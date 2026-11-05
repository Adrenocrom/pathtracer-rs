//! Backend-agnostic renderer interface.
//!
//! `CpuRenderer` wraps the existing rayon + BVH path (the reference oracle);
//! `GpuRenderer` (src/gpu.rs) implements the same interface on wgpu. The
//! terminal UI only talks to this trait, so switching backends is a one-line
//! change and the CPU path stays fully testable.

use crate::camera::Cam;
use crate::math::{Vec3, VEC_ZERO};
use crate::output::PixelBuffer;
use crate::scene::SceneData;

/// How many preview frames to average while the camera is stationary.
pub const MAX_ACCUMULATED_FRAMES: usize = 128;

/// Full-quality export (the `P` key in both frontends): 1920×1080 at
/// 1024 samples per pixel.
pub const FHD_WIDTH: u32 = 1920;
pub const FHD_HEIGHT: u32 = 1080;
pub const SAMPLES_FHD: u32 = 1289;
pub const MAX_DEPTH_FHD: u32 = 8;

pub trait Renderer {
    /// Render one frame (one sample per pixel) into the internal buffer.
    fn render_frame(&mut self, cam: &Cam, max_depth: usize);

    /// Copy the accumulated pixels out of the renderer.
    fn read_pixels(&mut self) -> PixelBuffer;

    /// Signal that the camera moved: restart temporal accumulation.
    fn reset_accumulation(&mut self);

    /// Signal that the *scene* changed (e.g. an animated mirror moved):
    /// re-ingest the geometry and restart temporal accumulation. The
    /// laser bridge ladders are re-derived from the new geometry (they
    /// are never stored), so the beam's sampling follows the movement.
    /// The default is a no-op for backends that cannot re-ingest a
    /// scene.
    fn update_scene(&mut self, _scene: &SceneData) {}

    /// Rebuild internal buffers at a new resolution. The default is a
    /// no-op: backends that allocate per-pixel state (the GPU accumulation
    /// buffer) override it, the CPU renderer rebuilds itself instead.
    fn resize(&mut self, _width: u32, _height: u32) {}

    /// Human-readable backend name for the status line.
    fn name(&self) -> String;
}

/// CPU backend: rayon parallelism over the BVH scene, one frame per call.
pub struct CpuRenderer {
    scene: Box<dyn crate::geometry::Intersectable>,
    lights: Vec<crate::geometry::AreaLight>,
    /// The specular bridges (the eye walk's diffuse sampling mixture):
    /// each laser's whole ladder, one rung per mirror its beam crosses.
    bridges: Vec<crate::bdpt::Bridge>,
    width: usize,
    height: usize,
    samples: usize,
    accum: Option<Vec<crate::math::Vec3>>,
    /// Running second moment of the sample luminance per pixel,
    /// E[lum²], the temporal-outlier-rejection statistic. Kept in lockstep
    /// with `accum` (same length, same sample count); the GPU stores the
    /// same statistic in the accumulation buffer's free `.w` channel.
    moment2: Vec<f32>,
    /// Samples accumulated in this epoch (each `render_frame` call adds
    /// `samples` of them). The accumulation weights batches by sample
    /// count, matching the GPU's `samples`/`spp` uniform pair.
    samples_accumulated: usize,
}

impl CpuRenderer {
    pub fn new(scene_data: &SceneData, width: usize, height: usize, samples: usize) -> Self {
        let (scene, lights) = scene_data.build_cpu_scene();
        let bridges = scene_data.bridges();
        Self {
            scene,
            lights,
            bridges,
            width,
            height,
            samples,
            accum: None,
            moment2: Vec::new(),
            samples_accumulated: 0,
        }
    }
}

/// Progressive temporal accumulation with outlier rejection.
///
/// The plain running mean is defenseless against fireflies: a sample
/// with radiance ~50 in a pixel whose true mean is ~0.05 enters at full
/// weight and then only dilutes as 1/n — a white dot that persists for
/// hundreds of frames. The rejection winsorizes each *batch* against
/// the pixel's own running statistics before it enters the mean: a
/// batch whose mean luminance exceeds the running mean by more than
/// `WINSOR_K` standard deviations is scaled down to that bound.
///
/// The statistic is E[lum²], an exponential moving average over
/// roughly `MOMENT_WINDOW` samples (on the GPU it lives in the free
/// `.w` channel of the accumulation buffer): σ = sqrt(E[lum²] − μ²).
/// It is an EMA rather than a full running average for a subtle
/// reason: capping the *running* second moment at the rejection bound
/// would deadlock a caustic pixel whose first samples are dark (σ = 0
/// keeps the bound at μ, every bright batch is clamped to μ, and the
/// moment never grows). The EMA keeps the raw (pre-winsorization)
/// luminance in the statistic, so a genuine caustic — many bright
/// batches — keeps σ loose and converges to its true value, while a
/// single firefly's σ-poisoning decays within ~MOMENT_WINDOW samples.
/// That asymmetry is exactly the statistic that tells a caustic from
/// a firefly.
///
/// `n` is the sample count already accumulated; the batch carries `m`
/// samples per pixel (the same sample-count weighting as the GPU's
/// accumulation block, which mixes 1-spp preview and 32-spp export
/// batches).
fn accumulate(
    accum: &mut [Vec3],
    moment2: &mut [f32],
    frame: &[Vec3],
    n: usize,
    m: usize,
) {
    let n = n as f32;
    let m = m as f32;
    // EMA rate: the batch's weight in the statistic. Per-sample, not
    // per-batch: a batch of m samples moves the statistic by
    // 1 − (1 − 1/window)^m, so a large batch (the 32-spp export) can
    // never *replace* the statistic — the per-batch form m/window
    // clamps to 1 there, and a batch whose spot pixel drew zero bridge
    // directions (probability (1 − p)³² ≈ 19% at BRIDGE_PROB = 0.05)
    // would collapse σ to ~0 and winsorize the next bright batch down
    // to μ, erasing the laser's spot from the accumulation.
    let alpha = 1.0 - (1.0 - 1.0 / crate::bdpt::MOMENT_WINDOW as f32).powf(m);
    for ((a, m2), f) in accum.iter_mut().zip(moment2.iter_mut()).zip(frame) {
        // Winsorize the batch mean against the running statistics.
        let mut batch = *f;
        if n > 0.0 {
            let mu = a.luminance();
            let sigma = (*m2 - mu * mu).max(0.0).sqrt();
            let bound = mu + crate::bdpt::WINSOR_K * sigma;
            let lum = batch.luminance();
            if lum > bound && lum > 0.0 {
                batch = batch * (bound / lum);
            }
        }
        if n + m > 0.0 {
            *a = (*a * n + batch * m) / (n + m);
        }
        // The statistic tracks the *raw* luminance (pre-winsorization),
        // so a genuine caustic keeps σ loose; the EMA bounds how long a
        // single outlier can keep it loose.
        let raw_sq = f.luminance() * f.luminance();
        *m2 = *m2 * (1.0 - alpha) + raw_sq * alpha;
    }
}

impl Renderer for CpuRenderer {
    fn render_frame(&mut self, cam: &Cam, max_depth: usize) {
        let frame = cam.render(
            &*self.scene,
            self.width,
            self.height,
            self.samples,
            max_depth,
            &self.lights,
            &self.bridges,
        );
        match &mut self.accum {
            Some(a) if a.len() == frame.pixels.len() => {
                accumulate(
                    a,
                    &mut self.moment2,
                    &frame.pixels,
                    self.samples_accumulated,
                    self.samples,
                );
                self.samples_accumulated += self.samples;
            }
            _ => {
                // First frame: the mean is the batch itself; the EMA
                // statistic starts at the batch's own raw luminance²
                // (the same initialization the GPU's reset path
                // performs).
                self.moment2 = frame.pixels.iter().map(|p| p.luminance().powi(2)).collect();
                self.accum = Some(frame.pixels);
                self.samples_accumulated = self.samples;
            }
        }
    }

    fn read_pixels(&mut self) -> PixelBuffer {
        // Before the first `render_frame` nothing is accumulated yet. Return
        // a zero-filled buffer of the full size so callers can index or
        // display it without panicking on an empty vec.
        let pixels = self
            .accum
            .clone()
            .unwrap_or_else(|| vec![VEC_ZERO; self.width * self.height]);
        PixelBuffer {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    fn reset_accumulation(&mut self) {
        self.accum = None;
        self.moment2.clear();
        self.samples_accumulated = 0;
    }

    /// The scene changed: rebuild the BVH scene, the lights and the
    /// (derived) laser bridges from the new geometry, and restart the
    /// accumulation.
    fn update_scene(&mut self, scene_data: &SceneData) {
        let (scene, lights) = scene_data.build_cpu_scene();
        self.scene = scene;
        self.lights = lights;
        self.bridges = scene_data.bridges();
        self.reset_accumulation();
    }

    /// The CPU renderer has no per-pixel GPU state; resizing just changes
    /// the target resolution and drops the stale accumulation.
    fn resize(&mut self, width: u32, height: u32) {
        self.width = width as usize;
        self.height = height as usize;
        self.reset_accumulation();
    }

    fn name(&self) -> String {
        format!("CPU (rayon, {} spp/frame)", self.samples)
    }
}

impl CpuRenderer {
    /// The scene's laser bridges, re-derived from the current geometry
    /// by `update_scene`. Exposed for tests that pin the movement
    /// pipeline (a moved mirror must move the bridge).
    #[cfg(test)]
    pub fn bridges(&self) -> &[crate::bdpt::Bridge] {
        &self.bridges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn accumulate_converges_to_the_mean() {
        let mut accum = vec![Vec3::new(1.0, 1.0, 1.0); 4];
        let mut moment2 = vec![1.0f32; 4];
        let frame = vec![Vec3::new(0.0, 0.0, 0.0); 4];
        // After one update the average must be exactly halfway.
        accumulate(&mut accum, &mut moment2, &frame, 1, 1);
        assert!((accum[0].x - 0.5).abs() < 1e-6);
        // After many updates it approaches the frame value.
        for n in 2..=100 {
            accumulate(&mut accum, &mut moment2, &frame, n, 1);
        }
        assert!(accum[0].x < 0.02);
    }

    /// The outlier rejection: a batch far above the running mean+3σ must
    /// be scaled down before entering the average, so a single firefly
    /// cannot paint a persistent dot — while a *sequence* of bright
    /// batches (a genuine caustic) raises the second moment, loosens the
    /// cap, and converges to its true value.
    #[test]
    fn accumulate_rejects_temporal_outliers() {
        // Steady state: mean 0.1, tiny variance.
        let mut accum = vec![Vec3::new(0.1, 0.1, 0.1)];
        let mut moment2 = vec![0.1f32 * 0.1 + 1e-6];
        // One firefly batch: luminance 50 against mean 0.1, σ ~ 0.001.
        let firefly = vec![Vec3::new(50.0, 50.0, 50.0)];
        accumulate(&mut accum, &mut moment2, &firefly, 100, 1);
        // The firefly must be winsorized to roughly mean + 3σ ≈ 0.103,
        // so the new mean moves by ~0.0003, not 0.5.
        assert!(
            accum[0].x < 0.11,
            "firefly not rejected: mean jumped to {accum:?}"
        );

        // A genuine caustic: many bright batches. The second moment
        // grows with each, the cap loosens, and the mean converges
        // toward the bright value.
        let mut accum = vec![Vec3::new(0.1, 0.1, 0.1)];
        let mut moment2 = vec![0.1f32 * 0.1 + 1e-6];
        for n in 1..=200 {
            let bright = vec![Vec3::new(2.0, 2.0, 2.0)];
            accumulate(&mut accum, &mut moment2, &bright, n, 1);
        }
        assert!(
            accum[0].x > 1.0,
            "genuine bright region suppressed: {accum:?}"
        );

        // The deadlock case: a caustic pixel whose first samples are
        // dark (σ = 0 pins the bound at μ). The EMA statistic must
        // still let the bright batches through after a few frames —
        // a capped running average would clamp every bright batch to
        // μ forever and the caustic would never appear.
        let mut accum = vec![Vec3::new(0.0, 0.0, 0.0)];
        let mut moment2 = vec![0.0f32];
        for n in 1..=64 {
            let bright = vec![Vec3::new(2.0, 2.0, 2.0)];
            accumulate(&mut accum, &mut moment2, &bright, n, 1);
        }
        assert!(
            accum[0].x > 1.0,
            "caustic from a dark start never converged: {accum:?}"
        );
    }

    /// `read_pixels` before the first `render_frame` used to return an
    /// empty vec; indexing it (`get_pixel`, terminal display) panicked.
    #[test]
    fn read_pixels_before_first_frame_is_zero_filled() {
        let mut scene = crate::scene::SceneData::new();
        scene.add_plane(
            crate::math::Vec3::new(0.0, 0.0, 0.0),
            crate::math::Vec3::new(0.0, 1.0, 0.0),
            crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
        );
        let mut renderer = CpuRenderer::new(&scene, 4, 3, 1);
        let buffer = renderer.read_pixels();
        assert_eq!(buffer.pixels.len(), 4 * 3);
        assert_eq!(buffer.get_pixel(3, 2), Vec3::new(0.0, 0.0, 0.0));
        // Displaying the buffer must not panic either.
        assert!(!buffer.to_string().is_empty());
    }

    /// The movement pipeline: a scene mutation followed by
    /// `update_scene` must move the derived laser bridge with the
    /// geometry. The renderer's bridges are re-derived from the new
    /// scene, never stored — this pins that contract end to end.
    #[test]
    fn update_scene_moves_the_derived_bridge() {
        use crate::scene::{MaterialData, SceneData};

        // A small closed room with a laser aimed at a 45° floor mirror
        // and a 45° wall mirror (the same chain as the scene tests).
        let build = |wall_z: f32| -> SceneData {
            let mut scene = SceneData::new();
            let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
            scene.add_wall(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(0.0, 1.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, -1.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 2.0, 2.0, white);
            scene.add_laser(
                Vec3::new(0.4, 1.9, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                0.2,
                0.01,
                0.01,
                Vec3::new(1.0, 1.0, 1.0),
            );
            scene.add_quad(
                Vec3::new(0.4, 0.002, 0.0),
                Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
                0.6,
                0.6,
                MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
            );
            scene.add_quad(
                Vec3::new(0.4, 0.0027, wall_z),
                Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),
                0.6,
                0.6,
                MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
            );
            scene
        };

        let mut renderer = CpuRenderer::new(&build(0.998), 4, 4, 1);
        let before = renderer.bridges().to_vec();
        assert_eq!(before.len(), 2, "the laser must carry one rung per mirror");

        // Move the wall mirror 0.5 toward the room center and update.
        let moved = build(0.5);
        renderer.update_scene(&moved);
        let after = renderer.bridges().to_vec();
        assert_eq!(after.len(), 2, "the ladder must survive the move");
        assert!(
            (before[1].image_center - after[1].image_center).length() > 0.1,
            "the whole-chain rung must follow the moved mirror: {:?} vs {:?}",
            before[1].image_center,
            after[1].image_center
        );
        // The moved ladder must be the correct ladder for the moved
        // geometry (the scene-level test pins the exact values; here it
        // suffices that the renderer serves the *new* scene's ladder).
        let laser = moved
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("the laser");
        let expected = moved
            .bridges_of_light(laser, 4);
        assert_eq!(expected.len(), 2, "the moved scene's ladder");
        for (a, e) in after.iter().zip(&expected) {
            assert!(
                (a.image_center - e.image_center).length() < 1e-6,
                "the renderer must serve the moved scene's ladder"
            );
        }
    }
}
