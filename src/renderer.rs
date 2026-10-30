//! Backend-agnostic renderer interface.
//!
//! `CpuRenderer` wraps the existing rayon + BVH path (the reference oracle);
//! `GpuRenderer` (src/gpu.rs) implements the same interface on wgpu. The
//! terminal UI only talks to this trait, so switching backends is a one-line
//! change and the CPU path stays fully testable.

use crate::camera::Cam;
use crate::math::Vec3;
use crate::output::PixelBuffer;
use crate::scene::SceneData;

/// How many preview frames to average while the camera is stationary.
pub const MAX_ACCUMULATED_FRAMES: usize = 128;

pub trait Renderer {
    /// Render one frame (one sample per pixel) into the internal buffer.
    fn render_frame(&mut self, cam: &Cam, max_depth: usize);

    /// Copy the accumulated pixels out of the renderer.
    fn read_pixels(&mut self) -> PixelBuffer;

    /// Signal that the camera moved: restart temporal accumulation.
    fn reset_accumulation(&mut self);

    /// Human-readable backend name for the status line.
    fn name(&self) -> String;
}

/// CPU backend: rayon parallelism over the BVH scene, one frame per call.
pub struct CpuRenderer {
    scene: Box<dyn crate::geometry::Intersectable>,
    lights: Vec<crate::geometry::AreaLight>,
    width: usize,
    height: usize,
    samples: usize,
    accum: Option<Vec<crate::math::Vec3>>,
    frames: usize,
}

impl CpuRenderer {
    pub fn new(scene_data: &SceneData, width: usize, height: usize, samples: usize) -> Self {
        let (scene, lights) = scene_data.build_cpu_scene();
        Self {
            scene,
            lights,
            width,
            height,
            samples,
            accum: None,
            frames: 0,
        }
    }
}

impl Renderer for CpuRenderer {
    fn render_frame(&mut self, cam: &Cam, max_depth: usize) {
        let frame = cam.render(&*self.scene, self.width, self.height, self.samples, max_depth, &self.lights);
        match &mut self.accum {
            Some(a) if a.len() == frame.pixels.len() => {
                accumulate(a, &frame.pixels, self.frames);
                self.frames += 1;
            }
            _ => {
                self.accum = Some(frame.pixels);
                self.frames = 1;
            }
        }
    }

    fn read_pixels(&mut self) -> PixelBuffer {
        PixelBuffer {
            width: self.width,
            height: self.height,
            pixels: self.accum.clone().unwrap_or_default(),
        }
    }

    fn reset_accumulation(&mut self) {
        self.accum = None;
        self.frames = 0;
    }

    fn name(&self) -> String {
        format!("CPU (rayon, {} spp/frame)", self.samples)
    }
}

/// Progressive temporal accumulation: average the new frame into `accum`.
/// `frame_count` is the number of frames already accumulated (>= 1).
fn accumulate(accum: &mut [Vec3], frame: &[Vec3], frame_count: usize) {
    let n = frame_count as f32;
    for (a, f) in accum.iter_mut().zip(frame) {
        *a = *a * (n / (n + 1.0)) + *f * (1.0 / (n + 1.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn accumulate_converges_to_the_mean() {
        let mut accum = vec![Vec3::new(1.0, 1.0, 1.0); 4];
        let frame = vec![Vec3::new(0.0, 0.0, 0.0); 4];
        // After one update the average must be exactly halfway.
        accumulate(&mut accum, &frame, 1);
        assert!((accum[0].x - 0.5).abs() < 1e-6);
        // After many updates it approaches the frame value.
        for n in 2..=100 {
            accumulate(&mut accum, &frame, n);
        }
        assert!(accum[0].x < 0.02);
    }
}
