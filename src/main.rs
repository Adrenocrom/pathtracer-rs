mod bdpt;
mod bvh;
mod camera;
mod font;
mod geometry;
mod gpu;
mod math;
mod output;
mod renderer;
mod scene;
mod window;

use crate::camera::Cam;
use crate::math::{Vec3, VEC_ZERO};
use crate::output::PixelBuffer;
use crate::renderer::{
    Renderer, FHD_HEIGHT, FHD_WIDTH, MAX_ACCUMULATED_FRAMES, MAX_DEPTH_FHD, SAMPLES_FHD,
};
use crate::scene::{LightData, MaterialData, SceneData};
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;

// --- CONFIGURATION ---
const SAMPLES_PREVIEW: usize = 1;
/// Path depth. Russian roulette terminates paths by throughput instead of
/// this cap, so it can sit at the vertex-array maximum without cost: short
/// paths (which carry most of the energy) are never killed, and deep ones
/// survive in proportion to their contribution.
const MAX_DEPTH_PREVIEW: usize = 8;

/// Build the scene with multiple objects for interesting lighting.
fn build_scene() -> SceneData {
    let emission = Vec3::new(1000.0, 1000.0, 1000.0);
    let room_size = 2.5;
    let room_height = 2.0;

    let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
    let red = MaterialData::diffuse(Vec3::new(0.75, 0.25, 0.25));
    let green = MaterialData::diffuse(Vec3::new(0.25, 0.75, 0.25));

    let mut scene = SceneData::new();

    // The room: finite quads, not infinite planes. Planes get a ±1e6
    // bounding box, which makes the BVH useless (the root box spans the
    // whole world and no wall is ever culled). Quads have tight boxes.
    //
    // `add_wall` reads `width`/`height` as world-axis extents (width
    // horizontal, height vertical) and centers the quad on `point`, so
    // the four side walls are centered at mid-height — a wall centered
    // at floor level would cover only the bottom half of the room.
    let floor_w = room_size * 2.0;
    let wall_h = room_height;
    let mid_y = room_height * 0.5;

    // Floor and ceiling (width spans x, height spans z)
    scene.add_wall(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        floor_w,
        floor_w,
        white,
    );
    scene.add_wall(
        Vec3::new(0.0, room_height, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        floor_w,
        floor_w,
        white,
    );

    // Back wall (width spans x, height spans y)
    scene.add_wall(
        Vec3::new(0.0, mid_y, -room_size),
        Vec3::new(0.0, 0.0, 1.0),
        floor_w,
        wall_h,
        white,
    );

    // Left wall (red) (width spans z, height spans y)
    scene.add_wall(
        Vec3::new(-room_size, mid_y, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        floor_w,
        wall_h,
        red,
    );

    // Right wall (green)
    scene.add_wall(
        Vec3::new(room_size, mid_y, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        floor_w,
        wall_h,
        green,
    );

    // Front wall
    scene.add_wall(
        Vec3::new(0.0, mid_y, room_size),
        Vec3::new(0.0, 0.0, -1.0),
        floor_w,
        wall_h,
        white,
    );

    // Ceiling light panel (one-sided: emits along its normal, i.e.
    // downward). `add_light` registers the quad as geometry too. It
    // provides the room's ambient illumination; the lamp itself is not
    // in frame from the default camera (it sits above the top edge of
    // the view).
    scene.add_light(LightData::from_quad(
            Vec3::new(0.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,
    ));

    // The mirror tile: 0.6 × 0.6, lying on the floor under the laser,
    // tilted 30° (its lower edge 2 mm above the floor). The normal
    // (0, cos 30°, sin 30°) points up and toward the front wall, so the
    // downward beam reflects up the wall at 60° from vertical.
    // Added *before* the laser: `add_laser` traces the beam's axis ray
    // through the scene to build the specular bridge (the aperture's
    // mirror image), so the mirror must already be registered.
    let mirror_tilt = 30.0f32.to_radians();
    scene.add_quad(
        Vec3::new(-0.9, 0.3 * mirror_tilt.sin() + 0.002, 0.6),
        Vec3::new(0.0, mirror_tilt.cos(), mirror_tilt.sin()),
        0.6,
        0.6,
        MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
    );

    // The laser showcase: a green laser hanging from the ceiling, aimed
    // straight down at the mirror tile, which bounces the beam up onto
    // the front wall — the wall the default camera faces. (A
    // wall-mounted flat mirror cannot do this: it flips the beam's
    // z-travel, so the beam can never land back on the wall it is
    // mounted on. The floor tilt is what turns the downward beam into
    // an upward one.)
    //
    // The beam's spot on the wall is *transported light*, not a decal:
    // `add_laser` traces the beam's axis ray, and when it hits the
    // mirror it registers the aperture's mirror image as a "specular
    // bridge" — a sampling technique the BDPT eye walk uses at diffuse
    // vertices (with probability BRIDGE_PROB it aims at the image disc,
    // so the path camera → wall → mirror → aperture is carried by
    // ordinary subpath walking; the direct-hit strategy then credits
    // the aperture's emission, cone-gated). No BDPT connection can
    // sample that path (the mirror is a delta endpoint) and no light
    // subpath can reach the aperture (a ~1 cm disc behind the mirror),
    // so without the bridge the spot is invisible — the estimator's
    // blind spot. There is deliberately no beam in flight: in clean air
    // a laser beam is invisible from the side; only the spot is real.
    scene.add_laser(
        Vec3::new(-0.9, 1.899, 0.6), // housing base cap 1 mm below the ceiling
        Vec3::new(0.0, -1.0, 0.0),  // beam straight down at the floor mirror
        0.2,                        // housing height
        0.01,                       // aperture radius
        0.01,                       // beam half-angle (radians, ~0.57°)
        Vec3::new(0.8, 4.0e8, 0.8), // green laser
    );

    scene
}

/// The default camera: eye height 1 m, level view down the room's +z
/// axis. Level (not the old slightly-down lookat, which was tuned for
/// the removed floor caustic) so the whole laser showcase is in frame:
/// the ceiling-mounted housing, the floor mirror, and the spot on the
/// front wall.
fn default_cam() -> Cam {
    Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 1.0, 0.0), 60.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `args()` yields argv[0] (the binary path) first, so the flags start
    // at index 1. `--terminal` forces the terminal UI; `--window` forces
    // the windowed one; with no flags the windowed frontend is preferred
    // and falls back to the terminal when unavailable.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let force_terminal = args.iter().any(|a| a == "--terminal");
    let force_window = args.iter().any(|a| a == "--window");

    if !force_terminal || force_window {
        let scene = build_scene();
        let cam = default_cam();
        match window::run_window(scene, cam) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!(
                    "Windowed frontend unavailable ({}), falling back to terminal UI",
                    e
                );
            }
        }
    }

    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;

    let (term_w, term_h) = {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 40));
        (w as usize, h as usize)
    };

    let width = term_w;
    let height = term_h * 2;

    let scene = build_scene();

    // Prefer the GPU backend; fall back to the CPU when no adapter exists.
    // The device is created once and reused across resizes and FHD
    // renders (a fresh `GpuRenderer::new` would re-create the instance,
    // adapter, device and re-upload the whole scene every time). Only the
    // cheap `Arc` handles are kept for the FHD export; the boxed renderer
    // itself drives the loop.
    let mut shared_device: Option<(wgpu::Device, wgpu::Queue)> = None;
    let mut renderer: Box<dyn Renderer> =
        match gpu::GpuRenderer::new(&scene, width as u32, height as u32) {
            Ok(gpu) => {
                eprintln!("GPU backend: {}", gpu.adapter_info());
                shared_device = Some((gpu.device_clone(), gpu.queue_clone()));
                Box::new(gpu)
            }
            Err(e) => {
                eprintln!("GPU unavailable ({}), falling back to CPU", e);
                Box::new(renderer::CpuRenderer::new(
                    &scene,
                    width,
                    height,
                    SAMPLES_PREVIEW,
                ))
            }
        };
    eprintln!("Renderer: {}", renderer.name());

    let mut cam = default_cam();
    let mut needs_render = true;
    let mut filter_enabled = false;
    let mut showing_help = false;

    // Temporal accumulation state (frames accumulated in this epoch).
    let mut frames_accumulated: usize = 0;

    loop {
        // Progressive rendering: keep accumulating while the camera is
        // stationary so the image cleans up over time. Cap the accumulation
        // so the average stops changing once it is converged enough.
        if !showing_help && (needs_render || frames_accumulated < MAX_ACCUMULATED_FRAMES) {
            renderer.render_frame(&cam, MAX_DEPTH_PREVIEW);
            frames_accumulated += 1;

            let mut buffer = renderer.read_pixels();

            if filter_enabled {
                // Apply the denoiser to the real buffer, not a clone.
                crate::output::apply_filters(&mut buffer);
            }

            let frame_string = buffer.to_string();
            crossterm::execute!(stdout, crossterm::cursor::MoveTo(0, 0))?;
            write!(stdout, "{}", frame_string).unwrap();
            stdout.flush()?;
            needs_render = false;
        }

        if crossterm::event::poll(std::time::Duration::from_millis(16))? {
            match crossterm::event::read()? {
                // While the help is on screen, any key just dismisses it.
                crossterm::event::Event::Key(_) if showing_help => {
                    showing_help = false;
                    needs_render = true;
                }
                // Windows crossterm emits Press *and* Release for every
                // key; without this guard each keypress would act twice.
                crossterm::event::Event::Key(key)
                    if key.kind == crossterm::event::KeyEventKind::Press =>
                {
                    match key.code {
                        crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => {
                            break
                        }
                        crossterm::event::KeyCode::Char('f') => {
                            filter_enabled = !filter_enabled;
                            needs_render = true;
                        }
                        crossterm::event::KeyCode::Char('h') => {
                            // Show help overlay; any key dismisses it.
                            showing_help = true;
                            let help = "\x1b[2J\x1b[H\
                            \x1b[1;36m╔══════════════════════════════════════╗\x1b[0m\n\
                            \x1b[1;36m║         Keyboard Controls             ║\x1b[0m\n\
                            \x1b[1;36m╠══════════════════════════════════════╣\x1b[0m\n\
                            \x1b[1;37m  W/S\x1b[0m - Move forward/backward         \n\
                            \x1b[1;37m  A/D\x1b[0m - Rotate left/right            \n\
                            \x1b[1;37m  F    \x1b[0m - Toggle denoiser filter      \n\
                            \x1b[1;37m  P    \x1b[0m - Render FHD PNG screenshot   \n\
                            \x1b[1;37m  H    \x1b[0m - Show this help              \n\
                            \x1b[1;37m  Q/Esc\x1b[0m - Quit                        \n\
                            \x1b[1;36m╚══════════════════════════════════════╝\x1b[0m\n\
                            \x1b[0m\nPress any key to return...\x1b[0m";
                            write!(stdout, "{}", help).unwrap();
                            stdout.flush()?;
                        }
                        crossterm::event::KeyCode::Char('p') => {
                            write!(stdout, "\r\nRendering FHD screenshot... ").unwrap();
                            stdout.flush()?;

                            // Render the FHD image with a dedicated renderer at
                            // 1920x1080, sharing the preview renderer's device
                            // (no new adapter, no scene re-upload). On the GPU
                            // this is a single dispatch per sample batch; on
                            // the CPU it renders in bands with a progress bar.
                            let fhd_buffer =
                                render_fhd(&scene, &cam, filter_enabled, shared_device.clone());

                            match fhd_buffer.save_as_png("screenshot.png") {
                                Ok(()) => write!(stdout, "Saved to screenshot.png!").unwrap(),
                                Err(e) => write!(stdout, "Failed to save: {}", e).unwrap(),
                            }
                            stdout.flush()?;
                            std::thread::sleep(std::time::Duration::from_secs(2));
                            // Redraw the preview over the progress-bar text.
                            needs_render = true;
                        }
                        crossterm::event::KeyCode::Char('w') => {
                            cam.move_forward(0.1);
                            renderer.reset_accumulation();
                            frames_accumulated = 0;
                            needs_render = true;
                        }
                        crossterm::event::KeyCode::Char('s') => {
                            cam.move_forward(-0.1);
                            renderer.reset_accumulation();
                            frames_accumulated = 0;
                            needs_render = true;
                        }
                        crossterm::event::KeyCode::Char('a') => {
                            cam.rotate(0.05);
                            renderer.reset_accumulation();
                            frames_accumulated = 0;
                            needs_render = true;
                        }
                        crossterm::event::KeyCode::Char('d') => {
                            cam.rotate(-0.05);
                            renderer.reset_accumulation();
                            frames_accumulated = 0;
                            needs_render = true;
                        }
                        _ => {}
                    }
                }
                crossterm::event::Event::Resize(w, h) => {
                    // Terminal resized: clear stale characters from the
                    // previous (possibly larger) frame and rebuild buffers.
                    crossterm::execute!(
                        stdout,
                        crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
                    )?;
                    // Reuse the existing renderer (device, pipeline, scene
                    // buffers) at the new resolution instead of creating a
                    // whole new GPU renderer.
                    renderer.resize(w as u32, h as u32 * 2);
                    frames_accumulated = 0;
                    needs_render = true;
                }
                _ => {}
            }
        }
    }

    crossterm::execute!(stdout, crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

/// Render the FHD screenshot. Uses the shared GPU device when one exists
/// (single dispatch per sample batch, no progress bar needed); otherwise
/// falls back to the CPU band renderer with a progress bar.
///
/// `shared_device` is the preview renderer's device/queue: reusing them
/// avoids re-creating the adapter and re-uploading the scene per export.
fn render_fhd(
    scene: &SceneData,
    cam: &Cam,
    filter_enabled: bool,
    shared_device: Option<(wgpu::Device, wgpu::Queue)>,
) -> PixelBuffer {
    let fhd_w = FHD_WIDTH;
    let fhd_h = FHD_HEIGHT;

    if let Some((device, queue)) = shared_device {
        let mut gpu = gpu::GpuRenderer::from_parts(
            device,
            queue,
            scene,
            fhd_w,
            fhd_h,
            String::from("shared device"),
        );
        // Accumulate SAMPLES_FHD samples on the GPU in batched dispatches
        // (one submit per batch instead of one per sample), then read back.
        const SPP_BATCH: u32 = 32;
        let mut remaining = SAMPLES_FHD;
        while remaining > 0 {
            let spp = SPP_BATCH.min(remaining);
            gpu.render_frame_spp(cam, MAX_DEPTH_FHD, spp);
            remaining -= spp;
        }
        let mut buffer = gpu.read_pixels();
        if filter_enabled {
            crate::output::apply_filters(&mut buffer);
        }
        return buffer;
    }

    // CPU fallback: render in bands with a progress bar.
    let (cpu_scene, lights) = scene.build_cpu_scene();
    let bridges = scene.bridges();
    let aspect = fhd_w as f32 / fhd_h as f32;
    let theta = cam.fov_deg.to_radians();
    let h = (theta * 0.5).tan();
    let w = aspect * h;

    let forward = (cam.lookat - cam.origin).normalize();
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = world_up.cross(forward).normalize();
    let up = forward.cross(right).normalize();

    let band: u32 = 32;
    let total_pixels = (fhd_w * fhd_h) as usize;
    let mut pixels: Vec<Vec3> = Vec::with_capacity(total_pixels);

    for y0 in (0..fhd_h).step_by(band as usize) {
        let y_end = (y0 + band).min(fhd_h);
        let rows: Vec<Vec<Vec3>> = (y0..y_end)
            .into_par_iter()
            .map(|y| {
                let mut rng = rand::thread_rng();
                (0..fhd_w)
                    .into_iter()
                    .map(|x| {
                        let mut color = VEC_ZERO;
                        for _ in 0..SAMPLES_FHD {
                            let u_jitter = (x as f32 + rng.gen::<f32>()) / fhd_w as f32;
                            let v_jitter = (y as f32 + rng.gen::<f32>()) / fhd_h as f32;
                            let px = (u_jitter * 2.0 - 1.0) * w;
                            let py = -(v_jitter * 2.0 - 1.0) * h;
                            let dir = (right * px + up * py + forward).normalize();
                            let ray = crate::geometry::Ray {
                                origin: cam.origin,
                                direction: dir,
                            };
                            color = color
                                + crate::bdpt::bdpt_trace(
                                    &ray,
                                    &*cpu_scene,
                                    MAX_DEPTH_FHD as usize,
                                    &lights,
                                    &bridges,
                                    &mut rng,
                                );
                        }
                        color / SAMPLES_FHD as f32
                    })
                    .collect()
            })
            .collect();
        pixels.extend(rows.into_iter().flatten());

        let rendered = (y_end * fhd_w) as usize;
        print!("\r{}", PixelBuffer::render_progress(rendered, total_pixels));
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }

    println!(
        "{}",
        PixelBuffer::render_progress(total_pixels, total_pixels)
    );

    let mut buffer = PixelBuffer {
        width: fhd_w as usize,
        height: fhd_h as usize,
        pixels,
    };
    if filter_enabled {
        crate::output::apply_filters(&mut buffer);
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Bsdf;

    #[test]
    fn scene_builds_and_is_sampleable() {
        let scene = build_scene();
        // The demo scene has two emitters: the ceiling panel and the
        // laser aperture. (No decal: the beam's spot is transported
        // light, not a fake emitter.)
        assert_eq!(scene.lights.len(), 2);
        for light in &scene.lights {
            assert!(light.area() > 0.0);
            assert!(light.emission.x > 0.0 || light.emission.y > 0.0 || light.emission.z > 0.0);
        }
        // A ray from the camera position must hit the scene.
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let ray = crate::geometry::Ray {
            origin: Vec3::new(0.0, 1.0, -1.5),
            direction: Vec3::new(0.0, 0.0, 1.0),
        };
        assert!(cpu_scene.intersect(&ray).is_some());
        assert_eq!(lights.len(), 2);
    }

    /// The laser light source: a diffuse cylinder housing plus a thin
    /// cylinder puck of laser material whose top cap is the aperture.
    /// The scene must register both as real geometry on both backends,
    /// the housing must carry a normal (non-emissive) material, and the
    /// aperture disc must be the puck's top cap.
    #[test]
    fn laser_is_a_cylinder_with_an_emitting_top_cap() {
        let scene = build_scene();
        // Exactly two cylinders: the diffuse housing and the laser puck.
        assert_eq!(scene.cylinders.len(), 2);
        let housing = scene
            .cylinders
            .iter()
            .find(|c| !c.mat.is_laser())
            .expect("the housing must carry a normal material");
        let puck = scene
            .cylinders
            .iter()
            .find(|c| c.mat.is_laser())
            .expect("the puck must carry the laser material");
        assert!(puck.mat.is_emissive());
        assert!(!housing.mat.is_emissive());
        // The light's aperture is the puck's top cap center.
        let laser_light = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("the laser must be registered as a light");
        let top = puck.base + puck.axis * (2.0 * puck.half_h);
        assert!((laser_light.center - top).length() < 1e-5);
        assert!((laser_light.normal - puck.axis).length() < 1e-5);
        assert!((laser_light.radius - puck.radius).abs() < 1e-5);
        // The housing is fatter than the puck and reaches further along
        // the axis, so the puck's rim never peeks out from behind it.
        assert!(housing.radius > puck.radius);
        assert!(
            housing.base.dot(housing.axis) < puck.base.dot(puck.axis),
            "the housing must start behind the puck"
        );
        // The CPU scene must contain the puck as intersectable geometry:
        // a ray along the axis from beyond the top cap hits it.
        let (cpu_scene, _) = scene.build_cpu_scene();
        let along_axis = crate::geometry::Ray {
            origin: top + puck.axis * 0.5,
            direction: puck.axis * -1.0,
        };
        let hit = cpu_scene
            .intersect(&along_axis)
            .expect("the laser housing must be intersectable");
        assert!(hit.mat.is_emissive());
        // ...and the hit is the aperture (the puck's top cap), in the beam.
        assert!((hit.p - top).length() < 1e-4);
    }

    /// The room must be closed: the six walls actually cover the full
    /// extents of the box (x, z ∈ [-2.5, 2.5], y ∈ [0, 2]). The wall
    /// sizing bug left two escape routes — the back/front walls were
    /// rotated (2 wide instead of 5) and the side walls were centered at
    /// floor level (covering only y ∈ [-1, 1]) — so rays through those
    /// regions returned the background instead of a wall.
    #[test]
    fn room_is_closed_on_all_six_sides() {
        let scene = build_scene();
        let (cpu_scene, _) = scene.build_cpu_scene();
        let center = Vec3::new(0.0, 1.0, 0.0);

        // Every axis ray from the room center must hit a wall.
        for dir in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let ray = crate::geometry::Ray {
                origin: center,
                direction: dir,
            };
            assert!(
                cpu_scene.intersect(&ray).is_some(),
                "ray {dir:?} escaped the room"
            );
        }

        // Through the old rotated-back-wall gap: x = 2 is outside the
        // old 2-wide back wall but inside the correct 5-wide one.
        let beside_back = crate::geometry::Ray {
            origin: Vec3::new(2.0, 1.0, 0.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
        };
        assert!(
            cpu_scene.intersect(&beside_back).is_some(),
            "back wall too narrow"
        );

        // Through the old half-height side-wall gap: y ≈ 1.9 at the wall
        // was above the old y ∈ [-1, 1] coverage.
        let above_side = crate::geometry::Ray {
            origin: Vec3::new(0.0, 1.8, 0.0),
            direction: Vec3::new(1.0, 0.05, 0.0).normalize(),
        };
        assert!(
            cpu_scene.intersect(&above_side).is_some(),
            "side wall too short"
        );
    }

    #[test]
    fn preview_render_produces_finite_pixels() {
        let scene = build_scene();
        let mut renderer = renderer::CpuRenderer::new(&scene, 16, 16, 4);
        let cam = default_cam();
        renderer.render_frame(&cam, 4);
        let buffer = renderer.read_pixels();
        assert_eq!(buffer.pixels.len(), 16 * 16);
        for p in &buffer.pixels {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
        // The image must not be black: the scene contains bright lights.
        let sum: f32 = buffer.pixels.iter().map(|p| p.x).sum();
        assert!(sum > 0.0);
    }

    /// The laser showcase, pinned by the beam tracer itself: the beam
    /// must leave the aperture, bounce off the mirror, and land on a
    /// diffuse wall — with the landing point, the mirror hit, and the
    /// housing all in frame from the default camera, and the landing
    /// point visible (not occluded). This is the regression guard for
    /// the whole showcase: a moved mirror, a re-aimed laser, or a
    /// camera change that pushes the spot out of frame fails here.
    #[test]
    fn laser_beam_reflects_onto_a_visible_spot() {
        let scene = build_scene();
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("demo scene must contain a laser");
        let trace = scene.trace_laser_beam(laser, 4);
        // Two legs: aperture -> mirror -> spot.
        assert_eq!(
            trace.legs.len(),
            2,
            "the beam must bounce exactly once (aperture -> mirror -> spot), got {:?}",
            trace.legs
        );
        let spot = trace
            .spot
            .expect("the beam must land on a diffuse surface");
        // The beam lands on the bare front wall (no decal: the spot is
        // transported light, not a fake emitter).
        assert_eq!(spot.mat.bsdf, Bsdf::Diffuse, "the spot must be diffuse");
        assert!(!spot.mat.is_emissive(), "the wall must not be emissive");
        // The spot is at the front wall (z = 2.5, normal -z).
        assert!(
            (spot.p.z - 2.5).abs() < 2e-3,
            "the beam must land at the front wall, got {:?}",
            spot.p
        );
        assert!((spot.normal + Vec3::new(0.0, 0.0, 1.0)).length() < 1e-4);

        // The mirror hit is the first leg's end: on the tilted mirror.
        let mirror_hit = trace.legs[0].1;
        let mirror = scene
            .quads
            .iter()
            .find(|q| q.mat.bsdf == Bsdf::Mirror)
            .expect("the mirror quad");
        let local = mirror_hit - mirror.center;
        assert!(
            local.dot(mirror.tangent).abs() <= mirror.half_w + 1e-3
                && local.dot(mirror.bitangent).abs() <= mirror.half_h + 1e-3,
            "the mirror hit {:?} is outside the mirror quad",
            mirror_hit
        );

        // The specular bridge: the aperture's mirror image, registered
        // by add_laser when the beam's first delta hit is the mirror.
        // This is the sampling technique that carries the beam's path
        // camera → wall → mirror → aperture — the whole fix for the
        // estimator's blind spot.
        let bridge = laser
            .bridge
            .expect("the laser must carry a specular bridge");
        // The image is the aperture reflected across the mirror quad's
        // plane: image = aperture − 2·(d·n)·n with d = aperture − hit.
        let aperture = laser.center;
        let d = aperture - mirror_hit;
        let expected_image = aperture - mirror.normal * (2.0 * d.dot(mirror.normal));
        assert!(
            (bridge.image_center - expected_image).length() < 1e-3,
            "the bridge image {:?} must be the aperture's reflection {:?}",
            bridge.image_center,
            expected_image
        );
        // The image disc's radius is the aperture radius.
        assert!((bridge.radius - laser.radius).abs() < 1e-6);

        // The bridge must be *usable* from the spot: the image disc's
        // axis lies in the wall's hemisphere (the spot can aim at it)
        // and the cone is non-degenerate.
        let d = bridge.image_center - spot.p;
        assert!(
            d.dot(spot.normal) > 0.0,
            "the bridge must point out of the wall"
        );
        assert!(
            d.length_squared() > bridge.radius * bridge.radius,
            "the bridge cone must be non-degenerate"
        );

        // The housing, the mirror hit, and the spot must be in frame from
        // the default camera.
        let cam = default_cam();
        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        let half_fov = (cam.fov_deg * 0.5).to_radians().cos();
        let in_frame = |name: &str, p: Vec3| {
            let rel = p - cam.origin;
            let f = rel.dot(forward);
            assert!(f > 0.0, "{name} is behind the camera");
            let cos_h = f / (rel.dot(right).abs().hypot(f));
            let cos_v = f / (rel.dot(up).abs().hypot(f));
            assert!(
                cos_h >= half_fov && cos_v >= half_fov,
                "{name} out of frame (cos_h {cos_h}, cos_v {cos_v}, need >= {half_fov})"
            );
        };
        let housing = scene
            .cylinders
            .iter()
            .find(|c| !c.mat.is_laser())
            .expect("the laser housing");
        let mid = housing.base + housing.axis * housing.half_h;
        in_frame("laser housing", mid);
        in_frame("mirror hit", mirror_hit);
        in_frame("laser spot", spot.p);

        // The spot must be visible from the camera: the camera -> spot
        // segment must not be occluded.
        let (cpu_scene, _) = scene.build_cpu_scene();
        let seg = spot.p - cam.origin;
        let len = seg.length();
        let ray = crate::geometry::Ray {
            origin: cam.origin,
            direction: seg * (1.0 / len),
        };
        if let Some(hit) = cpu_scene.intersect(&ray) {
            assert!(
                hit.t + 1e-3 >= len,
                "laser spot occluded at t {} (spot at {})",
                hit.t,
                len
            );
        }
    }

    /// TEMPORARY probe: the CPU estimator's mean at the spot pixel vs the
    /// analytic expectation. Run with --nocapture.
    #[test]
    fn temp_spot_probe_cpu() {
        let scene = build_scene();
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let bridges = scene.bridges();
        let cam = default_cam();
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("laser");
        let trace = scene.trace_laser_beam(laser, 4);
        let spot = trace.spot.expect("spot").p;
        let dir = (spot - cam.origin).normalize();
        let ray = crate::geometry::Ray {
            origin: cam.origin,
            direction: dir,
        };
        let mut rng = rand::thread_rng();
        let n = 4000;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for _ in 0..n {
            let c = crate::bdpt::bdpt_trace(&ray, &*cpu_scene, 8, &lights, &bridges, &mut rng);
            sum += c.y as f64;
            sum_sq += (c.y * c.y) as f64;
        }
        let mean = sum / n as f64;
        let std = (sum_sq / n as f64 - mean * mean).max(0.0).sqrt();
        std::fs::write(
            "target/spot_probe_cpu.txt",
            format!(
                "CPU spot pixel: mean {mean:.1} ± {:.1}\n",
                std / (n as f64).sqrt()
            ),
        )
        .ok();
        // The analytic expectation at the spot center:
        // albedo/π · Le·R · cos·Ω_image, Ω_image = π r²/d² (face-on).
        let d = (spot - laser.bridge.expect("bridge").image_center).length();
        let omega = std::f32::consts::PI * laser.radius * laser.radius / (d * d);
        let mirror = scene
            .quads
            .iter()
            .find(|q| q.mat.bsdf == Bsdf::Mirror)
            .expect("mirror");
        let expected = 0.8 / std::f32::consts::PI
            * laser.emission.y
            * mirror.mat.albedo.y
            * 0.8660254
            * omega;
        std::fs::write(
            "target/spot_analytic.txt",
            format!("analytic spot radiance: {expected:.1} (d {d:.3}, Ω {omega:.3e})\n"),
        )
        .ok();
    }

    /// Regenerate `screenshot.png` headlessly through the exact production
    /// path (`render_fhd` → `save_as_png`), so the acceptance artifact
    /// (the laser spot visible on the front wall) can be produced
    /// without pressing `P` in the interactive UI. Also probes the raw
    /// HDR buffer for the spot: the traced landing point must render
    /// clearly brighter in the laser's color channel than the wall
    /// around it.
    ///
    /// Runs as part of the default suite: it prefers the GPU (the FHD
    /// render takes seconds) and falls back to the CPU renderer, where
    /// the full render takes minutes. The `SCREENSHOT_SPP` env var
    /// overrides the sample count for quick probe runs (e.g.
    /// `SCREENSHOT_SPP=16`); run it alone with
    /// `cargo test regenerate_screenshot -- --nocapture`.
    #[test]
    fn regenerate_screenshot() {
        let scene = build_scene();
        let cam = default_cam();

        // Optional quick-probe override for the sample count.
        let spp = std::env::var("SCREENSHOT_SPP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(SAMPLES_FHD);

        // Prefer the GPU (the production path does too); fall back to the
        // CPU band renderer when no adapter exists.
        let shared_device = gpu::GpuRenderer::new(&scene, 8, 8).ok().map(|g| {
            let d = g.device_clone();
            let q = g.queue_clone();
            (d, q)
        });

        let buffer = render_fhd_with_spp(&scene, &cam, false, shared_device, spp);

        // --- Laser spot probe: the traced landing point must render
        // clearly brighter in the laser's color channel (green) than the
        // wall around it — the visual acceptance of the laser showcase.
        // The spot is the traced beam's landing point (never a formula:
        // the trace is the ground truth); the reference is a wall disc
        // 0.4 to the +x, outside the decal (half-width ~0.03) and away
        // from every object.
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("demo scene must contain a laser");
        let trace = scene.trace_laser_beam(laser, 4);
        let spot_world = trace
            .spot
            .expect("the beam must land on a diffuse surface")
            .p;
        let (sx, sy) = project_to_pixel(&cam, spot_world, buffer.width, buffer.height);
        let spot_ref_world = spot_world + Vec3::new(0.4, 0.0, 0.0);
        let (srx, sry) = project_to_pixel(&cam, spot_ref_world, buffer.width, buffer.height);
        let disc_mean = |px: usize, py: usize| -> f32 {
            let mut sum = 0.0f32;
            let mut n = 0usize;
            for dy in -40i64..=40 {
                for dx in -40i64..=40 {
                    if dx * dx + dy * dy > 40 * 40 {
                        continue;
                    }
                    let x = px as i64 + dx;
                    let y = py as i64 + dy;
                    if x < 0 || y < 0 || x >= buffer.width as i64 || y >= buffer.height as i64 {
                        continue;
                    }
                    // The green channel: the laser's color. A green
                    // laser is invisible in a red-channel probe.
                    sum += buffer.get_pixel(x as usize, y as usize).y;
                    n += 1;
                }
            }
            sum / n.max(1) as f32
        };
        let spot_mean = disc_mean(sx, sy);
        let spot_ref_mean = disc_mean(srx, sry);
        // TEMPORARY: dump the center pixel and a cross-section through
        // the spot, to compare against the CPU estimator's probe.
        let center = buffer.get_pixel(sx, sy);
        let mut cross = String::new();
        for dx in -60i64..=60 {
            let x = (sx as i64 + dx).clamp(0, buffer.width as i64 - 1) as usize;
            cross.push_str(&format!("{:.0} ", buffer.get_pixel(x, sy).y));
        }
        std::fs::write(
            "target/laser_probe.txt",
            format!(
                "laser spot ({sx},{sy}) mean {spot_mean}\nreference ({srx},{sry}) mean {spot_ref_mean}\nratio {}\ncenter {center:?}\ncross {cross}\n",
                spot_mean / spot_ref_mean
            ),
        )
        .ok();
        // The beam concentrates the aperture's power into a tiny disc:
        // the spot must be several times brighter than the ambient-lit
        // wall around it. 2x is a conservative floor for "clearly
        // visible".
        assert!(
            spot_mean > spot_ref_mean * 2.0,
            "no laser spot visible: spot {spot_mean} vs reference {spot_ref_mean}"
        );

        buffer
            .save_as_png("screenshot.png")
            .expect("failed to save screenshot.png");
    }

    /// Project a world point to image pixel coordinates (the inverse of
    /// the camera ray generation: u in [0,1] left->right, v in [0,1]
    /// top->bottom).
    fn project_to_pixel(cam: &Cam, p: Vec3, width: usize, height: usize) -> (usize, usize) {
        let aspect = width as f32 / height as f32;
        let theta = cam.fov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let w = aspect * h;
        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        let rel = p - cam.origin;
        let px = rel.dot(right) / rel.dot(forward);
        let py = rel.dot(up) / rel.dot(forward);
        let u = (px / w + 1.0) * 0.5;
        let v = (1.0 - py / h) * 0.5;
        (
            (u * width as f32).clamp(0.0, width as f32 - 1.0) as usize,
            (v * height as f32).clamp(0.0, height as f32 - 1.0) as usize,
        )
    }

    /// `render_fhd` with an explicit sample count (used by the screenshot
    /// test's quick-probe override; the production path always uses
    /// `SAMPLES_FHD`).
    fn render_fhd_with_spp(
        scene: &SceneData,
        cam: &Cam,
        filter_enabled: bool,
        shared_device: Option<(wgpu::Device, wgpu::Queue)>,
        spp: u32,
    ) -> PixelBuffer {
        let fhd_w = FHD_WIDTH;
        let fhd_h = FHD_HEIGHT;

        if let Some((device, queue)) = shared_device {
            let mut gpu = gpu::GpuRenderer::from_parts(
                device,
                queue,
                scene,
                fhd_w,
                fhd_h,
                String::from("shared device"),
            );
            const SPP_BATCH: u32 = 32;
            let mut remaining = spp;
            while remaining > 0 {
                let batch = SPP_BATCH.min(remaining);
                gpu.render_frame_spp(cam, MAX_DEPTH_FHD, batch);
                remaining -= batch;
            }
            let mut buffer = gpu.read_pixels();
            if filter_enabled {
                crate::output::apply_filters(&mut buffer);
            }
            return buffer;
        }

        // CPU fallback: render in bands with a progress bar.
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let bridges = scene.bridges();
        let aspect = fhd_w as f32 / fhd_h as f32;
        let theta = cam.fov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let w = aspect * h;

        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

        let band: u32 = 32;
        let total_pixels = (fhd_w * fhd_h) as usize;
        let mut pixels: Vec<Vec3> = Vec::with_capacity(total_pixels);

        for y0 in (0..fhd_h).step_by(band as usize) {
            let y_end = (y0 + band).min(fhd_h);
            let rows: Vec<Vec<Vec3>> = (y0..y_end)
                .into_par_iter()
                .map(|y| {
                    let mut rng = rand::thread_rng();
                    (0..fhd_w)
                        .into_iter()
                        .map(|x| {
                            let mut color = VEC_ZERO;
                            for _ in 0..spp {
                                let u_jitter = (x as f32 + rng.gen::<f32>()) / fhd_w as f32;
                                let v_jitter = (y as f32 + rng.gen::<f32>()) / fhd_h as f32;
                                let px = (u_jitter * 2.0 - 1.0) * w;
                                let py = -(v_jitter * 2.0 - 1.0) * h;
                                let dir = (right * px + up * py + forward).normalize();
                                let ray = crate::geometry::Ray {
                                    origin: cam.origin,
                                    direction: dir,
                                };
                                color = color
                                    + crate::bdpt::bdpt_trace(
                                        &ray,
                                        &*cpu_scene,
                                        MAX_DEPTH_FHD as usize,
                                        &lights,
                                        &bridges,
                                        &mut rng,
                                    );
                            }
                            color / spp as f32
                        })
                        .collect()
                })
                .collect();
            pixels.extend(rows.into_iter().flatten());

            let rendered = (y_end * fhd_w) as usize;
            print!("\r{}", PixelBuffer::render_progress(rendered, total_pixels));
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        let mut buffer = PixelBuffer {
            width: fhd_w as usize,
            height: fhd_h as usize,
            pixels,
        };
        if filter_enabled {
            crate::output::apply_filters(&mut buffer);
        }
        buffer
    }
}
