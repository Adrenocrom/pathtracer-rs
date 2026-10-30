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
    let emission = Vec3::new(10.0, 10.0, 10.0);
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

    // Central spheres
    scene.add_sphere(Vec3::new(-0.9, 0.4, -0.9), 0.4, white);
    scene.add_sphere(Vec3::new(-0.3, 0.4, -0.2), 0.4, white);

    scene.add_cube(Vec3::new(1.0, 0.3, -0.8), 0.6, white);
    scene.add_sphere(Vec3::new(1.0, 1.0, -0.8), 0.4, white);

    // Stacked cubes for interesting shadows
    scene.add_cube(Vec3::new(0.3, 0.3, 0.8), 0.6, white);
    scene.add_cube(Vec3::new(0.94, 0.3, 0.8), 0.6, white);
    scene.add_cube(Vec3::new(1.58, 0.3, 0.8), 0.6, white);

    scene.add_cube(Vec3::new(0.45, 0.9, 0.8), 0.6, white);
    scene.add_cube(Vec3::new(1.09, 0.9, 0.8), 0.6, white);

    // Hanging light (small emissive sphere). `add_light` registers the
    // emitter as geometry too, so both backends can hit it.
    scene.add_light(LightData::from_sphere(
        Vec3::new(0.65, 0.1, 1.5),
        0.2,
        emission,
    ));

    // Ceiling light panel (one-sided: emits along its normal, i.e.
    // downward). `add_light` registers the quad as geometry too.
    //scene.add_light(LightData::from_quad(
    //        Vec3::new(0.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,
    //));

    //scene.add_light(LightData::from_quad(
    //        Vec3::new(-15.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,
    //));

    //scene.add_light(LightData::from_quad(
    //        Vec3::new(15.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,
    //));

    scene
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_builds_and_is_sampleable() {
        let scene = build_scene();
        assert_eq!(scene.lights.len(), 1);
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
        assert_eq!(lights.len(), 1);
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
        let cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.7, 0.0), 60.0);
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
        let cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.7, 0.0), 60.0);
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

    let mut cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.7, 0.0), 60.0);
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
