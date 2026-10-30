mod bdpt;
mod bvh;
mod camera;
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
use crate::renderer::{Renderer, MAX_ACCUMULATED_FRAMES};
use crate::scene::{LightData, MaterialData, SceneData};
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;

// --- CONFIGURATION ---
const SAMPLES_PREVIEW: usize = 1;
const MAX_DEPTH_PREVIEW: usize = 4;
const SAMPLES_FHD: usize = 1024;
const MAX_DEPTH_FHD: usize = 4;

/// Build the scene with multiple objects for interesting lighting.
fn build_scene() -> SceneData {
    let emision = Vec3::new(10.0, 10.0, 10.0);
    let room_size = 2.5;
    let room_height = 2.0;

    let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
    let red = MaterialData::diffuse(Vec3::new(0.75, 0.25, 0.25));
    let green = MaterialData::diffuse(Vec3::new(0.25, 0.75, 0.25));
    let light_mat = MaterialData::emissive(emision);

    let mut scene = SceneData::new();

    // Floor and ceiling
    scene.add_plane(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), white);
    scene.add_plane(Vec3::new(0.0, room_height, 0.0), Vec3::new(0.0, -1.0, 0.0), white);

    // Back wall
    scene.add_plane(Vec3::new(0.0, 0.0, -room_size), Vec3::new(0.0, 0.0, 1.0), white);

    // Left wall (red)
    scene.add_plane(Vec3::new(-room_size, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), red);

    // Right wall (green)
    scene.add_plane(Vec3::new(room_size, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), green);

    // Front
    scene.add_plane(Vec3::new(0.0, 0.0, room_size), Vec3::new(0.0, 0.0, -1.0), white);

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

    // Hanging light (small emissive sphere)
    scene.add_sphere(Vec3::new(0.65, 0.1, 1.5), 0.2, light_mat);
    scene.add_light(LightData::from_sphere(
        Vec3::new(0.65, 0.1, 1.5), 0.2, emision,
    ));

    //// Ceiling light panel (one-sided: emits along its normal, i.e. downward)
    //scene.add_quad(Vec3::new(0.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, light_mat);
    //scene.add_light(LightData::from_quad(
    //        Vec3::new(0.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emision,
    //));

    //scene.add_quad(Vec3::new(-15.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, light_mat);
    //scene.add_light(LightData::from_quad(
    //        Vec3::new(-15.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emision,
    //));

    //scene.add_quad(Vec3::new(15.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, light_mat);
    //scene.add_light(LightData::from_quad(
    //        Vec3::new(15.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emision,
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
    // Windowed frontend: winit window + wgpu surface, GPU-only.
    if std::env::args().any(|a| a != "--terminal") {
        let scene = build_scene();
        let cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.7, 0.0), 60.0);
        match window::run_window(scene, cam) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("Windowed frontend unavailable ({}), falling back to terminal UI", e);
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

    let mut width = term_w;
    let mut height = term_h * 2;

    let scene = build_scene();

    // Prefer the GPU backend; fall back to the CPU when no adapter exists.
    let mut renderer: Box<dyn Renderer> = match gpu::GpuRenderer::new(&scene, width as u32, height as u32) {
        Ok(gpu) => {
            eprintln!("GPU backend: {}", gpu.adapter_info());
            Box::new(gpu)
        }
        Err(e) => {
            eprintln!("GPU unavailable ({}), falling back to CPU", e);
            Box::new(renderer::CpuRenderer::new(&scene, width, height, SAMPLES_PREVIEW))
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
                crossterm::event::Event::Key(key) => match key.code {
                    crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => break,
                    crossterm::event::KeyCode::Char('f') => {
                        filter_enabled = !filter_enabled;
                        needs_render = true;
                    },
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
                    },
                    crossterm::event::KeyCode::Char('p') => {
                        write!(stdout, "\r\nRendering FHD screenshot... ").unwrap();
                        stdout.flush()?;

                        // Render the FHD image with a dedicated renderer at
                        // 1920x1080. On the GPU this is a single dispatch per
                        // sample batch; on the CPU it renders in bands with a
                        // progress bar.
                        let fhd_buffer = render_fhd(&scene, &cam, filter_enabled);

                        match fhd_buffer.save_as_png("screenshot.png") {
                            Ok(()) => write!(stdout, "Saved to screenshot.png!").unwrap(),
                            Err(e) => write!(stdout, "Failed to save: {}", e).unwrap(),
                        }
                        stdout.flush()?;
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        // Redraw the preview over the progress-bar text.
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('w') => {
                        cam.move_forward(0.1);
                        renderer.reset_accumulation();
                        frames_accumulated = 0;
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('s') => {
                        cam.move_forward(-0.1);
                        renderer.reset_accumulation();
                        frames_accumulated = 0;
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('a') => {
                        cam.rotate(0.05);
                        renderer.reset_accumulation();
                        frames_accumulated = 0;
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('d') => {
                        cam.rotate(-0.05);
                        renderer.reset_accumulation();
                        frames_accumulated = 0;
                        needs_render = true;
                    },
                    _ => {}
                },
                crossterm::event::Event::Resize(w, h) => {
                    // Terminal resized: clear stale characters from the
                    // previous (possibly larger) frame and rebuild buffers.
                    crossterm::execute!(stdout, crossterm::terminal::Clear(crossterm::terminal::ClearType::All))?;
                    width = w as usize;
                    height = h as usize * 2;
                    // Rebuild the renderer at the new resolution.
                    renderer = match gpu::GpuRenderer::new(&scene, width as u32, height as u32) {
                        Ok(gpu) => Box::new(gpu),
                        Err(_) => Box::new(renderer::CpuRenderer::new(&scene, width, height, SAMPLES_PREVIEW)),
                    };
                    frames_accumulated = 0;
                    needs_render = true;
                },
                _ => {}
            }
        }
    }

    crossterm::execute!(stdout, crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

/// Render the FHD screenshot. Tries the GPU first (single dispatch per
/// sample batch, no progress bar needed), falls back to the CPU band
/// renderer with a progress bar.
fn render_fhd(scene: &SceneData, cam: &Cam, filter_enabled: bool) -> PixelBuffer {
    const FHD_W: usize = 1920;
    const FHD_H: usize = 1080;

    if let Ok(mut gpu) = gpu::GpuRenderer::new(scene, FHD_W as u32, FHD_H as u32) {
        // Accumulate SAMPLES_FHD samples on the GPU in batched dispatches
        // (one submit per batch instead of one per sample), then read back.
        const SPP_BATCH: u32 = 32;
        let mut remaining = SAMPLES_FHD as u32;
        while remaining > 0 {
            let spp = SPP_BATCH.min(remaining);
            gpu.render_frame_spp(cam, MAX_DEPTH_FHD as u32, spp);
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
    let aspect = FHD_W as f32 / FHD_H as f32;
    let theta = cam.fov_deg.to_radians();
    let h = (theta * 0.5).tan();
    let w = aspect * h;

    let forward = (cam.lookat - cam.origin).normalize();
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = world_up.cross(forward).normalize();
    let up = forward.cross(right).normalize();

    let band = 32;
    let total_pixels = FHD_W * FHD_H;
    let mut pixels: Vec<Vec3> = Vec::with_capacity(total_pixels);

    for y0 in (0..FHD_H).step_by(band) {
        let y_end = (y0 + band).min(FHD_H);
        let rows: Vec<Vec<Vec3>> = (y0..y_end)
            .into_par_iter()
            .map(|y| {
                let mut rng = rand::thread_rng();
                (0..FHD_W)
                    .into_iter()
                    .map(|x| {
                        let mut color = VEC_ZERO;
                        for _ in 0..SAMPLES_FHD {
                            let u_jitter = (x as f32 + rng.gen::<f32>()) / FHD_W as f32;
                            let v_jitter = (y as f32 + rng.gen::<f32>()) / FHD_H as f32;
                            let px = (u_jitter * 2.0 - 1.0) * w;
                            let py = -(v_jitter * 2.0 - 1.0) * h;
                            let dir = (right * px + up * py + forward).normalize();
                            let ray = crate::geometry::Ray { origin: cam.origin, direction: dir };
                            color = color + crate::bdpt::bdpt_trace(&ray, &*cpu_scene, MAX_DEPTH_FHD, &lights, &mut rng);
                        }
                        color / SAMPLES_FHD as f32
                    })
                    .collect()
            })
            .collect();
        pixels.extend(rows.into_iter().flatten());

        let rendered = y_end * FHD_W;
        print!("\r{}", PixelBuffer::render_progress(rendered, total_pixels));
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }

    println!("{}", PixelBuffer::render_progress(total_pixels, total_pixels));

    let mut buffer = PixelBuffer {
        width: FHD_W,
        height: FHD_H,
        pixels,
    };
    if filter_enabled {
        crate::output::apply_filters(&mut buffer);
    }
    buffer
}
