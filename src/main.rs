mod bdpt;
mod bvh;
mod camera;
mod geometry;
mod math;
mod output;

use crate::bvh::BVHNode;
use crate::camera::Cam;
use crate::geometry::{AreaLight, Cube, Intersectable, Material, MaterialType, Plane, Quad, Sphere};
use crate::math::{Vec3, VEC_ZERO};
use crate::output::PixelBuffer;
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;

// --- CONFIGURATION ---
const SAMPLES_PREVIEW: usize = 16;
const MAX_DEPTH_PREVIEW: usize = 4;
const SAMPLES_FHD: usize = 64;
const MAX_DEPTH_FHD: usize = 6;

/// Build the scene with multiple objects for interesting lighting.
fn build_scene() -> (Box<dyn Intersectable>, Vec<AreaLight>) {
    let white = Material { albedo: Vec3::new(0.8, 0.8, 0.8), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let red = Material { albedo: Vec3::new(0.75, 0.25, 0.25), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let green = Material { albedo: Vec3::new(0.25, 0.75, 0.25), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let blue = Material { albedo: Vec3::new(0.25, 0.4, 0.75), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let yellow = Material { albedo: Vec3::new(0.8, 0.7, 0.2), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let light_mat = Material { albedo: VEC_ZERO, emission: Vec3::new(3.0, 2.8, 1.5), mat_type: MaterialType::Emissive };

    // Ceiling light panel (finite quad, so it can be area-sampled).
    let ceiling_light = Quad::new(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        1.5,
        1.5,
        light_mat,
    );

    let objects: Vec<Box<dyn Intersectable>> = vec![
        // Floor - large plane
        Box::new(Plane { point: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(0.0, 1.0, 0.0), mat: white }),

        // Back wall
        Box::new(Plane { point: Vec3::new(0.0, 2.0, -2.5), normal: Vec3::new(0.0, 0.0, 1.0), mat: blue }),

        // Left wall (red)
        Box::new(Plane { point: Vec3::new(-2.5, 1.0, 0.0), normal: Vec3::new(1.0, 0.0, 0.0), mat: red }),

        // Right wall (green)
        Box::new(Plane { point: Vec3::new(2.5, 1.0, 0.0), normal: Vec3::new(-1.0, 0.0, 0.0), mat: green }),

        // Ceiling light panel
        Box::new(ceiling_light.clone()),

        // Central sphere
        Box::new(Sphere { center: Vec3::new(0.0, 0.5, 0.0), radius: 0.5, mat: yellow }),

        // Small spheres
        Box::new(Sphere { center: Vec3::new(-1.2, 0.3, -0.5), radius: 0.3, mat: blue }),
        Box::new(Sphere { center: Vec3::new(1.2, 0.3, -0.5), radius: 0.3, mat: red }),

        // Stacked cubes for interesting shadows
        Box::new(Cube::new(Vec3::new(-0.8, 0.4, 0.8), 0.6, white)),
        Box::new(Cube::new(Vec3::new(0.9, 0.4, 0.8), 0.6, green)),
        Box::new(Cube::new(Vec3::new(-0.2, 1.0, -0.5), 0.4, red)),

        // Hanging light (small emissive sphere)
        Box::new(Sphere { center: Vec3::new(0.0, 1.8, 0.5), radius: 0.1, mat: light_mat }),
    ];

    let scene = BVHNode::build(objects).expect("scene is non-empty");

    // Area lights must mirror the emissive geometry so BDPT can connect
    // paths to real surfaces.
    let lights = vec![
        AreaLight::from_quad(&ceiling_light),
        AreaLight::from_sphere(&Sphere { center: Vec3::new(0.0, 1.8, 0.5), radius: 0.1, mat: light_mat }),
    ];

    (scene, lights)
}

/// Progressive temporal accumulation: average the new frame into `accum`.
/// `frame_count` is the number of frames already accumulated (>= 1).
fn accumulate(accum: &mut [Vec3], frame: &[Vec3], frame_count: usize) {
    let n = frame_count as f32;
    for (a, f) in accum.iter_mut().zip(frame) {
        *a = *a * (n / (n + 1.0)) + *f * (1.0 / (n + 1.0));
    }
}

/// How many preview frames to average while the camera is stationary.
const MAX_ACCUMULATED_FRAMES: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn scene_builds_and_is_sampleable() {
        let (scene, lights) = build_scene();
        assert_eq!(lights.len(), 2);
        for light in &lights {
            assert!(light.area() > 0.0);
            assert!(light.emission.x > 0.0 || light.emission.y > 0.0 || light.emission.z > 0.0);
        }
        // A ray from the camera position must hit the scene.
        let ray = crate::geometry::Ray {
            origin: Vec3::new(0.0, 1.0, -1.5),
            direction: Vec3::new(0.0, 0.0, 1.0),
        };
        assert!(scene.intersect(&ray).is_some());
    }

    #[test]
    fn preview_render_produces_finite_pixels() {
        let (scene, lights) = build_scene();
        let cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.5, 0.0), 90.0);
        let buffer = cam.render(&*scene, 16, 16, 4, 4, &lights);
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
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;

    let (term_w, term_h) = {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 40));
        (w as usize, h as usize)
    };

    let mut width = term_w;
    let mut height = term_h * 2;

    let (scene, lights) = build_scene();

    let mut cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.5, 0.0), 90.0);
    let mut needs_render = true;
    let mut filter_enabled = true;
    let mut showing_help = false;

    // Temporal accumulation state.
    let mut accum: Option<Vec<Vec3>> = None;
    let mut frames_accumulated: usize = 0;

    // Render with progress reporting (used for FHD)
    fn render_with_progress(
        cam: &Cam,
        scene: &dyn crate::geometry::Intersectable,
        width: usize,
        height: usize,
        samples: usize,
        max_depth: usize,
        lights: &[crate::geometry::AreaLight],
    ) -> PixelBuffer {
        let aspect = width as f32 / height as f32;
        let theta = cam.fov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let w = aspect * h;

        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

        // Render in horizontal bands so we can report progress per band.
        let band = 32;
        let total_pixels = width * height;
        let mut pixels: Vec<Vec3> = Vec::with_capacity(total_pixels);

        for y0 in (0..height).step_by(band) {
            let y_end = (y0 + band).min(height);
            let rows: Vec<Vec<Vec3>> = (y0..y_end)
                .into_par_iter()
                .map(|y| {
                    let mut rng = rand::thread_rng();
                    (0..width)
                        .into_iter()
                        .map(|x| {
                            let mut color = VEC_ZERO;
                            for _ in 0..samples {
                                let u_jitter = (x as f32 + rng.gen::<f32>()) / width as f32;
                                let v_jitter = (y as f32 + rng.gen::<f32>()) / height as f32;
                                let px = (u_jitter * 2.0 - 1.0) * w;
                                let py = -(v_jitter * 2.0 - 1.0) * h;
                                let dir = (right * px + up * py + forward).normalize();
                                let ray = crate::geometry::Ray { origin: cam.origin, direction: dir };
                                color = color + crate::bdpt::bdpt_trace(&ray, scene, max_depth, lights, &mut rng);
                            }
                            color / samples as f32
                        })
                        .collect()
                })
                .collect();
            pixels.extend(rows.into_iter().flatten());

            let rendered = y_end * width;
            print!("\r{}", PixelBuffer::render_progress(rendered, total_pixels));
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        println!("{}", PixelBuffer::render_progress(total_pixels, total_pixels));

        PixelBuffer { width, height, pixels }
    }

    loop {
        // Progressive rendering: keep accumulating while the camera is
        // stationary so the image cleans up over time. Cap the accumulation
        // so the average stops changing once it is converged enough.
        if !showing_help && (needs_render || frames_accumulated < MAX_ACCUMULATED_FRAMES) {
            let frame = cam.render(&*scene, width, height, SAMPLES_PREVIEW, MAX_DEPTH_PREVIEW, &lights);

            // Temporal accumulation: average successive frames while the
            // camera is stationary, which progressively cleans up the noise.
            match &mut accum {
                Some(a) if a.len() == frame.pixels.len() => {
                    accumulate(a, &frame.pixels, frames_accumulated);
                    frames_accumulated += 1;
                }
                _ => {
                    accum = Some(frame.pixels.clone());
                    frames_accumulated = 1;
                }
            }

            let mut buffer = PixelBuffer {
                width,
                height,
                pixels: accum.clone().unwrap(),
            };

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

                        let mut fhd_buffer = render_with_progress(
                            &cam, &*scene, 1920, 1080, SAMPLES_FHD, MAX_DEPTH_FHD, &lights,
                        );

                        if filter_enabled {
                            crate::output::apply_filters(&mut fhd_buffer);
                        }
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
                        accum = None;
                        frames_accumulated = 0;
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('s') => {
                        cam.move_forward(-0.1);
                        accum = None;
                        frames_accumulated = 0;
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('a') => {
                        cam.rotate(0.05);
                        accum = None;
                        frames_accumulated = 0;
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('d') => {
                        cam.rotate(-0.05);
                        accum = None;
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
                    accum = None;
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