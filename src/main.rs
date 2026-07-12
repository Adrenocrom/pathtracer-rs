use crate::math::{Vec3, VEC_ZERO};
use crate::geometry::{Intersectable, MaterialType, Sphere, Plane, Cube};
use crate::bvh::BVHNode;
use crate::camera::Cam;
use rand::Rng;

// --- CONFIGURATION ---
const SAMPLES_PREVIEW: usize = 64; 
const MAX_DEPTH_PREVIEW: i32 = 1;
const SAMPLES_FHD: usize = 4096;
const MAX_DEPTH_FHD: i32 = 1;

/// Build the scene with multiple objects for interesting lighting.
fn build_scene() -> (Box<dyn Intersectable>, Vec<(Vec3, f32)>) {
    let white = crate::geometry::Material { albedo: Vec3::new(0.8, 0.8, 0.8), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let red = crate::geometry::Material { albedo: Vec3::new(1.5, 0.2, 0.1), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let green = crate::geometry::Material { albedo: Vec3::new(0.2, 1.0, 0.2), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let blue = crate::geometry::Material { albedo: Vec3::new(0.2, 0.4, 1.5), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let yellow = crate::geometry::Material { albedo: Vec3::new(1.5, 1.2, 0.1), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let light_mat = crate::geometry::Material { albedo: Vec3::new(0.5, 0.5, 0.5), emission: Vec3::new(3.0, 2.8, 1.5), mat_type: MaterialType::Emissive };

    let mut objects: Vec<Box<dyn Intersectable>> = vec![
        // Floor - large plane with subtle gradient feel
        Box::new(Plane { point: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(0.0, 1.0, 0.0), mat: white }),
        
        // Back wall
        Box::new(Plane { point: Vec3::new(0.0, 2.0, -2.5), normal: Vec3::new(0.0, 0.0, 1.0), mat: blue }),
        
        // Left wall (red)
        Box::new(Plane { point: Vec3::new(-2.5, 1.0, 0.0), normal: Vec3::new(1.0, 0.0, 0.0), mat: red }),
        
        // Right wall (green)
        Box::new(Plane { point: Vec3::new(2.5, 1.0, 0.0), normal: Vec3::new(-1.0, 0.0, 0.0), mat: green }),

        // Ceiling light panel
        Box::new(Plane { point: Vec3::new(0.0, 2.0, 0.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: light_mat.clone() }),

        // Central sphere (reflective-looking diffuse)
        Box::new(Sphere { center: Vec3::new(0.0, 0.5, 0.0), radius: 0.5, mat: yellow }),

        // Small spheres for caustics-like effect
        Box::new(Sphere { center: Vec3::new(-1.2, 0.3, -0.5), radius: 0.3, mat: blue }),
        Box::new(Sphere { center: Vec3::new(1.2, 0.3, -0.5), radius: 0.3, mat: red }),

        // Stacked cubes for interesting shadows
        Box::new(Cube::new(Vec3::new(-0.8, 0.4, 0.8), 0.6, white)),
        Box::new(Cube::new(Vec3::new(0.9, 0.4, 0.8), 0.6, green)),
        Box::new(Cube::new(Vec3::new(-0.2, 1.0, -0.5), 0.4, red)),

        // Hanging light (small emissive sphere)
        Box::new(Sphere { center: Vec3::new(0.0, 1.8, 0.5), radius: 0.1, mat: light_mat }),
    ];

    let scene = BVHNode::build(objects);
    
    // Return lights for BDPT sampling (position + area)
    let lights = vec![
        (Vec3::new(0.0, 2.0, 0.0), 5.0),   // Ceiling panel
        (Vec3::new(0.0, 1.8, 0.5), 2.0),   // Hanging bulb
    ];

    (scene, lights)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;

    let (term_w, term_h) = {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 40));
        (w as usize, h as usize)
    };

    let width = term_w;
    let height = term_h * 2;

    let (scene, lights) = build_scene();

    let mut cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.5, 0.0), 90.0);
    let mut needs_render = true;
    let mut filter_enabled = true;

    // Render with progress reporting (used for FHD)
    fn render_with_progress(
        cam: &Cam,
        scene: &dyn crate::geometry::Intersectable,
        width: usize,
        height: usize,
        samples: usize,
        max_depth: i32,
        lights: &[(Vec3, f32)],
    ) -> crate::output::PixelBuffer {
        let aspect = width as f32 / height as f32;
        let theta = cam.fov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let w = aspect * h;

        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

        // Render rows sequentially so we can report progress per row
        let total_pixels = width * height;
        let mut pixels = Vec::with_capacity(total_pixels);
        for y in 0..height {
            let mut rng = rand::thread_rng();
            for x in 0..width {
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
                pixels.push(color / samples as f32);
            }

            // Report progress every 10 rows to avoid excessive I/O
            if y % 10 == 0 {
                let rendered = (y + 1) * width;
                print!("\r{}", crate::output::PixelBuffer::render_progress(rendered, total_pixels));
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }

        // Final progress update
        println!("{}", crate::output::PixelBuffer::render_progress(total_pixels, total_pixels));

        crate::output::PixelBuffer { width, height, pixels }
    }

    loop {
        if needs_render {
            let buffer = cam.render(&*scene, width, height, SAMPLES_PREVIEW, MAX_DEPTH_PREVIEW, &lights);

            if filter_enabled {
                // Fix: apply filters directly to the real buffer (not a clone)
                crate::output::apply_filters(&mut buffer.clone());
            }

            let frame_string = buffer.to_string();
            crossterm::execute!(stdout, crossterm::cursor::MoveTo(0, 0))?;
            write!(stdout, "{}", frame_string).unwrap();
            stdout.flush()?;
            needs_render = false;
        }

        if crossterm::event::poll(std::time::Duration::from_millis(16))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                match key.code {
                    crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => break,
                    crossterm::event::KeyCode::Char('f') => {
                        filter_enabled = !filter_enabled;
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('h') => {
                        // Show help overlay
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
                            \x1b[1;36m╚══════════════════════════════════════╝\x1b[0m\n";
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
                    },
                    crossterm::event::KeyCode::Char('w') => {
                        cam.move_forward(0.1);
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('s') => {
                        cam.move_forward(-0.1);
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('a') => {
                        cam.rotate(0.05);
                        needs_render = true;
                    },
                    crossterm::event::KeyCode::Char('d') => {
                        cam.rotate(-0.05);
                        needs_render = true;
                    },
                    _ => {}
                }
            }
        }
    }

    crossterm::execute!(stdout, crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}