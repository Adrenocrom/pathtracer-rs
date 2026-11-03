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

    // Central spheres: one diffuse at the left edge, and the glass sphere
    // dead center — the caustic showcase (see the light below).
    scene.add_sphere(Vec3::new(-0.9, 0.4, -0.9), 0.4, white);
    // The glass sphere floats 0.2 above the floor instead of touching it:
    // at exact contact, a ray refracted out of the bottom cap would start
    // (after the EPS_NEAR ray offset) below the floor plane and miss it,
    // punching a hole in the center of the caustic. The height also keeps
    // the caustic center visible from the default camera: a sphere resting
    // on the floor hides the floor point directly beneath it behind its
    // own silhouette.
    scene.add_sphere(Vec3::new(-0.4, 0.5, 0.1), 0.3, MaterialData::glass(1.5));

    scene.add_cube(Vec3::new(1.0, 0.3, -0.8), 0.6, white);
    scene.add_sphere(Vec3::new(1.0, 1.0, -0.8), 0.4, white);

    // Stacked cubes for interesting shadows
    scene.add_cube(Vec3::new(0.3, 0.3, 0.8), 0.6, white);
    scene.add_cube(Vec3::new(0.94, 0.3, 0.8), 0.6, white);
    scene.add_cube(Vec3::new(1.58, 0.3, 0.8), 0.6, white);

    scene.add_cube(Vec3::new(0.45, 0.9, 0.8), 0.6, white);
    scene.add_cube(Vec3::new(1.09, 0.9, 0.8), 0.6, white);

    // Chrome sphere resting on the cube stack (its bottom touches the top
    // face of the right upper cube at y = 1.2): reflects the red/green
    // walls, the ceiling light and the glass sphere — the mirror BSDF
    // showcase.
    scene.add_sphere(
        Vec3::new(1.09, 1.45, 0.8),
        0.25,
        MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
    );

    // The old hanging light (small emissive sphere, directly above the
    // glass sphere) is replaced by the ceiling panel below; kept here
    // for reference. `add_light` registers the emitter as geometry too.
    //scene.add_light(LightData::from_sphere(
    //    Vec3::new(-0.4, 1.2, 0.1),
    //    0.2,
    //    emission,
    //));

    // Ceiling light panel (one-sided: emits along its normal, i.e.
    // downward). `add_light` registers the quad as geometry too. It
    // shines down on the glass sphere at a shallow off-axis angle
    // (~15°), so the ball lens focuses its light into a caustic on the
    // floor beside the sphere's shadow — the Phase 1 payoff
    // (paper.md §6.2): a light path the unidirectional tracer cannot
    // find. (The old hanging sphere light sat directly overhead; the
    // panel is the same showcase with a lamp that is not in frame.)
    scene.add_light(LightData::from_quad(
            Vec3::new(0.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,
    ));

    // Laser light source: a cylinder housing whose top cap (the circular
    // face) is the aperture, emitting a narrow beam along the cylinder's
    // axis. The housing hangs from the ceiling (its base cap sunk 1 cm
    // into the ceiling plane, so the ceiling quad always shades the mount
    // patch — a base cap exactly coplanar with the ceiling would tie at
    // the same t, and the winner would depend on BVH ordering); the beam
    // exits the bottom circular face and paints a bright red disc on the
    // open floor left of the glass sphere and the cube stack. Both the
    // housing and the spot are in frame from the default camera, and the
    // sight lines to both clear every object — the first placement, at
    // (0.8, ·, 1.5), painted the floor *behind* the cube stack, so the
    // spot was occluded and the beam invisible.
    //
    // The housing carries a normal diffuse material, so it shades like
    // any other object; the aperture is a thin puck of laser material
    // whose top cap radiates into the cone (the rest of the puck is
    // hidden inside the housing). The aperture is dark from the camera
    // too — physically correct: the emission cone is ±1.1° around the
    // downward normal and the camera sits ~85° off-axis, so the
    // aperture glows only when viewed from inside the beam (e.g. a
    // floor-level camera near the spot). The visible showcase is the
    // shaded housing plus the blazing red floor spot.
    // The radiance is enormous on purpose. The cone is so narrow that a
    // floor point sees only a tiny central patch of the aperture inside
    // the beam: the spot's irradiance is L·π·tan²θ, ~50 W/m² here —
    // at the old emission (8000, 160, 160) the spot's radiance was
    // ~2.6 against ~1.8 of ambient floor, an invisible ~20% bump. A
    // real laser's radiance is astronomically high; 5× the emission
    // (same chromaticity) is what makes the narrow beam deposit a
    // clearly visible spot.
    scene.add_laser(
        Vec3::new(-1.2, room_height - 0.34, 1.0), // housing center, base sunk into the ceiling
        Vec3::new(0.0, -1.0, 0.0),                // beam straight down
        0.2,                                      // housing height
        0.12,                                    // aperture radius
        0.02,                                    // beam half-angle (radians, ~1.1°)
        Vec3::new(40000.0, 800.0, 800.0),        // red laser
    );

    scene
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_builds_and_is_sampleable() {
        let scene = build_scene();
        // The demo scene has two emitters: the ceiling panel and the
        // laser aperture.
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

    /// The caustic showcase (plan §2.7): the light must illuminate the
    /// glass sphere from above so its light refracts through the ball
    /// and focuses onto the floor below, and the sphere plus the
    /// caustic point must be in frame from the default camera. This
    /// pins the geometry the screenshot acceptance criterion depends
    /// on — moving the light off the sphere or the sphere out of view
    /// silently kills the caustic.
    ///
    /// The scene's emitter is a one-sided ceiling quad panel (the
    /// hanging sphere light it replaced sat directly overhead); it
    /// shines down on the sphere at a shallow off-axis angle (~15°),
    /// so the caustic lands on the floor near the sphere's shadow. The
    /// lamp itself is *not* in frame from the default camera (it sits
    /// above the top edge of the view) — the caustic is the showcase,
    /// not the lamp, so the in-frame checks cover the sphere and the
    /// caustic point (the light→sphere axis crossing on the floor).
    #[test]
    fn caustic_geometry_is_in_frame() {
        let scene = build_scene();
        let light = &scene.lights[0];
        // The glass sphere: the only non-diffuse, non-emissive sphere
        // with ior != 1.
        let glass = scene
            .spheres
            .iter()
            .find(|s| s.mat.bsdf == crate::geometry::Bsdf::Glass { ior: 1.5 })
            .expect("demo scene must contain a glass sphere");
        // The light shines down on the glass sphere from above, at a
        // shallow enough off-axis angle that the sphere sits well inside
        // its beam (the old hanging sphere light was directly overhead,
        // horiz = 0; the ceiling panel is ~15° — both focus a caustic
        // below the ball).
        let to_sphere = glass.center - light.center;
        assert!(to_sphere.y < 0.0, "light not above the glass sphere");
        let horiz = (to_sphere.x * to_sphere.x + to_sphere.z * to_sphere.z).sqrt();
        assert!(
            horiz < to_sphere.y.abs() * 0.5,
            "light too far off-axis to focus a caustic through the sphere \
             (horiz {horiz}, height {})",
            to_sphere.y.abs()
        );
        // The glass sphere floats above the floor (not touching it), so
        // refracted rays leaving the bottom cap can still hit the floor
        // after the EPS_NEAR offset.
        assert!(
            glass.center.y - glass.radius > 0.01,
            "glass sphere touches the floor"
        );

        // The sphere and the caustic point must be in frame from the
        // default camera. The camera looks at (0, 0.7, 0) from
        // (0, 1, -1.5) with a 60° fov: a point is in frame when its
        // angle from the forward axis is under the half fov (30°) in
        // both the horizontal and vertical projection. The caustic
        // point is the light→sphere axis crossing on the floor (a ball
        // lens focuses on its axis; with the off-axis ceiling panel
        // that is beside the sphere's shadow, not directly below the
        // sphere).
        let cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.7, 0.0), 60.0);
        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        let half_fov = (cam.fov_deg * 0.5).to_radians().cos();
        let axis = (glass.center - light.center).normalize();
        let caustic = glass.center + axis * (-glass.center.y / axis.y);
        for (name, p) in [("glass", glass.center), ("caustic", caustic)] {
            let rel = p - cam.origin;
            let f = rel.dot(forward);
            assert!(f > 0.0, "{name} is behind the camera");
            let cos_h = f / (rel.dot(right).abs().hypot(f));
            let cos_v = f / (rel.dot(up).abs().hypot(f));
            assert!(
                cos_h >= half_fov && cos_v >= half_fov,
                "{name} out of frame (cos_h {cos_h}, cos_v {cos_v}, need >= {half_fov})"
            );
        }

        // The caustic center must be visible from the camera, not hidden
        // behind the glass sphere's own silhouette: the distance from the
        // sphere center to the camera→caustic segment must exceed the
        // radius. (A sphere resting on the floor fails this — the floor
        // point directly beneath it is behind the ball.)
        let seg = caustic - cam.origin;
        let to_center = glass.center - cam.origin;
        let t = (to_center.dot(seg) / seg.dot(seg)).clamp(0.0, 1.0);
        let closest = cam.origin + seg * t;
        let dist = (glass.center - closest).length();
        assert!(
            dist > glass.radius,
            "caustic center hidden behind the glass sphere (dist {dist} <= r {})",
            glass.radius
        );
    }

    /// The laser showcase: the housing (a cylinder hanging from the
    /// ceiling) and the beam's floor spot must both be in frame from the
    /// default camera, and the spot must be on open floor (not hidden
    /// behind an object). This pins the geometry the visual acceptance
    /// depends on — a laser that is out of frame or whose spot is
    /// occluded silently produces no visible beam.
    #[test]
    fn laser_housing_and_spot_are_in_frame() {
        let scene = build_scene();
        // The laser: the only shape-2 light.
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("demo scene must contain a laser");
        // The housing: the non-laser cylinder (the other one is the
        // aperture puck, hidden inside it).
        let cyl = scene
            .cylinders
            .iter()
            .find(|c| !c.mat.is_laser())
            .expect("the laser housing");
        let aperture = laser.center;

        let cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.7, 0.0), 60.0);
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

        // The housing's midpoint and the aperture must be in frame.
        let mid = cyl.base + cyl.axis * cyl.half_h;
        in_frame("laser housing", mid);
        in_frame("laser aperture", aperture);

        // The beam spot: where the beam axis crosses the floor. The beam
        // points straight down in the demo, so the spot is directly
        // below the aperture.
        let spot = aperture + laser.normal * (-aperture.y / laser.normal.y);
        in_frame("laser spot", spot);

        // The spot must be visible from the camera, not hidden behind
        // the housing itself: the segment camera -> spot must not pass
        // through the cylinder.
        let (cpu_scene, _) = scene.build_cpu_scene();
        let seg = spot - cam.origin;
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

    /// Regenerate `screenshot.png` headlessly through the exact production
    /// path (`render_fhd` → `save_as_png`), so the Phase 1 acceptance
    /// artifact (the caustic visible in the screenshot) can be produced
    /// without pressing `P` in the interactive UI. Also probes the raw
    /// HDR buffer for the caustic: a bright spot on the floor near the
    /// glass sphere, brighter than the surrounding floor.
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
        let cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.7, 0.0), 60.0);

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

        // --- Caustic probe: the floor around the glass sphere must show a
        // bright focused spot. A ball lens focuses on its axis, so the
        // caustic lands where the light→sphere axis crosses the floor —
        // with the ceiling panel ~15° off vertical that point sits beside
        // the sphere's shadow, not directly below the sphere (where the
        // old overhead light focused it, and which is now the umbra).
        // Compare the mean HDR value in a small disc at that axis
        // crossing against a nearby fully-lit floor disc: the focused
        // light must be brighter by a clear margin — several times the
        // direct illumination there.
        let glass = scene
            .spheres
            .iter()
            .find(|s| s.mat.bsdf == crate::geometry::Bsdf::Glass { ior: 1.5 })
            .expect("glass sphere");
        let light = &scene.lights[0];
        let axis = (glass.center - light.center).normalize();
        let caustic_world = glass.center + axis * (-glass.center.y / axis.y);
        let (cx, cy) = project_to_pixel(&cam, caustic_world, buffer.width, buffer.height);
        // The reference: fully-lit floor 1.2 to the +x of the caustic,
        // outside the sphere's shadow (umbra ~0.4, penumbra ~0.6 for
        // this light/sphere geometry) and away from every object.
        let reference_world = caustic_world + Vec3::new(1.2, 0.0, 0.0);
        let (rx, ry) = project_to_pixel(&cam, reference_world, buffer.width, buffer.height);
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
                    sum += buffer.get_pixel(x as usize, y as usize).x;
                    n += 1;
                }
            }
            sum / n.max(1) as f32
        };
        let center_mean = disc_mean(cx, cy);
        let reference_mean = disc_mean(rx, ry);
        // Diagnostic dump (the harness hides stdout).
        std::fs::write(
            "target/caustic_probe.txt",
            format!(
                "caustic ({cx},{cy}) mean {center_mean}\nreference ({rx},{ry}) mean {reference_mean}\nratio {}\n",
                center_mean / reference_mean
            ),
        )
        .ok();
        // The focused light concentrates the light's power into a small
        // disc: several times the direct illumination on nearby floor.
        // 1.5x is a conservative floor for "clearly visible".
        assert!(
            center_mean > reference_mean * 1.5,
            "no caustic visible: center {center_mean} vs reference {reference_mean}"
        );

        // --- Laser spot probe: the beam's floor disc must be clearly
        // brighter (and redder) than the surrounding floor — the visual
        // acceptance of the laser showcase, pinned the same way as the
        // caustic. The spot sits directly below the aperture (the beam
        // points straight down); the reference is the floor disc 0.6 to
        // the +x, outside the beam disc (radius ~0.03) and away from
        // every object.
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("demo scene must contain a laser");
        let spot_world = laser.center + laser.normal * (-laser.center.y / laser.normal.y);
        let (sx, sy) = project_to_pixel(&cam, spot_world, buffer.width, buffer.height);
        let spot_ref_world = spot_world + Vec3::new(0.6, 0.0, 0.0);
        let (srx, sry) = project_to_pixel(&cam, spot_ref_world, buffer.width, buffer.height);
        let spot_mean = disc_mean(sx, sy);
        let spot_ref_mean = disc_mean(srx, sry);
        std::fs::write(
            "target/laser_probe.txt",
            format!(
                "laser spot ({sx},{sy}) mean {spot_mean}\nreference ({srx},{sry}) mean {spot_ref_mean}\nratio {}\n",
                spot_mean / spot_ref_mean
            ),
        )
        .ok();
        // The beam concentrates the aperture's power into a tiny disc:
        // the spot must be several times brighter than the ambient-lit
        // floor around it. 2x is a conservative floor for "clearly
        // visible" (the caustic probe uses 1.5x; the laser's narrower
        // concentration warrants the higher bar).
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
