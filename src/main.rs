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
mod spectrum;
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

/// The lasers' aperture radius — the beam waist w₀ (m). The physical
/// divergence is θ = M²·λ/(π·w₀) and the spot irradiance is
/// Φ/(π·w₀²), both derived from this one constant.
const LASER_WAIST: f32 = 0.01;

/// Build the scene with multiple objects for interesting lighting.
///
/// `laser_aim_y` is the red laser's aim height on the front-wall mirror
/// (1.0 = the mirror's center) — the one knob the movement demo
/// animates (`M` in the window frontend): the laser pivots on its
/// table mount, scanning the beam up and down the mirror, and the
/// spot slides up and down the partition wall. The scene is rebuilt
/// from scratch per frame, which is exactly the contract the bridge
/// ladder derivation supports (it is never stored).
fn build_scene() -> SceneData {
    build_scene_with(1.0)
}

fn build_scene_with(_laser_aim_y: f32) -> SceneData {
    let emission = Vec3::new(1000.0, 1000.0, 1000.0);
    let room_size = 2.5;
    let room_height = 2.0;

    let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
    let red = MaterialData::diffuse(Vec3::new(0.75, 0.25, 0.25));
    let green = MaterialData::diffuse(Vec3::new(0.25, 0.75, 0.25));
    let brown = MaterialData::diffuse(Vec3::new(0.2, 0.25, 0.25));

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
    scene.add_light(LightData::from_quad( Vec3::new(0.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,));
    scene.add_light(LightData::from_quad( Vec3::new(2.0, room_height - 0.01, 2.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,));

    scene.add_box( Vec3::new(room_size/2.0, 1.0, 1.0), Vec3::new(room_size, room_height, 0.02), white);
    scene.add_shelf(Vec3::new(0.4, 0.0, 0.8), Vec3::new(0.8, 1.5, 0.2), white, brown);

    scene.add_sphere(Vec3::new(0.0, 0.2, 0.0), 0.2, MaterialData::glass_bk7());
    scene.add_sphere(Vec3::new(-1.4, 0.2, -1.2), 0.2, MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)));

    // The laser showcase: a red laser on the table, aimed backward at a
    // mirror tile on the back wall, which bounces the beam forward across
    // the room onto the partition wall's camera-facing side. One
    // bounce, one spot — plus the glow beside the beam (the bridge
    // ladder carries it on *every* leg now).
    //
    // The laser is added *before* the mirror on purpose: the specular
    // bridge ladder (one rung per mirror the beam crosses — see
    // `SceneData::bridges_of_light`) is derived on demand from the
    // fully-built scene, never baked at add-time, so the registration
    // order is irrelevant — and a scene that moves between frames
    // simply re-derives every rung from the new geometry.
    //
    // `laser_aim_y` is the movement knob (the window frontend's `M`
    // key): the laser pivots on its table mount, scanning the beam up
    // and down the back-wall mirror; the spot slides up and down the
    // partition, and the derived ladder follows.
    //
    // Spectral (paper §9.2): the laser is a *monochromatic line* —
    // 650 nm, the physically correct replacement for the old RGB trick
    // of putting 4×10⁸ in the red channel. The out-of-gamut negative
    // channels clamp at display, where they belong.
    //
    // The beam model (the collimated-beam rethink): the laser is
    // power-parametrized — Φ = 50 mW, a bright presentation pointer —
    // with the *physical* divergence θ = M²·λ/(π·w₀) (M² = 30, a
    // deliberately imperfect beam so the spot grows visibly across
    // the room; a diffraction-limited M² = 1 would stay a constant
    // 1 cm disc, which is the physically correct but visually
    // uninformative limit). The spot irradiance E = Φ/(π·w₀²) ≈
    // 160 W/m² is θ-invariant — θ controls only the divergence, as
    // it physically does. The old radiance-parametrized form coupled
    // them (Φ ∝ Ω at fixed Le, so θ was secretly a brightness knob —
    // which is why it was left artificially wide, reading as a
    // spotlight).
    let laser_pos = Vec3::new(-1.3, 0.62, 1.8);
    let back_mirror_center = Vec3::new(0.0, 1.0, -(room_size - 0.01));
    //let aim_point = Vec3::new(0.0, laser_aim_y, -(room_size - 0.01));
    let aim = Vec3::new(1.0, 0.0, -room_size).normalize();
    // The red laser: 650 nm, 50 mW, M² = 30.
    let red_lambda = 650.0;
    let red_power_w = 0.05;
    let red_m2 = 30.0;
    let red_theta = red_m2 * red_lambda * 1e-9 / (std::f32::consts::PI * LASER_WAIST);
    scene.add_laser_beam(
        laser_pos,
        aim,
        0.2,                        // housing height
        LASER_WAIST,                // aperture radius = the beam waist
        red_theta,                  // physical divergence (rad)
        red_lambda,                 // nm — the red laser line
        red_power_w,                // W — the beam power
    );
    scene.add_quad(
        back_mirror_center,
        Vec3::new(0.0, 0.0, 1.0),
        2.0,
        1.0,
        MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
    );

    // The glass cube: N-BK7 borosilicate (paper §9.2) — a dispersive
    // dielectric, so its caustic picks up chromatic fringes (blue bends
    // more than red). The Sellmeier coefficients replace the constant
    // n = 1.5.
    scene.add_box(
        Vec3::new(-1.4, 0.65, 1.5),
        Vec3::new(0.1, 0.1, 0.1),
        MaterialData::glass_bk7(),
    );
    // The green laser's ping-pong tiles: the first is a GGX *rough
    // mirror* (paper §9.5) — the tile picks up soft reflections of the
    // room, the beam still lands (the trace follows the lobe's mean),
    // the ladder still derives rung-per-crossing, and the spot softens.
    // The second stays a perfect mirror — the visible contrast.
    scene.add_box(
        Vec3::new(-1.9, 0.65, 1.7),
        Vec3::new(0.001, 0.1, 0.2),
        MaterialData::glossy(Vec3::new(1.0, 1.0, 1.0), 0.08, 0.9),
    );
    scene.add_box( Vec3::new(-1.7, 0.65, 1.6), Vec3::new(0.001, 0.1, 0.4), MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)));

    let pos = Vec3::new(-1.7, 0.6, 1.7);
    let dim = Vec3::new(1.0, 0.02, 0.7);
    scene.add_table(pos, dim, white, brown);


    // A second laser (green) on the table, aimed across the room at the
    // thin mirror boxes — its beam ping-pongs between them, and the
    // wide cone (0.05 rad) paints the table under *every* leg: the glow
    // under the middle legs (after one box, before the next) is exactly
    // what the bridge ladder's rungs carry — before the ladder, only
    // the first leg (direct light) and the last leg (the whole-chain
    // rung) glowed.
    //
    // Spectral (paper §9.2): a 532 nm line — the classic green laser
    // wavelength. A pure 532 nm green is outside the sRGB gamut, so the
    // rendered sample carries negative R/B channels; the display stage
    // gamut-maps them (§9.2: "the out-of-gamut problem moves to the
    // display stage where it belongs").
    //
    // The beam model: 5 mW (a standard green pointer — green is
    // ~6× more luminous than 650 nm red per watt, so 5 mW of green
    // renders a spot of comparable luminance to 50 mW of red), M² = 30
    // — the same divergence law as the red laser.
    let green_lambda = 532.0;
    let green_power_w = 0.005;
    let green_m2 = 30.0;
    let green_theta = green_m2 * green_lambda * 1e-9 / (std::f32::consts::PI * LASER_WAIST);
    scene.add_laser_beam(
        Vec3::new(-1.6, 0.62, 1.9),
        Vec3::new(-1.0, 0.01, -0.5),
        0.2,
        LASER_WAIST,
        green_theta,
        green_lambda,               // nm — the green laser line
        green_power_w,              // W — the beam power
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

    #[test]
    fn scene_builds_and_is_sampleable() {
        let scene = build_scene();
        // The demo scene has four emitters: the two ceiling panels and
        // the two laser apertures. (No decal: the beams' spots are
        // transported light, not fake emitters.)
        assert_eq!(scene.lights.len(), 4);
        for light in &scene.lights {
            assert!(light.area() > 0.0);
            assert!(light.spectrum.is_emissive());
        }
        // The two lasers are monochromatic lines (paper §9.2): 650 nm
        // (red) and 532 nm (green).
        let lambdas: Vec<f32> = scene
            .lights
            .iter()
            .filter_map(|l| match l.spectrum {
                crate::spectrum::SpectrumKind::Line { lambda, .. } => Some(lambda),
                _ => None,
            })
            .collect();
        assert_eq!(lambdas.len(), 2, "the two lasers must be line emitters");
        assert!(lambdas.contains(&650.0), "the red laser: {lambdas:?}");
        assert!(lambdas.contains(&532.0), "the green laser: {lambdas:?}");
        // A ray from the camera position must hit the scene.
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let ray = crate::geometry::Ray {
            origin: Vec3::new(0.0, 1.0, -1.5),
            direction: Vec3::new(0.0, 0.0, 1.0),
        };
        assert!(cpu_scene.intersect(&ray).is_some());
        assert_eq!(lights.len(), 4);
    }

    /// DIAGNOSTIC (temporary): where do the demo beams land, and does the
    /// estimator produce light there? Traces both beams, prints the
    /// ladders, and probes the spot with the CPU estimator.
    /// Ignored by default: a temporary diagnostic for the ongoing
    /// GPU-bridge investigation (slow: it renders a full view on the
    /// CPU). Run with `cargo test -- --ignored diagnostic_demo_beams_and_spots`.
    #[test]
    #[ignore]
    fn diagnostic_demo_beams_and_spots() {
        let scene = build_scene();
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let bridges = scene.bridges();
        let mut out = String::new();
        out.push_str("=== demo beam diagnostics ===\n");
        out.push_str(&format!("n_bridges total: {}\n", bridges.len()));
        for light in &scene.lights {
            if light.shape != 2 {
                continue;
            }
            let lambda = match light.spectrum {
                crate::spectrum::SpectrumKind::Line { lambda, .. } => lambda,
                _ => 0.0,
            };
            let trace = scene.trace_laser_beam(light, 8);
            out.push_str(&format!(
                "laser λ={lambda} at {:?} dir {:?}: {} legs, spot={:?}\n",
                light.center,
                light.normal,
                trace.legs.len(),
                trace.spot.as_ref().map(|s| s.p)
            ));
            for (i, (a, b)) in trace.legs.iter().enumerate() {
                out.push_str(&format!("  leg {i}: {a:?} -> {b:?}\n"));
            }
            let ladder = scene.bridges_of_light(light, 8);
            out.push_str(&format!("  ladder: {} rungs\n", ladder.len()));
            for (i, r) in ladder.iter().enumerate() {
                out.push_str(&format!(
                    "  rung {i}: image {:?} r={} beam_dir {:?} versine {:.3e}\n",
                    r.image_center, r.radius, r.beam_dir, r.versine
                ));
            }
        }

        // Follow the green beam across the glossy tile the way the ladder
        // does (trace_laser_beam stops at the glossy surface because it
        // is non-delta): aim at rung 0's image, then continue along the
        // reversed rung-1 beam direction.
        let green = scene
            .lights
            .iter()
            .find(|l| match l.spectrum {
                crate::spectrum::SpectrumKind::Line { lambda, .. } => lambda == 532.0,
                _ => false,
            })
            .expect("the green laser");
        let green_ladder = scene.bridges_of_light(green, 8);
        if green_ladder.len() >= 2 {
            // Leg 1: aperture -> glossy tile. Aim from the aperture along
            // the beam axis.
            let mut origin = green.center + green.normal * 0.001;
            let mut dir = green.normal;
            let mut legs = Vec::new();
            for _ in 0..8 {
                let ray = crate::geometry::Ray { origin, direction: dir };
                let Some(hit) = cpu_scene.intersect(&ray) else { break };
                legs.push((origin, hit.p));
                let front_face = hit.normal.dot(dir) < 0.0;
                let normal = if front_face { hit.normal } else { hit.normal * -1.0 };
                match hit.mat.bsdf {
                    crate::geometry::Bsdf::Mirror | crate::geometry::Bsdf::Glossy { .. } => {
                        dir = crate::geometry::reflect_dir(dir, normal);
                        origin = hit.p + normal * 0.001;
                    }
                    _ => {
                        out.push_str(&format!(
                            "λ=532 full chain lands at {:?} (mat {:?}) after {} legs\n",
                            hit.p,
                            hit.mat.bsdf,
                            legs.len()
                        ));
                        for (i, (a, b)) in legs.iter().enumerate() {
                            out.push_str(&format!("  λ=532 leg {i}: {a:?} -> {b:?}\n"));
                        }
                        break;
                    }
                }
            }
        }

        // Render the default camera view and scan for the spots: the
        // brightest red-dominant and green-dominant pixels.
        let cam = default_cam();
        let (forward, right, up) = {
            let forward = (cam.lookat - cam.origin).normalize();
            let world_up = Vec3::new(0.0, 1.0, 0.0);
            let right = world_up.cross(forward).normalize();
            let up = forward.cross(right).normalize();
            (forward, right, up)
        };
        let aspect = 16.0 / 9.0;
        let theta = cam.fov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let w = aspect * h;
        let mut rng = rand::thread_rng();
        let n = 64;
        let mut best_red = (0.0f32, 0usize, 0usize);
        let mut best_green = (0.0f32, 0usize, 0usize);
        let mut red_sum = 0.0f32;
        let mut green_sum = 0.0f32;
        for y in 0..48 {
            for x in 0..85 {
                let px = (x as f32 / 85.0 * 2.0 - 1.0) * w;
                let py = -(y as f32 / 48.0 * 2.0 - 1.0) * h;
                let dir = (right * px + up * py + forward).normalize();
                let ray = crate::geometry::Ray { origin: cam.origin, direction: dir };
                let mut mean = VEC_ZERO;
                for _ in 0..n {
                    mean = mean
                        + crate::bdpt::bdpt_trace(&ray, &*cpu_scene, 8, &lights, &bridges, &mut rng);
                }
                mean = mean * (1.0 / n as f32);
                // Red-dominant: R much greater than G and B.
                if mean.x > mean.y * 3.0 && mean.x > mean.z * 3.0 && mean.x > best_red.0 {
                    best_red = (mean.x, x, y);
                }
                if mean.y > mean.x * 3.0 && mean.y > mean.z * 3.0 && mean.y > best_green.0 {
                    best_green = (mean.y, x, y);
                }
                red_sum += mean.x.max(0.0);
                green_sum += mean.y.max(0.0);
            }
        }
        out.push_str(&format!(
            "default view scan (85x48, {n} spp): best red pixel {best_red:?}, best green pixel {best_green:?}\n"
        ));
        out.push_str(&format!(
            "default view: total red {red_sum:.2}, total green {green_sum:.2}\n"
        ));
        std::fs::write("target/demo_beam_diag.txt", &out).ok();
    }

    /// DIAGNOSTIC (temporary): analyze the rendered screenshots for the
    /// laser spots — saturated red / green blobs.
    /// Ignored by default: a temporary diagnostic (it depends on
    /// screenshot files in the repo root). Run with
    /// `cargo test -- --ignored diagnostic_scan_screenshots`.
    #[test]
    #[ignore]
    fn diagnostic_scan_screenshots() {
        let mut report = String::new();
        for name in [
            "screenshot.png",
            "screenshot2.png",
            "screenshot_15000.png",
            "example.png",
        ] {
            let Ok(img) = image::open(name) else {
                report.push_str(&format!("{name}: could not open\n"));
                continue;
            };
            let rgb = img.to_rgb8();
            let (w, h) = (rgb.width(), rgb.height());
            let mut red_pixels = 0usize;
            let mut green_pixels = 0usize;
            let mut best_red = (0u8, 0u32, 0u32);
            let mut best_green = (0u8, 0u32, 0u32);
            // The brightest pixels overall, to see what dominates.
            let mut brightest: Vec<(u32, u32, u32)> = Vec::new(); // (lum, x, y)
            for y in 0..h {
                for x in 0..w {
                    let p = rgb.get_pixel(x, y);
                    let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
                    // Saturated red: R dominant, G/B low.
                    if r > 150 && g * 3 < r && b * 3 < r {
                        red_pixels += 1;
                        if r as u8 > best_red.0 {
                            best_red = (r as u8, x, y);
                        }
                    }
                    // Saturated green: G dominant, R/B low.
                    if g > 150 && r * 3 < g && b * 3 < g {
                        green_pixels += 1;
                        if g as u8 > best_green.0 {
                            best_green = (g as u8, x, y);
                        }
                    }
                    let lum = (r * 2 + g * 3 + b) / 6;
                    if brightest.len() < 8 || lum > brightest[0].0 {
                        brightest.push((lum, x, y));
                        brightest.sort();
                        if brightest.len() > 8 {
                            brightest.remove(0);
                        }
                    }
                }
            }
            report.push_str(&format!(
                "{name} ({w}x{h}): red-ish {red_pixels} (best {best_red:?}), \
                 green-ish {green_pixels} (best {best_green:?})\n"
            ));
            // Sample the brightest pixels' actual colors.
            for (lum, x, y) in &brightest {
                let p = rgb.get_pixel(*x, *y);
                report.push_str(&format!(
                    "  bright ({x},{y}) lum {lum}: ({}, {}, {})\n",
                    p[0], p[1], p[2]
                ));
            }
        }
        std::fs::write("target/screenshot_scan.txt", &report).ok();
    }

    /// DIAGNOSTIC (temporary): render the current demo scene on the GPU at
    /// the default camera and scan for the laser spots. The CPU renders
    /// them strongly (see demo_beam_diag); if the GPU doesn't, the bug is
    /// in the WGSL port.
    /// Ignored by default: it creates a GPU instance, and several GPU
    /// tests running in parallel crashed the suite with SIGSEGV during
    /// the debug session. Run with
    /// `cargo test -- --ignored diagnostic_gpu_demo_spots`.
    #[test]
    #[ignore]
    fn diagnostic_gpu_demo_spots() {
        let scene = build_scene();
        const W: usize = 480;
        const H: usize = 270;
        let Ok(mut gpu) = crate::gpu::GpuRenderer::new(&scene, W as u32, H as u32) else {
            std::fs::write("target/gpu_demo_diag.txt", "no GPU adapter\n").ok();
            return;
        };
        let cam = default_cam();
        // Accumulate 64 frames x 4 spp = 256 spp.
        for _ in 0..64 {
            gpu.render_frame(&cam, 8);
        }
        let buf = gpu.read_pixels();
        let mut report = String::new();
        report.push_str(&format!(
            "GPU demo render {W}x{H}, {} spp: scanning for spots\n",
            gpu.samples_accumulated()
        ));
        // Project the true spot points through the camera (the same
        // convention as Cam::render).
        let (right, up) = {
            let forward = (cam.lookat - cam.origin).normalize();
            let world_up = Vec3::new(0.0, 1.0, 0.0);
            let right = world_up.cross(forward).normalize();
            let up = forward.cross(right).normalize();
            (right, up)
        };
        let aspect = W as f32 / H as f32;
        let theta = cam.fov_deg.to_radians();
        let hh = (theta * 0.5).tan();
        let ww = aspect * hh;
        let project = |p: Vec3| -> (usize, usize) {
            let d = p - cam.origin;
            let len = d.length();
            let dir = d * (1.0 / len);
            let u = (dir.dot(right) / ww + 1.0) * 0.5;
            let v = (1.0 - dir.dot(up) / hh) * 0.5;
            ((u * W as f32) as usize, (v * H as f32) as usize)
        };
        let red_spot = Vec3::new(1.8076, 0.62, 0.99);
        let green_spot = Vec3::new(-2.5, 0.633, 1.252);
        for (label, p) in [("red spot", red_spot), ("green spot", green_spot)] {
            let (cx, cy) = project(p);
            report.push_str(&format!(
                "--- {label} world {p:?} -> pixel ({cx},{cy}) ---\n"
            ));
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let x = (cx as i32 + dx).clamp(0, W as i32 - 1) as usize;
                    let y = (cy as i32 + dy).clamp(0, H as i32 - 1) as usize;
                    let g = buf.get_pixel(x, y);
                    report.push_str(&format!(
                        "  ({x:3},{y:3}) gpu ({:8.3},{:8.3},{:8.3})\n",
                        g.x, g.y, g.z
                    ));
                }
            }
        }
        // The overall image mean, to confirm the render isn't black.
        let mean = buf
            .pixels
            .iter()
            .fold(VEC_ZERO, |a, p| a + *p)
            * (1.0 / buf.pixels.len() as f32);
        report.push_str(&format!("image mean: {mean:?}\n"));
        std::fs::write("target/gpu_demo_diag.txt", &report).ok();
    }

    /// DIAGNOSTIC (temporary): CPU-vs-GPU comparison on the actual demo
    /// scene (the oracle test uses a different scene with a wide-θ,
    /// mirror-less laser, so it never exercises the bridge path). Render
    /// the same view on both backends and dump the most-divergent pixels.
    /// Ignored by default: it creates a GPU instance, and several GPU
    /// tests running in parallel crashed the suite with SIGSEGV during
    /// the debug session. Run with
    /// `cargo test -- --ignored diagnostic_cpu_vs_gpu_demo`.
    #[test]
    #[ignore]
    fn diagnostic_cpu_vs_gpu_demo() {
        let scene = build_scene();
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let bridges = scene.bridges();
        let cam = default_cam();

        const W: usize = 120;
        const H: usize = 68;
        const SPP: usize = 48;

        // CPU render.
        let (forward, right, up) = {
            let forward = (cam.lookat - cam.origin).normalize();
            let world_up = Vec3::new(0.0, 1.0, 0.0);
            let right = world_up.cross(forward).normalize();
            let up = forward.cross(right).normalize();
            (forward, right, up)
        };
        let aspect = W as f32 / H as f32;
        let theta = cam.fov_deg.to_radians();
        let hh = (theta * 0.5).tan();
        let ww = aspect * hh;
        let mut cpu_pixels = vec![VEC_ZERO; W * H];
        {
            use rayon::prelude::*;
            cpu_pixels
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, px)| {
                    let x = i % W;
                    let y = i / W;
                    let mut rng = rand::thread_rng();
                    let mut mean = VEC_ZERO;
                    for _ in 0..SPP {
                        let u = (x as f32 + rng.gen::<f32>()) / W as f32;
                        let v = (y as f32 + rng.gen::<f32>()) / H as f32;
                        let px_ = (u * 2.0 - 1.0) * ww;
                        let py = -(v * 2.0 - 1.0) * hh;
                        let dir = (right * px_ + up * py + forward).normalize();
                        let ray = crate::geometry::Ray { origin: cam.origin, direction: dir };
                        mean = mean
                            + crate::bdpt::bdpt_trace(
                                &ray, &*cpu_scene, 8, &lights, &bridges, &mut rng,
                            );
                    }
                    *px = mean * (1.0 / SPP as f32);
                });
        }

        // GPU render: same resolution, same spp.
        let Ok(mut gpu) = crate::gpu::GpuRenderer::new(&scene, W as u32, H as u32) else {
            std::fs::write("target/cpu_gpu_demo_diag.txt", "no GPU adapter\n").ok();
            return;
        };
        gpu.render_frame_spp(&cam, 8, SPP as u32);
        let gpu_buf = gpu.read_pixels();

        // Compare: the top divergences and the spot pixels.
        let mut report = String::new();
        report.push_str(&format!(
            "CPU vs GPU demo render {W}x{H} @ {SPP} spp\n"
        ));
        let mut divs: Vec<(f32, usize)> = cpu_pixels
            .iter()
            .zip(&gpu_buf.pixels)
            .map(|(c, g)| ((c.x - g.x).abs() + (c.y - g.y).abs() + (c.z - g.z).abs(), 0))
            .zip(0..)
            .map(|((d, _), i)| (d, i))
            .collect();
        divs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (d, i) in divs.iter().take(12) {
            let x = i % W;
            let y = i / W;
            let c = cpu_pixels[*i];
            let g = gpu_buf.pixels[*i];
            // What surface does this pixel look at (CPU)?
            let u = (x as f32 + 0.5) / W as f32;
            let v = (y as f32 + 0.5) / H as f32;
            let px_ = (u * 2.0 - 1.0) * ww;
            let py = -(v * 2.0 - 1.0) * hh;
            let dir = (right * px_ + up * py + forward).normalize();
            let hit = cpu_scene.intersect(&crate::geometry::Ray { origin: cam.origin, direction: dir });
            let surf = hit
                .map(|h| format!("{:?} mat {:?}", h.p, h.mat.bsdf))
                .unwrap_or_else(|| "escape".into());
            report.push_str(&format!(
                "({x:3},{y:3}) div {d:8.3}  cpu ({:7.3},{:7.3},{:7.3})  gpu ({:7.3},{:7.3},{:7.3})  -> {surf}\n",
                c.x, c.y, c.z, g.x, g.y, g.z
            ));
        }
        // Project the true spot points through the camera to pixel
        // coordinates (the same convention as Cam::render: u left->right,
        // v top->bottom, py negated), then probe neighborhoods.
        let project = |p: Vec3| -> (usize, usize) {
            let d = p - cam.origin;
            let len = d.length();
            let dir = d * (1.0 / len);
            let px = dir.dot(right);
            let py = dir.dot(up);
            let u = (px / ww + 1.0) * 0.5;
            let v = (1.0 - py / hh) * 0.5;
            ((u * W as f32) as usize, (v * H as f32) as usize)
        };
        let red_spot = Vec3::new(1.8076, 0.62, 0.99);
        let green_spot = Vec3::new(-2.5, 0.633, 1.252);
        for (label, p) in [("red spot", red_spot), ("green spot", green_spot)] {
            let (cx, cy) = project(p);
            report.push_str(&format!(
                "--- {label} at world {p:?} -> pixel ({cx},{cy}) ---\n"
            ));
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let x = (cx as i32 + dx).clamp(0, W as i32 - 1) as usize;
                    let y = (cy as i32 + dy).clamp(0, H as i32 - 1) as usize;
                    let c = cpu_pixels[y * W + x];
                    let g = gpu_buf.pixels[y * W + x];
                    report.push_str(&format!(
                        "  ({x:3},{y:3}) cpu ({:8.3},{:8.3},{:8.3})  gpu ({:8.3},{:8.3},{:8.3})\n",
                        c.x, c.y, c.z, g.x, g.y, g.z
                    ));
                }
            }
        }
        // What does the GPU's best-red pixel (238, 226) at 480x270
        // correspond to here (120x68)? Scale: (59.5, 56.5).
        for (label, x, y) in [("gpu best-red scaled", 59usize, 56usize)] {
            let c = cpu_pixels[y * W + x];
            let g = gpu_buf.pixels[y * W + x];
            let u = (x as f32 + 0.5) / W as f32;
            let v = (y as f32 + 0.5) / H as f32;
            let px_ = (u * 2.0 - 1.0) * ww;
            let py = -(v * 2.0 - 1.0) * hh;
            let dir = (right * px_ + up * py + forward).normalize();
            let hit = cpu_scene.intersect(&crate::geometry::Ray { origin: cam.origin, direction: dir });
            let surf = hit
                .map(|h| format!("{:?} mat {:?}", h.p, h.mat.bsdf))
                .unwrap_or_else(|| "escape".into());
            report.push_str(&format!(
                "{label} ({x},{y}): cpu {c:?}  gpu {g:?}  -> {surf}\n"
            ));
        }
        // Channel means over the whole image.
        let cm = cpu_pixels.iter().fold(VEC_ZERO, |a, p| a + *p) * (1.0 / (W * H) as f32);
        let gm = gpu_buf.pixels.iter().fold(VEC_ZERO, |a, p| a + *p) * (1.0 / (W * H) as f32);
        report.push_str(&format!("image mean: cpu {cm:?}  gpu {gm:?}\n"));
        std::fs::write("target/cpu_gpu_demo_diag.txt", &report).ok();
    }

    /// DIAGNOSTIC (temporary): the decisive probe. The CPU renders the red
    /// spot at R≈24 when a ray is aimed straight at it (see demo_beam_diag).
    /// Render the same probe on the GPU in ONE dispatch with 4096 spp —
    /// a pure batch mean, no accumulation blending, no winsorization
    /// (the first frame's batch is stored raw). If the GPU ever credits
    /// the spot, the mean must approach ~24; if it never does, the bug
    /// is in the WGSL trace, not the accumulation layer.
    /// Ignored by default: it creates a GPU instance, and several GPU
    /// tests running in parallel crashed the suite with SIGSEGV during
    /// the debug session. Run with
    /// `cargo test -- --ignored diagnostic_gpu_spot_probe`.
    #[test]
    #[ignore]
    fn diagnostic_gpu_spot_probe() {
        let scene = build_scene();
        let mut report = String::new();
        // The red spot on the partition wall, its normal (toward the
        // camera side), and the probe eye 0.5 m out.
        let spot = Vec3::new(1.8076, 0.62, 0.99);
        let normal = Vec3::new(0.0, 0.0, -1.0);
        let eye = spot + normal * 0.5;
        let cam = Cam::new(eye, spot, 60.0);

        // CPU reference: the center ray.
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let bridges = scene.bridges();
        let dir = (spot - eye).normalize();
        let ray = crate::geometry::Ray { origin: eye, direction: dir };
        let mut rng = rand::thread_rng();
        let n = 4000;
        let mut mean = VEC_ZERO;
        let mut hits = 0;
        for _ in 0..n {
            let c = crate::bdpt::bdpt_trace(&ray, &*cpu_scene, 8, &lights, &bridges, &mut rng);
            mean = mean + c;
            if c.x > 10.0 {
                hits += 1;
            }
        }
        mean = mean * (1.0 / n as f32);
        report.push_str(&format!(
            "CPU probe at the red spot: mean {mean:?} ({hits}/{n} samples with R>10)\n"
        ));

        // GPU: per-frame batch values at the center pixel, 64 frames of
        // 4 spp (the preview path). The seed varies per frame (it mixes
        // uniforms.samples), so this is the true per-frame distribution:
        // ~23% of frames should carry a bridge hit (batch R ≈ 60), the
        // rest ambient (~0.5). Also track the running accumulated mean.
        let Ok(mut gpu) = crate::gpu::GpuRenderer::new(&scene, 32, 32) else {
            report.push_str("no GPU adapter\n");
            std::fs::write("target/gpu_spot_probe.txt", &report).ok();
            return;
        };
        const FRAMES: usize = 64;
        let mut batches: Vec<f32> = Vec::new();
        let mut running: Vec<f32> = Vec::new();
        for f in 0..FRAMES {
            gpu.render_frame(&cam, 8);
            let buf = gpu.read_pixels();
            batches.push(buf.get_pixel(16, 16).x);
            running.push(buf.get_pixel(16, 16).x);
            if f % 16 == 15 {
                report.push_str(&format!(
                    "  frames {f}: last-16 batches {:?}\n",
                    &batches[batches.len() - 16..]
                ));
            }
        }
        let big = batches.iter().filter(|&&s| s > 5.0).count();
        let mean: f32 = batches.iter().sum::<f32>() / FRAMES as f32;
        report.push_str(&format!(
            "GPU per-frame batches at the spot: mean {mean:.3}, frames with R>5: {big}/{FRAMES}\n"
        ));
        report.push_str(&format!("  all batches: {batches:?}\n"));
        report.push_str(&format!("  running mean (every 8th): {:?}\n", {
            let mut v = Vec::new();
            for i in (0..FRAMES).step_by(8) {
                v.push(running[i]);
            }
            v
        }));

        // The same probe for the green spot on the left wall.
        let gspot = Vec3::new(-2.5, 0.633, 1.252);
        let gnormal = Vec3::new(1.0, 0.0, 0.0);
        let geye = gspot + gnormal * 0.5;
        let gdir = (gspot - geye).normalize();
        let gray = crate::geometry::Ray { origin: geye, direction: gdir };
        let mut rng = rand::thread_rng();
        let mut gmean = VEC_ZERO;
        let mut ghits = 0;
        for _ in 0..n {
            let c = crate::bdpt::bdpt_trace(&gray, &*cpu_scene, 8, &lights, &bridges, &mut rng);
            gmean = gmean + c;
            if c.y > 10.0 {
                ghits += 1;
            }
        }
        gmean = gmean * (1.0 / n as f32);
        report.push_str(&format!(
            "CPU probe at the green spot: mean {gmean:?} ({ghits}/{n} samples with G>10)\n"
        ));
        let gcam = Cam::new(geye, gspot, 60.0);
        let Ok(mut ggpu) = crate::gpu::GpuRenderer::new(&scene, 32, 32) else {
            std::fs::write("target/gpu_spot_probe.txt", &report).ok();
            return;
        };
        ggpu.render_frame_spp(&gcam, 8, 64);
        let gbuf = ggpu.read_pixels();
        report.push_str(&format!(
            "GPU green probe (center pixel, one 64-spp dispatch): {:?}\n",
            gbuf.get_pixel(16, 16)
        ));
        std::fs::write("target/gpu_spot_probe.txt", &report).ok();
    }

    /// DIAGNOSTIC (temporary): minimal repro. A small room with ONE laser
    /// using the demo's exact model (spectral 650 nm line, power-
    /// parametrized 50 mW, physical divergence θ = 6.2e-4) aimed at a
    /// 45° mirror that bounces the beam onto the ceiling. Probe the spot
    /// on CPU and GPU. This exercises: the line branch rule, the power
    /// parametrization, the versine cone, the gate-mapped bridge cone,
    /// the mirror chain, and the direct-hit crediting — with none of the
    /// demo scene's other confounds.
    /// Ignored by default: it creates a GPU instance, and several GPU
    /// tests running in parallel crashed the suite with SIGSEGV during
    /// the debug session. Run with
    /// `cargo test -- --ignored diagnostic_gpu_minimal_bridge_repro`.
    #[test]
    #[ignore]
    fn diagnostic_gpu_minimal_bridge_repro() {
        let mut scene = SceneData::new();
        let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
        // A 2 x 2 x 2 room.
        scene.add_wall(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(0.0, 1.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, -1.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 2.0, 2.0, white);
        // The laser: the demo's exact model. 70 cm up, aimed straight
        // down at the mirror.
        let lambda = 650.0f32;
        let power_w = 0.05f32;
        let m2 = 30.0f32;
        let theta = m2 * lambda * 1e-9 / (std::f32::consts::PI * 0.01);
        scene.add_laser_beam(
            Vec3::new(0.4, 1.7, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.2,
            0.01,
            theta,
            lambda,
            power_w,
        );
        // One 45° floor mirror: the downward beam reflects horizontal
        // (+z)... then hits the front wall. Simpler: aim the mirror to
        // send the beam to the CEILING — 45° tilted so the downward
        // beam reflects horizontally, then a second wall mirror sends
        // it up? Keep it minimal: ONE mirror, the beam lands on a wall.
        // A 45° mirror at the floor reflects the down-beam to +z; the
        // beam lands on the front wall (z=1) at height ~0.002.
        scene.add_quad(
            Vec3::new(0.4, 0.002, 0.0),
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
            0.6,
            0.6,
            MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
        );

        // Where does the beam land? Trace it.
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("the laser");
        let trace = scene.trace_laser_beam(laser, 4);
        let mut report = String::new();
        report.push_str(&format!(
            "minimal repro: {} legs, spot {:?}\n",
            trace.legs.len(),
            trace.spot.as_ref().map(|s| s.p)
        ));
        let ladder = scene.bridges_of_light(laser, 4);
        report.push_str(&format!("ladder: {} rungs\n", ladder.len()));
        for (i, r) in ladder.iter().enumerate() {
            report.push_str(&format!(
            "  rung {i}: image {:?} r={} beam_dir {:?} versine {:.3e}\n",
            r.image_center, r.radius, r.beam_dir, r.versine
        ));
        }
        let spot = trace.spot.expect("the beam must land").p;

        // CPU probe: look at the spot from 0.5 m out along the wall
        // normal (the front wall's normal is -z).
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let bridges = scene.bridges();
        let normal = Vec3::new(0.0, 0.0, -1.0);
        let eye = spot + normal * 0.5;
        let dir = (spot - eye).normalize();
        let ray = crate::geometry::Ray { origin: eye, direction: dir };
        let mut rng = rand::thread_rng();
        let n = 4000;
        let mut mean = VEC_ZERO;
        let mut hits = 0;
        for _ in 0..n {
            let c = crate::bdpt::bdpt_trace(&ray, &*cpu_scene, 8, &lights, &bridges, &mut rng);
            mean = mean + c;
            if c.x > 10.0 {
                hits += 1;
            }
        }
        mean = mean * (1.0 / n as f32);
        report.push_str(&format!(
            "CPU probe at the spot: mean {mean:?} ({hits}/{n} samples with R>10)\n"
        ));

        // GPU probe: the same camera geometry.
        let Ok(mut gpu) = crate::gpu::GpuRenderer::new(&scene, 32, 32) else {
            report.push_str("no GPU adapter\n");
            std::fs::write("target/gpu_minimal_repro.txt", &report).ok();
            return;
        };
        let cam = Cam::new(eye, spot, 60.0);
        for _ in 0..64 {
            gpu.render_frame(&cam, 8);
        }
        let buf = gpu.read_pixels();
        report.push_str(&format!(
            "GPU probe (center pixel, {} frames x 4 spp): {:?}\n",
            64,
            buf.get_pixel(16, 16)
        ));
        std::fs::write("target/gpu_minimal_repro.txt", &report).ok();
    }

    /// DIAGNOSTIC (temporary): bisect the WGSL bridge failure. Three
    /// variants of the minimal repro, each isolating one piece:
    ///   A. no mirror — the direct hit on the floor (cone gate, line
    ///      branch, power parametrization; no bridge).
    ///   B. mirror + WIDE θ (0.05 rad) — the bridge path with the disc
    ///      cone binding (the old behavior).
    ///   C. mirror + PHYSICAL θ (6.2e-4) — the bridge path with the
    ///      gate-mapped cone binding (the demo's regime).
    /// Each variant is probed on CPU and GPU.
    /// Ignored by default: it creates GPU instances, and several GPU
    /// tests running in parallel crashed the suite with SIGSEGV during
    /// the debug session. Run with
    /// `cargo test -- --ignored diagnostic_gpu_bridge_bisect`.
    #[test]
    #[ignore]
    fn diagnostic_gpu_bridge_bisect() {
        let mut report = String::new();
        let build = |with_mirror: bool, theta: f32| -> SceneData {
            let mut scene = SceneData::new();
            let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
            scene.add_wall(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(0.0, 1.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, -1.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 2.0, 2.0, white);
            scene.add_wall(Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 2.0, 2.0, white);
            scene.add_laser_beam(
                Vec3::new(0.4, 1.7, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                0.2,
                0.01,
                theta,
                650.0,
                0.05,
            );
            if with_mirror {
                scene.add_quad(
                    Vec3::new(0.4, 0.002, 0.0),
                    Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
                    0.6,
                    0.6,
                    MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
                );
            }
            scene
        };

        for (label, with_mirror, theta) in [
            ("A no-mirror", false, 6.2e-4),
            ("B mirror wide-theta", true, 0.05),
            ("C mirror physical-theta", true, 6.2e-4),
        ] {
            let scene = build(with_mirror, theta);
            let laser = scene.lights.iter().find(|l| l.shape == 2).unwrap();
            let trace = scene.trace_laser_beam(laser, 4);
            let spot = trace.spot.expect("the beam must land").p;
            // The probe eye: 0.5 m out along the surface normal. For the
            // no-mirror variant the spot is on the floor (normal +y);
            // for the mirror variants it is on the front wall (normal
            // -z).
            let normal = if with_mirror {
                Vec3::new(0.0, 0.0, -1.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            let eye = spot + normal * 0.5;

            // CPU probe.
            let (cpu_scene, lights) = scene.build_cpu_scene();
            let bridges = scene.bridges();
            let dir = (spot - eye).normalize();
            let ray = crate::geometry::Ray { origin: eye, direction: dir };
            let mut rng = rand::thread_rng();
            let n = 4000;
            let mut mean = VEC_ZERO;
            let mut hits = 0;
            for _ in 0..n {
                let c = crate::bdpt::bdpt_trace(&ray, &*cpu_scene, 8, &lights, &bridges, &mut rng);
                mean = mean + c;
                if c.x > 10.0 {
                    hits += 1;
                }
            }
            mean = mean * (1.0 / n as f32);

            // GPU probe.
            let gpu_mean = match crate::gpu::GpuRenderer::new(&scene, 32, 32) {
                Ok(mut gpu) => {
                    let cam = Cam::new(eye, spot, 60.0);
                    for _ in 0..64 {
                        gpu.render_frame(&cam, 8);
                    }
                    let buf = gpu.read_pixels();
                    buf.get_pixel(16, 16)
                }
                Err(_) => {
                    report.push_str("no GPU adapter\n");
                    VEC_ZERO
                }
            };
            report.push_str(&format!(
                "{label}: spot {spot:?}  cpu {mean:?} ({hits}/{n} R>10)  gpu {gpu_mean:?}\n"
            ));
        }
        std::fs::write("target/gpu_bridge_bisect.txt", &report).ok();
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

    /// The middle-leg glow — the whole point of the bridge ladder. A
    /// diffuse point under the beam's *middle* leg (after one mirror,
    /// before the next) receives light that crossed exactly one mirror:
    /// unreachable by any BDPT connection (the mirror is a delta
    /// endpoint) and by any light subpath (the light walk's reflection
    /// off the first mirror continues along the beam, and the floor
    /// spot it eventually lands on cannot connect back to a floor point
    /// — the floor-to-floor geometry term is exactly 0), so without the
    /// ladder's rung 0 the point renders black. With the ladder, the
    /// eye walk aims at rung 0's image disc and the glow appears.
    ///
    /// The probe uses a controlled synthetic scene (not the demo) so
    /// the signal is clean: the beam hugs a white floor (the cone
    /// footprint paints it), the walls and ceiling are black (no
    /// ambient bounce to pollute the comparison), and the probe sits
    /// under the middle leg inside the footprint. The whole-chain rung
    /// cannot carry this light: its path from the probe arrives at the
    /// aperture far off the beam axis, outside the emission cone.
    #[test]
    fn middle_leg_glow_is_carried_by_the_ladder() {
        // A closed 2 x 2 x 2 room: a white floor, black walls (the
        // only light is the laser, and the only diffuse light at the
        // probe must be the glow itself).
        let mut scene = SceneData::new();
        let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
        let black = MaterialData::diffuse(Vec3::new(0.02, 0.02, 0.02));
        scene.add_wall(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 2.0, 2.0, black);
        scene.add_wall(Vec3::new(0.0, 1.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 2.0, 2.0, black);
        scene.add_wall(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, -1.0), 2.0, 2.0, black);
        scene.add_wall(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 2.0, 2.0, black);
        scene.add_wall(Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 2.0, 2.0, black);
        // The laser: 70 cm up, aimed straight down at the first
        // mirror. The cone (0.03 rad) paints the floor the beam hugs
        // (the mirrors sit 1 cm above it, so the beam's legs run 1 cm
        // above the floor and the footprint starts ~0.33 m past each
        // mirror).
        scene.add_laser(
            Vec3::new(0.0, 0.7, -0.5),
            Vec3::new(0.0, -1.0, 0.0),
            0.2,
            0.02,
            0.03,
            Vec3::new(1.0, 50.0, 1.0),
        );
        // Mirror 1: 45°, 1 cm above the floor — reflects the downward
        // beam horizontal (+z).
        scene.add_quad(
            Vec3::new(0.0, 0.01, -0.5),
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
            0.4,
            0.4,
            MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
        );
        // Mirror 2: 45° the other way — reflects the horizontal beam
        // straight down, to the spot on the floor 1 cm below.
        scene.add_quad(
            Vec3::new(0.0, 0.01, 0.5),
            Vec3::new(0.0, -std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),
            0.4,
            0.4,
            MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
        );

        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("the laser");
        let trace = scene.trace_laser_beam(laser, 4);
        assert_eq!(
            trace.legs.len(),
            3,
            "two bounces: laser -> floor mirror -> wall mirror -> floor spot, got {:?}",
            trace.legs
        );
        // The ladder: one rung per mirror crossed.
        let ladder = scene.bridges_of_light(laser, 4);
        assert_eq!(ladder.len(), 2, "one rung per mirror");
        // Rung 0 is the aperture's single reflection across the first
        // mirror's plane (the unfolding invariant).
        let aperture = laser.center;
        let first_hit = trace.legs[0].1;
        let first_n = Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2);
        let d = aperture - first_hit;
        let expected_image = aperture - first_n * (2.0 * d.dot(first_n));
        assert!(
            (ladder[0].image_center - expected_image).length() < 1e-3,
            "rung 0 {:?} must be the aperture's single reflection {:?}",
            ladder[0].image_center,
            expected_image
        );

        // The probe: the floor point under the middle leg, 0.8 m past
        // the first mirror — inside the cone footprint (which starts
        // 0.33 m past it), so the glow is physically present. The
        // whole-chain rung's path from here arrives at the aperture
        // ~36° off the beam axis — far outside the 1.7° emission cone
        // — so only rung 0 can carry this light.
        let probe = Vec3::new(0.0, 0.0, 0.3);
        let to_image = ladder[0].image_center - probe;
        assert!(
            to_image.dot(Vec3::new(0.0, 1.0, 0.0)) > 0.0,
            "rung 0 must point up out of the floor under the middle leg"
        );
        assert!(
            to_image.length_squared() > ladder[0].radius * ladder[0].radius,
            "rung 0's cone must be non-degenerate at the probe"
        );

        // The camera: inside the room, looking at the probe; the ray
        // must hit the probe first (the mirrors are off to the sides).
        let (cpu_scene, lights) = scene.build_cpu_scene();
        let cam = Cam::new(Vec3::new(0.9, 0.3, 0.3), probe, 60.0);
        let seg = probe - cam.origin;
        let first = cpu_scene
            .intersect(&crate::geometry::Ray {
                origin: cam.origin,
                direction: seg * (1.0 / seg.length()),
            })
            .expect("the camera must see the floor");
        assert!(
            (first.p - probe).length() < 1e-4,
            "the camera ray must hit the probe first, got {:?}",
            first.p
        );

        // The estimator at the probe, with and without the ladder: the
        // difference is exactly the ladder's contribution. Without any
        // bridge the glow is structurally unreachable — the mirror is a
        // delta endpoint (no connection through it), the light walk
        // reflects along the beam (never toward the probe), and the
        // floor spot it lands on cannot connect to another floor point
        // (the floor-to-floor geometry term is 0) — so only the black
        // walls' faint bounce remains.
        let ladder = scene.bridges();
        let no_bridges: Vec<crate::bdpt::Bridge> = Vec::new();
        let mut rng = rand::thread_rng();
        let n = 2000;
        let mut mean_at = |bridges: &[crate::bdpt::Bridge]| -> f64 {
            let dir = (probe - cam.origin).normalize();
            let ray = crate::geometry::Ray {
                origin: cam.origin,
                direction: dir,
            };
            let mut sum = 0.0f64;
            for _ in 0..n {
                let c = crate::bdpt::bdpt_trace(&ray, &*cpu_scene, 8, &lights, bridges, &mut rng);
                assert!(c.y.is_finite(), "non-finite sample {c:?}");
                sum += c.y as f64;
            }
            sum / n as f64
        };
        let with_ladder = mean_at(&ladder);
        let without = mean_at(&no_bridges);

        // The ladder must light the probe up clearly above the
        // bridge-less value — the middle-leg glow. Analytically the
        // glow is (albedo/π)·Le·cos_grazing·Ω_image ≈ 5e-5 (the image
        // disc of the 2 cm aperture at 1.5 m, arriving 0.4° above the
        // horizon), vs the black-wall bounce ≈ 1e-6: a ~30x ratio. 3x
        // is a conservative floor; the absolute floor guards against
        // both values collapsing to ~0.
        assert!(
            with_ladder > without * 3.0,
            "the middle-leg glow {with_ladder} must exceed the bridge-less value {without} by 3x"
        );
        assert!(
            with_ladder > 1e-6,
            "the middle-leg glow {with_ladder} must be absolutely nonzero"
        );
        std::fs::write(
            "target/middle_leg_probe.txt",
            format!(
                "middle-leg glow with ladder {with_ladder:.3e} vs without {without:.3e} (ratio {})\n",
                with_ladder / without.max(1e-9)
            ),
        )
        .ok();
    }
}
