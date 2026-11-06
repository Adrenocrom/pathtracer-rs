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
    scene.add_light(LightData::from_quad(
            Vec3::new(0.0, room_height - 0.01, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.2, 0.2, emission,
    ));

    //// The laser showcase: a green laser hanging from the ceiling, aimed
    //// straight down at a mirror tile on the floor, which bounces the
    //// beam up onto a flat mirror on the front wall, which bounces it
    //// up onto the ceiling. Two bounces, one spot.
    ////
    //// The laser is added *before* the mirrors on purpose: the specular
    //// bridge (the aperture's mirror image through the beam's whole
    //// mirror chain) is derived on demand from the fully-built scene
    //// (`SceneData::bridges_of_light`), never baked at add-time, so the
    //// registration order is irrelevant — and a scene that moves
    //// between frames simply re-derives the bridge from the new
    //// geometry.
    ////
    //// The beam's spot on the wall is *transported light*, not a decal:
    //// the bridge is a sampling technique the BDPT eye walk uses at
    //// diffuse vertices (with probability BRIDGE_PROB it aims at the
    //// image disc, so the path camera → wall → wall mirror → floor
    //// mirror → aperture is carried by ordinary subpath walking; the
    //// direct-hit strategy then credits the aperture's emission,
    //// cone-gated). No BDPT connection can sample that path (the mirrors
    //// are delta endpoints) and no light subpath can reach the aperture
    //// (a ~1 cm disc behind the first mirror), so without the bridge
    //// the spot is invisible — the estimator's blind spot. There is
    //// deliberately no beam in flight: in clean air a laser beam is
    //// invisible from the side; only the spot is real.
    //scene.add_laser(
    //    Vec3::new(-0.9, 1.899, 0.6), // housing base cap 1 mm below the ceiling
    //    Vec3::new(0.0, -1.0, 0.0),  // beam straight down at the floor mirror
    //    0.2,                        // housing height
    //    0.01,                       // aperture radius
    //    0.01,                       // beam half-angle (radians, ~0.57°)
    //    Vec3::new(0.8, 4.0e8, 0.8), // green laser
    //);

    //// Mirror 1: the floor tile under the laser, 0.6 × 0.6, tilted 30°
    //// (its lower edge 2 mm above the floor). The normal
    //// (0, cos 30°, sin 30°) points up and toward the front wall, so the
    //// downward beam reflects up the wall at 60° from vertical.
    //let mirror_tilt = 30.0f32.to_radians();
    //scene.add_quad(
    //    Vec3::new(-0.9, 0.3 * mirror_tilt.sin() + 0.002, 0.6),
    //    Vec3::new(0.0, mirror_tilt.cos(), mirror_tilt.sin()),
    //    0.6,
    //    0.6,
    //    MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
    //);

    //// Mirror 2: a flat mirror on the front wall, right where the beam
    //// from mirror 1 arrives — the center sits on the beam (the beam
    //// leaves mirror 1 at (-0.9, 0.152, 0.6) traveling (0, 0.5, 0.866),
    //// crossing z = 2.498 at y = 1.2478), so the beam hits it dead
    //// center. The flat mirror flips the beam's z-travel, sending it
    //// back up across the room to the ceiling — the second bounce. The
    //// spot lands on the ceiling at (-0.9, 2.0, 1.195), in frame from
    //// the default camera.
    ////
    //// `wall_mirror_tilt` is the movement knob: the demo animates it,
    //// and the beam's second bounce (and the spot, and the derived
    //// bridge) follow. The tilt is around the x axis (the quad's
    //// tangent for a ±z normal), so the beam stays in the y-z plane.
    //let wall_normal = Vec3::new(0.0, wall_mirror_tilt.sin(), -wall_mirror_tilt.cos());
    //scene.add_quad(
    //    Vec3::new(-0.9, 1.2478, room_size - 0.002),
    //    wall_normal,
    //    0.6,
    //    0.6,
    //    MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
    //);

    scene.add_box( Vec3::new(room_size/2.0, 1.0, 1.0), Vec3::new(room_size, room_height, 0.02), white);

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
    // of putting 4×10⁸ in the red channel. The line's power is chosen
    // so the rendered spot's luminance matches the old RGB hack (the
    // 1/p(λ) sensor factor ≈ ×3 replaces the channel stuffing); the
    // out-of-gamut negative channels clamp at display, where they
    // belong.
    let laser_pos = Vec3::new(-1.3, 0.62, 1.8);
    let back_mirror_center = Vec3::new(0.0, 1.0, -(room_size - 0.01));
    //let aim_point = Vec3::new(0.0, laser_aim_y, -(room_size - 0.01));
    let aim = Vec3::new(1.0, 0.0, -room_size).normalize();
    scene.add_laser_line(
        laser_pos,
        aim,
        0.2,                        // housing height
        0.01,                       // aperture radius
        0.01,                       // beam half-angle (radians, ~0.57°)
        650.0,                      // nm — the red laser line
        1.2e9,                      // spectral radiance at the line
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
    scene.add_box(pos, dim, white);
    scene.add_box(Vec3::new(pos.x -  dim.x / 2.0 + 0.1, pos.y / 2.0, pos.z - dim.z / 2.0 + 0.1), Vec3::new(0.01, pos.y, 0.01), brown);
    scene.add_box(Vec3::new(pos.x +  dim.x / 2.0 - 0.1, pos.y / 2.0, pos.z - dim.z / 2.0 + 0.1), Vec3::new(0.01, pos.y, 0.01), brown);
    scene.add_box(Vec3::new(pos.x -  dim.x / 2.0 + 0.1, pos.y / 2.0, pos.z + dim.z / 2.0 - 0.1), Vec3::new(0.01, pos.y, 0.01), brown);
    scene.add_box(Vec3::new(pos.x +  dim.x / 2.0 - 0.1, pos.y / 2.0, pos.z + dim.z / 2.0 - 0.1), Vec3::new(0.01, pos.y, 0.01), brown);

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
    scene.add_laser_line(
        Vec3::new(-1.6, 0.62, 1.9),
        Vec3::new(-1.0, 0.0, -0.5),
        0.2,
        0.01,
        0.01,
        532.0,                      // nm — the green laser line
        1.2e9,
    );

    scene.add_table(Vec3::new(0.0, 0.7, 0.0), Vec3::new(1.0, 0.05, 0.7), brown); // top slab + 4 legs to the floor
    scene.add_chair(Vec3::new(1.0, 0.45, 0.0), brown);                            // seat + 4 legs + backrest, faces +z
    scene.add_shelf(Vec3::new(-1.0, 0.8, -1.0), Vec3::new(1.0, 1.6, 0.3), brown);   // 2 side panels + 4 boards

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
        // The demo scene has three emitters: the ceiling panel and the
        // two laser apertures. (No decal: the beams' spots are
        // transported light, not fake emitters.)
        assert_eq!(scene.lights.len(), 3);
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
        assert_eq!(lights.len(), 3);
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
