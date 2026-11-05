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

fn build_scene_with(laser_aim_y: f32) -> SceneData {
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
    let laser_pos = Vec3::new(-1.3, 0.62, 1.8);
    let back_mirror_center = Vec3::new(0.0, 1.0, -(room_size - 0.01));
    //let aim_point = Vec3::new(0.0, laser_aim_y, -(room_size - 0.01));
    let aim = Vec3::new(1.0, 0.0, -room_size).normalize();
    scene.add_laser(
        laser_pos,
        aim,
        0.2,                        // housing height
        0.01,                       // aperture radius
        0.01,                       // beam half-angle (radians, ~0.57°)
        Vec3::new(4.0e8, 800.0, 800.0), // red laser
    );
    scene.add_quad(
        back_mirror_center,
        Vec3::new(0.0, 0.0, 1.0),
        2.0,
        1.0,
        MaterialData::mirror(Vec3::new(0.9, 0.9, 0.9)),
    );

    scene.add_box( Vec3::new(-1.4, 0.65, 1.5), Vec3::new(0.1, 0.1, 0.1), MaterialData::glass(1.5));
    scene.add_box( Vec3::new(-1.9, 0.65, 1.7), Vec3::new(0.001, 0.1, 0.2), MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)));
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
    scene.add_laser(
        Vec3::new(-1.6, 0.62, 1.9),
        Vec3::new(-1.0, 0.0, -0.5),
        0.2,
        0.01,
        0.01,
        Vec3::new(800.0, 4.0e8, 800.0),
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
    use crate::geometry::Bsdf;

    #[test]
    fn scene_builds_and_is_sampleable() {
        let scene = build_scene();
        // The demo scene has three emitters: the ceiling panel and the
        // two laser apertures. (No decal: the beams' spots are
        // transported light, not fake emitters.)
        assert_eq!(scene.lights.len(), 3);
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
        assert_eq!(lights.len(), 3);
    }

    /// The laser light sources: each is a diffuse cylinder housing plus
    /// a thin cylinder puck of laser material whose top cap is the
    /// aperture. The scene must register both as real geometry on both
    /// backends, the housing must carry a normal (non-emissive)
    /// material, and the aperture disc must be the puck's top cap.
    #[test]
    fn laser_is_a_cylinder_with_an_emitting_top_cap() {
        let scene = build_scene();
        // Two lasers: two housings and two pucks.
        assert_eq!(scene.cylinders.len(), 4);
        let housings: Vec<_> = scene
            .cylinders
            .iter()
            .filter(|c| !c.mat.is_laser())
            .collect();
        let pucks: Vec<_> = scene.cylinders.iter().filter(|c| c.mat.is_laser()).collect();
        assert_eq!(housings.len(), 2);
        assert_eq!(pucks.len(), 2);
        for puck in &pucks {
            assert!(puck.mat.is_emissive());
        }
        for housing in &housings {
            assert!(!housing.mat.is_emissive());
        }
        // Each laser light's aperture is its puck's top cap center.
        for laser_light in scene.lights.iter().filter(|l| l.shape == 2) {
            let puck = pucks
                .iter()
                .find(|c| (c.base + c.axis * (2.0 * c.half_h) - laser_light.center).length()
                    < 1e-5)
                .expect("each laser light must have its puck");
            assert!((laser_light.normal - puck.axis).length() < 1e-5);
            assert!((laser_light.radius - puck.radius).abs() < 1e-5);
            // The housing is fatter than the puck and reaches further
            // along the axis, so the puck's rim never peeks out from
            // behind it.
            let housing = housings
                .iter()
                .find(|h| (h.axis - puck.axis).length() < 1e-5
                    && (h.base - puck.base).length() < 0.25)
                .expect("each puck must have its housing");
            assert!(housing.radius > puck.radius);
            assert!(
                housing.base.dot(housing.axis) < puck.base.dot(puck.axis),
                "the housing must start behind the puck"
            );
        }
        // The CPU scene must contain each puck as intersectable geometry:
        // a ray along the axis from just beyond the top cap hits it.
        // (Just beyond — a long probe can cross other scene geometry
        // before reaching the puck, e.g. a mirror box in front of the
        // aperture.)
        let (cpu_scene, _) = scene.build_cpu_scene();
        for puck in &pucks {
            let top = puck.base + puck.axis * (2.0 * puck.half_h);
            let along_axis = crate::geometry::Ray {
                origin: top + puck.axis * 0.02,
                direction: puck.axis * -1.0,
            };
            let hit = cpu_scene
                .intersect(&along_axis)
                .expect("the laser housing must be intersectable");
            assert!(hit.mat.is_emissive());
            // ...and the hit is the aperture (the puck's top cap), in the beam.
            assert!((hit.p - top).length() < 1e-4);
        }
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
    /// must leave the aperture, bounce off the front-wall mirror, and
    /// land on a diffuse surface — with the landing point, the mirror
    /// hit, and the housing all in frame from the default camera, and
    /// the landing point visible (not occluded). This is the regression
    /// guard for the whole showcase: a moved mirror, a re-aimed laser,
    /// or a camera change that pushes the spot out of frame fails here.
    #[test]
    fn laser_beam_reflects_onto_a_visible_spot() {
        let scene = build_scene();
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("demo scene must contain a laser");
        let trace = scene.trace_laser_beam(laser, 4);
        // Two legs: aperture -> back-wall mirror -> spot.
        assert_eq!(
            trace.legs.len(),
            2,
            "the beam must bounce once (aperture -> back-wall mirror -> spot), got {:?}",
            trace.legs
        );
        let spot = trace
            .spot
            .expect("the beam must land on a diffuse surface");
        assert_eq!(spot.mat.bsdf, Bsdf::Diffuse, "the spot must be diffuse");
        assert!(!spot.mat.is_emissive(), "the spot surface must not be emissive");

        // The mirror hit lies on the back-wall mirror quad.
        let mirror = scene
            .quads
            .iter()
            .find(|q| q.mat.bsdf == Bsdf::Mirror)
            .expect("the back-wall mirror");
        let mirror_hit = trace.legs[0].1;
        let local = mirror_hit - mirror.center;
        assert!(
            local.dot(mirror.tangent).abs() <= mirror.half_w + 1e-3
                && local.dot(mirror.bitangent).abs() <= mirror.half_h + 1e-3,
            "the mirror hit {:?} is outside the mirror quad",
            mirror_hit
        );

        // The specular bridge ladder: one rung per mirror (here one),
        // derived on demand from the fully-built scene (never baked at
        // add-time — the laser is registered before the mirror). The
        // rung is the aperture's mirror image through the mirror plane,
        // and it must be usable from the spot: the image disc's axis
        // lies in the spot's hemisphere and the cone is non-degenerate.
        let ladder = scene.bridges_of_light(laser, 4);
        assert_eq!(ladder.len(), 1, "one mirror, one rung");
        let aperture = laser.center;
        let unfold = |p: Vec3, hit: Vec3, normal: Vec3| {
            let d = p - hit;
            p - normal * (2.0 * d.dot(normal))
        };
        let expected_image = unfold(aperture, mirror_hit, mirror.normal);
        assert!(
            (ladder[0].image_center - expected_image).length() < 1e-3,
            "the rung image {:?} must be the aperture's reflection {:?}",
            ladder[0].image_center,
            expected_image
        );
        assert!((ladder[0].radius - laser.radius).abs() < 1e-6);
        let d = ladder[0].image_center - spot.p;
        assert!(
            d.dot(spot.normal) > 0.0,
            "the rung must point out of the spot's surface"
        );
        assert!(
            d.length_squared() > ladder[0].radius * ladder[0].radius,
            "the rung cone must be non-degenerate"
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
        // (The back-wall mirror hit is behind the camera by design: the
        // showcase shows the housing, the beam's glow, and the spot on
        // the partition — not the mirror itself.)
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

    /// The movement demo's contract, pinned across the whole sweep: for
    /// every laser aim height in the animated range, the beam must
    /// still bounce off the front-wall mirror, land on a diffuse
    /// surface in frame, and carry a *correctly re-derived* ladder
    /// (the image distance equals the beam path length — the unfolding
    /// invariant).
    #[test]
    fn laser_aim_sweep_keeps_the_beam_valid() {
        let cam = default_cam();
        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        let half_fov = (cam.fov_deg * 0.5).to_radians().cos();
        let in_frame = |p: Vec3| -> bool {
            let rel = p - cam.origin;
            let f = rel.dot(forward);
            if f <= 0.0 {
                return false;
            }
            let cos_h = f / (rel.dot(right).abs().hypot(f));
            let cos_v = f / (rel.dot(up).abs().hypot(f));
            cos_h >= half_fov && cos_v >= half_fov
        };

        // The demo's sweep: aim heights 1.0 ± 0.2 around the mirror's
        // center (the window frontend's MIRROR_SWEEP), inside the
        // mirror's y-extent [0.75, 1.25] so the beam always bounces.
        const SWEEP: f32 = 0.2;
        let steps = 9;
        for i in 0..steps {
            let aim_y = 1.0 + SWEEP * (2.0 * (i as f32 / (steps - 1) as f32) - 1.0);
            let scene = build_scene_with(aim_y);
            let laser = scene
                .lights
                .iter()
                .find(|l| l.shape == 2)
                .expect("the laser");
            let trace = scene.trace_laser_beam(laser, 4);
            assert_eq!(
                trace.legs.len(),
                2,
                "aim {aim_y}: the beam must bounce once, got {:?}",
                trace.legs
            );
            let spot = trace
                .spot
                .expect("aim {aim_y}: the beam must land on a diffuse surface");
            assert!(
                in_frame(spot.p),
                "aim {aim_y}: the spot {:?} must stay in frame",
                spot.p
            );
            // The re-derived ladder must be correct for this aim: the
            // unfolding preserves path length, so the spot's distance
            // to the rung's image equals the total beam path length.
            let ladder = scene.bridges_of_light(laser, 4);
            assert_eq!(ladder.len(), 1, "aim {aim_y}: one mirror, one rung");
            let path_len: f32 = trace.legs.iter().map(|(a, b)| (*b - *a).length()).sum();
            let d = (ladder[0].image_center - spot.p).length();
            assert!(
                (d - path_len).abs() < 0.01,
                "aim {aim_y}: the image distance {d} must equal the beam path length {path_len}"
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
        // albedo/π · Le·R^n · cos·Ω_image, Ω_image = π r²/d² (face-on),
        // with n = the number of mirror bounces (the mirror albedo
        // enters once per bounce); the arrival cosine comes from the
        // trace itself (the last leg's direction against the spot's
        // oriented normal).
        let bridge = scene
            .bridge_of_light(laser, 4)
            .expect("the laser must carry a specular bridge");
        let d = (spot - bridge.image_center).length();
        let omega = std::f32::consts::PI * laser.radius * laser.radius / (d * d);
        let mirror = scene
            .quads
            .iter()
            .find(|q| q.mat.bsdf == Bsdf::Mirror)
            .expect("mirror");
        let n_bounces = trace.legs.len() - 1;
        let arrival = (spot - trace.legs[trace.legs.len() - 2].1).normalize();
        let cos = -trace.spot.expect("spot").normal.dot(arrival);
        let expected = 0.8 / std::f32::consts::PI
            * laser.emission.y
            * mirror.mat.albedo.y.powi(n_bounces as i32)
            * cos
            * omega;
        std::fs::write(
            "target/spot_analytic.txt",
            format!("analytic spot radiance: {expected:.1} (d {d:.3}, Ω {omega:.3e}, cos {cos:.3}, bounces {n_bounces})\n"),
        )
        .ok();
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
        // clearly brighter in the laser's color channel than the wall
        // around it — the visual acceptance of the laser showcase.
        // The spot is the traced beam's landing point (never a formula:
        // the trace is the ground truth); the reference is a disc 0.4
        // to the +x, outside the spot's footprint and away from every
        // object. The channel is the laser's dominant color: the first
        // laser is red, so the probe reads the red channel.
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
                    // The red channel: the first laser's color.
                    sum += buffer.get_pixel(x as usize, y as usize).x;
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
            cross.push_str(&format!("{:.0} ", buffer.get_pixel(x, sy).x));
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
