//! Windowed frontend: winit window + wgpu surface displaying the BDPT
//! accumulation buffer directly on the GPU.
//!
//! The compute pass (src/gpu.rs) keeps accumulating HDR samples into a
//! storage buffer; the display pass (shaders/display.wgsl) reads that
//! buffer and applies the same bilateral denoise + Reinhard tone mapping
//! as the terminal path. Pixels never round-trip through the CPU while
//! the camera is stationary, so the preview runs at full GPU speed.

use crate::camera::Cam;
use crate::gpu::GpuRenderer;
use crate::renderer::{
    MAX_ACCUMULATED_FRAMES, MAX_DEPTH_FHD, SAMPLES_FHD, FHD_HEIGHT, FHD_WIDTH,
};
use crate::scene::SceneData;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Preview path depth (same as the terminal frontend). Russian roulette
/// terminates paths by throughput, so the cap can sit at the maximum.
const MAX_DEPTH_PREVIEW: u32 = 8;

/// Camera speed in world units per second (W/S).
const MOVE_SPEED: f32 = 1.0;
/// Camera strafe speed in world units per second (Q/E).
const STRAFE_SPEED: f32 = 1.0;
/// Camera yaw speed in radians per second (A/D).
const ROT_SPEED: f32 = 1.0;
/// Camera pitch speed in radians per second (arrow up/down).
const PITCH_SPEED: f32 = 1.0;

/// The keys the window frontend reacts to. The discriminants are used as
/// indices into the `Keys` state array, so they must stay contiguous.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Key {
    W = 0,
    A,
    S,
    D,
    Q,
    E,
    Up,
    Down,
    COUNT,
}

impl Key {
    /// Map a physical key code to a tracked key, if any.
    fn from_code(code: KeyCode) -> Option<Self> {
        match code {
            KeyCode::KeyW => Some(Key::W),
            KeyCode::KeyA => Some(Key::A),
            KeyCode::KeyS => Some(Key::S),
            KeyCode::KeyD => Some(Key::D),
            KeyCode::KeyQ => Some(Key::Q),
            KeyCode::KeyE => Some(Key::E),
            KeyCode::ArrowUp => Some(Key::Up),
            KeyCode::ArrowDown => Some(Key::Down),
            _ => None,
        }
    }
}

/// Held-key state: one bool per tracked key, allowing any number of keys
/// to be pressed simultaneously (e.g. W+D = forward + turn).
#[derive(Clone, Copy)]
struct Keys {
    pressed: [bool; Key::COUNT as usize],
}

impl Keys {
    fn new() -> Self {
        Self { pressed: [false; Key::COUNT as usize] }
    }

    fn set(&mut self, key: Key, state: ElementState) {
        self.pressed[key as usize] = state == ElementState::Pressed;
    }

    fn is_down(&self, key: Key) -> bool {
        self.pressed[key as usize]
    }
}

/// The display pass: reads the accumulation buffer and post-processes it
/// onto the swapchain (shaders/display.wgsl).
struct Display {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    /// The overlay text texture (status line / help panel). Re-created
    /// whenever the text changes; 1×1 transparent when no overlay is shown.
    overlay_tex: wgpu::Texture,
    overlay_view: wgpu::TextureView,
    /// True when the surface format is linear (not sRGB): the shader must
    /// apply gamma 2.2 itself in that case.
    apply_gamma: bool,
}

/// Uniform bytes for the display pass. Must match `DisplayUniforms` in
/// shaders/display.wgsl (7 u32 fields, padded to 8 for 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DisplayUniformsBytes {
    width: u32,
    height: u32,
    denoise: u32,
    apply_gamma: u32,
    overlay: u32,
    overlay_w: u32,
    overlay_h: u32,
    _pad0: u32,
}

const _: () = {
    assert!(std::mem::size_of::<DisplayUniformsBytes>() == 32);
};

impl Display {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &GpuRenderer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/display.wgsl"));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display-uniforms"),
            size: std::mem::size_of::<DisplayUniformsBytes>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Start with a 1×1 fully transparent overlay texture.
        let (overlay_tex, overlay_view) =
            Self::create_overlay_texture(device, queue, 1, 1, &[0, 0, 0, 0]);

        let bind_group = Self::bind_group(device, &bgl, &uniform_buf, renderer, &overlay_view);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display-pipeline-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buf,
            overlay_tex,
            overlay_view,
            apply_gamma: !surface_format.is_srgb(),
        }
    }

    /// Create the overlay texture and upload `data` (RGBA8, row-major).
    fn create_overlay_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        if data.len() == (width * height * 4) as usize {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
        }
        (tex, view)
    }

    fn bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        renderer: &GpuRenderer,
        overlay_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display-bind-group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: renderer.accum_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(overlay_view) },
            ],
        })
    }

    /// Rebind the accumulation buffer after a resize.
    fn rebind(&mut self, device: &wgpu::Device, renderer: &GpuRenderer) {
        let bgl = self.pipeline.get_bind_group_layout(0);
        self.bind_group = Self::bind_group(device, &bgl, &self.uniform_buf, renderer, &self.overlay_view);
    }

    /// Upload new overlay text. `lines` are drawn white-on-transparent; the
    /// shader composites them over the rendered image. The bind group is
    /// rebuilt because it references the old texture view.
    fn set_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &GpuRenderer,
        lines: &[&str],
    ) {
        let mut canvas = crate::font::TextCanvas::new(lines.len());
        let (w, h) = canvas.draw(lines);
        let (tex, view) = Self::create_overlay_texture(device, queue, w, h, &canvas.data);
        self.overlay_tex = tex;
        self.overlay_view = view;
        let bgl = self.pipeline.get_bind_group_layout(0);
        self.bind_group =
            Self::bind_group(device, &bgl, &self.uniform_buf, renderer, &self.overlay_view);
    }

    fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        filter: bool,
        overlay_visible: bool,
    ) {
        let bytes = DisplayUniformsBytes {
            width,
            height,
            denoise: filter as u32,
            apply_gamma: self.apply_gamma as u32,
            overlay: overlay_visible as u32,
            overlay_w: self.overlay_tex.width(),
            overlay_h: self.overlay_tex.height(),
            _pad0: 0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&bytes));
    }
}

/// The winit application: owns the window, surface and renderer state.
struct App {
    scene: SceneData,
    cam: Cam,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    /// The one-time surface configuration. Its format is the single
    /// source of truth for the display pipeline and is never re-queried
    /// (see `configure_surface`).
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<GpuRenderer>,
    display: Option<Display>,
    filter_enabled: bool,
    frames_accumulated: usize,
    /// Held-key state (press + release), updated by keyboard events.
    keys: Keys,
    /// Timestamp of the previous frame, for delta-time scaling.
    last_frame: Option<Instant>,
    /// Frames per second, smoothed with an exponential moving average so
    /// the title does not flicker.
    fps: f32,
    /// True while the help overlay is on screen (any key dismisses it).
    showing_help: bool,
    /// Screenshot in flight: the FHD renderer accumulates in the
    /// background while the preview keeps running.
    screenshot: Option<Screenshot>,
}

/// A full-quality export in progress. The FHD renderer shares the
/// preview's device; batches are dispatched from `render()` so the window
/// stays responsive instead of blocking for the whole export. The camera
/// is snapshotted at `P`-press time so moving during the export cannot
/// smear the image.
struct Screenshot {
    gpu: GpuRenderer,
    /// Camera frozen at the moment the export started.
    cam: Cam,
    /// Samples still to trace.
    remaining: u32,
    /// Batches dispatched so far, for the status line.
    batches: u32,
}

/// Samples traced per FHD dispatch batch (one submit per batch).
const SPP_BATCH: u32 = 32;

impl App {
    fn new(scene: SceneData, cam: Cam) -> Self {
        Self {
            scene,
            cam,
            window: None,
            surface: None,
            config: None,
            renderer: None,
            display: None,
            filter_enabled: false,
            frames_accumulated: 0,
            keys: Keys::new(),
            last_frame: None,
            fps: 0.0,
            showing_help: false,
            screenshot: None,
        }
    }

    /// (Re)configure the surface for the current window size and rebuild
    /// the resolution-dependent GPU resources.
    fn configure_surface(&mut self) {
        let Some(window) = self.window.as_ref() else { return };
        let Some(surface) = self.surface.as_ref() else { return };
        let Some(renderer) = self.renderer.as_ref() else { return };
        let Some(config) = self.config.as_mut() else { return };

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Always reconfigure: this is also the recovery path for an
        // `Outdated` swapchain, which needs it even at unchanged size.
        config.width = width;
        config.height = height;
        surface.configure(renderer.device(), config);

        // Unchanged resolution (e.g. a spurious Resized event): the
        // accumulation buffer is still valid, keep the samples.
        if (width, height) != renderer.size() {
            let renderer = self.renderer.as_mut().unwrap();
            renderer.resize(width, height);
            let display = self.display.as_mut().unwrap();
            display.rebind(renderer.device(), renderer);
            display.write_uniforms(
                renderer.queue(),
                width,
                height,
                self.filter_enabled,
                self.showing_help,
            );
            self.frames_accumulated = 0;
        }
        window.request_redraw();
    }

    /// Rebuild the overlay texture from the current state: the help panel
    /// while `showing_help`, otherwise nothing (the status readout lives
    /// in the window title, which needs no texture).
    fn update_overlay(&mut self) {
        let Some(renderer) = self.renderer.as_ref() else { return };
        let Some(display) = self.display.as_mut() else { return };
        if self.showing_help {
            display.set_overlay(
                renderer.device(),
                renderer.queue(),
                renderer,
                &[
                    "PATHTRACER-RS CONTROLS",
                    "",
                    "W/S     MOVE FORWARD/BACK",
                    "A/D     TURN LEFT/RIGHT",
                    "Q/E     STRAFE LEFT/RIGHT",
                    "UP/DOWN PITCH",
                    "F       TOGGLE DENOISER",
                    "P       EXPORT FHD PNG",
                    "H       TOGGLE THIS HELP",
                    "ESC     QUIT",
                ],
            );
        } else {
            // 1×1 transparent texture: the shader skips it via the
            // overlay flag in the uniforms.
            display.set_overlay(renderer.device(), renderer.queue(), renderer, &[]);
        }
        let (w, h) = renderer.size();
        display.write_uniforms(renderer.queue(), w, h, self.filter_enabled, self.showing_help);
    }

    /// One iteration: trace one sample per pixel (if still accumulating),
    /// then present the display pass.
    fn render(&mut self) {
        let Some(window) = self.window.as_ref() else { return };
        let Some(surface) = self.surface.as_ref() else { return };
        let Some(renderer) = self.renderer.as_mut() else { return };
        let Some(display) = self.display.as_ref() else { return };

        // Delta time since the previous frame, so movement speed is
        // independent of the display refresh rate.
        let now = Instant::now();
        let dt = self
            .last_frame
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.0);
        self.last_frame = Some(now);
        // Smoothed FPS for the status line (EMA; avoids flicker).
        if dt > 0.0 {
            let instant_fps = 1.0 / dt;
            self.fps = if self.fps == 0.0 {
                instant_fps
            } else {
                self.fps * 0.9 + instant_fps * 0.1
            };
        }

        // Continuous movement from held keys: any combination of W/A/S/D
        // can be active at once (e.g. W+D = forward + turn).
        let mut moved = false;
        if self.keys.is_down(Key::W) {
            self.cam.move_forward(MOVE_SPEED * dt);
            moved = true;
        }
        if self.keys.is_down(Key::S) {
            self.cam.move_forward(-MOVE_SPEED * dt);
            moved = true;
        }
        if self.keys.is_down(Key::A) {
            self.cam.rotate(ROT_SPEED * dt);
            moved = true;
        }
        if self.keys.is_down(Key::D) {
            self.cam.rotate(-ROT_SPEED * dt);
            moved = true;
        }
        if self.keys.is_down(Key::Q) {
            self.cam.strafe(-STRAFE_SPEED * dt);
            moved = true;
        }
        if self.keys.is_down(Key::E) {
            self.cam.strafe(STRAFE_SPEED * dt);
            moved = true;
        }
        if self.keys.is_down(Key::Up) {
            self.cam.pitch(PITCH_SPEED * dt);
            moved = true;
        }
        if self.keys.is_down(Key::Down) {
            self.cam.pitch(-PITCH_SPEED * dt);
            moved = true;
        }
        if moved {
            // Camera moved: the accumulated image is stale, start over.
            renderer.reset_accumulation();
            self.frames_accumulated = 0;
        }

        // Screenshot in flight: dispatch one batch per frame so the
        // window keeps presenting while the export accumulates. The
        // camera is the one frozen at `P`-press time.
        if let Some(shot) = self.screenshot.as_mut() {
            let spp = SPP_BATCH.min(shot.remaining);
            shot.gpu.render_frame_spp(&shot.cam, MAX_DEPTH_FHD, spp);
            shot.remaining -= spp;
            shot.batches += 1;
            if shot.remaining == 0 {
                // Done: read back, tone-map, save. `take()` moves the
                // renderer out so its buffers can be read before drop.
                let mut shot = self.screenshot.take().unwrap();
                let mut buffer = shot.gpu.read_pixels();
                if self.filter_enabled {
                    crate::output::apply_filters(&mut buffer);
                }
                match buffer.save_as_png("screenshot.png") {
                    Ok(()) => eprintln!("saved screenshot.png"),
                    Err(e) => eprintln!("failed to save screenshot: {e}"),
                }
            }
        }

        // Progressive accumulation while the camera is stationary.
        if self.frames_accumulated < MAX_ACCUMULATED_FRAMES {
            renderer.render_frame(&self.cam, MAX_DEPTH_PREVIEW);
            self.frames_accumulated += 1;
        }

        // Status line in the window title: backend, accumulated samples
        // and fps. (The title is the cheapest place for a status readout
        // that needs no font rendering pipeline.)
        {
            let spp = renderer.samples_accumulated();
            let shot = match &self.screenshot {
                Some(s) => format!(" | exporting {}%", (s.batches * SPP_BATCH * 100 / SAMPLES_FHD).min(100)),
                None => String::new(),
            };
            window.set_title(&format!(
                "pathtracer-rs — {} spp | {:.0} fps{}",
                spp, self.fps, shot
            ));
        }

        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.configure_surface();
                return;
            }
            // Timeout/Occluded: skip this frame and try again later.
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = renderer.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("display-encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&display.pipeline);
            pass.set_bind_group(0, &display.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        renderer.queue().submit(Some(encoder.finish()));
        renderer.queue().present(frame);

        // Keep the loop running at display rate even when converged.
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // Already initialized (e.g. redundant Resumed).
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("pathtracer-rs"))
                .expect("failed to create window"),
        );

        // Instance carrying the windowing system's display handle, as
        // required when creating a surface from a window.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(event_loop.owned_display_handle()),
        ));
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no suitable GPU adapter");
        let info = adapter.get_info();
        let adapter_info = format!("{} ({})", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(&crate::gpu::device_descriptor()))
            .expect("failed to create GPU device");

        let size = window.inner_size();
        let renderer = GpuRenderer::from_parts(
            device,
            queue,
            &self.scene,
            size.width.max(1),
            size.height.max(1),
            adapter_info,
        );

        // The display pipeline needs the surface format, so configure
        // the surface first.
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(renderer.device(), &config);

        let display = Display::new(renderer.device(), renderer.queue(), &renderer, config.format);
        display.write_uniforms(
            renderer.queue(),
            size.width.max(1),
            size.height.max(1),
            false,
            false,
        );

        self.window = Some(window);
        self.surface = Some(surface);
        self.config = Some(config);
        self.renderer = Some(renderer);
        self.display = Some(display);
        self.frames_accumulated = 0;
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => event_loop.exit(),
            WindowEvent::Resized(_) => self.configure_surface(),
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                // Held-key tracking: press and release both update the
                // state array; the actual motion is applied per frame in
                // `render()`, so multiple keys can be held at once.
                if let Some(key) = Key::from_code(code) {
                    self.keys.set(key, state);
                }

                // Discrete actions fire on the initial press only
                // (`repeat` is false), not on OS key-repeat.
                if state == ElementState::Pressed && !repeat {
                    match code {
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::KeyF => {
                            self.filter_enabled = !self.filter_enabled;
                            if let Some(renderer) = self.renderer.as_ref() {
                                if let Some(display) = self.display.as_ref() {
                                    let (w, h) = renderer.size();
                                    display.write_uniforms(
                                        renderer.queue(),
                                        w,
                                        h,
                                        self.filter_enabled,
                                        self.showing_help,
                                    );
                                }
                            }
                            if let Some(window) = self.window.as_ref() {
                                window.request_redraw();
                            }
                        }
                        KeyCode::KeyH => {
                            // Toggle the help overlay; any key dismisses it
                            // (handled below via `showing_help`).
                            self.showing_help = !self.showing_help;
                            self.update_overlay();
                            if let Some(window) = self.window.as_ref() {
                                window.request_redraw();
                            }
                        }
                        KeyCode::KeyP => {
                            // Start a full-quality export. It accumulates
                            // in batches from `render()` so the window
                            // stays responsive; the result is saved to
                            // screenshot.png when the last batch lands.
                            // The camera is frozen now so moving during
                            // the export cannot smear the image.
                            if self.screenshot.is_none() {
                                if let Some(renderer) = self.renderer.as_ref() {
                                    self.screenshot = Some(Screenshot {
                                        gpu: GpuRenderer::from_parts(
                                            renderer.device_clone(),
                                            renderer.queue_clone(),
                                            &self.scene,
                                            FHD_WIDTH,
                                            FHD_HEIGHT,
                                            renderer.adapter_info().to_string(),
                                        ),
                                        cam: self.cam,
                                        remaining: SAMPLES_FHD,
                                        batches: 0,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Any key dismisses the help overlay (except the H that
                // just toggled it on, handled above).
                if self.showing_help
                    && state == ElementState::Pressed
                    && !repeat
                    && code != KeyCode::KeyH
                {
                    self.showing_help = false;
                    self.update_overlay();
                }
            }
            _ => {}
        }
    }
}

/// Entry point for the windowed frontend. Returns an error when the
/// window system or GPU is unavailable (the caller falls back to the
/// terminal UI).
pub fn run_window(scene: SceneData, cam: Cam) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("failed to create event loop: {e}"))?;
    let mut app = App::new(scene, cam);
    event_loop.run_app(&mut app).map_err(|e| format!("event loop error: {e}"))
}

#[cfg(test)]
mod tests {
    /// The display shader must validate with the same naga version wgpu
    /// uses (runs on every machine, GPU or not).
    #[test]
    fn display_shader_validates() {
        let source = include_str!("../shaders/display.wgsl");
        let module = naga::front::wgsl::Frontend::new()
            .parse(source)
            .expect("WGSL parse failed");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("WGSL validation failed");
        assert!(!module.entry_points.is_empty());
    }
}
