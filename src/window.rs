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
use crate::renderer::MAX_ACCUMULATED_FRAMES;
use crate::scene::SceneData;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// How many preview frames to accumulate while the camera is stationary.
//const MAX_ACCUMULATED_FRAMES: u32 = 100000;
/// Preview path depth (same as the terminal frontend).
const MAX_DEPTH_PREVIEW: u32 = 4;

/// Camera speed in world units per second (W/S).
const MOVE_SPEED: f32 = 1.0;
/// Camera rotation speed in radians per second (A/D).
const ROT_SPEED: f32 = 1.0;

/// The keys the window frontend reacts to. The discriminants are used as
/// indices into the `Keys` state array, so they must stay contiguous.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Key {
    W = 0,
    A,
    S,
    D,
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
    /// True when the surface format is linear (not sRGB): the shader must
    /// apply gamma 2.2 itself in that case.
    apply_gamma: bool,
}

impl Display {
    fn new(
        device: &wgpu::Device,
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
            ],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display-uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = Self::bind_group(device, &bgl, &uniform_buf, renderer);

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
            apply_gamma: !surface_format.is_srgb(),
        }
    }

    fn bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        renderer: &GpuRenderer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display-bind-group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: renderer.accum_buf().as_entire_binding() },
            ],
        })
    }

    /// Rebind the accumulation buffer after a resize.
    fn rebind(&mut self, device: &wgpu::Device, renderer: &GpuRenderer) {
        let bgl = self.pipeline.get_bind_group_layout(0);
        self.bind_group = Self::bind_group(device, &bgl, &self.uniform_buf, renderer);
    }

    fn write_uniforms(&self, queue: &wgpu::Queue, width: u32, height: u32, filter: bool) {
        let bytes: [u8; 16] =
            bytemuck::cast([width, height, filter as u32, self.apply_gamma as u32]);
        queue.write_buffer(&self.uniform_buf, 0, &bytes);
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
}

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
        }
    }

    /// (Re)configure the surface for the current window size and rebuild
    /// the resolution-dependent GPU resources.
    ///
    /// The format comes from the stored configuration instead of a fresh
    /// `get_default_config` call: that returns the first entry of the
    /// adapter's capability list, which is not stable across queries on
    /// all platforms (on X11/Vulkan it has been observed to flip between
    /// Rgba8UnormSrgb and Bgra8UnormSrgb once the window is mapped).
    /// Configuring the surface with a format the display pipeline was not
    /// built for is a validation error, so the format is fixed for the
    /// lifetime of the surface.
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
            display.write_uniforms(renderer.queue(), width, height, self.filter_enabled);
            self.frames_accumulated = 0;
        }
        window.request_redraw();
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
        if moved {
            // Camera moved: the accumulated image is stale, start over.
            renderer.reset_accumulation();
            self.frames_accumulated = 0;
        }

        // Progressive accumulation while the camera is stationary.
        if self.frames_accumulated < MAX_ACCUMULATED_FRAMES {
            renderer.render_frame(&self.cam, MAX_DEPTH_PREVIEW);
            self.frames_accumulated += 1;
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

        let display = Display::new(renderer.device(), &renderer, config.format);
        display.write_uniforms(renderer.queue(), size.width.max(1), size.height.max(1), false);

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
                        KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
                        KeyCode::KeyF => {
                            self.filter_enabled = !self.filter_enabled;
                            if let Some(renderer) = self.renderer.as_ref() {
                                if let Some(display) = self.display.as_ref() {
                                    let (w, h) = renderer.size();
                                    display.write_uniforms(renderer.queue(), w, h, self.filter_enabled);
                                }
                            }
                            if let Some(window) = self.window.as_ref() {
                                window.request_redraw();
                            }
                        }
                        _ => {}
                    }
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
