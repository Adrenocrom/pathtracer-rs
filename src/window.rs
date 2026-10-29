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
use crate::scene::SceneData;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// How many preview frames to accumulate while the camera is stationary.
const MAX_ACCUMULATED_FRAMES: u32 = 128;
/// Preview path depth (same as the terminal frontend).
const MAX_DEPTH_PREVIEW: u32 = 4;

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
    /// Kept for `get_default_config` on resize (it needs the adapter).
    adapter: Option<wgpu::Adapter>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<GpuRenderer>,
    display: Option<Display>,
    filter_enabled: bool,
    frames_accumulated: u32,
}

impl App {
    fn new(scene: SceneData, cam: Cam) -> Self {
        Self {
            scene,
            cam,
            window: None,
            surface: None,
            adapter: None,
            config: None,
            renderer: None,
            display: None,
            filter_enabled: false,
            frames_accumulated: 0,
        }
    }

    /// (Re)configure the surface for the current window size and rebuild
    /// the resolution-dependent GPU resources.
    fn configure_surface(&mut self) {
        let Some(window) = self.window.as_ref() else { return };
        let Some(surface) = self.surface.as_ref() else { return };
        let Some(adapter) = self.adapter.as_ref() else { return };
        let Some(renderer) = self.renderer.as_ref() else { return };

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut config = surface
            .get_default_config(adapter, width, height)
            .expect("surface not supported by adapter");
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(renderer.device(), &config);
        self.config = Some(config);

        // Rebuild the accumulation buffer at the new resolution and
        // rebind the display pass to it.
        let renderer = self.renderer.as_mut().unwrap();
        renderer.resize(width, height);
        let display = self.display.as_mut().unwrap();
        display.rebind(renderer.device(), renderer);
        display.write_uniforms(renderer.queue(), width, height, self.filter_enabled);
        self.frames_accumulated = 0;
        window.request_redraw();
    }

    /// One iteration: trace one sample per pixel (if still accumulating),
    /// then present the display pass.
    fn render(&mut self) {
        let Some(window) = self.window.as_ref() else { return };
        let Some(surface) = self.surface.as_ref() else { return };
        let Some(renderer) = self.renderer.as_mut() else { return };
        let Some(display) = self.display.as_ref() else { return };

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

    /// Shared key handling: returns true when the camera moved.
    fn handle_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::KeyW => self.cam.move_forward(0.1),
            KeyCode::KeyS => self.cam.move_forward(-0.1),
            KeyCode::KeyA => self.cam.rotate(0.05),
            KeyCode::KeyD => self.cam.rotate(-0.05),
            _ => return false,
        }
        true
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
        self.adapter = Some(adapter);
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match code {
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
                _ => {
                    if self.handle_key(code) {
                        if let Some(renderer) = self.renderer.as_mut() {
                            renderer.reset_accumulation();
                        }
                        self.frames_accumulated = 0;
                        if let Some(window) = self.window.as_ref() {
                            window.request_redraw();
                        }
                    }
                }
            },
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