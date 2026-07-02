use wgpu::util::DeviceExt;
use crate::{Vec3, Material, MaterialType, VEC_ZERO, Cam, PixelBuffer};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVec3 {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for GpuVec3 {
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z, _pad: 0.0 }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    albedo: GpuVec3,
    emission: GpuVec3,
    mat_type: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSphere {
    center: GpuVec3,
    radius: f32,
    mat_id: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCube {
    min: GpuVec3,
    max: GpuVec3,
    mat_id: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPlane {
    point: GpuVec3,
    normal: GpuVec3,
    mat_id: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuScene {
    sphere_count: u32,
    cube_count: u32,
    plane_count: u32,
    material_count: u32,
}

pub async fn render_gpu(width: usize, height: usize, samples: usize, cam: &Cam) -> PixelBuffer {
    // Initialize WebGPU
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect("Failed to find adapter");
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.expect("Failed to create device");

    // Define Scene Data (Mirroring main.rs)
    let white_mat = Material { albedo: Vec3::new(1.0, 1.0, 1.0), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let red_mat = Material { albedo: Vec3::new(1.5, 0.1, 0.1), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let green_mat = Material { albedo: Vec3::new(0.1, 0.9, 0.1), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let light_mat = Material { albedo: Vec3::new(0.5, 0.5, 0.5), emission: Vec3::new(1.0, 1.0, 0.4), mat_type: MaterialType::Emissive };
    let dim_light_mat = Material { albedo: Vec3::new(1.0, 1.0, 1.0), emission: Vec3::new(0.001, 0.001, 0.001), mat_type: MaterialType::Emissive };

    let materials: Vec<GpuMaterial> = vec![white_mat, red_mat, green_mat, light_mat, dim_light_mat].into_iter().map(|m| GpuMaterial {
        albedo: m.albedo.into(),
        emission: m.emission.into(),
        mat_type: if let MaterialType::Diffuse = m.mat_type { 0 } else { 1 },
        _pad: [0; 3],
    }).collect();

    let spheres = vec![
        GpuSphere { center: Vec3::new(0.5, 0.4, 0.5).into(), radius: 0.4, mat_id: 0, _pad: [0; 2] },
        GpuSphere { center: Vec3::new(-1.5, 0.4, 0.1).into(), radius: 0.4, mat_id: 0, _pad: [0; 2] },
        GpuSphere { center: Vec3::new(2.0, 2.0, -2.0).into(), radius: 0.2, mat_id: 3, _pad: [0; 2] },
        GpuSphere { center: Vec3::new(-2.0, 2.0, -2.0).into(), radius: 0.2, mat_id: 3, _pad: [0; 2] },
    ];

    let cubes = vec![
        GpuCube { 
            min: (Vec3::new(0.0, 0.3, -0.5) - Vec3::new(0.3, 0.3, 0.3)).into(), 
            max: (Vec3::new(0.0, 0.3, -0.5) + Vec3::new(0.3, 0.3, 0.3)).into(), 
            mat_id: 0, _pad: [0; 3] 
        },
    ];

    let planes = vec![
        GpuPlane { point: Vec3::new(0.0, 0.0, 0.0).into(), normal: Vec3::new(0.0, 1.0, 0.0).into(), mat_id: 0, _pad: [0; 3] },
        GpuPlane { point: Vec3::new(0.0, 2.0, 0.0).into(), normal: Vec3::new(0.0, -1.0, 0.0).into(), mat_id: 0, _pad: [0; 3] },
        GpuPlane { point: Vec3::new(-2.0, 1.0, 0.0).into(), normal: Vec3::new(1.0, 0.0, 0.0).into(), mat_id: 1, _pad: [0; 3] },
        GpuPlane { point: Vec3::new(2.0, 1.0, 0.0).into(), normal: Vec3::new(-1.0, 0.0, 0.0).into(), mat_id: 2, _pad: [0; 3] },
        GpuPlane { point: Vec3::new(0.0, 1.0, 2.0).into(), normal: Vec3::new(0.0, 0.0, -1.0).into(), mat_id: 0, _pad: [0; 3] },
        GpuPlane { point: Vec3::new(0.0, 1.0, -2.0).into(), normal: Vec3::new(0.0, 0.0, 1.0).into(), mat_id: 0, _pad: [0; 3] },
        GpuPlane { point: Vec3::new(0.0, 1.9, 1.0).into(), normal: Vec3::new(0.0, -1.0, 0.0).into(), mat_id: 4, _pad: [0; 3] },
    ];

    let scene_info = GpuScene {
        sphere_count: spheres.len() as u32,
        cube_count: cubes.len() as u32,
        plane_count: planes.len() as u32,
        material_count: materials.len() as u32,
    };

    let mat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Materials"),
        contents: bytemuck::cast_slice(&materials),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Spheres"),
        contents: bytemuck::cast_slice(&spheres),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cube_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cubes"),
        contents: bytemuck::cast_slice(&cubes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let plane_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Planes"),
        contents: bytemuck::cast_slice(&planes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SceneInfo"),
        contents: bytemuck::cast_slice(&[scene_info]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: (width * height * 12) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: (width * height * 12) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: mat_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: sphere_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: cube_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: plane_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scene_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (width as u32 + 15) / 16,
            (height as u32 + 15) / 16,
            1,
        );
    }

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (width * height * 12) as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));
    
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::PollType::Poll);
    
    if let Ok(Ok(_)) = receiver.recv() {
        if let Ok(data) = slice.get_mapped_range() {
            let result = bytemuck::cast_slice::<u8, f32>(&data[..]).to_vec();
            drop(data);
            staging_buffer.unmap();
            
            let mut pixels = vec![Vec3::default(); width * height];
            for (i, chunk) in result.chunks(3).enumerate() {
                if i < pixels.len() {
                    pixels[i] = Vec3::new(chunk[0], chunk[1], chunk[2]);
                }
            }
            return PixelBuffer { width, height, pixels };
        }
    }

    PixelBuffer { 
        width, 
        height, 
        pixels: vec![Vec3::default(); width * height] 
    }
}
