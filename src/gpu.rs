//! GPU renderer: the BDPT compute pipeline on wgpu.
//!
//! One thread-per-pixel traces a full BDPT path and blends the result into a
//! persistent accumulation buffer on the GPU. The CPU reads the buffer back
//! only when pixels are needed (terminal frame or PNG export), so the render
//! loop never stalls on a GPU round-trip while the camera is stationary.
//!
//! Intersection runs on a flat BVH: the scene's prims (spheres, quads,
//! cubes, planes — one kind-tagged array) and the node tree are built
//! here and uploaded once as two storage buffers; the shader traverses
//! with an explicit stack (WGSL has no recursion) and culls every box
//! by the current best hit distance.

use crate::math::Vec3;
use crate::output::PixelBuffer;
use crate::scene::{LightData, SceneData};
use bytemuck::{Pod, Zeroable};

/// Camera + frame state, uploaded once per frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    cam_origin: [f32; 3],
    fov_h: f32,
    cam_forward: [f32; 3],
    fov_v: f32,
    cam_right: [f32; 3],
    width: f32,
    cam_up: [f32; 3],
    height: f32,
    max_depth: u32,
    n_lights: u32,
    /// Number of samples already accumulated in `accum` (0 = reset).
    samples: u32,
    reset: u32,
    /// Samples traced per pixel in this dispatch.
    spp: u32,
    /// Scene prims; 0 = empty scene, the shader skips traversal.
    n_prims: u32,
    /// Per-sample firefly clamp bound (0 = disable). Computed by
    /// `bdpt::firefly_clamp_bound` as a loose multiple of the scene's
    /// brightest emitter luminance.
    firefly_clamp: f32,
    _pad2: u32,
}

/// One scene primitive in GPU form (matches `Prim` in shaders/bdpt.wgsl).
///
/// All four families (sphere/quad/cube/plane) share this layout, kind-
/// tagged, so a single array — and a single BVH leaf range — can hold
/// them all. The families share slots: the cube's min/max live in
/// `center`/`normal`, the quad's `half_w` lives in `radius`.
///
/// Layout rule: every `vec3<f32>` is immediately followed by an `f32` so
/// WGSL's implicit 16-byte alignment never inserts padding that Rust's
/// `#[repr(C)]` doesn't have.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPrim {
    kind: u32,
    ior: f32,
    mat_type: u32,
    _pad0: f32,
    center: [f32; 3],
    radius: f32,
    normal: [f32; 3],
    half_h: f32,
    tangent: [f32; 3],
    _pad1: f32,
    bitangent: [f32; 3],
    _pad2: f32,
    albedo: [f32; 3],
    _pad3: f32,
    emission: [f32; 3],
    _pad4: f32,
}

/// One flat BVH node in GPU form (matches `BvhNode` in shaders/bdpt.wgsl).
///
/// * Leaf (`count > 0`): `right_first` indexes the first prim of a
///   contiguous run of `count` prims — the prim array is uploaded in
///   leaf order, so a leaf needs no per-prim indirection.
/// * Inner (`count == 0`): `right_first` is the right child's node
///   index; the left child is always this node's index + 1, because the
///   builder emits nodes in pre-order (the same layout trick as pbrt's
///   LinearBVHNode), so no second child field is needed.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBvhNode {
    min: [f32; 3],
    right_first: u32,
    max: [f32; 3],
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLight {
    shape: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    center: [f32; 3],
    radius: f32,
    normal: [f32; 3],
    _pad3: f32,
    tangent: [f32; 3],
    _pad4: f32,
    bitangent: [f32; 3],
    half_w: f32,
    half_h: f32,
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    emission: [f32; 3],
    _pad8: f32,
}

impl GpuLight {
    fn from_light_data(l: &LightData) -> Self {
        Self {
            shape: l.shape,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            center: [l.center.x, l.center.y, l.center.z],
            radius: l.radius,
            normal: [l.normal.x, l.normal.y, l.normal.z],
            _pad3: 0.0,
            tangent: [l.tangent.x, l.tangent.y, l.tangent.z],
            _pad4: 0.0,
            bitangent: [l.bitangent.x, l.bitangent.y, l.bitangent.z],
            half_w: l.half_w,
            half_h: l.half_h,
            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
            emission: [l.emission.x, l.emission.y, l.emission.z],
            _pad8: 0.0,
        }
    }
}

const MAT_DIFFUSE: u32 = 0;
const MAT_EMISSIVE: u32 = 1;
const MAT_MIRROR: u32 = 2;
const MAT_GLASS: u32 = 3;

// Prim kind tags, shared with shaders/bdpt.wgsl.
const KIND_SPHERE: u32 = 0;
const KIND_QUAD: u32 = 1;
const KIND_CUBE: u32 = 2;
const KIND_PLANE: u32 = 3;

/// Compile-time layout checks: the Rust structs must match the WGSL
/// structs byte-for-byte. WGSL aligns vec3<f32> to 16 bytes, so every
/// vec3 is followed by an explicit f32 pad in both languages.
const _: () = {
    assert!(std::mem::size_of::<GpuPrim>() == 112);
    assert!(std::mem::size_of::<GpuBvhNode>() == 32);
    assert!(std::mem::size_of::<GpuLight>() == 112);
    assert!(std::mem::size_of::<Uniforms>() == 96);
};

// --- Flat BVH construction ---
//
// The GPU analogue of src/bvh.rs's BVHNode: same median-split-along-
// the-widest-axis strategy, but emitted as a flat array the shader can
// traverse with an explicit stack (WGSL has no recursion). Nodes are
// written in pre-order, so an inner node's left child is always its own
// index + 1 and only the right child needs storing.

/// Maximum BVH depth the builder allows. A pop-based traversal of a
/// depth-d binary tree holds at most d+1 pending nodes; the shader's
/// stack is 32 entries, so 24 leaves plenty of headroom even for
/// degenerate trees.
const MAX_BVH_DEPTH: u32 = 24;

/// The bounding box of one prim, in the same terms as geometry.rs.
fn prim_bbox(p: &GpuPrim) -> ([f32; 3], [f32; 3]) {
    match p.kind {
        KIND_SPHERE => {
            let r = p.radius;
            (
                [p.center[0] - r, p.center[1] - r, p.center[2] - r],
                [p.center[0] + r, p.center[1] + r, p.center[2] + r],
            )
        }
        KIND_QUAD => {
            // The four corners of the rectangle.
            let mut bmin = [f32::INFINITY; 3];
            let mut bmax = [f32::NEG_INFINITY; 3];
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    for a in 0..3 {
                        let c = p.center[a]
                            + p.tangent[a] * (sx * p.radius)
                            + p.bitangent[a] * (sy * p.half_h);
                        bmin[a] = bmin[a].min(c);
                        bmax[a] = bmax[a].max(c);
                    }
                }
            }
            (bmin, bmax)
        }
        KIND_CUBE => (p.center, p.normal),
        _ => {
            // Planes are infinite; a ±1e6 box matches geometry.rs's
            // Plane::bounding_box (and makes the root box huge, exactly
            // as on the CPU — the demo scene uses finite quads instead).
            ([-1e6; 3], [1e6; 3])
        }
    }
}

/// Build the flat BVH over `prims` (reordered in place into leaf order)
/// and return the node array. Returns `None` for an empty scene.
///
/// The recursion is bounded by `MAX_BVH_DEPTH`: beyond it the split
/// falls back to a plain median cut, which guarantees progress (each
/// side gets at least one prim) and keeps the shader's fixed stack
/// safe.
fn build_bvh(prims: &mut Vec<GpuPrim>) -> Option<Vec<GpuBvhNode>> {
    if prims.is_empty() {
        return None;
    }
    let mut nodes: Vec<GpuBvhNode> = Vec::new();
    let n = prims.len();
    build_bvh_rec(prims, 0, n, 0, &mut nodes);
    Some(nodes)
}

/// Recursive builder. `first..end` is the prim slice still to process,
/// `depth` the current tree depth. Emits this node, then the left
/// subtree, then the right subtree (pre-order).
fn build_bvh_rec(
    prims: &mut [GpuPrim],
    first: usize,
    end: usize,
    depth: u32,
    nodes: &mut Vec<GpuBvhNode>,
) -> usize {
    let count = end - first;
    let node_idx = nodes.len();
    nodes.push(GpuBvhNode {
        min: [0.0; 3],
        right_first: 0,
        max: [0.0; 3],
        count: 0,
    });

    // Total bounds of this subtree.
    let mut bmin = [f32::INFINITY; 3];
    let mut bmax = [f32::NEG_INFINITY; 3];
    for p in &prims[first..end] {
        let (pmin, pmax) = prim_bbox(p);
        for a in 0..3 {
            bmin[a] = bmin[a].min(pmin[a]);
            bmax[a] = bmax[a].max(pmax[a]);
        }
    }

    if count <= 1 || depth >= MAX_BVH_DEPTH {
        // Leaf. The prims are already contiguous in leaf order.
        nodes[node_idx] = GpuBvhNode {
            min: bmin,
            right_first: first as u32,
            max: bmax,
            count: count as u32,
        };
        return node_idx;
    }

    // Median split along the widest axis (same heuristic as bvh.rs).
    let mut extent = [0.0f32; 3];
    for a in 0..3 {
        extent[a] = bmax[a] - bmin[a];
    }
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    let slice = &mut prims[first..end];
    slice.sort_unstable_by(|a, b| {
        let (amin, _) = prim_bbox(a);
        let (bmin, _) = prim_bbox(b);
        amin[axis]
            .partial_cmp(&bmin[axis])
            .expect("BVH sort: prim bbox min is not NaN")
    });
    let mid = first + count / 2;

    // Left subtree is emitted immediately after this node (pre-order),
    // so its index is node_idx + 1; the right subtree's index is known
    // only after the left recursion returns.
    build_bvh_rec(prims, first, mid, depth + 1, nodes);
    let right_idx = build_bvh_rec(prims, mid, end, depth + 1, nodes);
    nodes[node_idx] = GpuBvhNode {
        min: bmin,
        right_first: right_idx as u32,
        max: bmax,
        count: 0,
    };
    node_idx
}

fn mat_type_u32(m: &crate::scene::MaterialData) -> u32 {
    if m.is_emissive() {
        MAT_EMISSIVE
    } else {
        match m.bsdf {
            crate::geometry::Bsdf::Diffuse => MAT_DIFFUSE,
            crate::geometry::Bsdf::Mirror => MAT_MIRROR,
            crate::geometry::Bsdf::Glass { .. } => MAT_GLASS,
        }
    }
}

/// The ior of a glass material (1.0 for everything else, so the shader can
/// read the field unconditionally).
fn mat_ior(m: &crate::scene::MaterialData) -> f32 {
    match m.bsdf {
        crate::geometry::Bsdf::Glass { ior } => ior,
        _ => 1.0,
    }
}

fn vec3s(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// The GPU renderer. Owns the wgpu device and all per-scene resources.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout + uploaded scene buffers, kept together so the
    /// bind group can be rebuilt when the accumulation buffer is
    /// recreated on resize.
    res: StaticResources,
    bind_group: wgpu::BindGroup,
    accum_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    width: u32,
    height: u32,
    n_lights: u32,
    /// Scene prims; 0 = empty scene, the shader skips traversal.
    n_prims: u32,
    /// Per-sample firefly clamp bound, uploaded as a uniform (0 = the
    /// scene has no lights; the shader then disables the clamp).
    firefly_clamp: f32,
    /// Number of samples accumulated in the current accumulation epoch.
    samples: u32,
    /// True when the accumulation buffer must be overwritten this frame.
    reset: bool,
    adapter_info: String,
}

/// GPU resources that outlive a resize: the bind group layout, the
/// per-frame uniform buffer and the scene geometry uploaded once.
struct StaticResources {
    bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    prim_buf: wgpu::Buffer,
    bvh_buf: wgpu::Buffer,
    light_buf: wgpu::Buffer,
}

impl StaticResources {
    /// Bind group layout order: uniform + 3 read-only storage + 1
    /// read-write storage (the accumulation buffer).
    fn bind_group(&self, device: &wgpu::Device, accum_buf: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bdpt-bind-group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.prim_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bvh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: accum_buf.as_entire_binding(),
                },
            ],
        })
    }
}

impl GpuRenderer {
    /// Create a renderer for `scene` at the given resolution.
    pub fn new(scene: &SceneData, width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| format!("no suitable GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let adapter_info = format!("{} ({})", info.name, info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&device_descriptor()))
            .map_err(|e| format!("failed to create GPU device: {e}"))?;

        Ok(Self::from_parts(
            device,
            queue,
            scene,
            width,
            height,
            adapter_info,
        ))
    }

    /// Build a renderer from an existing device/queue (shared with the
    /// windowed frontend's surface) at the given resolution.
    pub fn from_parts(
        device: wgpu::Device,
        queue: wgpu::Queue,
        scene: &SceneData,
        width: u32,
        height: u32,
        adapter_info: String,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/bdpt.wgsl"));

        // --- Scene buffers (flat POD arrays, uploaded once) ---
        // All four prim families share one kind-tagged array so a BVH
        // leaf can reference a contiguous range with no per-prim
        // indirection.
        let mut prims: Vec<GpuPrim> = Vec::new();
        for s in &scene.spheres {
            prims.push(GpuPrim {
                kind: KIND_SPHERE,
                ior: mat_ior(&s.mat),
                mat_type: mat_type_u32(&s.mat),
                _pad0: 0.0,
                center: vec3s(s.center),
                radius: s.radius,
                normal: [0.0; 3],
                half_h: 0.0,
                tangent: [0.0; 3],
                _pad1: 0.0,
                bitangent: [0.0; 3],
                _pad2: 0.0,
                albedo: vec3s(s.mat.albedo),
                _pad3: 0.0,
                emission: vec3s(s.mat.emission),
                _pad4: 0.0,
            });
        }
        for q in &scene.quads {
            prims.push(GpuPrim {
                kind: KIND_QUAD,
                ior: mat_ior(&q.mat),
                mat_type: mat_type_u32(&q.mat),
                _pad0: 0.0,
                center: vec3s(q.center),
                radius: q.half_w,
                normal: vec3s(q.normal),
                half_h: q.half_h,
                tangent: vec3s(q.tangent),
                _pad1: 0.0,
                bitangent: vec3s(q.bitangent),
                _pad2: 0.0,
                albedo: vec3s(q.mat.albedo),
                _pad3: 0.0,
                emission: vec3s(q.mat.emission),
                _pad4: 0.0,
            });
        }
        for c in &scene.cubes {
            prims.push(GpuPrim {
                kind: KIND_CUBE,
                ior: mat_ior(&c.mat),
                mat_type: mat_type_u32(&c.mat),
                _pad0: 0.0,
                center: vec3s(c.min),
                radius: 0.0,
                normal: vec3s(c.max),
                half_h: 0.0,
                tangent: [0.0; 3],
                _pad1: 0.0,
                bitangent: [0.0; 3],
                _pad2: 0.0,
                albedo: vec3s(c.mat.albedo),
                _pad3: 0.0,
                emission: vec3s(c.mat.emission),
                _pad4: 0.0,
            });
        }
        for p in &scene.planes {
            prims.push(GpuPrim {
                kind: KIND_PLANE,
                ior: mat_ior(&p.mat),
                mat_type: mat_type_u32(&p.mat),
                _pad0: 0.0,
                center: vec3s(p.point),
                radius: 0.0,
                normal: vec3s(p.normal),
                half_h: 0.0,
                tangent: [0.0; 3],
                _pad1: 0.0,
                bitangent: [0.0; 3],
                _pad2: 0.0,
                albedo: vec3s(p.mat.albedo),
                _pad3: 0.0,
                emission: vec3s(p.mat.emission),
                _pad4: 0.0,
            });
        }
        let n_prims = prims.len() as u32;
        let bvh_nodes = build_bvh(&mut prims).unwrap_or_default();
        let lights: Vec<GpuLight> = scene.lights.iter().map(GpuLight::from_light_data).collect();
        let firefly_clamp = crate::bdpt::firefly_clamp_bound(scene.max_light_luminance());

        // Storage buffers must be non-empty for WGSL arrayLength to be valid;
        // pad empty families to one zeroed element of the right stride.
        let prim_buf = upload_storage(
            &device,
            &queue,
            bytemuck::cast_slice(&prims),
            std::mem::size_of::<GpuPrim>() as u64,
            "prims",
        );
        let bvh_buf = upload_storage(
            &device,
            &queue,
            bytemuck::cast_slice(&bvh_nodes),
            std::mem::size_of::<GpuBvhNode>() as u64,
            "bvh_nodes",
        );
        let light_buf = upload_storage(
            &device,
            &queue,
            bytemuck::cast_slice(&lights),
            std::mem::size_of::<GpuLight>() as u64,
            "lights",
        );

        let n_pixels = (width * height) as u64;
        let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accum"),
            size: n_pixels * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: n_pixels * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind group: uniform + 3 read-only storage + 1 read-write storage ---
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bdpt-bgl"),
            entries: &[
                uniform_entry,
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_rw(4),
            ],
        });

        let res = StaticResources {
            bgl,
            uniform_buf,
            prim_buf,
            bvh_buf,
            light_buf,
        };
        let bind_group = res.bind_group(&device, &accum_buf);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bdpt-pipeline-layout"),
            bind_group_layouts: &[Some(&res.bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bdpt-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            res,
            bind_group,
            accum_buf,
            readback_buf,
            width,
            height,
            n_lights: scene.lights.len() as u32,
            n_prims,
            firefly_clamp,
            samples: 0,
            reset: true,
            adapter_info,
        }
    }

    /// Human-readable description of the GPU in use.
    pub fn adapter_info(&self) -> &str {
        &self.adapter_info
    }

    /// The device shared with the windowed frontend's surface.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue shared with the windowed frontend's surface.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Cloned handles to the device and queue (both are `Arc`-backed in
    /// wgpu, so cloning is cheap and shares the same underlying GPU
    /// context). Used to build a second renderer — e.g. the FHD export —
    /// without re-creating the adapter and re-uploading the scene.
    pub fn device_clone(&self) -> wgpu::Device {
        self.device.clone()
    }

    pub fn queue_clone(&self) -> wgpu::Queue {
        self.queue.clone()
    }

    /// The accumulation buffer, read by the display pass.
    pub fn accum_buf(&self) -> &wgpu::Buffer {
        &self.accum_buf
    }

    /// Current render resolution.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Samples accumulated in the current epoch (for the status line).
    pub fn samples_accumulated(&self) -> u32 {
        self.samples
    }

    /// Rebuild the accumulation and readback buffers at a new resolution.
    /// The scene buffers and pipeline are reused; only the bind group is
    /// recreated (it references the new accumulation buffer).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width, height) == (self.width, self.height) {
            return;
        }
        let n_pixels = (width * height) as u64;
        self.accum_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accum"),
            size: n_pixels * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: n_pixels * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.bind_group = self.res.bind_group(&self.device, &self.accum_buf);
        self.width = width;
        self.height = height;
        self.reset_accumulation();
    }

    /// Reset the accumulation (camera moved or resolution changed).
    pub fn reset_accumulation(&mut self) {
        self.reset = true;
        self.samples = 0;
    }

    /// Render one frame (two samples per pixel) and accumulate it on the
    /// GPU. Does not block on the GPU; call `read_pixels` when pixels are
    /// needed.
    pub fn render_frame(&mut self, cam: &crate::camera::Cam, max_depth: u32) {
        self.render_frame_spp(cam, max_depth, 1);
    }

    /// Render one frame with `spp` samples per pixel (traced in a single
    /// dispatch) and accumulate it on the GPU. Does not block on the GPU;
    /// call `read_pixels` when pixels are needed.
    pub fn render_frame_spp(&mut self, cam: &crate::camera::Cam, max_depth: u32, spp: u32) {
        let aspect = self.width as f32 / self.height as f32;
        let theta = cam.fov_deg.to_radians();
        let fov_v = (theta * 0.5).tan();
        let fov_h = aspect * fov_v;

        let forward = (cam.lookat - cam.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

        let uniforms = Uniforms {
            cam_origin: vec3s(cam.origin),
            fov_h,
            cam_forward: vec3s(forward),
            fov_v,
            cam_right: vec3s(right),
            width: self.width as f32,
            cam_up: vec3s(up),
            height: self.height as f32,
            max_depth,
            n_lights: self.n_lights,
            samples: self.samples,
            reset: if self.reset { 1 } else { 0 },
            spp: spp.max(1),
            n_prims: self.n_prims,
            firefly_clamp: self.firefly_clamp,
            _pad2: 0,
        };
        self.queue
            .write_buffer(&self.res.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bdpt-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bdpt-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg_x = self.width.div_ceil(8);
            let wg_y = self.height.div_ceil(8);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.samples += spp.max(1);
        self.reset = false;
    }

    /// Copy the accumulation buffer to the CPU and return it as a PixelBuffer.
    /// Blocks until the GPU has finished the pending frame.
    pub fn read_pixels(&mut self) -> PixelBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bdpt-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.accum_buf,
            0,
            &self.readback_buf,
            0,
            self.readback_buf.size(),
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let data = slice
            .get_mapped_range()
            .expect("readback buffer not mapped");
        let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.readback_buf.unmap();

        let pixels: Vec<Vec3> = floats
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect();
        PixelBuffer {
            width: self.width as usize,
            height: self.height as usize,
            pixels,
        }
    }
}

/// Device descriptor shared by the standalone and windowed frontends.
///
/// The BDPT shader binds 4 storage buffers (prims + BVH nodes + lights +
/// accum); downlevel defaults only allow 4. Request the standard WebGPU
/// limits, which allow 8.
pub(crate) fn device_descriptor() -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        label: Some("bdpt-device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }
}

/// Create a STORAGE|COPY_DST buffer and fill it. Empty scenes get one zeroed
/// element so the WGSL `arrayLength` is never 0 and the buffer size is a
/// multiple of the element stride (avoids out-of-bounds reads and validation
/// errors).
fn upload_storage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    contents: &[u8],
    elem_size: u64,
    label: &str,
) -> wgpu::Buffer {
    let size = if contents.is_empty() {
        elem_size
    } else {
        contents.len() as u64
    };
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if contents.is_empty() {
        queue.write_buffer(&buf, 0, &vec![0u8; elem_size as usize]);
    } else {
        queue.write_buffer(&buf, 0, contents);
    }
    buf
}

impl crate::renderer::Renderer for GpuRenderer {
    fn render_frame(&mut self, cam: &crate::camera::Cam, max_depth: usize) {
        self.render_frame(cam, max_depth as u32);
    }

    fn read_pixels(&mut self) -> PixelBuffer {
        self.read_pixels()
    }

    fn reset_accumulation(&mut self) {
        self.reset_accumulation();
    }

    /// Rebuild the accumulation/readback buffers at a new resolution,
    /// reusing the device, pipeline and scene buffers. Creating a whole
    /// new `GpuRenderer` per resize would re-upload the scene and rebuild
    /// the pipeline every time.
    fn resize(&mut self, width: u32, height: u32) {
        self.resize(width, height);
    }

    fn name(&self) -> String {
        format!("GPU ({})", self.adapter_info())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::Renderer;

    /// The WGSL shader must parse and validate with the same naga version
    /// wgpu uses. This runs on every machine, GPU or not.
    #[test]
    fn wgsl_shader_validates() {
        let source = include_str!("../shaders/bdpt.wgsl");
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

    /// The flat BVH must be a valid traversal structure:
    /// * every prim appears in exactly one leaf;
    /// * leaf prim ranges are disjoint and cover all prims;
    /// * inner nodes point at children inside the array, with the left
    ///   child at index+1 (the pre-order invariant the shader relies on);
    /// * every node's box contains its subtree's prim boxes.
    #[test]
    fn flat_bvh_is_a_valid_traversal_structure() {
        use crate::scene::MaterialData;
        let mut scene = crate::scene::SceneData::new();
        // A spread of all four prim kinds, some overlapping.
        for i in 0..8 {
            let x = i as f32 * 0.7;
            scene.add_sphere(
                crate::math::Vec3::new(x, 0.4, -0.8),
                0.4,
                MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
        }
        scene.add_wall(
            crate::math::Vec3::new(0.0, 0.0, -2.5),
            crate::math::Vec3::new(0.0, 0.0, 1.0),
            5.0,
            2.0,
            MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
        );
        scene.add_cube(
            crate::math::Vec3::new(1.0, 0.3, 0.8),
            0.6,
            MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
        );
        scene.add_plane(
            crate::math::Vec3::new(0.0, 0.0, 0.0),
            crate::math::Vec3::new(0.0, 1.0, 0.0),
            MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
        );

        let mut prims: Vec<GpuPrim> = Vec::new();
        for s in &scene.spheres {
            prims.push(GpuPrim {
                kind: KIND_SPHERE,
                ior: 1.0,
                mat_type: 0,
                _pad0: 0.0,
                center: vec3s(s.center),
                radius: s.radius,
                normal: [0.0; 3],
                half_h: 0.0,
                tangent: [0.0; 3],
                _pad1: 0.0,
                bitangent: [0.0; 3],
                _pad2: 0.0,
                albedo: [0.0; 3],
                _pad3: 0.0,
                emission: [0.0; 3],
                _pad4: 0.0,
            });
        }
        for q in &scene.quads {
            prims.push(GpuPrim {
                kind: KIND_QUAD,
                ior: 1.0,
                mat_type: 0,
                _pad0: 0.0,
                center: vec3s(q.center),
                radius: q.half_w,
                normal: vec3s(q.normal),
                half_h: q.half_h,
                tangent: vec3s(q.tangent),
                _pad1: 0.0,
                bitangent: vec3s(q.bitangent),
                _pad2: 0.0,
                albedo: [0.0; 3],
                _pad3: 0.0,
                emission: [0.0; 3],
                _pad4: 0.0,
            });
        }
        for c in &scene.cubes {
            prims.push(GpuPrim {
                kind: KIND_CUBE,
                ior: 1.0,
                mat_type: 0,
                _pad0: 0.0,
                center: vec3s(c.min),
                radius: 0.0,
                normal: vec3s(c.max),
                half_h: 0.0,
                tangent: [0.0; 3],
                _pad1: 0.0,
                bitangent: [0.0; 3],
                _pad2: 0.0,
                albedo: [0.0; 3],
                _pad3: 0.0,
                emission: [0.0; 3],
                _pad4: 0.0,
            });
        }
        for p in &scene.planes {
            prims.push(GpuPrim {
                kind: KIND_PLANE,
                ior: 1.0,
                mat_type: 0,
                _pad0: 0.0,
                center: vec3s(p.point),
                radius: 0.0,
                normal: vec3s(p.normal),
                half_h: 0.0,
                tangent: [0.0; 3],
                _pad1: 0.0,
                bitangent: [0.0; 3],
                _pad2: 0.0,
                albedo: [0.0; 3],
                _pad3: 0.0,
                emission: [0.0; 3],
                _pad4: 0.0,
            });
        }
        let n = prims.len();
        let nodes = build_bvh(&mut prims).expect("non-empty scene must build a BVH");
        assert!(!nodes.is_empty());

        // Walk the tree, collecting leaf ranges and checking invariants.
        let mut covered = vec![false; n];
        let mut stack = vec![0usize];
        while let Some(idx) = stack.pop() {
            let node = nodes[idx];
            let (nmin, nmax) = (node.min, node.max);
            if node.count > 0 {
                let first = node.right_first as usize;
                let count = node.count as usize;
                assert!(first + count <= n, "leaf range out of bounds");
                for i in 0..count {
                    assert!(!covered[first + i], "prim covered by two leaves");
                    covered[first + i] = true;
                    let (pmin, pmax) = prim_bbox(&prims[first + i]);
                    for a in 0..3 {
                        assert!(pmin[a] >= nmin[a] - 1e-4 && pmax[a] <= nmax[a] + 1e-4);
                    }
                }
            } else {
                let left = idx + 1;
                let right = node.right_first as usize;
                assert!(left < nodes.len(), "left child out of bounds");
                assert!(right < nodes.len(), "right child out of bounds");
                assert!(right > idx, "right child must follow the node");
                stack.push(left);
                stack.push(right);
            }
        }
        assert!(covered.iter().all(|c| *c), "every prim must be in a leaf");
    }

    /// A degenerate case: all prims at the same position. The median
    /// split must still terminate and produce a valid tree (the sort is
    /// stable on equal keys, and each side gets at least one prim).
    #[test]
    fn flat_bvh_handles_degenerate_scenes() {
        let mut prims: Vec<GpuPrim> = Vec::new();
        for _ in 0..16 {
            prims.push(GpuPrim {
                kind: KIND_SPHERE,
                ior: 1.0,
                mat_type: 0,
                _pad0: 0.0,
                center: [0.0, 0.0, 0.0],
                radius: 0.5,
                normal: [0.0; 3],
                half_h: 0.0,
                tangent: [0.0; 3],
                _pad1: 0.0,
                bitangent: [0.0; 3],
                _pad2: 0.0,
                albedo: [0.0; 3],
                _pad3: 0.0,
                emission: [0.0; 3],
                _pad4: 0.0,
            });
        }
        let nodes = build_bvh(&mut prims).expect("non-empty scene must build a BVH");
        // 16 prims, median split: 15 inner nodes + 16 leaves = 31 nodes.
        assert_eq!(nodes.len(), 31);
        // Every leaf holds exactly one prim (the median split of an even
        // count gives 8/8, then 4/4, 2/2, 1/1).
        let leaves: Vec<_> = nodes.iter().filter(|nd| nd.count > 0).collect();
        assert_eq!(leaves.len(), 16);
        assert!(leaves.iter().all(|nd| nd.count == 1));
    }

    /// The empty scene must produce no nodes (the shader skips traversal
    /// when n_prims == 0).
    #[test]
    fn flat_bvh_empty_scene() {
        let mut prims: Vec<GpuPrim> = Vec::new();
        assert!(build_bvh(&mut prims).is_none());
    }

    /// Smoke test: GpuRenderer::new must either succeed or fail with a
    /// clean error (never panic). On machines with any adapter (including
    /// software rasterizers like llvmpipe) it also renders one frame and
    /// checks the pixels are finite.
    #[test]
    fn gpu_renderer_smoke() {
        // Capture any panic message to a file (test harness hides stdout).
        // Written under target/ so the working tree stays clean.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "non-string panic".to_string()
            };
            let loc = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_else(|| "unknown".to_string());
            std::fs::write(
                "target/gpu_smoke_error.txt",
                format!("panic at {loc}: {msg}"),
            )
            .ok();
            default_hook(info);
        }));
        // Clear stale error files so we only see fresh failures.
        let _ = std::fs::remove_file("target/gpu_smoke_error.txt");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu_renderer_smoke_inner();
        }));
        let _ = std::panic::take_hook();
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    fn gpu_renderer_smoke_inner() {
        let mut scene = crate::scene::SceneData::new();
        scene.add_plane(
            crate::math::Vec3::new(0.0, 0.0, 0.0),
            crate::math::Vec3::new(0.0, 1.0, 0.0),
            crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
        );
        // `add_light` registers the sphere as geometry too, so the GPU's
        // intersect_scene sees it (it never reads the light buffer).
        scene.add_light(crate::scene::LightData::from_sphere(
            crate::math::Vec3::new(0.0, 1.0, 0.0),
            0.5,
            crate::math::Vec3::new(5.0, 5.0, 5.0),
        ));

        let mut gpu = match super::GpuRenderer::new(&scene, 32, 32) {
            Ok(g) => g,
            Err(e) => {
                // No adapter at all: acceptable, the CPU fallback handles it.
                eprintln!("no GPU adapter in test env: {e}");
                return;
            }
        };
        let cam = crate::camera::Cam::new(
            crate::math::Vec3::new(0.0, 0.5, 2.0),
            crate::math::Vec3::new(0.0, 0.5, 0.0),
            90.0,
        );
        gpu.render_frame(&cam, 8);
        let buf = gpu.read_pixels();
        assert_eq!(buf.pixels.len(), 32 * 32);
        for p in &buf.pixels {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
        // The emissive sphere is in view; the image must not be black.
        let sum: f32 = buf.pixels.iter().map(|p| p.x).sum();
        assert!(sum > 0.0, "GPU render is black");
    }

    /// GPU-vs-CPU oracle: render the same scene on both backends and
    /// compare per-channel means. RNG streams differ, so images will not
    /// match sample-for-sample; the means must agree within a tolerance
    /// that catches systematic porting errors (wrong epsilon, missing MIS
    /// strategy, sign errors) while tolerating Monte-Carlo noise.
    ///
    /// Ignored by default: it needs a real GPU. Run with
    /// `cargo test -- --ignored gpu_matches_cpu`.
    #[test]
    #[ignore]
    fn gpu_matches_cpu() {
        // Capture any panic message to a file (test harness hides stdout).
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "non-string panic".to_string()
            };
            let loc = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_else(|| "unknown".to_string());
            std::fs::write(
                "target/gpu_oracle_error.txt",
                format!("panic at {loc}: {msg}"),
            )
            .ok();
            default_hook(info);
        }));
        // Clear stale error files so we only see fresh failures.
        let _ = std::fs::remove_file("target/gpu_oracle_error.txt");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu_matches_cpu_inner();
        }));
        let _ = std::panic::take_hook();
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    fn gpu_matches_cpu_inner() {
        let scene = crate::scene::SceneData::new();
        let scene = {
            // Reuse the demo scene builder from main.rs.
            let mut s = scene;
            s.add_plane(
                crate::math::Vec3::new(0.0, 0.0, 0.0),
                crate::math::Vec3::new(0.0, 1.0, 0.0),
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_plane(
                crate::math::Vec3::new(0.0, 2.0, 0.0),
                crate::math::Vec3::new(0.0, -1.0, 0.0),
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_plane(
                crate::math::Vec3::new(0.0, 2.0, -2.5),
                crate::math::Vec3::new(0.0, 0.0, 1.0),
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_plane(
                crate::math::Vec3::new(-2.5, 1.0, 0.0),
                crate::math::Vec3::new(1.0, 0.0, 0.0),
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.75, 0.25, 0.25)),
            );
            s.add_plane(
                crate::math::Vec3::new(2.5, 1.0, 0.0),
                crate::math::Vec3::new(-1.0, 0.0, 0.0),
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.25, 0.75, 0.25)),
            );
            s.add_plane(
                crate::math::Vec3::new(0.0, 2.0, 2.5),
                crate::math::Vec3::new(0.0, 0.0, -1.0),
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_sphere(
                crate::math::Vec3::new(-0.7, 0.4, -0.8),
                0.4,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_sphere(
                crate::math::Vec3::new(-0.3, 0.4, -0.2),
                0.4,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_cube(
                crate::math::Vec3::new(1.0, 0.3, -0.8),
                0.6,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            // Mirror sphere (was diffuse): the oracle must exercise the
            // Phase 1 BSDF paths (delta sampling, the symbolic-pdf MIS
            // rule), not just the diffuse reduction.
            s.add_sphere(
                crate::math::Vec3::new(1.0, 1.0, -0.8),
                0.4,
                crate::scene::MaterialData::mirror(crate::math::Vec3::new(0.9, 0.9, 0.9)),
            );
            s.add_cube(
                crate::math::Vec3::new(0.3, 0.3, 0.8),
                0.6,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_cube(
                crate::math::Vec3::new(0.94, 0.3, 0.8),
                0.6,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_cube(
                crate::math::Vec3::new(1.58, 0.3, 0.8),
                0.6,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_cube(
                crate::math::Vec3::new(0.45, 0.9, 0.8),
                0.6,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            s.add_cube(
                crate::math::Vec3::new(1.09, 0.9, 0.8),
                0.6,
                crate::scene::MaterialData::diffuse(crate::math::Vec3::new(0.8, 0.8, 0.8)),
            );
            // Glass sphere under the light: the oracle must exercise the
            // Phase 1 glass paths (Fresnel split, Snell refraction, the
            // eta^2 transport factor, caustic connections), not just the
            // diffuse reduction.
            s.add_sphere(
                crate::math::Vec3::new(-0.4, 0.42, -0.8),
                0.4,
                crate::scene::MaterialData::glass(1.5),
            );
            // `add_light` registers the emitter as geometry too.
            s.add_light(crate::scene::LightData::from_sphere(
                crate::math::Vec3::new(-0.4, 1.2, -0.8),
                0.2,
                crate::math::Vec3::new(10.0, 10.0, 4.0),
            ));
            s
        };

        const W: u32 = 32;
        const H: u32 = 32;
        // Glass caustics are high-variance; 256 frames (512 spp on the
        // GPU, 256 on the CPU) keep the channel means stable enough for
        // the 15% tolerance to isolate systematic porting errors from
        // Monte-Carlo noise.
        const FRAMES: u32 = 256;

        // GPU: accumulate FRAMES frames, read back once.
        let mut gpu = super::GpuRenderer::new(&scene, W, H).expect("GPU required for this test");
        let cam = crate::camera::Cam::new(
            crate::math::Vec3::new(0.0, 1.0, -1.5),
            crate::math::Vec3::new(0.0, 0.5, 0.0),
            90.0,
        );
        for _ in 0..FRAMES {
            gpu.render_frame(&cam, 8);
        }
        let gpu_buf = gpu.read_pixels();

        // CPU: same number of samples.
        let mut cpu =
            crate::renderer::CpuRenderer::new(&scene, W as usize, H as usize, FRAMES as usize);
        cpu.render_frame(&cam, 8);
        let cpu_buf = cpu.read_pixels();

        // Compare per-channel means and standard deviations.
        let mean = |buf: &crate::output::PixelBuffer, ch: usize| -> f32 {
            buf.pixels
                .iter()
                .map(|p| match ch {
                    0 => p.x,
                    1 => p.y,
                    _ => p.z,
                })
                .sum::<f32>()
                / buf.pixels.len() as f32
        };
        for ch in 0..3 {
            let m_gpu = mean(&gpu_buf, ch);
            let m_cpu = mean(&cpu_buf, ch);
            // Tolerance: 15% relative or 0.02 absolute, whichever is larger.
            let tol = (m_cpu * 0.15).max(0.02);
            assert!(
                (m_gpu - m_cpu).abs() <= tol,
                "channel {ch}: GPU mean {m_gpu} vs CPU mean {m_cpu} (tol {tol})"
            );
        }

        // Every pixel must be finite on the GPU side.
        for p in &gpu_buf.pixels {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
    }
}
