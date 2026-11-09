// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the shader's purpose statement, the shared structs, consts,
// bindings, the RNG, and the vector helpers.

// Bidirectional path tracing with MIS, ported 1:1 from src/bdpt.rs.
// Intersection runs on a flat BVH built on the CPU (src/gpu.rs) and
// traversed here with an explicit stack — see the Intersection section.
//
// Surfaces carry a general BSDF (diffuse, mirror, glass). Subpaths sample
// directions with the BSDF's own distribution, and the MIS weights are
// computed with Veach's incremental-ratio walk over the *actual* pdfs of
// every edge — computed on the fly from the path geometry through the
// PathView helpers, which address the unified path positions x_1..x_k
// (the eye prefix followed by the light vertices in reverse). Weights use
// the power heuristic (beta = 2).
//
// Delta BSDFs (mirror, glass) scatter into a single direction: their
// solid-angle pdf is a Dirac delta, which has no numeric value. The
// convention (shared with bdpt.rs) is that a delta-sampled edge
// contributes a *symbolic* factor of 1 to the pdf products — every valid
// strategy's pdf carries exactly one such factor per delta vertex on the
// path, so they cancel in the ratios between strategies. A delta vertex
// scattering in any *other* direction (a hypothetical extension that
// does not follow the specular pair) has pdf 0, which kills every
// strategy that would need it. Connections through delta vertices are
// invalid (their eval is 0) and are excluded from the MIS normalization;
// specular transport is handled purely by subpath sampling, which is
// the whole BDPT story for caustics.
//
// Paths are terminated by Russian roulette rather than a short fixed
// depth: from extension RR_START on, a subpath survives each bounce with
// probability q = clamp(max(throughput), RR_MIN, 1) and divides its
// throughput by q on survival, which keeps the estimator unbiased while
// letting deep paths through. The survival probabilities are part of the
// sampling pdfs, so they also enter the MIS weights, as the prefix
// products computed by rr_prefix_products — whose sweep recomputes each
// extension's sample weight from the path geometry, so the hypothetical
// rolls of strategies that would extend a subpath past where it stopped
// are exactly reproducible.

struct Uniforms {
    cam_origin: vec3<f32>,
    fov_h: f32,          // tan(fov/2) * aspect
    cam_forward: vec3<f32>,
    fov_v: f32,          // tan(fov/2)
    cam_right: vec3<f32>,
    width: f32,
    cam_up: vec3<f32>,
    height: f32,
    max_depth: u32,
    n_lights: u32,
    samples: u32,        // samples already accumulated in `accum`
    reset: u32,
    spp: u32,            // samples traced per pixel in this dispatch
    n_prims: u32,        // scene prims; 0 = empty scene, skip traversal
    firefly_clamp: f32,  // per-sample clamp *multiplier* (the bound is spectral, per-λ)
    debug: u32,          // DEBUG (temporary): 0 = off, 1 = dump bridge diagnostics
};

// One scene primitive, whichever kind — the GPU shape of the scene's
// spheres/quads/cubes/planes/cylinders, kind-tagged so a single array
// (and a single BVH leaf range) can hold them all. The families share
// slots: the cube's min/max live in center/normal, the quad's half_w
// lives in radius, the cylinder's base lives in center, its axis in
// tangent, its radius in radius and its half-height in half_h.
// mat_type is 0 diffuse / 1 emissive / 2 mirror / 3 glass / 4 laser /
// 5 glossy; ior carries the glass interior index (1.0 otherwise), or
// the laser emission cone's *versine* (1 - cos theta, computed in f64
// on the CPU — at a laser's divergence the f32 cosine rounds to 1.0
// and the cone would collapse; see bdpt.rs::Bsdf::Laser) for
// mat_type 4. The spectral fields
// (paper §9.2): sellmeier carries the glass dispersion coefficients
// (all zeros = achromatic, the constant ior); rough/f0 carry the GGX
// lobe for mat_type 5 (§9.5).
//
// Layout rule: every vec3<f32> is immediately followed by an f32/u32 so
// WGSL's implicit 16-byte alignment never inserts padding that Rust's
// #[repr(C)] doesn't have. Stride 128, matching GpuPrim in gpu.rs.
struct Prim {
    kind: u32,              // KIND_SPHERE / KIND_QUAD / KIND_CUBE / KIND_PLANE / KIND_CYLINDER
    ior: f32,
    mat_type: u32,
    _pad0: f32,
    center: vec3<f32>,      // sphere/quad center, plane point, cube min, cylinder base
    radius: f32,            // sphere radius, quad half_w, cylinder radius
    normal: vec3<f32>,      // quad/plane normal, cube max
    half_h: f32,            // quad half_h, cylinder half-height
    tangent: vec3<f32>,     // quad tangent, cylinder axis
    _pad1: f32,
    bitangent: vec3<f32>,   // quad bitangent
    _pad2: f32,
    albedo: vec3<f32>,
    _pad3: f32,
    sellmeier: array<f32, 6>,  // glass dispersion [B1,B2,B3,C1,C2,C3]; zeros = achromatic
    rough: f32,             // GGX roughness (mat_type 5)
    f0: f32,                // GGX normal-incidence reflectance (mat_type 5)
};

// One flat BVH node, 32 bytes (matching GpuBvhNode in gpu.rs).
//
// * Leaf (count > 0): right_first indexes the first prim of a contiguous
//   run of `count` prims — the prim array is uploaded in leaf order, so a
//   leaf needs no per-prim indirection.
// * Inner (count == 0): right_first is the right child's node index; the
//   left child is always this node's index + 1, because the builder emits
//   nodes in pre-order (the same layout trick as pbrt's LinearBVHNode),
//   so no second child field is needed.
struct BvhNode {
    min: vec3<f32>,
    right_first: u32,
    max: vec3<f32>,
    count: u32,
};

struct GpuLight {
    shape: u32,
    spectrum_kind: u32,   // 0 = smooth (CMF-basis coeffs), 1 = monochromatic line
    _pad1: f32,
    _pad2: f32,
    center: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    _pad3: f32,
    tangent: vec3<f32>,
    _pad4: f32,
    bitangent: vec3<f32>,
    half_w: f32,
    half_h: f32,
    // The smooth spectrum's CMF-basis coefficients (a line ignores
    // them): S(λ) = coeffs·CMF(λ), the unique SPD whose tristimulus
    // integral is the RGB intent (paper §9.2). Three *scalars*, not a
    // vec3: WGSL aligns vec3 to 16 bytes, which would misalign
    // bridge_rungs (16-byte vec4 elements) against the Rust layout.
    coeff_x: f32,
    coeff_y: f32,
    coeff_z: f32,
    // The line's wavelength (nm) and spectral radiance; 0 for a smooth
    // spectrum. A laser is monochromatic, so λ is deterministic on its
    // branch.
    lambda: f32,
    power: f32,
    _pad8: f32,
    _pad8b: f32,
    // The specular bridge *ladder* (a laser only), packed one rung per
    // *pair* of vec4s: [xyz = the rung's image-center, w = its radius]
    // then [xyz = the rung's beam direction (the unfolded beam axis,
    // pointing away from the image toward the surface), w = the
    // emission cone's versine]. Rung j is the aperture's mirror image
    // unfolded across the first j mirrors of the beam's chain (see
    // bdpt.rs::Bridge); the eye walk aims at these at diffuse vertices,
    // so the glow appears beside *every* leg of the beam, not just the
    // last. The beam direction and versine let the rung sample the
    // *gate-mapped* cone — the smaller of the image-disc cone and the
    // emission cone — so at a laser's physical divergence the spot's
    // draws are not wasted on gate-rejected aims. A zero beam_dir
    // marks a rung without a beam direction (the disc cone alone).
    // n_bridges = 0 marks "no ladder".
    bridge_rungs: array<vec4<f32>, 16>,
    n_bridges: u32,
    // Three scalar pads, not a vec3: WGSL aligns vec3 to 16 bytes, which
    // would push the struct past the Rust side's [f32; 3]. Scalars are
    // 4-byte aligned in both languages.
    _pad9: f32,
    _pad10: f32,
    _pad11: f32,
};

struct Hit {
    t: f32,
    p: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    mat_type: u32,
    ior: f32,
    // The glass Sellmeier coefficients (zeros = achromatic) and the GGX
    // lobe parameters, carried so the BSDF helpers can evaluate n(λ) and
    // the microfacet lobe without re-reading the prim.
    sellmeier: array<f32, 6>,
    rough: f32,
    f0: f32,
};

// One vertex of an eye or light subpath. `normal` is oriented toward the
// side the subpath arrived from; `front_face` records whether the
// geometric normal (as returned by the intersection) faces that side, so
// hypothetical events can re-orient it for their own arrival side.
// `throughput` excludes the BSDF applied when *leaving* the vertex (that
// factor belongs to the connection term) and is a *spectral* scalar at
// the path's hero wavelength (paper §9.2).
struct PathVertex {
    p: vec3<f32>,
    normal: vec3<f32>,
    front_face: bool,
    albedo: vec3<f32>,
    mat_type: u32,
    ior: f32,
    sellmeier: array<f32, 6>,
    rough: f32,
    f0: f32,
    throughput: f32,
    on_light: bool,
};

// A view of one full path x_1..x_k (camera at position 0, the light y_0
// at position k) as the eye prefix followed by the light vertices in
// reverse: x_m = eye[m-1] for m <= n_eye, x_m = light_path[k-m] above.
// Every pdf and sample weight the MIS machinery needs — realized edges,
// the connection edge, and the hypothetical extensions of strategies
// that would push a subpath past where it stopped — is computed from
// the path geometry through this view, so realized and hypothetical
// quantities are produced by the *same* code and cannot drift apart.
// (WGSL structs cannot hold pointers, so the subpath arrays are passed
// alongside the view to the pv_* helpers.)
struct PathView {
    k: u32,
    n_eye: u32,
    // n_active * area of the light the path ends on (the *active*
    // lights: those that radiate at the drawn λ — see the branch rule).
    inv_light_pdf: f32,
    // The primary camera ray direction (the incoming direction at x_1).
    cam_dir: vec3<f32>,
    // The path's hero wavelength (nm). Fixed per path, so realized and
    // hypothetical events evaluate the same spectral quantities.
    lambda: f32,
};

const PI: f32 = 3.141592653589793;
const EPS_NEAR: f32 = 0.001;
const MAX_PATH_VERTICES: u32 = 8u;
const BACKGROUND: vec3<f32> = vec3<f32>(0.02, 0.02, 0.05);
// Russian roulette: minimum survival probability and the extension number
// the roulette starts at (the first RR_START - 1 extensions are
// unconditional). Must match src/bdpt.rs.
const RR_MIN: f32 = 0.05;
const RR_START: u32 = 3u;
// Cosine threshold for recognizing a specular direction pair (1 - 1e-4).
const SPEC_DOT_EPS: f32 = 1.0 - 1e-4;
// Material type encoding, shared with gpu.rs.
const MAT_DIFFUSE: u32 = 0u;
const MAT_EMISSIVE: u32 = 1u;
const MAT_MIRROR: u32 = 2u;
const MAT_GLASS: u32 = 3u;
// A laser aperture: emissive, radiating only into the cone whose
// *versine* (1 - cos theta, computed in f64 on the CPU) rides in the
// prim's `ior` slot — at a laser's divergence the f32 cosine rounds to
// 1.0 and the cone would collapse (see bdpt.rs::Bsdf::Laser).
const MAT_LASER: u32 = 4u;
// A GGX microfacet "rough mirror" (paper §9.5): a continuous lobe, so
// connections through it are valid. The roughness rides in `rough`, the
// normal-incidence reflectance in `f0`.
const MAT_GLOSSY: u32 = 5u;
// Which quantity a subpath transports: the two agree for reflection but
// differ for transmission (radiance picks up (eta_t/eta_i)^2 crossing a
// refractive boundary, importance does not). The eye subpath carries
// radiance, the light subpath importance.
const TRANSPORT_RADIANCE: u32 = 0u;
const TRANSPORT_IMPORTANCE: u32 = 1u;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> prims: array<Prim>;
@group(0) @binding(2) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(3) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4<f32>>;

// --- RNG: PCG32 (stateless-seedable, standard for GPU path tracing) ---

fn pcg(state: ptr<function, u32>) -> f32 {
    *state = *state * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return f32((word >> 22u) ^ word) * (1.0 / 4294967296.0);
}

// --- Vector helpers ---

fn vlen(v: vec3<f32>) -> f32 { return sqrt(dot(v, v)); }

fn normalize3(v: vec3<f32>) -> vec3<f32> {
    let l = vlen(v);
    if (l == 0.0) { return v; }
    return v / l;
}
