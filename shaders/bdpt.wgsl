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
    firefly_clamp: f32,  // per-sample luminance clamp bound (0 = disable)
    _pad2: u32,
};

// One scene primitive, whichever kind — the GPU shape of the scene's
// spheres/quads/cubes/planes/cylinders, kind-tagged so a single array
// (and a single BVH leaf range) can hold them all. The families share
// slots: the cube's min/max live in center/normal, the quad's half_w
// lives in radius, the cylinder's base lives in center, its axis in
// tangent, its radius in radius and its half-height in half_h.
// mat_type is 0 diffuse / 1 emissive / 2 mirror / 3 glass / 4 laser;
// ior carries the glass interior index (1.0 otherwise), or the laser
// emission cone's cosine for mat_type 4.
//
// Layout rule: every vec3<f32> is immediately followed by an f32/u32 so
// WGSL's implicit 16-byte alignment never inserts padding that Rust's
// #[repr(C)] doesn't have. Stride 112, matching GpuPrim in gpu.rs.
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
    emission: vec3<f32>,
    _pad4: f32,
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
    _pad0: f32,
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
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    emission: vec3<f32>,
    _pad8: f32,
};

struct Hit {
    t: f32,
    p: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    emission: vec3<f32>,
    mat_type: u32,
    ior: f32,
};

// One vertex of an eye or light subpath. `normal` is oriented toward the
// side the subpath arrived from; `front_face` records whether the
// geometric normal (as returned by the intersection) faces that side, so
// hypothetical events can re-orient it for their own arrival side.
// `throughput` excludes the BSDF applied when *leaving* the vertex (that
// factor belongs to the connection term).
struct PathVertex {
    p: vec3<f32>,
    normal: vec3<f32>,
    front_face: bool,
    albedo: vec3<f32>,
    emission: vec3<f32>,
    mat_type: u32,
    ior: f32,
    throughput: vec3<f32>,
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
    // n_lights * area of the light the path ends on.
    inv_light_pdf: f32,
    // The primary camera ray direction (the incoming direction at x_1).
    cam_dir: vec3<f32>,
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
// A laser aperture: emissive, radiating only into the cone whose cosine
// rides in the prim's `ior` slot.
const MAT_LASER: u32 = 4u;
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

// --- Intersection: flat BVH traversal ---
//
// The scene is uploaded as a kind-tagged prim array plus a flat BVH
// (built on the CPU, src/gpu.rs). Traversal keeps an explicit stack of
// pending nodes (the GPU analogue of the CPU BVH's recursion) and
// culls every box by the current best hit distance — the analogue of
// the CPU's subtree early-exit test. The builder caps the tree depth
// at 24, and a pop-based traversal of a depth-d binary tree holds at
// most d+1 pending nodes, so the 32-entry stack cannot overflow (the
// bounds guards make it impossible even for a degenerate tree).

const KIND_SPHERE: u32 = 0u;
const KIND_QUAD: u32 = 1u;
const KIND_CUBE: u32 = 2u;
const KIND_PLANE: u32 = 3u;
const KIND_CYLINDER: u32 = 4u;
// The builder caps the tree depth at MAX_BVH_DEPTH; a pop-based
// traversal of a depth-d binary tree holds at most d+1 pending nodes,
// so a 32-entry stack can never overflow (the guards below make it
// impossible even for a degenerate tree).
const MAX_BVH_STACK: u32 = 32u;

// Slab test of one AABB against the ray segment [0, t_max). Returns
// false when the box is behind the ray or beyond the current best hit.
// Axis-parallel rays are handled explicitly (checking the direction,
// never the reciprocal — 1/0 is inf and 0*inf is NaN) so a parallel
// slab can never produce a NaN that silently drops valid hits. This is
// the WGSL shape of math.rs's BBox::intersect.
fn aabb_hit(bmin: vec3<f32>, bmax: vec3<f32>, origin: vec3<f32>,
            dir: vec3<f32>, inv_dir: vec3<f32>, t_max: f32) -> bool {
    var t_min = 0.0;
    var t_max_slab = t_max;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let d = dir[i];
        if (d == 0.0) {
            // Parallel to this slab: the origin must lie inside it.
            if (origin[i] < bmin[i] || origin[i] > bmax[i]) { return false; }
            continue;
        }
        let inv = inv_dir[i];
        var t0 = (bmin[i] - origin[i]) * inv;
        var t1 = (bmax[i] - origin[i]) * inv;
        if (t0 > t1) {
            let tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_min = max(t_min, t0);
        t_max_slab = min(t_max_slab, t1);
        if (t_min > t_max_slab) { return false; }
    }
    return true;
}

fn intersect_prim(p: Prim, origin: vec3<f32>, dir: vec3<f32>, best: ptr<function, Hit>) {
    if (p.kind == KIND_SPHERE) {
        let oc = origin - p.center;
        let a = dot(dir, dir);
        let b = 2.0 * dot(oc, dir);
        let c = dot(oc, oc) - p.radius * p.radius;
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) { return; }
        let sqrt_d = sqrt(disc);
        var t = (-b - sqrt_d) / (2.0 * a);
        if (t < EPS_NEAR) { t = (-b + sqrt_d) / (2.0 * a); }
        if (t < EPS_NEAR || t >= (*best).t) { return; }
        let hp = origin + dir * t;
        (*best).t = t;
        (*best).p = hp;
        (*best).normal = normalize3(hp - p.center);
        (*best).albedo = p.albedo;
        (*best).emission = p.emission;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    if (p.kind == KIND_QUAD) {
        let denom = dot(p.normal, dir);
        if (abs(denom) < 1e-6) { return; }
        let t = dot(p.center - origin, p.normal) / denom;
        if (t < EPS_NEAR || t >= (*best).t) { return; }
        let hp = origin + dir * t;
        let local = hp - p.center;
        if (abs(dot(local, p.tangent)) > p.radius || abs(dot(local, p.bitangent)) > p.half_h) { return; }
        (*best).t = t;
        (*best).p = hp;
        (*best).normal = p.normal;
        (*best).albedo = p.albedo;
        (*best).emission = p.emission;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    if (p.kind == KIND_CUBE) {
        var t_min = -3.4e38;
        var t_max = 3.4e38;
        // Slab test per axis; unrolled to avoid indexing overhead.
        for (var i = 0u; i < 3u; i = i + 1u) {
            let d = dir[i];
            let o = origin[i];
            let lo = p.center[i]; // cube min
            let hi = p.normal[i]; // cube max
            if (d == 0.0) {
                if (o < lo || o > hi) { return; }
                continue;
            }
            let inv_d = 1.0 / d;
            var t0 = (lo - o) * inv_d;
            var t1 = (hi - o) * inv_d;
            if (t0 > t1) {
                let tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            t_min = max(t_min, t0);
            t_max = min(t_max, t1);
        }
        if (!(t_max >= t_min && t_max > 0.0 && t_min < 1e6)) { return; }
        var t = t_min;
        if (t < EPS_NEAR) { t = t_max; }
        if (t < EPS_NEAR || t >= (*best).t) { return; }
        let hp = origin + dir * t;
        // Outward normal of the face closest to the hit point (same rule as
        // geometry.rs so top/+x/+z hits shade identically).
        let d0 = abs(hp.x - p.center.x); let d1 = abs(hp.x - p.normal.x);
        let d2 = abs(hp.y - p.center.y); let d3 = abs(hp.y - p.normal.y);
        let d4 = abs(hp.z - p.center.z); let d5 = abs(hp.z - p.normal.z);
        var best_d = d0;
        var normal = vec3<f32>(-1.0, 0.0, 0.0);
        if (d1 < best_d) { best_d = d1; normal = vec3<f32>(1.0, 0.0, 0.0); }
        if (d2 < best_d) { best_d = d2; normal = vec3<f32>(0.0, -1.0, 0.0); }
        if (d3 < best_d) { best_d = d3; normal = vec3<f32>(0.0, 1.0, 0.0); }
        if (d4 < best_d) { best_d = d4; normal = vec3<f32>(0.0, 0.0, -1.0); }
        if (d5 < best_d) { best_d = d5; normal = vec3<f32>(0.0, 0.0, 1.0); }
        (*best).t = t;
        (*best).p = hp;
        (*best).normal = normal;
        (*best).albedo = p.albedo;
        (*best).emission = p.emission;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    if (p.kind == KIND_CYLINDER) {
        // Closed cylinder: base cap at `center`, axis `tangent`,
        // half-height `half_h`, radius `radius`. Same decomposition as
        // geometry.rs: split the ray into axial and radial parts; the
        // side test is a 2D circle intersection in the radial plane.
        let a = p.tangent;
        let oc = origin - p.center;
        let d_perp = dir - a * dot(dir, a);
        let oc_perp = oc - a * dot(oc, a);
        let dz = dot(dir, a);
        let oz = dot(oc, a);

        var best_t = 3.4e38;
        var best_normal = vec3<f32>(0.0, 1.0, 0.0);

        // Curved side: |oc_perp + t * d_perp| = radius. The axial
        // coordinate z is measured from the base cap, so the cylinder
        // spans z in [0, 2 * half_h].
        let aa = dot(d_perp, d_perp);
        if (aa > 1e-12) {
            let bb = 2.0 * dot(oc_perp, d_perp);
            let cc = dot(oc_perp, oc_perp) - p.radius * p.radius;
            let disc = bb * bb - 4.0 * aa * cc;
            if (disc >= 0.0) {
                let sqrt_d = sqrt(disc);
                for (var k = 0u; k < 2u; k = k + 1u) {
                    var t = (-bb - sqrt_d) / (2.0 * aa);
                    if (k == 1u) { t = (-bb + sqrt_d) / (2.0 * aa); }
                    if (t < EPS_NEAR || t >= best_t) { continue; }
                    let z = oz + t * dz;
                    if (z < 0.0 || z > 2.0 * p.half_h) { continue; }
                    let hp = origin + dir * t;
                    best_t = t;
                    best_normal = normalize3(hp - p.center - a * z);
                }
            }
        }

        // Caps: two one-sided discs at z = 0 (base) and z = 2*half_h (top).
        for (var k = 0u; k < 2u; k = k + 1u) {
            let cap_z = select(0.0, 2.0 * p.half_h, k == 1u);
            let cap_n = select(a * -1.0, a, k == 1u);
            let cap_point = p.center + a * cap_z;
            let denom = dot(cap_n, dir);
            if (abs(denom) < 1e-6) { continue; }
            let t = dot(cap_point - origin, cap_n) / denom;
            if (t < EPS_NEAR || t >= best_t) { continue; }
            let hp = origin + dir * t;
            let radial = hp - cap_point;
            if (dot(radial, radial) > p.radius * p.radius) { continue; }
            best_t = t;
            best_normal = cap_n;
        }

        if (best_t >= (*best).t || best_t >= 3.4e38) { return; }
        let hp = origin + dir * best_t;
        (*best).t = best_t;
        (*best).p = hp;
        (*best).normal = best_normal;
        (*best).albedo = p.albedo;
        (*best).emission = p.emission;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    // KIND_PLANE
    let denom = dot(p.normal, dir);
    if (abs(denom) < 1e-6) { return; }
    let t = dot(p.center - origin, p.normal) / denom;
    if (t < EPS_NEAR || t >= (*best).t) { return; }
    (*best).t = t;
    (*best).p = origin + dir * t;
    (*best).normal = p.normal;
    (*best).albedo = p.albedo;
    (*best).emission = p.emission;
    (*best).mat_type = p.mat_type;
    (*best).ior = p.ior;
}

fn intersect_scene(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    var best: Hit;
    best.t = 1e30;
    best.p = vec3<f32>(0.0);
    best.normal = vec3<f32>(0.0, 1.0, 0.0);
    best.albedo = vec3<f32>(0.0);
    best.emission = vec3<f32>(0.0);
    best.mat_type = 0u;
    best.ior = 1.0;
    if (uniforms.n_prims == 0u) { return best; }

    // Precompute the reciprocals once per ray; the per-node slab tests
    // reuse them. A zero direction component would be a division by
    // zero, so those lanes keep a dummy 1.0 — aabb_hit tests the raw
    // direction for zero first and never consumes the dummy.
    var inv_dir = vec3<f32>(1.0);
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (dir[i] != 0.0) {
            inv_dir[i] = 1.0 / dir[i];
        }
    }
    var stack: array<u32, MAX_BVH_STACK>;
    var sp = 0u;
    stack[0] = 0u; // root
    sp = 1u;
    while (sp > 0u) {
        sp = sp - 1u;
        let cur = stack[sp];
        let node = bvh_nodes[cur];
        if (!aabb_hit(node.min, node.max, origin, dir, inv_dir, best.t)) { continue; }
        if (node.count > 0u) {
            // Leaf: a contiguous run of prims, tested in order.
            for (var i = 0u; i < node.count; i = i + 1u) {
                intersect_prim(prims[node.right_first + i], origin, dir, &best);
            }
            continue;
        }
        // Inner: left child is cur+1 (pre-order layout), right child is
        // right_first. Push both; the pop order visits the left subtree
        // first, which tends to find the near hit early and lets the
        // best.t culling skip the right subtree's boxes.
        if (sp < MAX_BVH_STACK) {
            stack[sp] = node.right_first;
            sp = sp + 1u;
        }
        if (sp < MAX_BVH_STACK) {
            stack[sp] = cur + 1u;
            sp = sp + 1u;
        }
    }
    return best;
}

fn has_hit(h: Hit) -> bool { return h.t < 1e29; }

// --- Sampling ---

fn cosine_hemisphere(normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let u1 = pcg(state);
    let u2 = pcg(state);
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let z = sqrt(1.0 - u1);
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
    let t = normalize3(cross(a, normal));
    let b = cross(normal, t);
    return t * (r * cos(phi)) + b * (r * sin(phi)) + normal * z;
}

fn g_term(p0: vec3<f32>, n0: vec3<f32>, p1: vec3<f32>, n1: vec3<f32>) -> f32 {
    let d = p1 - p0;
    let dist_sq = dot(d, d);
    if (dist_sq < 1e-8) { return 0.0; }
    let dist = sqrt(dist_sq);
    let dir = d / dist;
    let c0 = dot(n0, dir);
    let c1 = dot(n1, -dir);
    if (c0 <= 0.0 || c1 <= 0.0) { return 0.0; }
    return c0 * c1 / dist_sq;
}

fn visible(p_from: vec3<f32>, from_normal: vec3<f32>, to: vec3<f32>) -> bool {
    let origin = p_from + from_normal * EPS_NEAR;
    let d = to - origin;
    let dist = vlen(d);
    if (dist < EPS_NEAR) { return true; }
    let hit = intersect_scene(origin, d / dist);
    // Occluded only if something hits clearly *before* the target.
    return !has_hit(hit) || hit.t + 2.0 * EPS_NEAR >= dist;
}

// --- BSDFs (explicit if chains on mat_type — no dynamic dispatch) ---

// True for BSDFs that scatter into a single direction (a delta
// distribution): they have no continuous density, so connections
// through them are invalid and their pdfs cancel in MIS ratios.
// Emissive surfaces are never delta (they do not scatter at all).
fn is_delta_type(mat_type: u32) -> bool {
    return mat_type == MAT_MIRROR || mat_type == MAT_GLASS;
}

// Exact unpolarized dielectric Fresnel reflectance for an incidence
// cosine `cos_i` (measured against the normal on the incident side).
fn fresnel_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let ci = min(abs(cos_i), 1.0);
    let sin_i = sqrt(1.0 - ci * ci);
    let sin_t = eta_i / eta_t * sin_i;
    if (sin_t >= 1.0) {
        return 1.0; // total internal reflection
    }
    let cos_t = sqrt(1.0 - sin_t * sin_t);
    let r_s = (eta_i * ci - eta_t * cos_t) / (eta_i * ci + eta_t * cos_t);
    let r_p = (eta_t * ci - eta_i * cos_t) / (eta_t * ci + eta_i * cos_t);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

// Specular reflection of the travel direction `dir_in` (unit, with
// `normal` oriented against it).
fn reflect_dir(dir_in: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return dir_in - normal * (2.0 * dot(normal, dir_in));
}

// Snell refraction of the travel direction `dir_in` at a surface whose
// (unit) `normal` is oriented toward the side the ray arrives from.
// valid = false means total internal reflection.
struct RefractResult {
    dir: vec3<f32>,
    valid: bool,
};

fn refract_dir(dir_in: vec3<f32>, normal: vec3<f32>, eta_i: f32, eta_t: f32) -> RefractResult {
    let cos_i = -dot(normal, dir_in);
    let eta = eta_i / eta_t;
    let sin_t_sq = eta * eta * (1.0 - cos_i * cos_i);
    if (sin_t_sq >= 1.0) {
        return RefractResult(vec3<f32>(0.0), false);
    }
    let cos_t = sqrt(1.0 - sin_t_sq);
    return RefractResult(dir_in * eta + normal * (eta * cos_i - cos_t), true);
}

// The eta^2 transmission factor for the transport mode: radiance picks
// up (eta_t/eta_i)^2 crossing a refractive boundary, importance does not.
fn eta_factor(transport: u32, eta_i: f32, eta_t: f32) -> f32 {
    if (transport == TRANSPORT_RADIANCE) {
        let r = eta_t / eta_i;
        return r * r;
    }
    return 1.0;
}

// One scattering event sampled from a BSDF. `normal` must be oriented
// toward the side the ray arrives from (normal . dir_in < 0);
// `front_face` says whether the geometric normal faces that side (for
// glass: which medium is incident). Delta BSDFs sample
// deterministically; their pdf_omega is 0 (the delta has no numeric
// value — the MIS machinery replaces it symbolically).
struct BsdfSample {
    dir_out: vec3<f32>,
    pdf_omega: f32,
    // f . |cos| / pdf, the throughput multiplier of the event. For delta
    // BSDFs the delta cancels and this is the deterministic specular
    // weight; the Fresnel probability cancels against the BSDF value.
    weight: vec3<f32>,
};

fn bsdf_sample(albedo: vec3<f32>, mat_type: u32, ior: f32, normal: vec3<f32>,
               front_face: bool, dir_in: vec3<f32>, transport: u32,
               state: ptr<function, u32>) -> BsdfSample {
    if (mat_type == MAT_MIRROR) {
        return BsdfSample(reflect_dir(dir_in, normal), 0.0, albedo);
    }
    if (mat_type == MAT_GLASS) {
        let eta_i = select(ior, 1.0, front_face);
        let eta_t = select(1.0, ior, front_face);
        let cos_i = -dot(normal, dir_in);
        let f = fresnel_dielectric(cos_i, eta_i, eta_t);
        if (pcg(state) < f) {
            // Reflection. The Fresnel reflectance is both the discrete
            // choice probability and the BSDF value, so it cancels out
            // of the weight.
            return BsdfSample(reflect_dir(dir_in, normal), 0.0, albedo);
        }
        // Refraction (total internal reflection cannot reach this
        // branch: it gives f = 1, so the roll reflects); the fallback
        // mirrors the CPU's defensive unwrap_or_else.
        let refr = refract_dir(dir_in, normal, eta_i, eta_t);
        let dir_out = select(reflect_dir(dir_in, normal), refr.dir, refr.valid);
        return BsdfSample(dir_out, 0.0, albedo * eta_factor(transport, eta_i, eta_t));
    }
    // Diffuse (and emissive, which never scatters — unreachable here).
    let dir_out = cosine_hemisphere(normal, state);
    return BsdfSample(dir_out, dot(dir_out, normal) / PI, albedo);
}

// f (the raw BSDF, no cosine) and the solid-angle pdf of `dir_out`
// under the BSDF's sampling distribution, for an arbitrary direction
// pair. Delta BSDFs return (0, 0) — exactly what invalidates
// connections through them. Emissive materials carry a diffuse BSDF
// with albedo 0 (same as the CPU), so they fall into the same branch.
struct BsdfEval {
    f: vec3<f32>,
    pdf: f32,
};

fn eval_bsdf(albedo: vec3<f32>, mat_type: u32, normal: vec3<f32>,
             dir_in: vec3<f32>, dir_out: vec3<f32>) -> BsdfEval {
    if (is_delta_type(mat_type)) {
        return BsdfEval(vec3<f32>(0.0), 0.0);
    }
    let cos_i = -dot(normal, dir_in);
    let cos_o = dot(normal, dir_out);
    if (cos_i > 0.0 && cos_o > 0.0) {
        return BsdfEval(albedo * (1.0 / PI), cos_o / PI);
    }
    return BsdfEval(vec3<f32>(0.0), 0.0);
}

// The deterministic sample weight of a SPECULAR direction pair at a
// delta BSDF — needed for hypothetical path extensions, where the rolls
// never happened but the path geometry fixes the directions. valid =
// false if the pair is not the specular pair. `normal`/`front_face`
// describe the event being weighted.
struct SpecularWeight {
    weight: vec3<f32>,
    valid: bool,
};

fn specular_weight(albedo: vec3<f32>, mat_type: u32, ior: f32, normal: vec3<f32>,
                   front_face: bool, dir_in: vec3<f32>, dir_out: vec3<f32>,
                   transport: u32) -> SpecularWeight {
    if (mat_type == MAT_MIRROR) {
        if (dot(reflect_dir(dir_in, normal), dir_out) > SPEC_DOT_EPS) {
            return SpecularWeight(albedo, true);
        }
        return SpecularWeight(vec3<f32>(0.0), false);
    }
    if (mat_type == MAT_GLASS) {
        let eta_i = select(ior, 1.0, front_face);
        let eta_t = select(1.0, ior, front_face);
        if (dot(reflect_dir(dir_in, normal), dir_out) > SPEC_DOT_EPS) {
            return SpecularWeight(albedo, true);
        }
        let refr = refract_dir(dir_in, normal, eta_i, eta_t);
        if (refr.valid && dot(refr.dir, dir_out) > SPEC_DOT_EPS) {
            return SpecularWeight(albedo * eta_factor(transport, eta_i, eta_t), true);
        }
        return SpecularWeight(vec3<f32>(0.0), false);
    }
    return SpecularWeight(vec3<f32>(0.0), false);
}

// --- PathView helpers (the WGSL shape of bdpt.rs's PathView methods) ---

// The path vertex at position m (1-based).
fn pv_vertex(eye: ptr<function, array<PathVertex, 8>>,
             light_path: ptr<function, array<PathVertex, 8>>,
             view: PathView, m: u32) -> PathVertex {
    // Explicit branch, not select(): the discarded side would read slots
    // past the realized subpath.
    if (m <= view.n_eye) {
        return (*eye)[m - 1u];
    }
    return (*light_path)[view.k - m];
}

// Travel direction from position a to position b (adjacent). Position 0
// is the camera: its edge direction is the primary ray.
fn pv_dir(eye: ptr<function, array<PathVertex, 8>>,
          light_path: ptr<function, array<PathVertex, 8>>,
          view: PathView, a: u32, b: u32) -> vec3<f32> {
    if (a == 0u) { return view.cam_dir; }
    let d = pv_vertex(eye, light_path, view, b).p - pv_vertex(eye, light_path, view, a).p;
    let len = vlen(d);
    if (len < 1e-8) { return vec3<f32>(0.0); }
    return d * (1.0 / len);
}

// (normal, front_face) at position m for an event whose incoming travel
// direction is `dir_in`: the normal oriented toward the side the ray
// arrives from. The geometric normal is recovered from the stored
// orientation, then flipped when the event arrives from the other side
// (hypothetical extensions do).
struct OrientedNormal {
    normal: vec3<f32>,
    front_face: bool,
};

fn pv_oriented(eye: ptr<function, array<PathVertex, 8>>,
               light_path: ptr<function, array<PathVertex, 8>>,
               view: PathView, m: u32, dir_in: vec3<f32>) -> OrientedNormal {
    let v = pv_vertex(eye, light_path, view, m);
    let n_geo = select(v.normal * -1.0, v.normal, v.front_face);
    if (dot(n_geo, dir_in) < 0.0) {
        return OrientedNormal(n_geo, true);
    }
    return OrientedNormal(n_geo * -1.0, false);
}

// The solid-angle pdf of the scattering event at position m — arriving
// from `arrival`, scattering toward `to` (adjacent positions) — under
// the BSDF's own sampling distribution. Delta BSDFs: the symbolic 1
// when the pair is the specular pair, 0 otherwise.
fn pv_event_pdf(eye: ptr<function, array<PathVertex, 8>>,
                light_path: ptr<function, array<PathVertex, 8>>,
                view: PathView, m: u32, arrival: u32, to: u32) -> f32 {
    let v = pv_vertex(eye, light_path, view, m);
    let d_in = pv_dir(eye, light_path, view, arrival, m);
    let d_out = pv_dir(eye, light_path, view, m, to);
    let o = pv_oriented(eye, light_path, view, m, d_in);
    if (is_delta_type(v.mat_type)) {
        let sw = specular_weight(v.albedo, v.mat_type, v.ior, o.normal, o.front_face,
                                 d_in, d_out, TRANSPORT_RADIANCE);
        return select(0.0, 1.0, sw.valid);
    }
    return eval_bsdf(v.albedo, v.mat_type, o.normal, d_in, d_out).pdf;
}

// The sample weight (f.|cos|/pdf) of the scattering event at position m
// — arriving from `arrival`, scattering toward `to` — under
// `transport`. Delta BSDFs return the deterministic specular weight;
// impossible events return 0.
fn pv_event_weight(eye: ptr<function, array<PathVertex, 8>>,
                   light_path: ptr<function, array<PathVertex, 8>>,
                   view: PathView, m: u32, arrival: u32, to: u32,
                   transport: u32) -> vec3<f32> {
    let v = pv_vertex(eye, light_path, view, m);
    let d_in = pv_dir(eye, light_path, view, arrival, m);
    let d_out = pv_dir(eye, light_path, view, m, to);
    let o = pv_oriented(eye, light_path, view, m, d_in);
    if (is_delta_type(v.mat_type)) {
        let sw = specular_weight(v.albedo, v.mat_type, v.ior, o.normal, o.front_face,
                                 d_in, d_out, transport);
        return select(vec3<f32>(0.0), sw.weight, sw.valid);
    }
    let e = eval_bsdf(v.albedo, v.mat_type, o.normal, d_in, d_out);
    if (e.pdf <= 0.0) { return vec3<f32>(0.0); }
    return e.f * (dot(o.normal, d_out) / e.pdf);
}

// Solid-angle to area-measure conversion for the edge (a, b): the
// cosine at the destination vertex (absolute — the projected area is
// orientation-independent) over the squared distance.
fn pv_convert_to_area(eye: ptr<function, array<PathVertex, 8>>,
                      light_path: ptr<function, array<PathVertex, 8>>,
                      view: PathView, pdf_omega: f32, from_pos: u32, to_pos: u32) -> f32 {
    let d = pv_vertex(eye, light_path, view, to_pos).p - pv_vertex(eye, light_path, view, from_pos).p;
    let r_sq = dot(d, d);
    if (r_sq < 1e-8) { return 0.0; }
    let dir = d * (1.0 / sqrt(r_sq));
    let cos_dest = abs(dot(pv_vertex(eye, light_path, view, to_pos).normal, dir * -1.0));
    return pdf_omega * cos_dest / r_sq;
}

// The area pdf of edge e_m (positions m-1, m) as sampled at position
// m-1 toward position m — the eye-direction sampling a^e(e_m). Delta
// sampling vertices contribute the symbolic 1; impossible directions
// contribute 0.
fn pv_pdf_eye(eye: ptr<function, array<PathVertex, 8>>,
              light_path: ptr<function, array<PathVertex, 8>>,
              view: PathView, m: u32) -> f32 {
    if (m < 2u || m > view.k) { return 0.0; }
    let pdf_omega = pv_event_pdf(eye, light_path, view, m - 1u, m - 2u, m);
    if (pdf_omega <= 0.0) { return 0.0; }
    return pv_convert_to_area(eye, light_path, view, pdf_omega, m - 1u, m);
}

// The area pdf of edge e_m as sampled at position m toward position
// m-1 — the light-direction sampling a^l(e_m). m = k+1 is the light
// position sample (pdf 1/(n.A)); m = k is the emitter event, which has
// no incoming direction: the emitter scatters cosine-weighted around
// its stored (emission-side) normal with f = 1 — except a laser
// aperture, whose emission is uniform in its cone (pdf
// 1/(2π(1 - cos θ)) inside the beam, 0 outside).
fn pv_pdf_light(eye: ptr<function, array<PathVertex, 8>>,
                light_path: ptr<function, array<PathVertex, 8>>,
                view: PathView, m: u32) -> f32 {
    if (m == view.k + 1u) { return 1.0 / view.inv_light_pdf; }
    if (m < 2u || m > view.k) { return 0.0; }
    var pdf_omega = 0.0;
    if (m == view.k) {
        let d_out = pv_dir(eye, light_path, view, m, m - 1u);
        let v = pv_vertex(eye, light_path, view, m);
        let cos = dot(v.normal, d_out);
        if (v.mat_type == MAT_LASER) {
            // Uniform in the cone: pdf 1/(2π(1 - cos θ)) inside, 0 out.
            let cos_theta = v.ior;
            if (cos >= cos_theta) {
                pdf_omega = 1.0 / (2.0 * PI * (1.0 - cos_theta));
            }
        } else {
            if (cos > 0.0) { pdf_omega = cos / PI; }
        }
    } else {
        pdf_omega = pv_event_pdf(eye, light_path, view, m, m + 1u, m - 1u);
    }
    if (pdf_omega <= 0.0) { return 0.0; }
    return pv_convert_to_area(eye, light_path, view, pdf_omega, m, m - 1u);
}

// --- MIS weight (power heuristic, beta = 2), f32 port of bdpt.rs ---
//
// Strategy s' (1..k-1) connects x_{s'} to x_{s'+1} across e_{s'+1};
// strategy k has no connection edge (the eye subpath hit the emitter).
//
//   p[s'] = (prod_{m=2}^{s'} a^e(e_m)) . (1/nA) . (prod_{m=s'+2}^{k} a^l(e_m)) . Qe(s') . Ql(k-s'-1)
//   p[k]  = (prod_{m=2}^{k} a^e(e_m)) . Qe(k)
//
// The step ratio between adjacent strategies is
//
//   p[s'+1]/p[s'] = a^e(e_{s'+1}) / a^l(e_{s'+2}) . Qe(s'+1)/Qe(s') . Ql(k-s'-2)/Ql(k-s'-1)
//
// with a^l(e_{k+1}) := 1/nA and Ql(-1) := 1. The weight is
// 1 / sum (p[s']/p[s])^2 over the *valid* strategies, computed by
// walking both directions from the actual strategy s and accumulating
// the step products — p[s] itself cancels.
//
// Validity: a strategy's connection edge must have nonzero pdf from
// both sides and non-delta endpoints. Its contribution is then 0 (eval
// is 0), and leaving it out of the normalization is what keeps the
// weights of the contributing strategies summing to 1. The divisions
// in the walk are always by realized (positive) pdfs — the up-walk
// divides by light-side pdfs at or beyond the connection, the down-walk
// by eye edges — so a zero factor only ever appears as a multiplicative
// 0 (or an INF step, which the finite check below excludes).

// True when strategy s2 can sample the path at all.
fn mis_valid(eye: ptr<function, array<PathVertex, 8>>,
             light_path: ptr<function, array<PathVertex, 8>>,
             view: PathView, s2: u32) -> bool {
    if (s2 == view.k) { return true; }
    let m = s2 + 1u;
    if (is_delta_type(pv_vertex(eye, light_path, view, s2).mat_type) ||
        is_delta_type(pv_vertex(eye, light_path, view, m).mat_type)) {
        return false;
    }
    return pv_pdf_eye(eye, light_path, view, m) > 0.0 &&
           pv_pdf_light(eye, light_path, view, m) > 0.0;
}

// One step of the ratio walk: p[s2+1]/p[s2].
fn mis_step(eye: ptr<function, array<PathVertex, 8>>,
            light_path: ptr<function, array<PathVertex, 8>>,
            view: PathView, qe: ptr<function, array<f32, 9>>,
            ql: ptr<function, array<f32, 9>>, s2: u32) -> f32 {
    let a_e = pv_pdf_eye(eye, light_path, view, s2 + 1u);
    let a_l = pv_pdf_light(eye, light_path, view, s2 + 2u);
    let qe_ratio = (*qe)[s2 + 1u] / (*qe)[s2];
    var ql_ratio = 1.0; // Ql(-1) := 1
    // Explicit branch, not select(): the discarded side would index ql
    // with an unsigned underflow.
    if (view.k >= s2 + 2u) {
        ql_ratio = (*ql)[view.k - s2 - 2u] / (*ql)[view.k - s2 - 1u];
    }
    return a_e / a_l * qe_ratio * ql_ratio;
}

// Rust's f64::max ignores NaN (returns the other operand); the WGSL max
// builtin has no such guarantee across backends, and the ratio walk can
// produce NaN steps (0/0 at degenerate delta edges). Replicating the CPU
// semantics explicitly keeps the normalization set identical to bdpt.rs.
fn max_ignore_nan(a: f32, b: f32) -> f32 {
    if (a > b) { return a; }
    if (b > a) { return b; }
    // Equal, or at least one NaN: prefer whichever operand is not NaN.
    if (a == a) { return a; }
    return b;
}

fn mis_weight(eye: ptr<function, array<PathVertex, 8>>,
              light_path: ptr<function, array<PathVertex, 8>>,
              view: PathView, qe: ptr<function, array<f32, 9>>,
              ql: ptr<function, array<f32, 9>>, s: u32) -> f32 {
    let k = view.k;
    var up = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var down = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // Walk up from s: each step multiplies by p[s2+1]/p[s2]. The loop
    // must reach s2 = k-1 so that up[k] — the direct-hit strategy, a
    // valid competitor of every connection strategy — is computed.
    up[s] = 1.0;
    for (var s2 = s; s2 < k; s2 = s2 + 1u) {
        up[s2 + 1u] = up[s2] * mis_step(eye, light_path, view, qe, ql, s2);
    }

    // Walk down from s: each step multiplies by p[s2]/p[s2+1].
    down[s] = 1.0;
    for (var s2 = s - 1u; s2 >= 1u; s2 = s2 - 1u) {
        down[s2] = down[s2 + 1u] / mis_step(eye, light_path, view, qe, ql, s2);
    }

    // Power heuristic (beta = 2), normalized by the largest valid ratio
    // to keep the squares in a safe floating-point range. WGSL has no
    // isFinite builtin; `ratio < 3e38` is the f32 equivalent (NaN and
    // INF both compare false).
    var r_max = 1.0;
    for (var s2 = 1u; s2 <= k; s2 = s2 + 1u) {
        if (mis_valid(eye, light_path, view, s2)) {
            let ratio = max_ignore_nan(up[s2], down[s2]);
            if (ratio < 3.0e38) { r_max = max(r_max, ratio); }
        }
    }
    var sum = 0.0;
    for (var s2 = 1u; s2 <= k; s2 = s2 + 1u) {
        if (mis_valid(eye, light_path, view, s2)) {
            let ratio = max_ignore_nan(up[s2], down[s2]);
            if (ratio < 3.0e38) {
                let q = ratio / r_max;
                sum += q * q;
            }
        }
    }
    if (!(sum > 0.0)) { return 1.0; }
    let q = 1.0 / r_max;
    return q * q / sum;
}

// --- Russian roulette ---

// Survival probability for extending a subpath whose post-scatter
// throughput is `beta`, on extension number `ext` (1-based: the
// extension x_l -> x_{l+1} is number l, y_j -> y_{j+1} is number j+1).
fn rr_survival(beta: vec3<f32>, ext: u32) -> f32 {
    if (ext < RR_START) { return 1.0; }
    return clamp(max(beta.x, max(beta.y, beta.z)), RR_MIN, 1.0);
}

// Compute the Russian-roulette prefix products the MIS weights need:
// qe[m] is Qe(m), the product of the survival probabilities of the
// eye-subpath prefix x_1..x_m (the rolls at x_1..x_{m-1}); ql[j] is
// Ql(j), the same for the light-subpath prefix y_0..y_j.
//
// Both sweeps recompute the post-scatter throughputs the rolls were
// made from — each extension's factor is the sample weight of the
// scattering event at its start vertex, evaluated from the path
// geometry — so the q's here are exactly the ones the subpath sampling
// rolled. That also covers the *hypothetical* rolls of strategies that
// would extend a subpath past where it actually stopped (killed by
// roulette, escaped, or depth-capped): the events those extensions
// would perform belong to the path, so their weights and q's are
// computable even though the rolls never happened.
//
// `alpha_light0` is the light-subpath throughput at y_0: the realized
// `emission * n * A` for connection strategies, or the same expression
// for the direct-hit strategy's hypothetical light subpath starting at
// the emitter hit.
fn rr_prefix_products(eye: ptr<function, array<PathVertex, 8>>,
                      light_path: ptr<function, array<PathVertex, 8>>,
                      view: PathView, alpha_light0: vec3<f32>,
                      qe: ptr<function, array<f32, 9>>,
                      ql: ptr<function, array<f32, 9>>) {
    let k = view.k;

    // Eye-side sweep: the roll at x_l governs the extension to x_{l+1},
    // whose factor is the sample weight of the event at x_l toward
    // x_{l+1}. The eye subpath carries radiance.
    var alpha = vec3<f32>(1.0);
    (*qe)[1u] = 1.0;
    for (var l = 1u; l < k; l = l + 1u) {
        let factor = pv_event_weight(eye, light_path, view, l, l - 1u, l + 1u,
                                      TRANSPORT_RADIANCE);
        let beta = alpha * factor;
        let q = rr_survival(beta, l);
        (*qe)[l + 1u] = (*qe)[l] * q;
        alpha = beta * (1.0 / q);
    }

    // Light-side sweep: the roll at y_j governs the extension to y_{j+1},
    // whose factor is the sample weight of the event at y_j = x_{k-j}
    // toward y_{j+1} = x_{k-j-1}. The first extension has no BSDF at the
    // emitter (f = 1): cosine sampling gives the weight PI, a laser
    // aperture's uniform cone sampling gives cos0 * 2*PI*(1 - cos theta)
    // (0 outside the beam). The light subpath carries importance.
    var alpha_l = alpha_light0;
    (*ql)[0u] = 1.0;
    for (var j = 0u; j < k - min(k, 2u); j = j + 1u) {
        var factor: vec3<f32>;
        if (j == 0u) {
            // The emitter event at x_k = y_0, scattering toward x_{k-1}.
            let v = pv_vertex(eye, light_path, view, k);
            let d_out = pv_dir(eye, light_path, view, k, k - 1u);
            let cos0 = dot(v.normal, d_out);
            if (v.mat_type == MAT_LASER) {
                let cos_theta = v.ior;
                if (cos0 >= cos_theta) {
                    let f = cos0 * 2.0 * PI * (1.0 - cos_theta);
                    factor = vec3<f32>(f);
                } else {
                    factor = vec3<f32>(0.0);
                }
            } else {
                factor = vec3<f32>(PI);
            }
        } else {
            factor = pv_event_weight(eye, light_path, view, k - j, k - j + 1u, k - j - 1u,
                                     TRANSPORT_IMPORTANCE);
        }
        let beta = alpha_l * factor;
        let q = rr_survival(beta, j + 1u);
        (*ql)[j + 1u] = (*ql)[j] * q;
        alpha_l = beta * (1.0 / q);
    }
}

// --- Light sampling (uniform surface sample, pdf = 1/area) ---

// Build an orthonormal frame around `normal` (the same helper-axis rule
// as cosine_hemisphere).
fn frame(normal: vec3<f32>) -> mat3x3<f32> {
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
    let t = normalize3(cross(a, normal));
    let b = cross(normal, t);
    return mat3x3<f32>(t, b, normal);
}

fn sample_light(l: GpuLight, state: ptr<function, u32>) -> vec3<f32> {
    if (l.shape == 0u) {
        let z = 1.0 - 2.0 * pcg(state);
        let phi = 2.0 * PI * pcg(state);
        let r = sqrt(max(0.0, 1.0 - z * z));
        return l.center + vec3<f32>(r * cos(phi), r * sin(phi), z) * l.radius;
    }
    if (l.shape == 2u) {
        // Laser aperture: uniform point on the disc.
        let f = frame(l.normal);
        let r = l.radius * sqrt(pcg(state));
        let phi = 2.0 * PI * pcg(state);
        return l.center + f[0] * (r * cos(phi)) + f[1] * (r * sin(phi));
    }
    let u = pcg(state) * 2.0 - 1.0;
    let v = pcg(state) * 2.0 - 1.0;
    return l.center + l.tangent * (u * l.half_w) + l.bitangent * (v * l.half_h);
}

// The direction the light subpath leaves the sampled point in: uniform
// in the laser's cone (pdf 1/(2π(1 - cos θ))), cosine-weighted around
// the normal otherwise. The CPU's AreaLight::sample returns this as the
// second tuple element; WGSL returns it separately because the cone
// sampling needs its own random numbers.
fn sample_light_dir(l: GpuLight, p: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    if (l.shape == 2u) {
        let f = frame(l.normal);
        let cos_theta = cos(l.half_w);
        let u = 1.0 - pcg(state) * (1.0 - cos_theta);
        let phi = 2.0 * PI * pcg(state);
        let sin_theta = sqrt(max(0.0, 1.0 - u * u));
        return f[0] * (sin_theta * cos(phi)) + f[1] * (sin_theta * sin(phi)) + l.normal * u;
    }
    return cosine_hemisphere(light_normal(l, p), state);
}

fn light_contains(l: GpuLight, p: vec3<f32>) -> bool {
    if (l.shape == 0u) {
        return abs(vlen(p - l.center) - l.radius) < 1e-3;
    }
    if (l.shape == 2u) {
        let local = p - l.center;
        return abs(dot(local, l.normal)) < 1e-3
            && dot(local, local) <= l.radius * l.radius + 1e-3;
    }
    let local = p - l.center;
    return abs(dot(local, l.normal)) < 1e-3
        && abs(dot(local, l.tangent)) <= l.half_w + 1e-3
        && abs(dot(local, l.bitangent)) <= l.half_h + 1e-3;
}

fn light_area(l: GpuLight) -> f32 {
    if (l.shape == 0u) { return 4.0 * PI * l.radius * l.radius; }
    if (l.shape == 2u) { return PI * l.radius * l.radius; }
    return 4.0 * l.half_w * l.half_h;
}

fn light_normal(l: GpuLight, p: vec3<f32>) -> vec3<f32> {
    if (l.shape == 0u) {
        // Outward normal at the sampled point — the same direction
        // geometry.rs AreaLight::sample returns for spheres.
        return normalize3(p - l.center);
    }
    return l.normal;
}

// --- BDPT ---

fn bdpt_trace(ray_origin: vec3<f32>, ray_dir: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let max_depth = min(uniforms.max_depth, MAX_PATH_VERTICES);
    var total = vec3<f32>(0.0);

    // ---- Eye subpath: x_1 .. x_s ----
    var eye: array<PathVertex, 8>;
    var eye_len = 0u;
    var throughput = vec3<f32>(1.0);
    var cur_origin = ray_origin;
    var cur_dir = ray_dir;

    for (var bounce = 0u; bounce < max_depth; bounce = bounce + 1u) {
        let hit = intersect_scene(cur_origin, cur_dir);
        if (!has_hit(hit)) {
            if (eye_len == 0u) { return BACKGROUND; }
            break;
        }
        // Orient the normal toward the side the ray arrived from.
        // Diffuse surfaces are one-sided: a back-face hit cannot
        // scatter, so the walk stops there. Mirror and glass flip the
        // normal and continue — glass needs `front_face` to pick
        // eta_i/eta_t.
        let front_face = dot(hit.normal, cur_dir) < 0.0;
        if (!front_face && !is_delta_type(hit.mat_type)) { break; }
        let normal = select(hit.normal * -1.0, hit.normal, front_face);
        // MAT_LASER is an emitter too (a laser aperture): the walk stops
        // there, exactly like MAT_EMISSIVE.
        let on_light = hit.mat_type == MAT_EMISSIVE || hit.mat_type == MAT_LASER;
        eye[eye_len] = PathVertex(hit.p, normal, front_face, hit.albedo, hit.emission,
                                  hit.mat_type, hit.ior, throughput, on_light);
        eye_len = eye_len + 1u;
        if (on_light) { break; }
        // Sample the BSDF, then roll the roulette on the post-scatter
        // throughput: survive with probability q and divide by q, which
        // keeps the expected contribution of the subpath unchanged.
        let s = bsdf_sample(hit.albedo, hit.mat_type, hit.ior, normal, front_face,
                           cur_dir, TRANSPORT_RADIANCE, state);
        let beta = throughput * s.weight;
        let q = rr_survival(beta, eye_len);
        if (q < 1.0 && pcg(state) >= q) { break; }
        throughput = beta * (1.0 / q);
        // Offset along the side the new ray travels to: a refracted ray
        // leaves through the surface, so offsetting by the arrival-side
        // normal would start it inside the surface.
        let offset = select(normal * -1.0, normal, dot(s.dir_out, normal) > 0.0);
        cur_origin = hit.p + offset * EPS_NEAR;
        cur_dir = s.dir_out;
    }

    // ---- Light subpath: y_0 .. y_t ----
    var light_path: array<PathVertex, 8>;
    var light_len = 0u;
    var inv_light_pdf = 0.0;
    if (uniforms.n_lights > 0u) {
        let pick = u32(f32(uniforms.n_lights) * pcg(state));
        let light = lights[min(pick, uniforms.n_lights - 1u)];
        inv_light_pdf = f32(uniforms.n_lights) * light_area(light);
        let p0 = sample_light(light, state);
        let n0 = light_normal(light, p0);
        // The direction the subpath leaves the aperture in: uniform in
        // the laser's cone, cosine-weighted otherwise. The CPU's
        // AreaLight::sample_dir returns this.
        let d0 = sample_light_dir(light, p0, state);
        let is_laser = light.shape == 2u;
        // The first light extension (y_0 -> y_1) is unconditional
        // (RR_START >= 2). The emitter has no BSDF (f = 1) and scatters
        // cosine-weighted, so the edge factor is cos0 * PI. A laser
        // aperture samples uniformly in its cone, so the edge factor is
        // cos0 * 2*PI*(1 - cos theta) — and the direction must lie
        // inside the beam, where the emitter's radiance is nonzero.
        let cos0 = dot(n0, d0);
        var edge_factor = PI;
        if (is_laser) {
            let cos_theta = cos(light.half_w);
            if (cos0 >= cos_theta) {
                edge_factor = cos0 * 2.0 * PI * (1.0 - cos_theta);
            } else {
                edge_factor = 0.0;
            }
        }
        var ltp = light.emission * inv_light_pdf;
        // The y_0 vertex material carries the emission cone so the MIS
        // pdf machinery (pv_pdf_light at m = k) can see it.
        let y0_mat = select(MAT_EMISSIVE, MAT_LASER, is_laser);
        let y0_ior = select(1.0, cos(light.half_w), is_laser);
        light_path[0] = PathVertex(p0, n0, true, vec3<f32>(0.0), light.emission,
                                   y0_mat, y0_ior, ltp, true);
        light_len = 1u;

        if (edge_factor > 0.0) {
            ltp = ltp * edge_factor;
            cur_origin = p0 + n0 * EPS_NEAR;
            cur_dir = d0;
            // The connection strategies need k = i + j + 2 <= max_depth, so the
            // last light vertex (j = max_depth - 1) could never be connected.
            // Building one vertex fewer avoids a wasted intersection.
            let steps = max_depth - min(max_depth, 2u);
            for (var bounce = 0u; bounce < steps; bounce = bounce + 1u) {
                let hit = intersect_scene(cur_origin, cur_dir);
                if (!has_hit(hit)) { break; }
                // Pure emitters do not reflect.
                if (hit.mat_type == MAT_EMISSIVE || hit.mat_type == MAT_LASER) { break; }
                let front_face = dot(hit.normal, cur_dir) < 0.0;
                if (!front_face && !is_delta_type(hit.mat_type)) { break; }
                let normal = select(hit.normal * -1.0, hit.normal, front_face);
                light_path[light_len] = PathVertex(hit.p, normal, front_face, hit.albedo,
                                                   hit.emission, hit.mat_type, hit.ior,
                                                   ltp, false);
                light_len = light_len + 1u;
                // Russian roulette for the extension y_j -> y_{j+1}
                // (j = light_len - 1u >= 1, number j + 1). Surviving divides
                // ltp by q, which also applies the factor for the next vertex.
                let j = light_len - 1u;
                let s = bsdf_sample(hit.albedo, hit.mat_type, hit.ior, normal, front_face,
                                   cur_dir, TRANSPORT_IMPORTANCE, state);
                let beta = ltp * s.weight;
                let q = rr_survival(beta, j + 1u);
                if (q < 1.0 && pcg(state) >= q) { break; }
                ltp = beta * (1.0 / q);
                let offset = select(normal * -1.0, normal, dot(s.dir_out, normal) > 0.0);
                cur_origin = hit.p + offset * EPS_NEAR;
                cur_dir = s.dir_out;
            }
        }
    }

    // ---- Strategy "eye subpath hits a light" ----
    if (eye_len > 0u && eye[eye_len - 1u].on_light) {
        let last = eye[eye_len - 1u];
        let k = eye_len;
        // Default 0: a hit on emissive geometry that is no sampleable
        // light surface (the dark curved side of a laser housing)
        // carries no emission. A found light overwrites this.
        var w = 0.0;
        if (uniforms.n_lights > 0u) {
            var found = false;
            for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
                if (!found && light_contains(lights[li], last.p)) {
                    found = true;
                    // A laser aperture radiates only into its cone: a hit
                    // outside the beam (the dark side of the housing, or
                    // the aperture seen from behind) carries no emission.
                    // The eye vertex's stored normal is the aperture's
                    // outward normal; the incoming direction is the
                    // camera ray's.
                    var in_beam = true;
                    if (last.mat_type == MAT_LASER) {
                        let cos_theta = last.ior;
                        in_beam = dot(last.normal, ray_dir * -1.0) >= cos_theta;
                    }
                    if (in_beam) {
                        let inv_pdf = f32(uniforms.n_lights) * light_area(lights[li]);
                        let view = PathView(k, k, inv_pdf, ray_dir);
                        var qe = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                        var ql = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                        // The hypothetical light subpath of the other
                        // strategies starts at this hit point with the
                        // emitter's throughput (emission * n * A, the
                        // inverse light-sampling pdf).
                        rr_prefix_products(&eye, &light_path, view,
                                           lights[li].emission * inv_pdf, &qe, &ql);
                        w = mis_weight(&eye, &light_path, view, &qe, &ql, k);
                    } else {
                        w = 0.0;
                    }
                }
            }
        }
        total += last.throughput * last.emission * w;
    }

    // ---- Connection strategies: eye vertex x_{i+1} to light vertex y_j ----
    for (var i = 0u; i < eye_len; i = i + 1u) {
        if (eye[i].on_light) { break; } // only the last eye vertex can be emissive
        for (var j = 0u; j < light_len; j = j + 1u) {
            let k = i + j + 2u; // vertices after the camera
            if (k > max_depth) { break; }
            // Delta BSDFs cannot be connection endpoints: their eval is
            // 0, so the contribution would be 0 anyway.
            if (is_delta_type(eye[i].mat_type) || is_delta_type(light_path[j].mat_type)) {
                continue;
            }
            let view = PathView(k, i + 1u, inv_light_pdf, ray_dir);

            // Contribution: alpha_eye * f(x) * G * f(y) * alpha_light.
            // The path arrives at x_{i+1} from x_i (the camera for i = 0)
            // and at y_j from x_{i+1}, leaving toward y_{j-1}.
            let d_conn = pv_dir(&eye, &light_path, view, i + 1u, i + 2u);
            var d_in_eye = ray_dir;
            if (i > 0u) {
                d_in_eye = pv_dir(&eye, &light_path, view, i, i + 1u);
            }
            let o_eye = pv_oriented(&eye, &light_path, view, i + 1u, d_in_eye);
            let e_eye = eval_bsdf(eye[i].albedo, eye[i].mat_type, o_eye.normal,
                                  d_in_eye, d_conn);
            var f_light = vec3<f32>(1.0);
            var n_light_end = light_path[j].normal;
            if (j == 0u) {
                // y_0 is the emitter itself: no BSDF applies there — but
                // a laser aperture radiates only into its cone, so the
                // connection direction must lie inside the beam. The
                // emission direction is y_0 -> x_{i+1}, i.e. -d_conn
                // (d_conn points from the eye vertex to the light
                // vertex); testing d_conn itself inverts the gate and
                // zeroes every connection to the aperture.
                let y0 = light_path[0u];
                if (y0.mat_type == MAT_LASER) {
                    let cos_theta = y0.ior;
                    if (dot(y0.normal, d_conn * -1.0) < cos_theta) {
                        f_light = vec3<f32>(0.0);
                    }
                }
            }
            if (j > 0u) {
                let d_in_light = d_conn * -1.0;
                let d_out_light = pv_dir(&eye, &light_path, view, i + 2u, i + 3u);
                let o_light = pv_oriented(&eye, &light_path, view, i + 2u, d_in_light);
                f_light = eval_bsdf(light_path[j].albedo, light_path[j].mat_type,
                                   o_light.normal, d_in_light, d_out_light).f;
                n_light_end = o_light.normal;
            }
            let g_conn = g_term(eye[i].p, o_eye.normal, light_path[j].p, n_light_end);
            if (g_conn <= 0.0 || all(e_eye.f == vec3<f32>(0.0)) || all(f_light == vec3<f32>(0.0))) {
                continue;
            }
            if (!visible(eye[i].p, o_eye.normal, light_path[j].p)) { continue; }
            let contrib = eye[i].throughput * e_eye.f * g_conn * f_light
                * light_path[j].throughput;

            var qe = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            var ql = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            rr_prefix_products(&eye, &light_path, view, light_path[0].throughput, &qe, &ql);
            let w = mis_weight(&eye, &light_path, view, &qe, &ql, i + 1u);
            total += contrib * w;
        }
    }

    return total;
}

// --- Firefly rejection ---
//
// Caustic and mirror paths are sampled by BDPT's specular-chain
// strategies, which succeed rarely and carry a huge throughput when
// they do. One lucky sample with radiance ~50 landing in a pixel whose
// true mean is ~0.05 enters the running average at full weight and
// then only dilutes as 1/n — a persistent white dot (everything at or
// above the tone-map white point displays as pure white, so the dot is
// saturated from the first frame it lands). Two complementary controls
// remove it: the per-sample clamp below (bounds the magnitude) and the
// temporal outlier rejection in the accumulation block of `main` (stops
// it from sticking). The CPU applies the same two controls in
// src/bdpt.rs and src/renderer.rs.

// Rec. 709 relative luminance (same weights as Vec3::luminance).
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Clamp one sample's luminance to `bound`, preserving its chroma.
// `bound` comes from the `firefly_clamp` uniform, computed on the CPU
// (src/bdpt.rs::firefly_clamp_bound) as a loose multiple of the
// brightest emitter's luminance — loose because a rare legitimate
// caustic sample is per-sample indistinguishable from a firefly (both
// are huge relative to the pixel mean; the caustic's estimator is a
// rare-event one), so only the physically implausible tail is clamped
// here and the mean-aware temporal rejection handles the rest.
// 0 disables the clamp.
fn clamp_sample(c: vec3<f32>, bound: f32) -> vec3<f32> {
    if (bound <= 0.0) { return c; }
    let lum = luminance(c);
    if (lum > bound && lum > 0.0) {
        return c * (bound / lum);
    }
    return c;
}

// Temporal outlier rejection: a batch whose mean luminance exceeds the
// running mean by more than this many standard deviations is scaled
// down to that bound before entering the accumulation statistics. Must
// match WINSOR_K in src/bdpt.rs.
const WINSOR_K: f32 = 3.0;
// The second-moment memory, in samples: E[lum²] is an exponential
// moving average over roughly this many samples, so one outlier can
// only keep the rejection bound loose for a bounded time. Must match
// MOMENT_WINDOW in src/bdpt.rs.
const MOMENT_WINDOW: f32 = 16.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(uniforms.width);
    let height = u32(uniforms.height);
    if (gid.x >= width || gid.y >= height) { return; }

    let pixel = gid.y * width + gid.x;
    let seed = pixel * 719393u + uniforms.samples * 1013904223u + 1u;
    var state = seed;

    var color = vec3<f32>(0.0);
    for (var s = 0u; s < uniforms.spp; s = s + 1u) {
        // Jittered ray through the pixel, same convention as camera.rs:
        // u in [0,1] left->right, v in [0,1] top->bottom, py negated.
        let u = (f32(gid.x) + pcg(&state)) / uniforms.width;
        let v = (f32(gid.y) + pcg(&state)) / uniforms.height;
        let px = (u * 2.0 - 1.0) * uniforms.fov_h;
        let py = -(v * 2.0 - 1.0) * uniforms.fov_v;
        let dir = normalize3(uniforms.cam_right * px + uniforms.cam_up * py + uniforms.cam_forward);
        color += clamp_sample(bdpt_trace(uniforms.cam_origin, dir, &state),
                              uniforms.firefly_clamp);
    }
    color = color / f32(uniforms.spp);

    // Temporal accumulation with outlier rejection.
    //
    // The accumulation buffer's `.w` channel carries the running second
    // moment of the sample luminance, E[lum²] — the channel was
    // previously hard-coded to 1.0 and unused. σ = sqrt(E[lum²] − μ²)
    // gives the per-pixel standard deviation; a batch whose mean
    // luminance exceeds μ + WINSOR_K·σ is scaled down to that bound
    // before entering the mean, so a firefly leaves no trace.
    //
    // The statistic is an exponential moving average over roughly
    // MOMENT_WINDOW samples, and it tracks the *raw* (pre-winsorization)
    // luminance. Capping a full running average at the bound would
    // deadlock a caustic pixel whose first samples are dark (σ = 0 pins
    // the bound at μ, every bright batch is clamped to μ, and the
    // moment never grows). The EMA keeps a genuine caustic — many
    // bright batches — converging to its true value, while a single
    // firefly's σ-poisoning decays within ~MOMENT_WINDOW samples. That
    // asymmetry is exactly the statistic that tells a caustic from a
    // firefly.
    let prev = accum[pixel];
    let n = f32(uniforms.samples);
    let m = f32(uniforms.spp);
    var batch = color;
    if (n > 0.0 && uniforms.reset == 0u) {
        let mu = luminance(prev.xyz);
        let sigma = sqrt(max(prev.w - mu * mu, 0.0));
        let bound = mu + WINSOR_K * sigma;
        let lum = luminance(batch);
        if (lum > bound && lum > 0.0) {
            batch = batch * (bound / lum);
        }
    }
    let blended = select(batch, (prev.xyz * n + batch * m) / (n + m), uniforms.reset == 0u);
    // EMA of the raw luminance², batch weight ~m/window.
    let alpha = clamp(m / MOMENT_WINDOW, 0.0, 1.0);
    let raw_sq = luminance(color) * luminance(color);
    let blended_sq = select(raw_sq, prev.w * (1.0 - alpha) + raw_sq * alpha, uniforms.reset == 0u);
    accum[pixel] = vec4<f32>(blended, blended_sq);
}