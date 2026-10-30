// Bidirectional path tracing with MIS, ported 1:1 from src/bdpt.rs.
//
// Every surface is diffuse and every direction is cosine-sampled, so the
// area-measure pdf of every sampled path edge is G/PI and the MIS weights
// reduce to per-edge ratios (power heuristic, beta = 2). See bdpt.rs for
// the derivation.
//
// Paths are terminated by Russian roulette rather than a short fixed
// depth: from extension RR_START on, a subpath survives each bounce with
// probability q = clamp(max(throughput), RR_MIN, 1) and divides its
// throughput by q on survival, which keeps the estimator unbiased while
// letting deep paths through. The survival probabilities are part of the
// sampling pdfs, so they also enter the MIS weights, as the prefix
// products computed by rr_prefix_products.

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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct GpuSphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    _pad0: f32,
    emission: vec3<f32>,
    mat_type: u32,
    _pad1: vec3<f32>,
    _pad2: f32,
};

struct GpuQuad {
    center: vec3<f32>,
    half_w: f32,
    normal: vec3<f32>,
    half_h: f32,
    tangent: vec3<f32>,
    _pad0: f32,
    bitangent: vec3<f32>,
    _pad1: f32,
    albedo: vec3<f32>,
    _pad2: f32,
    emission: vec3<f32>,
    mat_type: u32,
    _pad3: vec3<f32>,
    _pad4: f32,
};

struct GpuCube {
    min: vec3<f32>,
    _pad0: f32,
    max: vec3<f32>,
    _pad1: f32,
    albedo: vec3<f32>,
    _pad2: f32,
    emission: vec3<f32>,
    mat_type: u32,
    _pad3: vec3<f32>,
    _pad4: f32,
};

struct GpuPlane {
    point: vec3<f32>,
    _pad0: f32,
    normal: vec3<f32>,
    _pad1: f32,
    albedo: vec3<f32>,
    _pad2: f32,
    emission: vec3<f32>,
    mat_type: u32,
    _pad3: vec3<f32>,
    _pad4: f32,
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
};

struct PathVertex {
    p: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    emission: vec3<f32>,
    throughput: vec3<f32>,
    edge_g: f32,
    on_light: bool,
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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> spheres: array<GpuSphere>;
@group(0) @binding(2) var<storage, read> quads: array<GpuQuad>;
@group(0) @binding(3) var<storage, read> cubes: array<GpuCube>;
@group(0) @binding(4) var<storage, read> planes: array<GpuPlane>;
@group(0) @binding(5) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(6) var<storage, read_write> accum: array<vec4<f32>>;

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

// --- Intersection: brute force over all primitives (scene is small) ---

fn intersect_sphere(s: GpuSphere, origin: vec3<f32>, dir: vec3<f32>, best: ptr<function, Hit>) {
    let oc = origin - s.center;
    let a = dot(dir, dir);
    let b = 2.0 * dot(oc, dir);
    let c = dot(oc, oc) - s.radius * s.radius;
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) { return; }
    let sqrt_d = sqrt(disc);
    var t = (-b - sqrt_d) / (2.0 * a);
    if (t < EPS_NEAR) { t = (-b + sqrt_d) / (2.0 * a); }
    if (t < EPS_NEAR || t >= (*best).t) { return; }
    let p = origin + dir * t;
    (*best).t = t;
    (*best).p = p;
    (*best).normal = normalize3(p - s.center);
    (*best).albedo = s.albedo;
    (*best).emission = s.emission;
    (*best).mat_type = s.mat_type;
}

fn intersect_quad(q: GpuQuad, origin: vec3<f32>, dir: vec3<f32>, best: ptr<function, Hit>) {
    let denom = dot(q.normal, dir);
    if (abs(denom) < 1e-6) { return; }
    let t = dot(q.center - origin, q.normal) / denom;
    if (t < EPS_NEAR || t >= (*best).t) { return; }
    let p = origin + dir * t;
    let local = p - q.center;
    if (abs(dot(local, q.tangent)) > q.half_w || abs(dot(local, q.bitangent)) > q.half_h) { return; }
    (*best).t = t;
    (*best).p = p;
    (*best).normal = q.normal;
    (*best).albedo = q.albedo;
    (*best).emission = q.emission;
    (*best).mat_type = q.mat_type;
}

fn intersect_cube(c: GpuCube, origin: vec3<f32>, dir: vec3<f32>, best: ptr<function, Hit>) {
    var t_min = -3.4e38;
    var t_max = 3.4e38;
    // Slab test per axis; unrolled to avoid indexing overhead.
    for (var i = 0u; i < 3u; i = i + 1u) {
        let d = dir[i];
        let o = origin[i];
        let lo = c.min[i];
        let hi = c.max[i];
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
    let p = origin + dir * t;
    // Outward normal of the face closest to the hit point (same rule as
    // geometry.rs so top/+x/+z hits shade identically).
    let d0 = abs(p.x - c.min.x); let d1 = abs(p.x - c.max.x);
    let d2 = abs(p.y - c.min.y); let d3 = abs(p.y - c.max.y);
    let d4 = abs(p.z - c.min.z); let d5 = abs(p.z - c.max.z);
    var best_d = d0;
    var normal = vec3<f32>(-1.0, 0.0, 0.0);
    if (d1 < best_d) { best_d = d1; normal = vec3<f32>(1.0, 0.0, 0.0); }
    if (d2 < best_d) { best_d = d2; normal = vec3<f32>(0.0, -1.0, 0.0); }
    if (d3 < best_d) { best_d = d3; normal = vec3<f32>(0.0, 1.0, 0.0); }
    if (d4 < best_d) { best_d = d4; normal = vec3<f32>(0.0, 0.0, -1.0); }
    if (d5 < best_d) { best_d = d5; normal = vec3<f32>(0.0, 0.0, 1.0); }
    (*best).t = t;
    (*best).p = p;
    (*best).normal = normal;
    (*best).albedo = c.albedo;
    (*best).emission = c.emission;
    (*best).mat_type = c.mat_type;
}

fn intersect_plane(pl: GpuPlane, origin: vec3<f32>, dir: vec3<f32>, best: ptr<function, Hit>) {
    let denom = dot(pl.normal, dir);
    if (abs(denom) < 1e-6) { return; }
    let t = dot(pl.point - origin, pl.normal) / denom;
    if (t < EPS_NEAR || t >= (*best).t) { return; }
    (*best).t = t;
    (*best).p = origin + dir * t;
    (*best).normal = pl.normal;
    (*best).albedo = pl.albedo;
    (*best).emission = pl.emission;
    (*best).mat_type = pl.mat_type;
}

fn intersect_scene(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    var best: Hit;
    best.t = 1e30;
    best.p = vec3<f32>(0.0);
    best.normal = vec3<f32>(0.0, 1.0, 0.0);
    best.albedo = vec3<f32>(0.0);
    best.emission = vec3<f32>(0.0);
    best.mat_type = 0u;
    for (var i = 0u; i < arrayLength(&spheres); i = i + 1u) {
        intersect_sphere(spheres[i], origin, dir, &best);
    }
    for (var i = 0u; i < arrayLength(&quads); i = i + 1u) {
        intersect_quad(quads[i], origin, dir, &best);
    }
    for (var i = 0u; i < arrayLength(&cubes); i = i + 1u) {
        intersect_cube(cubes[i], origin, dir, &best);
    }
    for (var i = 0u; i < arrayLength(&planes); i = i + 1u) {
        intersect_plane(planes[i], origin, dir, &best);
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

// --- MIS weight (power heuristic, beta = 2), f32 port of bdpt.rs ---
//
// qe[m] is Qe(m), the product of the Russian-roulette survival
// probabilities of the eye-subpath prefix x_1..x_m (the rolls at
// x_1..x_{m-1}); ql[j] is Ql(j), the same for the light-subpath prefix
// y_0..y_j (the rolls at y_0..y_{j-1}). Without roulette they are all 1.

fn mis_weight(edges: ptr<function, array<f32, 8>>, qe: ptr<function, array<f32, 9>>,
              ql: ptr<function, array<f32, 9>>, k: u32, inv_light_pdf: f32, s: u32) -> f32 {
    var r = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // r[k] = Qe(k); r[m] = PI * Qe(m) * Ql(k-m-1) / (nA * e_{m+1}) for m < k.
    r[k] = (*qe)[k];
    for (var m = 1u; m < k; m = m + 1u) {
        let e = (*edges)[m - 1u]; // e_{m+1}
        r[m] = select(0.0, PI * (*qe)[m] * (*ql)[k - m - 1u] / (inv_light_pdf * e), e > 0.0);
    }
    var r_max = 0.0;
    for (var m = 1u; m <= k; m = m + 1u) {
        r_max = max(r_max, r[m]);
    }
    if (!(r_max > 0.0)) { return 1.0; }
    var sum = 0.0;
    for (var m = 1u; m <= k; m = m + 1u) {
        let q = r[m] / r_max;
        sum += q * q;
    }
    let q = r[s] / r_max;
    return q * q / sum;
}

// --- Russian roulette ---

fn rr_survival(beta: vec3<f32>, ext: u32) -> f32 {
    if (ext < RR_START) { return 1.0; }
    return clamp(max(beta.x, max(beta.y, beta.z)), RR_MIN, 1.0);
}

// Fill the Russian-roulette prefix products the MIS weights need.
//
// The path is the eye prefix x_1..x_{n_eye} followed by the light vertices
// y_{n_light-1}..y_0 (k = n_eye + n_light vertices after the camera). The
// direct-hit strategy passes the whole eye subpath (n_light == 0) plus the
// light-subpath throughput at the hit point.
//
// Both sweeps recompute the compensated throughputs the rolls were made
// from - those are deterministic functions of the path prefix - so the
// q's here are exactly the ones the subpath sampling rolled. That also
// covers the hypothetical rolls of strategies that would extend a
// subpath past where it actually stopped (killed by roulette, escaped, or
// depth-capped): the albedos those extensions would scatter at belong to
// the path, so their q's are computable even though the rolls never
// happened.
fn rr_prefix_products(eye: ptr<function, array<PathVertex, 8>>,
                      light_path: ptr<function, array<PathVertex, 8>>,
                      k: u32, n_eye: u32, n_light: u32, alpha_light0: vec3<f32>,
                      qe: ptr<function, array<f32, 9>>,
                      ql: ptr<function, array<f32, 9>>) {
    // Eye-side sweep: the roll at x_l governs the extension to x_{l+1}.
    var alpha = vec3<f32>(1.0);
    (*qe)[1u] = 1.0;
    for (var l = 1u; l < k; l = l + 1u) {
        // Position l of the path: an eye vertex while l <= n_eye, then
        // the light vertices in reverse order (x_{n_eye+m} = y_{n_light-m}).
        // Explicit branches, not select(): the discarded side would read
        // slots past the realized subpath.
        var alb: vec3<f32>;
        if (l <= n_eye) {
            alb = (*eye)[l - 1u].albedo;
        } else {
            alb = (*light_path)[k - l].albedo;
        }
        let beta = alpha * alb;
        let q = rr_survival(beta, l);
        (*qe)[l + 1u] = (*qe)[l] * q;
        alpha = beta / q;
    }

    // Light-side sweep: the roll at y_j governs the extension to y_{j+1}.
    // The first step has no BSDF at the emitter, hence the PI factor.
    var alpha_l = alpha_light0;
    (*ql)[0u] = 1.0;
    for (var j = 0u; j < k - min(k, 2u); j = j + 1u) {
        // y_j is a real light vertex while j < n_light, then the eye
        // vertices in reverse order (y_{n_light+m} = x_{n_eye-m}).
        var factor: vec3<f32>;
        if (j == 0u) {
            factor = vec3<f32>(PI);
        } else if (j < n_light) {
            factor = (*light_path)[j].albedo;
        } else {
            factor = (*eye)[k - j - 1u].albedo;
        }
        let beta = alpha_l * factor;
        let q = rr_survival(beta, j + 1u);
        (*ql)[j + 1u] = (*ql)[j] * q;
        alpha_l = beta / q;
    }
}

// --- Light sampling (uniform surface sample, pdf = 1/area) ---

fn sample_light(l: GpuLight, state: ptr<function, u32>) -> vec3<f32> {
    if (l.shape == 0u) {
        let z = 1.0 - 2.0 * pcg(state);
        let phi = 2.0 * PI * pcg(state);
        let r = sqrt(max(0.0, 1.0 - z * z));
        return l.center + vec3<f32>(r * cos(phi), r * sin(phi), z) * l.radius;
    }
    let u = pcg(state) * 2.0 - 1.0;
    let v = pcg(state) * 2.0 - 1.0;
    return l.center + l.tangent * (u * l.half_w) + l.bitangent * (v * l.half_h);
}

fn light_contains(l: GpuLight, p: vec3<f32>) -> bool {
    if (l.shape == 0u) {
        return abs(vlen(p - l.center) - l.radius) < 1e-3;
    }
    let local = p - l.center;
    return abs(dot(local, l.normal)) < 1e-3
        && abs(dot(local, l.tangent)) <= l.half_w + 1e-3
        && abs(dot(local, l.bitangent)) <= l.half_h + 1e-3;
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

    for (var step = 0u; step < max_depth; step = step + 1u) {
        let hit = intersect_scene(cur_origin, cur_dir);
        if (!has_hit(hit)) {
            if (eye_len == 0u) { return BACKGROUND; }
            break;
        }
        // One-sided surfaces: a back-face hit cannot scatter.
        if (dot(hit.normal, cur_dir) > 0.0) { break; }
        // select() evaluates both branches, so guard the eye[eye_len - 1u]
        // read with an explicit branch to avoid an out-of-bounds access on
        // the first step (eye_len == 0).
        var edge_g = 0.0;
        if (eye_len > 0u) {
            edge_g = g_term(eye[eye_len - 1u].p, eye[eye_len - 1u].normal, hit.p, hit.normal);
        }
        let on_light = hit.mat_type == 1u;
        eye[eye_len] = PathVertex(
            hit.p, hit.normal, hit.albedo, hit.emission, throughput, edge_g, on_light);
        eye_len = eye_len + 1u;
        if (on_light) { break; }
        // Russian roulette for the extension x_l -> x_{l+1} (l = eye_len):
        // survive with probability q and divide the throughput by q, which
        // keeps the expected contribution of the subpath unchanged.
        let beta = throughput * hit.albedo;
        let q = rr_survival(beta, eye_len);
        if (q < 1.0 && pcg(state) >= q) { break; }
        throughput = beta / q;
        let dir = cosine_hemisphere(hit.normal, state);
        cur_origin = hit.p + hit.normal * EPS_NEAR;
        cur_dir = dir;
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
        var ltp = light.emission * inv_light_pdf;
        light_path[0] = PathVertex(p0, n0, vec3<f32>(0.0), light.emission, ltp, 0.0, true);
        light_len = 1u;

        // The first light extension (y_0 -> y_1) is unconditional
        // (RR_START >= 2). The emitter has no BSDF, so the edge contributes
        // a factor PI.
        ltp = ltp * PI;
        cur_origin = p0 + n0 * EPS_NEAR;
        cur_dir = cosine_hemisphere(n0, state);
        // The connection strategies need k = i + j + 2 <= max_depth, so the
        // last light vertex (j = max_depth - 1) could never be connected.
        // Building one vertex fewer avoids a wasted intersection.
        let steps = max_depth - min(max_depth, 2u);
        for (var step = 0u; step < steps; step = step + 1u) {
            let hit = intersect_scene(cur_origin, cur_dir);
            if (!has_hit(hit)) { break; }
            // Pure emitters do not reflect; back faces cannot scatter.
            if (hit.mat_type == 1u || dot(hit.normal, cur_dir) > 0.0) { break; }
            let prev = light_path[light_len - 1u];
            let edge_g = g_term(prev.p, prev.normal, hit.p, hit.normal);
            light_path[light_len] = PathVertex(
                hit.p, hit.normal, hit.albedo, hit.emission, ltp, edge_g, false);
            light_len = light_len + 1u;
            // Russian roulette for the extension y_j -> y_{j+1}
            // (j = light_len - 1 >= 1, number j + 1). The factor is the
            // vertex's own albedo; the emitter's PI case is applied before
            // the loop. Surviving divides ltp by q, which also applies the
            // factor for the next vertex.
            let j = light_len - 1u;
            let beta = ltp * hit.albedo;
            let q = rr_survival(beta, j + 1u);
            if (q < 1.0 && pcg(state) >= q) { break; }
            ltp = beta / q;
            let dir = cosine_hemisphere(hit.normal, state);
            cur_origin = hit.p + hit.normal * EPS_NEAR;
            cur_dir = dir;
        }
    }

    // ---- Strategy "eye subpath hits a light" ----
    if (eye_len > 0u && eye[eye_len - 1u].on_light) {
        let last = eye[eye_len - 1u];
        let k = eye_len;
        var w = 1.0;
        if (uniforms.n_lights > 0u) {
            var found = false;
            for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
                if (!found && light_contains(lights[li], last.p)) {
                    found = true;
                    var edges = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    for (var idx = 0u; idx < k - 1u; idx = idx + 1u) {
                        edges[idx] = eye[idx + 1u].edge_g;
                    }
                    // The hypothetical light subpath of the other strategies
                    // starts at this hit point with the emitter's throughput
                    // (emission * n * A, the inverse light-sampling pdf).
                    var qe = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    var ql = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    rr_prefix_products(&eye, &light_path, k, k, 0u,
                                       lights[li].emission * (f32(uniforms.n_lights) * light_area(lights[li])),
                                       &qe, &ql);
                    w = mis_weight(&edges, &qe, &ql, k,
                                   f32(uniforms.n_lights) * light_area(lights[li]), k);
                }
            }
        }
        total += last.throughput * last.emission * w;
    }

    // ---- Connection strategies: eye vertex x_{i+1} to light vertex y_j ----
    var edges = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var i = 0u; i < eye_len; i = i + 1u) {
        if (eye[i].on_light) { break; }
        for (var j = 0u; j < light_len; j = j + 1u) {
            let k = i + j + 2u; // vertices after the camera
            if (k > max_depth) { break; }
            let g_conn = g_term(eye[i].p, eye[i].normal, light_path[j].p, light_path[j].normal);
            if (g_conn <= 0.0) { continue; }
            if (!visible(eye[i].p, eye[i].normal, light_path[j].p)) { continue; }

            // Contribution: alpha_eye * f(x) * G * f(y) * alpha_light.
            // y_0 is the emitter itself, so no BSDF applies there.
            let f_light = select(light_path[j].albedo * (1.0 / PI), vec3<f32>(1.0), j == 0u);
            let contrib = eye[i].throughput
                * (eye[i].albedo * (1.0 / PI))
                * g_conn
                * f_light
                * light_path[j].throughput;

            // Path edges e_2..e_k: eye edges, the connection, light edges.
            for (var m = 2u; m <= i + 1u; m = m + 1u) {
                edges[m - 2u] = eye[m - 1u].edge_g;
            }
            edges[i] = g_conn;
            for (var m = i + 3u; m <= k; m = m + 1u) {
                edges[m - 2u] = light_path[k - m + 1u].edge_g;
            }

            var qe = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            var ql = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            rr_prefix_products(&eye, &light_path, k, i + 1u, j + 1u,
                               light_path[0].throughput, &qe, &ql);
            let w = mis_weight(&edges, &qe, &ql, k, inv_light_pdf, i + 1u);
            total += contrib * w;
        }
    }

    return total;
}

fn light_area(l: GpuLight) -> f32 {
    if (l.shape == 0u) { return 4.0 * PI * l.radius * l.radius; }
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
        color += bdpt_trace(uniforms.cam_origin, dir, &state);
    }
    color = color / f32(uniforms.spp);

    // Temporal accumulation: running average weighted by sample counts, so
    // mixing batch sizes (e.g. 1 spp preview + 8 spp batches) stays unbiased.
    let prev = accum[pixel];
    let n = f32(uniforms.samples);
    let blended = select(color, (prev.xyz * n + color * f32(uniforms.spp)) / (n + f32(uniforms.spp)), uniforms.reset == 0u);
    accum[pixel] = vec4<f32>(blended, 1.0);
}