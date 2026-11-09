// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the generic sampling helpers — the cosine-hemisphere draw,
// the geometric term, and the shadow-ray visibility test.

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
