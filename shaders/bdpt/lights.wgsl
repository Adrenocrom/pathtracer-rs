// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: light sampling — the uniform surface sample (pdf = 1/area)
// and the light-subpath starting direction.

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
// in the laser's cone (pdf 1/(2*pi*versine)), cosine-weighted around
// the normal otherwise. The CPU's AreaLight::sample returns this as the
// second tuple element; WGSL returns it separately because the cone
// sampling needs its own random numbers. The cone's versine rides in
// `half_w` (computed in f64 on the CPU — at a laser's divergence the
// f32 cosine rounds to 1.0 and the cone would collapse; see
// bdpt.rs::Bsdf::Laser).
fn sample_light_dir(l: GpuLight, p: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    if (l.shape == 2u) {
        let f = frame(l.normal);
        let v = pcg(state) * l.half_w;
        let cos_phi = 1.0 - v;
        let sin_phi = sqrt(max(0.0, 2.0 * v - v * v));
        let phi = 2.0 * PI * pcg(state);
        return f[0] * (sin_phi * cos(phi)) + f[1] * (sin_phi * sin(phi)) + l.normal * cos_phi;
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
