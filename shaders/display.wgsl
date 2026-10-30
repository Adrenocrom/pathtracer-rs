// Display pass for the windowed frontend (src/window.rs).
//
// Reads the BDPT accumulation buffer (linear HDR, rgba32float, row-major
// top-to-bottom — the same layout the compute shader writes) and applies
// the same post-processing as the terminal path (src/output.rs): optional
// bilateral denoise (radius 1, sigma_r 0.15) and Reinhard tone mapping.
//
// Gamma: when the surface format is an sRGB variant the hardware applies
// the transfer curve on store, so the shader outputs linear values. When
// the surface is linear the shader applies the same gamma 2.2 curve as
// the PNG export. `apply_gamma` selects between the two.

struct DisplayUniforms {
    width: u32,
    height: u32,
    denoise: u32,
    apply_gamma: u32,
    // Overlay: 0 = hidden, 1 = visible. The overlay texture is bound at
    // binding 2; its size comes from overlay_w/overlay_h.
    overlay: u32,
    overlay_w: u32,
    overlay_h: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> du: DisplayUniforms;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var overlay_tex: texture_2d<f32>;

const SIGMA_R: f32 = 0.15;

// Edge-preserving bilateral denoise, identical to `bilateral` in output.rs.
fn bilateral(px: u32, py: u32) -> vec3<f32> {
    let center = accum[py * du.width + px].xyz;
    let inv_two_sigma_sq = 1.0 / (2.0 * SIGMA_R * SIGMA_R);
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx = i32(px) + dx;
            let ny = i32(py) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(du.width) || ny >= i32(du.height)) {
                continue;
            }
            let neighbor = accum[u32(ny) * du.width + u32(nx)].xyz;
            let d = neighbor - center;
            let w = exp(-dot(d, d) * inv_two_sigma_sq);
            sum = sum + neighbor * w;
            weight_sum = weight_sum + w;
        }
    }

    if (weight_sum > 0.0) {
        return sum / weight_sum;
    }
    return center;
}

// Reinhard local operator, identical to `PixelBuffer::tone_map`'s first half.
fn reinhard(c: vec3<f32>) -> vec3<f32> {
    return c / (vec3<f32>(1.0) + c);
}

// Extended Reinhard with white point W (matches src/output.rs): maps W to
// exactly 1.0 so bright emitters roll off to pure white instead of
// plateauing below it.
const TONE_MAP_WHITE_POINT: f32 = 10.0;
fn reinhard_extended(c: vec3<f32>) -> vec3<f32> {
    let w = TONE_MAP_WHITE_POINT;
    return c * (vec3<f32>(1.0) + c / (w * w)) / (vec3<f32>(1.0) + c);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle covering the whole viewport.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Framebuffer coords: origin top-left, matching the accum layout.
    let px = u32(frag_pos.x);
    let py = u32(frag_pos.y);

    var color = accum[py * du.width + px].xyz;
    if (du.denoise == 1u) {
        color = bilateral(px, py);
    }

    var mapped = reinhard_extended(color);
    if (du.apply_gamma == 1u) {
        mapped = pow(mapped, vec3<f32>(1.0 / 2.2));
    }

    // Composite the overlay (status line / help panel) on top. The
    // texture is white text on transparent; its alpha is the coverage.
    // The blend happens after tone mapping + gamma so the text is crisp
    // regardless of the underlying image brightness.
    if (du.overlay == 1u) {
        if (px < du.overlay_w && py < du.overlay_h) {
            let o = textureLoad(overlay_tex, vec2<i32>(i32(px), i32(py)), 0);
            mapped = mix(mapped, o.xyz, o.a);
        }
    }

    return vec4<f32>(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}