// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the specular bridges — the eye walk's cosine–bridge
// diffuse sampling mixture that aims at a laser's mirror-image rungs.

// --- Specular bridges (the eye walk's diffuse sampling mixture) ---
//
// A specular bridge is the aperture's mirror image through a *prefix*
// of the beam's mirror chain — a disc. Each laser carries a *ladder*:
// one rung per mirror the beam crosses, in beam order. Rung j is the
// aperture's image unfolded across the first j mirrors, so it is the
// analytic description of the specular path "diffuse vertex → mirror₁
// → … → mirrorⱼ → aperture" — exactly what a diffuse vertex beside
// the beam's (j+1)-th leg sees. At a diffuse eye vertex the walk
// draws a direction toward one of the qualifying rungs with
// probability BRIDGE_PROB, so the glow appears beside *every* leg of
// the beam, not just the last. The ladders are derived on the CPU per
// frame from the current geometry (`SceneData::bridges_of_light`), so
// a moving scene's rungs follow the mirrors. See bdpt.rs::Bridge for
// the full story.

const BRIDGE_PROB: f32 = 0.25;

// The solid-angle pdf of the bridge direction `dir` from `p`: uniform
// over the *binding* cone — the smaller of the image-disc cone
// (half-angle asin(r/D) around the aim axis) and the gate-mapped
// emission cone (half-angle theta around -beam_dir; see
// bdpt.rs::Bridge::cone) — about its own axis. pdf 1/(2*pi*versine).
// 0 when `p` lies within the image sphere (degenerate) or `dir`
// points outside the cone. The gate is the versine-native form
// |dir - axis|^2 <= 2*versine (unit vectors): the component
// differences are Sterbenz-exact where 1 - dot cancels
// catastrophically at the tiny physical divergences.
//
// The disc cone's versine is computed in the cancellation-free form
// x/(1 + sqrt(1-x)) with x = (r/D)^2 — the naive 1 - sqrt(1-x) loses
// half its significant bits at the small ratios, and this must match
// the CPU arithmetic bit-for-bit (the 1:1 port).
fn bridge_pdf(center: vec3<f32>, radius: f32, beam_dir: vec3<f32>, versine: f32,
              p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let d = center - p;
    let len_sq = dot(d, d);
    if (len_sq <= radius * radius) { return 0.0; }
    let len = sqrt(len_sq);
    let aim = d * (1.0 / len);
    let x = radius * radius / len_sq;
    let disc_versine = x / (1.0 + sqrt(max(0.0, 1.0 - x)));
    let has_beam = dot(beam_dir, beam_dir) > 0.5;
    var axis = aim;
    var v = disc_versine;
    if (versine < disc_versine && has_beam) {
        axis = beam_dir * -1.0;
        v = versine;
    }
    let dx = dir.x - axis.x;
    let dy = dir.y - axis.y;
    let dz = dir.z - axis.z;
    if (dx * dx + dy * dy + dz * dz > 2.0 * v) { return 0.0; }
    return 1.0 / (2.0 * PI * v);
}

// Sample a direction uniformly over the cone `bridge_cone` describes.
// Returns vec3(0) when `p` lies within the image sphere (degenerate;
// the caller falls back to the cosine draw).
fn bridge_sample_dir(center: vec3<f32>, radius: f32, beam_dir: vec3<f32>, versine: f32,
                     p: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let d = center - p;
    let len_sq = dot(d, d);
    if (len_sq <= radius * radius) { return vec3<f32>(0.0); }
    let len = sqrt(len_sq);
    let aim = d * (1.0 / len);
    let x = radius * radius / len_sq;
    let disc_versine = x / (1.0 + sqrt(max(0.0, 1.0 - x)));
    let has_beam = dot(beam_dir, beam_dir) > 0.5;
    var axis = aim;
    var v = disc_versine;
    if (versine < disc_versine && has_beam) {
        axis = beam_dir * -1.0;
        v = versine;
    }
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.x) > 0.9);
    let t = normalize3(cross(a, axis));
    let b = cross(axis, t);
    // Uniform in the cone: the versine of the drawn angle is uniform
    // in [0, versine] (the versine is the natural measure — cos phi
    // uniform in [cos theta, 1] is exactly this). The sine is
    // computed from the versine directly (sin^2 = 2v - v^2), never
    // from the rounded 1 - v.
    let vv = pcg(state) * v;
    let cos_phi = 1.0 - vv;
    let sin_phi = sqrt(max(0.0, 2.0 * vv - vv * vv));
    let phi = 2.0 * PI * pcg(state);
    return t * (sin_phi * cos(phi)) + b * (sin_phi * sin(phi)) + axis * cos_phi;
}

// The number of bridge rungs that qualify at a diffuse vertex `p` with
// normal `normal`: the image disc's axis lies in the vertex's
// hemisphere and the cone is non-degenerate. A laser with a ladder
// contributes every rung that qualifies there.
fn qualifying_bridge_count(p: vec3<f32>, normal: vec3<f32>) -> u32 {
    var count = 0u;
    for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
        let l = lights[li];
        for (var bi = 0u; bi < l.n_bridges; bi = bi + 1u) {
            let d = l.bridge_rungs[2u * bi].xyz - p;
            let r = l.bridge_rungs[2u * bi].w;
            if (dot(d, normal) > 0.0 && dot(d, d) > r * r) {
                count = count + 1u;
            }
        }
    }
    return count;
}

// The eye walk's diffuse sampling density at a diffuse vertex: the
// cosine–bridge mixture. Must match sample_diffuse_mixture exactly —
// realized and hypothetical eye extensions both query it.
fn mixture_pdf(p: vec3<f32>, normal: vec3<f32>, dir_in: vec3<f32>, dir_out: vec3<f32>) -> f32 {
    let cos_i = -dot(normal, dir_in);
    let cos_o = dot(normal, dir_out);
    if (cos_i <= 0.0 || cos_o <= 0.0) { return 0.0; }
    let n_qual = qualifying_bridge_count(p, normal);
    if (n_qual == 0u) { return cos_o / PI; }
    var bridge_density = 0.0;
    for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
        let l = lights[li];
        for (var bi = 0u; bi < l.n_bridges; bi = bi + 1u) {
            let d = l.bridge_rungs[2u * bi].xyz - p;
            let r = l.bridge_rungs[2u * bi].w;
            if (dot(d, normal) > 0.0 && dot(d, d) > r * r) {
                let beam_dir = l.bridge_rungs[2u * bi + 1u].xyz;
                let versine = l.bridge_rungs[2u * bi + 1u].w;
                bridge_density += bridge_pdf(l.bridge_rungs[2u * bi].xyz, r, beam_dir, versine, p, dir_out);
            }
        }
    }
    return (1.0 - BRIDGE_PROB) * cos_o / PI + BRIDGE_PROB * bridge_density / f32(n_qual);
}

// The eye walk's diffuse sampling: a mixture of the cosine distribution
// (weight 1 − BRIDGE_PROB) and the specular-bridge cones (weight
// BRIDGE_PROB, split uniformly over the qualifying bridges). The
// realized direction's pdf is the full mixture density — the direction
// could have been drawn by either branch — and the sample weight is
// f·cos/pdf under that density, so the estimator stays unbiased for
// every direction the mixture can propose.
fn sample_diffuse_mixture(albedo: vec3<f32>, p: vec3<f32>, normal: vec3<f32>,
                          dir_in: vec3<f32>, lambda: f32,
                          state: ptr<function, u32>) -> BsdfSample {
    let n_qual = qualifying_bridge_count(p, normal);
    var dir_out = cosine_hemisphere(normal, state);
    if (n_qual > 0u && pcg(state) < BRIDGE_PROB) {
        // Pick a qualifying rung uniformly.
        let pick = u32(f32(n_qual) * pcg(state));
        var seen = 0u;
        var chosen_light = -1i;
        var chosen_bridge = -1i;
        for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
            let l = lights[li];
            for (var bi = 0u; bi < l.n_bridges; bi = bi + 1u) {
                let d = l.bridge_rungs[2u * bi].xyz - p;
                let r = l.bridge_rungs[2u * bi].w;
                if (dot(d, normal) > 0.0 && dot(d, d) > r * r) {
                    if (seen == pick) { chosen_light = i32(li); chosen_bridge = i32(bi); }
                    seen = seen + 1u;
                }
            }
        }
        if (chosen_light >= 0) {
            let l = lights[u32(chosen_light)];
            let rung = l.bridge_rungs[2u * u32(chosen_bridge)];
            let beam_dir = l.bridge_rungs[2u * u32(chosen_bridge) + 1u].xyz;
            let versine = l.bridge_rungs[2u * u32(chosen_bridge) + 1u].w;
            let d = bridge_sample_dir(rung.xyz, rung.w, beam_dir, versine, p, state);
            // A degenerate cone (unreachable for a qualifying rung)
            // falls back to the cosine draw already made.
            if (dot(d, d) > 0.0) { dir_out = d; }
        }
    }
    let pdf = mixture_pdf(p, normal, dir_in, dir_out);
    // A bridge direction can dip below the horizon: the image disc of
    // a grazing rung lies nearly in the surface plane, so the cone can
    // be wider than the axis elevation, and a uniform draw inside it
    // can land below the horizon. The BSDF is zero there, so the
    // sample carries no light — the weight must be 0, not
    // albedo·cos/(π·0) = ±inf. (Unbiased: those directions contribute
    // nothing to the integral; the draw is merely wasted.)
    var weight = 0.0;
    if (pdf > 0.0) {
        weight = rgb_to_albedo(albedo, lambda) * dot(normal, dir_out) / (PI * pdf);
    }
    return BsdfSample(dir_out, pdf, weight);
}
