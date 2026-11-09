// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the per-sample spectral firefly clamp bound and the
// firefly rejection machinery (the per-sample luminance clamp and the
// temporal outlier rejection's constants) — placed before the BDPT
// walk, which calls clamp_sample on every returned sample.

// The per-sample firefly clamp bound at the drawn wavelength (paper
// §9.2): 8× the brightest *active* emitter's direct-view sample
// luminance at λ — the faithful generalization of the old
// `8·max emission luminance`. The max is clamped at 0 (a smooth
// CMF-basis spectrum can evaluate negative at some wavelengths; a
// negative bound would disable the clamp). 0 when no active emitter
// radiates positively (disabled).
fn spectral_clamp_bound(wave: WaveSample) -> f32 {
    // No light_active filter here: light_eval already returns 0 for
    // inactive lights (the branch rule), and max(max_le, 0) is a no-op
    // since max_le starts at 0 — the filter is redundant. Dropping it
    // also keeps the loop body's dynamic-comparison count low: the
    // NVIDIA 615.78.08 shader compiler crashes (SIGILL in
    // libnvidia-gpucomp, the gc_driver_f32_compare_constant bug class)
    // when a loop body carries enough dynamic bools for the vectorizer
    // to pack them into an <8 x i4> (light_active + light_eval inlined
    // together cross that threshold; light_eval alone does not).
    var max_le = 0.0;
    for (var i = 0u; i < uniforms.n_lights; i = i + 1u) {
        let l = lights[i];
        max_le = max(max_le, light_eval(l, wave.lambda, wave.line_branch));
    }
    if (max_le <= 0.0) { return 0.0; }
    return uniforms.firefly_clamp * cmf(wave.lambda).y * max_le * wave.inv_pdf;
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
// `bound` is the *spectral* clamp bound at the drawn hero wavelength
// (see spectral_clamp_bound): a loose multiple of the brightest active
// emitter's direct-view sample luminance at λ — loose because a rare
// legitimate caustic sample is per-sample indistinguishable from a
// firefly (both are huge relative to the pixel mean; the caustic's
// estimator is a rare-event one), so only the physically implausible
// tail is clamped here and the mean-aware temporal rejection handles
// the rest. 0 disables the clamp.
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
