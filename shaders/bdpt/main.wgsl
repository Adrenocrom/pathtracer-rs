// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the compute entry point — the jittered camera rays, the
// per-sample BDPT loop, and the temporal accumulation with outlier
// rejection.

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
        // The per-sample firefly clamp is applied *inside* bdpt_trace,
        // at the drawn hero wavelength (the bound is a spectral
        // quantity — paper §9.2 — computed per-λ from the active
        // emitters' spectra and the λ sample; see spectral_clamp_bound).
        color += bdpt_trace(uniforms.cam_origin, dir, &state);
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
    // EMA of the raw luminance², per-sample rate: a batch of m samples
    // moves the statistic by 1 − (1 − 1/window)^m, so a large batch (the
    // 32-spp export) can never *replace* the statistic — the per-batch
    // form m/window clamps to 1 there, and a batch whose spot pixel drew
    // zero bridge directions (probability (1 − p)³² ≈ 19% at
    // BRIDGE_PROB = 0.05) would collapse σ to ~0 and winsorize the next
    // bright batch down to μ, erasing the laser's spot.
    let alpha = 1.0 - pow(1.0 - 1.0 / MOMENT_WINDOW, m);
    let raw_sq = luminance(color) * luminance(color);
    let blended_sq = select(raw_sq, prev.w * (1.0 - alpha) + raw_sq * alpha, uniforms.reset == 0u);
    accum[pixel] = vec4<f32>(blended, blended_sq);
}
