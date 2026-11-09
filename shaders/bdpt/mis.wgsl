// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the MIS weight (Veach's incremental-ratio walk, power
// heuristic) and the Russian-roulette prefix products that feed it.

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
// throughput is `beta` (a spectral scalar at the hero wavelength), on
// extension number `ext` (1-based: the extension x_l -> x_{l+1} is
// number l, y_j -> y_{j+1} is number j+1).
fn rr_survival(beta: f32, ext: u32) -> f32 {
    if (ext < RR_START) { return 1.0; }
    return clamp(beta, RR_MIN, 1.0);
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
                      view: PathView, alpha_light0: f32,
                      qe: ptr<function, array<f32, 9>>,
                      ql: ptr<function, array<f32, 9>>) {
    let k = view.k;

    // Eye-side sweep: the roll at x_l governs the extension to x_{l+1},
    // whose factor is the sample weight of the event at x_l toward
    // x_{l+1}. The eye subpath carries radiance.
    var alpha = 1.0;
    (*qe)[1u] = 1.0;
    for (var l = 1u; l < k; l = l + 1u) {
        let factor = pv_event_weight(eye, light_path, view, l, l - 1u, l + 1u,
                                      TRANSPORT_RADIANCE, WALK_EYE);
        let beta = alpha * factor;
        let q = rr_survival(beta, l);
        (*qe)[l + 1u] = (*qe)[l] * q;
        alpha = beta * (1.0 / q);
    }

    // Light-side sweep: the roll at y_j governs the extension to y_{j+1},
    // whose factor is the sample weight of the event at y_j = x_{k-j}
    // toward y_{j+1} = x_{k-j-1}. The first extension has no BSDF at the
    // emitter (f = 1): cosine sampling gives the weight PI, a laser
    // aperture's uniform cone sampling gives cos0 * 2*PI*versine
    // (0 outside the beam). The light subpath carries importance.
    var alpha_l = alpha_light0;
    (*ql)[0u] = 1.0;
    for (var j = 0u; j < k - min(k, 2u); j = j + 1u) {
        var factor: f32;
        if (j == 0u) {
            // The emitter event at x_k = y_0, scattering toward x_{k-1}.
            let v = pv_vertex(eye, light_path, view, k);
            let d_out = pv_dir(eye, light_path, view, k, k - 1u);
            let cos0 = dot(v.normal, d_out);
            if (v.mat_type == MAT_LASER) {
                // The versine gate (see pv_pdf_light).
                let versine = v.ior;
                let dx = d_out.x - v.normal.x;
                let dy = d_out.y - v.normal.y;
                let dz = d_out.z - v.normal.z;
                if (dx * dx + dy * dy + dz * dz <= 2.0 * versine) {
                    factor = cos0 * 2.0 * PI * versine;
                } else {
                    factor = 0.0;
                }
            } else {
                factor = PI;
            }
        } else {
            factor = pv_event_weight(eye, light_path, view, k - j, k - j + 1u, k - j - 1u,
                                     TRANSPORT_IMPORTANCE, WALK_LIGHT);
        }
        let beta = alpha_l * factor;
        let q = rr_survival(beta, j + 1u);
        (*ql)[j + 1u] = (*ql)[j] * q;
        alpha_l = beta * (1.0 / q);
    }
}
