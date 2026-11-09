// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the BDPT walk itself — the hero-wavelength draw, the eye
// and light subpath construction, and the strategy connection loop.

// --- BDPT ---

fn bdpt_trace(ray_origin: vec3<f32>, ray_dir: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let max_depth = min(uniforms.max_depth, MAX_PATH_VERTICES);

    // ---- The hero wavelength (paper §9.2) ----
    // Sampled once per path from the mixture over the emitters, before
    // the eye walk (glass refraction directions depend on n(λ)). The
    // light subpath below starts on an *active* emitter — one that
    // radiates at this λ — and every strategy in this trace uses the
    // same active count, so the pick pdf and the MIS machinery's
    // inv_light_pdf = n_active·A stay consistent.
    let wave = sample_wavelength(state);
    let lambda = wave.lambda;
    // The active lights: those that radiate at the drawn λ (the branch
    // rule — see the header comment).
    var n_active = 0u;
    for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
        if (light_active(lights[li], lambda, wave.line_branch)) { n_active = n_active + 1u; }
    }
    // The per-sample clamp bound at this λ (0 = disabled).
    let clamp_bound = spectral_clamp_bound(wave);

    var total = 0.0;

    // ---- Eye subpath: x_1 .. x_s ----
    var eye: array<PathVertex, 8>;
    var eye_len = 0u;
    var throughput = 1.0;
    var cur_origin = ray_origin;
    var cur_dir = ray_dir;

    for (var bounce = 0u; bounce < max_depth; bounce = bounce + 1u) {
        let hit = intersect_scene(cur_origin, cur_dir);
        if (!has_hit(hit)) {
            if (eye_len == 0u) {
                // The background is a smooth (incoherent) spectrum: it
                // radiates only on the smooth branch. Its spectral
                // radiance at λ is the CMF-basis SPD value of the RGB
                // intent; the 1/p(λ) sensor weighting is applied at the
                // conversion below.
                var bg = 0.0;
                if (!wave.line_branch) {
                    // The CMF-basis coefficients of the background's
                    // intent, evaluated at λ. The coefficients are
                    // G⁻¹·XYZ(BACKGROUND); the CPU computes them once —
                    // here they are inlined as constants (the same
                    // values spectrum.rs's background_coeffs produces,
                    // pinned by a test).
                    let coeffs = vec3<f32>(5.4379e-5, 2.1023e-4, 3.3559e-4);
                    bg = dot(cmf(lambda), coeffs);
                }
                return clamp_sample(sensor_rgb(lambda, bg, wave.inv_pdf), clamp_bound);
            }
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
        eye[eye_len] = PathVertex(hit.p, normal, front_face, hit.albedo,
                                  hit.mat_type, hit.ior, hit.sellmeier, hit.rough,
                                  hit.f0, throughput, on_light);
        eye_len = eye_len + 1u;
        if (on_light) { break; }
        // Sample the BSDF, then roll the roulette on the post-scatter
        // throughput: survive with probability q and divide by q, which
        // keeps the expected contribution of the subpath unchanged.
        // At a diffuse vertex the eye walk uses the cosine–bridge
        // mixture (see sample_diffuse_mixture): with probability
        // BRIDGE_PROB it aims at a laser aperture's mirror image, so
        // the beam's path camera → wall → mirror → aperture is carried
        // by ordinary subpath walking.
        var s = bsdf_sample(hit.albedo, hit.mat_type, hit.ior, hit.sellmeier,
                            hit.rough, hit.f0, normal, front_face,
                            cur_dir, TRANSPORT_RADIANCE, lambda, state);
        if (hit.mat_type == MAT_DIFFUSE) {
            s = sample_diffuse_mixture(hit.albedo, hit.p, normal, cur_dir, lambda, state);
        }
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
    // Starts on an *active* emitter (one that radiates at the drawn λ):
    // the pick pdf is 1/n_active and the MIS machinery's inv_light_pdf
    // is n_active·A — consistent for every strategy in this trace.
    var light_path: array<PathVertex, 8>;
    var light_len = 0u;
    var inv_light_pdf = 0.0;
    if (n_active > 0u) {
        // Pick uniformly among the active lights.
        let pick = min(u32(f32(n_active) * pcg(state)), n_active - 1u);
        var light = lights[0];
        var seen = 0u;
        for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
            let l = lights[li];
            if (light_active(l, lambda, wave.line_branch)) {
                if (seen == pick) { light = l; }
                seen = seen + 1u;
            }
        }
        inv_light_pdf = f32(n_active) * light_area(light);
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
        // cos0 * 2*PI*versine — and the direction must lie inside the
        // beam, where the emitter's radiance is nonzero.
        let cos0 = dot(n0, d0);
        var edge_factor = PI;
        if (is_laser) {
            // The cone's versine rides in `half_w` (see
            // `sample_light_dir`). The gate is the versine-native form
            // (see pv_pdf_light): exact under the cone sampler's f32
            // quantization, where 1 - dot would lose the tiny cones.
            let versine = light.half_w;
            let dx = d0.x - n0.x;
            let dy = d0.y - n0.y;
            let dz = d0.z - n0.z;
            if (dx * dx + dy * dy + dz * dz <= 2.0 * versine) {
                edge_factor = cos0 * 2.0 * PI * versine;
            } else {
                edge_factor = 0.0;
            }
        }
        // The spectral radiance at the drawn λ (the branch rule gates
        // it: only the active emitters radiate, and the pick above
        // already filtered on exactly that).
        let le = light_eval(light, lambda, wave.line_branch);
        var ltp = le * inv_light_pdf;
        // The y_0 vertex material carries the emission cone so the MIS
        // pdf machinery (pv_pdf_light at m = k) can see it.
        let y0_mat = select(MAT_EMISSIVE, MAT_LASER, is_laser);
        let y0_ior = select(1.0, light.half_w, is_laser);
        light_path[0] = PathVertex(p0, n0, true, vec3<f32>(0.0),
                                   y0_mat, y0_ior, array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                   0.0, 0.0, ltp, true);
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
                                                   hit.mat_type, hit.ior, hit.sellmeier,
                                                   hit.rough, hit.f0, ltp, false);
                light_len = light_len + 1u;
                // Russian roulette for the extension y_j -> y_{j+1}
                // (j = light_len - 1u >= 1, number j + 1). Surviving divides
                // ltp by q, which also applies the factor for the next vertex.
                let j = light_len - 1u;
                let s = bsdf_sample(hit.albedo, hit.mat_type, hit.ior, hit.sellmeier,
                                    hit.rough, hit.f0, normal, front_face,
                                    cur_dir, TRANSPORT_IMPORTANCE, lambda, state);
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
        // light surface (the dark curved side of the laser's aperture
        // puck) carries no emission. A found light overwrites this.
        var contribution = 0.0;
        if (uniforms.n_lights > 0u) {
            var found = false;
            for (var li = 0u; li < uniforms.n_lights; li = li + 1u) {
                let l = lights[li];
                if (!found && light_contains(l, last.p)) {
                    found = true;
                    // A laser aperture radiates only into its cone: a hit
                    // outside the beam (the dark side of the housing, or
                    // the aperture seen from behind) carries no emission.
                    // The eye vertex's stored normal is the aperture's
                    // outward normal; the incoming direction is the
                    // direction of the *last* segment — the segment that
                    // actually arrives at the aperture (the primary ray
                    // for a direct hit, the wall -> mirror segment for a
                    // bridged path).
                    var d_in = ray_dir;
                    if (k >= 2u) {
                        d_in = normalize3(last.p - eye[k - 2u].p);
                    }
                    var in_beam = true;
                    if (last.mat_type == MAT_LASER) {
                        // The versine gate: |d - n|^2 = 2(1 - d.n) <=
                        // 2*versine for unit vectors — Sterbenz-exact
                        // where 1 - dot cancels catastrophically at the
                        // tiny physical divergences.
                        let versine = last.ior;
                        let u = d_in * -1.0;
                        let dx = u.x - last.normal.x;
                        let dy = u.y - last.normal.y;
                        let dz = u.z - last.normal.z;
                        in_beam = dx * dx + dy * dy + dz * dz <= 2.0 * versine;
                    }
                    // The branch rule: only an emitter active at the
                    // drawn λ radiates. Nested ifs, not `&&` (see
                    // light_eval — the NVIDIA 615.78.08 compiler bug).
                    let active_here = light_active(l, lambda, wave.line_branch);
                    if (in_beam) {
                        if (active_here) {
                            let inv_pdf = f32(n_active) * light_area(l);
                            let view = PathView(k, k, inv_pdf, ray_dir, lambda);
                            var qe = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                            var ql = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                            // The hypothetical light subpath of the other
                            // strategies starts at this hit point with the
                            // emitter's throughput (emission(λ) * n * A, the
                            // inverse light-sampling pdf).
                            rr_prefix_products(&eye, &light_path, view,
                                               light_eval(l, lambda, wave.line_branch) * inv_pdf,
                                               &qe, &ql);
                            let w = mis_weight(&eye, &light_path, view, &qe, &ql, k);
                            contribution = last.throughput
                                * light_eval(l, lambda, wave.line_branch) * w;
                        }
                    }
                }
            }
        }
        total += contribution;
    }

    // ---- Connection strategies: eye vertex x_{i+1} to light vertex y_j ----
    for (var i = 0u; i < eye_len; i = i + 1u) {
        if (eye[i].on_light) { break; } // only the last eye vertex can be emissive
        for (var j = 0u; j < light_len; j = j + 1u) {
            let k = i + j + 2u; // vertices after the camera
            if (k > max_depth) { break; }
            // Delta BSDFs cannot be connection endpoints: their eval is
            // 0, so the contribution would be 0 anyway. (Glossy
            // vertices are *not* delta — §9.5 — so connections through
            // them are valid and carried by the generic eval path.)
            if (is_delta_type(eye[i].mat_type) || is_delta_type(light_path[j].mat_type)) {
                continue;
            }
            let view = PathView(k, i + 1u, inv_light_pdf, ray_dir, lambda);

            // Contribution: alpha_eye * f(x) * G * f(y) * alpha_light.
            // The path arrives at x_{i+1} from x_i (the camera for i = 0)
            // and at y_j from x_{i+1}, leaving toward y_{j-1}.
            let d_conn = pv_dir(&eye, &light_path, view, i + 1u, i + 2u);
            var d_in_eye = ray_dir;
            if (i > 0u) {
                d_in_eye = pv_dir(&eye, &light_path, view, i, i + 1u);
            }
            let o_eye = pv_oriented(&eye, &light_path, view, i + 1u, d_in_eye);
            let e_eye = eval_bsdf(eye[i].albedo, eye[i].mat_type, eye[i].rough, eye[i].f0,
                                  o_eye.normal, d_in_eye, d_conn, lambda);
            var f_light = 1.0;
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
                    // The versine gate (see the direct-hit site above):
                    // the emission direction is y_0 -> x_{i+1}, i.e.
                    // -d_conn (d_conn points from the eye vertex to the
                    // light vertex); testing d_conn itself inverts the
                    // gate and zeroes every connection to the aperture.
                    let versine = y0.ior;
                    let u = d_conn * -1.0;
                    let dx = u.x - y0.normal.x;
                    let dy = u.y - y0.normal.y;
                    let dz = u.z - y0.normal.z;
                    if (dx * dx + dy * dy + dz * dz > 2.0 * versine) {
                        f_light = 0.0;
                    }
                }
            }
            if (j > 0u) {
                let d_in_light = d_conn * -1.0;
                let d_out_light = pv_dir(&eye, &light_path, view, i + 2u, i + 3u);
                let o_light = pv_oriented(&eye, &light_path, view, i + 2u, d_in_light);
                f_light = eval_bsdf(light_path[j].albedo, light_path[j].mat_type,
                                    light_path[j].rough, light_path[j].f0,
                                    o_light.normal, d_in_light, d_out_light, lambda).f;
                n_light_end = o_light.normal;
            }
            let g_conn = g_term(eye[i].p, o_eye.normal, light_path[j].p, n_light_end);
            if (g_conn <= 0.0 || e_eye.f == 0.0 || f_light == 0.0) {
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

    // ---- The sensor conversion (paper §9.2) ----
    // The spectral estimate at the hero wavelength, weighted by 1/p(λ)
    // and converted to linear sRGB through the CIE CMFs. Out-of-gamut
    // samples (a pure 532 nm green) carry negative channels by design;
    // the display stage gamut-maps them.
    return clamp_sample(sensor_rgb(lambda, total, wave.inv_pdf), clamp_bound);
}
