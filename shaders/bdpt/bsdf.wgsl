// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the BSDFs — sampling (explicit if chains on mat_type, no
// dynamic dispatch), the GGX microfacet lobe (paper §9.5), and the
// evaluation/specular-weight helpers the MIS machinery queries.

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
// glass: which medium is incident); `lambda` is the path's hero
// wavelength (nm) — the albedo evaluates to ρ(λ) and dispersive glass
// to n(λ). Delta BSDFs sample deterministically; their pdf_omega is 0
// (the delta has no numeric value — the MIS machinery replaces it
// symbolically).
struct BsdfSample {
    dir_out: vec3<f32>,
    pdf_omega: f32,
    // f . |cos| / pdf, the throughput multiplier of the event — a
    // *spectral* scalar at the path's hero wavelength. For delta
    // BSDFs the delta cancels and this is the deterministic specular
    // weight; the Fresnel probability cancels against the BSDF value.
    weight: f32,
};

fn bsdf_sample(albedo: vec3<f32>, mat_type: u32, ior: f32,
               sellmeier: array<f32, 6>, rough: f32, f0: f32,
               normal: vec3<f32>, front_face: bool, dir_in: vec3<f32>,
               transport: u32, lambda: f32,
               state: ptr<function, u32>) -> BsdfSample {
    let rho = rgb_to_albedo(albedo, lambda);
    if (mat_type == MAT_MIRROR) {
        return BsdfSample(reflect_dir(dir_in, normal), 0.0, rho);
    }
    if (mat_type == MAT_GLASS) {
        let n = glass_n(ior, sellmeier, lambda);
        let eta_i = select(n, 1.0, front_face);
        let eta_t = select(1.0, n, front_face);
        let cos_i = -dot(normal, dir_in);
        let f = fresnel_dielectric(cos_i, eta_i, eta_t);
        if (pcg(state) < f) {
            // Reflection. The Fresnel reflectance is both the discrete
            // choice probability and the BSDF value, so it cancels out
            // of the weight.
            return BsdfSample(reflect_dir(dir_in, normal), 0.0, rho);
        }
        // Refraction (total internal reflection cannot reach this
        // branch: it gives f = 1, so the roll reflects); the fallback
        // mirrors the CPU's defensive unwrap_or_else.
        let refr = refract_dir(dir_in, normal, eta_i, eta_t);
        let dir_out = select(reflect_dir(dir_in, normal), refr.dir, refr.valid);
        return BsdfSample(dir_out, 0.0, rho * eta_factor(transport, eta_i, eta_t));
    }
    if (mat_type == MAT_GLOSSY) {
        // The GGX microfacet lobe (paper §9.5): sample a half-vector
        // from the NDF, reflect about it. A below-horizon draw is
        // possible at high roughness and carries no light (weight 0,
        // never infinite).
        let h = ggx_sample_half_vector(rough, normal, state);
        let dir_out = dir_in - h * (2.0 * dot(dir_in, h));
        let e = ggx_eval_pdf(rough, f0, rho, normal, dir_in, dir_out);
        var weight = 0.0;
        if (e.pdf > 0.0) {
            weight = e.f * dot(normal, dir_out) / e.pdf;
        }
        return BsdfSample(dir_out, e.pdf, weight);
    }
    // Diffuse (and emissive, which never scatters — unreachable here).
    let dir_out = cosine_hemisphere(normal, state);
    return BsdfSample(dir_out, dot(dir_out, normal) / PI, rho);
}

// --- GGX microfacet BSDF (paper §9.5) ---
//
// The "rough mirror": a GGX lobe replaces the perfect mirror's delta.
// The convention matches the rest of the shader — `dir_in` is the
// *travel* direction (normal . dir_in < 0), `normal` orients the
// hemisphere — so the microfacet half-vector is
// h = normalize(−dir_in + dir_out) and the reflection about it is
// dir_out = dir_in − 2(dir_in·h)h.

// The GGX normal distribution D(h) for roughness `alpha` and
// cos θ_h = normal·h.
fn ggx_d(alpha: f32, cos_h: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = cos_h * cos_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// The Smith GGX masking term G1 (Walter et al. 2007, eq. 99).
fn ggx_g1(alpha: f32, cos_v: f32) -> f32 {
    let c = clamp(cos_v, 1e-6, 1.0);
    let a2 = alpha * alpha;
    return 2.0 * c / (c + sqrt(a2 + c * c * (1.0 - a2)));
}

// Schlick's Fresnel approximation from the normal-incidence
// reflectance `f0`.
fn schlick_f(f0: f32, cos: f32) -> f32 {
    let c = clamp(cos, 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - c, 5.0);
}

// Sample a GGX half-vector around `normal` for roughness `alpha`
// (the standard NDF inversion).
fn ggx_sample_half_vector(alpha: f32, normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let u = clamp(pcg(state), 1e-6, 1.0 - 1e-6);
    let v = pcg(state);
    let a2 = alpha * alpha;
    let cos_h = sqrt((1.0 - u) / (1.0 + (a2 - 1.0) * u));
    let sin_h = sqrt(max(0.0, 1.0 - cos_h * cos_h));
    let phi = 2.0 * PI * v;
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
    let t = normalize3(cross(a, normal));
    let b = cross(normal, t);
    return t * (sin_h * cos(phi)) + b * (sin_h * sin(phi)) + normal * cos_h;
}

// The GGX microfacet BSDF value f (no cosine) and its NDF-sampling pdf
// for the direction pair, at spectral reflectance `rho`.
struct GgxEval {
    f: f32,
    pdf: f32,
};

fn ggx_eval_pdf(alpha: f32, f0: f32, rho: f32, normal: vec3<f32>,
                dir_in: vec3<f32>, dir_out: vec3<f32>) -> GgxEval {
    let cos_i = -dot(normal, dir_in);
    let cos_o = dot(normal, dir_out);
    if (cos_i <= 0.0 || cos_o <= 0.0) { return GgxEval(0.0, 0.0); }
    let h = normalize3(dir_out - dir_in);
    let cos_h = dot(normal, h);
    if (cos_h <= 0.0) { return GgxEval(0.0, 0.0); }
    let v_dot_h = -dot(dir_in, h); // ω_i·h, positive on a reflection pair
    if (v_dot_h <= 0.0) { return GgxEval(0.0, 0.0); }
    let d = ggx_d(alpha, cos_h);
    let g = ggx_g1(alpha, cos_i) * ggx_g1(alpha, cos_o);
    let f = schlick_f(f0, v_dot_h);
    let bsdf = rho * d * g * f / (4.0 * cos_i * cos_o);
    let pdf = d * cos_h / (4.0 * v_dot_h);
    return GgxEval(bsdf, pdf);
}

// f (the raw BSDF, no cosine) and the solid-angle pdf of `dir_out`
// under the BSDF's sampling distribution, for an arbitrary direction
// pair at wavelength `lambda`. Delta BSDFs return (0, 0) — exactly what
// invalidates connections through them. Glossy (§9.5) is *not* delta:
// its lobe evaluates nonzero, which is what makes connections through
// it valid. Emissive materials carry a diffuse BSDF with albedo 0
// (same as the CPU), so they fall into the same branch.
struct BsdfEval {
    f: f32,
    pdf: f32,
};

fn eval_bsdf(albedo: vec3<f32>, mat_type: u32, rough: f32, f0: f32,
             normal: vec3<f32>, dir_in: vec3<f32>, dir_out: vec3<f32>,
             lambda: f32) -> BsdfEval {
    if (is_delta_type(mat_type)) {
        return BsdfEval(0.0, 0.0);
    }
    if (mat_type == MAT_GLOSSY) {
        let e = ggx_eval_pdf(rough, f0, rgb_to_albedo(albedo, lambda), normal, dir_in, dir_out);
        return BsdfEval(e.f, e.pdf);
    }
    let cos_i = -dot(normal, dir_in);
    let cos_o = dot(normal, dir_out);
    if (cos_i > 0.0 && cos_o > 0.0) {
        return BsdfEval(rgb_to_albedo(albedo, lambda) * (1.0 / PI), cos_o / PI);
    }
    return BsdfEval(0.0, 0.0);
}

// The deterministic sample weight of a SPECULAR direction pair at a
// delta BSDF — needed for hypothetical path extensions, where the rolls
// never happened but the path geometry fixes the directions. valid =
// false if the pair is not the specular pair. `normal`/`front_face`
// describe the event being weighted; `lambda` evaluates dispersive
// glass (the hypothetical extension uses the same hero wavelength as
// the realized walk — λ is fixed per path).
struct SpecularWeight {
    weight: f32,
    valid: bool,
};

fn specular_weight(albedo: vec3<f32>, mat_type: u32, ior: f32,
                   sellmeier: array<f32, 6>, normal: vec3<f32>,
                   front_face: bool, dir_in: vec3<f32>, dir_out: vec3<f32>,
                   transport: u32, lambda: f32) -> SpecularWeight {
    let rho = rgb_to_albedo(albedo, lambda);
    if (mat_type == MAT_MIRROR) {
        if (dot(reflect_dir(dir_in, normal), dir_out) > SPEC_DOT_EPS) {
            return SpecularWeight(rho, true);
        }
        return SpecularWeight(0.0, false);
    }
    if (mat_type == MAT_GLASS) {
        let n = glass_n(ior, sellmeier, lambda);
        let eta_i = select(n, 1.0, front_face);
        let eta_t = select(1.0, n, front_face);
        if (dot(reflect_dir(dir_in, normal), dir_out) > SPEC_DOT_EPS) {
            return SpecularWeight(rho, true);
        }
        let refr = refract_dir(dir_in, normal, eta_i, eta_t);
        if (refr.valid && dot(refr.dir, dir_out) > SPEC_DOT_EPS) {
            return SpecularWeight(rho * eta_factor(transport, eta_i, eta_t), true);
        }
        return SpecularWeight(0.0, false);
    }
    // Glossy has no specular pair (its density is continuous — the MIS
    // machinery uses `eval_bsdf` for it instead); pure emitters never
    // scatter.
    return SpecularWeight(0.0, false);
}
