// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the PathView helpers — the WGSL shape of bdpt.rs's
// PathView methods, addressing the unified path positions x_1..x_k.

// --- PathView helpers (the WGSL shape of bdpt.rs's PathView methods) ---

// The path vertex at position m (1-based).
fn pv_vertex(eye: ptr<function, array<PathVertex, 8>>,
             light_path: ptr<function, array<PathVertex, 8>>,
             view: PathView, m: u32) -> PathVertex {
    // Explicit branch, not select(): the discarded side would read slots
    // past the realized subpath.
    if (m <= view.n_eye) {
        return (*eye)[m - 1u];
    }
    return (*light_path)[view.k - m];
}

// Travel direction from position a to position b (adjacent). Position 0
// is the camera: its edge direction is the primary ray.
fn pv_dir(eye: ptr<function, array<PathVertex, 8>>,
          light_path: ptr<function, array<PathVertex, 8>>,
          view: PathView, a: u32, b: u32) -> vec3<f32> {
    if (a == 0u) { return view.cam_dir; }
    let d = pv_vertex(eye, light_path, view, b).p - pv_vertex(eye, light_path, view, a).p;
    let len = vlen(d);
    if (len < 1e-8) { return vec3<f32>(0.0); }
    return d * (1.0 / len);
}

// (normal, front_face) at position m for an event whose incoming travel
// direction is `dir_in`: the normal oriented toward the side the ray
// arrives from. The geometric normal is recovered from the stored
// orientation, then flipped when the event arrives from the other side
// (hypothetical extensions do).
struct OrientedNormal {
    normal: vec3<f32>,
    front_face: bool,
};

fn pv_oriented(eye: ptr<function, array<PathVertex, 8>>,
               light_path: ptr<function, array<PathVertex, 8>>,
               view: PathView, m: u32, dir_in: vec3<f32>) -> OrientedNormal {
    let v = pv_vertex(eye, light_path, view, m);
    let n_geo = select(v.normal * -1.0, v.normal, v.front_face);
    if (dot(n_geo, dir_in) < 0.0) {
        return OrientedNormal(n_geo, true);
    }
    return OrientedNormal(n_geo * -1.0, false);
}

// The solid-angle pdf of the scattering event at position m — arriving
// from `arrival`, scattering toward `to` (adjacent positions) — under
// the BSDF's own sampling distribution. Delta BSDFs: the symbolic 1
// when the pair is the specular pair, 0 otherwise. Diffuse events on
// the eye side use the cosine–bridge mixture density (the eye walk's
// own distribution); on the light side, plain cosine.
// WALK_EYE / WALK_LIGHT select the walk (see bdpt.rs::Walk).
const WALK_EYE: u32 = 0u;
const WALK_LIGHT: u32 = 1u;

fn pv_event_pdf(eye: ptr<function, array<PathVertex, 8>>,
                light_path: ptr<function, array<PathVertex, 8>>,
                view: PathView, m: u32, arrival: u32, to: u32, walk: u32) -> f32 {
    let v = pv_vertex(eye, light_path, view, m);
    let d_in = pv_dir(eye, light_path, view, arrival, m);
    let d_out = pv_dir(eye, light_path, view, m, to);
    let o = pv_oriented(eye, light_path, view, m, d_in);
    if (is_delta_type(v.mat_type)) {
        let sw = specular_weight(v.albedo, v.mat_type, v.ior, v.sellmeier, o.normal,
                                 o.front_face, d_in, d_out, TRANSPORT_RADIANCE,
                                 view.lambda);
        return select(0.0, 1.0, sw.valid);
    }
    if (walk == WALK_EYE && v.mat_type == MAT_DIFFUSE) {
        return mixture_pdf(v.p, o.normal, d_in, d_out);
    }
    return eval_bsdf(v.albedo, v.mat_type, v.rough, v.f0, o.normal, d_in, d_out,
                     view.lambda).pdf;
}

// The sample weight (f.|cos|/pdf) of the scattering event at position m
// — arriving from `arrival`, scattering toward `to` — under
// `transport`. Delta BSDFs return the deterministic specular weight;
// impossible events return 0. Diffuse events on the eye side use the
// cosine–bridge mixture density.
fn pv_event_weight(eye: ptr<function, array<PathVertex, 8>>,
                   light_path: ptr<function, array<PathVertex, 8>>,
                   view: PathView, m: u32, arrival: u32, to: u32,
                   transport: u32, walk: u32) -> f32 {
    let v = pv_vertex(eye, light_path, view, m);
    let d_in = pv_dir(eye, light_path, view, arrival, m);
    let d_out = pv_dir(eye, light_path, view, m, to);
    let o = pv_oriented(eye, light_path, view, m, d_in);
    if (is_delta_type(v.mat_type)) {
        let sw = specular_weight(v.albedo, v.mat_type, v.ior, v.sellmeier, o.normal,
                                 o.front_face, d_in, d_out, transport, view.lambda);
        return select(0.0, sw.weight, sw.valid);
    }
    if (walk == WALK_EYE && v.mat_type == MAT_DIFFUSE) {
        let pdf = mixture_pdf(v.p, o.normal, d_in, d_out);
        if (pdf <= 0.0) { return 0.0; }
        // f·cos/pdf with f = ρ(λ)/π (the diffuse BSDF); only the
        // sampling distribution changed, not the BSDF.
        return rgb_to_albedo(v.albedo, view.lambda) * (1.0 / PI) * (dot(o.normal, d_out) / pdf);
    }
    let e = eval_bsdf(v.albedo, v.mat_type, v.rough, v.f0, o.normal, d_in, d_out,
                      view.lambda);
    if (e.pdf <= 0.0) { return 0.0; }
    return e.f * (dot(o.normal, d_out) / e.pdf);
}

// Solid-angle to area-measure conversion for the edge (a, b): the
// cosine at the destination vertex (absolute — the projected area is
// orientation-independent) over the squared distance.
fn pv_convert_to_area(eye: ptr<function, array<PathVertex, 8>>,
                      light_path: ptr<function, array<PathVertex, 8>>,
                      view: PathView, pdf_omega: f32, from_pos: u32, to_pos: u32) -> f32 {
    let d = pv_vertex(eye, light_path, view, to_pos).p - pv_vertex(eye, light_path, view, from_pos).p;
    let r_sq = dot(d, d);
    if (r_sq < 1e-8) { return 0.0; }
    let dir = d * (1.0 / sqrt(r_sq));
    let cos_dest = abs(dot(pv_vertex(eye, light_path, view, to_pos).normal, dir * -1.0));
    return pdf_omega * cos_dest / r_sq;
}

// The area pdf of edge e_m (positions m-1, m) as sampled at position
// m-1 toward position m — the eye-direction sampling a^e(e_m). Delta
// sampling vertices contribute the symbolic 1; impossible directions
// contribute 0.
fn pv_pdf_eye(eye: ptr<function, array<PathVertex, 8>>,
              light_path: ptr<function, array<PathVertex, 8>>,
              view: PathView, m: u32) -> f32 {
    if (m < 2u || m > view.k) { return 0.0; }
    let pdf_omega = pv_event_pdf(eye, light_path, view, m - 1u, m - 2u, m, WALK_EYE);
    if (pdf_omega <= 0.0) { return 0.0; }
    return pv_convert_to_area(eye, light_path, view, pdf_omega, m - 1u, m);
}

// The area pdf of edge e_m as sampled at position m toward position
// m-1 — the light-direction sampling a^l(e_m). m = k+1 is the light
// position sample (pdf 1/(n.A)); m = k is the emitter event, which has
// no incoming direction: the emitter scatters cosine-weighted around
// its stored (emission-side) normal with f = 1 — except a laser
// aperture, whose emission is uniform in its cone (pdf
// 1/(2π(1 - cos θ)) inside the beam, 0 outside).
fn pv_pdf_light(eye: ptr<function, array<PathVertex, 8>>,
                light_path: ptr<function, array<PathVertex, 8>>,
                view: PathView, m: u32) -> f32 {
    if (m == view.k + 1u) { return 1.0 / view.inv_light_pdf; }
    if (m < 2u || m > view.k) { return 0.0; }
    var pdf_omega = 0.0;
    if (m == view.k) {
        let d_out = pv_dir(eye, light_path, view, m, m - 1u);
        let v = pv_vertex(eye, light_path, view, m);
        let cos = dot(v.normal, d_out);
        if (v.mat_type == MAT_LASER) {
            // Uniform in the cone: pdf 1/(2*pi*versine) inside, 0 out.
            // The versine rides in the prim's `ior` slot (see
            // MAT_LASER); the gate is |d - n|^2 <= 2*versine (unit
            // vectors) — the component differences are Sterbenz-exact
            // where 1 - dot cancels catastrophically at the tiny
            // physical divergences (and exactly absorbs the cone
            // sampler's f32 cos-quantization).
            let versine = v.ior;
            let dx = d_out.x - v.normal.x;
            let dy = d_out.y - v.normal.y;
            let dz = d_out.z - v.normal.z;
            if (dx * dx + dy * dy + dz * dz <= 2.0 * versine) {
                pdf_omega = 1.0 / (2.0 * PI * versine);
            }
        } else {
            if (cos > 0.0) { pdf_omega = cos / PI; }
        }
    } else {
        pdf_omega = pv_event_pdf(eye, light_path, view, m, m + 1u, m - 1u, WALK_LIGHT);
    }
    if (pdf_omega <= 0.0) { return 0.0; }
    return pv_convert_to_area(eye, light_path, view, pdf_omega, m, m - 1u);
}
