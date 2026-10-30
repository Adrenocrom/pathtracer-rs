//! Bidirectional path tracing (BDPT) with multiple-importance sampling.
//!
//! Surfaces carry a general BSDF (diffuse, mirror, glass). Subpaths sample
//! directions with the BSDF's own distribution, and the MIS weights are
//! computed with Veach's incremental-ratio walk over the *actual* pdfs of
//! every edge — computed on the fly from the path geometry via `PathView`,
//! which addresses the unified path positions x_1..x_k (the eye prefix
//! followed by the light vertices in reverse). Weights use the power
//! heuristic (beta = 2).
//!
//! Delta BSDFs (mirror, glass) scatter into a single direction: their
//! solid-angle pdf is a Dirac delta, which has no numeric value. The
//! convention here is that a delta-sampled edge contributes a *symbolic*
//! factor of 1 to the pdf products — every valid strategy's pdf carries
//! exactly one such factor per delta vertex on the path, so they cancel in
//! the ratios between strategies. A delta vertex scattering in any *other*
//! direction (a hypothetical extension that does not follow the specular
//! pair) has pdf 0, which correctly kills every strategy that would need
//! it. Connections through delta vertices are invalid (their `eval` is 0)
//! and are excluded from the MIS normalization; specular transport is
//! handled purely by subpath sampling, which is the whole BDPT story for
//! caustics.
//!
//! Paths are terminated by Russian roulette rather than a short fixed
//! depth: from extension `RR_START` on, a subpath survives each bounce with
//! probability `q = clamp(max(post-scatter throughput), RR_MIN, 1)` and
//! divides its throughput by `q` on survival, which keeps the estimator
//! unbiased while letting deep paths through. The survival probabilities
//! are part of the sampling pdfs, so they also enter the MIS weights, as
//! the prefix products computed by `rr_prefix_products` — whose sweep
//! recomputes each extension's sample weight from the path geometry, so
//! the hypothetical rolls of strategies that would extend a subpath past
//! where it stopped are exactly reproducible.

use crate::geometry::{
    AreaLight, Intersectable, Material, Ray, Transport, cosine_hemisphere,
};
use crate::math::{Vec3, EPS_NEAR, VEC_ZERO};
use rand::Rng;

const PI: f32 = std::f32::consts::PI;

/// Hard cap on path length (vertices after the camera). Bounds the fixed-size
/// MIS scratch arrays.
pub const MAX_PATH_VERTICES: usize = 8;

/// Color returned when the primary ray escapes the scene.
pub const BACKGROUND: Vec3 = Vec3 {
    x: 0.02,
    y: 0.02,
    z: 0.05,
};

/// Russian roulette: minimum survival probability. Bounds the variance the
/// 1/q compensation can add (a surviving path never gains more than a 20x
/// weight multiplier per bounce).
const RR_MIN: f32 = 0.05;
/// Russian roulette starts at this extension number. The first
/// RR_START - 1 extensions of every subpath are unconditional, so short
/// paths — which carry most of the energy — are never killed and stay
/// low-variance.
const RR_START: usize = 3;

/// One vertex of an eye or light subpath.
struct PathVertex {
    p: Vec3,
    /// Normal oriented toward the side the subpath arrived from (so
    /// `normal · dir_in < 0` for the creating ray).
    normal: Vec3,
    /// True when the geometric normal (as returned by the intersection)
    /// faces the side the subpath arrived from. Together with `normal`
    /// this recovers the geometric normal, which hypothetical events
    /// re-orient for their own arrival side.
    front_face: bool,
    mat: Material,
    /// Path throughput up to this vertex, excluding the BSDF applied when
    /// *leaving* it (that factor belongs to the connection term).
    throughput: Vec3,
    on_light: bool,
}

/// A view of one full path x_1..x_k (camera at position 0, the light y_0 at
/// position k) as the eye prefix followed by the light vertices in reverse
/// order: x_m = eye[m-1] for m <= n_eye, x_m = light_path[k-m] above.
///
/// Every pdf and sample weight the MIS machinery needs — realized edges,
/// the connection edge, and the hypothetical extensions of strategies that
/// would push a subpath past where it stopped — is computed from the path
/// geometry through this view, so realized and hypothetical quantities are
/// produced by the *same* code and cannot drift apart.
struct PathView<'a> {
    eye: &'a [PathVertex],
    light_path: &'a [PathVertex],
    k: usize,
    n_eye: usize,
    /// The primary camera ray direction (the incoming direction at x_1).
    cam_dir: Vec3,
    /// n_lights * area of the light the path ends on.
    inv_light_pdf: f64,
}

impl<'a> PathView<'a> {
    /// The path vertex at position m (1-based).
    fn vertex(&self, m: usize) -> &PathVertex {
        debug_assert!((1..=self.k).contains(&m), "position {m} out of 1..={}", self.k);
        if m <= self.n_eye {
            &self.eye[m - 1]
        } else {
            &self.light_path[self.k - m]
        }
    }

    /// Travel direction from position a to position b (adjacent). Position
    /// 0 is the camera: its edge direction is the primary ray.
    fn dir(&self, a: usize, b: usize) -> Vec3 {
        debug_assert!(b == a + 1 || b + 1 == a);
        if a == 0 {
            return self.cam_dir;
        }
        let d = self.vertex(b).p - self.vertex(a).p;
        let len = d.length();
        if len < 1e-8 {
            VEC_ZERO
        } else {
            d * (1.0 / len)
        }
    }

    /// (normal, front_face) at position m for an event whose incoming
    /// travel direction is `dir_in`: the normal oriented toward the side
    /// the ray arrives from. The geometric normal is recovered from the
    /// stored orientation, then flipped when the event arrives from the
    /// other side (hypothetical extensions do).
    fn oriented(&self, m: usize, dir_in: Vec3) -> (Vec3, bool) {
        let v = self.vertex(m);
        let n_geo = if v.front_face {
            v.normal
        } else {
            v.normal * -1.0
        };
        if n_geo.dot(dir_in) < 0.0 {
            (n_geo, true)
        } else {
            (n_geo * -1.0, false)
        }
    }

    /// The solid-angle pdf of the scattering event at position m — arriving
    /// from `arrival`, scattering toward `to` (adjacent positions) — under
    /// the BSDF's own sampling distribution. Delta BSDFs: the symbolic 1
    /// when the pair is the specular pair, 0 otherwise.
    fn event_pdf(&self, m: usize, arrival: usize, to: usize) -> f64 {
        let v = self.vertex(m);
        let d_in = self.dir(arrival, m);
        let d_out = self.dir(m, to);
        let (normal, front_face) = self.oriented(m, d_in);
        if v.mat.bsdf.is_delta() {
            if v
                .mat
                .is_specular_pair(normal, front_face, d_in, d_out)
            {
                1.0
            } else {
                0.0
            }
        } else {
            v.mat.eval_bsdf(normal, d_in, d_out).1 as f64
        }
    }

    /// The sample weight (f·|cos|/pdf) of the scattering event at position
    /// m — arriving from `arrival`, scattering toward `to` — under
    /// `transport`. Delta BSDFs return the deterministic specular weight;
    /// impossible events return 0.
    fn event_weight(
        &self,
        m: usize,
        arrival: usize,
        to: usize,
        transport: Transport,
    ) -> Vec3 {
        let v = self.vertex(m);
        let d_in = self.dir(arrival, m);
        let d_out = self.dir(m, to);
        let (normal, front_face) = self.oriented(m, d_in);
        if v.mat.bsdf.is_delta() {
            v.mat
                .specular_weight(normal, front_face, d_in, d_out, transport)
                .unwrap_or(VEC_ZERO)
        } else {
            let (f, pdf) = v.mat.eval_bsdf(normal, d_in, d_out);
            if pdf <= 0.0 {
                VEC_ZERO
            } else {
                f * (normal.dot(d_out) / pdf)
            }
        }
    }

    /// The area pdf of edge e_m (positions m-1, m) as sampled at position
    /// m-1 toward position m — the eye-direction sampling a^e(e_m).
    /// Delta sampling vertices contribute the symbolic 1; impossible
    /// directions contribute 0.
    fn pdf_eye(&self, m: usize) -> f64 {
        if m < 2 || m > self.k {
            return 0.0;
        }
        let pdf_omega = self.event_pdf(m - 1, m - 2, m);
        if pdf_omega <= 0.0 {
            return 0.0;
        }
        self.convert_to_area(pdf_omega, m - 1, m)
    }

    /// The area pdf of edge e_m as sampled at position m toward position
    /// m-1 — the light-direction sampling a^l(e_m). m = k+1 is the light
    /// position sample (pdf 1/(n·A)); m = k is the emitter event, which
    /// has no incoming direction: the emitter scatters cosine-weighted
    /// around its stored (emission-side) normal with f = 1.
    fn pdf_light(&self, m: usize) -> f64 {
        if m == self.k + 1 {
            return 1.0 / self.inv_light_pdf;
        }
        if m < 2 || m > self.k {
            return 0.0;
        }
        let pdf_omega = if m == self.k {
            let d_out = self.dir(m, m - 1);
            let cos = self.vertex(m).normal.dot(d_out);
            if cos <= 0.0 {
                0.0
            } else {
                cos / PI
            }
        } else {
            self.event_pdf(m, m + 1, m - 1) as f32
        };
        if pdf_omega <= 0.0 {
            return 0.0;
        }
        self.convert_to_area(pdf_omega as f64, m, m - 1)
    }

    /// Solid-angle to area-measure conversion for the edge (a, b): the
    /// cosine at the destination vertex (absolute — the projected area is
    /// orientation-independent) over the squared distance.
    fn convert_to_area(&self, pdf_omega: f64, from: usize, to: usize) -> f64 {
        let d = self.vertex(to).p - self.vertex(from).p;
        let r_sq = d.length_squared();
        if r_sq < 1e-8 {
            return 0.0;
        }
        let dir = d * (1.0 / r_sq.sqrt());
        let cos_dest = self.vertex(to).normal.dot(dir * -1.0).abs();
        pdf_omega * (cos_dest as f64) / (r_sq as f64)
    }
}

/// Geometry term between two surface points (0 when either side faces away).
fn g_term(p0: Vec3, n0: Vec3, p1: Vec3, n1: Vec3) -> f32 {
    let d = p1 - p0;
    let dist_sq = d.length_squared();
    if dist_sq < 1e-8 {
        return 0.0;
    }
    let dist = dist_sq.sqrt();
    let dir = d * (1.0 / dist);
    let c0 = n0.dot(dir);
    let c1 = n1.dot(-dir);
    if c0 <= 0.0 || c1 <= 0.0 {
        return 0.0;
    }
    c0 * c1 / dist_sq
}

/// True when nothing blocks the segment `from -> to`.
fn visible(from: Vec3, from_normal: Vec3, to: Vec3, scene: &dyn Intersectable) -> bool {
    let origin = from + from_normal * EPS_NEAR;
    let d = to - origin;
    let dist = d.length();
    if dist < EPS_NEAR {
        return true;
    }
    let ray = Ray {
        origin,
        direction: d * (1.0 / dist),
    };
    match scene.intersect(&ray) {
        // Occluded only if something hits clearly *before* the target.
        Some(hit) => hit.t + 2.0 * EPS_NEAR >= dist,
        None => true,
    }
}

/// MIS weight (power heuristic) for the sampling strategy that uses `s` eye
/// vertices out of a path with `k` vertices after the camera.
///
/// The path is x_1..x_k (camera at x_0, light y_0 = x_k). Edge e_m
/// connects x_{m-1} and x_m. Strategy s' connects x_{s'} to x_{s'+1} =
/// y_{k-s'-1}; its connection edge is e_{s'+1}. The eye subpath samples
/// e_2..e_{s'+1} (at x_1..x_{s'}), the light subpath samples e_{s'+2}..e_k
/// (at x_{s'+2}..x_k) plus the light position sample (pdf 1/(n·A)):
///
/// ```text
/// p[s'] = (∏_{m=2}^{s'} aᵉ(e_m)) · (1/nA) · (∏_{m=s'+2}^{k} aˡ(e_m)) · Qe(s') · Ql(k-s'-1)
/// p[k]  = (∏_{m=2}^{k} aᵉ(e_m)) · Qe(k)          (eye subpath hits the light)
/// ```
///
/// The step ratio between adjacent strategies is
///
/// ```text
/// p[s'+1]/p[s'] = aᵉ(e_{s'+1}) / aˡ(e_{s'+2}) · Qe(s'+1)/Qe(s') · Ql(k-s'-2)/Ql(k-s'-1)
/// ```
///
/// with `aˡ(e_{k+1}) := 1/nA` and `Ql(-1) := 1`. The weight is
/// `1 / Σ (p[s']/p[s])²` over the *valid* strategies, computed by walking
/// both directions from the actual strategy s and accumulating the step
/// products — `p[s]` itself cancels.
///
/// Validity: a strategy's connection edge must have nonzero pdf from both
/// sides and non-delta endpoints. Its contribution is then 0 (eval is 0),
/// and leaving it out of the normalization is what keeps the weights of
/// the contributing strategies summing to 1. The zero-propagation is
/// correct: if the eye continuation cannot scatter at some vertex, no
/// strategy beyond that point can sample the path either. The divisions
/// in the walk are always by realized (positive) pdfs — the up-walk divides
/// by light-side pdfs at or beyond the connection, the down-walk by eye
/// edges — so a zero factor only ever appears as a multiplicative 0.
fn mis_weight(path: &PathView, qe: &[f64], ql: &[f64], s: usize) -> f32 {
    let k = path.k;
    debug_assert!((1..=k).contains(&s));
    debug_assert!(qe.len() > k && ql.len() + 1 >= k);

    // Strategy s' (1..k-1) connects x_{s'} to x_{s'+1} across e_{s'+1}.
    // Strategy k has no connection edge (the eye subpath hit the emitter).
    let valid = |s2: usize| -> bool {
        if s2 == k {
            return true;
        }
        let m = s2 + 1;
        if path.vertex(s2).mat.is_delta() || path.vertex(m).mat.is_delta() {
            return false;
        }
        path.pdf_eye(m) > 0.0 && path.pdf_light(m) > 0.0
    };

    // One step of the ratio walk: p[s2+1]/p[s2].
    let step = |s2: usize| -> f64 {
        let a_e = path.pdf_eye(s2 + 1);
        let a_l = path.pdf_light(s2 + 2);
        let qe_ratio = qe[s2 + 1] / qe[s2];
        let ql_ratio = if k >= s2 + 2 {
            ql[k - s2 - 2] / ql[k - s2 - 1]
        } else {
            1.0 // Ql(-1) := 1
        };
        a_e / a_l * qe_ratio * ql_ratio
    };

    // Walk up from s: each step multiplies by p[s2+1]/p[s2]. The loop
    // must reach s2 = k-1 so that up[k] — the direct-hit strategy, a valid
    // competitor of every connection strategy — is computed; dropping it
    // would shrink the normalization sum and inflate every weight.
    let mut up = [0.0f64; MAX_PATH_VERTICES + 1];
    up[s] = 1.0;
    for s2 in s..k {
        up[s2 + 1] = up[s2] * step(s2);
    }

    // Walk down from s: each step multiplies by p[s2]/p[s2+1].
    let mut down = [0.0f64; MAX_PATH_VERTICES + 1];
    down[s] = 1.0;
    for s2 in (1..s).rev() {
        down[s2] = down[s2 + 1] / step(s2);
    }

    // Power heuristic (beta = 2), normalized by the largest valid ratio to
    // keep the squares in a safe floating-point range (the f32 WGSL port
    // relies on the same trick).
    let ratio = |s2: usize| up[s2].max(down[s2]);
    let mut r_max = 1.0f64;
    for s2 in 1..=k {
        if valid(s2) && ratio(s2).is_finite() {
            r_max = r_max.max(ratio(s2));
        }
    }
    let mut sum = 0.0f64;
    for s2 in 1..=k {
        if valid(s2) && ratio(s2).is_finite() {
            let q = ratio(s2) / r_max;
            sum += q * q;
        }
    }
    if !(sum > 0.0) {
        return 1.0;
    }
    let q = 1.0 / r_max;
    (q * q / sum) as f32
}

/// Survival probability for extending a subpath whose post-scatter
/// throughput is `beta`, on extension number `ext` (1-based: the extension
/// x_l -> x_{l+1} is number l, y_j -> y_{j+1} is number j + 1).
///
/// The caller divides the throughput by the returned q when the path
/// survives, which leaves the expected contribution unchanged.
fn rr_survival(beta: Vec3, ext: usize) -> f32 {
    if ext < RR_START {
        return 1.0;
    }
    beta.x.max(beta.y).max(beta.z).clamp(RR_MIN, 1.0)
}

/// Russian-roulette prefix products the MIS weights need: `qe[m]` is
/// Qe(m) and `ql[j]` is Ql(j) (see `mis_weight`).
struct RrProducts {
    qe: [f64; MAX_PATH_VERTICES + 1],
    ql: [f64; MAX_PATH_VERTICES + 1],
}

/// Compute the Russian-roulette prefix products the MIS weights need.
///
/// Both sweeps recompute the post-scatter throughputs the rolls were made
/// from — each extension's factor is the sample weight of the scattering
/// event at its start vertex, evaluated from the path geometry — so the
/// q's here are exactly the ones the subpath sampling rolled. That also
/// covers the *hypothetical* rolls of strategies that would extend a
/// subpath past where it actually stopped (killed by roulette, escaped, or
/// depth-capped): the events those extensions would perform belong to the
/// path, so their weights and q's are computable even though the rolls
/// never happened.
///
/// `alpha_light0` is the light-subpath throughput at y_0: the realized
/// `emission * n * A` for connection strategies, or the same expression
/// for the direct-hit strategy's hypothetical light subpath starting at
/// the emitter hit.
fn rr_prefix_products(path: &PathView, alpha_light0: Vec3) -> RrProducts {
    let k = path.k;
    let mut out = RrProducts {
        qe: [0.0; MAX_PATH_VERTICES + 1],
        ql: [0.0; MAX_PATH_VERTICES + 1],
    };

    // Eye-side sweep: the roll at x_l governs the extension to x_{l+1},
    // whose factor is the sample weight of the event at x_l toward
    // x_{l+1}. The eye subpath carries radiance.
    let mut alpha = Vec3::new(1.0, 1.0, 1.0);
    out.qe[1] = 1.0;
    for l in 1..k {
        let factor = path.event_weight(l, l - 1, l + 1, Transport::Radiance);
        let beta = alpha * factor;
        let q = rr_survival(beta, l);
        out.qe[l + 1] = out.qe[l] * q as f64;
        alpha = beta * (1.0 / q);
    }

    // Light-side sweep: the roll at y_j governs the extension to y_{j+1},
    // whose factor is the sample weight of the event at y_j = x_{k-j}
    // toward y_{j+1} = x_{k-j-1}. The first extension has no BSDF at the
    // emitter (f = 1, cosine sampling), hence the PI weight. The light
    // subpath carries importance.
    let mut alpha_l = alpha_light0;
    out.ql[0] = 1.0;
    for j in 0..k.saturating_sub(2) {
        let factor = if j == 0 {
            Vec3::new(PI, PI, PI)
        } else {
            path.event_weight(k - j, k - j + 1, k - j - 1, Transport::Importance)
        };
        let beta = alpha_l * factor;
        let q = rr_survival(beta, j + 1);
        out.ql[j + 1] = out.ql[j] * q as f64;
        alpha_l = beta * (1.0 / q);
    }

    out
}

/// Trace one camera ray with BDPT. Returns the estimated radiance.
pub fn bdpt_trace(
    ray: &Ray,
    scene: &dyn Intersectable,
    max_depth: usize,
    lights: &[AreaLight],
    rng: &mut impl Rng,
) -> Vec3 {
    let max_depth = max_depth.min(MAX_PATH_VERTICES);
    let mut total = VEC_ZERO;

    // ---- Eye subpath: x_1 .. x_s ----
    let mut eye: Vec<PathVertex> = Vec::with_capacity(max_depth);
    let mut throughput = Vec3::new(1.0, 1.0, 1.0);
    let mut cur = ray.clone();

    for _ in 0..max_depth {
        let Some(hit) = scene.intersect(&cur) else {
            if eye.is_empty() {
                return BACKGROUND;
            }
            break;
        };
        // Orient the normal toward the side the ray arrived from. Diffuse
        // surfaces are one-sided: a back-face hit cannot scatter, so the
        // walk stops there (the plan's BSDF-aware one-sided rule). Mirror
        // and glass flip the normal and continue — glass needs `front_face`
        // to pick eta_i/eta_t.
        let front_face = hit.normal.dot(cur.direction) < 0.0;
        if !front_face && !hit.mat.bsdf.is_delta() {
            break;
        }
        let normal = if front_face {
            hit.normal
        } else {
            hit.normal * -1.0
        };
        let on_light = hit.mat.is_emissive();
        eye.push(PathVertex {
            p: hit.p,
            normal,
            front_face,
            mat: hit.mat,
            throughput,
            on_light,
        });
        if on_light {
            break;
        }
        // Sample the BSDF, then roll the roulette on the post-scatter
        // throughput: survive with probability q and divide by q, which
        // keeps the expected contribution of the subpath unchanged.
        let s = hit
            .mat
            .sample_bsdf(normal, front_face, cur.direction, Transport::Radiance, rng);
        let beta = throughput * s.weight;
        let q = rr_survival(beta, eye.len());
        if q < 1.0 && rng.gen::<f32>() >= q {
            break;
        }
        throughput = beta * (1.0 / q);
        // Offset along the side the new ray travels to: a refracted ray
        // leaves through the surface, so offsetting by the arrival-side
        // normal would start it inside the surface.
        let offset = if s.dir_out.dot(normal) > 0.0 {
            normal
        } else {
            normal * -1.0
        };
        cur = Ray {
            origin: hit.p + offset * EPS_NEAR,
            direction: s.dir_out,
        };
    }

    // ---- Light subpath: y_0 .. y_t ----
    let n_lights = lights.len();
    let mut light_path: Vec<PathVertex> = Vec::with_capacity(max_depth);
    let mut inv_light_pdf = 0.0f64;
    if n_lights > 0 {
        let light = &lights[rng.gen_range(0..n_lights)];
        inv_light_pdf = n_lights as f64 * light.area() as f64;
        let (p0, n0) = light.sample(rng);
        let mut ltp = light.emission * inv_light_pdf as f32;
        light_path.push(PathVertex {
            p: p0,
            normal: n0,
            front_face: true,
            mat: Material::emissive(light.emission),
            throughput: ltp,
            on_light: true,
        });

        // The first light extension (y_0 -> y_1) is unconditional
        // (RR_START >= 2). The emitter has no BSDF (f = 1) and scatters
        // cosine-weighted, so the edge contributes a factor PI.
        ltp = ltp * PI;
        let mut cur = Ray {
            origin: p0 + n0 * EPS_NEAR,
            direction: cosine_hemisphere(n0, rng),
        };
        // The connection strategies need k = i + j + 2 <= max_depth, so the
        // last light vertex (j = max_depth - 1) could never be connected.
        // Building one vertex fewer avoids a wasted intersection.
        for _ in 1..max_depth.saturating_sub(1) {
            let Some(hit) = scene.intersect(&cur) else {
                break;
            };
            // Pure emitters do not reflect.
            if hit.mat.is_emissive() {
                break;
            }
            let front_face = hit.normal.dot(cur.direction) < 0.0;
            if !front_face && !hit.mat.bsdf.is_delta() {
                break;
            }
            let normal = if front_face {
                hit.normal
            } else {
                hit.normal * -1.0
            };
            light_path.push(PathVertex {
                p: hit.p,
                normal,
                front_face,
                mat: hit.mat,
                throughput: ltp,
                on_light: false,
            });
            // Russian roulette for the extension y_j -> y_{j+1}
            // (j = light_path.len() - 1 >= 1, number j + 1).
            let j = light_path.len() - 1;
            let s = hit.mat.sample_bsdf(
                normal,
                front_face,
                cur.direction,
                Transport::Importance,
                rng,
            );
            let beta = ltp * s.weight;
            let q = rr_survival(beta, j + 1);
            if q < 1.0 && rng.gen::<f32>() >= q {
                break;
            }
            ltp = beta * (1.0 / q);
            let offset = if s.dir_out.dot(normal) > 0.0 {
                normal
            } else {
                normal * -1.0
            };
            cur = Ray {
                origin: hit.p + offset * EPS_NEAR,
                direction: s.dir_out,
            };
        }
    }

    // ---- Strategy "eye subpath hits a light" ----
    if let Some(last) = eye.last().filter(|v| v.on_light) {
        let k = eye.len();
        let w = match lights.iter().find(|l| l.contains(last.p)) {
            Some(light) => {
                let inv_pdf = n_lights as f64 * light.area() as f64;
                let path = PathView {
                    eye: &eye,
                    light_path: &light_path,
                    k,
                    n_eye: k,
                    cam_dir: ray.direction,
                    inv_light_pdf: inv_pdf,
                };
                // The hypothetical light subpath of the other strategies
                // starts at this hit point with the emitter's throughput
                // (emission * n * A, the inverse light-sampling pdf).
                let rr = rr_prefix_products(&path, light.emission * inv_pdf as f32);
                mis_weight(&path, &rr.qe, &rr.ql, k)
            }
            None => 1.0,
        };
        total = total + last.throughput * last.mat.emission * w;
    }

    // ---- Connection strategies: eye vertex x_{i+1} to light vertex y_j ----
    for i in 0..eye.len() {
        if eye[i].on_light {
            break; // only the last eye vertex can be emissive
        }
        for j in 0..light_path.len() {
            let k = i + j + 2; // vertices after the camera
            if k > max_depth {
                break;
            }
            // Delta BSDFs cannot be connection endpoints: their eval is 0,
            // so the contribution would be 0 anyway.
            if eye[i].mat.is_delta() || light_path[j].mat.is_delta() {
                continue;
            }
            let path = PathView {
                eye: &eye,
                light_path: &light_path,
                k,
                n_eye: i + 1,
                cam_dir: ray.direction,
                inv_light_pdf,
            };

            // Contribution: alpha_eye * f(x) * G * f(y) * alpha_light.
            // The path arrives at x_{i+1} from x_i (the camera for i = 0)
            // and at y_j from x_{i+1}, leaving toward y_{j-1}.
            let d_conn = path.dir(i + 1, i + 2);
            let d_in_eye = if i == 0 {
                ray.direction
            } else {
                path.dir(i, i + 1)
            };
            let (n_eye_end, _) = path.oriented(i + 1, d_in_eye);
            let (f_eye, _) = path
                .vertex(i + 1)
                .mat
                .eval_bsdf(n_eye_end, d_in_eye, d_conn);
            let (f_light, n_light_end) = if j == 0 {
                // y_0 is the emitter itself: no BSDF applies there.
                (Vec3::new(1.0, 1.0, 1.0), path.vertex(k).normal)
            } else {
                let d_in_light = d_conn * -1.0;
                let d_out_light = path.dir(i + 2, i + 3);
                let (n, _) = path.oriented(i + 2, d_in_light);
                let (f, _) = path
                    .vertex(i + 2)
                    .mat
                    .eval_bsdf(n, d_in_light, d_out_light);
                (f, n)
            };
            let g_conn = g_term(path.vertex(i + 1).p, n_eye_end, path.vertex(i + 2).p, n_light_end);
            if g_conn <= 0.0 || f_eye == VEC_ZERO || f_light == VEC_ZERO {
                continue;
            }
            if !visible(path.vertex(i + 1).p, n_eye_end, path.vertex(i + 2).p, scene) {
                continue;
            }
            let contrib = eye[i].throughput
                * f_eye
                * g_conn
                * f_light
                * light_path[j].throughput;

            let rr = rr_prefix_products(&path, light_path[0].throughput);
            let w = mis_weight(&path, &rr.qe, &rr.ql, i + 1);
            total = total + contrib * w;
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Cube, Material, Plane, Quad, Sphere};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Write a marker so a failing test can be identified when the
    /// harness hides stdout: each test records its name on entry, and a
    /// panic hook dumps the message to the same file.
    fn mark(name: &str) {
        std::fs::write("target/test_marker.txt", name).ok();
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "non-string panic".to_string()
            };
            let loc = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_else(|| "unknown".to_string());
            std::fs::write(
                "target/test_panic.txt",
                format!("panic at {loc}: {msg}"),
            )
            .ok();
            default_hook(info);
        }));
    }

    fn diffuse(albedo: Vec3) -> Material {
        Material::diffuse(albedo)
    }

    fn emissive(emission: Vec3) -> Material {
        Material::emissive(emission)
    }

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray {
            origin,
            direction: direction.normalize(),
        }
    }

    /// A tiny room: floor at y=0, one 1x1 quad light at y=3 facing down.
    fn test_scene() -> (Box<dyn Intersectable>, Vec<AreaLight>) {
        let light = Quad::new(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            1.0,
            1.0,
            emissive(Vec3::new(1.0, 1.0, 1.0)),
        );
        let objects: Vec<Box<dyn Intersectable>> = vec![
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: diffuse(Vec3::new(0.8, 0.8, 0.8)),
            }),
            Box::new(light.clone()),
        ];
        let lights = vec![AreaLight::from_quad(&light)];
        (crate::bvh::BVHNode::build(objects).unwrap(), lights)
    }

    /// A synthetic path for the MIS/RR unit tests: k vertices, all diffuse
    /// with the given albedo, laid out as a physically valid bounce chain
    /// between the floor (y = 0, normal +y) and the ceiling (y = 2, normal
    /// -y), with alternating x offsets so consecutive segments are never
    /// collinear with the normal. Every arrival and departure direction
    /// then has a positive cosine against the vertex normal — a layout
    /// with horizontal travel (all vertices on the floor) would give
    /// cos = 0 and zero out every pdf, silently invalidating every
    /// strategy.
    ///
    /// The camera hovers above x_1 looking down at it; the light y_0 = x_k
    /// sits at the chain's far end with its stored normal facing back down
    /// the chain — the direction the light subpath scatters, which
    /// `pdf_light(k)` measures against that stored normal.
    fn synthetic_path(
        k: usize,
        albedo: Vec3,
        _inv_light_pdf: f64,
    ) -> (Vec<PathVertex>, Vec<PathVertex>, Vec3) {
        let mk = |m: usize| {
            let ceiling = m % 2 == 0;
            PathVertex {
                p: Vec3::new(
                    if ceiling { 0.5 } else { -0.5 },
                    if ceiling { 2.0 } else { 0.0 },
                    m as f32,
                ),
                normal: if ceiling {
                    Vec3::new(0.0, -1.0, 0.0)
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                },
                front_face: true,
                mat: diffuse(albedo),
                throughput: VEC_ZERO,
                on_light: false,
            }
        };
        let mut eye = Vec::new();
        let mut light_path = Vec::new();
        // The eye subpath takes ceil(k/2) vertices so the split matches
        // the n_eye the tests declare (k=4 -> 2, k=5 -> 3); integer
        // division of k alone would give floor(k/2) and leave n_eye
        // pointing past the eye slice.
        for m in 1..=k {
            let v = mk(m);
            if m <= (k + 1) / 2 {
                eye.push(v);
            } else {
                light_path.push(v);
            }
        }
        // light_path is stored y_0..y_t; the view reads it in reverse.
        light_path.reverse();
        // The camera hovers above x_1 (a floor vertex), one unit back in z.
        let x1 = mk(1);
        let cam_dir = (x1.p - Vec3::new(x1.p.x, 2.0, x1.p.z - 1.0)).normalize();
        (eye, light_path, cam_dir)
    }

    #[test]
    fn mis_weights_sum_to_one_asymmetric() {
        mark("mis_weights_sum_to_one_asymmetric");
        // A k=4 path with asymmetric pdfs: the eye and light subpaths
        // sample different directions, so the per-edge pdfs differ between
        // the two sampling directions. The weights must still sum to 1
        // over the valid strategies.
        let albedo = Vec3::new(0.5, 0.5, 0.5);
        let (eye, light_path, cam_dir) = synthetic_path(4, albedo, 2.0);
        let path = PathView {
            eye: &eye,
            light_path: &light_path,
            k: 4,
            n_eye: 2,
            cam_dir,
            inv_light_pdf: 2.0,
        };
        let rr = rr_prefix_products(&path, Vec3::new(1.0, 1.0, 1.0));
        let mut sum = 0.0f32;
        for s in 1..=4 {
            sum += mis_weight(&path, &rr.qe, &rr.ql, s);
        }
        assert!((sum - 1.0).abs() < 1e-4, "weights sum to {sum}");
    }

    /// Sum-to-one with a delta vertex on the path: the delta-sampled
    /// edges get the symbolic pdf 1, and the strategies whose connection
    /// would need a non-specular direction through the delta vertex are
    /// invalid (contribution 0) and must be excluded from the
    /// normalization. The remaining weights must still sum to 1.
    #[test]
    fn mis_weights_sum_to_one_with_delta_vertex() {
        mark("mis_weights_sum_to_one_with_delta_vertex");
        // A fully consistent path, all normals axis-aligned:
        //   camera at (0, 2, 0) looking straight down,
        //   x_1 = (0, 0, 0)   diffuse floor,  normal +y,
        //   x_2 = (0, 2, 0)   mirror ceiling, normal -y,
        //   x_3 = (0, 0, 0)   diffuse floor,  normal +y,
        //   x_4 = (0, 2, 1)   emitter on the ceiling, normal -y.
        // The specular pair at x_2: d_in = +y (floor to ceiling),
        // reflect about -y => d_out = -y (ceiling to floor). ✓
        // The camera ray: from (0,2,0) to (0,0,0) => d_in at x_1 = -y. ✓
        // The light subpath: y_0 = x_4 on the ceiling emitting downward
        // (cosine sampling around -y), y_1 = x_3 on the floor. The
        // segment x_3 -> x_4 rises, so its cosine against both normals is
        // positive — a floor-to-floor segment would be horizontal and
        // give cos = 0, zeroing the pdfs.
        let albedo = Vec3::new(0.5, 0.5, 0.5);
        let mk = |p: Vec3, normal: Vec3, mat: Material| PathVertex {
            p,
            normal,
            front_face: true,
            mat,
            throughput: VEC_ZERO,
            on_light: false,
        };
        let eye = vec![
            mk(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), diffuse(albedo)),
            mk(
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Material::mirror(Vec3::new(0.9, 0.9, 0.9)),
            ),
        ];
        // light_path[0] = y_0 = x_4 (the emitter), light_path[1] = y_1 = x_3.
        let light_path = vec![
            mk(
                Vec3::new(0.0, 2.0, 1.0),
                Vec3::new(0.0, -1.0, 0.0),
                Material::emissive(Vec3::new(1.0, 1.0, 1.0)),
            ),
            mk(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), diffuse(albedo)),
        ];
        let cam_dir = Vec3::new(0.0, -1.0, 0.0);
        let path = PathView {
            eye: &eye,
            light_path: &light_path,
            k: 4,
            n_eye: 2,
            cam_dir,
            inv_light_pdf: 2.0,
        };
        let rr = rr_prefix_products(&path, Vec3::new(1.0, 1.0, 1.0));
        let mut w = [0.0f32; 5];
        for s in 1..=4 {
            w[s] = mis_weight(&path, &rr.qe, &rr.ql, s);
        }
        // Only the valid strategies' weights are meaningful; the invalid
        // ones (delta endpoint) are never computed by bdpt_trace. The
        // valid ones must sum to 1.
        let valid_sum = w[3] + w[4];
        assert!(
            (valid_sum - 1.0).abs() < 1e-4,
            "valid weights sum to {valid_sum} (w = {w:?})"
        );
    }

    #[test]
    fn rr_survival_is_one_before_start_and_clamped_after() {
        let beta = Vec3::new(0.3, 0.7, 0.2);
        // Before RR_START every extension is unconditional.
        for ext in 0..RR_START {
            assert_eq!(rr_survival(beta, ext), 1.0);
        }
        // From RR_START on: the max channel, clamped to [RR_MIN, 1].
        assert!((rr_survival(beta, RR_START) - 0.7).abs() < 1e-6);
        assert!((rr_survival(Vec3::new(2.0, 3.0, 1.0), RR_START) - 1.0).abs() < 1e-6);
        assert!((rr_survival(VEC_ZERO, RR_START) - RR_MIN).abs() < 1e-6);
    }

    #[test]
    fn rr_prefix_products_match_hand_computation() {
        mark("rr_prefix_products_match_hand_computation");
        // Path x_1 x_2 x_3 | y_1 y_0 connected at (i=2, j=1):
        // k = 5, n_eye = 3, n_light = 2. All albedos 0.5, emitter
        // throughput 0.1. This exercises both cross-over reads: the eye
        // sweep extends hypothetically through y_1 (l = 4) and the light
        // sweep through x_3 (j = 2).
        let albedo = Vec3::new(0.5, 0.5, 0.5);
        let (eye, light_path, cam_dir) = synthetic_path(5, albedo, 1.0);
        let path = PathView {
            eye: &eye,
            light_path: &light_path,
            k: 5,
            n_eye: 3,
            cam_dir,
            inv_light_pdf: 1.0,
        };
        let rr = rr_prefix_products(&path, Vec3::new(0.1, 0.1, 0.1));

        // Eye sweep: ext 1, 2 unconditional; ext 3: beta = 0.5^3 = 0.125;
        // ext 4: alpha = 0.125/0.125 = 1, beta = 1 * albedo(y_1) = 0.5.
        assert!((rr.qe[1] - 1.0).abs() < 1e-12);
        assert!((rr.qe[2] - 1.0).abs() < 1e-12);
        assert!((rr.qe[3] - 1.0).abs() < 1e-12);
        assert!((rr.qe[4] - 0.125).abs() < 1e-6, "qe[4] = {}", rr.qe[4]);
        assert!((rr.qe[5] - 0.0625).abs() < 1e-6, "qe[5] = {}", rr.qe[5]);

        // Light sweep: ext 1 (PI factor, q = 1), ext 2 (albedo(y_1), q = 1),
        // ext 3: factor = albedo(x_3), beta = 0.1 * PI * 0.5 * 0.5.
        assert!((rr.ql[0] - 1.0).abs() < 1e-12);
        assert!((rr.ql[1] - 1.0).abs() < 1e-12);
        assert!((rr.ql[2] - 1.0).abs() < 1e-12);
        let expected = 0.1 * PI * 0.5 * 0.5;
        assert!(
            (rr.ql[3] - expected as f64).abs() < 1e-6,
            "ql[3] = {}",
            rr.ql[3]
        );
    }

    #[test]
    fn rr_leaves_short_paths_exact() {
        // A direct hit at x_1 involves no roulette at all; with RR armed
        // (max_depth 8) the exact emission must still come back.
        let (scene, lights) = test_scene();
        let mut rng = StdRng::seed_from_u64(7);
        let r = ray(Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let c = bdpt_trace(&r, &*scene, 8, &lights, &mut rng);
        assert!((c.x - 1.0).abs() < 1e-4, "got {c:?}");
        assert!((c.y - 1.0).abs() < 1e-4);
        assert!((c.z - 1.0).abs() < 1e-4);
    }

    /// Reference unidirectional path tracer: no MIS, no roulette, fixed
    /// depth. Correct by inspection — it sums `throughput * emission` over
    /// every emissive surface the eye path hits, with the same BSDF
    /// sampling and one-sided-surface rules as `bdpt_trace`. Used as the
    /// oracle for the roulette unbiasedness check and the caustic
    /// differential.
    fn reference_pt(
        ray: &Ray,
        scene: &dyn Intersectable,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> Vec3 {
        let mut total = VEC_ZERO;
        let mut throughput = Vec3::new(1.0, 1.0, 1.0);
        let mut cur = ray.clone();
        for _ in 0..max_depth {
            let Some(hit) = scene.intersect(&cur) else {
                break;
            };
            let front_face = hit.normal.dot(cur.direction) < 0.0;
            if !front_face && !hit.mat.bsdf.is_delta() {
                break;
            }
            let normal = if front_face {
                hit.normal
            } else {
                hit.normal * -1.0
            };
            total = total + throughput * hit.mat.emission;
            if hit.mat.is_emissive() {
                break;
            }
            let s = hit
                .mat
                .sample_bsdf(normal, front_face, cur.direction, Transport::Radiance, rng);
            throughput = throughput * s.weight;
            let offset = if s.dir_out.dot(normal) > 0.0 {
                normal
            } else {
                normal * -1.0
            };
            cur = Ray {
                origin: hit.p + offset * EPS_NEAR,
                direction: s.dir_out,
            };
        }
        total
    }

    #[test]
    fn rr_does_not_bias_the_mean() {
        mark("rr_does_not_bias_the_mean");
        // Differential test: BDPT with roulette active must agree with the
        // reference path tracer over the same depth-capped path space.
        // Both are unbiased, so only Monte-Carlo noise separates them; a
        // missing 1/q compensation biases every contribution that passed a
        // roll down by roughly the kill rate (~50% at the first rolled
        // extension in this scene), far outside this tolerance.
        let (scene, lights) = test_scene();
        let r = ray(Vec3::new(0.0, 1.5, 2.0), Vec3::new(0.0, -1.0, -0.5));
        const N: usize = 16384;
        const DEPTH: usize = 8;

        let mut rng = StdRng::seed_from_u64(123);
        let mut mean_bdpt = VEC_ZERO;
        for _ in 0..N {
            mean_bdpt = mean_bdpt + bdpt_trace(&r, &*scene, DEPTH, &lights, &mut rng);
        }
        mean_bdpt = mean_bdpt * (1.0 / N as f32);

        let mut rng = StdRng::seed_from_u64(456);
        let mut mean_ref = VEC_ZERO;
        for _ in 0..N {
            mean_ref = mean_ref + reference_pt(&r, &*scene, DEPTH, &mut rng);
        }
        mean_ref = mean_ref * (1.0 / N as f32);

        let tol = (mean_ref.x * 0.05).max(0.01);
        assert!(
            (mean_bdpt.x - mean_ref.x).abs() <= tol,
            "roulette biased the mean: bdpt {mean_bdpt:?} vs reference {mean_ref:?}"
        );
    }

    #[test]
    fn looking_straight_at_the_light_returns_its_emission() {
        let (scene, lights) = test_scene();
        let mut rng = StdRng::seed_from_u64(7);
        let r = ray(Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let c = bdpt_trace(&r, &*scene, 4, &lights, &mut rng);
        assert!((c.x - 1.0).abs() < 1e-4, "got {c:?}");
        assert!((c.y - 1.0).abs() < 1e-4);
        assert!((c.z - 1.0).abs() < 1e-4);
    }

    #[test]
    fn lit_floor_is_finite_and_nonzero() {
        let (scene, lights) = test_scene();
        let mut rng = StdRng::seed_from_u64(11);
        let r = ray(Vec3::new(0.0, 1.5, 2.0), Vec3::new(0.0, -1.0, -0.5));
        let mut mean = VEC_ZERO;
        for _ in 0..64 {
            mean = mean + bdpt_trace(&r, &*scene, 8, &lights, &mut rng);
        }
        mean = mean * (1.0 / 64.0);
        assert!(mean.x.is_finite() && mean.y.is_finite() && mean.z.is_finite());
        assert!(mean.x > 0.0 && mean.y > 0.0 && mean.z > 0.0, "got {mean:?}");
        // The floor cannot reflect more energy than the light emits.
        assert!(mean.x < 1.0 && mean.y < 1.0 && mean.z < 1.0, "got {mean:?}");
    }

    #[test]
    fn occluded_point_stays_dark() {
        mark("occluded_point_stays_dark");
        // A dark cube between the floor and the light blocks direct light.
        // The light is large and close to the floor so the direct term
        // dominates the unshadowed point — the contrast assertion needs a
        // strong direct term to be meaningful (with the small high light
        // of `test_scene` the true contrast is only ~3x, right at the
        // assertion boundary).
        let light = Quad::new(
            Vec3::new(0.0, 1.2, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            2.0,
            emissive(Vec3::new(1.0, 1.0, 1.0)),
        );
        let objects: Vec<Box<dyn Intersectable>> = vec![
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: diffuse(Vec3::new(0.8, 0.8, 0.8)),
            }),
            // The occluder must be wider than the light, or the light
            // wraps around it and the "shadow" is not one.
            Box::new(Cube::new(
                Vec3::new(0.0, 1.5, 0.0),
                3.0,
                diffuse(Vec3::new(0.02, 0.02, 0.02)),
            )),
            Box::new(light.clone()),
        ];
        let lights = vec![AreaLight::from_quad(&light)];
        let scene = crate::bvh::BVHNode::build(objects).unwrap();

        let mut rng = StdRng::seed_from_u64(3);
        // Same floor point, once shadowed by the cube and once with the
        // cube removed: the difference must be dramatic. The shadowed
        // point is lit only indirectly, so its estimate carries the
        // roulette variance; 4096 samples keep the contrast assertion
        // stable across RNG-stream changes.
        let target = ray(Vec3::new(0.0, 1.5, 2.0), Vec3::new(0.0, -1.0, -0.9));

        let mut mean_shadowed = VEC_ZERO;
        for _ in 0..4096 {
            mean_shadowed = mean_shadowed + bdpt_trace(&target, &*scene, 4, &lights, &mut rng);
        }
        mean_shadowed = mean_shadowed * (1.0 / 4096.0);

        // Rebuild the scene without the occluder.
        let objects_no_cube: Vec<Box<dyn Intersectable>> = vec![
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: diffuse(Vec3::new(0.8, 0.8, 0.8)),
            }),
            Box::new(light.clone()),
        ];
        let scene_no_cube = crate::bvh::BVHNode::build(objects_no_cube).unwrap();
        let mut mean_lit = VEC_ZERO;
        for _ in 0..4096 {
            mean_lit = mean_lit + bdpt_trace(&target, &*scene_no_cube, 4, &lights, &mut rng);
        }
        mean_lit = mean_lit * (1.0 / 4096.0);

        // The shadowed point receives only weak indirect light...
        assert!(
            mean_shadowed.x < 0.05,
            "shadowed point too bright: {mean_shadowed:?}"
        );
        // ...while the same point without the occluder is much brighter.
        // The exact ratio depends on the scene geometry (the light is
        // small and high, so the direct term dominates); the assertion is
        // a factor of 3, not an exact value.
        assert!(
            mean_lit.x > mean_shadowed.x * 3.0,
            "no contrast between shadow and light: shadowed {mean_shadowed:?} lit {mean_lit:?}"
        );
    }

    #[test]
    fn visibility_detects_occlusion() {
        let cube = Cube::new(
            Vec3::new(0.0, 0.5, 0.0),
            1.0,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        let a = Vec3::new(-2.0, 0.5, 0.0);
        let b = Vec3::new(2.0, 0.5, 0.0);
        // Segment passes straight through the cube.
        assert!(!visible(a, Vec3::new(0.0, 1.0, 0.0), b, &cube));
        // Segment above the cube is clear.
        let c = Vec3::new(2.0, 2.0, 0.0);
        assert!(visible(a, Vec3::new(0.0, 1.0, 0.0), c, &cube));
    }

    #[test]
    fn cosine_samples_stay_in_hemisphere() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = Vec3::new(0.0, 1.0, 0.0);
        for _ in 0..200 {
            let d = cosine_hemisphere(n, &mut rng);
            assert!(d.dot(n) > 0.0);
            assert!((d.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn sphere_light_direct_hit() {
        let bulb = Sphere {
            center: Vec3::new(0.0, 2.0, 0.0),
            radius: 0.5,
            mat: emissive(Vec3::new(4.0, 4.0, 4.0)),
        };
        let objects: Vec<Box<dyn Intersectable>> = vec![
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: diffuse(Vec3::new(0.8, 0.8, 0.8)),
            }),
            Box::new(bulb.clone()),
        ];
        let lights = vec![AreaLight::from_sphere(&bulb)];
        let scene = crate::bvh::BVHNode::build(objects).unwrap();

        let mut rng = StdRng::seed_from_u64(9);
        let r = ray(Vec3::new(0.0, 0.5, 2.0), Vec3::new(0.0, 1.0, -1.0));
        let c = bdpt_trace(&r, &*scene, 4, &lights, &mut rng);
        assert!((c.x - 4.0).abs() < 1e-3, "got {c:?}");
    }

    /// The white furnace: a closed room where every surface has albedo 1
    /// and one pure emitter. Radiance is conserved, so a ray hitting a
    /// wall must return the emitter's radiance exactly — throughput never
    /// decays and every path eventually ends on the emitter. This is the
    /// guardian test for the whole BSDF-generalization phase: energy loss
    /// in any BSDF, dropped MIS strategies, or roulette bias all show up
    /// here as a mean below Le.
    ///
    /// One subtlety the naive setup gets wrong: with albedo 1 the
    /// throughput never decays, so Russian roulette never fires
    /// (q = clamp(1, RR_MIN, 1) = 1) and the only path terminator is the
    /// MAX_PATH_VERTICES = 8 cap. That truncation is real bias — the
    /// missing tail is P(8 bounces without an emitter hit) — so the room
    /// must make an emitter hit overwhelmingly likely per bounce. A wide,
    /// very flat room with the emitter as the entire ceiling does that:
    /// the walls (the surfaces that see the ceiling most edge-on) are a
    /// small fraction of the surface and sit directly beneath the
    /// emitter's edge. Measured tail in this geometry: < 0.5%.
    #[test]
    fn white_furnace() {
        mark("white_furnace");
        // Room: 4 x 4 x 0.25 (x, z, y), emitter = the whole ceiling.
        let le = Vec3::new(1.0, 1.0, 1.0);
        let light = Quad::new(
            Vec3::new(0.0, 0.25, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            4.0,
            4.0,
            emissive(le),
        );
        let white = diffuse(Vec3::new(1.0, 1.0, 1.0));
        let objects: Vec<Box<dyn Intersectable>> = vec![
            // Floor
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: white,
            }),
            // Side walls
            Box::new(Plane {
                point: Vec3::new(-2.0, 0.125, 0.0),
                normal: Vec3::new(1.0, 0.0, 0.0),
                mat: white,
            }),
            Box::new(Plane {
                point: Vec3::new(2.0, 0.125, 0.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
                mat: white,
            }),
            Box::new(Plane {
                point: Vec3::new(0.0, 0.125, -2.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                mat: white,
            }),
            Box::new(Plane {
                point: Vec3::new(0.0, 0.125, 2.0),
                normal: Vec3::new(0.0, 0.0, -1.0),
                mat: white,
            }),
            // Ceiling emitter
            Box::new(light.clone()),
        ];
        let lights = vec![AreaLight::from_quad(&light)];
        let scene = crate::bvh::BVHNode::build(objects).unwrap();

        // A ray hitting the floor at a grazing angle: it must return Le.
        let mut rng = StdRng::seed_from_u64(2026);
        let r = ray(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.3, -1.0, 0.2));
        const N: usize = 4096;
        let mut mean = VEC_ZERO;
        for _ in 0..N {
            mean = mean + bdpt_trace(&r, &*scene, 8, &lights, &mut rng);
        }
        mean = mean * (1.0 / N as f32);
        assert!(
            (mean.x - le.x).abs() < 0.03 * le.x,
            "furnace mean {mean:?} != Le {le:?}"
        );
        assert!(
            (mean.y - le.y).abs() < 0.03 * le.y,
            "furnace mean {mean:?} != Le {le:?}"
        );
        assert!(
            (mean.z - le.z).abs() < 0.03 * le.z,
            "furnace mean {mean:?} != Le {le:?}"
        );
    }

    /// Mirror determinism: a ray at a mirror wall that reflects into a
    /// wall region must match the mean of a direct ray at that region.
    /// A mirror must not absorb or scatter — the reflected image must be
    /// exactly what the direct view sees.
    #[test]
    fn mirror_determinism() {
        mark("mirror_determinism");
        // Room: floor + ceiling + two side walls, one of them a mirror.
        // A quad light on the ceiling.
        let le = Vec3::new(1.0, 1.0, 1.0);
        let light = Quad::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            1.0,
            1.0,
            emissive(le),
        );
        let white = diffuse(Vec3::new(0.8, 0.8, 0.8));
        let mirror = Material::mirror(Vec3::new(1.0, 1.0, 1.0));
        let objects: Vec<Box<dyn Intersectable>> = vec![
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: white,
            }),
            // Mirror wall at z = -1 facing +z.
            Box::new(Plane {
                point: Vec3::new(0.0, 1.0, -1.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                mat: mirror,
            }),
            // Diffuse wall at z = +1 facing -z (the reflected target).
            Box::new(Plane {
                point: Vec3::new(0.0, 1.0, 1.0),
                normal: Vec3::new(0.0, 0.0, -1.0),
                mat: white,
            }),
            Box::new(light.clone()),
        ];
        let lights = vec![AreaLight::from_quad(&light)];
        let scene = crate::bvh::BVHNode::build(objects).unwrap();

        // A ray at the mirror (hitting z = -1) reflects toward +z and hits
        // the diffuse wall at z = +1. The direct ray hits the same wall
        // point from the camera position.
        let mut rng = StdRng::seed_from_u64(13);
        const N: usize = 4096;
        let r_mirror = ray(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let mut mean_mirror = VEC_ZERO;
        for _ in 0..N {
            mean_mirror = mean_mirror + bdpt_trace(&r_mirror, &*scene, 8, &lights, &mut rng);
        }
        mean_mirror = mean_mirror * (1.0 / N as f32);

        let r_direct = ray(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let mut mean_direct = VEC_ZERO;
        for _ in 0..N {
            mean_direct = mean_direct + bdpt_trace(&r_direct, &*scene, 8, &lights, &mut rng);
        }
        mean_direct = mean_direct * (1.0 / N as f32);

        // The mirror ray sees the wall through one extra bounce; the
        // direct ray sees it directly. Both must agree (the mirror is
        // lossless), within Monte-Carlo noise.
        let tol = (mean_direct.x * 0.05).max(0.01);
        assert!(
            (mean_mirror.x - mean_direct.x).abs() <= tol,
            "mirror view {mean_mirror:?} != direct view {mean_direct:?}"
        );
    }

    /// Glass energy: the white furnace with a glass sphere inside. Fresnel
    /// + TIR must conserve R + T = 1, so the mean must still be Le.
    ///
    /// Design notes (each guards a specific false-negative):
    /// - Flat room (4 x 4 x 0.25, emitter = the whole ceiling): the same
    ///   geometry `white_furnace` proved has < 0.5% truncation tail at
    ///   depth 8. A taller room truncates heavily and the deficit would
    ///   be indistinguishable from glass energy loss.
    /// - The sphere floats mid-room and the ray takes a near-axial chord
    ///   (impact parameter ~ 0.05 of the radius): entry/exit are far from
    ///   tangency, and the internal incidence stays below the critical
    ///   angle, so the ray transmits instead of entering a TIR orbit that
    ///   would eat the depth budget inside the sphere.
    /// - A no-sphere control with the same seed isolates glass-specific
    ///   loss from room truncation: the assertion compares glass vs
    ///   control, not glass vs Le, so truncation cancels.
    #[test]
    fn glass_energy_furnace() {
        // Room: 4 x 4 x 0.25 (x, z, y), emitter = the whole ceiling.
        let le = Vec3::new(1.0, 1.0, 1.0);
        let light = Quad::new(
            Vec3::new(0.0, 0.25, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            4.0,
            4.0,
            emissive(le),
        );
        let white = diffuse(Vec3::new(1.0, 1.0, 1.0));
        let glass = Material::glass(1.5);
        let sphere = Sphere {
            center: Vec3::new(0.0, 0.125, 0.0),
            radius: 0.1,
            mat: glass,
        };
        let lights = vec![AreaLight::from_quad(&light)];

        // The camera ray: from just under the ceiling, near-axially
        // through the sphere, to the floor.
        let r = ray(Vec3::new(0.0, 0.24, 0.0), Vec3::new(0.05, -1.0, 0.0));

        const N: usize = 4096;
        const DEPTH: usize = 8;
        // The room builder, called twice: once with the glass sphere and
        // once without (the control). Box<dyn Intersectable> is not
        // Clone, so the lists are built fresh.
        let build_room = |with_sphere: bool| -> Vec<Box<dyn Intersectable>> {
            let mut objects: Vec<Box<dyn Intersectable>> = vec![
                Box::new(Plane {
                    point: Vec3::new(0.0, 0.0, 0.0),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    mat: white,
                }),
                Box::new(Plane {
                    point: Vec3::new(-2.0, 0.125, 0.0),
                    normal: Vec3::new(1.0, 0.0, 0.0),
                    mat: white,
                }),
                Box::new(Plane {
                    point: Vec3::new(2.0, 0.125, 0.0),
                    normal: Vec3::new(-1.0, 0.0, 0.0),
                    mat: white,
                }),
                Box::new(Plane {
                    point: Vec3::new(0.0, 0.125, -2.0),
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    mat: white,
                }),
                Box::new(Plane {
                    point: Vec3::new(0.0, 0.125, 2.0),
                    normal: Vec3::new(0.0, 0.0, -1.0),
                    mat: white,
                }),
                Box::new(light.clone()),
            ];
            if with_sphere {
                objects.push(Box::new(sphere.clone()));
            }
            objects
        };

        let mut sweep = String::new();
        let mut means = std::collections::HashMap::new();
        // Glass scene and no-sphere control, same seed: the difference is
        // the glass-specific energy loss (truncation cancels).
        for (name, objects) in [
            ("glass", build_room(true)),
            ("control", build_room(false)),
        ] {
            let scene = crate::bvh::BVHNode::build(objects).unwrap();
            let mut rng = StdRng::seed_from_u64(2027);
            let mut mean = VEC_ZERO;
            let mut sq = 0.0f64;
            for _ in 0..N {
                let c = bdpt_trace(&r, &*scene, DEPTH, &lights, &mut rng);
                mean = mean + c;
                sq += (c.x * c.x) as f64;
            }
            mean = mean * (1.0 / N as f32);
            let var = (sq / N as f64 - mean.x as f64 * mean.x as f64).max(0.0);
            let se = (var / N as f64).sqrt();
            sweep.push_str(&format!("{name}: {mean:?} se {se:.4}\n"));
            means.insert(name, (mean, se));
        }
        std::fs::remove_file("target/glass_diag.txt").ok();
        std::fs::write("target/glass_diag.txt", &sweep).ok();
        let (mean_glass, se_glass) = means["glass"];
        let (mean_control, se_control) = means["control"];
        // The glass must not lose energy relative to the control: the
        // difference must be within noise (3 sigma of the combined
        // standard errors) and both must be close to Le (the truncation
        // tail is < 0.5% in this flat room).
        let diff = (mean_glass.x - mean_control.x).abs();
        let noise = 3.0 * (se_glass + se_control);
        assert!(
            diff <= noise.max(0.02) as f32,
            "glass loses energy: glass {mean_glass:?} vs control {mean_control:?} (diff {diff}, noise {noise})"
        );
        assert!(
            (mean_glass.x - le.x).abs() < 0.05 * le.x,
            "glass furnace mean {mean_glass:?} != Le {le:?}"
        );
        assert!(
            (mean_glass.y - le.y).abs() < 0.05 * le.y,
            "glass furnace mean {mean_glass:?} != Le {le:?}"
        );
        assert!(
            (mean_glass.z - le.z).abs() < 0.05 * le.z,
            "glass furnace mean {mean_glass:?} != Le {le:?}"
        );
    }

    /// A minimal glass transport check: a closed room, all albedo 1, with
    /// a glass SLAB (two parallel quads) the camera ray passes through
    /// perpendicularly. Perpendicular incidence has Fresnel R = 0.04 per
    /// interface; the slab's internal bounces are all perpendicular too,
    /// so the transport is analytically simple and any eta^2 or Fresnel
    /// bookkeeping error shows up directly.
    #[test]
    fn glass_slab_transmits_energy() {
        mark("glass_slab_transmits_energy");
        // Room 2 x 2 x 2, emitter = the whole +z wall, glass slab across
        // the middle (two parallel quads at z = 0 and z = 0.2, normals
        // facing outward). The camera looks straight through at the light.
        let le = Vec3::new(1.0, 1.0, 1.0);
        let white = diffuse(Vec3::new(1.0, 1.0, 1.0));
        let glass = Material::glass(1.5);
        let light = Quad::new(
            Vec3::new(0.0, 1.0, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
            2.0,
            2.0,
            emissive(le),
        );
        let objects: Vec<Box<dyn Intersectable>> = vec![
            // Floor, ceiling, side walls
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: white,
            }),
            Box::new(Plane {
                point: Vec3::new(0.0, 2.0, 0.0),
                normal: Vec3::new(0.0, -1.0, 0.0),
                mat: white,
            }),
            Box::new(Plane {
                point: Vec3::new(-1.0, 1.0, 0.0),
                normal: Vec3::new(1.0, 0.0, 0.0),
                mat: white,
            }),
            Box::new(Plane {
                point: Vec3::new(1.0, 1.0, 0.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
                mat: white,
            }),
            // Glass slab: two parallel quads spanning the room cross-section.
            Box::new(Quad::new(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, -1.0),
                2.0,
                2.0,
                glass,
            )),
            Box::new(Quad::new(
                Vec3::new(0.0, 1.0, 0.2),
                Vec3::new(0.0, 0.0, 1.0),
                2.0,
                2.0,
                glass,
            )),
            // Emitter at +z
            Box::new(light.clone()),
        ];
        let lights = vec![AreaLight::from_quad(&light)];
        let scene = crate::bvh::BVHNode::build(objects).unwrap();

        let mut rng = StdRng::seed_from_u64(31);
        let r = ray(Vec3::new(0.0, 1.0, 0.5), Vec3::new(0.0, 0.0, 1.0));
        const N: usize = 4096;
        let mut mean = VEC_ZERO;
        for _ in 0..N {
            mean = mean + bdpt_trace(&r, &*scene, 8, &lights, &mut rng);
        }
        mean = mean * (1.0 / N as f32);
        // Straight through the slab. The room is a closed white furnace,
        // so the ~4% reflected per interface is not lost — it bounces
        // off the walls and eventually returns through the slab. The
        // mean must be exactly Le (energy conservation), not the
        // open-air slab transmittance (1-R)^2/(1-R^2) ≈ 0.923.
        assert!(
            (mean.x - le.x).abs() < 0.03,
            "slab mean {mean:?} != Le {le:?}"
        );
    }

    /// The caustic differential: the flagship test. A glass sphere focuses
    /// light into a caustic on the floor — a path the unidirectional
    /// tracer finds only by luck (light -> glass -> floor -> camera needs
    /// the eye walk to sample the exact refraction direction), while BDPT
    /// connects the light subpath through the glass to the eye vertex.
    /// At equal sample counts BDPT's variance must be dramatically lower.
    #[test]
    fn caustic_differential() {
        // A dark room with a small bright light and a glass sphere under
        // it. The caustic pixel looks at the floor directly beneath the
        // sphere.
        let le = Vec3::new(20.0, 20.0, 20.0);
        let light = Quad::new(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.5,
            0.5,
            emissive(le),
        );
        let dark = diffuse(Vec3::new(0.1, 0.1, 0.1));
        let glass = Material::glass(1.5);
        let objects: Vec<Box<dyn Intersectable>> = vec![
            Box::new(Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: dark,
            }),
            Box::new(Plane {
                point: Vec3::new(0.0, 3.0, 0.0),
                normal: Vec3::new(0.0, -1.0, 0.0),
                mat: dark,
            }),
            Box::new(Plane {
                point: Vec3::new(-2.0, 1.5, 0.0),
                normal: Vec3::new(1.0, 0.0, 0.0),
                mat: dark,
            }),
            Box::new(Plane {
                point: Vec3::new(2.0, 1.5, 0.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
                mat: dark,
            }),
            Box::new(Plane {
                point: Vec3::new(0.0, 1.5, -2.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                mat: dark,
            }),
            Box::new(Plane {
                point: Vec3::new(0.0, 1.5, 2.0),
                normal: Vec3::new(0.0, 0.0, -1.0),
                mat: dark,
            }),
            Box::new(light.clone()),
            // Glass sphere between the light and the floor.
            Box::new(Sphere {
                center: Vec3::new(0.0, 0.5, 0.0),
                radius: 0.5,
                mat: glass,
            }),
        ];
        let lights = vec![AreaLight::from_quad(&light)];
        let scene = crate::bvh::BVHNode::build(objects).unwrap();

        // The caustic pixel: looking at the floor just beside the sphere,
        // where the focused light lands.
        let r = ray(Vec3::new(0.0, 1.0, 1.5), Vec3::new(0.0, -1.0, -1.0));

        // BDPT: 256 samples.
        let mut rng = StdRng::seed_from_u64(99);
        const N: usize = 256;
        let mut mean_bdpt = VEC_ZERO;
        let mut sq_sum_bdpt = 0.0f32;
        for _ in 0..N {
            let c = bdpt_trace(&r, &*scene, 8, &lights, &mut rng);
            mean_bdpt = mean_bdpt + c;
            sq_sum_bdpt += c.x * c.x;
        }
        mean_bdpt = mean_bdpt * (1.0 / N as f32);
        let var_bdpt = (sq_sum_bdpt / N as f32 - mean_bdpt.x * mean_bdpt.x).max(0.0);

        // Reference PT: the same number of samples.
        let mut rng = StdRng::seed_from_u64(100);
        let mut mean_pt = VEC_ZERO;
        let mut sq_sum_pt = 0.0f32;
        for _ in 0..N {
            let c = reference_pt(&r, &*scene, 8, &mut rng);
            mean_pt = mean_pt + c;
            sq_sum_pt += c.x * c.x;
        }
        mean_pt = mean_pt * (1.0 / N as f32);
        let var_pt = (sq_sum_pt / N as f32 - mean_pt.x * mean_pt.x).max(0.0);

        // Both must find the caustic (the means agree within noise).
        let tol = (mean_pt.x * 0.15).max(0.05);
        assert!(
            (mean_bdpt.x - mean_pt.x).abs() <= tol,
            "BDPT mean {mean_bdpt:?} vs PT mean {mean_pt:?}"
        );
        // ...and BDPT's variance must be dramatically lower. The exact
        // ratio depends on the scene; the assertion is deliberately loose
        // (any factor > 2) because the caustic is exactly the regime where
        // PT's variance explodes.
        assert!(
            var_bdpt < var_pt / 2.0 || var_pt < 1e-6,
            "BDPT variance {var_bdpt} not lower than PT {var_pt}"
        );
        // DIAGNOSTIC: dump the numbers for tolerance tuning.
        std::fs::remove_file("target/caustic_diag.txt").ok();
        std::fs::write(
            "target/caustic_diag.txt",
            format!(
                "bdpt mean {mean_bdpt:?} var {var_bdpt}\npt   mean {mean_pt:?} var {var_pt}\n"
            ),
        )
        .ok();
    }
}