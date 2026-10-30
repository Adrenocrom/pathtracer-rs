//! Bidirectional path tracing (BDPT) with multiple-importance sampling.
//!
//! Every surface is diffuse and every direction is sampled with a
//! cosine-weighted distribution. That makes the math pleasantly uniform: the
//! area-measure pdf of *every* path edge is `G / PI` (G = geometry term),
//! no matter which subpath sampled it. As a result each strategy's pdf is
//! the full edge product `P` divided by its one unsampled connection edge,
//! so the MIS weights depend only on simple per-edge ratios. Weights use
//! the power heuristic (beta = 2).
//!
//! Paths are terminated by Russian roulette rather than a short fixed
//! depth: from extension `RR_START` on, a subpath survives each bounce with
//! probability `q = clamp(max(throughput), RR_MIN, 1)` and divides its
//! throughput by `q` on survival, which keeps the estimator unbiased while
//! letting deep paths through. The survival probabilities are part of the
//! sampling pdfs, so they also enter the MIS weights, as the prefix
//! products computed by `rr_prefix_products`.

use crate::geometry::{AreaLight, Intersectable, MaterialType, Ray};
use crate::math::{Vec3, EPS_NEAR, VEC_ZERO};
use rand::Rng;

const PI: f32 = std::f32::consts::PI;
const PI_F64: f64 = std::f64::consts::PI;

/// Hard cap on path length (vertices after the camera). Bounds the fixed-size
/// MIS scratch arrays.
pub const MAX_PATH_VERTICES: usize = 8;

/// Color returned when the primary ray escapes the scene.
pub const BACKGROUND: Vec3 = Vec3 { x: 0.02, y: 0.02, z: 0.05 };

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
    normal: Vec3,
    albedo: Vec3,
    emission: Vec3,
    /// Path throughput up to this vertex, excluding the BSDF applied when
    /// *leaving* it (that factor belongs to the connection term).
    throughput: Vec3,
    /// G(prev, this): the area pdf of this vertex is `edge_g / PI`.
    /// 0 for the first eye vertex (camera edge) and for the light start.
    edge_g: f32,
    on_light: bool,
}

/// Cosine-weighted direction in the hemisphere around `normal` (pdf = cos/pi).
fn cosine_hemisphere(normal: Vec3, rng: &mut impl Rng) -> Vec3 {
    let u1 = rng.gen::<f32>();
    let u2 = rng.gen::<f32>();
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let z = (1.0 - u1).sqrt();

    let a = if normal.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t = a.cross(normal).normalize();
    let b = normal.cross(t);

    t * (r * phi.cos()) + b * (r * phi.sin()) + normal * z
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
/// `edges[m - 2]` holds the geometry term of path edge e_m (m = 2..=k).
/// `inv_light_pdf` is `n_lights * area` of the light the path ends on.
/// `qe[m]` is Qe(m), the product of the Russian-roulette survival
/// probabilities of the eye-subpath prefix x_1..x_m (the rolls at
/// x_1..x_{m-1}); `ql[j]` is Ql(j), the same for the light-subpath prefix
/// y_0..y_j (the rolls at y_0..y_{j-1}). Both come from
/// `rr_prefix_products`; without Russian roulette they are all 1.
///
/// Strategy s' samples every edge except its connection edge e_{s'+1}, and
/// every sampled edge has area pdf `G / PI` times the survival probability
/// of the roll governing it. So with P = prod_{m=2}^{k} (e_m / PI):
///   p[s'] = (1 / inv_light_pdf) * P / (e_{s'+1} / PI) * Qe(s') * Ql(k-s'-1)  for s' < k
///   p[k]  = P * Qe(k)                                                        (eye subpath hits light)
/// All strategies share the factor P, so the weights depend only on the
/// ratios r[s'] = p[s'] / P.
fn mis_weight(edges: &[f64], qe: &[f64], ql: &[f64], k: usize, inv_light_pdf: f64, s: usize) -> f32 {
    debug_assert!(edges.len() + 1 == k);
    debug_assert!(qe.len() > k && ql.len() + 1 >= k);

    // r[k] = Qe(k); r[m] = PI * Qe(m) * Ql(k-m-1) / (nA * e_{m+1}) for m < k.
    let mut r = [0.0f64; MAX_PATH_VERTICES + 1];
    r[k] = qe[k];
    for m in 1..k {
        let e = edges[m - 1]; // e_{m+1}
        r[m] = if e > 0.0 {
            PI_F64 * qe[m] * ql[k - m - 1] / (inv_light_pdf * e)
        } else {
            0.0
        };
    }

    // Power heuristic (beta = 2), normalized by the largest ratio to keep
    // the squares in a safe floating-point range.
    let mut r_max = 0.0f64;
    for m in 1..=k {
        r_max = r_max.max(r[m]);
    }
    if !(r_max > 0.0) {
        return 1.0;
    }
    let mut sum = 0.0f64;
    for m in 1..=k {
        let q = r[m] / r_max;
        sum += q * q;
    }
    let q = r[s] / r_max;
    (q * q / sum) as f32
}

/// Survival probability for extending a subpath whose post-BSDF throughput
/// is `beta`, on extension number `ext` (1-based: the extension x_l ->
/// x_{l+1} is number l, y_j -> y_{j+1} is number j + 1).
///
/// The caller divides the throughput by the returned q when the path
/// survives, which leaves the expected contribution unchanged.
fn rr_survival(beta: Vec3, ext: usize) -> f32 {
    if ext < RR_START {
        return 1.0;
    }
    beta.x.max(beta.y).max(beta.z).clamp(RR_MIN, 1.0)
}

/// Fill the Russian-roulette prefix products the MIS weights need.
///
/// The path is the eye prefix x_1..x_{n_eye} followed by the light vertices
/// y_{n_light-1}..y_0 (k = n_eye + n_light vertices after the camera). The
/// direct-hit strategy passes the whole eye subpath (`n_light == 0`) plus
/// the light-subpath throughput at the hit point.
///
/// `qe[m]` = Qe(m): product of the survival probabilities of the rolls the
/// *eye* subpath makes at x_1..x_{m-1}, for m = 1..=k.
/// `ql[j]` = Ql(j): product of the rolls the *light* subpath makes at
/// y_0..y_{j-1}, for j = 0..=k-2.
///
/// Both sweeps recompute the compensated throughputs the rolls were made
/// from — those are deterministic functions of the path prefix — so the
/// q's here are exactly the ones the subpath sampling rolled. That also
/// covers the *hypothetical* rolls of strategies that would extend a
/// subpath past where it actually stopped (killed by roulette, escaped, or
/// depth-capped): the albedos those extensions would scatter at belong to
/// the path, so their q's are computable even though the rolls never
/// happened.
fn rr_prefix_products(
    eye: &[PathVertex],
    light_path: &[PathVertex],
    k: usize,
    n_eye: usize,
    n_light: usize,
    alpha_light0: Vec3,
    qe: &mut [f64],
    ql: &mut [f64],
) {
    debug_assert!(n_eye + n_light == k);
    debug_assert!(n_eye >= 1);

    // Eye-side sweep: the roll at x_l governs the extension to x_{l+1}.
    let mut alpha = Vec3::new(1.0, 1.0, 1.0);
    qe[1] = 1.0;
    for l in 1..k {
        let alb = if l <= n_eye {
            eye[l - 1].albedo
        } else {
            light_path[k - l].albedo
        };
        let beta = alpha * alb;
        let q = rr_survival(beta, l);
        qe[l + 1] = qe[l] * q as f64;
        alpha = beta * (1.0 / q);
    }

    // Light-side sweep: the roll at y_j governs the extension to y_{j+1}.
    // The first step has no BSDF at the emitter, hence the PI factor.
    let mut alpha_l = alpha_light0;
    ql[0] = 1.0;
    for j in 0..k.saturating_sub(2) {
        let factor = if j == 0 {
            Vec3::new(PI, PI, PI)
        } else if j < n_light {
            light_path[j].albedo
        } else {
            eye[k - j - 1].albedo
        };
        let beta = alpha_l * factor;
        let q = rr_survival(beta, j + 1);
        ql[j + 1] = ql[j] * q as f64;
        alpha_l = beta * (1.0 / q);
    }
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
        // One-sided surfaces: a back-face hit cannot scatter.
        if hit.normal.dot(cur.direction) > 0.0 {
            break;
        }
        let edge_g = match eye.last() {
            Some(prev) => g_term(prev.p, prev.normal, hit.p, hit.normal),
            None => 0.0,
        };
        let on_light = hit.mat.mat_type == MaterialType::Emissive;
        eye.push(PathVertex {
            p: hit.p,
            normal: hit.normal,
            albedo: hit.mat.albedo,
            emission: hit.mat.emission,
            throughput,
            edge_g,
            on_light,
        });
        if on_light {
            break;
        }
        // Russian roulette for the extension x_l -> x_{l+1} (l = eye.len()):
        // survive with probability q and divide the throughput by q, which
        // keeps the expected contribution of the subpath unchanged.
        let beta = throughput * hit.mat.albedo;
        let q = rr_survival(beta, eye.len());
        if q < 1.0 && rng.gen::<f32>() >= q {
            break;
        }
        throughput = beta * (1.0 / q);
        let dir = cosine_hemisphere(hit.normal, rng);
        cur = Ray {
            origin: hit.p + hit.normal * EPS_NEAR,
            direction: dir,
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
            albedo: VEC_ZERO,
            emission: light.emission,
            throughput: ltp,
            edge_g: 0.0,
            on_light: true,
        });

        // The first light extension (y_0 -> y_1) is unconditional
        // (RR_START >= 2). The emitter has no BSDF, so the edge contributes
        // a factor PI.
        ltp = ltp * PI;
        let mut cur = Ray {
            origin: p0 + n0 * EPS_NEAR,
            direction: cosine_hemisphere(n0, rng),
        };
        // The connection strategies need k = i + j + 2 <= max_depth, so the
        // last light vertex (j = max_depth - 1) could never be connected.
        // Building one vertex fewer avoids a wasted intersection.
        for _ in 1..max_depth.saturating_sub(1) {
            let Some(hit) = scene.intersect(&cur) else { break };
            // Pure emitters do not reflect; back faces cannot scatter.
            if hit.mat.mat_type == MaterialType::Emissive
                || hit.normal.dot(cur.direction) > 0.0
            {
                break;
            }
            let prev = light_path.last().unwrap();
            let edge_g = g_term(prev.p, prev.normal, hit.p, hit.normal);
            light_path.push(PathVertex {
                p: hit.p,
                normal: hit.normal,
                albedo: hit.mat.albedo,
                emission: hit.mat.emission,
                throughput: ltp,
                edge_g,
                on_light: false,
            });
            // Russian roulette for the extension y_j -> y_{j+1}
            // (j = light_path.len() - 1 >= 1, number j + 1). The factor is
            // the vertex's own albedo; the emitter's PI case is handled
            // before the loop. Surviving divides ltp by q, which also
            // applies the factor for the next vertex.
            let j = light_path.len() - 1;
            let beta = ltp * hit.mat.albedo;
            let q = rr_survival(beta, j + 1);
            if q < 1.0 && rng.gen::<f32>() >= q {
                break;
            }
            ltp = beta * (1.0 / q);
            let dir = cosine_hemisphere(hit.normal, rng);
            cur = Ray {
                origin: hit.p + hit.normal * EPS_NEAR,
                direction: dir,
            };
        }
    }

    // ---- Strategy "eye subpath hits a light" ----
    if let Some(last) = eye.last().filter(|v| v.on_light) {
        let k = eye.len();
        let w = match lights.iter().find(|l| l.contains(last.p)) {
            Some(light) => {
                let mut edges = [0.0f64; MAX_PATH_VERTICES];
                for idx in 0..k - 1 {
                    edges[idx] = eye[idx + 1].edge_g as f64;
                }
                let inv_pdf = n_lights as f64 * light.area() as f64;
                // The hypothetical light subpath of the other strategies
                // starts at this hit point with the emitter's throughput
                // (emission * n * A, the inverse light-sampling pdf).
                let mut qe = [0.0f64; MAX_PATH_VERTICES + 1];
                let mut ql = [0.0f64; MAX_PATH_VERTICES + 1];
                rr_prefix_products(&eye, &light_path, k, k, 0, light.emission * inv_pdf as f32, &mut qe, &mut ql);
                mis_weight(&edges[..k - 1], &qe, &ql, k, inv_pdf, k)
            }
            None => 1.0,
        };
        total = total + last.throughput * last.emission * w;
    }

    // ---- Connection strategies: eye vertex x_{i+1} to light vertex y_j ----
    let mut edges = [0.0f64; MAX_PATH_VERTICES];
    for i in 0..eye.len() {
        if eye[i].on_light {
            break; // only the last eye vertex can be emissive
        }
        for j in 0..light_path.len() {
            let k = i + j + 2; // vertices after the camera
            if k > max_depth {
                break;
            }
            let g_conn = g_term(eye[i].p, eye[i].normal, light_path[j].p, light_path[j].normal);
            if g_conn <= 0.0 {
                continue;
            }
            if !visible(eye[i].p, eye[i].normal, light_path[j].p, scene) {
                continue;
            }

            // Contribution: alpha_eye * f(x) * G * f(y) * alpha_light.
            // y_0 is the emitter itself, so no BSDF applies there.
            let f_light = if j == 0 {
                Vec3::new(1.0, 1.0, 1.0)
            } else {
                light_path[j].albedo * (1.0 / PI)
            };
            let contrib = eye[i].throughput
                * (eye[i].albedo * (1.0 / PI))
                * g_conn
                * f_light
                * light_path[j].throughput;

            // Path edges e_2..e_k: eye edges, the connection, light edges.
            for m in 2..=i + 1 {
                edges[m - 2] = eye[m - 1].edge_g as f64;
            }
            edges[i] = g_conn as f64;
            for m in i + 3..=k {
                edges[m - 2] = light_path[k - m + 1].edge_g as f64;
            }

            let mut qe = [0.0f64; MAX_PATH_VERTICES + 1];
            let mut ql = [0.0f64; MAX_PATH_VERTICES + 1];
            rr_prefix_products(&eye, &light_path, k, i + 1, j + 1, light_path[0].throughput, &mut qe, &mut ql);
            let w = mis_weight(&edges[..k - 1], &qe, &ql, k, inv_light_pdf, i + 1);
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

    fn diffuse(albedo: Vec3) -> Material {
        Material { albedo, emission: VEC_ZERO, mat_type: MaterialType::Diffuse }
    }

    fn emissive(emission: Vec3) -> Material {
        Material { albedo: VEC_ZERO, emission, mat_type: MaterialType::Emissive }
    }

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction: direction.normalize() }
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

    #[test]
    fn mis_weight_two_strategies_matches_hand_computation() {
        // k = 2, single edge with G = PI, n*A = 1, no RR (all q = 1):
        // p[1] = 1/(nA) = 1, p[2] = G/PI = 1  =>  both weights are 0.5.
        let ones = [1.0f64; MAX_PATH_VERTICES + 1];
        let w = mis_weight(&[PI_F64], &ones, &ones, 2, 1.0, 1);
        assert!((w - 0.5).abs() < 1e-6);
        let w2 = mis_weight(&[PI_F64], &ones, &ones, 2, 1.0, 2);
        assert!((w2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mis_weights_sum_to_one() {
        // Nontrivial RR prefix products: the normalization must still hold.
        let edges = [0.3, 0.7, 1.2, 0.4];
        let k = edges.len() + 1;
        let inv = 3.5;
        let qe = [1.0, 1.0, 1.0, 0.5, 0.25, 0.125, 0.5, 0.5, 0.5];
        let ql = [1.0, 1.0, 0.8, 0.4, 0.2, 0.8, 0.8, 0.8, 0.8];
        let mut sum = 0.0f32;
        for s in 1..=k {
            sum += mis_weight(&edges, &qe, &ql, k, inv, s);
        }
        assert!((sum - 1.0).abs() < 1e-4, "weights sum to {sum}");
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
        // Path x_1 x_2 x_3 | y_1 y_0 connected at (i=2, j=1):
        // k = 5, n_eye = 3, n_light = 2. All albedos 0.5, emitter
        // throughput 0.1. This exercises both cross-over reads: the eye
        // sweep extends hypothetically through y_1 (l = 4) and the light
        // sweep through x_3 (j = 2).
        let vertex = || PathVertex {
            p: VEC_ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            albedo: Vec3::new(0.5, 0.5, 0.5),
            emission: VEC_ZERO,
            throughput: VEC_ZERO,
            edge_g: 0.0,
            on_light: false,
        };
        let eye = vec![vertex(), vertex(), vertex()];
        let mut y0 = vertex();
        y0.albedo = VEC_ZERO;
        y0.throughput = Vec3::new(0.1, 0.1, 0.1);
        let light_path = vec![y0, vertex()];

        let mut qe = [0.0f64; MAX_PATH_VERTICES + 1];
        let mut ql = [0.0f64; MAX_PATH_VERTICES + 1];
        rr_prefix_products(
            &eye, &light_path, 5, 3, 2,
            Vec3::new(0.1, 0.1, 0.1),
            &mut qe, &mut ql,
        );

        // Eye sweep: ext 1, 2 unconditional; ext 3: beta = 0.5^3 = 0.125;
        // ext 4: alpha = 0.125/0.125 = 1, beta = 1 * albedo(y_1) = 0.5.
        assert!((qe[1] - 1.0).abs() < 1e-12);
        assert!((qe[2] - 1.0).abs() < 1e-12);
        assert!((qe[3] - 1.0).abs() < 1e-12);
        assert!((qe[4] - 0.125).abs() < 1e-6, "qe[4] = {}", qe[4]);
        assert!((qe[5] - 0.0625).abs() < 1e-6, "qe[5] = {}", qe[5]);

        // Light sweep: ext 1 (PI factor, q = 1), ext 2 (albedo(y_1), q = 1),
        // ext 3: factor = albedo(x_3), beta = 0.1 * PI * 0.5 * 0.5.
        assert!((ql[0] - 1.0).abs() < 1e-12);
        assert!((ql[1] - 1.0).abs() < 1e-12);
        assert!((ql[2] - 1.0).abs() < 1e-12);
        let expected = 0.1 * PI * 0.5 * 0.5;
        assert!((ql[3] - expected as f64).abs() < 1e-6, "ql[3] = {}", ql[3]);
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
    /// every emissive surface the eye path hits, with the same cosine
    /// sampling and one-sided-surface rules as `bdpt_trace`. Used as the
    /// oracle for the roulette unbiasedness check.
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
            let Some(hit) = scene.intersect(&cur) else { break };
            if hit.normal.dot(cur.direction) > 0.0 {
                break;
            }
            total = total + throughput * hit.mat.emission;
            if hit.mat.mat_type == MaterialType::Emissive {
                break;
            }
            throughput = throughput * hit.mat.albedo;
            cur = Ray {
                origin: hit.p + hit.normal * EPS_NEAR,
                direction: cosine_hemisphere(hit.normal, rng),
            };
        }
        total
    }

    #[test]
    fn rr_does_not_bias_the_mean() {
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
        // A dark cube between the floor and the light blocks direct light.
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
            Box::new(Cube::new(
                Vec3::new(0.0, 0.5, 0.0),
                1.0,
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
        assert!(mean_shadowed.x < 0.05, "shadowed point too bright: {mean_shadowed:?}");
        // ...while the same point without the occluder is much brighter.
        assert!(
            mean_lit.x > mean_shadowed.x * 3.0,
            "no contrast between shadow and light: shadowed {mean_shadowed:?} lit {mean_lit:?}"
        );
    }

    #[test]
    fn visibility_detects_occlusion() {
        let cube = Cube::new(Vec3::new(0.0, 0.5, 0.0), 1.0, diffuse(Vec3::new(1.0, 1.0, 1.0)));
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
}