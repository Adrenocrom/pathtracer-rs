use crate::math::{BBox, Vec3, EPS_NEAR, VEC_ZERO};
use crate::spectrum::{rgb_to_albedo, sellmeier_n, SpectrumKind};
use rand::Rng;

const PI: f32 = std::f32::consts::PI;

// --- RAY ---
#[derive(Clone)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Minimum positive t at which this ray could enter a bounding box.
    /// Returns None if the ray starts inside the box or can never reach it.
    pub fn min_t_intersect(&self, bbox: &BBox) -> Option<f32> {
        let mut t_min = f32::NEG_INFINITY;

        for i in 0..3 {
            let d = self.direction[i];
            if d == 0.0 {
                // Parallel to this slab: unreachable unless inside it.
                if self.origin[i] < bbox.min[i] || self.origin[i] > bbox.max[i] {
                    return None;
                }
                continue;
            }
            let inv_d = 1.0 / d;
            let t0 = (bbox.min[i] - self.origin[i]) * inv_d;
            let t1 = (bbox.max[i] - self.origin[i]) * inv_d;
            t_min = t_min.max(t0.min(t1));
        }

        if t_min <= EPS_NEAR {
            None
        } else {
            Some(t_min)
        }
    }
}

// --- HIT RECORD ---
pub struct HitRecord {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub mat: Material,
}

// --- MATERIALS ---
/// The scattering model of a surface. Emissive materials ignore the BSDF:
/// emitters are treated as pure light sources that do not reflect.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Bsdf {
    /// Lambertian, sampled with a cosine-weighted distribution.
    Diffuse,
    /// Perfect specular reflection (a delta distribution).
    Mirror,
    /// Perfect specular dielectric (a delta distribution): exact dielectric
    /// Fresnel splits reflection and refraction, total internal reflection
    /// falls back to reflection. `ior` is the interior index of refraction;
    /// the exterior is vacuum (ior = 1), so glass objects must be closed.
    /// `sellmeier` carries the Sellmeier dispersion coefficients
    /// `[B₁, B₂, B₃, C₁, C₂, C₃]` (C in µm²) — all zeros means achromatic
    /// glass with the constant `ior` (paper §9.2: real dielectrics
    /// disperse, n(λ), so a prism fans white light into a spectrum).
    Glass {
        ior: f32,
        sellmeier: [f32; 6],
    },
    /// A GGX microfacet reflection lobe (paper §9.5): the "rough mirror".
    /// `alpha` is the GGX roughness (α = 0 is the perfect-mirror limit);
    /// `f0` is the normal-incidence reflectance (Schlick's F0 — a
    /// mirror-like coating sits near 0.9, a bare dielectric at
    /// ((ior−1)/(ior+1))² ≈ 0.04). Unlike Mirror this is a *continuous*
    /// distribution: connections through glossy vertices are valid
    /// (nonzero eval), so the BDPT blind spot softens on its own.
    Glossy { alpha: f32, f0: f32 },
    /// Tag for a *laser emitter*: an emissive surface that radiates only
    /// into a narrow cone of half-angle `acos(1 − versine)` around its
    /// geometric normal (the beam). Not a scattering model at all — pure
    /// emitters never scatter — but the BSDF slot is the one place the
    /// emission cone can travel with the material, so the BDPT pdf/weight/
    /// gate helpers can read it from any path vertex. Deliberately NOT
    /// delta: the cone has a real (if tiny) solid angle, connections to
    /// the aperture stay valid, and one-sided shading applies (the
    /// aperture is dark from behind).
    ///
    /// The cone is stored as its *versine* (1 − cos θ), computed in f64
    /// by `cone_versine`: at a laser's divergence (θ ~ 10⁻⁴ rad) the f32
    /// `theta.cos()` rounds to exactly 1.0, which would collapse the
    /// cone's solid angle 2π(1 − cos θ) to zero and gate the beam away
    /// entirely. Every consumer uses versine-native forms: the solid
    /// angle is 2π·versine, the gate is |d − n|² ≤ 2·versine (unit
    /// vectors — the component differences are Sterbenz-exact where
    /// 1 − dot cancels catastrophically), and the cone sampler draws
    /// the versine directly.
    Laser { versine: f32 },
}

impl Bsdf {
    /// True for BSDFs that scatter into a single direction (a delta
    /// distribution): they have no continuous density, so connections
    /// through them are invalid and their pdfs cancel in MIS ratios.
    /// `Laser` is excluded: its cone is a continuous (if narrow)
    /// distribution, so the aperture remains a valid connection endpoint.
    /// `Glossy` is excluded too — its lobe is continuous, which is the
    /// whole §9.5 point: connections through it are valid.
    pub fn is_delta(&self) -> bool {
        matches!(self, Bsdf::Mirror | Bsdf::Glass { .. })
    }

    /// The refractive index at `lambda` (nm): Sellmeier's dispersion
    /// formula when the coefficients are set, the constant otherwise.
    /// Only meaningful for `Glass`.
    pub fn glass_n(&self, lambda: f32) -> f32 {
        match self {
            Bsdf::Glass { ior, sellmeier } => {
                if sellmeier.iter().any(|c| *c != 0.0) {
                    sellmeier_n(sellmeier, lambda)
                } else {
                    *ior
                }
            }
            _ => 1.0,
        }
    }
}

/// The versine (1 − cos θ) of a laser's emission-cone half-angle,
/// computed in f64: at a laser's divergence (θ ~ 10⁻⁴ rad) the f32
/// `theta.cos()` rounds to exactly 1.0, which would collapse the cone's
/// solid angle 2π(1 − cos θ) to zero and gate the beam away entirely.
/// The versine is O(θ²/2) — representable down to θ ~ 10⁻¹⁹ in f32.
pub fn cone_versine(theta: f32) -> f32 {
    let t = theta as f64;
    (1.0 - t.cos()) as f32
}

/// Which quantity a subpath transports. The two agree for reflection but
/// differ for transmission: basic radiance L/eta^2 is invariant across a
/// refractive boundary, so radiance picks up (eta_t/eta_i)^2 crossing one
/// (Veach's thesis; pbrt's TransportMode::Radiance factor), while
/// importance does not. The eye subpath carries radiance, the light
/// subpath importance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transport {
    Radiance,
    Importance,
}

/// A surface material. `spectrum != None` marks a pure emitter: the BSDF
/// is ignored and the surface does not scatter. The emission is a
/// *spectral* quantity (paper §9.2) — a smooth SPD reconstructed from
/// the RGB intent, or a monochromatic line for a laser — evaluated at
/// the path's hero wavelength.
#[derive(Clone, Copy)]
pub struct Material {
    pub albedo: Vec3,
    pub spectrum: SpectrumKind,
    pub bsdf: Bsdf,
}

impl Material {
    pub fn diffuse(albedo: Vec3) -> Self {
        Self {
            albedo,
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Diffuse,
        }
    }

    /// A smooth emitter radiating the RGB `intent` (in expectation — the
    /// SPD is the unique CMF-basis spectrum whose tristimulus integral
    /// is exactly the intent; see `SpectrumKind::smooth_from_rgb`).
    pub fn emissive(emission: Vec3) -> Self {
        Self {
            albedo: VEC_ZERO,
            spectrum: SpectrumKind::smooth_from_rgb(emission),
            bsdf: Bsdf::Diffuse,
        }
    }

    /// An emitter with an explicit spectrum (a laser line, for instance).
    pub fn emissive_spec(spectrum: SpectrumKind) -> Self {
        Self {
            albedo: VEC_ZERO,
            spectrum,
            bsdf: Bsdf::Diffuse,
        }
    }

    pub fn mirror(albedo: Vec3) -> Self {
        Self {
            albedo,
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Mirror,
        }
    }

    /// Achromatic glass with the constant index `ior` (no dispersion).
    pub fn glass(ior: f32) -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Glass {
                ior,
                sellmeier: [0.0; 6],
            },
        }
    }

    /// Dispersive glass from Sellmeier coefficients (paper §9.2). `ior`
    /// is the mid-visible fallback, unused while `sellmeier` is set.
    pub fn glass_sellmeier(ior: f32, sellmeier: [f32; 6]) -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Glass { ior, sellmeier },
        }
    }

    /// A GGX microfacet "rough mirror" (paper §9.5): `alpha` is the
    /// roughness, `f0` the normal-incidence reflectance. The albedo
    /// scales the lobe per wavelength.
    pub fn glossy(albedo: Vec3, alpha: f32, f0: f32) -> Self {
        Self {
            albedo,
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Glossy { alpha, f0 },
        }
    }

    /// A cone-gated emitter with a *smooth* spectrum (an RGB spotlight):
    /// radiates `intent` only into the cone of half-angle `theta` around
    /// the geometric normal. The cone is stored as its versine (see
    /// `Bsdf::Laser`) so the gate survives the tiny physical divergences.
    #[allow(dead_code)]
    pub fn laser(emission: Vec3, theta: f32) -> Self {
        Self {
            albedo: VEC_ZERO,
            spectrum: SpectrumKind::smooth_from_rgb(emission),
            bsdf: Bsdf::Laser {
                versine: cone_versine(theta),
            },
        }
    }

    /// A *laser*: a monochromatic line emitter, cone-gated — the
    /// physically correct replacement for the old RGB trick of putting
    /// 4×10⁸ in one channel (paper §9.2: "a laser is monochromatic, so
    /// λ is deterministic"). All `power` (spectral radiance) radiates at
    /// `lambda` (nm) inside the cone of half-angle `theta`.
    pub fn laser_line(lambda: f32, power: f32, theta: f32) -> Self {
        Self {
            albedo: VEC_ZERO,
            spectrum: SpectrumKind::Line { lambda, power },
            bsdf: Bsdf::Laser {
                versine: cone_versine(theta),
            },
        }
    }

    /// A laser-aperture material from the cone's versine directly — the
    /// y_0 vertex material in bdpt.rs, whose cone comes from the
    /// AreaLight's stored versine (no θ round-trip, which would lose
    /// the tiny physical cones to the f32 rounding). The spectrum is
    /// the same dummy white smooth SPD as `Material::laser`: the y_0
    /// vertex's emission is read from the light's spectrum, never the
    /// material's.
    pub fn laser_cone(versine: f32) -> Self {
        Self {
            albedo: VEC_ZERO,
            spectrum: SpectrumKind::smooth_from_rgb(Vec3::new(1.0, 1.0, 1.0)),
            bsdf: Bsdf::Laser { versine },
        }
    }

    /// True when this material radiates into a cone (a laser aperture).
    pub fn is_laser(&self) -> bool {
        matches!(self.bsdf, Bsdf::Laser { .. })
    }

    /// True when the direction `p -> target` lies inside this laser
    /// material's emission cone around `normal` (the beam). The cone is
    /// closed: a direction exactly on the boundary counts as inside.
    /// False for non-laser materials.
    ///
    /// Versine form: for unit vectors the gate is
    /// |d − n|² = 2(1 − d·n) ≤ 2·versine — the component differences
    /// are Sterbenz-exact where `1 − dot` cancels catastrophically at
    /// the tiny divergences.
    pub fn in_beam(&self, normal: Vec3, p: Vec3, target: Vec3) -> bool {
        match self.bsdf {
            Bsdf::Laser { versine } => {
                let d = target - p;
                let len = d.length();
                if len < 1e-8 {
                    return true;
                }
                let u = d * (1.0 / len);
                let dx = u.x - normal.x;
                let dy = u.y - normal.y;
                let dz = u.z - normal.z;
                dx * dx + dy * dy + dz * dz <= 2.0 * versine
            }
            _ => false,
        }
    }

    /// Pure emitters do not scatter; the BSDF is ignored for them.
    pub fn is_emissive(&self) -> bool {
        self.spectrum.is_emissive()
    }

    /// Delta BSDFs cannot be connection endpoints. Emitters are never
    /// delta (they do not scatter at all).
    pub fn is_delta(&self) -> bool {
        !self.is_emissive() && self.bsdf.is_delta()
    }

    /// The spectral reflectance at `lambda`: the RGB albedo intent
    /// reconstructed through the complementary-ramp partition of unity
    /// (grays exact, energy conservation automatic; see
    /// `spectrum::rgb_to_albedo`).
    fn rho(&self, lambda: f32) -> f32 {
        rgb_to_albedo(self.albedo, lambda)
    }
}

/// One scattering event sampled from a BSDF.
pub struct BsdfSample {
    /// Unit travel direction away from the surface.
    pub dir_out: Vec3,
    /// Solid-angle pdf of `dir_out`. 0 for delta BSDFs: their density is a
    /// Dirac delta, which the MIS machinery replaces symbolically.
    pub pdf_omega: f32,
    /// f · |cos| / pdf, the throughput multiplier of the event — a
    /// *spectral* scalar at the path's hero wavelength. For delta
    /// BSDFs the delta cancels and this is the deterministic specular
    /// weight; the Fresnel probability cancels against the BSDF value.
    pub weight: f32,
}

/// Cosine-weighted direction in the hemisphere around `normal` (pdf = cos/pi).
pub(crate) fn cosine_hemisphere(normal: Vec3, rng: &mut impl Rng) -> Vec3 {
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

/// Exact unpolarized dielectric Fresnel reflectance for an incidence
/// cosine `cos_i` (measured against the normal on the incident side).
pub(crate) fn fresnel_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cos_i = cos_i.abs().min(1.0);
    let sin_i = (1.0 - cos_i * cos_i).sqrt();
    let sin_t = eta_i / eta_t * sin_i;
    if sin_t >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    let r_s = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    let r_p = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    0.5 * (r_s * r_s + r_p * r_p)
}

/// Snell refraction of the travel direction `dir_in` at a surface whose
/// (unit) `normal` is oriented toward the side the ray arrives from.
/// None = total internal reflection.
pub(crate) fn refract_dir(dir_in: Vec3, normal: Vec3, eta_i: f32, eta_t: f32) -> Option<Vec3> {
    let cos_i = -normal.dot(dir_in);
    let eta = eta_i / eta_t;
    let sin_t_sq = eta * eta * (1.0 - cos_i * cos_i);
    if sin_t_sq >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin_t_sq).sqrt();
    Some(dir_in * eta + normal * (eta * cos_i - cos_t))
}

/// Specular reflection of the travel direction `dir_in` (unit, with `normal`
/// oriented against it).
pub(crate) fn reflect_dir(dir_in: Vec3, normal: Vec3) -> Vec3 {
    dir_in - normal * (2.0 * normal.dot(dir_in))
}

// --- GGX microfacet BSDF (paper §9.5) ---
//
// The "rough mirror": a GGX lobe replaces the perfect mirror's delta.
// The convention matches the rest of the file — `dir_in` is the *travel*
// direction (normal · dir_in < 0), `normal` orients the hemisphere — so
// the microfacet half-vector is h = normalize(−dir_in + dir_out) and the
// reflection about it is dir_out = dir_in − 2(dir_in·h)h (which reduces
// to reflect_dir at h = normal).

/// The GGX normal distribution D(h) for roughness `alpha` and
/// cos θ_h = normal·h.
fn ggx_d(alpha: f32, cos_h: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = cos_h * cos_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

/// The Smith GGX masking term G1 for roughness `alpha` and
/// cos θ_v = normal·v (Walter et al. 2007, eq. 99):
/// G1 = 2c / (c + sqrt(α² + c²(1−α²))).
fn ggx_g1(alpha: f32, cos_v: f32) -> f32 {
    let c = cos_v.clamp(1e-6, 1.0);
    let a2 = alpha * alpha;
    2.0 * c / (c + (a2 + c * c * (1.0 - a2)).sqrt())
}

/// Schlick's Fresnel approximation from the normal-incidence
/// reflectance `f0` at incidence cosine `cos`.
fn schlick_f(f0: f32, cos: f32) -> f32 {
    let c = cos.clamp(0.0, 1.0);
    f0 + (1.0 - f0) * (1.0 - c).powi(5)
}

/// Sample a GGX half-vector around `normal` for roughness `alpha`
/// (the standard NDF inversion: cos θ_h = sqrt((1−u)/(1+(α²−1)u))).
fn ggx_sample_half_vector(alpha: f32, normal: Vec3, rng: &mut impl Rng) -> Vec3 {
    let u = rng.gen::<f32>().clamp(1e-6, 1.0 - 1e-6);
    let v = rng.gen::<f32>();
    let a2 = alpha * alpha;
    let cos_h = ((1.0 - u) / (1.0 + (a2 - 1.0) * u)).sqrt();
    let sin_h = (1.0 - cos_h * cos_h).sqrt();
    let phi = 2.0 * PI * v;
    let a = if normal.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t = a.cross(normal).normalize();
    let b = normal.cross(t);
    t * (sin_h * phi.cos()) + b * (sin_h * phi.sin()) + normal * cos_h
}

/// The GGX microfacet BSDF value f (no cosine) and its NDF-sampling pdf
/// for the direction pair, at spectral reflectance `rho`. `normal` is
/// oriented toward the side `dir_in` arrives from. (0, 0) outside the
/// hemisphere — the same convention as the diffuse eval, and consistent
/// with the sampler: a below-horizon draw is possible at high roughness
/// and simply carries no light.)
fn ggx_eval_pdf(
    alpha: f32,
    f0: f32,
    rho: f32,
    normal: Vec3,
    dir_in: Vec3,
    dir_out: Vec3,
) -> (f32, f32) {
    let cos_i = -normal.dot(dir_in);
    let cos_o = normal.dot(dir_out);
    if cos_i <= 0.0 || cos_o <= 0.0 {
        return (0.0, 0.0);
    }
    let h = (dir_out - dir_in).normalize();
    let cos_h = normal.dot(h);
    if cos_h <= 0.0 {
        return (0.0, 0.0);
    }
    let v_dot_h = -(dir_in.dot(h)); // ω_i·h, positive on a reflection pair
    if v_dot_h <= 0.0 {
        return (0.0, 0.0);
    }
    let d = ggx_d(alpha, cos_h);
    let g = ggx_g1(alpha, cos_i) * ggx_g1(alpha, cos_o);
    let f = schlick_f(f0, v_dot_h);
    let bsdf = rho * d * g * f / (4.0 * cos_i * cos_o);
    let pdf = d * cos_h / (4.0 * v_dot_h);
    (bsdf, pdf)
}

/// The eta^2 transmission factor for the transport mode: radiance picks
/// up (eta_t/eta_i)^2 crossing a refractive boundary, importance does not.
fn eta_factor(transport: Transport, eta_i: f32, eta_t: f32) -> f32 {
    if transport == Transport::Radiance {
        (eta_t / eta_i) * (eta_t / eta_i)
    } else {
        1.0
    }
}

impl Material {
    /// Sample a scattering direction. `normal` must be oriented toward the
    /// side the ray arrives from (normal · dir_in < 0); `front_face` says
    /// whether the geometric normal faces that side (for glass: which
    /// medium is incident); `lambda` is the path's hero wavelength (nm) —
    /// the albedo evaluates to ρ(λ) and dispersive glass to n(λ). Delta
    /// BSDFs sample deterministically.
    pub fn sample_bsdf(
        &self,
        normal: Vec3,
        front_face: bool,
        dir_in: Vec3,
        transport: Transport,
        lambda: f32,
        rng: &mut impl Rng,
    ) -> BsdfSample {
        let rho = self.rho(lambda);
        match self.bsdf {
            Bsdf::Diffuse => {
                let dir_out = cosine_hemisphere(normal, rng);
                let pdf = dir_out.dot(normal) / PI;
                BsdfSample {
                    dir_out,
                    pdf_omega: pdf,
                    weight: rho,
                }
            }
            Bsdf::Mirror => BsdfSample {
                dir_out: reflect_dir(dir_in, normal),
                pdf_omega: 0.0,
                weight: rho,
            },
            Bsdf::Glass { .. } => {
                let n = self.bsdf.glass_n(lambda);
                let (eta_i, eta_t) = if front_face { (1.0, n) } else { (n, 1.0) };
                let cos_i = -normal.dot(dir_in);
                let f = fresnel_dielectric(cos_i, eta_i, eta_t);
                if rng.gen::<f32>() < f {
                    // Reflection. The Fresnel reflectance is both the
                    // discrete choice probability and the BSDF value, so
                    // it cancels out of the weight.
                    BsdfSample {
                        dir_out: reflect_dir(dir_in, normal),
                        pdf_omega: 0.0,
                        weight: rho,
                    }
                } else {
                    // Refraction (total internal reflection cannot reach
                    // this branch: it gives f = 1, so the roll reflects).
                    let dir_out = refract_dir(dir_in, normal, eta_i, eta_t)
                        .unwrap_or_else(|| reflect_dir(dir_in, normal));
                    BsdfSample {
                        dir_out,
                        pdf_omega: 0.0,
                        weight: rho * eta_factor(transport, eta_i, eta_t),
                    }
                }
            }
            Bsdf::Glossy { alpha, f0 } => {
                let h = ggx_sample_half_vector(alpha, normal, rng);
                let dir_out = dir_in - h * (2.0 * dir_in.dot(h));
                let (f, pdf) = ggx_eval_pdf(alpha, f0, rho, normal, dir_in, dir_out);
                BsdfSample {
                    dir_out,
                    pdf_omega: pdf,
                    // f·cos/pdf; 0 for a below-horizon draw (f = 0 there —
                    // the draw is merely wasted, exactly like a bridge
                    // direction that dips below the horizon).
                    weight: if pdf > 0.0 {
                        f * normal.dot(dir_out) / pdf
                    } else {
                        0.0
                    },
                }
            }
            // A laser material is a pure emitter: it never scatters, so
            // this arm is unreachable (the subpath walks stop at
            // emissive surfaces). Zero weight keeps any hypothetical
            // extension through such a vertex at zero contribution.
            Bsdf::Laser { .. } => BsdfSample {
                dir_out: normal,
                pdf_omega: 0.0,
                weight: 0.0,
            },
        }
    }

    /// f (the raw BSDF, no cosine) and the solid-angle pdf of `dir_out`
    /// under the BSDF's sampling distribution, for an arbitrary direction
    /// pair at wavelength `lambda`. Delta BSDFs return (0, 0) — exactly
    /// what invalidates connections through them. `normal` must be
    /// oriented toward the side `dir_in` arrives from.
    pub fn eval_bsdf(
        &self,
        normal: Vec3,
        dir_in: Vec3,
        dir_out: Vec3,
        lambda: f32,
    ) -> (f32, f32) {
        match self.bsdf {
            Bsdf::Diffuse => {
                let cos_i = -normal.dot(dir_in);
                let cos_o = normal.dot(dir_out);
                if cos_i > 0.0 && cos_o > 0.0 {
                    (self.rho(lambda) / PI, cos_o / PI)
                } else {
                    (0.0, 0.0)
                }
            }
            Bsdf::Glossy { alpha, f0 } => {
                ggx_eval_pdf(alpha, f0, self.rho(lambda), normal, dir_in, dir_out)
            }
            _ => (0.0, 0.0),
        }
    }

    /// True when (dir_in, dir_out) is the specular pair of this delta BSDF
    /// (the deterministic direction it scatters into). Used by the MIS
    /// machinery to give delta-sampled edges their symbolic pdf of 1.
    /// `lambda` is the path's hero wavelength — dispersive glass picks
    /// its refraction direction per-λ, so the pair test must use the
    /// same λ the walk used (the WGSL port passes `view.lambda`; a
    /// hardcoded 550 here would diverge from it on dispersive glass).
    pub fn is_specular_pair(
        &self,
        normal: Vec3,
        front_face: bool,
        dir_in: Vec3,
        dir_out: Vec3,
        lambda: f32,
    ) -> bool {
        self.specular_weight(normal, front_face, dir_in, dir_out, Transport::Radiance, lambda)
            .is_some()
    }

    /// The deterministic sample weight of a SPECULAR direction pair at a
    /// delta BSDF — needed for hypothetical path extensions, where the
    /// rolls never happened but the path geometry fixes the directions.
    /// None if the pair is not the specular pair. `normal`/`front_face`
    /// describe the event being weighted; `lambda` evaluates dispersive
    /// glass (the hypothetical extension uses the same hero wavelength
    /// as the realized walk — λ is fixed per path, so the PathView
    /// discipline is preserved).
    pub fn specular_weight(
        &self,
        normal: Vec3,
        front_face: bool,
        dir_in: Vec3,
        dir_out: Vec3,
        transport: Transport,
        lambda: f32,
    ) -> Option<f32> {
        const DOT_EPS: f32 = 1.0 - 1e-4;
        let rho = self.rho(lambda);
        match self.bsdf {
            Bsdf::Mirror => {
                if reflect_dir(dir_in, normal).dot(dir_out) > DOT_EPS {
                    Some(rho)
                } else {
                    None
                }
            }
            Bsdf::Glass { .. } => {
                let n = self.bsdf.glass_n(lambda);
                let (eta_i, eta_t) = if front_face { (1.0, n) } else { (n, 1.0) };
                if reflect_dir(dir_in, normal).dot(dir_out) > DOT_EPS {
                    return Some(rho);
                }
                let refr = refract_dir(dir_in, normal, eta_i, eta_t)?;
                if refr.dot(dir_out) > DOT_EPS {
                    Some(rho * eta_factor(transport, eta_i, eta_t))
                } else {
                    None
                }
            }
            // Pure emitters never scatter; glossy has no specular pair
            // (its density is continuous — the MIS machinery uses
            // `eval_bsdf` for it instead).
            Bsdf::Laser { .. } | Bsdf::Diffuse | Bsdf::Glossy { .. } => None,
        }
    }
}

// --- INTERSECTABLE TRAIT ---
pub trait Intersectable: Sync + Send {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord>;
    fn bounding_box(&self) -> BBox;
}

// --- SPHERE ---
#[derive(Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub mat: Material,
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        // Prefer the near root; fall back to the far root so the sphere is
        // still visible when the camera is inside it.
        let sqrt_d = discriminant.sqrt();
        let mut t = (-b - sqrt_d) / (2.0 * a);
        if t < EPS_NEAR {
            t = (-b + sqrt_d) / (2.0 * a);
        }
        if t < EPS_NEAR {
            return None;
        }

        let p = ray.at(t);
        let normal = (p - self.center).normalize();
        Some(HitRecord {
            t,
            p,
            normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        BBox {
            min: self.center - Vec3::new(self.radius, self.radius, self.radius),
            max: self.center + Vec3::new(self.radius, self.radius, self.radius),
        }
    }
}

// --- PLANE ---
#[derive(Clone)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
    pub mat: Material,
}

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < EPS_NEAR {
            return None;
        }

        Some(HitRecord {
            t,
            p: ray.at(t),
            normal: self.normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        // Planes have infinite extent - use a large box
        BBox {
            min: Vec3::new(-1e6, -1e6, -1e6),
            max: Vec3::new(1e6, 1e6, 1e6),
        }
    }
}

// --- QUAD ---
/// Tangent frame for a `width` x `height` rectangle facing `normal`.
///
/// For axis-aligned normals the extents are world-axis sized: `width`
/// spans the horizontal axis (x for normals along z or y, z for normals
/// along x) and `height` spans the vertical axis (y, or z for normals
/// along y). Sizing the extents along an arbitrary helper-axis tangent
/// frame instead *rotates* walls: a "5 wide, 2 tall" wall facing +z came
/// out 2 wide and 5 tall, because the helper axis for a +z normal is
/// vertical.
///
/// Non-axis-aligned normals have no world-axis mapping; they fall back
/// to a helper axis perpendicular to the normal, with `width` along the
/// tangent and `height` along the bitangent.
///
/// Returns `(tangent, bitangent, half_w, half_h)`.
pub(crate) fn quad_frame(normal: Vec3, width: f32, height: f32) -> (Vec3, Vec3, f32, f32) {
    // An axis normal is exactly unit length along its axis after
    // normalization, so a near-1 threshold separates it from diagonals.
    const AXIS: f32 = 1.0 - 1e-3;
    if normal.y.abs() > AXIS {
        // Floor/ceiling: width along x, height along z.
        (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            width * 0.5,
            height * 0.5,
        )
    } else if normal.x.abs() > AXIS {
        // Side walls: width along z, height along y.
        (
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            width * 0.5,
            height * 0.5,
        )
    } else if normal.z.abs() > AXIS {
        // Back/front walls: width along x, height along y.
        (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            width * 0.5,
            height * 0.5,
        )
    } else {
        // Diagonal normal: |normal.x| <= 1 - 1e-3, so the x axis is
        // never parallel to it and works as the helper axis.
        let tangent = Vec3::new(1.0, 0.0, 0.0).cross(normal).normalize();
        let bitangent = normal.cross(tangent);
        (tangent, bitangent, width * 0.5, height * 0.5)
    }
}

/// A finite rectangle, used for light panels.
#[derive(Clone)]
pub struct Quad {
    pub center: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub half_w: f32,
    pub half_h: f32,
    pub mat: Material,
}

impl Quad {
    /// A `width` x `height` rectangle centered at `center`, facing `normal`.
    /// Kept for API completeness (used by tests and future scenes).
    #[allow(dead_code)]
    pub fn new(center: Vec3, normal: Vec3, width: f32, height: f32, mat: Material) -> Self {
        let normal = normal.normalize();
        let (tangent, bitangent, half_w, half_h) = quad_frame(normal, width, height);
        Self {
            center,
            normal,
            tangent,
            bitangent,
            half_w,
            half_h,
            mat,
        }
    }
}

impl Intersectable for Quad {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (self.center - ray.origin).dot(self.normal) / denom;
        if t < EPS_NEAR {
            return None;
        }

        let p = ray.at(t);
        let local = p - self.center;
        if local.dot(self.tangent).abs() > self.half_w
            || local.dot(self.bitangent).abs() > self.half_h
        {
            return None;
        }
        Some(HitRecord {
            t,
            p,
            normal: self.normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        let mut bbox = BBox::empty();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                let corner = self.center
                    + self.tangent * (sx * self.half_w)
                    + self.bitangent * (sy * self.half_h);
                bbox = bbox.surround(BBox {
                    min: corner,
                    max: corner,
                });
            }
        }
        bbox
    }
}

// --- CUBE ---
#[derive(Clone)]
pub struct Cube {
    pub min: Vec3,
    pub max: Vec3,
    pub mat: Material,
}

impl Cube {
    /// Kept for API completeness (used by tests and future scenes).
    #[allow(dead_code)]
    pub fn new(center: Vec3, size: f32, mat: Material) -> Self {
        let half = size * 0.5;
        Self {
            min: center - Vec3::new(half, half, half),
            max: center + Vec3::new(half, half, half),
            mat,
        }
    }
}

impl Intersectable for Cube {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        let dirs = [ray.direction.x, ray.direction.y, ray.direction.z];
        let origins = [ray.origin.x, ray.origin.y, ray.origin.z];
        let mins = [self.min.x, self.min.y, self.min.z];
        let maxs = [self.max.x, self.max.y, self.max.z];

        for i in 0..3 {
            if dirs[i] == 0.0 {
                // Parallel to this slab: must lie within it.
                if origins[i] < mins[i] || origins[i] > maxs[i] {
                    return None;
                }
                continue;
            }
            let inv_d = 1.0 / dirs[i];
            let mut t0 = (mins[i] - origins[i]) * inv_d;
            let mut t1 = (maxs[i] - origins[i]) * inv_d;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
        }

        if !(t_max >= t_min && t_max > 0.0 && t_min < 1e6) {
            return None;
        }
        let t = if t_min < EPS_NEAR { t_max } else { t_min };
        if t < EPS_NEAR {
            return None;
        }

        let p = ray.at(t);

        // Outward normal of the face closest to the hit point. Checking all
        // six faces (not just the min faces) is what makes top/+x/+z hits
        // shade correctly.
        let candidates = [
            ((p.x - self.min.x).abs(), Vec3::new(-1.0, 0.0, 0.0)),
            ((p.x - self.max.x).abs(), Vec3::new(1.0, 0.0, 0.0)),
            ((p.y - self.min.y).abs(), Vec3::new(0.0, -1.0, 0.0)),
            ((p.y - self.max.y).abs(), Vec3::new(0.0, 1.0, 0.0)),
            ((p.z - self.min.z).abs(), Vec3::new(0.0, 0.0, -1.0)),
            ((p.z - self.max.z).abs(), Vec3::new(0.0, 0.0, 1.0)),
        ];
        let normal = candidates
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, n)| *n)
            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

        Some(HitRecord {
            t,
            p,
            normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        BBox {
            min: self.min,
            max: self.max,
        }
    }
}

// --- CYLINDER ---
/// A closed cylinder: two flat caps plus the curved side. `axis` is the
/// unit direction from the base cap to the top cap; `half_h` is the
/// half-height along it. The caps are one-sided (they shade only from
/// outside the cylinder), the curved side shades from both sides.
#[derive(Clone)]
pub struct Cylinder {
    pub base: Vec3,
    pub axis: Vec3,
    pub radius: f32,
    pub half_h: f32,
    pub mat: Material,
}

impl Cylinder {
    /// A cylinder centered at `center` with its axis along `axis` (any
    /// length; normalized internally), total height `height` and the
    /// given radius. Kept for API completeness (used by tests and future
    /// scenes), like `Quad::new`/`Cube::new`.
    #[allow(dead_code)]
    pub fn new(center: Vec3, axis: Vec3, height: f32, radius: f32, mat: Material) -> Self {
        let axis = axis.normalize();
        Self {
            base: center - axis * (height * 0.5),
            axis,
            radius,
            half_h: height * 0.5,
            mat,
        }
    }

    /// The center of the top cap (base + axis * height). Used by tests
    /// and future scenes to place the aperture.
    #[allow(dead_code)]
    pub fn top_center(&self) -> Vec3 {
        self.base + self.axis * (2.0 * self.half_h)
    }
}

impl Intersectable for Cylinder {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        // Work in the cylinder's local frame: origin at the base cap,
        // z along the axis. The ray is decomposed into its axial part
        // (z) and its radial part (x, y); the side test is then a 2D
        // circle intersection in the radial plane.
        let oc = ray.origin - self.base;
        let d = ray.direction;
        let a = self.axis;
        let d_perp = d - a * d.dot(a); // radial component of the direction
        let oc_perp = oc - a * oc.dot(a); // radial component of the offset
        let dz = d.dot(a);
        let oz = oc.dot(a);

        let mut best_t = f32::INFINITY;
        let mut best_normal = VEC_ZERO;

        // --- Curved side: |oc_perp + t * d_perp| = radius ---
        // The axial coordinate z is measured from the base cap, so the
        // cylinder spans z in [0, 2 * half_h].
        let aa = d_perp.length_squared();
        if aa > 1e-12 {
            let bb = 2.0 * oc_perp.dot(d_perp);
            let cc = oc_perp.length_squared() - self.radius * self.radius;
            let disc = bb * bb - 4.0 * aa * cc;
            if disc >= 0.0 {
                let sqrt_d = disc.sqrt();
                for t in [(-bb - sqrt_d) / (2.0 * aa), (-bb + sqrt_d) / (2.0 * aa)] {
                    if t < EPS_NEAR || t >= best_t {
                        continue;
                    }
                    let z = oz + t * dz;
                    if z < 0.0 || z > 2.0 * self.half_h {
                        continue; // outside the side's height range
                    }
                    let p = ray.at(t);
                    let n = (p - self.base - a * z).normalize();
                    best_t = t;
                    best_normal = n;
                }
            }
        }

        // --- Caps: two one-sided discs at z = 0 (base) and z = 2*half_h (top) ---
        for (cap_z, cap_n) in [(0.0, a * -1.0), (2.0 * self.half_h, a)] {
            let cap_point = self.base + a * cap_z;
            let denom = cap_n.dot(d);
            if denom.abs() < 1e-6 {
                continue;
            }
            let t = (cap_point - ray.origin).dot(cap_n) / denom;
            if t < EPS_NEAR || t >= best_t {
                continue;
            }
            let p = ray.at(t);
            let radial = p - cap_point;
            if radial.length_squared() > self.radius * self.radius {
                continue;
            }
            best_t = t;
            best_normal = cap_n;
        }

        if best_t.is_infinite() {
            return None;
        }
        Some(HitRecord {
            t: best_t,
            p: ray.at(best_t),
            normal: best_normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        // Exact box via the support function: along world axis e_k the
        // cylinder's extent is |a_k| * half_h (the caps) plus
        // r * sqrt(1 - a_k²) (the disc radius projected onto e_k),
        // where a is the unit axis. Centered on the cylinder's midpoint.
        let mid = self.base + self.axis * self.half_h;
        let r = self.radius;
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for k in 0..3 {
            let a_k = self.axis[k];
            let extent = a_k.abs() * self.half_h + r * (1.0 - a_k * a_k).sqrt();
            min[k] = mid[k] - extent;
            max[k] = mid[k] + extent;
        }
        BBox {
            min: Vec3::new(min[0], min[1], min[2]),
            max: Vec3::new(max[0], max[1], max[2]),
        }
    }
}

// --- AREA LIGHTS ---
/// A sampleable emitter surface. The shape parameters must match an emissive
/// object in the scene so that BDPT can connect paths to real geometry.
#[derive(Clone)]
pub enum LightShape {
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Quad {
        center: Vec3,
        normal: Vec3,
        tangent: Vec3,
        bitangent: Vec3,
        half_w: f32,
        half_h: f32,
    },
    /// A laser aperture: a disc of radius `radius` centered at `center`
    /// with outward normal `normal`, radiating only into the cone of
    /// half-angle `acos(1 − versine)` around `normal`. The disc is the
    /// top cap of the laser's cylinder housing. The cone is stored as
    /// its versine (1 − cos θ, computed in f64 — see `Bsdf::Laser`): at
    /// a laser's divergence the f32 cosine rounds to 1.0 and the cone
    /// would collapse.
    Laser {
        center: Vec3,
        normal: Vec3,
        radius: f32,
        versine: f32,
    },
}

#[derive(Clone)]
pub struct AreaLight {
    pub shape: LightShape,
    /// The emission spectrum (paper §9.2): a smooth SPD or a
    /// monochromatic line, evaluated at the path's hero wavelength.
    pub spectrum: SpectrumKind,
}

impl AreaLight {
    pub fn from_sphere(sphere: &Sphere) -> Self {
        Self {
            shape: LightShape::Sphere {
                center: sphere.center,
                radius: sphere.radius,
            },
            spectrum: sphere.mat.spectrum,
        }
    }

    pub fn from_quad(quad: &Quad) -> Self {
        Self {
            shape: LightShape::Quad {
                center: quad.center,
                normal: quad.normal,
                tangent: quad.tangent,
                bitangent: quad.bitangent,
                half_w: quad.half_w,
                half_h: quad.half_h,
            },
            spectrum: quad.mat.spectrum,
        }
    }

    /// A laser aperture light: a disc of `radius` centered at `center`
    /// with outward normal `normal`, radiating into the cone of
    /// half-angle `theta` around the normal. The disc is the emitting
    /// cap of the laser's cylinder housing (see `SceneData::add_laser`).
    /// The cone is stored as its versine (see `Bsdf::Laser`).
    #[allow(dead_code)]
    pub fn from_laser(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        theta: f32,
        spectrum: SpectrumKind,
    ) -> Self {
        Self::from_laser_versine(center, normal, radius, cone_versine(theta), spectrum)
    }

    /// A laser aperture light in versine form (see `Bsdf::Laser` for why
    /// the cone travels as its versine): the versine is computed once,
    /// in f64, at the scene-construction site — never re-derived from a
    /// cosine at a consumer.
    pub fn from_laser_versine(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        versine: f32,
        spectrum: SpectrumKind,
    ) -> Self {
        Self {
            shape: LightShape::Laser {
                center,
                normal: normal.normalize(),
                radius,
                versine,
            },
            spectrum,
        }
    }

    pub fn area(&self) -> f32 {
        match &self.shape {
            LightShape::Sphere { radius, .. } => 4.0 * PI * radius * radius,
            LightShape::Quad { half_w, half_h, .. } => 4.0 * half_w * half_h,
            LightShape::Laser { radius, .. } => PI * radius * radius,
        }
    }

    /// Uniform surface sample: (point, normal). The pdf is 1 / area.
    pub fn sample(&self, rng: &mut impl Rng) -> (Vec3, Vec3) {
        match &self.shape {
            LightShape::Sphere { center, radius } => {
                // Uniform point on the sphere via spherical coordinates.
                let z = 1.0 - 2.0 * rng.gen::<f32>(); // cos theta in [-1, 1]
                let phi = 2.0 * PI * rng.gen::<f32>();
                let r = (1.0 - z * z).sqrt();
                let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                (*center + dir * *radius, dir)
            }
            LightShape::Quad {
                center,
                normal,
                tangent,
                bitangent,
                half_w,
                half_h,
            } => {
                let u = rng.gen::<f32>() * 2.0 - 1.0;
                let v = rng.gen::<f32>() * 2.0 - 1.0;
                let p = *center + *tangent * (u * half_w) + *bitangent * (v * half_h);
                (p, *normal)
            }
            LightShape::Laser {
                center,
                normal,
                radius,
                ..
            } => {
                // Uniform point on the aperture disc; the normal is the
                // aperture's outward normal (the beam axis). The
                // departure direction is sampled separately by
                // `sample_dir` — uniformly in the cone — because the
                // vertex normal and the departure direction are
                // different vectors for a cone emitter.
                let r = radius * rng.gen::<f32>().sqrt();
                let phi = 2.0 * PI * rng.gen::<f32>();
                let a = if normal.x.abs() > 0.9 {
                    Vec3::new(0.0, 1.0, 0.0)
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let t = a.cross(*normal).normalize();
                let b = normal.cross(t);
                let p = *center + t * (r * phi.cos()) + b * (r * phi.sin());
                (p, *normal)
            }
        }
    }

    /// Sample the direction a light subpath leaves the sampled point in,
    /// around the normal `sample` returned: cosine-weighted for ordinary
    /// emitters (pdf cos/π), uniform in the cone for a laser aperture
    /// (pdf 1/(2π·versine)). Split from `sample` because a cone
    /// emitter's departure direction is not its normal.
    ///
    /// Versine form: the draw is `1 − u·versine` (u uniform in [0,1]),
    /// which is the versine of the drawn angle — exact at every
    /// divergence, where the old `1 − u(1 − cos θ)` form lost the
    /// entire cone to the f32 rounding of cos θ.
    pub fn sample_dir(&self, normal: Vec3, rng: &mut impl Rng) -> Vec3 {
        match &self.shape {
            LightShape::Laser { versine, .. } => {
                let a = if normal.x.abs() > 0.9 {
                    Vec3::new(0.0, 1.0, 0.0)
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let t = a.cross(normal).normalize();
                let b = normal.cross(t);
                // Uniform in the cone: the versine of the drawn angle is
                // uniform in [0, versine] (the versine is the natural
                // measure — cos φ uniform in [cos θ, 1] is exactly this).
                let v = rng.gen::<f32>() * *versine;
                let cos_phi = 1.0 - v;
                let sin_phi = (2.0 * v - v * v).sqrt();
                let phi = 2.0 * PI * rng.gen::<f32>();
                t * (sin_phi * phi.cos()) + b * (sin_phi * phi.sin()) + normal * cos_phi
            }
            _ => cosine_hemisphere(normal, rng),
        }
    }

    /// True if `p` lies on the light surface (within a small epsilon).
    pub fn contains(&self, p: Vec3) -> bool {
        match &self.shape {
            LightShape::Sphere { center, radius } => {
                ((p - *center).length() - *radius).abs() < 1e-3
            }
            LightShape::Quad {
                center,
                normal,
                tangent,
                bitangent,
                half_w,
                half_h,
            } => {
                let local = p - *center;
                local.dot(*normal).abs() < 1e-3
                    && local.dot(*tangent).abs() <= half_w + 1e-3
                    && local.dot(*bitangent).abs() <= half_h + 1e-3
            }
            LightShape::Laser {
                center,
                normal,
                radius,
                ..
            } => {
                let local = p - *center;
                local.dot(*normal).abs() < 1e-3
                    && local.length_squared() <= radius * radius + 1e-3
            }
        }
    }

    /// True when this light is a laser aperture (cone emission) rather
    /// than an ordinary hemisphere emitter.
    pub fn shape_is_laser(&self) -> bool {
        matches!(self.shape, LightShape::Laser { .. })
    }

    /// The laser cone's versine (1 − cos θ). Only meaningful for laser
    /// lights. The versine — not the cosine — is the canonical form:
    /// at a laser's divergence the f32 cosine rounds to 1.0 and the
    /// cone would collapse (see `Bsdf::Laser`).
    pub fn cone_versine(&self) -> f32 {
        match &self.shape {
            LightShape::Laser { versine, .. } => *versine,
            _ => 0.0,
        }
    }

    /// The laser cone's half-angle (radians). Only meaningful for laser
    /// lights; 0 for a degenerate (zero-solid-angle) cone. The inverse
    /// of the versine — used by tests and diagnostics, never on a
    /// transport path (the versine is the canonical form there).
    #[allow(dead_code)]
    pub fn cone_theta(&self) -> f32 {
        match &self.shape {
            LightShape::Laser { versine, .. } => {
                if *versine <= 0.0 {
                    0.0
                } else {
                    (2.0 * *versine as f64).sqrt() as f32
                }
            }
            _ => 0.0,
        }
    }

    /// True when the direction `p -> target` lies inside this laser's
    /// emission cone. Always true for non-laser lights (they radiate
    /// into the whole hemisphere). For a laser, `p` must also lie on
    /// the aperture disc — the rest of the housing is dark.
    pub fn in_beam(&self, p: Vec3, target: Vec3) -> bool {
        match &self.shape {
            LightShape::Laser {
                center,
                normal,
                radius,
                versine,
            } => {
                // p must lie on the aperture disc (the curved side of
                // the housing is dark): the same plane/extent test as
                // `contains`. A distance-from-*center* test here would
                // wrongly reject every off-center point of the disc.
                let local = p - *center;
                if local.dot(*normal).abs() > 1e-3
                    || local.length_squared() > radius * radius + 1e-3
                {
                    return false;
                }
                // ...and the direction p -> target must lie inside the
                // cone (closed: exactly on the boundary counts). The
                // versine gate: |d − n|² = 2(1 − d·n) ≤ 2·versine for
                // unit vectors — Sterbenz-exact where 1 − dot cancels.
                let d = target - p;
                let len = d.length();
                if len < 1e-8 {
                    return true;
                }
                let u = d * (1.0 / len);
                let dx = u.x - normal.x;
                let dy = u.y - normal.y;
                let dz = u.z - normal.z;
                dx * dx + dy * dy + dz * dz <= 2.0 * versine
            }
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::{rgb_to_albedo, BK7};
    use rand::SeedableRng;

    fn diffuse(albedo: Vec3) -> Material {
        Material::diffuse(albedo)
    }

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray {
            origin,
            direction: direction.normalize(),
        }
    }

    #[test]
    fn sphere_hit_from_outside_and_inside() {
        let s = Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            mat: diffuse(Vec3::new(1.0, 1.0, 1.0)),
        };
        let outside = s.intersect(&ray(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)));
        assert!(outside.is_some());
        assert!((outside.unwrap().t - 4.0).abs() < 1e-4);

        // A ray starting at the center must hit the far surface.
        let inside = s.intersect(&ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)));
        assert!(inside.is_some());
        assert!((inside.unwrap().t - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cube_normals_face_outward_on_all_sides() {
        let c = Cube::new(
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        let top = c
            .intersect(&ray(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0)))
            .unwrap();
        assert_eq!(top.normal, Vec3::new(0.0, 1.0, 0.0));

        let right = c
            .intersect(&ray(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)))
            .unwrap();
        assert_eq!(right.normal, Vec3::new(1.0, 0.0, 0.0));

        let front = c
            .intersect(&ray(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)))
            .unwrap();
        assert_eq!(front.normal, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn quad_intersect_respects_bounds() {
        let q = Quad::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            2.0,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        let hit = q.intersect(&ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 2.0).abs() < 1e-4);

        let miss = q.intersect(&ray(Vec3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        assert!(miss.is_none());
    }

    /// The cylinder: side hits, cap hits, misses outside the radius, and
    /// the correct outward normals on all three surface families.
    #[test]
    fn cylinder_intersects_sides_and_caps_with_correct_normals() {
        // A vertical cylinder: base at y=0, height 2, radius 0.5.
        let c = Cylinder::new(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0,
            0.5,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        // Side hit from +x: normal +x, t = 0.5.
        let side = c
            .intersect(&ray(Vec3::new(3.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)))
            .unwrap();
        assert!((side.t - 2.5).abs() < 1e-4);
        assert_eq!(side.normal, Vec3::new(1.0, 0.0, 0.0));
        // Top cap hit from above: normal +y.
        let top = c
            .intersect(&ray(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0)))
            .unwrap();
        assert!((top.t - 3.0).abs() < 1e-4);
        assert_eq!(top.normal, Vec3::new(0.0, 1.0, 0.0));
        // Base cap hit from below: normal -y.
        let base = c
            .intersect(&ray(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0)))
            .unwrap();
        assert_eq!(base.normal, Vec3::new(0.0, -1.0, 0.0));
        // A ray above the top cap misses (outside the height range).
        let above = c.intersect(&ray(Vec3::new(3.0, 2.5, 0.0), Vec3::new(-1.0, 0.0, 0.0)));
        assert!(above.is_none());
        // A ray outside the radius misses the caps.
        let beside = c.intersect(&ray(Vec3::new(2.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0)));
        assert!(beside.is_none());
        // A ray through the cap plane but outside the disc misses.
        let off_disc = c.intersect(&ray(Vec3::new(0.0, 5.0, 2.0), Vec3::new(0.0, -1.0, 0.0)));
        assert!(off_disc.is_none());
        // A tilted axis: the same tests must hold after rotation. The
        // cylinder points along +x now.
        let t = Cylinder::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            2.0,
            0.5,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        let side_t = t
            .intersect(&ray(Vec3::new(1.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0)))
            .unwrap();
        assert_eq!(side_t.normal, Vec3::new(0.0, 1.0, 0.0));
        let cap_t = t
            .intersect(&ray(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)))
            .unwrap();
        assert_eq!(cap_t.normal, Vec3::new(1.0, 0.0, 0.0));
        // The bounding box must be exact: the cylinder is centered at the
        // origin with height 2 along x, so it spans x in [-1, 1] (the
        // cap discs extend radially, in y/z, not along the axis) and
        // y, z in [-0.5, 0.5].
        let bb = t.bounding_box();
        assert!((bb.min.x + 1.0).abs() < 1e-4, "bb.min.x: {}", bb.min.x);
        assert!((bb.max.x - 1.0).abs() < 1e-4, "bb.max.x: {}", bb.max.x);
        assert!((bb.min.y + 0.5).abs() < 1e-4 && (bb.max.y - 0.5).abs() < 1e-4);
        assert!((bb.min.z + 0.5).abs() < 1e-4 && (bb.max.z - 0.5).abs() < 1e-4);
    }

    /// The laser aperture light: samples lie on the disc, the departure
    /// directions lie in the cone, and the cone gate accepts/rejects
    /// directions correctly. Also pins the versine representation at
    /// the tiny physical divergences, where the old f32 cosine form
    /// collapsed the cone entirely (cos(2e-4) rounds to 1.0).
    #[test]
    fn laser_light_samples_disc_and_cone() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let light = AreaLight::from_laser(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            0.1, // ~5.7° half-angle
            SpectrumKind::smooth_from_rgb(Vec3::new(100.0, 100.0, 100.0)),
        );
        assert!(light.shape_is_laser());
        assert!((light.area() - PI * 0.25).abs() < 1e-5);
        for _ in 0..200 {
            let (p, n) = light.sample(&mut rng);
            assert!(light.contains(p));
            assert_eq!(n, Vec3::new(0.0, 1.0, 0.0));
            let d = light.sample_dir(n, &mut rng);
            // Every departure direction lies in the cone (dot with the
            // axis >= cos θ) and is unit length.
            assert!(d.dot(n) >= 1.0 - light.cone_versine() - 1e-5);
            assert!((d.length() - 1.0).abs() < 1e-5);
        }
        // The beam gate: a direction along the axis is inside, one far
        // off-axis is outside.
        let p = Vec3::new(0.0, 1.0, 0.0);
        assert!(light.in_beam(p, p + Vec3::new(0.0, 1.0, 0.0)));
        assert!(!light.in_beam(p, p + Vec3::new(1.0, 0.0, 0.0)));
        // A point off the aperture is not "in beam" (guards the direct
        // hit on the dark housing side).
        assert!(!light.in_beam(Vec3::new(0.0, 0.5, 0.0), p));

        // The physical-divergence regime: a 532 nm beam with a 1 cm
        // waist has θ = λ/(πw₀) ≈ 1.7e-5 rad. The versine must be
        // representable (the f32 cosine rounds to 1.0 there), the cone
        // sampler must still produce in-cone unit directions, and the
        // gate must accept the axis and reject a 1e-3 rad off-axis
        // direction (60x the half-angle).
        let theta = 532.0e-9 / (PI * 0.01);
        let tight = AreaLight::from_laser(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.01,
            theta,
            SpectrumKind::Line {
                lambda: 532.0,
                power: 1.0,
            },
        );
        let versine = tight.cone_versine();
        assert!(
            versine > 0.0 && versine < 1e-8,
            "the diffraction-limited versine must be tiny but nonzero: {versine}"
        );
        assert!((tight.cone_theta() - theta).abs() < theta * 1e-3);
        for _ in 0..200 {
            let (_, n) = tight.sample(&mut rng);
            let d = tight.sample_dir(n, &mut rng);
            assert!((d.length() - 1.0).abs() < 1e-5);
            assert!(d.dot(n) >= 1.0 - versine - 1e-6);
        }
        assert!(tight.in_beam(p, p + Vec3::new(0.0, 1.0, 0.0)));
        // 1e-3 rad off-axis: versine 5e-7 vs the cone's 1.4e-10 — far
        // outside.
        assert!(!tight.in_beam(
            p,
            p + Vec3::new(1e-3, 1.0, 0.0).normalize()
        ));
    }

    #[test]
    fn area_light_samples_lie_on_the_surface() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sphere = Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
            mat: Material::emissive(Vec3::new(5.0, 5.0, 5.0)),
        };
        let light = AreaLight::from_sphere(&sphere);
        assert!((light.area() - 4.0 * PI * 0.25).abs() < 1e-5);
        for _ in 0..100 {
            let (p, _) = light.sample(&mut rng);
            assert!(light.contains(p));
        }

        let quad = Quad::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            4.0,
            sphere.mat,
        );
        let light = AreaLight::from_quad(&quad);
        assert!((light.area() - 8.0).abs() < 1e-5);
        for _ in 0..100 {
            let (p, n) = light.sample(&mut rng);
            assert!(light.contains(p));
            assert_eq!(n, Vec3::new(0.0, -1.0, 0.0));
        }
    }

    // --- BSDF tests ---

    /// The diffuse BSDF at a fixed hero wavelength: f = ρ(λ)/π, the
    /// cosine sampling pdf = cos/π, so the sample weight f·cos/pdf =
    /// ρ(λ) exactly — the spectral albedo of the RGB intent at λ.
    #[test]
    fn diffuse_bsdf_reduces_to_the_cosine_sampling_constants() {
        let m = Material::diffuse(Vec3::new(0.5, 0.6, 0.7));
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.0, -1.0, 0.0);
        let lambda = 555.0; // green peak: ρ = the G channel exactly
        let rho = rgb_to_albedo(m.albedo, lambda);
        assert!((rho - 0.6).abs() < 1e-6, "ρ(555) = {rho}");
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..100 {
            let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, lambda, &mut rng);
            assert!(s.dir_out.dot(n) > 0.0);
            assert!((s.pdf_omega - s.dir_out.dot(n) / PI).abs() < 1e-6);
            assert!((s.weight - rho).abs() < 1e-6);
            let (f, pdf) = m.eval_bsdf(n, dir_in, s.dir_out, lambda);
            assert!((f - rho / PI).abs() < 1e-6);
            assert!((pdf - s.pdf_omega).abs() < 1e-6);
        }
        // Below the horizon: no density.
        let (f, pdf) = m.eval_bsdf(n, dir_in, Vec3::new(0.0, -1.0, 0.0), lambda);
        assert_eq!(f, 0.0);
        assert_eq!(pdf, 0.0);
        // A gray albedo is wavelength-independent: the weight is the
        // gray value at every λ.
        let g = Material::diffuse(Vec3::new(0.42, 0.42, 0.42));
        for lambda in [400.0f32, 480.0, 555.0, 610.0, 700.0] {
            let s = g.sample_bsdf(n, true, dir_in, Transport::Radiance, lambda, &mut rng);
            assert!((s.weight - 0.42).abs() < 1e-6);
        }
    }

    #[test]
    fn mirror_bsdf_is_deterministic_and_delta() {
        let m = Material::mirror(Vec3::new(0.9, 0.9, 0.9));
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, 555.0, &mut rng);
        assert_eq!(s.dir_out, Vec3::new(0.6, 0.8, 0.0));
        assert_eq!(s.pdf_omega, 0.0);
        assert!((s.weight - 0.9).abs() < 1e-6); // gray albedo: ρ(λ) = 0.9
        // eval is (0, 0) for delta BSDFs — connections through them are
        // invalid.
        let (f, pdf) = m.eval_bsdf(n, dir_in, s.dir_out, 555.0);
        assert_eq!(f, 0.0);
        assert_eq!(pdf, 0.0);
        assert!(m.is_delta());
        // The specular weight of the actual pair is ρ(λ); any other
        // direction has none.
        assert_eq!(
            m.specular_weight(n, true, dir_in, s.dir_out, Transport::Radiance, 555.0),
            Some(0.9)
        );
        assert_eq!(
            m.specular_weight(
                n,
                true,
                dir_in,
                Vec3::new(0.0, 1.0, 0.0),
                Transport::Radiance,
                555.0
            ),
            None
        );
    }

    #[test]
    fn glass_fresnel_conserves_energy() {
        // R + T = 1 for every incidence angle, from both sides, including
        // the total-internal-reflection regime (T = 0 there).
        let ior = 1.5;
        for deg in [0.0f32, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0] {
            let cos_i = deg.to_radians().cos();
            let r_out = fresnel_dielectric(cos_i, 1.0, ior);
            let r_in = fresnel_dielectric(cos_i, ior, 1.0);
            assert!((r_out + (1.0 - r_out) - 1.0).abs() < 1e-6);
            assert!((r_in + (1.0 - r_in) - 1.0).abs() < 1e-6);
            // Beyond the critical angle (sin > 1/1.5, i.e. > 41.8°) the
            // internal reflectance must be exactly 1.
            if deg > 41.8 {
                assert!((r_in - 1.0).abs() < 1e-6, "TIR at {deg}°: {r_in}");
            }
        }
        // Normal incidence: the textbook value ((1-1.5)/(1+1.5))^2 = 0.04.
        assert!((fresnel_dielectric(1.0, 1.0, 1.5) - 0.04).abs() < 1e-4);
    }

    /// Achromatic glass keeps its exact constant-n behavior (the existing
    /// tests pin these values); dispersive glass evaluates n(λ) through
    /// Sellmeier, and blue bends more than red — the prism physics
    /// (paper §9.2). "Bends more" = closer to the surface normal: the
    /// refracted direction's *normal* component is larger in magnitude
    /// for blue (the ray travels down into the glass, so the y
    /// components are negative — compare magnitudes, not signs).
    #[test]
    fn dispersive_glass_bends_blue_more_than_red() {
        let m = Material::glass_sellmeier(1.5168, BK7);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0); // ~37° incidence
        let refr_blue = refract_dir(dir_in, n, 1.0, m.bsdf.glass_n(400.0)).unwrap();
        let refr_red = refract_dir(dir_in, n, 1.0, m.bsdf.glass_n(700.0)).unwrap();
        // Snell: sin_t = sin_i / n. Blue has the larger n, so it bends
        // toward the normal more: |cos_t| is larger for blue.
        assert!(
            refr_blue.y.abs() > refr_red.y.abs(),
            "blue must bend more: blue {refr_blue:?} red {refr_red:?}"
        );
        // The achromatic control: the same refraction at every λ.
        let ach = Material::glass(1.5);
        for lambda in [400.0f32, 555.0, 700.0] {
            assert!((ach.bsdf.glass_n(lambda) - 1.5).abs() < 1e-6);
        }
        // The dispersive n values match the Sellmeier evaluation, and
        // the datasheet's D-line value lands at 589 nm (n(555) is
        // *higher* — normal dispersion).
        assert!((m.bsdf.glass_n(400.0) - sellmeier_n(&BK7, 400.0)).abs() < 1e-6);
        assert!((m.bsdf.glass_n(589.0) - 1.5168).abs() < 2e-4);
        assert!(m.bsdf.glass_n(555.0) > 1.5168, "normal dispersion: n(555) > n(589)");
    }

    #[test]
    fn glass_sampling_matches_the_fresnel_split() {
        // Over many samples the reflection/refraction split must follow
        // the Fresnel reflectance, and the refracted direction must obey
        // Snell's law.
        let m = Material::glass(1.5);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let f = fresnel_dielectric(-n.dot(dir_in), 1.0, 1.5);
        let mut reflected = 0;
        const N: usize = 20000;
        for _ in 0..N {
            let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, 555.0, &mut rng);
            assert_eq!(s.pdf_omega, 0.0);
            if s.dir_out.y > 0.0 {
                reflected += 1;
            } else {
                // Snell: sin_t = sin_i / 1.5.
                let sin_i = (1.0f32 - 0.8 * 0.8).sqrt();
                let sin_t = (1.0 - s.dir_out.y * s.dir_out.y).sqrt();
                assert!((sin_t - sin_i / 1.5).abs() < 1e-4);
                // Radiance transport picks up (eta_t/eta_i)^2 = 1.5^2.
                assert!((s.weight - 2.25).abs() < 1e-6);
            }
        }
        let frac = reflected as f32 / N as f32;
        assert!((frac - f).abs() < 0.02, "split {frac} vs Fresnel {f}");
    }

    #[test]
    fn glass_tir_falls_back_to_reflection() {
        // Inside the glass (front_face = false), beyond the critical angle
        // every sample must reflect — no refraction exists.
        let m = Material::glass(1.5);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.9, -0.435889894354067, 0.0); // sin_i = 0.9 > 1/1.5
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        for _ in 0..100 {
            let s = m.sample_bsdf(n, false, dir_in, Transport::Radiance, 555.0, &mut rng);
            assert!(s.dir_out.y > 0.0, "TIR must reflect: {:?}", s.dir_out);
            assert!((s.weight - 1.0).abs() < 1e-6);
        }
        // And the specular weight of the refracted pair is None there.
        assert_eq!(
            m.specular_weight(
                n,
                false,
                dir_in,
                Vec3::new(0.0, -1.0, 0.0),
                Transport::Radiance,
                555.0
            ),
            None
        );
    }

    #[test]
    fn glass_specular_weight_covers_both_modes() {
        // The refracted pair's weight differs by transport mode: radiance
        // gets (eta_t/eta_i)^2, importance does not.
        let m = Material::glass(1.5);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let refr = refract_dir(dir_in, n, 1.0, 1.5).unwrap();
        assert_eq!(
            m.specular_weight(n, true, dir_in, refr, Transport::Radiance, 555.0),
            Some(2.25)
        );
        assert_eq!(
            m.specular_weight(n, true, dir_in, refr, Transport::Importance, 555.0),
            Some(1.0)
        );
        // Exiting the glass: eta_i = 1.5, eta_t = 1, so radiance gets
        // (1/1.5)^2 and importance 1. Compare within a few ulps: the
        // implementation computes (1/1.5)*(1/1.5), which rounds
        // differently from the literal 1/2.25.
        let refr_out = refract_dir(dir_in, n, 1.5, 1.0).unwrap();
        let w = m.specular_weight(n, false, dir_in, refr_out, Transport::Radiance, 555.0);
        let expected = 1.0 / 2.25;
        assert!(w.is_some());
        assert!(
            (w.unwrap() - expected).abs() < 1e-6,
            "got {w:?}, expected {expected}"
        );
    }

    // --- GGX glossy BSDF (paper §9.5) ---

    /// The sampler and the eval must agree on the pdf — the MIS machinery
    /// requires one distribution — and the samples must stay in the
    /// hemisphere (a below-horizon draw is possible at high roughness
    /// but carries no light; the weight is 0 there, never infinite).
    #[test]
    fn glossy_sampler_and_eval_agree() {
        let m = Material::glossy(Vec3::new(1.0, 1.0, 1.0), 0.08, 0.9);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(6);
        let mut below_horizon = 0usize;
        const N: usize = 5000;
        for _ in 0..N {
            let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, 555.0, &mut rng);
            let (f, pdf) = m.eval_bsdf(n, dir_in, s.dir_out, 555.0);
            assert!((pdf - s.pdf_omega).abs() < 1e-5, "pdf {pdf} vs {}", s.pdf_omega);
            if s.dir_out.dot(n) <= 0.0 {
                // Below the horizon: f = 0, the weight must be 0 (the
                // draw is wasted), never ±inf.
                below_horizon += 1;
                assert_eq!(s.weight, 0.0);
                assert_eq!(f, 0.0);
            } else {
                assert!(s.weight.is_finite() && s.weight > 0.0);
                assert!((s.weight - f * s.dir_out.dot(n) / pdf).abs() < 1e-4);
            }
        }
        // At this roughness essentially every draw stays in the
        // hemisphere; the guard exists for the grazing tail.
        assert!(below_horizon < N / 10);
    }

    /// The near-mirror limit (§9.5: "the bridge is the small-roughness
    /// limit of the glossy connection"): as α → 0 the lobe collapses
    /// onto the perfect-mirror direction. The GGX NDF has a heavy
    /// angular tail (a u-draw near 0 produces a large θ_h even at tiny
    /// α), so the check is on the lobe's *bulk* — the 99th-percentile
    /// angle — not the maximum.
    #[test]
    fn glossy_approaches_the_mirror_direction() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let mirror_dir = reflect_dir(dir_in, n);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        const N: usize = 4000;
        for alpha in [0.001f32, 0.005, 0.02] {
            let m = Material::glossy(Vec3::new(1.0, 1.0, 1.0), alpha, 0.9);
            let mut angles = [0.0f32; N];
            for i in 0..N {
                let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, 555.0, &mut rng);
                let dot = s.dir_out.dot(mirror_dir).clamp(-1.0, 1.0);
                angles[i] = dot.acos();
            }
            angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // The 99th percentile: the lobe's bulk scales with α; 30α
            // is a generous envelope over the GGX tail's 1% fringe.
            let p99 = angles[(N as f32 * 0.99) as usize];
            assert!(
                p99 < alpha * 30.0,
                "α = {alpha}: p99 spread {p99} rad"
            );
        }
    }

    /// Reciprocity: f(ω_i, ω_o) = f(ω_o, ω_i) — the microfacet formula is
    /// symmetric in the pair (the half-vector bisects, so the Fresnel
    /// cosine agrees from both sides).
    #[test]
    fn glossy_is_reciprocal() {
        let m = Material::glossy(Vec3::new(1.0, 1.0, 1.0), 0.25, 0.9);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(8);
        for _ in 0..500 {
            let a = cosine_hemisphere(n, &mut rng);
            let b = cosine_hemisphere(n, &mut rng);
            let (f_ab, _) = m.eval_bsdf(n, a, b, 555.0);
            let (f_ba, _) = m.eval_bsdf(n, b, a, 555.0);
            assert!((f_ab - f_ba).abs() < 1e-4, "{f_ab} vs {f_ba}");
        }
    }

    /// The energy guard: the directional-hemispherical reflectance
    /// ∫ f·cos_o dω_o must stay ≤ 1 (the single-scatter GGX masking term
    /// *loses* energy at high roughness — multiple scattering between
    /// microfacets is not modeled — so the demo uses low α and the
    /// assert has a sane floor).
    #[test]
    fn glossy_directional_hemispherical_reflectance_is_bounded() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);
        for &(alpha, f0) in &[(0.05f32, 0.9), (0.1, 0.9), (0.25, 0.5), (0.5, 0.04)] {
            let m = Material::glossy(Vec3::new(1.0, 1.0, 1.0), alpha, f0);
            // Average the DHR over incident directions.
            let mut dhr_mean = 0.0f32;
            const NI: usize = 64;
            const NO: usize = 256;
            for _ in 0..NI {
                let dir_in = cosine_hemisphere(n, &mut rng) * -1.0;
                let mut sum = 0.0f32;
                for _ in 0..NO {
                    let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, 555.0, &mut rng);
                    if s.pdf_omega > 0.0 && s.dir_out.dot(n) > 0.0 {
                        sum += s.weight; // f·cos/pdf is the DHR estimator
                    }
                }
                dhr_mean += sum / NO as f32;
            }
            dhr_mean /= NI as f32;
            assert!(
                dhr_mean <= 1.0 + 0.02,
                "α={alpha} f0={f0}: DHR {dhr_mean} exceeds 1"
            );
            assert!(
                dhr_mean >= 0.25 * f0,
                "α={alpha} f0={f0}: DHR {dhr_mean} implausibly small"
            );
        }
    }

    /// Glossy is NOT delta (§9.5): connections through glossy vertices
    /// are valid — `eval_bsdf` returns a nonzero f and pdf for ordinary
    /// direction pairs, and `specular_weight` is None (no specular pair;
    /// the MIS machinery uses the continuous pdf instead).
    #[test]
    fn glossy_is_a_continuous_bsdf() {
        let m = Material::glossy(Vec3::new(1.0, 1.0, 1.0), 0.1, 0.9);
        assert!(!m.bsdf.is_delta());
        assert!(!m.is_delta());
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        // An ordinary (non-mirror) direction pair evaluates nonzero.
        let dir_out = Vec3::new(0.0, 1.0, 0.0).normalize();
        let (f, pdf) = m.eval_bsdf(n, dir_in, dir_out, 555.0);
        assert!(f > 0.0, "glossy eval must be nonzero: {f}");
        assert!(pdf > 0.0);
        // No specular pair: hypothetical extensions use the continuous
        // pdf, exactly like diffuse.
        assert_eq!(
            m.specular_weight(n, true, dir_in, dir_out, Transport::Radiance, 555.0),
            None
        );
        // Even the exact mirror direction has no specular pair — the
        // lobe is continuous, not a delta.
        let mirror_dir = reflect_dir(dir_in, n);
        assert_eq!(
            m.specular_weight(n, true, dir_in, mirror_dir, Transport::Radiance, 555.0),
            None
        );
        let (f, pdf) = m.eval_bsdf(n, dir_in, mirror_dir, 555.0);
        assert!(f > 0.0 && pdf > 0.0);
    }
}
