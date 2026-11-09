//! Spectral transport: the CIE 1931 color-matching functions, spectral
//! emitters, and the hero-wavelength machinery (paper §9.2).
//!
//! Radiance is transported as a spectral density at one *hero
//! wavelength* per path rather than as an RGB triplet. The wavelength is
//! sampled once per path from a mixture over the emitters — a laser
//! line's atom (λ is deterministic on its branch; §9.2: "for lasers it
//! adds none") or the uniform distribution over the visible range for
//! smooth emitters — and the sensor converts the spectral estimate to
//! linear sRGB through the CIE CMFs. The λ sampling is independent of
//! all direction sampling, so every BDPT strategy pdf factorizes as
//! `p(directions)·p(λ)` and the λ factors cancel in every MIS ratio:
//! **the MIS machinery is untouched** (§9.2).
//!
//! # The branch rule
//!
//! The mixture is *branch-aware*, which is the subtle part. A laser
//! line's spectrum is a delta measure, not a density, so the estimator
//! must credit, on a **line branch**, only the lines at that λ (their
//! full power — the atom's mass), and, on a **smooth branch**, only the
//! smooth SPDs (their density). Crediting a smooth emitter's density on
//! a line branch would add a point sample of a continuous integral that
//! the branch's probability does not weight correctly — a bias. The
//! rule extends to *every* continuous spectrum in the scene, the
//! background included: `SpectrumKind::eval` takes the branch and
//! returns 0 for the inactive side.
//!
//! # Why the smooth branch is uniform
//!
//! Sampling λ ∝ ȳ (the photopic response) would seem the natural
//! importance choice, but it blows up the X/Z channels: the ratio
//! x̄/ȳ reaches ~12 at the blue end (x̄(445)/ȳ(445) = 0.348/0.0298), so
//! the occasional blue-end draw carries a sample ~10³× the mean — a
//! heavy tail no practical sample count tames. The uniform distribution
//! over [380, 780] nm has no tail at all (the integrand is bounded) and
//! costs ~2–4× luminance variance — exactly the range the paper
//! predicts for smooth emitters (§9.2).
//!
//! # Smooth spectra reconstruct their intent exactly
//!
//! A `Smooth` spectrum is the unique CMF-basis combination
//! `S(λ) = CMF(λ)ᵀ·G⁻¹·xyz` whose tristimulus integral is exactly the
//! intended XYZ (`G = ∫CMF·CMFᵀ dλ`, the CMF Gram matrix). The emitter
//! therefore renders to exactly its intended RGB *in expectation*,
//! which keeps every existing exact-color test meaningful as a mean
//! over N samples.

use crate::math::{Vec3, VEC_ZERO};
use rand::Rng;
use std::sync::LazyLock;

/// The visible range the CMF table covers (nm).
pub const LAMBDA_MIN: f32 = 380.0;
pub const LAMBDA_MAX: f32 = 780.0;
/// The table step (nm).
const CMF_STEP: f32 = 5.0;

/// The CIE 1931 2° color matching functions (x̄, ȳ, z̄), 380–780 nm at
/// 5 nm — 81 knots (source: CIE 018:2019, via the cvrl.ucl.ac.uk
/// distribution). `CMF[i] = (x̄, ȳ, z̄)` at `LAMBDA_MIN + i·5`.
const CMF: [[f32; 3]; 81] = [
    [0.001368, 0.000039, 0.006450],   // 380
    [0.002236, 0.000064, 0.010550],   // 385
    [0.004243, 0.000120, 0.020050],   // 390
    [0.007650, 0.000217, 0.036210],   // 395
    [0.014310, 0.000396, 0.067850],   // 400
    [0.023190, 0.000640, 0.110200],   // 405
    [0.043510, 0.001210, 0.207400],   // 410
    [0.077630, 0.002180, 0.371300],   // 415
    [0.134380, 0.004000, 0.645600],   // 420
    [0.214770, 0.007300, 1.039050],   // 425
    [0.283900, 0.011600, 1.385600],   // 430
    [0.328500, 0.016840, 1.622960],   // 435
    [0.348280, 0.023000, 1.747060],   // 440
    [0.348060, 0.029800, 1.782600],   // 445
    [0.336200, 0.038000, 1.772110],   // 450
    [0.318700, 0.048000, 1.744100],   // 455
    [0.290800, 0.060000, 1.669200],   // 460
    [0.251100, 0.073900, 1.528100],   // 465
    [0.195360, 0.090980, 1.287640],   // 470
    [0.142100, 0.112600, 1.041900],   // 475
    [0.095640, 0.139020, 0.812950],   // 480
    [0.057950, 0.169300, 0.616200],   // 485
    [0.032010, 0.208020, 0.465180],   // 490
    [0.014700, 0.258600, 0.353300],   // 495
    [0.004900, 0.323000, 0.272000],   // 500
    [0.002400, 0.407300, 0.212300],   // 505
    [0.009300, 0.503000, 0.158200],   // 510
    [0.029100, 0.608200, 0.111700],   // 515
    [0.063270, 0.710000, 0.078250],   // 520
    [0.109600, 0.793200, 0.057250],   // 525
    [0.165500, 0.862000, 0.042160],   // 530
    [0.225750, 0.914850, 0.029840],   // 535
    [0.290400, 0.954000, 0.020300],   // 540
    [0.359700, 0.980300, 0.013400],   // 545
    [0.433450, 0.994950, 0.008750],   // 550
    [0.512050, 1.000000, 0.005750],   // 555
    [0.594500, 0.995000, 0.003900],   // 560
    [0.678400, 0.978600, 0.002750],   // 565
    [0.762100, 0.952000, 0.002100],   // 570
    [0.842500, 0.915400, 0.001800],   // 575
    [0.916300, 0.870000, 0.001650],   // 580
    [0.978600, 0.816300, 0.001400],   // 585
    [1.026300, 0.757000, 0.001100],   // 590
    [1.056700, 0.694900, 0.001000],   // 595
    [1.062200, 0.631000, 0.000800],   // 600
    [1.045600, 0.566800, 0.000600],   // 605
    [1.002600, 0.503000, 0.000340],   // 610
    [0.938400, 0.441200, 0.000240],   // 615
    [0.854450, 0.381000, 0.000190],   // 620
    [0.751400, 0.321000, 0.000100],   // 625
    [0.642400, 0.265000, 0.000050],   // 630
    [0.541900, 0.217000, 0.000030],   // 635
    [0.447900, 0.175000, 0.000020],   // 640
    [0.360800, 0.138200, 0.000010],   // 645
    [0.283500, 0.107000, 0.000000],   // 650
    [0.218700, 0.081600, 0.000000],   // 655
    [0.164900, 0.061000, 0.000000],   // 660
    [0.121200, 0.044580, 0.000000],   // 665
    [0.087400, 0.032000, 0.000000],   // 670
    [0.063600, 0.023200, 0.000000],   // 675
    [0.046770, 0.017000, 0.000000],   // 680
    [0.032900, 0.011920, 0.000000],   // 685
    [0.022700, 0.008210, 0.000000],   // 690
    [0.015840, 0.005723, 0.000000],   // 695
    [0.011359, 0.004102, 0.000000],   // 700
    [0.008111, 0.002929, 0.000000],   // 705
    [0.005790, 0.002091, 0.000000],   // 710
    [0.004109, 0.001484, 0.000000],   // 715
    [0.002899, 0.001047, 0.000000],   // 720
    [0.002049, 0.000740, 0.000000],   // 725
    [0.001440, 0.000520, 0.000000],   // 730
    [0.001000, 0.000361, 0.000000],   // 735
    [0.000690, 0.000249, 0.000000],   // 740
    [0.000476, 0.000172, 0.000000],   // 745
    [0.000332, 0.000120, 0.000000],   // 750
    [0.000235, 0.000085, 0.000000],   // 755
    [0.000166, 0.000060, 0.000000],   // 760
    [0.000117, 0.000042, 0.000000],   // 765
    [0.000083, 0.000030, 0.000000],   // 770
    [0.000059, 0.000021, 0.000000],   // 775
    [0.000042, 0.000015, 0.000000],   // 780
];

/// The CMF at `lambda` (nm), linearly interpolated between knots and 0
/// outside the table range.
pub fn cmf(lambda: f32) -> Vec3 {
    if !(LAMBDA_MIN..=LAMBDA_MAX).contains(&lambda) {
        return VEC_ZERO;
    }
    let t = (lambda - LAMBDA_MIN) / CMF_STEP;
    let i = (t as usize).min(CMF.len() - 2);
    let f = t - i as f32;
    let a = CMF[i];
    let b = CMF[i + 1];
    Vec3::new(
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    )
}

/// The CMF Gram matrix `G = ∫CMF·CMFᵀ dλ` (trapezoidal over the table),
/// in f64 for the accumulation.
fn gram_matrix() -> [[f64; 3]; 3] {
    let mut g = [[0.0f64; 3]; 3];
    let step = CMF_STEP as f64;
    for i in 0..CMF.len() - 1 {
        for a in 0..3 {
            for b in 0..3 {
                let fa = CMF[i][a] as f64 * CMF[i][b] as f64;
                let fb = CMF[i + 1][a] as f64 * CMF[i + 1][b] as f64;
                g[a][b] += 0.5 * (fa + fb) * step;
            }
        }
    }
    g
}

/// The inverse of a 3×3 matrix (the adjugate over the determinant).
fn invert3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let s = 1.0 / det;
    [
        [
            s * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
            s * (m[0][2] * m[2][1] - m[0][1] * m[2][2]),
            s * (m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        ],
        [
            s * (m[1][2] * m[2][0] - m[1][0] * m[2][2]),
            s * (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
            s * (m[0][2] * m[1][0] - m[0][0] * m[1][2]),
        ],
        [
            s * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
            s * (m[0][1] * m[2][0] - m[0][0] * m[2][1]),
            s * (m[0][0] * m[1][1] - m[0][1] * m[1][0]),
        ],
    ]
}

/// G⁻¹ as three row vectors: `coeffs = G⁻¹·xyz` is
/// `coeffs[c] = G_INV[c]·xyz`.
static G_INV: LazyLock<[Vec3; 3]> = LazyLock::new(|| {
    let m = invert3(gram_matrix());
    [
        Vec3::new(m[0][0] as f32, m[0][1] as f32, m[0][2] as f32),
        Vec3::new(m[1][0] as f32, m[1][1] as f32, m[1][2] as f32),
        Vec3::new(m[2][0] as f32, m[2][1] as f32, m[2][2] as f32),
    ]
});

/// Linear sRGB → XYZ (IEC 61966-2-1). The Y row is the Rec. 709
/// luminance vector — the same weights as `Vec3::luminance`, so the
/// luminance of a linear-sRGB color equals its Y tristimulus.
pub fn srgb_to_xyz(rgb: Vec3) -> Vec3 {
    Vec3::new(
        0.4124 * rgb.x + 0.3576 * rgb.y + 0.1805 * rgb.z,
        0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z,
        0.0193 * rgb.x + 0.1192 * rgb.y + 0.9505 * rgb.z,
    )
}

/// XYZ → linear sRGB (IEC 61966-2-1). Out-of-gamut XYZ (a pure 532 nm
/// green, for instance) produces negative channels — correct, and
/// gamut-mapped at display (§9.2).
pub fn xyz_to_srgb(xyz: Vec3) -> Vec3 {
    Vec3::new(
        3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
        0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z,
    )
}

/// The emission spectrum of an emitter (paper §9.2).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SpectrumKind {
    /// No emission (a scattering surface).
    None,
    /// A smooth SPD `S(λ) = coeffs·CMF(λ)` — the unique CMF-basis
    /// spectrum whose tristimulus integral is exactly the intended
    /// color, so the emitter renders to exactly that RGB in
    /// expectation. Constructed by `smooth_from_rgb`.
    Smooth { coeffs: Vec3 },
    /// A monochromatic line: all `power` at `lambda` (nm). A laser is
    /// monochromatic, so λ is deterministic on its branch — the
    /// physically correct replacement for the old RGB trick of putting
    /// 4×10⁸ in one channel (§9.2).
    Line { lambda: f32, power: f32 },
}

impl SpectrumKind {
    /// The smooth spectrum that renders as `rgb` in expectation.
    pub fn smooth_from_rgb(rgb: Vec3) -> Self {
        let xyz = srgb_to_xyz(rgb);
        let g = *G_INV;
        Self::Smooth {
            coeffs: Vec3::new(g[0].dot(xyz), g[1].dot(xyz), g[2].dot(xyz)),
        }
    }

    /// True when this spectrum radiates anything.
    pub fn is_emissive(&self) -> bool {
        match self {
            SpectrumKind::None => false,
            SpectrumKind::Smooth { coeffs } => *coeffs != VEC_ZERO,
            SpectrumKind::Line { power, .. } => *power > 0.0,
        }
    }

    /// The CMF-basis coefficients of a smooth spectrum (None otherwise).
    /// Used by the WGSL port's background constant (pinned by a test so
    /// the CPU and the shader cannot drift apart).
    pub fn coeffs(&self) -> Option<Vec3> {
        match self {
            SpectrumKind::Smooth { coeffs } => Some(*coeffs),
            _ => None,
        }
    }

    /// The spectral radiance this emitter carries at `lambda` under the
    /// drawn branch (see `WaveSample`): a line radiates its full power
    /// on its own line branch (the atom's mass — the delta integrates
    /// out against it), a smooth SPD its density on the smooth branch,
    /// and everything else is dark. This branch rule is what keeps the
    /// hero-wavelength estimator unbiased in the presence of delta
    /// spectra.
    pub fn eval(&self, lambda: f32, line_branch: bool) -> f32 {
        match self {
            SpectrumKind::None => 0.0,
            SpectrumKind::Smooth { coeffs } => {
                if line_branch {
                    0.0
                } else {
                    cmf(lambda).dot(*coeffs)
                }
            }
            SpectrumKind::Line { lambda: l, power } => {
                if line_branch && *l == lambda {
                    *power
                } else {
                    0.0
                }
            }
        }
    }
}

/// The per-path wavelength sample: the hero wavelength, the sensor
/// weight 1/p(λ), and which branch of the emitter mixture was drawn.
#[derive(Clone, Copy, Debug)]
pub struct WaveSample {
    /// The hero wavelength (nm).
    pub lambda: f32,
    /// 1/p(λ): the sensor conversion divides by the sampling density.
    pub inv_pdf: f32,
    /// True on a line branch (only lines at `lambda` radiate); false on
    /// a smooth branch (only smooth SPDs radiate).
    pub line_branch: bool,
}

impl WaveSample {
    /// True when `spectrum` radiates at this sample — the branch rule
    /// as a set membership test, used to pick the light subpath's
    /// starting emitter among the *active* ones (the pick pdf and the
    /// MIS machinery's `inv_light_pdf = n_active·A` stay consistent
    /// because every strategy in one trace uses the same n_active).
    ///
    /// The test is *structural*, not sign-based: a smooth CMF-basis
    /// spectrum can evaluate negative at some wavelengths (it is the
    /// unique combination matching the XYZ integral, not a physically
    /// nonnegative SPD), and those negative values are legitimate
    /// parts of the response integral — dropping them would bias the
    /// estimator. A smooth spectrum is active on the whole smooth
    /// branch; a line is active only on its own line branch.
    pub fn active(&self, spectrum: &SpectrumKind) -> bool {
        match spectrum {
            SpectrumKind::None => false,
            SpectrumKind::Line { lambda, .. } => self.line_branch && *lambda == self.lambda,
            SpectrumKind::Smooth { .. } => !self.line_branch,
        }
    }
}

/// Sample the hero wavelength from the mixture over the emitters
/// (paper §9.2): pick an emitter uniformly, then take its branch — a
/// line's λ is its own line (deterministic), a smooth emitter's λ is
/// uniform over the visible range. The returned `inv_pdf` is 1/p(λ)
/// under the *full* mixture:
///
/// * line branch at λ₀: the atom's mass is k₀/n (k₀ lines at λ₀, n
///   emitters), so `inv_pdf = n/k₀`;
/// * smooth branch: the continuous density is `(n_smooth/n)/400`, so
///   `inv_pdf = 400·n/n_smooth`.
///
/// Both are exactly what makes
/// `E[CMF(λ)·Le(λ)·inv_pdf]` equal the emitter's true tristimulus
/// response — pinned by the estimator tests below.
pub fn sample_wavelength<T>(
    items: &[T],
    spectrum: impl Fn(&T) -> SpectrumKind,
    rng: &mut impl Rng,
) -> WaveSample {
    let n = items.len();
    if n == 0 {
        // No emitters: nothing radiates, λ is irrelevant.
        return WaveSample {
            lambda: 550.0,
            inv_pdf: 1.0,
            line_branch: false,
        };
    }
    let pick = rng.gen_range(0..n);
    match spectrum(&items[pick]) {
        SpectrumKind::Line { lambda, .. } => {
            // The atom's mass: every line at this exact λ₀.
            let k = items
                .iter()
                .filter(|it| {
                    matches!(spectrum(it), SpectrumKind::Line { lambda: l, .. } if l == lambda)
                })
                .count()
                .max(1);
            WaveSample {
                lambda,
                inv_pdf: n as f32 / k as f32,
                line_branch: true,
            }
        }
        _ => {
            // The smooth branch: λ uniform over the visible range (see
            // the module doc for why not ȳ-proportional).
            let n_smooth = items
                .iter()
                .filter(|it| !matches!(spectrum(it), SpectrumKind::Line { .. }))
                .count()
                .max(1);
            let lambda = LAMBDA_MIN + rng.gen::<f32>() * (LAMBDA_MAX - LAMBDA_MIN);
            WaveSample {
                lambda,
                inv_pdf: (LAMBDA_MAX - LAMBDA_MIN) * n as f32 / n_smooth as f32,
                line_branch: false,
            }
        }
    }
}

/// The sensor conversion: the rendered linear-sRGB sample for a spectral
/// radiance estimate `radiance` at `lambda`, weighted by 1/p(λ). The
/// response integral `∫CMF(λ)·L(λ) dλ` is estimated by
/// `CMF(λ)·L(λ)/p(λ)`; the XYZ→sRGB matrix finishes the conversion.
/// Out-of-gamut samples carry negative channels by design (§9.2) — the
/// display stage gamut-maps them.
pub fn sensor_rgb(lambda: f32, radiance: f32, inv_pdf: f32) -> Vec3 {
    let c = cmf(lambda) * (radiance * inv_pdf);
    xyz_to_srgb(c)
}

/// The smooth spectrum's CMF-basis coefficients for the *background*
/// intent — the values the WGSL port inlines as constants (the CPU
/// computes them once through the LazyLock; the shader cannot). Pinned
/// by a test so the two cannot drift apart.
pub fn background_coeffs() -> Vec3 {
    SpectrumKind::smooth_from_rgb(Vec3::new(0.02, 0.02, 0.05))
        .coeffs()
        .expect("the background is a smooth spectrum")
}

/// The spectral radiance of the *background* — a smooth (incoherent)
/// spectrum reconstructed from the RGB intent, evaluated at `lambda`.
/// The background radiates only on the smooth branch, like every other
/// continuous spectrum in the scene (see the module doc's branch rule):
/// a line branch sees a dark background.
pub fn smooth_spectrum_value(rgb: Vec3, lambda: f32) -> f32 {
    SpectrumKind::smooth_from_rgb(rgb).eval(lambda, false)
}

/// The smoothstep ramp on [0, 1] (0 outside).
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Reconstruct a spectral reflectance from an RGB albedo (paper §9.2):
/// three complementary smoothstep ramps — B falling 440→500 nm, G rising
/// 440→500 and falling 570→630, R rising 570→630 — that sum to **1 at
/// every wavelength**, so
///
/// * grays are **exact**: `(v, v, v) → ρ(λ) ≡ v` (the white furnace
///   keeps its exact contract),
/// * ρ is a convex combination of the channels: energy conservation is
///   automatic and ρ ∈ [min rgb, max rgb],
/// * hues are approximate (a Smits-style fit is a drop-in refinement).
pub fn rgb_to_albedo(rgb: Vec3, lambda: f32) -> f32 {
    let b = if lambda <= 440.0 {
        1.0
    } else if lambda >= 500.0 {
        0.0
    } else {
        1.0 - smoothstep((lambda - 440.0) / 60.0)
    };
    let g = if lambda <= 440.0 {
        0.0
    } else if lambda < 500.0 {
        smoothstep((lambda - 440.0) / 60.0)
    } else if lambda <= 570.0 {
        1.0
    } else if lambda < 630.0 {
        1.0 - smoothstep((lambda - 570.0) / 60.0)
    } else {
        0.0
    };
    let r = if lambda <= 570.0 {
        0.0
    } else if lambda >= 630.0 {
        1.0
    } else {
        smoothstep((lambda - 570.0) / 60.0)
    };
    rgb.x * r + rgb.y * g + rgb.z * b
}

/// Sellmeier's dispersion formula: `n²(λ) = 1 + Σ Bᵢλ²/(λ² − Cᵢ)` with
/// λ in µm and C in µm². `coeffs = [B₁, B₂, B₃, C₁, C₂, C₃]`.
pub fn sellmeier_n(coeffs: &[f32; 6], lambda_nm: f32) -> f32 {
    let l = lambda_nm / 1000.0;
    let l2 = l * l;
    let mut n2 = 1.0f32;
    for i in 0..3 {
        n2 += coeffs[i] * l2 / (l2 - coeffs[3 + i]);
    }
    n2.sqrt()
}

/// N-BK7 borosilicate glass (Schott's Sellmeier coefficients) — the
/// demo's dispersive glass. n(589 nm) ≈ 1.5168, n(400) − n(700) ≈ 0.019.
pub const BK7: [f32; 6] = [
    1.03961212,
    0.231792344,
    1.01046945,
    0.00600069867,
    0.0200179144,
    103.560653,
];

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn cmf_table_anchors() {
        // The photopic peak and a few canonical knots.
        assert!((cmf(555.0).y - 1.0).abs() < 1e-6);
        assert!((cmf(555.0).x - 0.512050).abs() < 1e-6);
        assert!((cmf(540.0).y - 0.954).abs() < 1e-6);
        assert!((cmf(650.0).x - 0.2835).abs() < 1e-6);
        assert!((cmf(650.0).y - 0.107).abs() < 1e-6);
        assert!((cmf(440.0).z - 1.747060).abs() < 1e-6);
        // Linear interpolation between knots.
        assert!((cmf(532.0).y - 0.883140).abs() < 1e-5);
        // Zero outside the visible range.
        assert_eq!(cmf(379.0), VEC_ZERO);
        assert_eq!(cmf(781.0), VEC_ZERO);
    }

    #[test]
    fn xyz_srgb_round_trip() {
        // White: the canonical XYZ of linear-sRGB white.
        let xyz = srgb_to_xyz(Vec3::new(1.0, 1.0, 1.0));
        assert!((xyz.x - 0.9505).abs() < 1e-3);
        assert!((xyz.y - 1.0).abs() < 1e-3);
        assert!((xyz.z - 1.0890).abs() < 1e-3);
        // The round trip is the identity (the matrices are inverses).
        for rgb in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.2, 0.5, 0.9),
            Vec3::new(1000.0, 1000.0, 1000.0),
        ] {
            let back = xyz_to_srgb(srgb_to_xyz(rgb));
            assert!((back.x - rgb.x).abs() < 1e-3 * rgb.x.max(1.0));
            assert!((back.y - rgb.y).abs() < 1e-3 * rgb.y.max(1.0));
            assert!((back.z - rgb.z).abs() < 1e-3 * rgb.z.max(1.0));
        }
        // The Y row of sRGB→XYZ is the Rec. 709 luminance vector, so the
        // luminance of a linear-sRGB color equals its Y — the firefly
        // statistic stays Y-based.
        let c = Vec3::new(0.3, 0.7, 0.1);
        assert!((srgb_to_xyz(c).y - c.luminance()).abs() < 1e-5);
    }

    /// The killer property of the CMF-comb construction: the smooth
    /// spectrum's tristimulus integral is *exactly* the intent. The
    /// test integrates with the same trapezoidal rule the Gram matrix
    /// used, so the equality is exact up to f32 rounding.
    #[test]
    fn smooth_spectrum_reproduces_its_intent() {
        for rgb in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(50.0, 50.0, 50.0),
            Vec3::new(0.75, 0.25, 0.25),
            Vec3::new(0.25, 0.75, 0.25),
            Vec3::new(0.1, 0.2, 0.9),
            Vec3::new(1000.0, 1000.0, 1000.0),
        ] {
            let s = SpectrumKind::smooth_from_rgb(rgb);
            let SpectrumKind::Smooth { coeffs } = s else {
                panic!("not a smooth spectrum");
            };
            let xyz = srgb_to_xyz(rgb);
            let mut integral = VEC_ZERO;
            for i in 0..CMF.len() - 1 {
                let a = Vec3::new(CMF[i][0], CMF[i][1], CMF[i][2]);
                let b = Vec3::new(CMF[i + 1][0], CMF[i + 1][1], CMF[i + 1][2]);
                integral = integral + (a * a.dot(coeffs) + b * b.dot(coeffs)) * (CMF_STEP * 0.5);
            }
            let scale = xyz.x.max(xyz.y).max(xyz.z).max(1.0);
            assert!(
                (integral.x - xyz.x).abs() < 1e-3 * scale,
                "X: {integral:?} vs {xyz:?} (rgb {rgb:?})"
            );
            assert!(
                (integral.y - xyz.y).abs() < 1e-3 * scale,
                "Y: {integral:?} vs {xyz:?} (rgb {rgb:?})"
            );
            assert!(
                (integral.z - xyz.z).abs() < 1e-3 * scale,
                "Z: {integral:?} vs {xyz:?} (rgb {rgb:?})"
            );
        }
    }

    #[test]
    fn line_spectra_gate_on_the_branch() {
        let line = SpectrumKind::Line {
            lambda: 650.0,
            power: 8.0,
        };
        assert_eq!(line.eval(650.0, true), 8.0);
        // The smooth branch: lines are dark (their density is 0 off
        // the atom).
        assert_eq!(line.eval(650.0, false), 0.0);
        // Another line's branch: dark.
        assert_eq!(line.eval(532.0, true), 0.0);
        let smooth = SpectrumKind::smooth_from_rgb(Vec3::new(1.0, 1.0, 1.0));
        assert!(smooth.eval(555.0, false) > 0.0);
        // The line branch: smooth SPDs are dark (the branch rule).
        assert_eq!(smooth.eval(555.0, true), 0.0);
        assert_eq!(SpectrumKind::None.eval(555.0, false), 0.0);
        assert!(!SpectrumKind::None.is_emissive());
        assert!(line.is_emissive());
        assert!(smooth.is_emissive());
        assert!(!SpectrumKind::smooth_from_rgb(VEC_ZERO).is_emissive());
    }

    #[test]
    fn wavelength_mixture_covers_every_emitter() {
        let spectra = [
            SpectrumKind::smooth_from_rgb(Vec3::new(1.0, 1.0, 1.0)),
            SpectrumKind::Line {
                lambda: 650.0,
                power: 8.0,
            },
            SpectrumKind::Line {
                lambda: 532.0,
                power: 3.0,
            },
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let mut red = 0usize;
        let mut green = 0usize;
        let mut smooth = 0usize;
        const N: usize = 30000;
        for _ in 0..N {
            let w = sample_wavelength(&spectra, |s| *s, &mut rng);
            assert!(w.inv_pdf > 0.0);
            if w.line_branch {
                // A lone line at its λ: the atom's mass is 1/3, so
                // 1/p = 3.
                assert!((w.inv_pdf - 3.0).abs() < 1e-6);
                assert!(w.lambda == 650.0 || w.lambda == 532.0);
                if w.lambda == 650.0 {
                    red += 1;
                } else {
                    green += 1;
                }
            } else {
                // One smooth emitter of three: the continuous density
                // is (1/3)/400, so 1/p = 1200.
                assert!((w.inv_pdf - 1200.0).abs() < 1e-3);
                assert!(w.lambda >= LAMBDA_MIN && w.lambda < LAMBDA_MAX);
                smooth += 1;
            }
        }
        // Every emitter's branch carries ~1/3 of the draws — the
        // support condition: without the line components the lasers
        // would never be sampled (biased), without the smooth component
        // the panel's off-line light never would be.
        let third = N as f64 / 3.0;
        assert!((red as f64 - third).abs() < 0.03 * third);
        assert!((green as f64 - third).abs() < 0.03 * third);
        assert!((smooth as f64 - third).abs() < 0.03 * third);
    }

    /// The estimator identity the renderer relies on: over the λ
    /// mixture, `E[CMF(λ)·S(λ)/p(λ)]` equals `∫CMF·S dλ` — the intent's
    /// XYZ. This pins the whole hero-wavelength machinery (the mixture,
    /// the branch rule, inv_pdf) without any transport.
    #[test]
    fn hero_wavelength_estimator_is_unbiased_for_a_smooth_spd() {
        let intent = Vec3::new(0.8, 0.6, 0.3);
        let s = SpectrumKind::smooth_from_rgb(intent);
        let spectra = [s];
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let mut mean = VEC_ZERO;
        const N: usize = 200000;
        for _ in 0..N {
            let w = sample_wavelength(&spectra, |x| *x, &mut rng);
            let le = s.eval(w.lambda, w.line_branch);
            mean = mean + cmf(w.lambda) * (le * w.inv_pdf);
        }
        mean = mean * (1.0 / N as f32);
        let xyz = srgb_to_xyz(intent);
        // The sample SD is ~2x the mean (uniform-λ over a bounded
        // integrand), so the SE at N = 200000 is ~0.5%; 2% is ~4 sigma.
        let scale = xyz.x.max(xyz.y).max(xyz.z);
        assert!((mean.x - xyz.x).abs() < 0.02 * scale, "{mean:?} vs {xyz:?}");
        assert!((mean.y - xyz.y).abs() < 0.02 * scale, "{mean:?} vs {xyz:?}");
        assert!((mean.z - xyz.z).abs() < 0.02 * scale, "{mean:?} vs {xyz:?}");
    }

    /// The mixed case: a line plus a smooth emitter. The line's response
    /// is `CMF(λ₀)·P` exactly (the atom's mass times the full power),
    /// the smooth emitter's its XYZ — and the mixture estimates the sum.
    #[test]
    fn hero_wavelength_estimator_is_unbiased_for_a_line_plus_smooth() {
        let p = 8.0f32;
        let line = SpectrumKind::Line { lambda: 650.0, power: p };
        let intent = Vec3::new(1.0, 1.0, 1.0);
        let smooth = SpectrumKind::smooth_from_rgb(intent);
        let spectra = [line, smooth];
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let mut mean = VEC_ZERO;
        const N: usize = 200000;
        for _ in 0..N {
            let w = sample_wavelength(&spectra, |x| *x, &mut rng);
            let le = if w.line_branch {
                line.eval(w.lambda, true)
            } else {
                smooth.eval(w.lambda, false)
            };
            mean = mean + cmf(w.lambda) * (le * w.inv_pdf);
        }
        mean = mean * (1.0 / N as f32);
        // The exact expectation: the line's CMF(650)*P plus the panel's
        // XYZ.
        let expected = cmf(650.0) * p + srgb_to_xyz(intent);
        let scale = expected.x.max(expected.y).max(expected.z);
        assert!((mean.x - expected.x).abs() < 0.02 * scale, "{mean:?} vs {expected:?}");
        assert!((mean.y - expected.y).abs() < 0.02 * scale, "{mean:?} vs {expected:?}");
        assert!((mean.z - expected.z).abs() < 0.02 * scale, "{mean:?} vs {expected:?}");
    }

    #[test]
    fn rgb_albedo_is_a_partition_of_unity() {
        // R+G+B = 1 at every wavelength (the white albedo is exactly 1).
        for i in 0..=80 {
            let lambda = LAMBDA_MIN + i as f32 * CMF_STEP;
            let sum = rgb_to_albedo(Vec3::new(1.0, 1.0, 1.0), lambda);
            assert!((sum - 1.0).abs() < 1e-5, "sum {sum} at {lambda}");
        }
        // Grays are exact at every wavelength.
        for i in 0..=80 {
            let lambda = LAMBDA_MIN + i as f32 * CMF_STEP;
            let rho = rgb_to_albedo(Vec3::new(0.37, 0.37, 0.37), lambda);
            assert!((rho - 0.37).abs() < 1e-6, "gray {rho} at {lambda}");
        }
        // Convexity: rho lies between the extreme channels.
        let rgb = Vec3::new(0.1, 0.5, 0.9);
        for i in 0..=80 {
            let lambda = LAMBDA_MIN + i as f32 * CMF_STEP;
            let rho = rgb_to_albedo(rgb, lambda);
            assert!(rho >= 0.1 - 1e-6 && rho <= 0.9 + 1e-6, "{rho} at {lambda}");
        }
        // Hues: red reflects only long wavelengths, blue only short.
        assert!(rgb_to_albedo(Vec3::new(1.0, 0.0, 0.0), 500.0) < 1e-6);
        assert!(rgb_to_albedo(Vec3::new(1.0, 0.0, 0.0), 650.0) > 0.99);
        assert!(rgb_to_albedo(Vec3::new(0.0, 0.0, 1.0), 650.0) < 1e-6);
        assert!(rgb_to_albedo(Vec3::new(0.0, 0.0, 1.0), 420.0) > 0.99);
        assert!(rgb_to_albedo(Vec3::new(0.0, 1.0, 0.0), 550.0) > 0.99);
    }

    #[test]
    fn sellmeier_bk7_matches_the_datasheet() {
        // n at the sodium D line (589 nm): the N-BK7 datasheet value is
        // 1.51680 (at 587.6 nm).
        assert!((sellmeier_n(&BK7, 589.0) - 1.5168).abs() < 2e-4);
        // Normal dispersion: n decreases monotonically across the
        // visible range.
        let mut prev = sellmeier_n(&BK7, 380.0);
        for i in 1..=80 {
            let lambda = LAMBDA_MIN + i as f32 * CMF_STEP;
            let n = sellmeier_n(&BK7, lambda);
            assert!(n < prev + 1e-9, "n rose at {lambda}: {n} > {prev}");
            prev = n;
        }
        // The visible-range dispersion magnitude.
        let spread = sellmeier_n(&BK7, 400.0) - sellmeier_n(&BK7, 700.0);
        assert!(spread > 0.015, "dispersion too weak: {spread}");
    }

    /// The background's CMF-basis coefficients — the WGSL port inlines
    /// them as constants, so this pins the exact values the shader must
    /// carry (the CPU and the shader cannot drift apart).
    #[test]
    fn background_coeffs_match_the_wgsl_constants() {
        let c = background_coeffs();
        // The exact values hardcoded in shaders/bdpt/trace.wgsl's background
        // path. If this test fails after a CMF-table or matrix change,
        // update the shader's `coeffs` constant to match.
        assert!((c.x - 5.4379e-5).abs() < 5e-8, "x: {c:?}");
        assert!((c.y - 2.1023e-4).abs() < 5e-7, "y: {c:?}");
        assert!((c.z - 3.3559e-4).abs() < 5e-7, "z: {c:?}");
    }
}
