// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the spectral transport machinery (paper §9.2) — the CIE
// CMF, the sensor conversion, the albedo reconstruction, Sellmeier
// dispersion, the emitter branch rule, and the hero-wavelength draw.

// --- Spectral transport (paper §9.2) ---
//
// The CIE 1931 2° color matching functions (x̄, ȳ, z̄), 380–780 nm at
// 5 nm — 81 knots, the same table as src/spectrum.rs (source: CIE
// 018:2019). CMF[i] = (x̄, ȳ, z̄) at 380 + i·5 nm. The sensor converts
// the spectral estimate through these; the emitters' smooth spectra
// are CMF-basis combinations whose tristimulus integral is exactly the
// RGB intent.

const LAMBDA_MIN: f32 = 380.0;
const LAMBDA_MAX: f32 = 780.0;
const CMF_STEP: f32 = 5.0;

const CMF = array<vec3<f32>, 81>(
    vec3<f32>(0.001368, 0.000039, 0.006450), // 380
    vec3<f32>(0.002236, 0.000064, 0.010550), // 385
    vec3<f32>(0.004243, 0.000120, 0.020050), // 390
    vec3<f32>(0.007650, 0.000217, 0.036210), // 395
    vec3<f32>(0.014310, 0.000396, 0.067850), // 400
    vec3<f32>(0.023190, 0.000640, 0.110200), // 405
    vec3<f32>(0.043510, 0.001210, 0.207400), // 410
    vec3<f32>(0.077630, 0.002180, 0.371300), // 415
    vec3<f32>(0.134380, 0.004000, 0.645600), // 420
    vec3<f32>(0.214770, 0.007300, 1.039050), // 425
    vec3<f32>(0.283900, 0.011600, 1.385600), // 430
    vec3<f32>(0.328500, 0.016840, 1.622960), // 435
    vec3<f32>(0.348280, 0.023000, 1.747060), // 440
    vec3<f32>(0.348060, 0.029800, 1.782600), // 445
    vec3<f32>(0.336200, 0.038000, 1.772110), // 450
    vec3<f32>(0.318700, 0.048000, 1.744100), // 455
    vec3<f32>(0.290800, 0.060000, 1.669200), // 460
    vec3<f32>(0.251100, 0.073900, 1.528100), // 465
    vec3<f32>(0.195360, 0.090980, 1.287640), // 470
    vec3<f32>(0.142100, 0.112600, 1.041900), // 475
    vec3<f32>(0.095640, 0.139020, 0.812950), // 480
    vec3<f32>(0.057950, 0.169300, 0.616200), // 485
    vec3<f32>(0.032010, 0.208020, 0.465180), // 490
    vec3<f32>(0.014700, 0.258600, 0.353300), // 495
    vec3<f32>(0.004900, 0.323000, 0.272000), // 500
    vec3<f32>(0.002400, 0.407300, 0.212300), // 505
    vec3<f32>(0.009300, 0.503000, 0.158200), // 510
    vec3<f32>(0.029100, 0.608200, 0.111700), // 515
    vec3<f32>(0.063270, 0.710000, 0.078250), // 520
    vec3<f32>(0.109600, 0.793200, 0.057250), // 525
    vec3<f32>(0.165500, 0.862000, 0.042160), // 530
    vec3<f32>(0.225750, 0.914850, 0.029840), // 535
    vec3<f32>(0.290400, 0.954000, 0.020300), // 540
    vec3<f32>(0.359700, 0.980300, 0.013400), // 545
    vec3<f32>(0.433450, 0.994950, 0.008750), // 550
    vec3<f32>(0.512050, 1.000000, 0.005750), // 555
    vec3<f32>(0.594500, 0.995000, 0.003900), // 560
    vec3<f32>(0.678400, 0.978600, 0.002750), // 565
    vec3<f32>(0.762100, 0.952000, 0.002100), // 570
    vec3<f32>(0.842500, 0.915400, 0.001800), // 575
    vec3<f32>(0.916300, 0.870000, 0.001650), // 580
    vec3<f32>(0.978600, 0.816300, 0.001400), // 585
    vec3<f32>(1.026300, 0.757000, 0.001100), // 590
    vec3<f32>(1.056700, 0.694900, 0.001000), // 595
    vec3<f32>(1.062200, 0.631000, 0.000800), // 600
    vec3<f32>(1.045600, 0.566800, 0.000600), // 605
    vec3<f32>(1.002600, 0.503000, 0.000340), // 610
    vec3<f32>(0.938400, 0.441200, 0.000240), // 615
    vec3<f32>(0.854450, 0.381000, 0.000190), // 620
    vec3<f32>(0.751400, 0.321000, 0.000100), // 625
    vec3<f32>(0.642400, 0.265000, 0.000050), // 630
    vec3<f32>(0.541900, 0.217000, 0.000030), // 635
    vec3<f32>(0.447900, 0.175000, 0.000020), // 640
    vec3<f32>(0.360800, 0.138200, 0.000010), // 645
    vec3<f32>(0.283500, 0.107000, 0.000000), // 650
    vec3<f32>(0.218700, 0.081600, 0.000000), // 655
    vec3<f32>(0.164900, 0.061000, 0.000000), // 660
    vec3<f32>(0.121200, 0.044580, 0.000000), // 665
    vec3<f32>(0.087400, 0.032000, 0.000000), // 670
    vec3<f32>(0.063600, 0.023200, 0.000000), // 675
    vec3<f32>(0.046770, 0.017000, 0.000000), // 680
    vec3<f32>(0.032900, 0.011920, 0.000000), // 685
    vec3<f32>(0.022700, 0.008210, 0.000000), // 690
    vec3<f32>(0.015840, 0.005723, 0.000000), // 695
    vec3<f32>(0.011359, 0.004102, 0.000000), // 700
    vec3<f32>(0.008111, 0.002929, 0.000000), // 705
    vec3<f32>(0.005790, 0.002091, 0.000000), // 710
    vec3<f32>(0.004109, 0.001484, 0.000000), // 715
    vec3<f32>(0.002899, 0.001047, 0.000000), // 720
    vec3<f32>(0.002049, 0.000740, 0.000000), // 725
    vec3<f32>(0.001440, 0.000520, 0.000000), // 730
    vec3<f32>(0.001000, 0.000361, 0.000000), // 735
    vec3<f32>(0.000690, 0.000249, 0.000000), // 740
    vec3<f32>(0.000476, 0.000172, 0.000000), // 745
    vec3<f32>(0.000332, 0.000120, 0.000000), // 750
    vec3<f32>(0.000235, 0.000085, 0.000000), // 755
    vec3<f32>(0.000166, 0.000060, 0.000000), // 760
    vec3<f32>(0.000117, 0.000042, 0.000000), // 765
    vec3<f32>(0.000083, 0.000030, 0.000000), // 770
    vec3<f32>(0.000059, 0.000021, 0.000000), // 775
    vec3<f32>(0.000042, 0.000015, 0.000000), // 780
);

// The CMF at `lambda` (nm), linearly interpolated between knots and 0
// outside the table range.
fn cmf(lambda: f32) -> vec3<f32> {
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) { return vec3<f32>(0.0); }
    let t = (lambda - LAMBDA_MIN) / CMF_STEP;
    var i = u32(t);
    if (i > 79u) { i = 79u; }
    let f = t - f32(i);
    let a = CMF[i];
    let b = CMF[i + 1u];
    return vec3<f32>(
        a.x + (b.x - a.x) * f,
        a.y + (b.y - a.y) * f,
        a.z + (b.z - a.z) * f,
    );
}

// XYZ → linear sRGB (IEC 61966-2-1). Out-of-gamut XYZ (a pure 532 nm
// green) produces negative channels — correct; the display stage
// gamut-maps them (§9.2).
fn xyz_to_srgb(xyz: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
        0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z,
    );
}

// The sensor conversion: the rendered linear-sRGB sample for a spectral
// radiance estimate at `lambda`, weighted by 1/p(λ).
fn sensor_rgb(lambda: f32, radiance: f32, inv_pdf: f32) -> vec3<f32> {
    return xyz_to_srgb(cmf(lambda) * (radiance * inv_pdf));
}

// The spectral reflectance of an RGB albedo at `lambda`: the
// complementary-ramp partition of unity (grays exact, energy
// conservation automatic; see spectrum.rs::rgb_to_albedo).
fn rgb_to_albedo(rgb: vec3<f32>, lambda: f32) -> f32 {
    let b = 1.0 - smoothstep01((lambda - 440.0) / 60.0);
    var g = 0.0;
    if (lambda > 440.0 && lambda < 500.0) {
        g = smoothstep01((lambda - 440.0) / 60.0);
    } else if (lambda >= 500.0 && lambda <= 570.0) {
        g = 1.0;
    } else if (lambda > 570.0 && lambda < 630.0) {
        g = 1.0 - smoothstep01((lambda - 570.0) / 60.0);
    }
    let r = smoothstep01((lambda - 570.0) / 60.0);
    return rgb.x * r + rgb.y * g + rgb.z * b;
}

// The smoothstep ramp on [0, 1] (0 outside) — the shape of the albedo
// reconstruction's ramps.
fn smoothstep01(t: f32) -> f32 {
    let c = clamp(t, 0.0, 1.0);
    return c * c * (3.0 - 2.0 * c);
}

// Sellmeier's dispersion formula: n²(λ) = 1 + Σ Bᵢλ²/(λ² − Cᵢ), λ in µm.
fn sellmeier_n(coeffs: array<f32, 6>, lambda_nm: f32) -> f32 {
    let l = lambda_nm / 1000.0;
    let l2 = l * l;
    var n2 = 1.0;
    // Unrolled: WGSL array indexing in a loop is fine, but the unroll
    // keeps the constant-foldable zeros cheap.
    n2 = n2 + coeffs[0] * l2 / (l2 - coeffs[3]);
    n2 = n2 + coeffs[1] * l2 / (l2 - coeffs[4]);
    n2 = n2 + coeffs[2] * l2 / (l2 - coeffs[5]);
    return sqrt(n2);
}

// The refractive index of a glass vertex at the path's hero wavelength:
// Sellmeier when the coefficients are set, the constant otherwise.
fn glass_n(ior: f32, sellmeier: array<f32, 6>, lambda: f32) -> f32 {
    if (sellmeier[0] != 0.0 || sellmeier[1] != 0.0 || sellmeier[2] != 0.0) {
        return sellmeier_n(sellmeier, lambda);
    }
    return ior;
}

// Bit-exact f32 equality as a u32 compare: the values are exact line
// wavelengths (650.0, 532.0 — whole floats, no arithmetic), so the bit
// patterns are identical iff the values are. The NVIDIA 615.78.08
// shader compiler crashes (SIGILL in libnvidia-gpucomp) on the plain
// f32 `==` between a buffer-loaded value and a dynamic value in this
// shader's context (the known `gc_driver_f32_compare_constant` bug
// class); the integer compare avoids the broken path entirely.
fn lambda_eq(a: f32, b: f32) -> bool {
    return bitcast<u32>(a) == bitcast<u32>(b);
}

// The spectral radiance of a light at `lambda` under the drawn branch
// (the branch rule): a line radiates its full power on its own line
// branch, a smooth SPD its density on the smooth branch, everything
// else is dark.
fn light_eval(l: GpuLight, lambda: f32, line_branch: bool) -> f32 {
    if (l.spectrum_kind == 1u) {
        // A line: radiates only on its own line branch. Nested ifs,
        // not `&&`: the NVIDIA 615.78.08 shader compiler crashes
        // (SIGILL in libnvidia-gpucomp) on the vectorized
        // bool-and-f32-compare pattern this used to form.
        if (line_branch) {
            if (lambda_eq(l.lambda, lambda)) { return l.power; }
        }
        return 0.0;
    }
    // A smooth SPD: radiates only on the smooth branch.
    if (line_branch) { return 0.0; }
    return dot(cmf(lambda), vec3<f32>(l.coeff_x, l.coeff_y, l.coeff_z));
}

// True when the light radiates at `lambda` under the drawn branch —
// the branch rule as a set-membership test, used to pick the light
// subpath's starting emitter among the *active* ones. The test is
// *structural*, not sign-based: a smooth CMF-basis spectrum can
// evaluate negative at some wavelengths (a legitimate part of the
// response integral — dropping it would bias the estimator), so a
// smooth spectrum is active on the whole smooth branch and a line only
// on its own line branch.
fn light_active(l: GpuLight, lambda: f32, line_branch: bool) -> bool {
    // Structured exactly like light_eval: the lambda comparison feeds
    // control flow (`if (...) return true`), never a returned bool
    // value. The NVIDIA 615.78.08 shader compiler crashes (SIGILL in
    // libnvidia-gpucomp, the gc_driver_f32_compare_constant bug class)
    // when an f32 comparison against the dynamic hero wavelength is
    // materialized as a value; consumed by a branch it compiles fine
    // (light_eval has carried the same comparison all along).
    if (l.spectrum_kind == 1u) {
        if (line_branch) {
            if (lambda_eq(l.lambda, lambda)) {
                return true;
            }
        }
        return false;
    }
    return !line_branch;
}

// The per-path wavelength sample: the hero wavelength, the sensor
// weight 1/p(λ), and which branch of the emitter mixture was drawn.
// Mirrors `spectrum::sample_wavelength` exactly (the mixture over the
// emitters: pick one uniformly, then take its branch — a line's λ is
// its own line, a smooth emitter's λ uniform over the visible range).
struct WaveSample {
    lambda: f32,
    inv_pdf: f32,
    line_branch: bool,
};

fn sample_wavelength(state: ptr<function, u32>) -> WaveSample {
    let n = uniforms.n_lights;
    if (n == 0u) {
        return WaveSample(550.0, 1.0, false);
    }
    let pick = min(u32(f32(n) * pcg(state)), n - 1u);
    let l = lights[pick];
    if (l.spectrum_kind == 1u) {
        // The line branch: the atom's mass is k₀/n (k₀ lines at this
        // exact λ₀), so 1/p = n/k₀. Nested ifs, not `&&` (see
        // light_eval — the NVIDIA 615.78.08 compiler bug).
        var k = 0u;
        for (var i = 0u; i < n; i = i + 1u) {
            let li = lights[i];
            if (li.spectrum_kind == 1u) {
                if (lambda_eq(li.lambda, l.lambda)) { k = k + 1u; }
            }
        }
        let kk = max(k, 1u);
        return WaveSample(l.lambda, f32(n) / f32(kk), true);
    }
    // The smooth branch: λ uniform over the visible range; the
    // continuous density is (n_smooth/n)/400, so 1/p = 400·n/n_smooth.
    var n_smooth = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        if (lights[i].spectrum_kind == 0u) { n_smooth = n_smooth + 1u; }
    }
    let ns = max(n_smooth, 1u);
    let lambda = LAMBDA_MIN + pcg(state) * (LAMBDA_MAX - LAMBDA_MIN);
    return WaveSample(lambda, (LAMBDA_MAX - LAMBDA_MIN) * f32(n) / f32(ns), false);
}
