# plan.md — Applying the Paper's Advanced Techniques (§9 "Changing the Model")

Status: **in progress** — see the checklist at the end.

The paper (`paper.md`) documents the renderer as it stands: BDPT + MIS over all
connection strategies, the specular bridge ladder for laser beams through delta
mirror chains, two-tier firefly rejection, dual CPU/GPU backends with a 1:1
WGSL port. §9 then analyzes five *model changes* — the advanced techniques —
ordered by cost. This plan implements them in the paper's own order, keeping
every guarantee §9 says must survive.

## What the paper guarantees survives (§9.7 summary)

| Technique | Physics added | MIS | Bridge ladder | Firefly machinery | Cost |
|---|---|---|---|---|---|
| **Spectral + dispersion (§9.2)** | n(λ), prisms, true laser colors | **invariant** (λ factorizes out of every ratio) | survives (mirrors achromatic); glass rungs would become per-λ fans | bound becomes spectral | **low — plumbing** |
| **Glossy microfacets (§9.5)** | rough mirrors, soft reflections | connections through glossy vertices become *valid* | rungs become fuzzy images; bridge = small-roughness limit of the glossy connection | unchanged | **moderate** |
| Polarization (§9.4) | s/p split, Brewster, coatings | invariant | survives | statistic reads I | 4× state |
| Participating media (§9.3) | beam visible in fog, RTE | gains distance pdfs (absorbed by the ratio walk) | **medium-agnostic** — fire the mixture at medium vertices | unchanged | medium |
| Wave optics (§9.1) | diffraction, interference, speckle | bypassed (coherent channel is deterministic) | survives as the geometric shadow; rungs gain phase (path length already preserved) | survives by design | high |

**Session scope:** Phase 1 (spectral + dispersion) and Phase 2 (glossy
microfacets), end to end — CPU backend, WGSL port, tests, demo scene. Phases
3–5 are designed below for follow-up sessions, in the paper's cost order.

---

## Phase 1 — Spectral transport + dispersion (§9.2)

The paper's own pick: *"the cheapest of the model changes and the most
visually rewarding for the demo scene (dispersive caustics through the glass
cube; physically correct laser colors)"*.

### Design decisions

1. **Hero wavelength.** One `lambda: f32` (nm) is sampled per path *before*
   the eye walk (glass refraction directions depend on n(λ)), carried through
   both subpaths, and converted to RGB only at the sensor. The transport
   loop's structure is unchanged — "a float rides along" (§9.2).

2. **Scalar throughput.** Radiance becomes a spectral density: every
   `throughput: Vec3` in the walks, `PathVertex`, `BsdfSample.weight`,
   `rr_prefix_products`'s alphas and the strategy contributions becomes
   `f32`. `bdpt_trace` still returns a `Vec3` — the *sensor-converted* sample
   — so `camera.rs`, `renderer.rs` (accumulation, winsorization, bilateral,
   tone map) and `output.rs` are untouched.

3. **λ sampling = a mixture over emitters** (importance-sampled by the
   emitter spectrum, §9.2). Component `e` is picked with weight `1/n_lights`;
   within a component:
   - **Line** (laser): λ is *deterministic* — §9.2: "for lasers it adds none".
   - **CmfComb** (smooth emitter): λ ∝ ȳ(λ) by rejection sampling.
   
   The mixture density `p(λ) = (1/n)·[Σ_lines δ(λ−λ_e) + Σ_cmf ȳ(λ)/∫ȳ]` is
   what the `1/p(λ)` sensor weighting divides by. Support check: the smooth
   components cover every continuous SPD; the delta components cover the
   laser lines — without a delta component the lasers would never be sampled
   (biased), without a smooth component the panel's off-line light would
   never be sampled (biased). Both are required.

4. **MIS is untouched — and must stay untouched.** λ is sampled once per
   path, independent of all direction sampling, so every strategy pdf
   factorizes as `p(directions)·p(λ)` and the λ factors cancel in every
   incremental-ratio (§3.2). The `1/p(λ)` weighting is applied exactly once,
   at the sensor conversion. No change to `mis_weight`, `pdf_eye/pdf_light`,
   the RR prefix products, or the delta convention.

5. **Emitter spectra** — a `SpectrumKind` on `Material`/`AreaLight`/`LightData`:
   - `CmfComb` — the payload is the existing `emission: Vec3` *intent*. The
     SPD is `S(λ) = CMF(λ)ᵀ·G⁻¹·XYZ(intent)` with `G = ∫CMF·CMFᵀ dλ` (the CMF
     Gram matrix, computed once from the table). Then
     `∫CMF(λ)·S(λ) dλ = XYZ(intent)` **exactly**, so the emitter renders to
     exactly its intended RGB in expectation. This keeps every existing
     exact-color test meaningful (as means over N samples) and the demo's
     ceiling panel renders exactly white 1000 as before. `Material::emissive`
     and `Material::laser` construct CmfComb — the existing tests keep their
     semantics.
   - `Line { lambda, power }` — the lasers. §9.2: "a laser is monochromatic,
     so λ is deterministic (the current RGB trick of putting 4×10⁸ in one
     channel is replaced by the physically correct spectral delta, and the
     out-of-gamut problem — a pure 532 nm green is outside sRGB — moves to
     the display stage where it belongs)". Demo: red laser = 650 nm line,
     green laser = 532 nm line; `power` chosen so the rendered spot
     luminance matches the old RGB hack (the `1/p(λ)` factor ≈ ×3 replaces
     the channel stuffing). Out-of-gamut samples carry negative G/B
     channels through the linear pipeline and clamp at display — by design.

6. **Sensor conversion.** Sample XYZ `= CMF(λ)·L(λ)/p(λ)`, then the standard
   XYZ→linear-sRGB matrix. CIE 1931 2° CMFs, 380–780 nm at 5 nm (81 entries),
   linear interpolation between knots; the table is taken from a canonical
   source and pinned by anchor tests. `G⁻¹` is computed once (LazyLock) on
   the CPU; for the WGSL port the nine constants are printed by a unit test
   and hardcoded.

7. **Albedo spectra.** `rgb_to_albedo(rgb, λ)` reconstructs a spectral
   reflectance from the material's RGB albedo as a **partition of unity**:
   three complementary smoothstep ramps R/G/B (B: 1 below 440 → 0 at 500;
   G: 0 at 440 → 1 at 500 → 1 until 570 → 0 at 630; R: 0 at 570 → 1 at 630)
   with `R+G+B ≡ 1`, and `ρ(λ) = r·R + g·G + b·B`. Properties: grays are
   **exact** (`(1,1,1) → ρ ≡ 1`, so the white furnace keeps its exact
   contract), ρ is a convex combination (energy conservation is automatic,
   ρ ∈ [min rgb, max rgb]), hues are approximate (documented; Smits fits are
   a drop-in refinement). The test suite's albedos are almost all grays.

8. **Dispersion.** `Bsdf::Glass { ior }` gains `sellmeier: Option<[f32; 6]>`;
   `n(λ) = Sellmeier(λ)` when present, the constant `ior` otherwise (the
   existing glass tests pin exact constant-n Fresnel/Snell values and must
   keep passing). BK7 coefficients (λ in µm): B = (1.03961212, 0.231792344,
   1.01046945), C = (0.00600069867, 0.0200179144, 103.560653) µm² —
   n(400) ≈ 1.530, n(588) ≈ 1.517, n(700) ≈ 1.513. The demo's glass cube
   becomes BK7: its caustic picks up dispersive fringes. All BSDF methods
   (`sample_bsdf`, `eval_bsdf`, `is_specular_pair`, `specular_weight`) gain a
   `lambda` parameter; hypothetical MIS extensions use the same λ as the
   realized walk (λ is fixed per path, so the `PathView` discipline is
   preserved — realized and hypothetical quantities still cannot drift).

9. **Firefly clamp goes spectral** (§9.2: "the bound becomes a spectral
   quantity — 8× the brightest emitter's luminance *at the sampled λ*").
   The direct-view sample luminance of emitter `e` at λ is
   `ȳ(λ)·Le_e(λ)/p(λ)` (the Rec709 luminance of sRGB(XYZ) is Y by the duality
   of the matrices), so `bound(λ) = 8·max_e[ȳ(λ)·Le_e(λ)]/p(λ)` — the faithful
   generalization of the old `8·max emission luminance` (the old bound was
   exactly "8× the direct-hit sample's luminance"). The clamp still operates
   on the final RGB sample, chroma-preserving; the two-tier rationale (§5)
   is unchanged — the loose tier stays physically motivated, the tight tier
   stays statistical. Rare high-`1/p(λ)` draws at the spectrum edges carry
   proportionally tiny `Le(λ)` there, so their absolute sample magnitude
   stays moderate; the residual tail is what the winsorization tier eats.

10. **The bridge ladder survives unchanged** (§9.2: "reflection is
    achromatic, so the rungs are λ-independent"). The mixture pdf is
    geometric; the bridge sample weight uses `ρ(λ)/π`. Glass already ends
    the ladder (§4.3) — with dispersion a glass rung would become a per-λ
    fan, which is out of scope (the paper's "rainbow ladder" is noted as a
    follow-up).

11. **Variance budget.** §9.2: "for smooth emitters this adds ~2–4×
    variance; for lasers it adds none". With three emitters the panel keeps
    1/3 of λ draws → ~3× variance on ambient — cleaned up by the existing
    128-frame accumulation. The lasers keep 1/3 of draws each (vs. all
    samples before); the spots converge correspondingly slower.

### Files

- **new `src/spectrum.rs`** — CMF table + interpolation, XYZ↔sRGB matrices,
  `G`/`G⁻¹`, `SpectrumKind` (eval at λ, component sampling, mixture pdf,
  `sample_wavelength(lights, rng)`), `rgb_to_albedo`, Sellmeier eval,
  `cmf_luminance(λ)`; unit tests.
- **`src/geometry.rs`** — `Material` gains `spectrum: SpectrumKind`;
  `Bsdf::Glass` gains `sellmeier`; λ threads through the BSDF methods;
  `AreaLight` gains the spectrum; `BsdfSample.weight` → `f32`.
- **`src/scene.rs`** — `MaterialData`/`LightData` mirror the spectrum;
  `glass_bk7()`, `add_laser_line`; the beam trace/ladder are untouched
  (geometric).
- **`src/bdpt.rs`** — scalar-throughput refactor; λ sampling at path start;
  sensor conversion + spectral clamp at the end; `PathView` carries λ.
- **`src/main.rs`** — demo: Line lasers (650/532 nm), BK7 glass cube.
- **`src/gpu.rs` + `shaders/bdpt/`** — the 1:1 port: `GpuPrim` grows to
  144 B (Sellmeier[6] + roughness + pad), `GpuLight` reuses free pads for
  the spectrum kind/λ/power, `PathVertex` fields go scalar, the CMF table +
  `G⁻¹` + the albedo ramps are hardcoded consts, λ sampling + conversion +
  spectral clamp in-shader, `intersect_scene` evaluates `Le(λ)`, `ρ(λ)`,
  `n(λ)` at hit time. The `firefly_clamp` uniform becomes a reserved pad
  (the bound is now λ-dependent and computed in-shader). Layout asserts and
  the naga validation test are the gates.

### Tests

New: CIE table anchors; XYZ↔sRGB white round-trip; **CmfComb renders exactly
its RGB** (MC — the killer test); Line spectra render saturated colors
(650 nm → r ≫ g, b); Sellmeier BK7 values (monotone decreasing, anchors);
**dispersion bends blue more** (oblique `refract_dir` at 400 vs 700 nm);
albedo reconstruction (exact grays, convex, smooth); the λ mixture pdf
normalizes and covers every emitter's support; a Line-laser direct hit
matches its analytic rendered value.

Reworked (semantics preserved, assertions become means over N samples with
Monte-Carlo tolerances — a spectral sample is random per-λ, where the RGB
pipeline returned a deterministic value): `looking_straight_at_the_light…`,
`rr_leaves_short_paths_exact`, `sphere_light_direct_hit`,
`laser_beam_lights_only_inside_the_cone`, `white_furnace`,
`glass_energy_furnace`, `glass_slab_transmits_energy`,
`diffuse_bsdf_reduces_to_the_cosine_sampling_constants`,
`glass_sampling_matches_the_fresnel_split` (scalar weights). The MIS
sum-to-one, RR-product and bridge-ladder tests are λ-agnostic and keep
their hand-computed values.

---

## Phase 2 — Glossy microfacet BSDFs (§9.5)

§9.5: *"Replace the perfect mirror with a GGX microfacet lobe. The delta
becomes continuous: connections through glossy vertices become valid
(nonzero eval), so the BDPT blind spot softens on its own."*

### Design decisions

1. **`Bsdf::Glossy { alpha, ior }`** — a dielectric GGX reflection lobe
   ("rough mirror"): D = α²/(π·(cosθ_h(α²−1)+1)²), separable Smith G1 =
   2c/(c+√(α²+c²(1−α²))), Schlick F from F0 = ((ior−1)/(ior+1))²,
   f = D·G·F/(4 cosθ_i cosθ_o). Sampling: the standard GGX NDF half-vector
   draw (cosθ_h = √((1−u)/(1+(α²−1)u))), reflect about h;
   pdf_ω = D·cosθ_h/(4|v·h|). `eval_bsdf` returns the same pdf — sampler and
   eval are one distribution, which is what the MIS machinery requires.

2. **Connections through glossy vertices become valid — for free.** Glossy
   is not delta: `is_delta()` is false, so `mis_weight`'s validity check
   admits glossy endpoints and the generic non-delta path (eval + G +
   shadow test) carries the connection. This is the paper's "the blind
   spot softens on its own": no MIS code changes, only the new `eval_bsdf`
   arm. `specular_weight` returns None for glossy (correct — it has no
   specular pair), so hypothetical extensions use the continuous pdf.

3. **The ladder crosses glossy mirrors with exact rungs.** In
   `bridges_of_light_in` and `trace_laser_beam`, `Glossy` is handled like
   `Mirror`: the beam's central ray reflects along the specular direction
   (the lobe's mean) and the image unfolds across the plane exactly. The
   rung stays the *exact* image disc: as a sampling technique it remains
   unbiased regardless of what happens downstream — the eye walk aims at
   the rung, hits the glossy mirror, and the GGX draw there must be
   near-specular to continue. That is precisely §9.5's "the bridge is the
   small-roughness limit of the glossy connection": the technique helps
   exactly where the lobe is narrow. The "fuzzy image" rung (the lobe's
   reflection of the disc, a variance optimization) is noted as a follow-up.

4. **No bridge mixture at glossy vertices.** The cosine–bridge mixture
   fires only at diffuse eye vertices, as now; glossy vertices sample their
   own lobe. §9.5's generalized mixture ("the bridge cone becomes the
   specular lobe's reflection of the image disc") is a follow-up.

5. **One-sided rule unchanged.** Glossy is not delta, so back-face hits
   break the walk exactly like diffuse — consistent with the existing rule.

6. **Demo.** One of the green laser's ping-pong mirror tiles becomes
   `glossy(alpha = 0.08, ior = 1.5)`: the tile picks up soft reflections of
   the room, the beam still lands (the trace follows the specular mean),
   the ladder still derives rung-per-crossing, and the spot softens — the
   visible contrast against the red laser's perfect-mirror chain. The
   red laser's back-wall mirror stays perfect.

### Tests

New: GGX samples stay in the hemisphere and the sampler/eval pdfs agree;
the near-mirror limit (α → 0.001 samples within ~1° of the reflection);
reciprocity f(in,out) = f(out,in); **directional-hemispherical reflectance
≤ 1** (MC — the energy guard; single-scatter GGX loses energy at high α, so
the demo uses low α and the test asserts ≤ 1 + ε and a sane floor); MIS
weights sum to one with a glossy vertex on the path (connections through it
now valid); the middle-leg glow survives glossy mirrors (α = 0.05, a looser
ratio than the perfect-mirror test).

---

## Phase 3 — Polarization (§9.4) — follow-up design

Stokes vectors (I,Q,U,V) replace the scalar spectral throughput (4 floats per
vertex at the hero λ); BSDFs become 4×4 Mueller matrices. The building blocks
already exist: `fresnel_dielectric` computes r_s and r_p separately — refactor
it to return both and assemble the standard dielectric interface Mueller
matrices (mirror coatings add distinct s/p *phases*, ellipticity upon
reflection). Diffuse = depolarizing (energy in I). The η² basic-radiance
factor multiplies all four components. MIS is untouched (sampling stays
polarization-independent → pdfs factorize); the firefly statistic reads I
(§9.4). Demo payoff: the lasers start linearly polarized (1,1,0,0) and the
glass cube's Brewster angle (~56°) selectively dims one polarization.
Tests: the Mueller matrix at normal incidence reduces to scalar r·diag;
at Brewster r_p = 0 kills the p component; energy in I conserved by the
furnace.

## Phase 4 — Participating media (§9.3) — follow-up design

A homogeneous fog volume (a box with σ_a, σ_s, Henyey–Greenstein g) makes the
laser legs *visible* — single scattering off the medium along each leg. The
walks sample distances by σ_t (clamped to the segment), gain medium
scattering vertices (the phase function plays the BSDF's role — connectable,
non-delta), and carry transmittance e^(−σ_t d) in the connection geometry
term. The bridge ladder is **medium-agnostic** (§9.3): a medium point beside
leg j+1 sees the aperture through exactly the first j mirrors — the one
required extension is firing the cosine–bridge mixture at *medium* vertices
with the phase function in the weight. The MIS machinery gains the
distance-sampling pdfs as additional per-edge factors — the incremental-ratio
walk absorbs them without structural change (each strategy's pdf gains the
product of its own edges' distance pdfs; the step ratio generalizes).
GPU: the medium joins the prim array (a kind tag + σ's), and the traversal
must return medium collisions — the largest WGSL change of the three.

## Phase 5 — Wave optics (§9.1) — follow-up design

The hybrid of §9.1: keep BDPT for the incoherent ambient; add a deterministic
**coherent channel** for the lasers. Propagate each beam as a Gaussian beam
through its mirror chain via ABCD ray-transfer matrices (free space and
mirror reflection are both ABCD operations; each rung's image disc is the
beam's virtual waist — the ladder *is* the geometric shadow of this
formalism). The rungs already preserve path length exactly (unit-tested), so
phase = 2π·(path length)/λ is one multiplication away: where two prefixes
illuminate the same point, deposit |u₁ + u₂|² (multi-rung interference).
Deposit |u|² modulated by a speckle pattern whose statistics match the
surface roughness PSD (Harvey–Shack) — static speckle is temporally
persistent, so the EMA σ grows and the grains converge to their true mean:
the firefly machinery survives *by design* (§9.1). The coherent channel
bypasses MIS entirely (deterministic). Note the paper's own caveat: the demo
aperture (w₀ = 1 cm, λ = 532 nm) sits ~600× inside the geometric-optics
regime (θ_d ≈ 1.7×10⁻⁵ rad vs the 0.01 rad cone), so the visible payoff
needs either a smaller-aperture showcase or the speckle texture.

---

## Parity & validation gates

- The WGSL stays a **1:1 port** of `bdpt.rs` (§2: "identical estimator
  semantics"). Every CPU change lands in the same edit set as its WGSL
  counterpart; compile-time layout asserts pin the structs byte-for-byte;
  the offline naga validation test must pass (it runs without a GPU).
- The `gpu_matches_cpu` oracle (ignored, needs GPU) keeps compiling and its
  tolerance rationale is re-checked.
- The test suite is the paper's evidence (§8): every existing test keeps its
  physical meaning. Exact single-sample color asserts become mean-over-N
  asserts (spectral samples are random per-λ); tolerances come from the
  Monte-Carlo standard error, not convenience.
- Gates, in order: `cargo test` green after Phase 1 CPU; green after
  Phase 2 CPU; naga validation + layout asserts green after the WGSL port;
  full `cargo test` green at the end.

## Risks & mitigations

1. **CMF table transcription** — fetch from a canonical source; anchor tests
   pin ȳ(555), the peak, and the ∫ȳ scale.
2. **Variance increase on smooth emitters** (~2–4×, per the paper) — the
   128-frame accumulation absorbs it; the lasers lose no conditional
   variance (λ deterministic on their lines).
3. **Out-of-gamut laser colors** (negative G/B in linear sRGB) — expected
   and correct; they flow through the linear pipeline and clamp at display
   ("moves to the display stage where it belongs", §9.2).
4. **GGX single-scatter energy loss at high roughness** — documented; the
   demo uses low α; the reflectance test asserts the ≤ 1 guard.
5. **Session overrun** — the checklist below tracks exactly where to resume;
   the CPU backend is the reference oracle and lands first.

## Checklist

- [x] plan.md written
- [x] Phase 1 CPU: `src/spectrum.rs` + geometry/scene/bdpt refactor + tests green
- [x] Phase 1 demo: Line lasers (650/532 nm), BK7 glass cube
- [x] Phase 2 CPU: GGX BSDF + valid connections + ladder crossing + tests green
- [x] Phase 2 demo: glossy ping-pong tile
- [x] WGSL port of Phases 1+2 (layout asserts + naga validation green)
- [x] README updated; full `cargo test` green (85 passed, 2 GPU tests ignored)

## Session log

- **Phase 1 (spectral + dispersion, §9.2)** — implemented end to end:
  hero-wavelength sampling from a branch-aware emitter mixture (lines
  deterministic on their branch, smooth SPDs uniform over the visible
  range — see spectrum.rs's module doc for why not ȳ-proportional),
  scalar spectral throughputs through both walks and the MIS machinery
  (λ factorizes out of every ratio — the machinery is untouched, as
  §9.2 guarantees), the CMF-comb smooth-spectrum construction (renders
  exactly its RGB intent in expectation — pinned by
  `smooth_emitter_renders_exactly_its_intent`), the complementary-ramp
  albedo reconstruction (grays exact, energy conservation automatic),
  Sellmeier dispersion (BK7 glass in the demo), the spectral firefly
  clamp (8× the brightest active emitter's direct-view sample luminance
  at the drawn λ), and the sensor conversion through the CIE CMFs.
- **Phase 2 (glossy microfacets, §9.5)** — implemented end to end: the
  GGX lobe (NDF sampling, Smith G1, Schlick F), connections through
  glossy vertices valid by construction (not delta — the generic eval
  path carries them, no MIS changes), the beam trace and bridge ladder
  crossing glossy mirrors along the specular mean (the rung stays the
  exact image disc; the technique remains unbiased), and the demo's
  glossy ping-pong tile.
- **Bugs found and fixed during validation:**
  1. The branch rule must be *structural*, not sign-based: the
     CMF-comb spectrum evaluates negative at some wavelengths (it is
     the unique XYZ-matching combination, not a physically nonnegative
     SPD), and dropping those wavelengths biased every smooth emitter
     ~4% dark. `WaveSample::active` now tests the spectrum's *kind*.
  2. The WGSL `GpuLight` spectrum fields must be scalars, not a vec3:
     WGSL's 16-byte vec3 alignment misaligns the 16-byte-strided
     `bridge_rungs` array against the Rust layout.
  3. `-(a.dot(b))` is not valid WGSL — `-dot(a, b)`.
- **Test-suite changes:** the exact single-sample color asserts became
  mean-over-N asserts with SE-derived tolerances (a spectral sample is
  random per-λ where the RGB pipeline was deterministic); the dispersion
  test uses two line emitters and a strongly dispersive Sellmeier glass
  (the test verifies the machinery, not BK7's magnitude — the BSDF-level
  test pins that); `gpu_renderer_smoke` is now `#[ignore]`d like
  `gpu_matches_cpu` (a driver crash takes down the test process with
  SIGILL rather than failing the test).
- **Phases 3–5** (polarization, participating media, wave optics) remain
  designed in this plan for follow-up sessions, in the paper's cost
  order.

---

# Session 2 — the laser model, performance, single backend

Status: **reviewed and designed** — see the checklist at the end.

A full code review (all of `src/`, both shaders, the paper, this plan)
preceded the three phases below. Two baseline bugs were found and fixed
immediately (see Hygiene); the suite is green (83 passed, 2 GPU tests
ignored).

## Phase 6 — The laser as a collimated beam (model rethink)

**Diagnosis — why the current laser reads as a spotlight.** The laser is
modeled as a tiny disc emitter radiating uniformly into a cone, with the
emission direction *independent* of the emission position — which is
literally the definition of a spotlight. Two independent tells:

1. **The cone is 50–300× too wide.** The demo uses θ = 0.001 rad (red) and
   0.005 rad (green); the physical divergence of a beam with waist
   w₀ = 1 cm is θ_d = λ/(πw₀) ≈ 2×10⁻⁵ rad. At a 5 m throw the cone
   spread (d·tanθ ≈ 5 mm) is comparable to the aperture radius (1 cm), so
   the spot's *penumbra* (width 2dθ ≈ 1 cm) is as wide as the beam
   itself: the spot is mostly soft edge. That soft blob is exactly what
   reads as "spotlight"; a laser spot is a sharp, *constant-size* beam
   cross-section.
2. **The radiance magic number hides the power.** `Le = 1.2e9` with
   A = πr² ≈ 3.1×10⁻⁴ m² and Ω = πθ² ≈ 3.1×10⁻⁶ sr corresponds to a beam
   power Φ = Le·A·Ω ≈ **1.2 W** — a Class-4 burning laser, not a pointer.
   Worse, because Φ ∝ Ω at fixed Le, θ is secretly a *brightness* knob:
   shrinking the cone dims the spot. That coupling is why θ was left
   artificially wide in the first place.

**The better model: a power-parametrized collimated beam.** A laser is
the *collimated* limit — position and direction perfectly correlated,
spot = the translated beam cross-section, divergence set by
diffraction. Concretely:

1. **Power parametrization.** `add_laser_line(center, axis, height,
   radius, power_w, lambda)`: specify the beam power Φ (W) and derive
   the line's spectral radiance `P = Φ/(A_ap·Ω(θ))`. The spot irradiance
   becomes E = Φ/(πw₀²) — *invariant to θ* (θ now controls only the
   divergence, as it physically does). A 5 mW pointer deposits
   E ≈ 16 W/m² — brighter than the floor under the ceiling panel
   (E ≈ 10 W/m²): physically correct and visually correct.
2. **Physical divergence.** θ = M²·λ/(π·w₀) with w₀ = the aperture
   radius; M² = 1 is diffraction-limited. The demo may use M² ≈ 10–30 so
   the spot grows *visibly* across the room — still 10–30× tighter than
   today. The spot becomes a sharp disc that stays constant along the
   mirror chain; the penumbra disappears because position and direction
   are now (nearly) perfectly correlated — which *is* the physical
   definition of collimation.
3. **Why the cone survives — and the direction-delta does not.** The
   "obvious" alternative — a direction-delta emitter (radiance
   ∝ δ(ω − axis), the exact GO limit) — is *worse* on all three counts:
   (a) the aperture becomes a delta connection endpoint, killing the
   s=1 direct-light strategy that carries the first-leg glow (the
   floor beside the beam connects straight to y₀ only because the cone
   is continuous); (b) the emitter departure pdf becomes a delta,
   requiring the symbolic-pdf convention in the MIS machinery (new
   edge cases); (c) numerically it needs a tolerance cone anyway — and
   the physically correct tolerance cone *is* the diffraction cone. The
   cone emitter with physical θ **is** the GO collimated beam with the
   wave-optics divergence as its tolerance: zero machinery changes,
   unbiased, connectable.
4. **The upgrade path to §9.1.** This model is the wave-optics
   channel's geometric skeleton: flat disc → Gaussian cross-section
   profile (needs the position-pdf plumbing in `pdf_light(m=k+1)` —
   moderate, deferred) → complex field |u|² with speckle (Phase 5's
   coherent channel). The ladder already preserves path length =
   phase, so each rung's image disc becomes the beam's virtual waist.

**Files.** `scene.rs` (`LightData::from_laser_line` gains `power_w`,
derives P; keep the radiance-based constructor for the analytic tests),
`main.rs` (demo constants: Φ = 5 mW, physical θ), `gpu.rs`/`bdpt/`
(nothing — the machinery is θ-agnostic).

**Tests.** Power invariance (fixed Φ, θ varying over decades: the spot
irradiance is unchanged); the footprint law (spot radius = w₀ + d·tanθ);
the rendered spot matches Φ/(πw₀²)·ρ/π analytically; the existing
cone-gate and ladder tests keep their wide-θ synthetic scenes (they
test the machinery, which is θ-agnostic).

## Phase 7 — Performance

**The hot spot: bridge qualification inside the MIS inner loop.**
Every pdf/weight query at a diffuse eye vertex calls `mixture_pdf`,
which loops over *all lights × all their rungs* — on the GPU loading
the full 256-byte `GpuLight` per light, **twice** (count + density); on
the CPU allocating a `Vec` per call. These queries run O(k²) times per
path (per connection strategy × per edge in the ratio walk and the RR
sweep): several hundred `mixture_pdf` calls per path, each doing ~8–10
storage loads. The bridge loop is plausibly 5–10× the rest of the
shader combined — the single biggest win available.

1. **Flat rung buffer.** Move the rungs out of `GpuLight` into a
   dedicated `array<vec4<f32>>` storage buffer (xyz = image center,
   w = radius), uploaded once per scene. The qualification loop reads
   16 bytes per rung instead of 256 bytes per light; `GpuLight`
   shrinks by 128 bytes (the rung array and its pads go — which also
   retires the vec3-alignment trap that array caused).
2. **Per-vertex cached qualifying count.** `n_qual` depends only on
   (p, normal) — and the oriented normal at a path position is
   *query-independent*: every MIS/RR query uses the realized edge
   directions (`pv_event_pdf`'s `d_in` is always a realized path edge),
   and both walks stop at back-face diffuse hits, so the orientation
   never flips between queries. Compute `n_qual` once at vertex
   creation (one pass over the flat rung array), store it in
   `PathVertex`, early-out at 0 — the overwhelmingly common case.
   Semantics unchanged: the current code recomputes the identical
   value hundreds of times per path.
3. **CPU: kill the hot-path allocations.** `qualifying_bridges`
   returns `Vec<usize>` — a heap allocation in the innermost MIS
   query, dozens per path; `bdpt_trace` also collects `active` into a
   `Vec` per trace. After Phase 8 the CPU path is the test oracle, so
   this is test-suite speed, not production speed — but the same
   per-vertex cache applies 1:1.
4. **`mis_weight` per-edge caching.** `mis_valid(s2)` and
   `mis_step(s2)` both compute `pv_pdf_eye(s2+1)`; cache the per-m pdfs
   in small local arrays within one `mis_weight` call (~2× off that
   walk). Same for the CPU.
5. **Shared-prefix RR caching** (optional — measure after 1–4): qe[l]
   for l ≤ n_eye and ql[j] for j ≤ light_len are strategy-independent
   (the subpath prefixes are shared across strategies; only the
   crossover extensions differ). Precompute the prefix sweeps once
   per path. A delicate refactor of the MIS machinery — guard with the
   sum-to-one and differential tests.
6. **Ladder derivation: one BVH build.** `GpuRenderer::build_scene_buffers`
   calls per-laser `bridges_of_light`, and each call rebuilds the whole
   trait-object scene + BVH (`bridges_of_light` → `build_cpu_scene`) —
   N lasers → N full scene builds per upload, per frame under the M
   animation. Add a per-laser variant that shares one build (the flat
   `bridges()` already does exactly this).
7. **`update_scene` buffer reuse.** The M animation re-creates the
   prim/BVH/light buffers per frame; reuse allocations when capacity
   suffices (`write_buffer` only).
8. **Optional: SAH split** in both BVH builders (the room mixes
   5 m walls with cm-scale mirror tiles — median-split-on-widest-axis
   is weak there). Measure after 1–7.

## Phase 8 — Remove the CPU renderer

**Scope: remove the CPU *renderer*, keep the CPU *BDPT core*.** The CPU
implementation remains (a) the test oracle — the whole evidence suite
(§8) runs `bdpt_trace` on CPU scenes — and (b) the ladder-derivation
engine (`bridges_of_light` traces beams through the trait-object BVH).
Removing either would gut the paper's validation story; the renderer
frontend is what goes.

Remove:
- `renderer::CpuRenderer` and the `Renderer` trait (only `GpuRenderer`
  remains; the frontends use it concretely).
- `Cam::render` (the rayon pixel loop) and the CPU FHD fallback in
  `render_fhd` — the export is GPU-only; no adapter → clean error.
- The `rayon` dependency (nothing else uses it).
- The terminal frontend's CPU fallback branch (it becomes GPU-only; it
  already prefers the GPU).

Keep:
- `bdpt.rs`, `bvh.rs`, `geometry.rs`, `SceneData::build_cpu_scene` —
  the oracle + the ladder engine.
- `accumulate` + its unit tests as the *tested reference* of the
  winsorization policy the WGSL accumulation block implements (the
  parity discipline: the shader's policy stays pinned by a CPU-side
  test even though no CPU renderer consumes it).
- The `gpu_matches_cpu` oracle: its CPU reference becomes a small
  `#[cfg(test)]` render helper (a plain loop over pixels calling
  `bdpt_trace`; 32×32 needs no parallelism).
- The `renderer.rs` export constants (FHD_*, MAX_ACCUMULATED_FRAMES,
  SAMPLES_FHD) move to `gpu.rs` or a small config module.

README updates land with the phase (the "Two Frontends" / CPU-fallback
sections).

## Hygiene (found in review; first two fixed immediately)

- **[FIXED]** `scene_builds_and_is_sampleable` expected 3 lights; the
  demo has 4 (a second ceiling panel was added without updating the
  test).
- **[FIXED]** `Material::is_specular_pair` hardcoded λ = 550 nm while
  the WGSL port passes the path's hero λ — a 1:1-port violation that
  corrupts the MIS weights of dispersive-glass paths on the CPU: the
  550-nm refraction direction is not the realized λ's, the
  specular-pair test fails, and the realized strategy's pdf collapses
  to 0. Now takes `lambda` like every other BSDF query.
- **The `M` sweep animation is a no-op**: `build_scene_with(_laser_aim_y)`
  ignores its parameter (the `aim_point` line is commented out), so the
  window frontend rebuilds and re-uploads the whole scene every frame
  *while changing nothing* — a documented README feature that animates
  nothing at full cost. Restore the aim (note: this moves the default
  spot — the fixed aim and the aim_point aim are different beams) or
  remove the feature. Either way Phase 7.6/7.7 make the per-frame
  re-upload cheap.
- **Doc drift**: `SAMPLES_FHD = 1280` vs README "1024 samples per pixel"
  vs paper "128 spp" — three different numbers for one constant; and
  `GpuRenderer::render_frame`'s doc says "two samples per pixel" but
  dispatches 4.
- **Debris**: `progress.patch` in the repo root (a leftover diff from a
  previous session); `screenshot.png` is a generated artifact (gitignore).
- **Dead-code warnings (9)**: mostly test-only items (`Material::*`
  constructors, `trace_laser_beam`, `background_coeffs`,
  `BeamTrace`/`BeamSpot`) — mark `#[cfg(test)]` where honest; delete
  `firefly_clamp_bound` (superseded by the spectral bound).

## Session 2 checklist

- [x] Review: code + paper + plan read end to end; baseline green
- [x] Hygiene: stale light-count test fixed; `is_specular_pair` λ fixed
- [x] Phase 6: power-parametrized beam emitter + physical divergence +
      demo constants + tests
- [ ] Phase 7.1–7.4: flat rung buffer, per-vertex n_qual cache, CPU
      allocation fix, mis_weight pdf caching (then measure)
- [ ] Phase 7.5–7.8 (optional, post-measurement): shared-prefix RR,
      SAH, buffer reuse
- [ ] Phase 8: CPU renderer removal (keep the BDPT core, `accumulate`
      tests, oracle helper)
- [ ] Hygiene: M-animation decision, doc sync, debris removal,
      dead-code silence
- [ ] README + paper Appendix A/B updated for the single backend and
      the new laser model

### Phase 6 session log

Implemented end to end (CPU + WGSL 1:1 port + tests + demo):

1. **Power parametrization.** `LightData::from_laser_beam` /
   `SceneData::add_laser_beam`: specify Φ (W) and θ; the line's
   spectral radiance is derived as P = Φ/(A_ap·Ω(θ)) with
   Ω = 2π·versine. The spot irradiance E = Φ/(π·w₀²) is θ-invariant
   (pinned by `laser_beam_power_parametrization_is_theta_invariant`
   and the analytic spot test). The radiance-parametrized
   constructors stay for the analytic tests.
2. **The versine representation.** The plan's "machinery is
   θ-agnostic" claim missed a hard numerical wall: at the physical
   divergences the f32 `theta.cos()` rounds to exactly 1.0 (at the
   diffraction limit, M² = 1) or keeps ~1 significant digit (M² = 30,
   an 8% solid-angle error) — the cone's solid angle 2π(1−cosθ)
   collapses and the beam gates to black. The cone now travels as its
   *versine* (1 − cosθ, computed in f64 by `cone_versine`) through
   `Bsdf::Laser`, `LightShape::Laser`, `LightData::half_w` and the
   prim's `ior` slot; every consumer uses versine-native forms: the
   solid angle is 2π·versine, the gate is |d − n|² ≤ 2·versine (unit
   vectors — Sterbenz-exact where 1 − dot cancels catastrophically,
   and it exactly absorbs the cone sampler's f32 cos-quantization),
   and the cone sampler draws the versine directly (sin² = 2v − v²,
   never from the rounded 1 − v).
3. **The gate-mapped bridge cone.** The plan's second miss: the rung
   sampled the full image-disc cone (angular radius w₀/D), but the
   aperture's cone gate accepts only a (Dθ/w₀)² fraction of those
   aims — at the demo's divergence that is ~1e-4, so the spot would
   converge ~10⁴× slower and fall below the accumulation layer's
   outlier-rejection threshold (p ≥ 0.1), never appearing. The
   unfolding is an isometry, so it maps the emission cone exactly onto
   a cone of half-angle θ around the reversed beam direction at the
   image; each rung now carries `beam_dir` + `versine` and samples the
   *binding* cone — the smaller of the disc cone and the gate-mapped
   cone, about its own axis (the disc cone's versine in the
   cancellation-free form x/(1+√(1−x)), x = (r/D)²). At the wide test
   angles the disc binds and the behavior is exactly the old one; at
   the physical divergences the gate binds and the spot's draws are
   zero-waste. `GpuLight` grew to 384 bytes (one rung = two vec4s).
4. **Demo constants.** Red: 650 nm, 50 mW, M² = 30 (θ ≈ 6.2e-4 rad);
   green: 532 nm, 5 mW, M² = 30 (θ ≈ 5.1e-4 rad) — green is ~6× more
   luminous per watt (ȳ(532) ≈ 0.86 vs ȳ(650) ≈ 0.107), so the spots
   read equally bright; both deposit E ≈ 160/16 W/m², above the floor
   under the ceiling panel (~10 W/m²). The plan's 5 mW red would have
   rendered ~6× dimmer than the lit floor — visible, but weak.
5. **Tests.** New: the power parametrization's θ-invariance; the
   versine surviving the diffraction limit (where the f32 cosine is
   exactly 1.0); the gate-mapped rung cone (samples inside the
   emission cone's image, pdf uniform over 2π·versine, the wide-angle
   control keeping the disc cone); the analytic spot value — the
   rendered axis radiance matches the closed form
   E(0) = Φ(1+cosθ)/(2πw₀²)·ρ/π (the plan's Φ/(πw₀²) is the θ→0
   limit; the exact form follows from E(0) = L·π·sin²(min(α₀,θ)) with
   sin²θ = (1−cosθ)(1+cosθ)), with Bernoulli-SE tolerances from the
   connection strategy's (dθ/w₀)² pass rate. The existing
   cone-gate/ladder/MIS tests keep their wide-θ scenes (the machinery
   is θ-agnostic; the binding-cone switch keeps their convergence).
6. **Bugs found and fixed during implementation:** the first draft of
   the rung's `pdf` used the disc versine r²/D² (2× the true
   1−cos(asin(r/D))) and sampled the smaller cone about the *disc*
   axis — the wide-angle regime regressed and the narrow regime
   rendered a half-width stripe; the binding cone must be sampled
   about its own axis. The emitter-cone gates in bdpt.rs first used
   `1 − cos ≤ versine` — at the tiny cones the cancellation leaves
   ~1 digit, killing ~7% of the light walk's reconstructed cone
   samples; the |d−n|² form is exact under the f32 quantization.
