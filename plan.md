# Plan: Generalized BSDFs, Caustics, and the Showcase Road

This file replaces the previous `plan.md`, which reviewed a simple
unidirectional tracer (`trace()` with an emissive early-return) that no
longer exists. The codebase is now a bidirectional path tracer with MIS,
Russian roulette, a wgpu port, and two frontends. This document plans the
remaining work from `todo.md`.

## 0. Where we stand

Done and verified in code (not just claimed in `todo.md`):

- BDPT with power-heuristic MIS (β = 2); every sampled edge has area pdf
  `G/π`, which reduces the weights to per-edge ratios.
- Russian roulette (`RR_MIN = 0.05`, `RR_START = 3`) with the survival
  probabilities entering the MIS weights as prefix products
  (`rr_prefix_products`, both backends).
- Finite-quad room (`add_wall` + `quad_frame`), correct wall dimensions,
  tight BVH boxes; the room-closure regression tests.
- GPU device reuse across resize and FHD export (`from_parts`); window
  frontend parity (screenshot, status overlay, help panel).
- Extended Reinhard (W = 10) in `output.rs` and `display.wgsl`.
- Camera pitch/strafe, walk-style `move_forward` (no y = 1.0 hack).
- CLI flags (`--terminal`/`--window`), key-kind guard, `#[ignore]` on the
  GPU oracle, panic dumps under `target/`.

Remaining, in the order I would do it:

- **Phase 0** — hygiene: clippy/fmt pass, CI workflow.
- **Phase 1** — BSDF generalization (`todo.md` Tier 2, "the big one").
- **Phase 2** — showcase features (`todo.md` Tier 3, ordered by value).
- Tests woven through all phases: white furnace, MIS sum-to-one with
  asymmetric pdfs, mirror/glass energy, caustic differential, variance-vs-spp.

---

## 1. Phase 0 — Hygiene (do first, small diffs, keeps the big change on green CI)

1. `cargo fmt` as its own commit (the tree is not rustfmt-clean; e.g. the
   big `match` in `main.rs`). Separate commit so later diffs stay reviewable.
2. [Done] `cargo clippy --all-targets -- -D warnings`, fix what it finds.
   Expect mostly mechanical fixes; keep the `#[allow(dead_code)]`
   scene-API items (`add_quad`, `add_plane`, `LightData::from_quad`) —
   they are deliberate API surface. (Fixed 2026-10-30: needless-range
   loops and negated comparison in `mis_weight`, `rr_prefix_products`
   restructured to return a struct instead of taking 8 arguments,
   `as_chunks` in `gpu.rs`, `Display` impl instead of inherent
   `to_string` in `output.rs`, `Key::Count`, test module moved to the
   end of `main.rs`, `identity_op`/`unnecessary_cast` in tests.)
3. Add `.github/workflows/ci.yml`: stable toolchain, then
   `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
   `cargo test`. Notes:
   - The WGSL validation test is pure naga — runs without a GPU.
   - `gpu_renderer_smoke` already early-returns when no adapter exists.
   - `gpu_matches_cpu` stays `#[ignore]` (needs a real GPU; run manually
     after every GPU-touching change instead).

**Acceptance:** CI green on a fresh checkout.

---

## 2. Phase 1 — BSDF generalization (the architectural core)

Goal: `enum Bsdf { Diffuse, Mirror, Glass { ior } }`, sampling that
returns the *actual* pdf, and the caustic demo that justifies BDPT.
No diffuse fast path — one general code path, per `todo.md`; it is ~30
lines more and removes a whole class of "works only for Lambertian"
reasoning.

### 2.1 Why the current math breaks (the trap to understand first)

Today every direction is cosine-sampled, so the area pdf of an edge is
`cos₀·cos₁/(π r²) = G/π` **regardless of which endpoint sampled it**.
That symmetry is what lets `mis_weight` work from a single `edges[]`
array of G values and closed-form per-edge ratios.

With a general BSDF the two directions of sampling differ:
`a = pdf_ω(sampling vertex) · cos(opposite vertex) / r²`. So each edge
needs *two* pdfs (eye-sampled and light-sampled), and the closed-form
ratio trick dies. The replacement is Veach's incremental-ratio walk.

### 2.2 Material model

```rust
// geometry.rs (and mirrored MaterialData in scene.rs)
pub enum Bsdf { Diffuse, Mirror, Glass { ior: f32 } }

pub struct Material {
    pub albedo: Vec3,
    pub emission: Vec3,   // != 0 => pure emitter, bsdf ignored (unchanged semantics)
    pub bsdf: Bsdf,
}

pub struct BsdfSample {
    pub dir_out: Vec3,
    pub pdf_omega: f32,   // 0 => absorption
    pub weight: Vec3,     // f · |cos| / pdf
}

impl Bsdf {
    /// Sample a scattering direction (delta BSDFs sample deterministically).
    fn sample(&self, normal: Vec3, dir_in: Vec3, rng: &mut impl Rng) -> Option<BsdfSample>;
    /// f and pdf for an arbitrary direction pair. Delta BSDFs return (0, 0)
    /// — which is exactly what invalidates connections through them.
    fn eval(&self, normal: Vec3, dir_in: Vec3, dir_out: Vec3) -> (Vec3, f32);
    fn is_delta(&self) -> bool;
}
```

- `eval` returns f and pdf separately: the connection *contribution*
  needs f, the MIS *walk* needs pdf.
- Diffuse must reduce exactly to today's constants: `f = albedo/π`,
  cosine sampling gives `weight = albedo`, edge pdf `G/π`. The existing
  hand-computed tests guard this reduction.
- Glass: exact dielectric Fresnel (cheap), Snell refraction, TIR falls
  back to reflection; `R + T = 1` must hold (white-furnace test checks it).
- GPU encoding: `mat_type: u32` becomes 0 diffuse / 1 emissive /
  2 mirror / 3 glass, plus `ior: f32`. **The material structs already
  have 16 bytes of trailing padding** (`_pad1/_pad2` etc.), so `ior`
  slots in without changing strides — the compile-time layout asserts
  in `gpu.rs` stay valid. Mirror the same fields in the WGSL structs.

### 2.3 Path vertices: store the actual pdfs

`PathVertex::edge_g` (with its implicit `/π`) becomes the **creation
pdf** `pdf_area: f32` — the area pdf of the edge that created the
vertex, *as sampled by the subpath that created it*:
`pdf_area = pdf_ω · cos / r²`. Plus the per-vertex sample `weight`
(f·cos/pdf) used to advance throughput and by the RR sweep.

Subpath generation changes:
- throughput advances by `weight` instead of `albedo`;
- the sampled direction comes from `Bsdf::sample` instead of
  `cosine_hemisphere`;
- the one-sided rule becomes BSDF-aware: Diffuse keeps the back-face
  break; Mirror/Glass flip the normal to the incident side and continue
  (glass needs `front_face` to pick η = 1/ior vs ior/1).

### 2.4 MIS weights: incremental ratios (replaces the closed form)

With vertices numbered 1..k along the path (camera→light) and edge `e_m`
connecting vertices m−1 and m: the eye subpath samples `e_m` at vertex
m−1, the light subpath at vertex m. Strategy s connects x_s to
y_{k−s−1}; its connection edge is `e_{s+1}`.

```
p[s]  = (∏_{m=2}^{s} aᵉₘ) · (1/nA) · (∏_{m=s+2}^{k} aˡₘ) · Qe(s) · Ql(k−s−1)
p[k]  = (∏_{m=2}^{k} aᵉₘ) · Qe(k)                    (eye subpath hits the light)
step: p[s+1]/p[s] = aᵉ(e_{s+1}) / aˡ(e_{s+2}) · Qe(s+1)/Qe(s) · Ql(k−s−2)/Ql(k−s−1)
```

where `aᵉ(e_{s+1})` is the area pdf of sampling, at vertex s, the
direction toward vertex s+1 — and `aˡ(e_{s+2})` the same at vertex s+2
toward vertex s+1. The downward step is the inverse ratio. The weight is

```
w[s₀] = 1 / Σ_{s=1..k} (p[s]/p[s₀])²        (power heuristic, β = 2)
```

computed by walking both directions from the actual strategy s₀ and
accumulating ratios — `p[s₀]` itself cancels, exactly like today's
r-max normalization. What each step needs:

- **Realized edges**: the stored `pdf_area` of the vertex the edge
  created (eye edges from eye vertices, light edges from light vertices).
- **The connection edge** (evaluated on the fly, from both sides):
  `Bsdf::eval` at x_{i+1} toward y_j and at y_j toward x_{i+1}.
- **Hypothetical extensions** (strategies that extend a subpath past
  where it stopped): `Bsdf::eval` in the known direction — the path
  geometry determines the directions, so the pdfs are computable even
  though the rolls never happened. Same principle the RR sweep already
  uses.

Boundary case: `p[k]/p[k−1] = aᵉ(e_k) · nA · Qe(k)/Qe(k−1)` with
`Ql(−1) := 1`.

**Delta rule:** a strategy whose connection endpoint has a delta BSDF
(or whose connection pdf is 0 from either side) is skipped — its
contribution is 0 anyway since `eval` returns (0, 0). Specular chains
are traversed only by subpath sampling; that is the whole BDPT story
for caustics.

The connection contribution becomes
`α(x_{i+1}) · f(x_{i+1}) · G · f(y_j) · α(y_j)` with f from `eval`
(y_0 keeps f = 1, unchanged).

### 2.5 Russian roulette prefix products

The q rule is unchanged; the sweep's throughput factors change from
`albedo` to the per-vertex sample `weight`. Realized steps read the
stored weight; hypothetical extensions evaluate f·cos/pdf on the fly.
For Diffuse this must reduce exactly to today's factors (`albedo`, and
π for the emitter's first step) — update
`rr_prefix_products_match_hand_computation` accordingly and keep it as
the guard.

### 2.6 GPU port

Mirror every change in `shaders/bdpt.wgsl` 1:1 (the file header says so
and the offline naga test enforces parse/validate):

- `PathVertex` gains `pdf_area` + `weight` fields (function-local
  array, no layout constraints);
- `mis_weight` becomes the ratio walk (keep the f32 math and the
  r-max-style normalization for range safety);
- `bsdf_sample`/`bsdf_eval` WGSL functions with explicit `if` chains on
  `mat_type` (no dynamic dispatch in WGSL — same pattern as today);
- `ior` in the material structs (fits existing padding, see 2.2).

Gate: `wgsl_shader_validates` in CI, then the ignored `gpu_matches_cpu`
oracle manually on a real GPU.

### 2.7 Demo scene: the caustic

Add a glass sphere (ior 1.5) between the hanging light and the floor
area visible from the default camera, plus optionally a small mirror
sphere. Tune positions so the focused caustic is in frame — this is
the `paper.md` §6.1 payoff: light paths the unidirectional tracer
cannot find.

### 2.8 Tests (white furnace lands FIRST, as a baseline)

1. **`white_furnace`** — add before touching anything: closed room via
   `add_wall`, every surface albedo 1, one pure emitter anywhere. A ray
   hitting a wall must return mean = Le exactly: throughput never decays
   and every path eventually ends on the emitter (RR is unbiased, so
   the *mean* is exact; only variance grows). ~4096 samples, fixed
   seed, ~3% tolerance. Catches energy loss, dropped MIS strategies,
   RR bias — the guardian for the entire phase.
2. **MIS sum-to-one with asymmetric pdfs** — generalize
   `mis_weights_sum_to_one` to non-`G/π` pdfs.
3. **Mirror determinism** — a ray at a mirror wall that reflects into a
   wall region must match the mean of a direct ray at that region.
4. **Glass energy** — white furnace with a glass sphere inside: mean
   still Le (Fresnel + TIR must conserve R + T = 1).
5. **Caustic differential** — the flagship: estimate the caustic pixel
   with `bdpt_trace` and with the existing `reference_pt` oracle at
   equal sample counts; BDPT variance must be dramatically lower
   (tolerances tuned on first run against a high-N BDPT reference).
6. All existing tests keep passing (several need updating to the new
   constructors/factors — that is expected, not regression).

**Acceptance:** all tests green; caustic visible in `screenshot.png`;
GPU oracle within tolerance; README "Light Transport" section rewritten
(the `G/π` claim is gone) and `paper.md` §6.1 references the realized
demo.

---

## 3. Phase 2 — Showcase features (ordered by value/effort)

1. **Sobol / low-discrepancy sampling** — biggest noise cut per spp,
   works on both backends. A tiny Owen-scrambled Sobol (~50 lines, no
   new dependency) with a deterministic dimension counter per bounce;
   the GPU uses the same matrices and dimension scheme in WGSL. Exact
   stream matching between backends is *not* required (the oracle
   compares means). Acceptance: unbiasedness (mean unchanged) + the
   variance-vs-spp test below shows the improvement.
2. **Depth of field (thin lens)** — small and self-contained:
   `camera.rs` + `lens_radius`/`focal_distance` uniforms + WGSL ray
   gen (sample lens disc, aim at the focal point on the principal
   ray). Acceptance: radius 0 reproduces today's output bit-for-bit
   (same seed); a focused sphere stays sharp while the background
   blurs.
3. **Joint bilateral denoiser with G-buffers** — fixes the known
   `sigma_r = 0.15`-in-linear-HDR problem (near the light, values ~10,
   the filter stops working). Compute pass writes first-hit
   normal/albedo buffers; the display shader and the CPU filter weight
   neighbors by normal/albedo agreement too, which lets σᵣ scale with
   local luminance without smearing silhouettes. Acceptance: a
   bright-region smoothing test that the current filter fails.
4. **JSON scene format** — serde + loader; `build_scene()` becomes
   `scenes/demo.json` (embedded via `include_str!` for tests). This is
   the prerequisite that justifies bigger scenes — and only then the
   WGSL BVH (explicitly deferred until it exists).
5. *(stretch)* **Environment map** — an infinite-area light vertex
   type in BDPT (directional pdf, no G/π structure; the "ray escapes"
   strategy connects to the env instead of returning `BACKGROUND`).
   Sketch only unless time permits.

---

## 4. Test plan summary

| Test | Lands | Catches |
| :--- | :--- | :--- |
| white furnace | before Phase 1 | energy loss, dropped strategies, RR bias |
| MIS sum-to-one (asymmetric) | Phase 1 | weight normalization bugs |
| mirror determinism | Phase 1 | reflection direction/side bugs |
| glass energy (furnace + glass) | Phase 1 | Fresnel/TIR energy leaks |
| caustic differential (BDPT vs PT) | Phase 1 | the reason BDPT exists |
| variance-decreases-with-spp | Phase 2.1 | estimator efficiency, LDS port |
| GPU oracle (`#[ignore]`) | after each GPU change | porting errors |

Variance test spec: 32 independent estimates at 64 spp vs 32 at
1024 spp on a fixed ray; the sample variance of the means must drop by
roughly the sample-count ratio (assert > 4×).

---

## 5. Risks and pitfalls

- **The G/π symmetry trap** (2.1) is the main one: forgetting that the
  connection edge needs *two* pdfs is the classic BDPT bug and shows up
  as subtle bias only in mixed-specular scenes. The asymmetric
  sum-to-one test is the guard.
- **f32 MIS products in WGSL**: the ratio walk is naturally stable
  (products of ratios); keep the normalize-by-max trick.
- **Buffer layout drift**: `ior` fits existing padding, but the
  compile-time asserts in `gpu.rs` must stay in sync with the WGSL
  structs — they are the enforcement mechanism, keep them.
- **RR sweep complexity**: the hypothetical-extension evaluations are
  the fiddliest part; the updated hand-computed test must pin them.
- **Emissive + reflective materials** stay a non-goal: the light
  subpath assumes pure emitters (`on_light` vertices do not scatter).
  Supporting both is possible later but touches the light-subpath
  rules and MIS for little demo value.
- **Shadow-acne epsilon** (`EPS_NEAR = 0.001`, absolute): fine at the
  current ~unit scene scale; revisit only if scene scale changes.

---

## 6. Suggested order of work

1. Phase 0 (fmt → clippy → CI), each its own commit.
2. White furnace test on the *current* code (baseline green).
3. Phase 1 in the sub-step order of §2 — material model, sampling,
   vertices, MIS walk, RR sweep, CPU tests, then the WGSL port, then
   the demo scene and the caustic differential.
4. Phase 2 items one per PR-sized change, each with its test.
5. Mark items `[Done]` in `todo.md` as they land; update README and
   `paper.md` alongside Phase 1 (not after).