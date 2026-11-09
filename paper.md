# Rendering Laser Beams Through Specular Chains: A Bidirectional Path Tracer with Analytic Bridge Sampling

**A technical paper on the render technique of `pathtracer-rs`**

*November 2026*

---

## Abstract

This paper documents the rendering technique of a physically based, dual-backend (CPU/wgpu) path tracer whose showcase scene contains laser beams that bounce through chains of perfect mirrors before landing on diffuse surfaces. The core estimator is bidirectional path tracing (BDPT) with multiple-importance sampling (MIS) over all vertex-pair connection strategies, using Veach's incremental-ratio walk over the *actual* per-edge pdfs, with Russian-roulette survival probabilities folded into the pdf products and delta BSDFs handled by a symbolic-pdf convention. The central problem — and the paper's main contribution — is a structural blind spot of every BDPT strategy for light transported through a *delta specular chain* from a *tiny emitter*: no connection can cross a mirror (eval = 0), no light subpath can find a ~1 cm aperture, and the eye subpath hits it with probability ~10⁻⁵. The fix is the **specular bridge ladder**: the laser aperture's mirror image is unfolded analytically across every *prefix* of the beam's mirror chain at scene-build time, and each image disc is registered as a *sampling technique* — a mixture component of the eye walk's diffuse BSDF sampling. The construction is exact (the unfolding preserves path length), derived on demand (a moving scene re-derives every rung), and unbiased (the mixture pdf enters the MIS machinery on both realized and hypothetical edges). A two-tier firefly rejection scheme — a loose, physically motivated per-sample clamp plus a mean-aware temporal outlier rejection over an exponential second-moment window — removes numerical outliers without destroying rare-event caustic estimators. The paper closes with an analysis of *changing the underlying physical model*: wave-based transport (diffraction, interference, speckle, polarization), spectral transport with dispersive prisms, participating media, and glossy microfacet BSDFs — how each interacts with the existing MIS, bridge, and firefly machinery, and what it costs.

---

## 1. Introduction

The rendering equation integrates radiance over all light-transport paths connecting emitters to the camera. Monte-Carlo path tracing estimates this integral by sampling paths; the difficulty is that interesting paths — caustics, mirror chains, tiny bright emitters — occupy vanishingly small measure under naive sampling.

This codebase renders a Cornell-box-style room containing two lasers: a red laser aimed at a back-wall mirror that bounces the beam onto a partition wall, and a green laser whose beam ping-pongs between two thin mirror tiles. Each beam is a *narrow cone* (half-angle 0.01 rad) emitted from a *tiny aperture* (radius 1 cm) and transported through a *chain of perfect mirrors* before depositing a spot on a diffuse surface. The renderer must show:

1. the spot where the beam lands,
2. the *glow beside every leg of the beam* — the light a diffuse surface point near the beam receives through a *prefix* of the mirror chain,
3. ordinary global illumination (color bleeding, soft shadows, glass caustics),

…all with an unbiased estimator, on both a CPU backend (rayon + trait-object BVH) and a GPU backend (wgpu compute + flat BVH), at interactive rates with progressive refinement.

The paper is organized as follows. §2 overviews the system. §3 extracts the light-transport method (BDPT + MIS + roulette + BSDFs). §4 presents the specular bridge ladder. §5 presents firefly rejection. §6–§8 cover the display pipeline, acceleration, and validation. §9 analyzes changes to the underlying physical model — wave optics, spectral dispersion, media, polarization — and §10 concludes.

---

## 2. System Overview

```
SceneData (flat, GPU-friendly scene description, src/scene.rs)
   │
   ├── CPU backend:  Box<dyn Intersectable> BVH tree (src/bvh.rs)
   │                 rayon-parallel BDPT (src/bdpt.rs, src/camera.rs)
   │                 → reference oracle, fully unit-tested
   │
   └── GPU backend:  kind-tagged prim array + flat pre-order BVH (src/gpu.rs)
                     WGSL compute, one thread/pixel (shaders/bdpt/, concatenated
                     by BDPT_WGSL in src/gpu.rs)
                     PCG32 RNG, persistent rgba32float accumulation buffer
                     (the .w channel carries the outlier-rejection statistic)

Both backends consume the same SceneData → identical geometry, identical
estimator semantics (the WGSL is a 1:1 port of bdpt.rs).

Pipeline:
  ray generation → BVH intersection → BDPT (eye+light subpaths, all
  connections, MIS) → per-sample firefly clamp → temporal accumulation
  with outlier rejection → bilateral filter (relative σ_r) → extended
  Reinhard tone map → gamma 2.2 → terminal ANSI half-blocks / wgpu surface
```

Key architectural decisions:

- **One source of truth for the scene.** `SceneData` is a flat, data-oriented description (spheres, quads, cubes, planes, cylinders, lights). The CPU backend builds a trait-object BVH from it; the GPU backend uploads it as two storage buffers (prims + flat BVH nodes). Emitter geometry is registered *both* as intersectable geometry and as a sampleable light — the GPU's `intersect_scene` never reads the light buffer, so a light missing from the geometry arrays would be invisible to eye subpaths.
- **No GPU readback while stationary.** The display pass reads the accumulation buffer directly in a fragment shader; pixels round-trip through the CPU only for PNG export or the terminal frontend.
- **Derived, never baked.** The laser bridge ladders (§4) are derived on demand from the fully-built scene, so registration order is irrelevant and a scene that moves between frames simply re-derives every rung.

---

## 3. The Light Transport Method

### 3.1 Path space and strategies

A full path is $x_1 \dots x_k$ — camera at $x_0$, emitter at $y_0 = x_k$. The eye subpath samples $x_1, x_2, \dots$; the light subpath samples $y_0, y_1, \dots$. Every partition of the path into an eye prefix and light suffix is a *strategy*: strategy $s$ uses $s$ eye vertices and connects $x_s$ to $x_{s+1} = y_{k-s-1}$; strategy $k$ is the eye subpath hitting the emitter directly. The estimator sums, over all strategies that can sample the path, the contribution weighted by the MIS weight $w_s$.

The implementation (`bdpt_trace` in `src/bdpt.rs`, ported 1:1 to the WGSL fragments in `shaders/bdpt/`):

1. **Eye walk.** Trace the camera ray; at each hit, orient the normal toward the arrival side; stop at emitters; sample the BSDF (diffuse vertices use the cosine–bridge mixture of §4.4); roll Russian roulette on the post-scatter throughput.
2. **Light walk.** Pick a light uniformly, sample a surface point (pdf $1/(n \cdot A)$), sample a departure direction (cosine-weighted, or uniform-in-cone for a laser aperture), then walk as above transporting *importance*.
3. **Direct-hit strategy.** If the eye subpath's last vertex is on a sampleable emitter (and, for a laser, inside the beam cone), credit `throughput · emission · w_k`.
4. **Connection strategies.** For every eye vertex $x_{i+1}$ and light vertex $y_j$ with $k = i+j+2 \le$ depth cap: skip delta endpoints, evaluate $f \cdot G \cdot f$ across the connection edge, shadow-test, credit `α_eye · f · G · f · α_light · w_{i+1}`.

### 3.2 MIS weights: the incremental-ratio walk

The power heuristic ($\beta = 2$) gives

$$w_s = \frac{1}{\sum_{s'} \left( \frac{p_{s'}}{p_s} \right)^2}$$

over the *valid* strategies. Rather than computing each strategy's pdf product from scratch, the implementation walks the ratio between adjacent strategies:

$$\frac{p_{s'+1}}{p_{s'}} = \frac{a^e(e_{s'+1})}{a^l(e_{s'+2})} \cdot \frac{Q_e(s'+1)}{Q_e(s')} \cdot \frac{Q_l(k{-}s'{-}2)}{Q_l(k{-}s'{-}1)}$$

where $a^e, a^l$ are area-measure pdfs of edges as sampled from the eye/light side, and $Q_e, Q_l$ are the Russian-roulette survival-probability prefix products (§3.4). The walk accumulates products up and down from the realized strategy $s$; $p_s$ itself cancels. Two numerical guards: the ratios are normalized by the largest valid ratio before squaring (keeps f32 squares in range), and invalid strategies are excluded from the normalization so the valid weights still sum to one (unit-tested).

**The `PathView` discipline.** Every pdf and sample weight — realized edges, the connection edge, *and the hypothetical extensions of strategies that would push a subpath past where it stopped* — is computed from the path geometry through one view (`PathView`), which addresses the unified positions $x_1..x_k$ (eye prefix, then light vertices reversed). Realized and hypothetical quantities are produced by the *same code* and cannot drift apart. This is the load-bearing design decision of the whole MIS implementation: a strategy that would have extended the eye subpath one vertex further needs the pdf of an event that never happened, and that pdf must be exactly the one the walk *would* have rolled.

### 3.3 Delta BSDFs

Mirror and glass scatter into a single direction: their solid-angle pdf is a Dirac delta with no numeric value. The convention:

- A delta-sampled edge contributes a **symbolic factor of 1** to the pdf products — every valid strategy's pdf carries exactly one such factor per delta vertex, so they cancel in the ratios.
- A delta vertex scattering in any *other* direction has pdf 0, killing every strategy that would need it.
- Connections *through* delta vertices are invalid (their `eval` is 0) and are excluded from the MIS normalization.

Specular transport is therefore handled purely by subpath sampling — which is the whole BDPT story for caustics, and also the origin of the blind spot of §4.

### 3.4 Russian roulette inside MIS

Paths are terminated by roulette, not a short fixed depth: from extension 3 on, a subpath survives with $q = \mathrm{clamp}(\max \beta, 0.05, 1)$ and divides its throughput by $q$ on survival (unbiased). The subtlety: **the survival probabilities are part of the sampling pdfs**, so they must enter the MIS weights. `rr_prefix_products` computes $Q_e(m)$ and $Q_l(j)$ by sweeping the path and *recomputing* each extension's post-scatter throughput from the path geometry — so the q's are exactly the ones the rolls used, including the hypothetical rolls of strategies that would extend a subpath past where it stopped (killed, escaped, or depth-capped). The differential test `rr_does_not_bias_the_mean` pins unbiasedness against a roulette-free reference path tracer.

### 3.5 BSDF and emitter models

| Model | Sampling | Notes |
|---|---|---|
| Lambertian diffuse | cosine-weighted ($p_\omega = \cos\theta/\pi$) | weight $f\cos/\text{pdf} = $ albedo exactly |
| Perfect mirror | deterministic reflection | delta; symbolic pdf 1 |
| Dielectric glass | Fresnel-split reflect/refract | exact unpolarized Fresnel $(r_s^2+r_p^2)/2$, Snell, TIR fallback; radiance picks up $(\eta_t/\eta_i)^2$ crossing a boundary, importance does not (basic-radiance invariance) |
| Laser aperture | uniform-in-cone emission | *not* delta: the cone has real (tiny) solid angle $\Omega = 2\pi(1-\cos\theta)$, so the aperture remains a valid connection endpoint; one-sided (dark from behind); the cone gate is a single dot product |

The laser's first light-subpath edge factor is $\cos\theta_0 \cdot 2\pi(1-\cos\theta)$ inside the beam (0 outside), replacing the ordinary emitter's $\pi$; the emitter event's pdf in the MIS machinery reads the cone from the vertex material.

---

## 4. The Specular Bridge Ladder

### 4.1 The blind spot

Consider the transport path *aperture → mirror₁ → … → mirrorⱼ → diffuse point beside the beam's (j+1)-th leg → camera*. Check every BDPT strategy:

- **Connections from the diffuse eye vertex** to a light vertex on the mirror: the mirror is a delta BSDF, `eval = 0` — invalid. To the aperture: the segment points away from the emission cone — gated. To a nearby light vertex: the segment lies in the surface plane — grazing, $G = 0$ exactly.
- **Light subpath threading the chain**: the light subpath follows the beam well (the cone is narrow, mirror bounces are deterministic), but that is exactly the problem — its vertices on the mirrors are delta (invalid connection endpoints), and a mirror cannot scatter *toward* $p$ (it only reflects deterministically, continuing down the chain). The light subpath can therefore never place a connectable vertex that carries the beam's light to $p$; it either dies on the mirrors' delta vertices or lands at the spot, where a connection to $p$ transports the *spot's* reflected light — a different path.
- **Eye subpath threading the chain**: the mirror bounces are deterministic, so reaching the aperture requires the *initial diffuse sample* at $p$ to land in the tiny solid angle whose mirror chain terminates on the 1 cm aperture disc — a ~10⁻⁵ measure. Effectively never sampled — and the rare hit is then winsorized away by temporal outlier rejection anyway.

This is the classic reason photon mapping, VCM, and manifold exploration exist: a pure BDPT with delta mirrors and a tiny emitter is *structurally blind* to the beam's light. Before the fix, only the first leg (direct light from the aperture) and the last leg (the whole-chain image) glowed; the middle legs rendered black.

### 4.2 The unfolding construction

The analytic description of the specular path *diffuse point $p$ → mirror₁ → … → mirrorⱼ → aperture* is the aperture's **mirror image** through the mirror chain's prefix: reflect the aperture across mirrorⱼ's plane, that image across mirrorⱼ₋₁'s plane, and so on. Each mirror plane is the perpendicular bisector of the segment image ↔ unfolded image, so a ray from $p$ aimed at a point $q$ of the image disc crosses every mirror plane in the prefix, in reverse order, and reflects onto the corresponding point of the aperture disc — the classic unfolding construction. The set of directions from $p$ that reach the aperture through the prefix is exactly the cone subtended by the image disc:

$$\theta_c = \arcsin\frac{r}{\lVert c - p \rVert}, \qquad \Omega = 2\pi(1 - \cos\theta_c)$$

The construction is exact and preserves path length (the unit tests assert $\lVert \text{image} - \text{spot} \rVert = $ total beam path length) — a property that becomes important for the wave-optics extension (§9.1), since path length is exactly the phase ingredient.

### 4.3 The ladder

One rung per mirror the beam crosses, in beam order: rung $j$ is the aperture unfolded across exactly the first $j$ mirrors. The rungs are free byproducts of the beam trace — the loop that follows the aperture's axis ray (`SceneData::bridges_of_light`) reflects the image across each mirror plane as it goes, and every intermediate image is a valid bridge for the corresponding prefix. A diffuse vertex beside leg $j{+}1$ sees the aperture through exactly the first $j$ mirrors, so **rung $j$ is what carries the glow on leg $j{+}1$**.

Boundary rules: glass ends the ladder (a refracted ray's bend depends on where it crosses the interface, so no rigid image describes the far side); the rungs before the glass remain valid. A beam that never hits a mirror has an empty ladder (the aperture is directly visible; ordinary sampling finds it). The ladder is capped at 8 bounces (matching the path vertex budget) so a mirror loop cannot spin forever.

### 4.4 Mixture sampling and MIS consistency

At a diffuse eye vertex, the walk samples from a **mixture**: with probability $1 - p_B$ a cosine-weighted direction, with probability $p_B = 0.25$ a direction uniform over the cone of one qualifying rung (picked uniformly among rungs whose image axis lies in the vertex's hemisphere and whose cone is non-degenerate). The realized direction's pdf is the *full mixture density*

$$p(\omega) = (1 - p_B)\frac{\cos\theta_o}{\pi} + p_B \cdot \frac{1}{n_\text{qual}} \sum_i \frac{\mathbb{1}[\omega \in \Omega_i]}{\Omega_i}$$

and the sample weight is $f \cos / p$ under that density — unbiased for every direction the mixture can propose. A bridge direction can dip below the horizon (a grazing rung's cone is wider than the axis elevation); the BSDF is zero there, so the weight is 0, not $\pm\infty$ — the draw is merely wasted.

The asymmetry is deliberate and *must* be reflected in the MIS machinery: the eye walk samples diffuse vertices with the cosine–bridge mixture; the light walk samples plain cosine. `event_pdf`/`event_weight` therefore take a `Walk` tag and query each walk's own distribution — realized *and* hypothetical edges alike. Get this wrong and the MIS weights are silently biased.

### 4.5 The winsorization constraint on $p_B$

The mixture probability is not free. The accumulation layer (§5.2) winsorizes each batch against $\mu + K\sigma$ with $K = 3$. A rare-event signal (the beam's glow: a pixel is bright only when a bridge direction was drawn) has its bright mode at roughly $\mu + \sqrt{(1-p)/p}\,\sigma$; the mode survives the winsorization iff

$$\sqrt{\frac{1-p}{p}} \le K \iff p \ge \frac{1}{1+K^2} = 0.1$$

$p_B = 0.25$ sits comfortably inside (mode at $\mu + 1.7\sigma$), keeping the beam's glow unbiased. The cost: non-footprint vertices waste 25% of their diffuse draws on bridge directions the cone gate kills — a ~33% ambient-variance increase, cleaned up by accumulation.

### 4.6 Movement and re-derivation

The ladder is *derived* per query from the fully-built scene, never stored: the beam is traced through the real geometry, so the laser may be registered before, after, or interleaved with the mirrors. The window frontend's sweep animation (`M`) pivots the laser on its table mount, rebuilds the scene per frame, and the renderer re-derives every rung — the spot, the glow beside every leg, and the ladder all follow the movement live. The regression tests pin this end to end: a moved mirror must move the whole-chain rung, and each re-derived rung must be the correct unfolding for the new geometry.

### 4.7 What the ladder is *not*

The ladder is a **sampling technique, not a light source**: no fake emitters, no decals, no mid-air beam. In clean air a laser beam is invisible from the side; only the light deposited on real surfaces is real. (The earlier "decal" design — registering the landing patch as a small quad emitter — was replaced by the bridge precisely because the bridge keeps the transport inside the unbiased estimator: the direct-hit strategy credits the aperture's real emission, cone-gated, when a bridged eye path lands on it.)

---

## 5. Firefly Rejection

Caustic and mirror paths are sampled by rare-event strategies that carry huge throughput when they succeed. One lucky sample with radiance ~50 in a pixel whose true mean is ~0.05 enters the running average at full weight and dilutes only as $1/n$ — a persistent white dot (everything at or above the tone-map white point displays as pure white, so the dot is saturated from the first frame it lands). Two complementary controls remove it.

### 5.1 Tier 1: the loose per-sample clamp

The per-sample clamp bound is a multiple (8×) of the **brightest emitter's luminance** — a deliberately loose, *physically motivated* bound, not a statistical one. The reasoning:

- Per-sample, a rare legitimate caustic sample is indistinguishable from a firefly: both are huge relative to the pixel's mean, because the caustic's estimator is rare-event (in the reference scene ~8% of samples carry ~$L_e$ each, so the sample magnitude is ~$L_e$, not ~the pixel mean). A *tight* clamp (e.g., at the tone-map white point) destroys caustics: it caps the rare-event samples below the pixel's true mean and the caustic never appears (measured: clamping at 2W collapsed the reference caustic from ~85 to ~1.6 — *darker than the lit floor around it*).
- What the loose bound *does* catch: a sample far above the brightest emitter is physically implausible (BSDF sample weights ≤ 1, MIS weights ≤ 1, the roulette's $1/q$ compensation exactly restores decayed throughput), so it can only be a numerical outlier. The 8× headroom covers the $\eta^2$ transmission factor and caustic concentration (measured: the reference caustic's samples peak at ~1.1× $L_e$).
- The clamp preserves chroma (scales all channels by bound/luminance) and is disabled when the scene has no lights (no physical bound to appeal to).

### 5.2 Tier 2: mean-aware temporal outlier rejection

The accumulation layer winsorizes each *batch* against the pixel's own running statistics: a batch whose mean luminance exceeds $\mu + K\sigma$ ($K = 3$) is scaled down to that bound before entering the average. The statistic is $\sigma = \sqrt{E[\text{lum}^2] - \mu^2}$ where $E[\text{lum}^2]$ is an **exponential moving average** over ~16 samples (on the GPU it lives in the accumulation buffer's otherwise-unused `.w` channel).

The EMA — not a full running average — is the subtle load-bearing choice. Capping a *running* second moment at the rejection bound would deadlock a caustic pixel whose first samples are dark: $\sigma = 0$ pins the bound at $\mu$, every bright batch is clamped to $\mu$, and the moment never grows. The EMA tracks the **raw** (pre-winsorization) luminance, so:

- a **genuine caustic** — many bright batches — keeps $\sigma$ loose and converges to its true value;
- a **single firefly**'s σ-poisoning decays within ~MOMENT_WINDOW samples and leaves no trace in the mean.

That asymmetry — temporal persistence — is exactly the statistic that tells a caustic from a firefly, and it is the *only* distinguisher available per-sample-agnostic. The EMA rate is per-sample, not per-batch: a batch of $m$ samples moves the statistic by $1 - (1 - 1/W)^m$, so a large batch (the 32-spp export) can never *replace* the statistic — the per-batch form $m/W$ clamps to 1 there, and a batch whose spot pixel drew zero bridge directions would collapse $\sigma$ to ~0 and erase the laser's spot from the accumulation.

### 5.3 Why two tiers

The per-sample clamp bounds the *magnitude* (removes the implausible tail); the temporal rejection stops the plausible tail from *sticking*. Neither alone suffices: the clamp alone leaves persistent bright dots from legitimate-magnitude samples in dark pixels; the temporal rejection alone chases its own statistic without a physical anchor. Both backends apply both tiers identically (the GPU receives the clamp bound as a uniform) so the CPU oracle compares like against like.

---

## 6. Accumulation, Filtering, Tone Mapping, Display

- **Progressive accumulation.** Up to 128 frames average while the camera is stationary (sample-count-weighted, so 1-spp preview and 32-spp export batches mix correctly); any camera move or scene change resets. The FHD export (1920×1080, 128 spp, depth 8) runs in background batches of 32 spp per frame so the window keeps presenting.
- **Bilateral filter** (radius 1, $\sigma_r = 0.15$): edge-preserving denoise on the *linear HDR* buffer, before tone mapping. The range sigma is **relative** — the color distance is scaled by the local mean — so one sigma covers both dark corners (values ~0.05) and the floor under a bright emitter (values ~10). An absolute sigma in linear HDR makes the filter a no-op exactly where fireflies live.
- **Extended Reinhard** with white point $W = 10$: $c(1 + c/W^2)/(1 + c)$ maps the scene white point to display white, so highlights roll off to pure white instead of plateauing below it. Then gamma 2.2 (or hardware sRGB on the window surface).
- **Terminal frontend**: ANSI 24-bit color with the half-block glyph `▀` (two pixel rows per character cell, doubling vertical resolution).

---

## 7. Acceleration and Backends

- **BVH, median split along the widest axis.** The room walls are finite quads, not infinite planes — planes get a ±10⁶ bounding box, which makes the root box span the world and culls nothing; quads have tight boxes, so the BVH actually culls. The CPU tree adds a subtree early-exit test (skip the right subtree when the left hit is closer than the right box's minimum possible $t$).
- **GPU flat BVH**: built on the CPU, uploaded as two storage buffers — a kind-tagged prim array (sphere/quad/cube/plane/cylinder share one 112-byte layout) and 32-byte pre-order nodes (the pbrt `LinearBVHNode` trick: the left child is always node index + 1, so only the right child is stored). Traversal uses an explicit 32-entry stack (WGSL has no recursion); the builder caps depth at 24, and a pop-based traversal of a depth-$d$ tree holds at most $d+1$ pending nodes, so the stack cannot overflow. Every box is culled by the current best-hit distance.
- **RNG**: PCG32, seeded per pixel per frame — stateless-seedable, standard for GPU path tracing.
- **Numerical parity discipline**: the WGSL port replicates Rust's NaN-ignoring `max`, uses explicit branches where `select()` would read past realized subpaths or underflow unsigned indices, and compile-time layout assertions pin the Rust structs to the WGSL structs byte-for-byte.

---

## 8. Validation

The test suite is the paper's evidence. Highlights:

- **MIS weights sum to one** over valid strategies, including with a delta vertex on the path (asymmetric pdfs; hand-computed values).
- **White furnace**: a closed albedo-1 room with the emitter as the entire ceiling must return $L_e$ exactly — the guardian for BSDF energy conservation, dropped MIS strategies, and roulette bias (the flat-room geometry keeps the depth-8 truncation tail < 0.5%).
- **Glass furnace with a no-glass control**: Fresnel + TIR must conserve $R + T = 1$; the control isolates glass-specific loss from room truncation.
- **Caustic differential**: a glass sphere focusing light onto the floor — BDPT's variance must be dramatically lower than a unidirectional reference tracer's at equal sample counts, with means agreeing.
- **Roulette unbiasedness differential** against a roulette-free reference.
- **Laser cone gating**: full emission on the beam axis and off-center aperture hits; darkness outside the cone and on the housing's dark side.
- **Bridge ladder**: one rung per mirror in beam order; each rung the exact unfolding for its prefix (path-length preservation asserted); the ladder follows a moved mirror; a beam without mirrors has no ladder.
- **Outlier rejection**: a firefly batch cannot move the mean; a sequence of bright batches (a genuine caustic) converges; the dark-start deadlock case converges.

---

## 9. Changing the Model

The renderer's physical model is **geometric optics**: light is rays carrying radiance; surfaces are local BSDFs; transport is straight lines between events. This section analyzes what changes — and what survives — under four model changes, ordered from "drop-in" to "research".

### 9.1 Wave-based transport

Geometric optics is the $L \gg \lambda$ limit of the electromagnetic wave. A wave model adds four phenomena the current renderer cannot express:

1. **Diffraction** — beam spreading beyond the geometric cone. A Gaussian beam with waist $w_0$ diverges at half-angle $\theta_d = \lambda / (\pi w_0)$. For the demo laser ($w_0 = 1$ cm, $\lambda = 532$ nm): $\theta_d \approx 1.7 \times 10^{-5}$ rad — the geometric cone (0.01 rad) is ~600× wider, so the demo sits safely in the GO regime. The wave model matters when the aperture shrinks or the path length grows: a diffraction-limited 1 mm beam over 10 m spreads to ~3 mm — visible in the spot's size and ring structure (Airy fringes at the mirror edges).
2. **Interference** — coherent superposition. Lasers *are* coherent; the spot on a rough wall is not a smooth disc but **speckle**. Two bridge rungs illuminating the same surface point (a point that sees the aperture through two different mirror prefixes) interfere; the fringe period is set by the path-length difference, which the unfolding construction already computes exactly.
3. **Speckle statistics** — fully developed speckle has *negative-exponential* intensity statistics, $P(I) = e^{-I/\langle I\rangle}/\langle I\rangle$, contrast $\sigma/\langle I\rangle = 1$. This is a physically real heavy-tailed distribution — precisely the regime the firefly machinery was built to protect. The two-tier design handles it correctly by construction: static speckle is temporally persistent, so the EMA σ grows and the grains converge to their true (grainy) mean rather than being clamped away. A tight per-sample clamp would erase speckle exactly as it erased the caustic.
4. **Polarization** — lasers are typically linearly polarized; the current Fresnel is the unpolarized average $(r_s^2 + r_p^2)/2$, exact for unpolarized light, an approximation for polarized. See §9.4.

**Why not full-wave simulation?** FDTD needs ~$\lambda/10$ cells: 53 nm for green light. A 5 m room is $10^8$ cells per dimension — $10^{24}$ total. Full-wave methods are viable only at micron scale (metasurfaces, photonic crystals). The scene-scale resolution is a **hybrid**:

- Keep BDPT for the incoherent ambient light (the ceiling panel is a spatially incoherent thermal source; radiance transport is correct for it).
- Add a **coherent channel** for lasers: propagate the beam as a Gaussian beam through the mirror chain using ABCD ray-transfer matrices (free space and mirror reflection are both ABCD operations; the unfolding ladder *is* the geometric shadow of this formalism — each rung's image disc becomes the beam's virtual waist). Deposit the complex field $u$ on the final surface; the physical spot radiance is $|u|^2$ times the surface's scatter operator.
- **Speckle rendering**: the surface's scattered field is a random phasor sum determined by the surface's roughness power spectral density (Harvey–Shack scatter theory connects the micro-roughness PSD to the BRDF). Render the spot as $|u|^2$ modulated by a speckle pattern whose statistics match the surface PSD — or stochastically, by sampling random phases per sub-ray and summing complex amplitudes.
- **Phase from the ladder**: the rungs already preserve path length; phase $= 2\pi \cdot (\text{path length})/\lambda$ is one multiplication away. Multi-rung interference fringes follow.

**Cost**: the coherent channel is a bounded addition (one Gaussian beam per laser, analytic through the chain — no Monte Carlo at all for the propagation); the expensive part is the speckle texture and its temporal behavior. **Interaction with existing machinery**: the bridge ladder survives unchanged as the sampling technique for the geometric part; the firefly machinery survives by design (§5 was built around heavy-tailed-but-real signals); the MIS machinery is untouched (the coherent channel bypasses it, being deterministic).

### 9.2 Spectral transport and prisms (dispersion)

The current model transports RGB triplets with a single refractive index — glass is achromatic. Real dielectrics disperse: $n(\lambda)$ via Sellmeier coefficients, so a **prism** fans white light into a spectrum and the glass cube casts dispersive caustic fringes.

**The change**: transport radiance as a function of wavelength $\lambda$ instead of RGB.

- **Hero wavelength sampling**: sample one $\lambda$ per path (importance-sampled by the emitter spectrum), carry it through, convert to RGB at the sensor via CIE color matching functions. For smooth emitters this adds ~2–4× variance; for **lasers it adds none** — a laser is monochromatic, so $\lambda$ is deterministic (the current RGB trick of putting $4 \times 10^8$ in one channel is replaced by the physically correct spectral delta, and the out-of-gamut problem — a pure 532 nm green is outside sRGB — moves to the display stage where it belongs).
- **MIS is untouched** — the standard factorization result: $\lambda$ is sampled once per path, independent of all direction sampling, so every strategy pdf factorizes as $p(\text{directions}) \cdot p(\lambda)$ and the $\lambda$ factors cancel in every MIS ratio. The incremental-ratio walk, the roulette products, the delta convention: all invariant.
- **The bridge ladder survives for mirrors** — reflection is achromatic, so the rungs are $\lambda$-independent. Glass already ends the ladder (§4.3); with dispersion, a glass prism in the chain would split the beam into a per-$\lambda$ fan, and the beam trace would register a *rainbow ladder* — one rung set per wavelength band, each following its own refracted chain.
- **Fresnel** becomes $\lambda$-dependent through $n(\lambda)$ — already supported by the formula's structure.
- **Firefly clamp**: the bound becomes a spectral quantity (8× the brightest emitter's luminance *at the sampled $\lambda$*); nothing structural changes.

**Cost**: the transport loop is unchanged (a float rides along); the costs are the emitter/sensor spectra plumbing and the color conversion. This is the cheapest of the model changes and the most visually rewarding for the demo scene (dispersive caustics through the glass cube; physically correct laser colors).

### 9.3 Participating media

The renderer's design decision "no beam in flight — in clean air a laser beam is invisible from the side" is a special case of $\sigma_s = 0$. Add a participating medium (fog, smoke) with absorption $\sigma_a$, scattering $\sigma_s$, and a phase function $p$ (Henyey–Greenstein), and the beam legs *become* real: single scattering off the medium along each leg is what makes a laser visible in fog.

- **Model**: the radiative transfer equation replaces the rendering equation's surface-only integral; volumetric BDPT adds vertices at scattering events, samples distances by $\sigma_t$, and carries transmittance ratios in the geometry term (delta tracking / null collision for heterogeneous media).
- **The bridge ladder is medium-agnostic** — this is the elegant part. A *medium point* beside leg $j{+}1$ sees the aperture through exactly the first $j$ mirrors, exactly like a surface point: aim at rung $j$. The ladder needs only one extension: the cosine–bridge mixture must also fire at *medium* vertices (currently only at diffuse surface vertices), with the medium's phase function playing the BSDF's role in the weight.
- **Cost**: distance sampling per segment, more vertices per path, a 3D representation of the medium (grid or BVH over volumes). The flat-BVH traversal generalizes (the medium's density lookup is an extra texture fetch); the MIS machinery gains the distance-sampling pdfs as additional edge factors — the incremental-ratio walk absorbs them without structural change.

### 9.4 Polarization

Replace scalar radiance with **Stokes vectors** $(I, Q, U, V)$ and BSDFs with 4×4 **Mueller matrices**. The Fresnel s/p coefficients — already computed separately in `fresnel_dielectric` — are the building blocks of the interface matrices; metallic mirror coatings add distinct s/p *phases* (ellipticity upon reflection). Lasers are typically linearly polarized, so the demo scene's transport becomes genuinely polarization-dependent (the Brewster angle of the glass cube selectively dims one polarization).

- **MIS is untouched**: sampling remains polarization-independent, so pdfs factorize and the weights are invariant.
- **Cost**: 4× state per vertex, a 4×4 matrix product per event; the firefly statistic operates on $I$ (the first Stokes component) unchanged.
- **Synergy with §9.1**: polarization is the *vector* part of wave optics; the two upgrades compose into a full coherent-polarized laser channel.

### 9.5 Glossy microfacet BSDFs

Replace the perfect mirror with a GGX microfacet lobe. The delta becomes continuous: connections through glossy vertices become *valid* (nonzero eval), so the BDPT blind spot softens on its own — the beam's glow becomes reachable by ordinary connections, at the price of high variance for narrow lobes.

- The **ladder's exact unfolding breaks** (no single reflection direction); the rungs become *fuzzy images* — the image disc convolved with the lobe — and remain the low-variance technique exactly when the lobe is narrow (the bridge is the small-roughness limit of the glossy connection).
- The mixture sampling generalizes: the bridge cone becomes the specular lobe's reflection of the image disc; the MIS machinery already supports arbitrary mixture components (§4.4).

### 9.6 Estimator-level alternatives (for context)

Not model changes but worth naming as complements to the ladder, since they address the same blind spot by different means: **photon mapping** (caustics via density estimation — biased but robust), **VCM** (BDPT + photon merging), **MLT** (Metropolis perturbation of hard specular chains), **manifold exploration** (projective-Newton perturbation through specular manifolds). The bridge ladder's advantage over all of them: it is *analytic, unbiased, and free* — the rungs are byproducts of a beam trace the scene builder already performs — but it is specialized to delta chains from known tiny emitters, whereas the general estimators pay their cost on every path.

### 9.7 Summary

| Model change | Physics added | MIS | Bridge ladder | Firefly machinery | Cost |
|---|---|---|---|---|---|
| Spectral + dispersion (§9.2) | $n(\lambda)$, prisms, true laser colors | invariant (λ factorizes) | survives (mirrors achromatic); glass rungs become per-λ fans | bound becomes spectral | low — plumbing |
| Polarization (§9.4) | s/p split, Brewster, coatings | invariant | survives | statistic reads $I$ | 4× state |
| Participating media (§9.3) | beam visible in fog, RTE | gains distance pdfs (absorbed by the ratio walk) | **medium-agnostic** — fire the mixture at medium vertices | unchanged | medium — distance sampling, volume representation |
| Wave optics (§9.1) | diffraction, interference, speckle | bypassed (coherent channel is deterministic) | survives as the geometric shadow; rungs gain phase (path length already preserved) | **survives by design** — speckle is a real heavy-tailed signal | high — hybrid GO+PO; full-wave infeasible at scene scale |
| Glossy microfacets (§9.5) | rough mirrors, soft reflections | connections through glossy vertices become valid | rungs become fuzzy images; bridge = small-roughness limit | unchanged | moderate |

---

## 10. Conclusion

The renderer demonstrates that a *pure, unbiased* BDPT — with no photon maps, no decals, no fake lights — can render laser beams through arbitrary mirror chains, provided the estimator is taught the one thing Monte Carlo cannot discover on its own: the analytic geometry of the delta path. The specular bridge ladder turns the classic unfolding construction into a *sampling technique*: derived on demand from the moving scene, carried through the MIS machinery with full mixture-density consistency, and constrained (not free) by the downstream outlier rejection's winsorization bound. Around it, the standard machinery is implemented with unusual care — hypothetical-extension pdfs recomputed from path geometry through a single `PathView`, roulette probabilities inside the MIS products, a two-tier firefly rejection whose loose tier is *physically* motivated precisely so the tight tier can be *statistically* motivated — and the whole is validated by furnace tests, differentials, and sum-to-one invariants rather than by eyeballing.

The model-change analysis shows the architecture's forward compatibility: spectral and polarization upgrades leave the MIS machinery provably invariant; participating media extend the bridge ladder rather than replace it (the image-disc construction is medium-agnostic); and the wave-optics upgrade — the deepest change — finds its geometric skeleton already in place, because the unfolding construction preserves exactly the quantity waves need: path length.

---

## Appendix A: Parameters

| Parameter | Value | Where |
|---|---|---|
| Max path vertices | 8 | `bdpt::MAX_PATH_VERTICES` |
| RR start / min survival | ext 3 / 0.05 | `bdpt::RR_START`, `RR_MIN` |
| Bridge mixture prob $p_B$ | 0.25 (must be ≥ 1/(1+K²) = 0.1) | `bdpt::BRIDGE_PROB` |
| Firefly clamp | 8× brightest emitter luminance | `bdpt::FIREFLY_LIGHT_MULT` |
| Winsorization K | 3.0 | `bdpt::WINSOR_K` |
| Moment EMA window | 16 samples | `bdpt::MOMENT_WINDOW` |
| Accumulated frames | 128 | `renderer::MAX_ACCUMULATED_FRAMES` |
| Bilateral | radius 1, relative σ_r 0.15 | `output::bilateral`, `display.wgsl` |
| Tone map white point | 10 | `output::TONE_MAP_WHITE_POINT` |
| Laser aperture / cone | r = 0.01 m, θ = 0.01 rad | `main.rs` scene |
| Max beam bounces | 8 | `scene::MAX_BEAM_BOUNCES` |
| GPU BVH depth cap / stack | 24 / 32 | `gpu::MAX_BVH_DEPTH`, WGSL `MAX_BVH_STACK` |
| FHD export | 1920×1080, 128 spp, depth 8, 32-spp batches | `renderer` consts |

## Appendix B: Code Map

| Component | File |
|---|---|
| BDPT + MIS + roulette + bridges (CPU) | `src/bdpt.rs` |
| BDPT + MIS + accumulation (GPU, 1:1 port) | `shaders/bdpt/` |
| Scene description, beam trace, ladder derivation | `src/scene.rs` |
| BSDFs, Fresnel/Snell, prims, area lights | `src/geometry.rs` |
| BVH (CPU tree) | `src/bvh.rs` |
| Flat BVH builder, GPU renderer, layout checks | `src/gpu.rs` |
| Renderer trait, CPU backend, accumulation + outlier rejection | `src/renderer.rs` |
| Camera, ray generation, rayon parallelism | `src/camera.rs` |
| Tone map, bilateral filter, PNG, terminal display | `src/output.rs` |
| Display pass (denoise + tone map + gamma + overlay) | `shaders/display.wgsl` |
| Window frontend, sweep animation, FHD export | `src/window.rs` |
| Demo scene, frontends | `src/main.rs` |
