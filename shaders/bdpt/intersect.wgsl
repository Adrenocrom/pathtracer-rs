// BDPT shader fragment, concatenated with its siblings in dependency
// order by BDPT_WGSL in src/gpu.rs (WGSL has no #include). This
// fragment: the flat-BVH intersection machinery.

// --- Intersection: flat BVH traversal ---
//
// The scene is uploaded as a kind-tagged prim array plus a flat BVH
// (built on the CPU, src/gpu.rs). Traversal keeps an explicit stack of
// pending nodes (the GPU analogue of the CPU BVH's recursion) and
// culls every box by the current best hit distance — the analogue of
// the CPU's subtree early-exit test. The builder caps the tree depth
// at 24, and a pop-based traversal of a depth-d binary tree holds at
// most d+1 pending nodes, so the 32-entry stack cannot overflow (the
// bounds guards make it impossible even for a degenerate tree).

const KIND_SPHERE: u32 = 0u;
const KIND_QUAD: u32 = 1u;
const KIND_CUBE: u32 = 2u;
const KIND_PLANE: u32 = 3u;
const KIND_CYLINDER: u32 = 4u;
// The builder caps the tree depth at MAX_BVH_DEPTH; a pop-based
// traversal of a depth-d binary tree holds at most d+1 pending nodes,
// so a 32-entry stack can never overflow (the guards below make it
// impossible even for a degenerate tree).
const MAX_BVH_STACK: u32 = 32u;

// Slab test of one AABB against the ray segment [0, t_max). Returns
// false when the box is behind the ray or beyond the current best hit.
// Axis-parallel rays are handled explicitly (checking the direction,
// never the reciprocal — 1/0 is inf and 0*inf is NaN) so a parallel
// slab can never produce a NaN that silently drops valid hits. This is
// the WGSL shape of math.rs's BBox::intersect.
fn aabb_hit(bmin: vec3<f32>, bmax: vec3<f32>, origin: vec3<f32>,
            dir: vec3<f32>, inv_dir: vec3<f32>, t_max: f32) -> bool {
    var t_min = 0.0;
    var t_max_slab = t_max;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let d = dir[i];
        if (d == 0.0) {
            // Parallel to this slab: the origin must lie inside it.
            if (origin[i] < bmin[i] || origin[i] > bmax[i]) { return false; }
            continue;
        }
        let inv = inv_dir[i];
        var t0 = (bmin[i] - origin[i]) * inv;
        var t1 = (bmax[i] - origin[i]) * inv;
        if (t0 > t1) {
            let tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_min = max(t_min, t0);
        t_max_slab = min(t_max_slab, t1);
        if (t_min > t_max_slab) { return false; }
    }
    return true;
}

fn intersect_prim(p: Prim, origin: vec3<f32>, dir: vec3<f32>, best: ptr<function, Hit>) {
    if (p.kind == KIND_SPHERE) {
        let oc = origin - p.center;
        let a = dot(dir, dir);
        let b = 2.0 * dot(oc, dir);
        let c = dot(oc, oc) - p.radius * p.radius;
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) { return; }
        let sqrt_d = sqrt(disc);
        var t = (-b - sqrt_d) / (2.0 * a);
        if (t < EPS_NEAR) { t = (-b + sqrt_d) / (2.0 * a); }
        if (t < EPS_NEAR || t >= (*best).t) { return; }
        let hp = origin + dir * t;
        (*best).t = t;
        (*best).p = hp;
        (*best).normal = normalize3(hp - p.center);
        (*best).albedo = p.albedo;
        (*best).sellmeier = p.sellmeier;
        (*best).rough = p.rough;
        (*best).f0 = p.f0;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    if (p.kind == KIND_QUAD) {
        let denom = dot(p.normal, dir);
        if (abs(denom) < 1e-6) { return; }
        let t = dot(p.center - origin, p.normal) / denom;
        if (t < EPS_NEAR || t >= (*best).t) { return; }
        let hp = origin + dir * t;
        let local = hp - p.center;
        if (abs(dot(local, p.tangent)) > p.radius || abs(dot(local, p.bitangent)) > p.half_h) { return; }
        (*best).t = t;
        (*best).p = hp;
        (*best).normal = p.normal;
        (*best).albedo = p.albedo;
        (*best).sellmeier = p.sellmeier;
        (*best).rough = p.rough;
        (*best).f0 = p.f0;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    if (p.kind == KIND_CUBE) {
        var t_min = -3.4e38;
        var t_max = 3.4e38;
        // Slab test per axis; unrolled to avoid indexing overhead.
        for (var i = 0u; i < 3u; i = i + 1u) {
            let d = dir[i];
            let o = origin[i];
            let lo = p.center[i]; // cube min
            let hi = p.normal[i]; // cube max
            if (d == 0.0) {
                if (o < lo || o > hi) { return; }
                continue;
            }
            let inv_d = 1.0 / d;
            var t0 = (lo - o) * inv_d;
            var t1 = (hi - o) * inv_d;
            if (t0 > t1) {
                let tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            t_min = max(t_min, t0);
            t_max = min(t_max, t1);
        }
        if (!(t_max >= t_min && t_max > 0.0 && t_min < 1e6)) { return; }
        var t = t_min;
        if (t < EPS_NEAR) { t = t_max; }
        if (t < EPS_NEAR || t >= (*best).t) { return; }
        let hp = origin + dir * t;
        // Outward normal of the face closest to the hit point (same rule as
        // geometry.rs so top/+x/+z hits shade identically).
        let d0 = abs(hp.x - p.center.x); let d1 = abs(hp.x - p.normal.x);
        let d2 = abs(hp.y - p.center.y); let d3 = abs(hp.y - p.normal.y);
        let d4 = abs(hp.z - p.center.z); let d5 = abs(hp.z - p.normal.z);
        var best_d = d0;
        var normal = vec3<f32>(-1.0, 0.0, 0.0);
        if (d1 < best_d) { best_d = d1; normal = vec3<f32>(1.0, 0.0, 0.0); }
        if (d2 < best_d) { best_d = d2; normal = vec3<f32>(0.0, -1.0, 0.0); }
        if (d3 < best_d) { best_d = d3; normal = vec3<f32>(0.0, 1.0, 0.0); }
        if (d4 < best_d) { best_d = d4; normal = vec3<f32>(0.0, 0.0, -1.0); }
        if (d5 < best_d) { best_d = d5; normal = vec3<f32>(0.0, 0.0, 1.0); }
        (*best).t = t;
        (*best).p = hp;
        (*best).normal = normal;
        (*best).albedo = p.albedo;
        (*best).sellmeier = p.sellmeier;
        (*best).rough = p.rough;
        (*best).f0 = p.f0;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    if (p.kind == KIND_CYLINDER) {
        // Closed cylinder: base cap at `center`, axis `tangent`,
        // half-height `half_h`, radius `radius`. Same decomposition as
        // geometry.rs: split the ray into axial and radial parts; the
        // side test is a 2D circle intersection in the radial plane.
        let a = p.tangent;
        let oc = origin - p.center;
        let d_perp = dir - a * dot(dir, a);
        let oc_perp = oc - a * dot(oc, a);
        let dz = dot(dir, a);
        let oz = dot(oc, a);

        var best_t = 3.4e38;
        var best_normal = vec3<f32>(0.0, 1.0, 0.0);

        // Curved side: |oc_perp + t * d_perp| = radius. The axial
        // coordinate z is measured from the base cap, so the cylinder
        // spans z in [0, 2 * half_h].
        let aa = dot(d_perp, d_perp);
        if (aa > 1e-12) {
            let bb = 2.0 * dot(oc_perp, d_perp);
            let cc = dot(oc_perp, oc_perp) - p.radius * p.radius;
            let disc = bb * bb - 4.0 * aa * cc;
            if (disc >= 0.0) {
                let sqrt_d = sqrt(disc);
                for (var k = 0u; k < 2u; k = k + 1u) {
                    var t = (-bb - sqrt_d) / (2.0 * aa);
                    if (k == 1u) { t = (-bb + sqrt_d) / (2.0 * aa); }
                    if (t < EPS_NEAR || t >= best_t) { continue; }
                    let z = oz + t * dz;
                    if (z < 0.0 || z > 2.0 * p.half_h) { continue; }
                    let hp = origin + dir * t;
                    best_t = t;
                    best_normal = normalize3(hp - p.center - a * z);
                }
            }
        }

        // Caps: two one-sided discs at z = 0 (base) and z = 2*half_h (top).
        for (var k = 0u; k < 2u; k = k + 1u) {
            let cap_z = select(0.0, 2.0 * p.half_h, k == 1u);
            let cap_n = select(a * -1.0, a, k == 1u);
            let cap_point = p.center + a * cap_z;
            let denom = dot(cap_n, dir);
            if (abs(denom) < 1e-6) { continue; }
            let t = dot(cap_point - origin, cap_n) / denom;
            if (t < EPS_NEAR || t >= best_t) { continue; }
            let hp = origin + dir * t;
            let radial = hp - cap_point;
            if (dot(radial, radial) > p.radius * p.radius) { continue; }
            best_t = t;
            best_normal = cap_n;
        }

        if (best_t >= (*best).t || best_t >= 3.4e38) { return; }
        let hp = origin + dir * best_t;
        (*best).t = best_t;
        (*best).p = hp;
        (*best).normal = best_normal;
        (*best).albedo = p.albedo;
        (*best).sellmeier = p.sellmeier;
        (*best).rough = p.rough;
        (*best).f0 = p.f0;
        (*best).mat_type = p.mat_type;
        (*best).ior = p.ior;
        return;
    }
    // KIND_PLANE
    let denom = dot(p.normal, dir);
    if (abs(denom) < 1e-6) { return; }
    let t = dot(p.center - origin, p.normal) / denom;
    if (t < EPS_NEAR || t >= (*best).t) { return; }
    (*best).t = t;
    (*best).p = origin + dir * t;
    (*best).normal = p.normal;
    (*best).albedo = p.albedo;
    (*best).sellmeier = p.sellmeier;
    (*best).rough = p.rough;
    (*best).f0 = p.f0;
    (*best).mat_type = p.mat_type;
    (*best).ior = p.ior;
}

fn intersect_scene(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    var best: Hit;
    best.t = 1e30;
    best.p = vec3<f32>(0.0);
    best.normal = vec3<f32>(0.0, 1.0, 0.0);
    best.albedo = vec3<f32>(0.0);
    best.sellmeier = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    best.rough = 0.0;
    best.f0 = 0.0;
    best.mat_type = 0u;
    best.ior = 1.0;
    if (uniforms.n_prims == 0u) { return best; }

    // Precompute the reciprocals once per ray; the per-node slab tests
    // reuse them. A zero direction component would be a division by
    // zero, so those lanes keep a dummy 1.0 — aabb_hit tests the raw
    // direction for zero first and never consumes the dummy.
    var inv_dir = vec3<f32>(1.0);
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (dir[i] != 0.0) {
            inv_dir[i] = 1.0 / dir[i];
        }
    }
    var stack: array<u32, 32u>;
    var sp = 0u;
    stack[0] = 0u; // root
    sp = 1u;
    while (sp > 0u) {
        sp = sp - 1u;
        let cur = stack[sp];
        let node = bvh_nodes[cur];
        if (!aabb_hit(node.min, node.max, origin, dir, inv_dir, best.t)) { continue; }
        if (node.count > 0u) {
            // Leaf: a contiguous run of prims, tested in order.
            for (var i = 0u; i < node.count; i = i + 1u) {
                intersect_prim(prims[node.right_first + i], origin, dir, &best);
            }
            continue;
        }
        // Inner: left child is cur+1 (pre-order layout), right child is
        // right_first. Push both; the pop order visits the left subtree
        // first, which tends to find the near hit early and lets the
        // best.t culling skip the right subtree's boxes.
        if (sp < MAX_BVH_STACK) {
            stack[sp] = node.right_first;
            sp = sp + 1u;
        }
        if (sp < MAX_BVH_STACK) {
            stack[sp] = cur + 1u;
            sp = sp + 1u;
        }
    }
    return best;
}

fn has_hit(h: Hit) -> bool { return h.t < 1e29; }
