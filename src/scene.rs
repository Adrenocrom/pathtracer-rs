//! Data-oriented scene description shared by the CPU and GPU backends.
//!
//! The CPU renderer consumes `SceneData` through the existing
//! `Box<dyn Intersectable>` + BVH path (it stays the reference oracle); the
//! GPU backend uploads the same data as flat POD buffers. Keeping one source
//! of truth for the scene guarantees both backends trace identical geometry.

use crate::geometry::{
    fresnel_dielectric, reflect_dir, refract_dir, AreaLight, Bsdf, Cube, Cylinder, Intersectable,
    Material, Plane, Quad, Ray, Sphere,
};
use crate::math::{EPS_NEAR, Vec3};

/// The laser housing's material: a normal diffuse surface, so the
/// cylinder shades like any other object in the scene (the laser
/// material is a pure emitter — albedo 0, emission gated to the beam
/// cone — which renders the whole surface black).
const LASER_HOUSING_ALBEDO: Vec3 = Vec3 {
    x: 0.8,
    y: 0.8,
    z: 0.8,
};
/// How much fatter (and longer) the housing is than the aperture puck,
/// so the puck's rim never peeks out from behind it: 10%.
const LASER_HOUSING_MARGIN: f32 = 0.1;
/// The aperture puck's height as a fraction of the housing's: thin
/// (10%), so it reads as the aperture disc rather than a second
/// cylinder.
const LASER_PUCK_FRACTION: f32 = 0.1;

/// The maximum number of specular surfaces a laser beam is traced
/// across when deriving its specular bridge (see
/// `SceneData::bridge_of_light`). A beam caught in a mirror loop must
/// not spin forever; 8 matches the path tracer's vertex budget.
pub const MAX_BEAM_BOUNCES: usize = 8;

/// A surface material in flat, GPU-friendly form. `emission != 0` marks a
/// pure emitter: the BSDF is ignored and the surface does not scatter.
#[derive(Clone, Copy, Debug)]
pub struct MaterialData {
    pub albedo: Vec3,
    pub emission: Vec3,
    pub bsdf: Bsdf,
}

impl MaterialData {
    pub fn diffuse(albedo: Vec3) -> Self {
        Self {
            albedo,
            emission: Vec3::new(0.0, 0.0, 0.0),
            bsdf: Bsdf::Diffuse,
        }
    }

    pub fn emissive(emission: Vec3) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            emission,
            bsdf: Bsdf::Diffuse,
        }
    }

    pub fn mirror(albedo: Vec3) -> Self {
        Self {
            albedo,
            emission: Vec3::new(0.0, 0.0, 0.0),
            bsdf: Bsdf::Mirror,
        }
    }

    pub fn glass(ior: f32) -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            emission: Vec3::new(0.0, 0.0, 0.0),
            bsdf: Bsdf::Glass { ior },
        }
    }

    /// A laser emitter: radiates `emission` only into the cone of
    /// half-angle `theta` around the geometric normal (the beam).
    pub fn laser(emission: Vec3, theta: f32) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            emission,
            bsdf: Bsdf::Laser {
                cos_theta: theta.cos(),
            },
        }
    }

    /// True when this material radiates into a cone (a laser aperture).
    pub fn is_laser(&self) -> bool {
        matches!(self.bsdf, Bsdf::Laser { .. })
    }

    /// Pure emitters do not scatter; the BSDF is ignored for them.
    pub fn is_emissive(&self) -> bool {
        self.emission != Vec3::new(0.0, 0.0, 0.0)
    }

    /// Delta BSDFs cannot be connection endpoints. Kept as API surface
    /// (the flat-form mirror of `Material::is_delta`, which bdpt.rs uses);
    /// the GPU shader encodes the same test inline via `is_delta_type`.
    #[allow(dead_code)]
    pub fn is_delta(&self) -> bool {
        !self.is_emissive() && self.bsdf.is_delta()
    }

    fn to_material(self) -> Material {
        Material {
            albedo: self.albedo,
            emission: self.emission,
            bsdf: self.bsdf,
        }
    }
}

/// A sampleable emitter surface, mirroring `geometry::AreaLight` in flat form.
#[derive(Clone, Copy)]
pub struct LightData {
    /// 0 = sphere, 1 = quad, 2 = laser aperture (disc + cone).
    pub shape: u32,
    pub center: Vec3,
    /// Sphere radius (unused for quads).
    pub radius: f32,
    /// Quad normal (unused for spheres).
    pub normal: Vec3,
    /// Quad tangent (unused for spheres).
    pub tangent: Vec3,
    /// Quad bitangent (unused for spheres).
    pub bitangent: Vec3,
    /// Quad half-extents (unused for spheres).
    pub half_w: f32,
    pub half_h: f32,
    pub emission: Vec3,
}

impl LightData {
    pub fn from_sphere(center: Vec3, radius: f32, emission: Vec3) -> Self {
        Self {
            shape: 0,
            center,
            radius,
            normal: Vec3::new(0.0, 0.0, 0.0),
            tangent: Vec3::new(0.0, 0.0, 0.0),
            bitangent: Vec3::new(0.0, 0.0, 0.0),
            half_w: 0.0,
            half_h: 0.0,
            emission,
        }
    }

    /// Quad light constructor. Not used by the demo scene (its ceiling quad
    /// light is disabled), but part of the scene API. Uses the same
    /// world-axis tangent frame as `add_wall`/`add_quad` so the light
    /// geometry matches a wall of the same arguments.
    #[allow(dead_code)]
    pub fn from_quad(center: Vec3, normal: Vec3, width: f32, height: f32, emission: Vec3) -> Self {
        let normal = normal.normalize();
        let (tangent, bitangent, half_w, half_h) =
            crate::geometry::quad_frame(normal, width, height);
        Self {
            shape: 1,
            center,
            radius: 0.0,
            normal,
            tangent,
            bitangent,
            half_w,
            half_h,
            emission,
        }
    }

    /// A laser aperture light: a disc of `radius` centered at `center`
    /// with outward normal `normal`, radiating only into the cone of
    /// half-angle `theta` around the normal. The disc is the emitting
    /// cap of the laser's cylinder housing (see `SceneData::add_laser`);
    /// `height` is the housing's total height, carried so `add_light`
    /// can register the housing as geometry.
    pub fn from_laser(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        theta: f32,
        height: f32,
        emission: Vec3,
    ) -> Self {
        Self {
            shape: 2,
            center,
            radius,
            normal: normal.normalize(),
            tangent: Vec3::new(0.0, 0.0, 0.0),
            bitangent: Vec3::new(0.0, 0.0, 0.0),
            half_w: theta,   // the cone half-angle rides in the unused slot
            half_h: height, // the housing height rides in the unused slot
            emission,
        }
    }

    /// Surface area (used by tests; the GPU shader computes it inline).
    #[allow(dead_code)]
    pub fn area(&self) -> f32 {
        const PI: f32 = std::f32::consts::PI;
        if self.shape == 0 {
            4.0 * PI * self.radius * self.radius
        } else if self.shape == 2 {
            PI * self.radius * self.radius
        } else {
            4.0 * self.half_w * self.half_h
        }
    }
}

/// The whole scene in flat, GPU-friendly form.
#[derive(Clone)]
pub struct SceneData {
    pub spheres: Vec<SphereData>,
    pub quads: Vec<QuadData>,
    pub cubes: Vec<CubeData>,
    pub planes: Vec<PlaneData>,
    pub cylinders: Vec<CylinderData>,
    pub lights: Vec<LightData>,
}

/// The traced path of a laser beam: the legs (aperture → hit₁ → hit₂ →
/// … → spot) and, when the beam reaches a diffuse surface, the spot.
#[derive(Clone, Debug)]
pub struct BeamTrace {
    /// One (start, end) pair per leg. The last leg's end is the spot
    /// (or the escape point when the beam leaves the scene).
    pub legs: Vec<(Vec3, Vec3)>,
    /// The beam's final stop on a diffuse surface.
    pub spot: Option<BeamSpot>,
}

/// Where a traced laser beam lands: the point, the surface normal
/// (oriented toward the beam's arrival side), the surface material, and
/// the beam's color throughput across the specular chain (mirror
/// albedos, glass Fresnel transmittances).
#[derive(Clone, Copy, Debug)]
pub struct BeamSpot {
    pub p: Vec3,
    pub normal: Vec3,
    pub mat: MaterialData,
    pub throughput: Vec3,
}

#[derive(Clone, Copy)]
pub struct SphereData {
    pub center: Vec3,
    pub radius: f32,
    pub mat: MaterialData,
}

#[derive(Clone, Copy)]
pub struct QuadData {
    pub center: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub half_w: f32,
    pub half_h: f32,
    pub mat: MaterialData,
}

#[derive(Clone, Copy)]
pub struct CubeData {
    pub min: Vec3,
    pub max: Vec3,
    pub mat: MaterialData,
}

#[derive(Clone, Copy)]
pub struct PlaneData {
    pub point: Vec3,
    pub normal: Vec3,
    pub mat: MaterialData,
}

/// A closed cylinder in flat form: `base` is the center of the base cap,
/// `axis` the unit direction from base to top, `half_h` the half-height.
/// Regular cylinders are added via `add_cylinder`; for a laser, the
/// housing cylinder is diffuse and the aperture is a separate thin
/// cylinder puck of laser material (see `add_light`).
#[derive(Clone, Copy)]
pub struct CylinderData {
    pub base: Vec3,
    pub axis: Vec3,
    pub radius: f32,
    pub half_h: f32,
    pub mat: MaterialData,
}

impl SceneData {
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            quads: Vec::new(),
            cubes: Vec::new(),
            planes: Vec::new(),
            cylinders: Vec::new(),
            lights: Vec::new(),
        }
    }

    pub fn add_sphere(&mut self, center: Vec3, radius: f32, mat: MaterialData) {
        self.spheres.push(SphereData {
            center,
            radius,
            mat,
        });
    }

    /// Add a finite rectangle (e.g. a light panel). Not used by the demo
    /// scene, but part of the scene API — the original code had a
    /// commented-out ceiling quad light.
    #[allow(dead_code)]
    pub fn add_quad(
        &mut self,
        center: Vec3,
        normal: Vec3,
        width: f32,
        height: f32,
        mat: MaterialData,
    ) {
        let normal = normal.normalize();
        let (tangent, bitangent, half_w, half_h) =
            crate::geometry::quad_frame(normal, width, height);
        self.quads.push(QuadData {
            center,
            normal,
            tangent,
            bitangent,
            half_w,
            half_h,
            mat,
        });
    }

    /// Add a cube (equal extents on all axes). For a rectangular box
    /// (a "Quader") with different extents per axis, use `add_box`.
    pub fn add_cube(&mut self, center: Vec3, size: f32, mat: MaterialData) {
        let half = size * 0.5;
        self.cubes.push(CubeData {
            min: center - Vec3::new(half, half, half),
            max: center + Vec3::new(half, half, half),
            mat,
        });
    }

    /// Add an axis-aligned rectangular box (cuboid) — like `add_cube`,
    /// but with independent extents per axis. `size` is the *full*
    /// extent along each axis (x, y, z), matching `add_cube`'s
    /// convention: `add_box(c, Vec3::new(s, s, s), m)` equals
    /// `add_cube(c, s, m)`.
    ///
    /// Unlike a wall (a one-sided quad), a box shades from all sides:
    /// each of the six faces carries its own outward normal, so it can
    /// serve as a slab, pillar, table top or any block seen from
    /// multiple directions.
    ///
    /// The box is stored as min/max — exactly what both backends
    /// consume (the CPU `Cube` and the GPU's slab test both read
    /// min/max), so no backend changes are involved.
    ///
    /// Not used by the demo scene, but part of the scene API — covered
    /// by tests and available for future scenes.
    #[allow(dead_code)]
    pub fn add_box(&mut self, center: Vec3, size: Vec3, mat: MaterialData) {
        let half = size * 0.5;
        self.cubes.push(CubeData {
            min: center - half,
            max: center + half,
            mat,
        });
    }

    /// Add a closed cylinder as a regular scene object: centered at
    /// `center`, axis along `axis` (any length; normalized internally),
    /// total height `height`, radius `radius`. The caps are one-sided
    /// (they shade only from outside), the curved side shades from both
    /// sides. Works with any material — diffuse, mirror, glass, or even
    /// emissive (though for a light source prefer `add_light`, which
    /// also registers it as a sampleable emitter).
    ///
    /// Not used by the demo scene (its only cylinders are the laser
    /// housing and puck, registered via `add_light`), but part of the
    /// scene API — covered by tests and available for future scenes.
    #[allow(dead_code)]
    pub fn add_cylinder(
        &mut self,
        center: Vec3,
        axis: Vec3,
        height: f32,
        radius: f32,
        mat: MaterialData,
    ) {
        let axis = axis.normalize();
        self.cylinders.push(CylinderData {
            base: center - axis * (height * 0.5),
            axis,
            radius,
            half_h: height * 0.5,
            mat,
        });
    }

    /// Add an infinite plane. Kept for tests and future scenes; the demo
    /// room uses `add_wall` (finite quads) so the BVH can cull.
    #[allow(dead_code)]
    pub fn add_plane(&mut self, point: Vec3, normal: Vec3, mat: MaterialData) {
        self.planes.push(PlaneData { point, normal, mat });
    }

    /// Add a finite rectangular wall (a quad) instead of an infinite
    /// plane. Planes get a ±1e6 bounding box, so a room built from them
    /// makes the BVH useless: the root box is huge and no wall is ever
    /// culled. A quad has a tight box, so the BVH can actually cull.
    ///
    /// The quad is centered at `point` with the given outward `normal`.
    /// For axis-aligned normals the extents are world-axis sized:
    /// `width` spans the horizontal axis (x for floor/ceiling/back/front
    /// walls, z for side walls) and `height` spans the vertical axis (y,
    /// or z for floor/ceiling). The caller centers side walls at
    /// mid-height — the quad is centered on `point`, so a wall centered
    /// at floor level would cover only the bottom half of the room.
    ///
    /// The tangent frame is derived exactly like `geometry::Quad::new`,
    /// so the CPU and GPU backends see identical geometry.
    pub fn add_wall(
        &mut self,
        point: Vec3,
        normal: Vec3,
        width: f32,
        height: f32,
        mat: MaterialData,
    ) {
        let normal = normal.normalize();
        // Same world-axis rule as `geometry::Quad::new` so the tangent
        // frame matches exactly.
        let (tangent, bitangent, half_w, half_h) =
            crate::geometry::quad_frame(normal, width, height);
        self.quads.push(QuadData {
            center: point,
            normal,
            tangent,
            bitangent,
            half_w,
            half_h,
            mat,
        });
    }

    /// Register an emitter. The light is stored in two places, by design:
    ///
    /// * as real intersectable geometry (`spheres`/`quads`/`cylinders`),
    ///   so eye subpaths can hit it on *both* backends — the GPU's
    ///   `intersect_scene` reads only the geometry buffers, never the
    ///   light buffer;
    /// * as a sampleable `LightData`, which drives the light subpath and
    ///   the MIS weight of the "eye subpath hits the emitter" strategy.
    ///
    /// Because this registers the primitive, do not also add the same
    /// shape via `add_sphere`/`add_quad` — that would duplicate it.
    pub fn add_light(&mut self, light: LightData) {
        let mat = MaterialData::emissive(light.emission);
        if light.shape == 0 {
            self.spheres.push(SphereData {
                center: light.center,
                radius: light.radius,
                mat,
            });
        } else if light.shape == 2 {
            // A laser aperture: the disc is the top cap of a small
            // cylinder "puck" of laser material, registered here so the
            // aperture is real geometry on both backends. The puck pokes
            // through the top cap of the housing cylinder, so the
            // aperture disc is the puck's own top cap and no ray can
            // tie between the two surfaces.
            let theta = light.half_w; // the cone half-angle rides in half_w
            let height = light.half_h; // the housing height rides in half_h
            let axis = light.normal;
            // The housing: a normal diffuse material, so the cylinder
            // shades like any other object (giving the whole housing
            // the laser material would render it black — a pure emitter
            // scatters nothing). It is fatter than the puck, and its top
            // cap sits at the puck's mid-height — behind the aperture
            // plane — so the beam exits unoccluded.
            self.cylinders.push(CylinderData {
                base: light.center - axis * height,
                axis,
                radius: light.radius * (1.0 + LASER_HOUSING_MARGIN),
                half_h: (height - height * LASER_PUCK_FRACTION * 0.5) * 0.5,
                mat: MaterialData::diffuse(LASER_HOUSING_ALBEDO),
            });
            // The aperture puck: thin, carrying the laser material.
            // Its top cap is the aperture disc; its lower half is sunk
            // into the housing. The laser material is a pure emitter
            // (albedo 0, emission gated to the beam cone), so only the
            // top cap radiates — the exposed sliver of the curved side
            // reads as the dark bezel around the aperture.
            self.cylinders.push(CylinderData {
                base: light.center - axis * (height * LASER_PUCK_FRACTION),
                axis,
                radius: light.radius,
                half_h: height * LASER_PUCK_FRACTION * 0.5,
                mat: MaterialData::laser(light.emission, theta),
            });
        } else {
            self.quads.push(QuadData {
                center: light.center,
                normal: light.normal,
                tangent: light.tangent,
                bitangent: light.bitangent,
                half_w: light.half_w,
                half_h: light.half_h,
                mat,
            });
        }
        self.lights.push(light);
    }

    /// Add a laser light source: a cylinder housing whose top cap is the
    /// aperture, emitting a beam along `axis` (the cylinder's axis, so
    /// the beam exits the circular face) into a cone of half-angle
    /// `theta`. `height` is the cylinder's total height; the aperture
    /// disc sits at `center + axis * height/2`.
    ///
    /// The housing carries a normal diffuse material, so it shades like
    /// any other object; the aperture is a thin cylinder puck of laser
    /// material whose top cap is the aperture disc, poking through the
    /// housing's top cap. The laser material is a pure emitter (albedo
    /// 0, emission gated to the beam cone), so only the puck's top cap
    /// radiates — the rest of it is hidden inside the housing.
    ///
    /// The specular bridge *ladder* (one bridge per mirror the beam
    /// crosses — see `bridges_of_light`) is *derived* on demand, never
    /// baked here: the beam is traced through the fully-built scene,
    /// so the laser may be added before, after or interleaved with the
    /// mirrors, and a scene that moves between frames simply re-derives
    /// every rung from the new geometry.
    pub fn add_laser(
        &mut self,
        center: Vec3,
        axis: Vec3,
        height: f32,
        radius: f32,
        theta: f32,
        emission: Vec3,
    ) {
        let axis = axis.normalize();
        let aperture = center + axis * (height * 0.5);
        let light = LightData::from_laser(aperture, axis, radius, theta, height, emission);
        self.add_light(light);
    }

    /// The specular bridge *ladder* of one laser: one bridge per mirror
    /// the beam crosses, in beam order. Rung j is the aperture's mirror
    /// image unfolded across the first j mirrors of the beam's chain —
    /// the analytic description of the specular path "diffuse vertex →
    /// mirror₁ → … → mirrorⱼ → aperture". A diffuse vertex beside the
    /// beam's j-th leg sees the aperture through exactly those j
    /// mirrors, so rung j is the bridge that carries the glow on that
    /// leg: without it the light is unreachable (every BDPT connection
    /// from such a vertex is either through a delta mirror — invalid —
    /// or coplanar-grazing, G = 0 exactly), and the leg renders black.
    ///
    /// The rungs are free byproducts of the unfolding: the loop that
    /// traces the beam reflects the image across each mirror plane as
    /// it goes, and every intermediate image is a valid bridge for the
    /// corresponding prefix of the chain — the classic unfolding
    /// construction works for any prefix (a ray aimed at rung j's
    /// image disc crosses the mirrors j…1 in reverse order and lands
    /// on the aperture).
    ///
    /// Glass ends the ladder (a refracted ray's bend depends on where
    /// it crosses the interface, so no image disc describes the far
    /// side); the rungs before the glass remain valid and are kept.
    /// The ladder is empty when the beam never hits a mirror (the
    /// aperture is then directly visible and ordinary sampling finds
    /// it).
    ///
    /// Derived on demand, never stored: the scene may move between
    /// frames, and every rung must follow the geometry.
    pub fn bridges_of_light(
        &self,
        laser: &LightData,
        max_bounces: usize,
    ) -> Vec<crate::bdpt::Bridge> {
        debug_assert!(laser.shape == 2, "not a laser light");
        let (cpu_scene, _) = self.build_cpu_scene();
        self.bridges_of_light_in(laser, &*cpu_scene, max_bounces)
    }

    /// The specular bridge of one laser's *whole* mirror chain: the
    /// last rung of the ladder (`bridges_of_light`). `None` when the
    /// beam never hits a mirror.
    pub fn bridge_of_light(
        &self,
        laser: &LightData,
        max_bounces: usize,
    ) -> Option<crate::bdpt::Bridge> {
        self.bridges_of_light(laser, max_bounces).pop()
    }

    /// The specular bridges of every laser in the scene: each laser's
    /// whole ladder, concatenated — the eye walk's diffuse sampling
    /// mixture (see `bdpt::Bridge`). One BVH build shared across the
    /// lasers.
    pub fn bridges(&self) -> Vec<crate::bdpt::Bridge> {
        let (cpu_scene, _) = self.build_cpu_scene();
        self.lights
            .iter()
            .filter(|l| l.shape == 2)
            .flat_map(|l| self.bridges_of_light_in(l, &*cpu_scene, MAX_BEAM_BOUNCES))
            .collect()
    }

    /// `bridges_of_light` against an already-built BVH (the per-laser
    /// variant rebuilds it; this one shares it across all lasers).
    fn bridges_of_light_in(
        &self,
        laser: &LightData,
        cpu_scene: &dyn Intersectable,
        max_bounces: usize,
    ) -> Vec<crate::bdpt::Bridge> {
        debug_assert!(laser.shape == 2, "not a laser light");
        let mut origin = laser.center + laser.normal * EPS_NEAR;
        let mut dir = laser.normal;
        // The aperture's image, unfolded across the mirrors hit so far.
        let mut image = laser.center;
        let mut ladder = Vec::new();
        for _ in 0..max_bounces {
            let ray = Ray {
                origin,
                direction: dir,
            };
            let Some(hit) = cpu_scene.intersect(&ray) else {
                break; // escaped: no (further) mirror
            };
            // Orient the normal toward the side the beam arrives from.
            let front_face = hit.normal.dot(dir) < 0.0;
            let normal = if front_face {
                hit.normal
            } else {
                hit.normal * -1.0
            };
            match hit.mat.bsdf {
                Bsdf::Mirror => {
                    // Unfold: reflect the current image across this
                    // mirror plane. The plane is the perpendicular
                    // bisector of image ↔ reflected image, so a ray
                    // aimed at the reflected image bounces onto the
                    // pre-image — and chaining the reflections lands on
                    // the aperture itself. Every intermediate image is
                    // a valid bridge for the prefix of the chain so
                    // far: push it as a rung.
                    let d = image - hit.p;
                    image = image - normal * (2.0 * d.dot(normal));
                    ladder.push(crate::bdpt::Bridge {
                        image_center: image,
                        radius: laser.radius,
                    });
                    dir = reflect_dir(dir, normal);
                    origin = hit.p + normal * EPS_NEAR;
                }
                Bsdf::Glass { .. } => break, // see the doc comment
                _ => break, // diffuse (or the beam's own housing): the chain ends
            }
        }
        ladder
    }

    /// The highest luminance any physically valid sample can carry:
    /// the brightest emitter's luminance. Radiance along a path is
    /// bounded by the emitter's emission (albedo ≤ 1, MIS weights ≤ 1,
    /// the eta² transmission factor is compensated by the BSDF's
    /// 1/eta²), so a sample above this bound is a statistical outlier
    /// — a firefly — and clamping it toward the bound is the standard
    /// outlier-rejection trade (small bias on rare samples for a large
    /// variance cut).
    ///
    /// Shared by both backends: the CPU clamps in `bdpt_trace`, the GPU
    /// uploads it as a uniform so the WGSL clamp and the CPU clamp use
    /// the same threshold.
    pub fn max_light_luminance(&self) -> f32 {
        self.lights
            .iter()
            .map(|l| l.emission.luminance())
            .fold(0.0f32, f32::max)
    }

    /// Trace the laser beam through the scene: follow the aperture's
    /// axis ray across every specular (delta) surface it hits — mirrors
    /// reflect it, glass refracts it — until it reaches a non-delta
    /// surface (a diffuse wall: the spot) or escapes.
    ///
    /// This is the *analytic* transport of the beam's central ray. The
    /// beam is a narrow cone around it, so the spot is the cone's
    /// footprint on the final surface; the caller accounts for the
    /// spread (a decal's half-width from the path length and cone
    /// angle). The `throughput` is the beam's color power fraction
    /// across the delta chain: a mirror multiplies by its albedo, a
    /// glass interface by its albedo times the Fresnel transmittance
    /// (or reflectance, under total internal reflection) — so the
    /// power the beam deposits at the spot is
    /// `emission · A_aperture · Ω_cone · throughput`.
    ///
    /// `max_bounces` bounds the specular chain (a beam caught in a
    /// mirror loop must not spin forever).
    pub fn trace_laser_beam(&self, laser: &LightData, max_bounces: usize) -> BeamTrace {
        debug_assert!(laser.shape == 2, "not a laser light");
        let mut legs: Vec<(Vec3, Vec3)> = Vec::new();
        let (cpu_scene, _) = self.build_cpu_scene();
        // Start just off the aperture, along the beam axis.
        let mut origin = laser.center + laser.normal * EPS_NEAR;
        let mut dir = laser.normal;
        let mut throughput = Vec3::new(1.0, 1.0, 1.0);
        for _ in 0..=max_bounces {
            let ray = Ray {
                origin,
                direction: dir,
            };
            let Some(hit) = cpu_scene.intersect(&ray) else {
                // Escaped the scene (should not happen in a closed room).
                legs.push((origin, origin + dir));
                return BeamTrace { legs, spot: None };
            };
            legs.push((origin, hit.p));
            let mat = MaterialData {
                albedo: hit.mat.albedo,
                emission: hit.mat.emission,
                bsdf: hit.mat.bsdf,
            };
            // Orient the normal toward the side the beam arrives from
            // (mirrors reflect the same from both sides; glass needs the
            // orientation to pick its eta pair).
            let front_face = hit.normal.dot(dir) < 0.0;
            let normal = if front_face {
                hit.normal
            } else {
                hit.normal * -1.0
            };
            if !mat.bsdf.is_delta() {
                // A non-delta surface: the beam's final stop. The spot's
                // material is what a decal's emission derives from.
                return BeamTrace {
                    legs,
                    spot: Some(BeamSpot {
                        p: hit.p,
                        normal,
                        mat,
                        throughput,
                    }),
                };
            }
            // Delta surface: continue the specular chain deterministically,
            // bookkeeping the power fraction the traced branch carries.
            let cos_i = -normal.dot(dir);
            dir = match mat.bsdf {
                Bsdf::Mirror => {
                    throughput = throughput * mat.albedo;
                    reflect_dir(dir, normal)
                }
                Bsdf::Glass { ior } => {
                    let (eta_i, eta_t) = if front_face { (1.0, ior) } else { (ior, 1.0) };
                    let r = fresnel_dielectric(cos_i, eta_i, eta_t);
                    match refract_dir(dir, normal, eta_i, eta_t) {
                        Some(refracted) => {
                            // The refracted branch carries T = 1 - R of the
                            // power (the eta^2 factor is a radiance
                            // artifact; power transport sees only R + T = 1).
                            throughput = throughput * mat.albedo * (1.0 - r);
                            refracted
                        }
                        None => {
                            // Total internal reflection: the whole power
                            // reflects.
                            throughput = throughput * mat.albedo;
                            reflect_dir(dir, normal)
                        }
                    }
                }
                _ => unreachable!("only Mirror and Glass are delta BSDFs"),
            };
            // Offset along the side the new ray travels to (a refracted
            // ray leaves through the surface, so offsetting by the
            // arrival-side normal would start it inside the glass).
            let offset = if dir.dot(normal) > 0.0 {
                normal
            } else {
                normal * -1.0
            };
            origin = hit.p + offset * EPS_NEAR;
        }
        // Specular chain longer than the bound: the beam never reached a
        // diffuse surface within `max_bounces`.
        BeamTrace { legs, spot: None }
    }

    /// Build the trait-object scene (BVH) used by the CPU renderer.
    ///
    /// Emitter geometry is already part of `spheres`/`quads` (registered
    /// by `add_light`), so it is intersectable here without any special
    /// handling; `lights` provides the area-sampling interface.
    pub fn build_cpu_scene(&self) -> (Box<dyn Intersectable>, Vec<AreaLight>) {
        let mut objects: Vec<Box<dyn Intersectable>> = Vec::new();
        for s in &self.spheres {
            objects.push(Box::new(Sphere {
                center: s.center,
                radius: s.radius,
                mat: s.mat.to_material(),
            }));
        }
        for q in &self.quads {
            objects.push(Box::new(Quad {
                center: q.center,
                normal: q.normal,
                tangent: q.tangent,
                bitangent: q.bitangent,
                half_w: q.half_w,
                half_h: q.half_h,
                mat: q.mat.to_material(),
            }));
        }
        for c in &self.cubes {
            objects.push(Box::new(Cube {
                min: c.min,
                max: c.max,
                mat: c.mat.to_material(),
            }));
        }
        for p in &self.planes {
            objects.push(Box::new(Plane {
                point: p.point,
                normal: p.normal,
                mat: p.mat.to_material(),
            }));
        }
        for c in &self.cylinders {
            objects.push(Box::new(Cylinder {
                base: c.base,
                axis: c.axis,
                radius: c.radius,
                half_h: c.half_h,
                mat: c.mat.to_material(),
            }));
        }
        // Emitter geometry is already part of `spheres`/`quads`/`cylinders`
        // — `add_light` registers it — so it must not be added again here.
        let scene = crate::bvh::BVHNode::build(objects).expect("scene is non-empty");
        let lights = self.lights.iter().map(|l| self.cpu_light(l)).collect();
        (scene, lights)
    }

    fn cpu_light(&self, l: &LightData) -> AreaLight {
        if l.shape == 0 {
            AreaLight::from_sphere(&Sphere {
                center: l.center,
                radius: l.radius,
                mat: Material::emissive(l.emission),
            })
        } else if l.shape == 2 {
            AreaLight::from_laser(l.center, l.normal, l.radius, l.half_w, l.emission)
        } else {
            AreaLight::from_quad(&Quad {
                center: l.center,
                normal: l.normal,
                tangent: l.tangent,
                bitangent: l.bitangent,
                half_w: l.half_w,
                half_h: l.half_h,
                mat: Material::emissive(l.emission),
            })
        }
    }
}

impl Default for SceneData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_area_matches_geometry_definitions() {
        let s = LightData::from_sphere(Vec3::new(0.0, 2.0, 0.0), 0.5, Vec3::new(1.0, 1.0, 1.0));
        assert!((s.area() - 4.0 * std::f32::consts::PI * 0.25).abs() < 1e-5);
        let q = LightData::from_quad(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            4.0,
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert!((q.area() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_scene_conversion_preserves_light_count() {
        let mut scene = SceneData::new();
        scene.add_plane(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        );
        scene.add_light(LightData::from_sphere(
            Vec3::new(0.0, 2.0, 0.0),
            0.5,
            Vec3::new(5.0, 5.0, 5.0),
        ));
        let (cpu_scene, lights) = scene.build_cpu_scene();
        assert_eq!(lights.len(), 1);
        // A ray straight up from the floor must hit the light sphere.
        let ray = crate::geometry::Ray {
            origin: Vec3::new(0.0, 0.1, 0.0),
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        assert!(cpu_scene.intersect(&ray).is_some());
    }

    /// A wall added via `add_wall` must be a quad with a tight bounding
    /// box, not an infinite plane with a ±1e6 box: that is the whole
    /// point of the helper (BVH culling). It must also map `width` to
    /// the horizontal world axis and `height` to the vertical one — the
    /// original helper-axis tangent frame rotated walls (a 5-wide,
    /// 2-tall wall facing +z came out 2 wide and 5 tall).
    #[test]
    fn add_wall_is_a_finite_quad_with_tight_bbox() {
        let mut scene = SceneData::new();
        // A 5×2 wall facing +z: width spans x, height spans y.
        scene.add_wall(
            Vec3::new(0.0, 0.0, -2.5),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            2.0,
            MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        );
        assert_eq!(scene.planes.len(), 0);
        assert_eq!(scene.quads.len(), 1);
        let q = &scene.quads[0];
        assert_eq!(q.half_w, 2.5);
        assert_eq!(q.half_h, 1.0);
        // The quad's bbox must be tight: no ±1e6 extents.
        let (cpu_scene, _) = scene.build_cpu_scene();
        let bbox = cpu_scene.bounding_box();
        assert!(
            bbox.max.x - bbox.min.x <= 5.0 + 1e-4,
            "x extent: {:?}",
            bbox
        );
        assert!(
            bbox.max.y - bbox.min.y <= 2.0 + 1e-4,
            "y extent: {:?}",
            bbox
        );
        assert!(bbox.max.z - bbox.min.z <= 1e-4, "z extent: {:?}", bbox);
        // A ray through the wall's center hits it...
        let hit = crate::geometry::Ray {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
        };
        assert!(cpu_scene.intersect(&hit).is_some());
        // ...and rays outside its extents miss (a plane would hit).
        let above = crate::geometry::Ray {
            origin: Vec3::new(0.0, 3.0, 0.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
        };
        assert!(cpu_scene.intersect(&above).is_none());
        let beside = crate::geometry::Ray {
            origin: Vec3::new(3.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
        };
        assert!(cpu_scene.intersect(&beside).is_none());
    }

    /// A side wall (normal along x) must span z horizontally and y
    /// vertically — the axis mapping differs from the back/front walls.
    #[test]
    fn add_wall_side_wall_spans_z_and_y() {
        let mut scene = SceneData::new();
        scene.add_wall(
            Vec3::new(-2.5, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            5.0,
            2.0,
            MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        );
        let q = &scene.quads[0];
        assert_eq!(q.half_w, 2.5);
        assert_eq!(q.half_h, 1.0);
        let (cpu_scene, _) = scene.build_cpu_scene();
        let bbox = cpu_scene.bounding_box();
        assert!(
            bbox.max.z - bbox.min.z <= 5.0 + 1e-4,
            "z extent: {:?}",
            bbox
        );
        assert!(
            bbox.max.y - bbox.min.y <= 2.0 + 1e-4,
            "y extent: {:?}",
            bbox
        );
        assert!(bbox.max.x - bbox.min.x <= 1e-4, "x extent: {:?}", bbox);
    }

    /// A cylinder added via `add_cylinder` must be a regular scene
    /// object: real geometry on both backends (the GPU uploads the
    /// `cylinders` array like any other primitive family), with the
    /// flat form matching `geometry::Cylinder::new` (base at
    /// center - axis*height/2, unit axis, half_h = height/2) — and
    /// *not* a light: only `add_light` registers sampleable emitters.
    #[test]
    fn add_cylinder_is_a_regular_scene_object() {
        let mut scene = SceneData::new();
        scene.add_cylinder(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0), // non-unit on purpose: must normalize
            2.0,
            0.5,
            MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        );
        assert_eq!(scene.cylinders.len(), 1);
        assert!(scene.lights.is_empty());
        let c = &scene.cylinders[0];
        assert!((c.base - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-6);
        assert!((c.axis - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert!((c.half_h - 1.0).abs() < 1e-6);
        assert!((c.radius - 0.5).abs() < 1e-6);
        // Intersectable on the CPU: the curved side and the top cap.
        let (cpu_scene, lights) = scene.build_cpu_scene();
        assert!(lights.is_empty());
        let side = crate::geometry::Ray {
            origin: Vec3::new(2.0, 1.0, 0.0),
            direction: Vec3::new(-1.0, 0.0, 0.0),
        };
        let hit = cpu_scene
            .intersect(&side)
            .expect("the curved side must be intersectable");
        assert!((hit.p - Vec3::new(0.5, 1.0, 0.0)).length() < 1e-4);
        let cap = crate::geometry::Ray {
            origin: Vec3::new(0.0, 3.0, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
        };
        let hit = cpu_scene
            .intersect(&cap)
            .expect("the top cap must be intersectable");
        assert!((hit.p - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4);
    }

    /// A box added via `add_box` must be a regular scene object with
    /// independent per-axis extents — the whole point over `add_cube`
    /// (which forces equal extents) and over `add_wall` (a one-sided
    /// quad: it shades only from its normal direction, so a wall used
    /// as a table top is black from below). A box must shade from all
    /// six sides, each face with its own outward normal.
    #[test]
    fn add_box_is_a_regular_scene_object() {
        let mut scene = SceneData::new();
        // A 4 x 1 x 2 box centered at (0, 0.5, 0): a flat slab.
        scene.add_box(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(4.0, 1.0, 2.0),
            MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        );
        assert_eq!(scene.cubes.len(), 1);
        assert!(scene.lights.is_empty());
        let c = &scene.cubes[0];
        assert_eq!(c.min, Vec3::new(-2.0, 0.0, -1.0));
        assert_eq!(c.max, Vec3::new(2.0, 1.0, 1.0));
        // The bbox must be tight (a box, not a ±1e6 plane box).
        let (cpu_scene, lights) = scene.build_cpu_scene();
        assert!(lights.is_empty());
        let bbox = cpu_scene.bounding_box();
        assert_eq!(bbox.min, Vec3::new(-2.0, 0.0, -1.0));
        assert_eq!(bbox.max, Vec3::new(2.0, 1.0, 1.0));

        // All six faces shade, each with its own outward normal.
        let faces = [
            // (ray origin, expected hit point, expected normal)
            (
                Vec3::new(0.0, 3.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            (
                Vec3::new(0.0, -3.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
            ),
            (
                Vec3::new(3.0, 0.5, 0.0),
                Vec3::new(2.0, 0.5, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            ),
            (
                Vec3::new(-3.0, 0.5, 0.0),
                Vec3::new(-2.0, 0.5, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
            ),
            (
                Vec3::new(0.0, 0.5, 3.0),
                Vec3::new(0.0, 0.5, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ),
            (
                Vec3::new(0.0, 0.5, -3.0),
                Vec3::new(0.0, 0.5, -1.0),
                Vec3::new(0.0, 0.0, -1.0),
            ),
        ];
        for (origin, expected_p, expected_n) in faces {
            let ray = crate::geometry::Ray {
                origin,
                direction: (Vec3::new(0.0, 0.5, 0.0) - origin).normalize(),
            };
            let hit = cpu_scene
                .intersect(&ray)
                .expect("every face of the box must be intersectable");
            assert!(
                (hit.p - expected_p).length() < 1e-4,
                "hit {:?}, expected {:?}",
                hit.p,
                expected_p
            );
            assert!(
                (hit.normal - expected_n).length() < 1e-4,
                "normal {:?}, expected {:?}",
                hit.normal,
                expected_n
            );
        }

        // Rays outside the box's extents miss (a plane would hit).
        let beside = crate::geometry::Ray {
            origin: Vec3::new(3.0, 3.0, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
        };
        assert!(cpu_scene.intersect(&beside).is_none());

        // `add_box` with equal extents equals `add_cube`.
        let mut a = SceneData::new();
        a.add_box(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.6, 0.6, 0.6),
            MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        );
        let mut b = SceneData::new();
        b.add_cube(
            Vec3::new(1.0, 2.0, 3.0),
            0.6,
            MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        );
        assert_eq!(a.cubes[0].min, b.cubes[0].min);
        assert_eq!(a.cubes[0].max, b.cubes[0].max);
    }

    /// A light added *only* via `add_light` must be real geometry: the GPU
    /// uploads `spheres`/`quads` and its `intersect_scene` never reads the
    /// light buffer, so a light missing from the geometry arrays would be
    /// invisible to eye subpaths on the GPU while visible on the CPU.
    #[test]
    fn add_light_registers_geometry_for_both_backends() {
        let mut scene = SceneData::new();
        scene.add_light(LightData::from_sphere(
            Vec3::new(0.0, 2.0, 0.0),
            0.5,
            Vec3::new(5.0, 5.0, 5.0),
        ));
        // Registered exactly once as geometry (no duplicate registration).
        assert_eq!(scene.spheres.len(), 1);
        assert!(scene.spheres[0].mat.is_emissive());
        assert_eq!(scene.spheres[0].mat.emission, Vec3::new(5.0, 5.0, 5.0));
        // ...and the CPU BVH hits it with the emissive material.
        let (cpu_scene, lights) = scene.build_cpu_scene();
        assert_eq!(lights.len(), 1);
        let ray = crate::geometry::Ray {
            origin: Vec3::new(0.0, 0.1, 0.0),
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        let hit = cpu_scene
            .intersect(&ray)
            .expect("light must be intersectable");
        assert!(hit.mat.is_emissive());

        // Same for quad lights.
        let mut scene = SceneData::new();
        scene.add_light(LightData::from_quad(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            1.0,
            1.0,
            Vec3::new(5.0, 5.0, 5.0),
        ));
        assert_eq!(scene.quads.len(), 1);
        assert!(scene.quads[0].mat.is_emissive());
    }

    /// A small closed room with a laser and a mirror chain, used by the
    /// bridge tests below.
    fn laser_room() -> SceneData {
        let mut scene = SceneData::new();
        let white = MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8));
        // A 2 x 2 x 2 room.
        scene.add_wall(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(0.0, 1.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, -1.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 2.0, 2.0, white);
        scene.add_wall(Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 2.0, 2.0, white);
        scene
    }

    /// The specular bridge must be the aperture's mirror image through
    /// the beam's *whole* mirror chain — not just the first mirror.
    /// A beam aimed at a 45° floor mirror that bounces onto a 45° wall
    /// mirror must unfold across both planes, in beam order.
    #[test]
    fn bridge_unfolds_across_the_whole_mirror_chain() {
        let mut scene = laser_room();
        // The laser first (order independence: the bridge is derived
        // from the fully-built scene, never baked at add-time).
        scene.add_laser(
            Vec3::new(0.4, 1.9, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.2,
            0.01,
            0.01,
            Vec3::new(1.0, 1.0, 1.0),
        );
        // Mirror 1: the floor, tilted 45° — the downward beam reflects
        // horizontal (+z).
        scene.add_quad(
            Vec3::new(0.4, 0.002, 0.0),
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
            0.6,
            0.6,
            MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
        );
        // Mirror 2: the front wall (z = 1), tilted 45° the other way —
        // the horizontal beam reflects straight up, landing on the
        // ceiling far from the laser housing.
        scene.add_quad(
            Vec3::new(0.4, 0.0027, 0.998),
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),
            0.6,
            0.6,
            MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
        );

        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("the laser");
        let trace = scene.trace_laser_beam(laser, 4);
        // Three legs: aperture -> floor mirror -> wall mirror -> spot.
        assert_eq!(
            trace.legs.len(),
            3,
            "the beam must bounce twice, got {:?}",
            trace.legs
        );
        let spot = trace.spot.expect("the beam must land on a diffuse surface");
        // The beam ends going straight up: it lands on the ceiling.
        assert!(
            (spot.p.y - 2.0).abs() < 2e-3,
            "the beam must land on the ceiling, got {:?}",
            spot.p
        );

        // The bridge: the aperture unfolded across both mirrors.
        let bridge = scene
            .bridge_of_light(laser, 4)
            .expect("the laser must carry a specular bridge");
        let aperture = laser.center;
        let floor_hit = trace.legs[0].1;
        let wall_hit = trace.legs[1].1;
        let unfold = |p: Vec3, hit: Vec3, normal: Vec3| {
            let d = p - hit;
            p - normal * (2.0 * d.dot(normal))
        };
        let image1 = unfold(
            aperture,
            floor_hit,
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
        );
        let expected = unfold(
            image1,
            wall_hit,
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),
        );
        assert!(
            (bridge.image_center - expected).length() < 1e-3,
            "the bridge image {:?} must be the double reflection {:?}",
            bridge.image_center,
            expected
        );
        assert!((bridge.radius - laser.radius).abs() < 1e-6);
    }

    /// The bridge must follow a *moved* mirror: the same scene with the
    /// wall mirror shifted must produce a different (correct) image.
    /// This is the movement-readiness guard — the bridge is derived
    /// per query, never stored.
    #[test]
    fn bridge_follows_a_moved_mirror() {
        let build = |wall_z: f32| -> (SceneData, crate::bdpt::Bridge) {
            let mut scene = laser_room();
            scene.add_laser(
                Vec3::new(0.4, 1.9, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                0.2,
                0.01,
                0.01,
                Vec3::new(1.0, 1.0, 1.0),
            );
            scene.add_quad(
                Vec3::new(0.4, 0.002, 0.0),
                Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
                0.6,
                0.6,
                MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
            );
            scene.add_quad(
                Vec3::new(0.4, 0.0027, wall_z),
                Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),
                0.6,
                0.6,
                MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
            );
            let laser = scene
                .lights
                .iter()
                .find(|l| l.shape == 2)
                .expect("the laser");
            let bridge = scene
                .bridge_of_light(laser, 4)
                .expect("the beam must hit both mirrors");
            (scene, bridge)
        };

        let (_, bridge_a) = build(0.998);
        let (_, bridge_b) = build(0.5);
        // The wall mirror moved 0.5 toward the room center; the image
        // must move with it (the second unfolding plane changed).
        assert!(
            (bridge_a.image_center - bridge_b.image_center).length() > 0.1,
            "the bridge must follow the moved mirror: {:?} vs {:?}",
            bridge_a.image_center,
            bridge_b.image_center
        );
        // And each image is the correct double reflection for its own
        // geometry: the unfolding preserves path length, so the spot's
        // distance to the image equals the total beam path length (up
        // to the per-bounce EPS_NEAR ray offsets).
        for (scene, bridge) in [build(0.998), build(0.5)] {
            let laser = scene
                .lights
                .iter()
                .find(|l| l.shape == 2)
                .expect("the laser");
            let trace = scene.trace_laser_beam(laser, 4);
            let spot = trace.spot.expect("spot");
            let path_len: f32 = trace
                .legs
                .iter()
                .map(|(a, b)| (*b - *a).length())
                .sum();
            let d = (bridge.image_center - spot.p).length();
            assert!(
                (d - path_len).abs() < 0.01,
                "the image distance {d} must equal the beam path length {path_len}"
            );
        }
    }

    /// A beam that never hits a mirror must produce no bridge: the
    /// aperture is directly visible and ordinary sampling finds it.
    #[test]
    fn beam_without_mirrors_has_no_bridge() {
        let mut scene = laser_room();
        scene.add_laser(
            Vec3::new(0.0, 1.9, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.2,
            0.01,
            0.01,
            Vec3::new(1.0, 1.0, 1.0),
        );
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("the laser");
        assert!(
            scene.bridge_of_light(laser, 4).is_none(),
            "a beam hitting only the diffuse floor must have no bridge"
        );
        assert!(
            scene.bridges_of_light(laser, 4).is_empty(),
            "a beam hitting only the diffuse floor must have no ladder"
        );
    }

    /// The bridge *ladder*: one rung per mirror, and rung j must be the
    /// correct bridge for the beam's j-th leg — the aperture's image
    /// unfolded across exactly the first j mirrors. A diffuse vertex
    /// beside leg j+1 sees the aperture through exactly those j
    /// mirrors, so rung j is what carries the glow there. This is the
    /// per-bounce generalization of the whole-chain test above.
    #[test]
    fn bridge_ladder_has_one_rung_per_mirror_in_beam_order() {
        let mut scene = laser_room();
        scene.add_laser(
            Vec3::new(0.4, 1.9, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.2,
            0.01,
            0.01,
            Vec3::new(1.0, 1.0, 1.0),
        );
        scene.add_quad(
            Vec3::new(0.4, 0.002, 0.0),
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),
            0.6,
            0.6,
            MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
        );
        scene.add_quad(
            Vec3::new(0.4, 0.0027, 0.998),
            Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),
            0.6,
            0.6,
            MaterialData::mirror(Vec3::new(1.0, 1.0, 1.0)),
        );
        let laser = scene
            .lights
            .iter()
            .find(|l| l.shape == 2)
            .expect("the laser");
        let trace = scene.trace_laser_beam(laser, 4);
        assert_eq!(trace.legs.len(), 3, "two bounces");

        let ladder = scene.bridges_of_light(laser, 4);
        assert_eq!(ladder.len(), 2, "one rung per mirror");
        // The last rung is the whole-chain bridge (the old contract).
        let whole = scene
            .bridge_of_light(laser, 4)
            .expect("the whole-chain bridge");
        assert!(
            (ladder[1].image_center - whole.image_center).length() < 1e-6,
            "the last rung must be the whole-chain bridge"
        );

        // Rung j is the aperture unfolded across exactly the first j
        // mirrors, in beam order. The unfolding preserves path length,
        // so the distance from the *end of leg j+1* (the point that sees
        // the aperture through mirrors 1..j) to rung j's image equals
        // the length of the specular path from there to the aperture:
        // legs j+1..1 plus the aperture leg.
        let unfold = |p: Vec3, hit: Vec3, normal: Vec3| {
            let d = p - hit;
            p - normal * (2.0 * d.dot(normal))
        };
        let aperture = laser.center;
        let floor_hit = trace.legs[0].1;
        let wall_hit = trace.legs[1].1;
        let floor_n = Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2);
        let wall_n = Vec3::new(0.0, std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2);
        // Rung 0: the aperture reflected across the floor mirror only.
        let expected0 = unfold(aperture, floor_hit, floor_n);
        assert!(
            (ladder[0].image_center - expected0).length() < 1e-3,
            "rung 0 {:?} must be the single reflection {:?}",
            ladder[0].image_center,
            expected0
        );
        // Rung 1: that image reflected across the wall mirror.
        let expected1 = unfold(expected0, wall_hit, wall_n);
        assert!(
            (ladder[1].image_center - expected1).length() < 1e-3,
            "rung 1 {:?} must be the double reflection {:?}",
            ladder[1].image_center,
            expected1
        );

        // The per-leg usability invariant: a diffuse vertex beside leg
        // j+1 aims at rung j. The vertex is the end of leg j+1 offset
        // perpendicular to the beam, on the surface that leg runs
        // along, and the image must lie in that surface's hemisphere.
        // Leg 2 runs along the floor: a floor point beside the beam
        // (offset in x) sees the aperture through mirror 1 only.
        let leg2_end = trace.legs[1].1; // the wall-mirror hit
        let beside_leg2 = Vec3::new(leg2_end.x + 0.05, 0.0, leg2_end.z * 0.5);
        let d0 = ladder[0].image_center - beside_leg2;
        assert!(
            d0.dot(Vec3::new(0.0, 1.0, 0.0)) > 0.0,
            "rung 0 must point out of the floor beside leg 2"
        );
        // Leg 3 runs from the front-wall mirror up to the ceiling: a
        // ceiling point beside the beam sees the aperture through both
        // mirrors.
        let leg3_end = trace.legs[2].1; // the spot on the ceiling
        let beside_leg3 = Vec3::new(leg3_end.x + 0.05, leg3_end.y, leg3_end.z - 0.05);
        let d1 = ladder[1].image_center - beside_leg3;
        assert!(
            d1.dot(Vec3::new(0.0, -1.0, 0.0)) > 0.0,
            "rung 1 must point out of the ceiling beside leg 3"
        );
    }
}
