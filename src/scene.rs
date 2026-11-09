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
use crate::spectrum::{SpectrumKind, BK7};

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

/// A surface material in flat, GPU-friendly form. `spectrum != None`
/// marks a pure emitter: the BSDF is ignored and the surface does not
/// scatter. The emission is spectral (paper §9.2) — a smooth SPD or a
/// monochromatic line — evaluated at the path's hero wavelength.
#[derive(Clone, Copy, Debug)]
pub struct MaterialData {
    pub albedo: Vec3,
    pub spectrum: SpectrumKind,
    pub bsdf: Bsdf,
}

impl MaterialData {
    pub fn diffuse(albedo: Vec3) -> Self {
        Self {
            albedo,
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Diffuse,
        }
    }

    #[allow(dead_code)]
    pub fn emissive(emission: Vec3) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            spectrum: SpectrumKind::smooth_from_rgb(emission),
            bsdf: Bsdf::Diffuse,
        }
    }

    /// An emitter with an explicit spectrum (a laser line, for instance).
    pub fn emissive_spec(spectrum: SpectrumKind) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            spectrum,
            bsdf: Bsdf::Diffuse,
        }
    }

    /// Attach the laser emission cone to an emissive material: the BSDF
    /// slot is the one place the cone can travel with the material so
    /// the BDPT pdf/weight/gate helpers can read it from any path
    /// vertex (see `Bsdf::Laser`). The cone is stored as its versine
    /// (see `Bsdf::Laser`).
    #[allow(dead_code)]
    pub fn with_laser_bsdf(mut self, theta: f32) -> Self {
        self.bsdf = Bsdf::Laser {
            versine: crate::geometry::cone_versine(theta),
        };
        self
    }

    #[allow(dead_code)]
    pub fn mirror(albedo: Vec3) -> Self {
        Self {
            albedo,
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Mirror,
        }
    }

    #[allow(dead_code)]
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

    /// Dispersive glass from Sellmeier coefficients (paper §9.2).
    pub fn glass_sellmeier(ior: f32, sellmeier: [f32; 6]) -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Glass { ior, sellmeier },
        }
    }

    /// N-BK7 borosilicate glass — the demo's dispersive glass cube.
    pub fn glass_bk7() -> Self {
        Self::glass_sellmeier(1.5168, BK7)
    }

    /// A GGX microfacet "rough mirror" (paper §9.5): `alpha` is the
    /// roughness, `f0` the normal-incidence reflectance.
    pub fn glossy(albedo: Vec3, alpha: f32, f0: f32) -> Self {
        Self {
            albedo,
            spectrum: SpectrumKind::None,
            bsdf: Bsdf::Glossy { alpha, f0 },
        }
    }

    /// A cone-gated emitter with a smooth spectrum (an RGB spotlight).
    /// The cone is stored as its versine (see `Bsdf::Laser`).
    #[allow(dead_code)]
    pub fn laser(emission: Vec3, theta: f32) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            spectrum: SpectrumKind::smooth_from_rgb(emission),
            bsdf: Bsdf::Laser {
                versine: crate::geometry::cone_versine(theta),
            },
        }
    }

    /// A *laser*: a monochromatic line emitter, cone-gated (paper §9.2:
    /// "a laser is monochromatic, so λ is deterministic"). All `power`
    /// (spectral radiance) radiates at `lambda` (nm) inside the cone of
    /// half-angle `theta`.
    #[allow(dead_code)]
    pub fn laser_line(lambda: f32, power: f32, theta: f32) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            spectrum: SpectrumKind::Line { lambda, power },
            bsdf: Bsdf::Laser {
                versine: crate::geometry::cone_versine(theta),
            },
        }
    }

    /// True when this material radiates into a cone (a laser aperture).
    pub fn is_laser(&self) -> bool {
        matches!(self.bsdf, Bsdf::Laser { .. })
    }

    /// Pure emitters do not scatter; the BSDF is ignored for them.
    pub fn is_emissive(&self) -> bool {
        self.spectrum.is_emissive()
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
            spectrum: self.spectrum,
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
    /// Quad half-width (unused for spheres); for a laser the cone
    /// half-angle rides in this slot.
    pub half_w: f32,
    /// Quad half-height (unused for spheres); for a laser the housing
    /// height rides in this slot.
    pub half_h: f32,
    /// The emission spectrum (paper §9.2): a smooth SPD or a
    /// monochromatic line, evaluated at the path's hero wavelength.
    pub spectrum: SpectrumKind,
}

impl LightData {
    #[allow(dead_code)]
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
            spectrum: SpectrumKind::smooth_from_rgb(emission),
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
            spectrum: SpectrumKind::smooth_from_rgb(emission),
        }
    }

    /// A laser aperture light: a disc of `radius` centered at `center`
    /// with outward normal `normal`, radiating only into the cone of
    /// half-angle `theta` around the normal. The disc is the emitting
    /// cap of the laser's cylinder housing (see `SceneData::add_laser`);
    /// `height` is the housing's total height, carried so `add_light`
    /// can register the housing as geometry.
    #[allow(dead_code)]
    pub fn from_laser(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        theta: f32,
        height: f32,
        emission: Vec3,
    ) -> Self {
        Self::from_laser_spec(
            center,
            normal,
            radius,
            theta,
            height,
            SpectrumKind::smooth_from_rgb(emission),
        )
    }

    /// A *laser*: a monochromatic line emitter, cone-gated (paper §9.2).
    /// All `power` (spectral radiance) radiates at `lambda` (nm) inside
    /// the cone of half-angle `theta`.
    #[allow(dead_code)]
    pub fn from_laser_line(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        theta: f32,
        height: f32,
        lambda: f32,
        power: f32,
    ) -> Self {
        Self::from_laser_spec(
            center,
            normal,
            radius,
            theta,
            height,
            SpectrumKind::Line { lambda, power },
        )
    }

    /// A *laser beam*, power-parametrized (the physically correct model —
    /// paper §9.2's "the laser is a collimated beam"): specify the beam
    /// power Φ (W) and the divergence half-angle θ, and the line's
    /// spectral radiance is derived as
    /// `P = Φ / (A_ap · Ω(θ))` with `A_ap = π·radius²` the aperture
    /// area and `Ω(θ) = 2π(1 − cos θ)` the cone's solid angle. The spot
    /// irradiance `E = Φ/(π·w₀²)` is then *invariant to θ* — θ controls
    /// only the divergence, as it physically does. (The old
    /// radiance-parametrized form coupled them: at fixed Le the power
    /// Φ ∝ Ω, so θ was secretly a brightness knob — which is why it was
    /// left artificially wide.)
    ///
    /// `theta` is the beam's *physical* divergence: a diffraction-
    /// limited beam has θ = M²·λ/(π·w₀) with w₀ = `radius` (M² = 1 is
    /// the ideal Gaussian; real pointers sit at M² ≈ 1–3, and the demo
    /// uses a larger M² so the spot grows visibly across the room).
    /// The versine is computed in f64 (see `cone_versine`): at these
    /// divergences the f32 cosine rounds to 1.0 and the cone would
    /// collapse.
    pub fn from_laser_beam(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        theta: f32,
        height: f32,
        lambda: f32,
        power_w: f32,
    ) -> Self {
        let versine = crate::geometry::cone_versine(theta);
        let omega = 2.0 * std::f32::consts::PI * versine;
        let aperture_area = std::f32::consts::PI * radius * radius;
        let power = power_w / (aperture_area * omega);
        Self::from_laser_spec_versine(
            center,
            normal,
            radius,
            versine,
            height,
            SpectrumKind::Line {
                lambda,
                power,
            },
        )
    }

    /// The shared laser-aperture constructor: the spectrum rides in the
    /// `spectrum` field; the cone half-angle and housing height ride in
    /// the unused quad slots (`half_w`/`half_h`).
    fn from_laser_spec(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        theta: f32,
        height: f32,
        spectrum: SpectrumKind,
    ) -> Self {
        Self::from_laser_spec_versine(
            center,
            normal,
            radius,
            crate::geometry::cone_versine(theta),
            height,
            spectrum,
        )
    }

    /// The shared laser-aperture constructor in versine form (see
    /// `Bsdf::Laser` for why the cone travels as its versine): the
    /// spectrum rides in the `spectrum` field; the cone versine and
    /// housing height ride in the unused quad slots (`half_w`/`half_h`).
    fn from_laser_spec_versine(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        versine: f32,
        height: f32,
        spectrum: SpectrumKind,
    ) -> Self {
        Self {
            shape: 2,
            center,
            radius,
            normal: normal.normalize(),
            tangent: Vec3::new(0.0, 0.0, 0.0),
            bitangent: Vec3::new(0.0, 0.0, 0.0),
            half_w: versine, // the cone versine rides in the unused slot
            half_h: height,  // the housing height rides in the unused slot
            spectrum,
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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    /// Add a simple table: a top slab plus four legs, all of `mat`.
    /// `center` is the center of the top slab, `size` its full extent
    /// (x width, y thickness, z depth). The legs are square, 10% of
    /// the smaller horizontal extent, inset 10% from the edges, and
    /// reach down to the floor (y = 0) — the convention the demo
    /// room's hand-built table uses.
    ///
    /// Like `add_box`/`add_cylinder`: composed of boxes, so both
    /// backends see it without any changes.
    #[allow(dead_code)]
    pub fn add_table(&mut self, center: Vec3, size: Vec3, top: MaterialData, leg: MaterialData) {
        let dim = Vec3::new(size.x / 2.0, 0.02, size.z / 2.0);
        self.add_box(center, size, top);
        self.add_box(Vec3::new(center.x -  dim.x + 0.1, center.y / 2.0, center.z - dim.z + 0.1), Vec3::new(0.01, center.y, 0.01), leg);
        self.add_box(Vec3::new(center.x +  dim.x - 0.1, center.y / 2.0, center.z - dim.z + 0.1), Vec3::new(0.01, center.y, 0.01), leg);
        self.add_box(Vec3::new(center.x -  dim.x + 0.1, center.y / 2.0, center.z + dim.z - 0.1), Vec3::new(0.01, center.y, 0.01), leg);
        self.add_box(Vec3::new(center.x +  dim.x - 0.1, center.y / 2.0, center.z + dim.z - 0.1), Vec3::new(0.01, center.y, 0.01), leg);
    }

    /// Add a simple chair: a seat, four legs and a backrest, all of
    /// `mat`. `center` is the center of the seat; the chair faces +z
    /// (the backrest sits at the -z edge). Fixed ~0.45 m proportions,
    /// legs reaching the floor (y = 0).
    ///
    /// Like `add_box`/`add_cylinder`: composed of boxes, so both
    /// backends see it without any changes.
    #[allow(dead_code)]
    pub fn add_chair(&mut self, center: Vec3, mat: MaterialData) {
        const SEAT: f32 = 0.45; // seat width/depth
        const SEAT_T: f32 = 0.05; // seat thickness
        const LEG: f32 = 0.04; // leg thickness
        const BACK_H: f32 = 0.5; // backrest height above the seat
        const BACK_T: f32 = 0.04; // backrest thickness
        self.add_box(center, Vec3::new(SEAT, SEAT_T, SEAT), mat);
        let leg_h = center.y - SEAT_T * 0.5;
        for sx in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                self.add_box(
                    Vec3::new(
                        center.x + sx * (SEAT * 0.5 - LEG),
                        leg_h * 0.5,
                        center.z + sz * (SEAT * 0.5 - LEG),
                    ),
                    Vec3::new(LEG, leg_h, LEG),
                    mat,
                );
            }
        }
        // Backrest, rising from the back edge of the seat.
        self.add_box(
            Vec3::new(
                center.x,
                center.y + SEAT_T * 0.5 + BACK_H * 0.5,
                center.z - SEAT * 0.5 + BACK_T * 0.5,
            ),
            Vec3::new(SEAT, BACK_H, BACK_T),
            mat,
        );
    }

    /// Add a simple shelf: two side panels and four boards, all of
    /// `mat`. `center` is the shelf's center, `size` its full extent
    /// (x width, y height, z depth). The panels are 5% of the width
    /// thick; the boards are 4% of the height thick, span between the
    /// panels and are evenly spaced.
    ///
    /// Unlike `add_table`/`add_chair` (whose legs are derived down to
    /// the floor), the shelf is placed exactly at `center`/`size` —
    /// like `add_box` — so it can also hang on a wall; to stand it on
    /// the floor, pass `center.y = size.y / 2`.
    ///
    /// Like `add_box`/`add_cylinder`: composed of boxes, so both
    /// backends see it without any changes.
    #[allow(dead_code)]
    pub fn add_shelf(&mut self, center: Vec3, size: Vec3, mat: MaterialData, leg: MaterialData) {
        let dim = Vec3::new(size.x / 2.0, size.y / 2.0, size.z / 2.0);
        self.add_box(Vec3::new(center.x -  dim.x, center.y + dim.y, center.z - dim.z), Vec3::new(0.01, size.y, 0.01), leg);
        self.add_box(Vec3::new(center.x +  dim.x, center.y + dim.y, center.z - dim.z), Vec3::new(0.01, size.y, 0.01), leg);
        self.add_box(Vec3::new(center.x -  dim.x, center.y + dim.y, center.z + dim.z), Vec3::new(0.01, size.y, 0.01), leg);
        self.add_box(Vec3::new(center.x +  dim.x, center.y + dim.y, center.z + dim.z), Vec3::new(0.01, size.y, 0.01), leg);
        let step_size = size.y / 4.0;
        for i in 1..5 {
            let height = i as f32 * step_size + center.y; 
            self.add_box(
                Vec3::new(center.x, height, center.z),
                Vec3::new(size.x, 0.02, size.z),
                mat,
            );
        }
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
        let mat = MaterialData::emissive_spec(light.spectrum);
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
                mat: MaterialData::emissive_spec(light.spectrum).with_laser_bsdf(theta),
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
    ///
    /// Radiance-parametrized (the smooth-spectrum form the synthetic
    /// test scenes use); the demo and any physical scene use
    /// `add_laser_beam`, the power-parametrized form.
    #[allow(dead_code)]
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

    /// Add a *laser*: a monochromatic line emitter (paper §9.2) — the
    /// physically correct replacement for the RGB channel-stuffing hack.
    /// All `power` (spectral radiance) radiates at `lambda` (nm) inside
    /// the cone of half-angle `theta` around the beam axis.
    ///
    /// Radiance-parametrized (the analytic tests use it to pin exact
    /// rendered values); the demo and any physical scene use
    /// `add_laser_beam`, the power-parametrized form.
    #[allow(dead_code)]
    pub fn add_laser_line(
        &mut self,
        center: Vec3,
        axis: Vec3,
        height: f32,
        radius: f32,
        theta: f32,
        lambda: f32,
        power: f32,
    ) {
        let axis = axis.normalize();
        let aperture = center + axis * (height * 0.5);
        let light = LightData::from_laser_line(
            aperture, axis, radius, theta, height, lambda, power,
        );
        self.add_light(light);
    }

    /// Add a *laser beam*, power-parametrized (the physically correct
    /// laser model — see `LightData::from_laser_beam`): a cylinder
    /// housing whose top cap is the aperture, emitting a beam of
    /// `power_w` watts along `axis` into a cone of half-angle `theta`.
    /// The spot irradiance is Φ/(π·w₀²), invariant to θ.
    pub fn add_laser_beam(
        &mut self,
        center: Vec3,
        axis: Vec3,
        height: f32,
        radius: f32,
        theta: f32,
        lambda: f32,
        power_w: f32,
    ) {
        let axis = axis.normalize();
        let aperture = center + axis * (height * 0.5);
        let light = LightData::from_laser_beam(
            aperture, axis, radius, theta, height, lambda, power_w,
        );
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
    ///
    /// Each rung carries the beam direction and the emission cone's
    /// versine alongside the image disc: the unfolding is an isometry,
    /// so it maps the beam axis to the reflected travel direction at
    /// the image, and the emission cone to the cone of half-angle θ
    /// around it — the *gate-mapped* cone the rung samples (see
    /// `bdpt::Bridge`). Without it, at a laser's physical divergence
    /// the rung would sample the full image-disc cone while the
    /// aperture's cone gate accepts only a (Dθ/r)² fraction of the
    /// aims — the spot would converge ~(Dθ/r)² times slower and fall
    /// below the accumulation layer's outlier-rejection threshold,
    /// never appearing.
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
        // The emission cone's versine (the versine — not the cosine —
        // is the canonical form; see `Bsdf::Laser`).
        let versine = laser.half_w;
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
                Bsdf::Mirror | Bsdf::Glossy { .. } => {
                    // Unfold: reflect the current image across this
                    // mirror plane. The plane is the perpendicular
                    // bisector of image ↔ reflected image, so a ray
                    // aimed at the reflected image bounces onto the
                    // pre-image — and chaining the reflections lands on
                    // the aperture itself. Every intermediate image is
                    // a valid bridge for the prefix of the chain so
                    // far: push it as a rung.
                    //
                    // A glossy mirror crosses like a perfect one: the
                    // beam's central ray follows the lobe's mean (the
                    // specular direction), and the unfolding is exact
                    // for that ray. The rung stays the *exact* image
                    // disc — as a sampling technique it remains unbiased
                    // regardless of what happens downstream; the eye
                    // walk aims at the rung, hits the glossy surface,
                    // and the GGX draw there must be near-specular to
                    // continue. That is precisely §9.5's "the bridge is
                    // the small-roughness limit of the glossy
                    // connection": the technique helps exactly where
                    // the lobe is narrow.
                    let d = image - hit.p;
                    image = image - normal * (2.0 * d.dot(normal));
                    dir = reflect_dir(dir, normal);
                    // The gate-mapped cone: the unfolded beam axis at
                    // the image is the *reflected* travel direction —
                    // a ray from the surface aimed inside the cone
                    // around it crosses the mirrors and arrives at the
                    // aperture inside the emission cone. (The image
                    // sits behind the mirror, so the axis points from
                    // the image toward the surface — the direction the
                    // eye walk aims along.)
                    ladder.push(crate::bdpt::Bridge {
                        image_center: image,
                        radius: laser.radius,
                        beam_dir: dir,
                        versine,
                    });
                    origin = hit.p + normal * EPS_NEAR;
                }
                Bsdf::Glass { .. } => break, // see the doc comment
                _ => break, // diffuse (or the beam's own housing): the chain ends
            }
        }
        ladder
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
                spectrum: hit.mat.spectrum,
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
                Bsdf::Glossy { .. } => {
                    // A glossy mirror: the central ray follows the lobe's
                    // mean (the specular direction). The traced branch
                    // carries the lobe's specular-albedo approximation
                    // (the near-mirror limit); the spot's softening is
                    // the visible §9.5 effect.
                    throughput = throughput * mat.albedo;
                    reflect_dir(dir, normal)
                }
                Bsdf::Glass { .. } => {
                    let n = mat.bsdf.glass_n(550.0);
                    let (eta_i, eta_t) = if front_face { (1.0, n) } else { (n, 1.0) };
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
                _ => unreachable!("only Mirror, Glossy and Glass are delta-like here"),
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
                mat: Material::emissive_spec(l.spectrum),
            })
        } else if l.shape == 2 {
            AreaLight::from_laser_versine(l.center, l.normal, l.radius, l.half_w, l.spectrum)
        } else {
            AreaLight::from_quad(&Quad {
                center: l.center,
                normal: l.normal,
                tangent: l.tangent,
                bitangent: l.bitangent,
                half_w: l.half_w,
                half_h: l.half_h,
                mat: Material::emissive_spec(l.spectrum),
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
    use rand::SeedableRng;

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
        // The spectrum is the smooth reconstruction of the RGB intent:
        // its tristimulus integral is exactly (5, 5, 5) — pinned by the
        // spectrum tests; here it suffices that it is a smooth SPD.
        assert!(matches!(
            scene.spheres[0].mat.spectrum,
            crate::spectrum::SpectrumKind::Smooth { .. }
        ));
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

    /// The power parametrization (the collimated-beam rethink): the
    /// line's spectral radiance must be P = Φ/(A_ap·Ω(θ)), so the spot
    /// irradiance E = Φ/(π·w₀²) is *invariant to θ* — θ controls only
    /// the divergence, as it physically does. (The old
    /// radiance-parametrized form coupled them: Φ ∝ Ω at fixed Le, so
    /// θ was secretly a brightness knob.)
    #[test]
    fn laser_beam_power_parametrization_is_theta_invariant() {
        const PI: f32 = std::f32::consts::PI;
        let radius = 0.01f32;
        let power_w = 0.05f32;
        // The spot irradiance E = Φ/(π·w₀²) — the same for every θ.
        let expected_e = power_w / (PI * radius * radius);
        for theta in [1e-5f32, 1e-4, 1e-3, 1e-2, 0.1] {
            let light = LightData::from_laser_beam(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                radius,
                theta,
                0.2,
                650.0,
                power_w,
            );
            let SpectrumKind::Line { power, .. } = light.spectrum else {
                panic!("a beam laser must be a line emitter");
            };
            // P = Φ/(A_ap·Ω): the derived radiance.
            let omega = 2.0 * PI * crate::geometry::cone_versine(theta);
            let aperture_area = PI * radius * radius;
            let expected_p = power_w / (aperture_area * omega);
            assert!(
                (power - expected_p).abs() <= expected_p * 1e-3,
                "theta {theta}: radiance {power} != {expected_p}"
            );
            // The spot irradiance: E = P·A_ap·Ω/ (π·w₀²) = Φ/(π·w₀²)
            // — invariant to θ by construction.
            let e = power * aperture_area * omega / (PI * radius * radius);
            assert!(
                (e - expected_e).abs() <= expected_e * 1e-3,
                "theta {theta}: spot irradiance {e} != {expected_e}"
            );
        }
    }

    /// The footprint law: the beam's cross-section radius at distance d
    /// is w(d) = w₀ + d·tanθ ≈ w₀ + d·θ. The versine representation
    /// must reproduce it at the physical divergences (the f32 cosine
    /// form collapsed there — at the diffraction limit cos(θ) is
    /// *exactly* 1.0 in f32, a zero-solid-angle cone: the pdf
    /// 1/(2π(1−cosθ)) would divide by zero and the gate would reject
    /// the beam's own axis).
    #[test]
    fn laser_beam_versine_survives_the_physical_divergences() {
        const PI: f32 = std::f32::consts::PI;
        let w0 = 0.01f32;
        // The diffraction-limited beam (M² = 1): θ = λ/(πw₀) ≈ 1.7e-4
        // rad for 532 nm. The f32 cosine collapses to exactly 1.0 here
        // (θ²/2 = 1.4e-10 is far below the 3e-8 rounding threshold).
        let theta_dl = 532.0e-9 / (PI * w0);
        assert!(
            theta_dl.cos() == 1.0,
            "the f32 cosine must collapse at the diffraction limit"
        );
        let versine_dl = crate::geometry::cone_versine(theta_dl);
        assert!(
            versine_dl > 0.0 && (versine_dl - 0.5 * theta_dl * theta_dl).abs()
                <= 0.5 * theta_dl * theta_dl * 1e-3,
            "the versine must survive where the cosine collapsed: {versine_dl}"
        );

        // The demo's M² = 30 beam: θ ≈ 5.1e-4 rad. The cosine keeps
        // ~1 significant digit here (1 − 1.19e-7 vs the true
        // 1 − 1.29e-7 — an 8% error in the solid angle); the versine
        // is exact.
        let theta = 30.0 * 532.0e-9 / (PI * w0);
        let light = LightData::from_laser_beam(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            w0,
            theta,
            0.2,
            532.0,
            0.005,
        );
        let versine = light.half_w;
        assert!(
            (versine - 0.5 * theta * theta).abs() <= 0.5 * theta * theta * 1e-3,
            "versine {versine} != theta^2/2 = {}",
            0.5 * theta * theta
        );
        // The footprint at 5 m: w = w₀ + d·θ ≈ 1 cm + 2.5 mm — the
        // spot grows visibly across the room, exactly what M² = 30
        // is for (a diffraction-limited beam would stay a constant
        // 1 cm disc).
        let d = 5.0f32;
        let footprint = w0 + d * theta;
        assert!(
            (footprint - 0.0125).abs() < 1e-3,
            "footprint {footprint}"
        );
    }

    /// The rung's gate-mapped cone: at a laser's physical divergence
    /// the emission cone (θ ~ 5e-4 rad) is far tighter than the image
    /// disc's cone (r/D ~ 1e-2), so the rung must propose the
    /// gate-mapped cone — its samples must lie inside the emission
    /// cone's gate-mapped image (the reversed beam direction), or the
    /// aperture's cone gate would reject ~all of them and the spot
    /// would never converge. At the wide test angles the disc cone is
    /// the binding one and the samples must aim at the image disc (the
    /// old behavior).
    #[test]
    fn bridge_rung_samples_the_gate_mapped_cone_at_physical_divergence() {
        use crate::bdpt::Bridge;
        // A rung 5 m away, aperture radius 1 cm: the disc cone's
        // half-angle is asin(0.01/5) ≈ 2e-3 rad.
        let image_center = Vec3::new(0.0, 5.0, 0.0);
        let radius = 0.01f32;
        // The physical divergence of the demo's green laser.
        let theta = 30.0 * 532.0e-9 / (std::f32::consts::PI * 0.01);
        let versine = crate::geometry::cone_versine(theta);
        let beam_dir = Vec3::new(0.0, -1.0, 0.0); // the beam travels -y
        let rung = Bridge {
            image_center,
            radius,
            beam_dir,
            versine,
        };
        let p = Vec3::new(0.0, 0.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        for _ in 0..500 {
            let d = rung.sample_dir(p, &mut rng).expect("non-degenerate");
            // Every sample must lie inside the gate-mapped cone: within
            // θ of the reversed beam direction (+y — the direction a
            // ray from p must travel to arrive at the aperture inside
            // the emission cone).
            let dx = d.x - 0.0;
            let dy = d.y - 1.0;
            let dz = d.z - 0.0;
            assert!(
                dx * dx + dy * dy + dz * dz <= 2.0 * versine * 1.001,
                "sample {d:?} outside the gate-mapped cone (versine {versine})"
            );
            // ...and the pdf must be the gate-mapped cone's: uniform
            // over 2π·versine.
            let pdf = rung.pdf(p, d);
            assert!(
                (pdf - 1.0 / (2.0 * std::f32::consts::PI * versine)).abs()
                    < 1e-3 / (2.0 * std::f32::consts::PI * versine),
                "pdf {pdf} is not the gate-mapped cone's"
            );
        }
        // A direction at the disc-cone's edge but outside the emission
        // cone (2e-3 rad off-axis, 4x the divergence) must have pdf 0:
        // the rung no longer proposes it.
        let off = Vec3::new(2e-3, 1.0, 0.0).normalize();
        assert_eq!(rung.pdf(p, off), 0.0);

        // The wide-angle control: θ = 0.1 rad > the disc cone's 2e-3,
        // so the disc binds and the samples aim at the image disc (the
        // old behavior — the wide-θ synthetic scenes keep their
        // convergence).
        let wide = Bridge {
            image_center,
            radius,
            beam_dir,
            versine: crate::geometry::cone_versine(0.1),
        };
        for _ in 0..500 {
            let d = wide.sample_dir(p, &mut rng).expect("non-degenerate");
            // Inside the disc cone: within asin(r/D) of the aim axis.
            let aim = (image_center - p).normalize();
            let dx = d.x - aim.x;
            let dy = d.y - aim.y;
            let dz = d.z - aim.z;
            let disc_versine = 0.01f32 * 0.01 / 25.0;
            assert!(
                dx * dx + dy * dy + dz * dz <= 2.0 * disc_versine * 1.001,
                "wide sample {d:?} outside the disc cone"
            );
        }
    }
}
