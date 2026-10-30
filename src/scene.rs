//! Data-oriented scene description shared by the CPU and GPU backends.
//!
//! The CPU renderer consumes `SceneData` through the existing
//! `Box<dyn Intersectable>` + BVH path (it stays the reference oracle); the
//! GPU backend uploads the same data as flat POD buffers. Keeping one source
//! of truth for the scene guarantees both backends trace identical geometry.

use crate::geometry::{
    AreaLight, Cube, Intersectable, Material, MaterialType, Plane, Quad, Sphere,
};
use crate::math::Vec3;

/// Diffuse surface. For emissive materials the albedo is ignored: emitters
/// are treated as pure light sources that do not reflect.
#[derive(Clone, Copy)]
pub struct MaterialData {
    pub albedo: Vec3,
    pub emission: Vec3,
    pub mat_type: MaterialType,
}

impl MaterialData {
    pub fn diffuse(albedo: Vec3) -> Self {
        Self {
            albedo,
            emission: Vec3::new(0.0, 0.0, 0.0),
            mat_type: MaterialType::Diffuse,
        }
    }

    pub fn emissive(emission: Vec3) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            emission,
            mat_type: MaterialType::Emissive,
        }
    }
}

/// A sampleable emitter surface, mirroring `geometry::AreaLight` in flat form.
#[derive(Clone, Copy)]
pub struct LightData {
    /// 0 = sphere, 1 = quad.
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

    /// Surface area (used by tests; the GPU shader computes it inline).
    #[allow(dead_code)]
    pub fn area(&self) -> f32 {
        const PI: f32 = std::f32::consts::PI;
        if self.shape == 0 {
            4.0 * PI * self.radius * self.radius
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
    pub lights: Vec<LightData>,
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

impl SceneData {
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            quads: Vec::new(),
            cubes: Vec::new(),
            planes: Vec::new(),
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

    pub fn add_cube(&mut self, center: Vec3, size: f32, mat: MaterialData) {
        let half = size * 0.5;
        self.cubes.push(CubeData {
            min: center - Vec3::new(half, half, half),
            max: center + Vec3::new(half, half, half),
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
    /// * as real intersectable geometry (`spheres`/`quads`), so eye
    ///   subpaths can hit it on *both* backends — the GPU's
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
                mat: Material {
                    albedo: s.mat.albedo,
                    emission: s.mat.emission,
                    mat_type: s.mat.mat_type,
                },
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
                mat: Material {
                    albedo: q.mat.albedo,
                    emission: q.mat.emission,
                    mat_type: q.mat.mat_type,
                },
            }));
        }
        for c in &self.cubes {
            objects.push(Box::new(Cube {
                min: c.min,
                max: c.max,
                mat: Material {
                    albedo: c.mat.albedo,
                    emission: c.mat.emission,
                    mat_type: c.mat.mat_type,
                },
            }));
        }
        for p in &self.planes {
            objects.push(Box::new(Plane {
                point: p.point,
                normal: p.normal,
                mat: Material {
                    albedo: p.mat.albedo,
                    emission: p.mat.emission,
                    mat_type: p.mat.mat_type,
                },
            }));
        }
        // Emitter geometry is already part of `spheres`/`quads` —
        // `add_light` registers it — so it must not be added again here.
        let scene = crate::bvh::BVHNode::build(objects).expect("scene is non-empty");
        let lights = self.lights.iter().map(|l| self.cpu_light(l)).collect();
        (scene, lights)
    }

    fn cpu_light(&self, l: &LightData) -> AreaLight {
        if l.shape == 0 {
            AreaLight::from_sphere(&Sphere {
                center: l.center,
                radius: l.radius,
                mat: Material {
                    albedo: Vec3::new(0.0, 0.0, 0.0),
                    emission: l.emission,
                    mat_type: MaterialType::Emissive,
                },
            })
        } else {
            AreaLight::from_quad(&Quad {
                center: l.center,
                normal: l.normal,
                tangent: l.tangent,
                bitangent: l.bitangent,
                half_w: l.half_w,
                half_h: l.half_h,
                mat: Material {
                    albedo: Vec3::new(0.0, 0.0, 0.0),
                    emission: l.emission,
                    mat_type: MaterialType::Emissive,
                },
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
        assert_eq!(scene.spheres[0].mat.mat_type, MaterialType::Emissive);
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
        assert_eq!(hit.mat.mat_type, MaterialType::Emissive);

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
        assert_eq!(scene.quads[0].mat.mat_type, MaterialType::Emissive);
    }
}
