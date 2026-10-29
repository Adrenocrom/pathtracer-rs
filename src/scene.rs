//! Data-oriented scene description shared by the CPU and GPU backends.
//!
//! The CPU renderer consumes `SceneData` through the existing
//! `Box<dyn Intersectable>` + BVH path (it stays the reference oracle); the
//! GPU backend uploads the same data as flat POD buffers. Keeping one source
//! of truth for the scene guarantees both backends trace identical geometry.

use crate::geometry::{AreaLight, Cube, Intersectable, Material, MaterialType, Plane, Quad, Sphere};
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
        Self { albedo, emission: Vec3::new(0.0, 0.0, 0.0), mat_type: MaterialType::Diffuse }
    }

    pub fn emissive(emission: Vec3) -> Self {
        Self { albedo: Vec3::new(0.0, 0.0, 0.0), emission, mat_type: MaterialType::Emissive }
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
    /// light is disabled), but part of the scene API.
    #[allow(dead_code)]
    pub fn from_quad(center: Vec3, normal: Vec3, width: f32, height: f32, emission: Vec3) -> Self {
        let normal = normal.normalize();
        // Same helper-axis rule as `geometry::Quad::new` so the tangent frame
        // matches exactly.
        let a = if normal.x.abs() > 0.9 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let tangent = a.cross(normal).normalize();
        let bitangent = normal.cross(tangent);
        Self {
            shape: 1,
            center,
            radius: 0.0,
            normal,
            tangent,
            bitangent,
            half_w: width * 0.5,
            half_h: height * 0.5,
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
        self.spheres.push(SphereData { center, radius, mat });
    }

    /// Add a finite rectangle (e.g. a light panel). Not used by the demo
    /// scene, but part of the scene API — the original code had a
    /// commented-out ceiling quad light.
    #[allow(dead_code)]
    pub fn add_quad(&mut self, center: Vec3, normal: Vec3, width: f32, height: f32, mat: MaterialData) {
        let normal = normal.normalize();
        let a = if normal.x.abs() > 0.9 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let tangent = a.cross(normal).normalize();
        let bitangent = normal.cross(tangent);
        self.quads.push(QuadData {
            center,
            normal,
            tangent,
            bitangent,
            half_w: width * 0.5,
            half_h: height * 0.5,
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

    pub fn add_plane(&mut self, point: Vec3, normal: Vec3, mat: MaterialData) {
        self.planes.push(PlaneData { point, normal, mat });
    }

    pub fn add_light(&mut self, light: LightData) {
        self.lights.push(light);
    }

    /// Build the trait-object scene (BVH) used by the CPU renderer.
    ///
    /// The light geometry is added as real intersectable objects (mirroring
    /// the original scene construction) so BDPT eye subpaths can hit the
    /// emitters, while `lights` provides the area-sampling interface.
    pub fn build_cpu_scene(&self) -> (Box<dyn Intersectable>, Vec<AreaLight>) {
        let mut objects: Vec<Box<dyn Intersectable>> = Vec::new();
        for s in &self.spheres {
            objects.push(Box::new(Sphere {
                center: s.center,
                radius: s.radius,
                mat: Material { albedo: s.mat.albedo, emission: s.mat.emission, mat_type: s.mat.mat_type },
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
                mat: Material { albedo: q.mat.albedo, emission: q.mat.emission, mat_type: q.mat.mat_type },
            }));
        }
        for c in &self.cubes {
            objects.push(Box::new(Cube {
                min: c.min,
                max: c.max,
                mat: Material { albedo: c.mat.albedo, emission: c.mat.emission, mat_type: c.mat.mat_type },
            }));
        }
        for p in &self.planes {
            objects.push(Box::new(Plane {
                point: p.point,
                normal: p.normal,
                mat: Material { albedo: p.mat.albedo, emission: p.mat.emission, mat_type: p.mat.mat_type },
            }));
        }
        // Light geometry must be intersectable too: BDPT connects paths to
        // real surfaces, and the eye subpath must be able to hit an emitter.
        for l in &self.lights {
            if l.shape == 0 {
                objects.push(Box::new(Sphere {
                    center: l.center,
                    radius: l.radius,
                    mat: Material {
                        albedo: Vec3::new(0.0, 0.0, 0.0),
                        emission: l.emission,
                        mat_type: MaterialType::Emissive,
                    },
                }));
            } else {
                objects.push(Box::new(Quad {
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
                }));
            }
        }
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
        let q = LightData::from_quad(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 2.0, 4.0, Vec3::new(1.0, 1.0, 1.0));
        assert!((q.area() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_scene_conversion_preserves_light_count() {
        let mut scene = SceneData::new();
        scene.add_plane(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), MaterialData::diffuse(Vec3::new(0.8, 0.8, 0.8)));
        scene.add_light(LightData::from_sphere(Vec3::new(0.0, 2.0, 0.0), 0.5, Vec3::new(5.0, 5.0, 5.0)));
        let (cpu_scene, lights) = scene.build_cpu_scene();
        assert_eq!(lights.len(), 1);
        // A ray straight up from the floor must hit the light sphere.
        let ray = crate::geometry::Ray {
            origin: Vec3::new(0.0, 0.1, 0.0),
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        assert!(cpu_scene.intersect(&ray).is_some());
    }
}