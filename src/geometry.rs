use crate::math::{BBox, Vec3, EPS_NEAR};
use rand::Rng;

const PI: f32 = std::f32::consts::PI;

// --- RAY ---
#[derive(Clone)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Minimum positive t at which this ray could enter a bounding box.
    /// Returns None if the ray starts inside the box or can never reach it.
    pub fn min_t_intersect(&self, bbox: &BBox) -> Option<f32> {
        let mut t_min = f32::NEG_INFINITY;

        for i in 0..3 {
            let d = self.direction[i];
            if d == 0.0 {
                // Parallel to this slab: unreachable unless inside it.
                if self.origin[i] < bbox.min[i] || self.origin[i] > bbox.max[i] {
                    return None;
                }
                continue;
            }
            let inv_d = 1.0 / d;
            let t0 = (bbox.min[i] - self.origin[i]) * inv_d;
            let t1 = (bbox.max[i] - self.origin[i]) * inv_d;
            t_min = t_min.max(t0.min(t1));
        }

        if t_min <= EPS_NEAR { None } else { Some(t_min) }
    }
}

// --- HIT RECORD ---
pub struct HitRecord {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub mat: Material,
}

// --- MATERIALS ---
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MaterialType {
    Diffuse,
    Emissive,
}

/// Diffuse surface. For emissive materials the albedo is ignored: emitters
/// are treated as pure light sources that do not reflect.
#[derive(Clone, Copy)]
pub struct Material {
    pub albedo: Vec3,
    pub emission: Vec3,
    pub mat_type: MaterialType,
}

// --- INTERSECTABLE TRAIT ---
pub trait Intersectable: Sync + Send {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord>;
    fn bounding_box(&self) -> BBox;
}

// --- SPHERE ---
#[derive(Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub mat: Material,
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 { return None; }

        // Prefer the near root; fall back to the far root so the sphere is
        // still visible when the camera is inside it.
        let sqrt_d = discriminant.sqrt();
        let mut t = (-b - sqrt_d) / (2.0 * a);
        if t < EPS_NEAR {
            t = (-b + sqrt_d) / (2.0 * a);
        }
        if t < EPS_NEAR { return None; }

        let p = ray.at(t);
        let normal = (p - self.center).normalize();
        Some(HitRecord { t, p, normal, mat: self.mat })
    }

    fn bounding_box(&self) -> BBox {
        BBox {
            min: self.center - Vec3::new(self.radius, self.radius, self.radius),
            max: self.center + Vec3::new(self.radius, self.radius, self.radius),
        }
    }
}

// --- PLANE ---
#[derive(Clone)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
    pub mat: Material,
}

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 { return None; }
        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < EPS_NEAR { return None; }

        Some(HitRecord {
            t,
            p: ray.at(t),
            normal: self.normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        // Planes have infinite extent - use a large box
        BBox {
            min: Vec3::new(-1e6, -1e6, -1e6),
            max: Vec3::new(1e6, 1e6, 1e6),
        }
    }
}

// --- QUAD ---
/// Tangent frame for a `width` x `height` rectangle facing `normal`.
///
/// For axis-aligned normals the extents are world-axis sized: `width`
/// spans the horizontal axis (x for normals along z or y, z for normals
/// along x) and `height` spans the vertical axis (y, or z for normals
/// along y). Sizing the extents along an arbitrary helper-axis tangent
/// frame instead *rotates* walls: a "5 wide, 2 tall" wall facing +z came
/// out 2 wide and 5 tall, because the helper axis for a +z normal is
/// vertical.
///
/// Non-axis-aligned normals have no world-axis mapping; they fall back
/// to a helper axis perpendicular to the normal, with `width` along the
/// tangent and `height` along the bitangent.
///
/// Returns `(tangent, bitangent, half_w, half_h)`.
pub(crate) fn quad_frame(normal: Vec3, width: f32, height: f32) -> (Vec3, Vec3, f32, f32) {
    // An axis normal is exactly unit length along its axis after
    // normalization, so a near-1 threshold separates it from diagonals.
    const AXIS: f32 = 1.0 - 1e-3;
    if normal.y.abs() > AXIS {
        // Floor/ceiling: width along x, height along z.
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), width * 0.5, height * 0.5)
    } else if normal.x.abs() > AXIS {
        // Side walls: width along z, height along y.
        (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0), width * 0.5, height * 0.5)
    } else if normal.z.abs() > AXIS {
        // Back/front walls: width along x, height along y.
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), width * 0.5, height * 0.5)
    } else {
        // Diagonal normal: |normal.x| <= 1 - 1e-3, so the x axis is
        // never parallel to it and works as the helper axis.
        let tangent = Vec3::new(1.0, 0.0, 0.0).cross(normal).normalize();
        let bitangent = normal.cross(tangent);
        (tangent, bitangent, width * 0.5, height * 0.5)
    }
}

/// A finite rectangle, used for light panels.
#[derive(Clone)]
pub struct Quad {
    pub center: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub half_w: f32,
    pub half_h: f32,
    pub mat: Material,
}

impl Quad {
    /// A `width` x `height` rectangle centered at `center`, facing `normal`.
    /// Kept for API completeness (used by tests and future scenes).
    #[allow(dead_code)]
    pub fn new(center: Vec3, normal: Vec3, width: f32, height: f32, mat: Material) -> Self {
        let normal = normal.normalize();
        let (tangent, bitangent, half_w, half_h) = quad_frame(normal, width, height);
        Self { center, normal, tangent, bitangent, half_w, half_h, mat }
    }
}

impl Intersectable for Quad {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 { return None; }
        let t = (self.center - ray.origin).dot(self.normal) / denom;
        if t < EPS_NEAR { return None; }

        let p = ray.at(t);
        let local = p - self.center;
        if local.dot(self.tangent).abs() > self.half_w
            || local.dot(self.bitangent).abs() > self.half_h
        {
            return None;
        }
        Some(HitRecord { t, p, normal: self.normal, mat: self.mat })
    }

    fn bounding_box(&self) -> BBox {
        let mut bbox = BBox::empty();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                let corner = self.center
                    + self.tangent * (sx * self.half_w)
                    + self.bitangent * (sy * self.half_h);
                bbox = bbox.surround(BBox { min: corner, max: corner });
            }
        }
        bbox
    }
}

// --- CUBE ---
#[derive(Clone)]
pub struct Cube {
    pub min: Vec3,
    pub max: Vec3,
    pub mat: Material,
}

impl Cube {
    /// Kept for API completeness (used by tests and future scenes).
    #[allow(dead_code)]
    pub fn new(center: Vec3, size: f32, mat: Material) -> Self {
        let half = size * 0.5;
        Self {
            min: center - Vec3::new(half, half, half),
            max: center + Vec3::new(half, half, half),
            mat,
        }
    }
}

impl Intersectable for Cube {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        let dirs = [ray.direction.x, ray.direction.y, ray.direction.z];
        let origins = [ray.origin.x, ray.origin.y, ray.origin.z];
        let mins = [self.min.x, self.min.y, self.min.z];
        let maxs = [self.max.x, self.max.y, self.max.z];

        for i in 0..3 {
            if dirs[i] == 0.0 {
                // Parallel to this slab: must lie within it.
                if origins[i] < mins[i] || origins[i] > maxs[i] {
                    return None;
                }
                continue;
            }
            let inv_d = 1.0 / dirs[i];
            let mut t0 = (mins[i] - origins[i]) * inv_d;
            let mut t1 = (maxs[i] - origins[i]) * inv_d;
            if t0 > t1 { std::mem::swap(&mut t0, &mut t1); }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
        }

        if !(t_max >= t_min && t_max > 0.0 && t_min < 1e6) {
            return None;
        }
        let t = if t_min < EPS_NEAR { t_max } else { t_min };
        if t < EPS_NEAR { return None; }

        let p = ray.at(t);

        // Outward normal of the face closest to the hit point. Checking all
        // six faces (not just the min faces) is what makes top/+x/+z hits
        // shade correctly.
        let candidates = [
            ((p.x - self.min.x).abs(), Vec3::new(-1.0, 0.0, 0.0)),
            ((p.x - self.max.x).abs(), Vec3::new(1.0, 0.0, 0.0)),
            ((p.y - self.min.y).abs(), Vec3::new(0.0, -1.0, 0.0)),
            ((p.y - self.max.y).abs(), Vec3::new(0.0, 1.0, 0.0)),
            ((p.z - self.min.z).abs(), Vec3::new(0.0, 0.0, -1.0)),
            ((p.z - self.max.z).abs(), Vec3::new(0.0, 0.0, 1.0)),
        ];
        let normal = candidates
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, n)| *n)
            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

        Some(HitRecord { t, p, normal, mat: self.mat })
    }

    fn bounding_box(&self) -> BBox {
        BBox { min: self.min, max: self.max }
    }
}

// --- AREA LIGHTS ---
/// A sampleable emitter surface. The shape parameters must match an emissive
/// object in the scene so that BDPT can connect paths to real geometry.
#[derive(Clone)]
pub enum LightShape {
    Sphere { center: Vec3, radius: f32 },
    Quad { center: Vec3, normal: Vec3, tangent: Vec3, bitangent: Vec3, half_w: f32, half_h: f32 },
}

#[derive(Clone)]
pub struct AreaLight {
    pub shape: LightShape,
    pub emission: Vec3,
}

impl AreaLight {
    pub fn from_sphere(sphere: &Sphere) -> Self {
        Self {
            shape: LightShape::Sphere {
                center: sphere.center,
                radius: sphere.radius,
            },
            emission: sphere.mat.emission,
        }
    }

    pub fn from_quad(quad: &Quad) -> Self {
        Self {
            shape: LightShape::Quad {
                center: quad.center,
                normal: quad.normal,
                tangent: quad.tangent,
                bitangent: quad.bitangent,
                half_w: quad.half_w,
                half_h: quad.half_h,
            },
            emission: quad.mat.emission,
        }
    }

    pub fn area(&self) -> f32 {
        match &self.shape {
            LightShape::Sphere { radius, .. } => 4.0 * PI * radius * radius,
            LightShape::Quad { half_w, half_h, .. } => 4.0 * half_w * half_h,
        }
    }

    /// Uniform surface sample: (point, normal). The pdf is 1 / area.
    pub fn sample(&self, rng: &mut impl Rng) -> (Vec3, Vec3) {
        match &self.shape {
            LightShape::Sphere { center, radius } => {
                // Uniform point on the sphere via spherical coordinates.
                let z = 1.0 - 2.0 * rng.gen::<f32>(); // cos theta in [-1, 1]
                let phi = 2.0 * PI * rng.gen::<f32>();
                let r = (1.0 - z * z).sqrt();
                let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                (*center + dir * *radius, dir)
            }
            LightShape::Quad { center, normal, tangent, bitangent, half_w, half_h } => {
                let u = rng.gen::<f32>() * 2.0 - 1.0;
                let v = rng.gen::<f32>() * 2.0 - 1.0;
                let p = *center + *tangent * (u * half_w) + *bitangent * (v * half_h);
                (p, *normal)
            }
        }
    }

    /// True if `p` lies on the light surface (within a small epsilon).
    pub fn contains(&self, p: Vec3) -> bool {
        match &self.shape {
            LightShape::Sphere { center, radius } => {
                ((p - *center).length() - *radius).abs() < 1e-3
            }
            LightShape::Quad { center, normal, tangent, bitangent, half_w, half_h } => {
                let local = p - *center;
                local.dot(*normal).abs() < 1e-3
                    && local.dot(*tangent).abs() <= half_w + 1e-3
                    && local.dot(*bitangent).abs() <= half_h + 1e-3
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn diffuse(albedo: Vec3) -> Material {
        Material { albedo, emission: Vec3::new(0.0, 0.0, 0.0), mat_type: MaterialType::Diffuse }
    }

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction: direction.normalize() }
    }

    #[test]
    fn sphere_hit_from_outside_and_inside() {
        let s = Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            mat: diffuse(Vec3::new(1.0, 1.0, 1.0)),
        };
        let outside = s.intersect(&ray(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)));
        assert!(outside.is_some());
        assert!((outside.unwrap().t - 4.0).abs() < 1e-4);

        // A ray starting at the center must hit the far surface.
        let inside = s.intersect(&ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)));
        assert!(inside.is_some());
        assert!((inside.unwrap().t - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cube_normals_face_outward_on_all_sides() {
        let c = Cube::new(Vec3::new(0.0, 0.0, 0.0), 1.0, diffuse(Vec3::new(1.0, 1.0, 1.0)));
        let top = c
            .intersect(&ray(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0)))
            .unwrap();
        assert_eq!(top.normal, Vec3::new(0.0, 1.0, 0.0));

        let right = c
            .intersect(&ray(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)))
            .unwrap();
        assert_eq!(right.normal, Vec3::new(1.0, 0.0, 0.0));

        let front = c
            .intersect(&ray(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)))
            .unwrap();
        assert_eq!(front.normal, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn quad_intersect_respects_bounds() {
        let q = Quad::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            2.0,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        let hit = q.intersect(&ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 2.0).abs() < 1e-4);

        let miss = q.intersect(&ray(Vec3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        assert!(miss.is_none());
    }

    #[test]
    fn area_light_samples_lie_on_the_surface() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sphere = Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
            mat: Material {
                albedo: Vec3::new(0.0, 0.0, 0.0),
                emission: Vec3::new(5.0, 5.0, 5.0),
                mat_type: MaterialType::Emissive,
            },
        };
        let light = AreaLight::from_sphere(&sphere);
        assert!((light.area() - 4.0 * PI * 0.25).abs() < 1e-5);
        for _ in 0..100 {
            let (p, _) = light.sample(&mut rng);
            assert!(light.contains(p));
        }

        let quad = Quad::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            4.0,
            sphere.mat,
        );
        let light = AreaLight::from_quad(&quad);
        assert!((light.area() - 8.0).abs() < 1e-5);
        for _ in 0..100 {
            let (p, n) = light.sample(&mut rng);
            assert!(light.contains(p));
            assert_eq!(n, Vec3::new(0.0, -1.0, 0.0));
        }
    }
}