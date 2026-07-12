use crate::math::{Vec3, BBox, EPS_NEAR};

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

    /// Compute the minimum positive t where this ray could intersect a bounding box.
    /// Returns None if the ray can never hit the box (e.g., parallel to an axis).
    pub fn min_t_intersect(&self, bbox: &BBox) -> Option<f32> {
        let mut t_min = f32::NEG_INFINITY;

        for i in 0..3 {
            let inv_d = 1.0 / self.direction[i];
            let t0 = (bbox.min[i] - self.origin[i]) * inv_d;
            let t1 = (bbox.max[i] - self.origin[i]) * inv_d;
            if inv_d < 0.0 { std::mem::swap(&mut t0, &mut t1); }
            t_min = t_min.max(t0);
        }

        if t_min > f32::INFINITY || t_min <= EPS_NEAR { None } else { Some(t_min) }
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
#[derive(Clone, Copy, PartialEq)]
pub enum MaterialType {
    Diffuse,
    Emissive,
}

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

        let t = (-b - discriminant.sqrt()) / (2.0 * a);
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

// --- CUBE ---
#[derive(Clone)]
pub struct Cube {
    pub min: Vec3,
    pub max: Vec3,
    pub mat: Material,
}

impl Cube {
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
            let inv_d = 1.0 / dirs[i];
            let mut t0 = (mins[i] - origins[i]) * inv_d;
            let mut t1 = (maxs[i] - origins[i]) * inv_d;
            if inv_d < 0.0 { std::mem::swap(&mut t0, &mut t1); }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
        }

        if t_max >= t_min && t_max > 0.0 && t_min < 1e6 {
            let t = if t_min < EPS_NEAR { t_max } else { t_min };
            if t < EPS_NEAR { return None; }

            let p = ray.at(t);
            
            // Compute normal based on closest face (more robust than epsilon check)
            let mut normal = Vec3::new(0.0, 0.0, 0.0);
            let dx = (p.x - self.min.x).abs();
            let dy = (p.y - self.min.y).abs();
            let dz = (p.z - self.min.z).abs();
            
            if dx < dy && dx < dz {
                normal = Vec3::new(-1.0, 0.0, 0.0);
            } else if dy < dz {
                normal = Vec3::new(0.0, -1.0, 0.0);
            } else {
                normal = Vec3::new(0.0, 0.0, -1.0);
            }

            Some(HitRecord { t, p, normal, mat: self.mat })
        } else {
            None
        }
    }

    fn bounding_box(&self) -> BBox {
        BBox { min: self.min, max: self.max }
    }
}
