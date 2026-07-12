use std::ops::{Add, Sub, Mul, Div};

// --- CONSTANTS ---
pub const EPS_NEAR: f32 = 0.001;
pub const EPS_NORMAL: f32 = 0.001;
pub const DENOM_TOLERANCE: f32 = 1e-6;
pub const VEC_ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

// --- VEC3 UTILS ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 { self } else { self * (1.0 / len) }
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn min(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x.min(other.x), self.y.min(other.y), self.z.min(other.z))
    }

    pub fn max(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x.max(other.x), self.y.max(other.y), self.z.max(other.z))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Vec3 {
        self * (1.0 / rhs)
    }
}

// --- BOUNDING BOX ---
#[derive(Clone, Copy, Debug)]
pub struct BBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BBox {
    pub fn empty() -> Self {
        BBox {
            min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub fn surround(self, other: BBox) -> BBox {
        BBox {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn intersect(&self, ray: &crate::geometry::Ray) -> bool {
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        // X
        let inv_dir_x = 1.0 / ray.direction.x;
        let mut t0x = (self.min.x - ray.origin.x) * inv_dir_x;
        let mut t1x = (self.max.x - ray.origin.x) * inv_dir_x;
        if inv_dir_x < 0.0 { std::mem::swap(&mut t0x, &mut t1x); }
        t_min = t_min.max(t0x);
        t_max = t_max.min(t1x);

        // Y
        let inv_dir_y = 1.0 / ray.direction.y;
        let mut t0y = (self.min.y - ray.origin.y) * inv_dir_y;
        let mut t1y = (self.max.y - ray.origin.y) * inv_dir_y;
        if inv_dir_y < 0.0 { std::mem::swap(&mut t0y, &mut t1y); }
        t_min = t_min.max(t0y);
        t_max = t_max.min(t1y);

        // Z
        let inv_dir_z = 1.0 / ray.direction.z;
        let mut t0z = (self.min.z - ray.origin.z) * inv_dir_z;
        let mut t1z = (self.max.z - ray.origin.z) * inv_dir_z;
        if inv_dir_z < 0.0 { std::mem::swap(&mut t0z, &mut t1z); }
        t_min = t_min.max(t0z);
        t_max = t_max.min(t1z);

        t_max >= t_min && t_max > 0.0
    }
}
