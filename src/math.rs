//! Minimal linear algebra: 3-component vectors and AABBs.

use std::ops::{Add, Div, Index, Mul, Neg, Sub};

// --- CONSTANTS ---
pub const EPS_NEAR: f32 = 0.001;
pub const VEC_ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

// --- VEC3 ---
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

impl Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
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

/// Component-wise product (used for throughput accumulation).
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

    /// Slab test. Handles axis-parallel rays explicitly so it never produces
    /// NaN from `0 * infinity` (which would silently drop valid hits).
    pub fn intersect(&self, ray: &crate::geometry::Ray) -> bool {
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        for i in 0..3 {
            let (o, d) = (ray.origin[i], ray.direction[i]);
            if d == 0.0 {
                // Parallel to this slab: the origin must lie inside it.
                if o < self.min[i] || o > self.max[i] {
                    return false;
                }
            } else {
                let inv = 1.0 / d;
                let mut t0 = (self.min[i] - o) * inv;
                let mut t1 = (self.max[i] - o) * inv;
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                t_min = t_min.max(t0);
                t_max = t_max.min(t1);
                if t_min > t_max {
                    return false;
                }
            }
        }

        t_max > 0.0
    }
}