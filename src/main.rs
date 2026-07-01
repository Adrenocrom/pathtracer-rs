use rand::Rng;
use rayon::prelude::*;
use std::io::{self, Write};

// --- VEC3 UTILS ---
#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 { self } else { self * (1.0 / len) }
    }

    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Vec3 {
        self * (1.0 / rhs)
    }
}

const Vec3_ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

// --- CONFIGURATION ---
const SAMPLES: usize = 16;
const MAX_DEPTH: i32 = 5;

// --- MATERIALS ---
#[derive(Clone, Copy)]
enum MaterialType {
    Diffuse,
    Emissive,
}

#[derive(Clone, Copy)]
struct Material {
    albedo: Vec3,
    emission: Vec3,
    mat_type: MaterialType,
}

// --- GEOMETRY ---
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

struct HitRecord {
    t: f32,
    p: Vec3,
    normal: Vec3,
    mat: Material,
}

trait Intersectable: Sync + Send {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord>;
}

struct Sphere {
    center: Vec3,
    radius: f32,
    mat: Material,
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
        if t < 0.001 { return None; }

        let p = ray.at(t);
        let normal = (p - self.center).normalize();
        Some(HitRecord { t, p, normal, mat: self.mat })
    }
}

struct Plane {
    point: Vec3,
    normal: Vec3,
    mat: Material,
}

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 { return None; }
        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < 0.001 { return None; }
        
        Some(HitRecord {
            t,
            p: ray.at(t),
            normal: self.normal,
            mat: self.mat,
        })
    }
}

// --- PATH TRACING ---
fn random_unit_vector() -> Vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let v = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if v.length_squared() <= 1.0 {
            return v.normalize();
        }
    }
}

fn trace(ray: &Ray, scene: &[Box<dyn Intersectable>], depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3_ZERO;
    }

    let mut closest_hit: Option<HitRecord> = None;
    let mut min_t = f32::INFINITY;

    for obj in scene {
        if let Some(hit) = obj.intersect(ray) {
            if hit.t < min_t {
                min_t = hit.t;
                closest_hit = Some(hit);
            }
        }
    }

    if let Some(hit) = closest_hit {
        let emission = hit.mat.emission;
        
        if let MaterialType::Emissive = hit.mat.mat_type {
            return emission;
        }

        let target = hit.normal + random_unit_vector();
        let scattered_ray = Ray {
            origin: hit.p,
            direction: target.normalize(),
        };

        let indirect = trace(&scattered_ray, scene, depth - 1);
        return emission + hit.mat.albedo * indirect;
    }

    Vec3::new(0.02, 0.02, 0.05) 
}

fn render_rgb(color: Vec3) -> String {
    let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
    let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
    let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
    format!("\x1b[38;2;{};{};{}m▀\x1b[0m", r, g, b)
}

fn main() {
    let (width, height) = {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 40));
        (w as usize, h as usize)
    };

    let white = Material { albedo: Vec3::new(0.7, 0.7, 0.7), emission: Vec3_ZERO, mat_type: MaterialType::Diffuse };
    let red = Material { albedo: Vec3::new(0.7, 0.1, 0.1), emission: Vec3_ZERO, mat_type: MaterialType::Diffuse };
    let green = Material { albedo: Vec3::new(0.1, 0.7, 0.1), emission: Vec3_ZERO, mat_type: MaterialType::Diffuse };
    let light = Material { albedo: Vec3_ZERO, emission: Vec3::new(10.0, 10.0, 10.0), mat_type: MaterialType::Emissive };

    let scene: Vec<Box<dyn Intersectable>> = vec![
        Box::new(Plane { point: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(0.0, 1.0, 0.0), mat: white }),
        Box::new(Plane { point: Vec3::new(0.0, 2.0, 0.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: white }),
        Box::new(Plane { point: Vec3::new(-1.0, 1.0, 0.0), normal: Vec3::new(1.0, 0.0, 0.0), mat: red }),
        Box::new(Plane { point: Vec3::new(1.0, 1.0, 0.0), normal: Vec3::new(-1.0, 0.0, 0.0), mat: green }),
        Box::new(Plane { point: Vec3::new(0.0, 1.0, 2.0), normal: Vec3::new(0.0, 0.0, -1.0), mat: white }),
        Box::new(Sphere { center: Vec3::new(0.0, 0.5, 1.0), radius: 0.4, mat: white }),
        Box::new(Plane { point: Vec3::new(0.0, 1.9, 1.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: light }),
    ];

    let cam_pos = Vec3::new(0.0, 1.0, -1.0);
    
    let results: Vec<String> = (0..height).into_par_iter().map(|y| {
        let mut row = String::new();
        for x in 0..width {
            let mut color = Vec3_ZERO;
            for _ in 0..SAMPLES {
                let px = (x as f32 / width as f32 * 2.0 - 1.0) * (width as f32 / height as f32) * 0.5;
                let py = (1.0 - y as f32 / height as f32) * 2.0 * 0.5; 
                let dir = Vec3::new(px, py - 1.0, 1.0).normalize();
                
                let ray = Ray { origin: cam_pos, direction: dir };
                color = color + trace(&ray, &scene, MAX_DEPTH);
            }
            color = color / SAMPLES as f32;
            row.push_str(&render_rgb(color));
        }
        row
    }).collect();

    let mut stdout = io::stdout();
    for row in results {
        writeln!(stdout, "{}", row).unwrap();
    }
}
