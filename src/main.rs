use glam::{Vec3, Vec4};
use rand::Rng;
use rayon::prelude::*;

// --- CONFIGURATION ---
const WIDTH: usize = 80;
const HEIGHT: usize = 40;
const SAMPLES: usize = 32;
const MAX_DEPTH: i32 = 5;
const CHARS: &str = " .:-=+*#%@";

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
        return Vec3::ZERO;
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

        // Diffuse bounce
        let mut rng = rand::thread_rng();
        let target = hit.normal + random_unit_vector();
        let scattered_ray = Ray {
            origin: hit.p,
            direction: target.normalize(),
        };

        let indirect = trace(&scattered_ray, scene, depth - 1);
        return emission + hit.mat.albedo * indirect;
    }

    Vec3::new(0.05, 0.05, 0.05) // Dark grey background
}

fn main() {
    let white = Material { albedo: Vec3::new(0.7, 0.7, 0.7), emission: Vec3::ZERO, mat_type: MaterialType::Diffuse };
    let red = Material { albedo: Vec3::new(0.7, 0.1, 0.1), emission: Vec3::ZERO, mat_type: MaterialType::Diffuse };
    let green = Material { albedo: Vec3::new(0.1, 0.7, 0.1), emission: Vec3::ZERO, mat_type: MaterialType::Diffuse };
    let light = Material { albedo: Vec3::ZERO, emission: Vec3::new(15.0, 15.0, 15.0), mat_type: MaterialType::Emissive };

    let scene: Vec<Box<dyn Intersectable>> = vec![
        Box::new(Plane { point: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(0.0, 1.0, 0.0), mat: white }),   // Floor
        Box::new(Plane { point: Vec3::new(0.0, 2.0, 0.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: white }),  // Ceiling
        Box::new(Plane { point: Vec3::new(-1.0, 1.0, 0.0), normal: Vec3::new(1.0, 0.0, 0.0), mat: red }),    // Left Wall
        Box::new(Plane { point: Vec3::new(1.0, 1.0, 0.0), normal: Vec3::new(-1.0, 0.0, 0.0), mat: green }),  // Right Wall
        Box::new(Plane { point: Vec3::new(0.0, 1.0, 2.0), normal: Vec3::new(0.0, 0.0, -1.0), mat: white }), // Back Wall
        Box::new(Sphere { center: Vec3::new(0.0, 0.5, 1.0), radius: 0.4, mat: white }),
        Box::new(Plane { point: Vec3::new(0.0, 1.9, 1.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: light }),  // Light panel
    ];

    let cam_pos = Vec3::new(0.0, 1.0, -1.0);
    
    let results: Vec<Vec<char>> = (0..HEIGHT).into_par_iter().map(|y| {
        (0..WIDTH).map(|x| {
            let mut color = Vec3::ZERO;
            for _ in 0..SAMPLES {
                let px = (x as f32 / WIDTH as f32 * 2.0 - 1.0) * (WIDTH as f32 / HEIGHT as f32) * 0.5;
                let py = (y as f32 / HEIGHT as f32 * 2.0 - 1.0) * -0.5;
                let dir = Vec3::new(px, py, 1.0).normalize();
                
                let ray = Ray { origin: cam_pos, direction: dir };
                color += trace(&ray, &scene, MAX_DEPTH);
            }
            color /= SAMPLES as f32;
            
            let luminance = (color.x + color.y + color.z) / 3.0;
            let norm_lum = (luminance / 5.0).clamp(0.0, 1.0);
            let char_idx = (norm_lum * (CHARS.len() - 1) as f32) as usize;
            CHARS.chars().nth(char_idx).unwrap()
        }).collect()
    }).collect();

    for row in results {
        let s: String = row.into_iter().collect();
        println!("{}", s);
    }
}
