use rayon::prelude::*;
use std::io::{self, Write};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, ClearType, Clear},
};
use rand::Rng;
use image::{RgbImage, Rgb};

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
const SAMPLES: usize = 128; // Lowered for responsiveness; increase for PNG
const MAX_DEPTH: i32 = 10;

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

// --- CAMERA ---
struct Cam {
    origin: Vec3,
    fov: f32,
}

impl Cam {
    fn new(origin: Vec3, fov_deg: f32) -> Self {
        Self {
            origin,
            fov: fov_deg.to_radians(),
        }
    }

    fn render(&self, scene: &[Box<dyn Intersectable>], width: usize, height: usize) -> PixelBuffer {
        let aspect = width as f32 / height as f32;
        let scale = (self.fov * 0.5).tan();

        let pixels: Vec<Vec3> = (0..height).into_par_iter().flat_map(|y| {
            (0..width).into_iter().map(|x| {
                let mut color = Vec3_ZERO;
                for _ in 0..SAMPLES {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;
                    let px = (u * 2.0 - 1.0) * 0.5 * aspect * scale;
                    let py = -(v * 2.0 - 1.0) * scale;
                    let dir = Vec3::new(px, py, 1.0).normalize();
                    let ray = Ray { origin: self.origin, direction: dir };
                    color = color + trace(&ray, scene, MAX_DEPTH);
                }
                color / SAMPLES as f32
            }).collect::<Vec<_>>()
        }).collect();

        PixelBuffer {
            width,
            height,
            pixels,
        }
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

// --- RENDERING PIPELINE ---

struct PixelBuffer {
    width: usize,
    height: usize,
    pixels: Vec<Vec3>,
}

impl PixelBuffer {
    fn get_pixel(&self, x: usize, y: usize) -> Vec3 {
        self.pixels[y * self.width + x]
    }

    fn save_as_png(&self, path: &str) -> Result<(), image::ImageError> {
        let mut img = RgbImage::new(self.width as u32, self.height as u32);
        for y in 0..self.height {
            for x in 0..self.width {
                let color = self.get_pixel(x, y);
                img.put_pixel(x as u32, y as u32, Rgb([
                    (color.x.clamp(0.0, 1.0) * 255.0) as u8,
                    (color.y.clamp(0.0, 1.0) * 255.0) as u8,
                    (color.z.clamp(0.0, 1.0) * 255.0) as u8,
                ]));
            }
        }
        img.save(path)
    }
}

fn buffer_to_string(buffer: &PixelBuffer) -> String {
    let mut output = String::new();
    for y in (0..buffer.height).step_by(2) {
        for x in 0..buffer.width {
            let top_pixel = buffer.get_pixel(x, y);
            let bottom_pixel = if y + 1 < buffer.height {
                buffer.get_pixel(x, y + 1)
            } else {
                Vec3_ZERO
            };

            let r_top = (top_pixel.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g_top = (top_pixel.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b_top = (top_pixel.z.clamp(0.0, 1.0) * 255.0) as u8;

            let r_bot = (bottom_pixel.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g_bot = (bottom_pixel.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b_bot = (bottom_pixel.z.clamp(0.0, 1.0) * 255.0) as u8;

            output.push_str(&format!(
                "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                r_top, g_top, b_top, r_bot, g_bot, b_bot
            ));
        }
        output.push_str("\x1b[0m\r\n");
    }
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, Clear(ClearType::All))?;

    let (term_w, term_h) = {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 40));
        (w as usize, h as usize)
    };

    let width = term_w;
    let height = term_h * 2;

    let white = Material { albedo: Vec3::new(0.5, 0.5, 0.5), emission: Vec3_ZERO, mat_type: MaterialType::Diffuse };
    let red = Material { albedo: Vec3::new(0.5, 0.1, 0.1), emission: Vec3_ZERO, mat_type: MaterialType::Diffuse };
    let green = Material { albedo: Vec3::new(0.1, 0.5, 0.1), emission: Vec3_ZERO, mat_type: MaterialType::Diffuse };
    let yellow = Material { albedo: Vec3::new(0.5, 0.5, 0.1), emission: Vec3_ZERO, mat_type: MaterialType::Diffuse };
    let light = Material { albedo: Vec3_ZERO, emission: Vec3::new(1.5, 1.5, 1.4), mat_type: MaterialType::Emissive };

    let scene: Vec<Box<dyn Intersectable>> = vec![
        Box::new(Plane { point: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(0.0, 1.0, 0.0), mat: white }),
        Box::new(Plane { point: Vec3::new(0.0, 2.0, 0.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: white }),
        Box::new(Plane { point: Vec3::new(-3.0, 1.0, 0.0), normal: Vec3::new(1.0, 0.0, 0.0), mat: red }),
        Box::new(Plane { point: Vec3::new(3.0, 1.0, 0.0), normal: Vec3::new(-1.0, 0.0, 0.0), mat: green }),
        Box::new(Plane { point: Vec3::new(0.0, 1.0, 2.0), normal: Vec3::new(0.0, 0.0, -1.0), mat: white }),
        Box::new(Sphere { center: Vec3::new(0.5, 0.5, 0.5), radius: 0.4, mat: yellow }),
        Box::new(Sphere { center: Vec3::new(-0.3, 0.5, 0.1), radius: 0.4, mat: white }),
        Box::new(Plane { point: Vec3::new(0.0, 1.9, 1.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: light }),
    ];

    let cam = Cam::new(Vec3::new(0.0, 1.0, -1.0), 90.0);
    let buffer = cam.render(&scene, width, height);
    
    // Save high-res screenshot
    if let Err(e) = buffer.save_as_png("screenshot.png") {
        eprintln!("Failed to save screenshot: {}", e);
    }

    let frame_string = buffer_to_string(&buffer);
    execute!(stdout, crossterm::cursor::MoveTo(0, 0))?;
    writeln!(stdout, "{}", frame_string).unwrap();
    stdout.flush()?;

    loop {
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                    break;
                }
            }
        }
    }

    execute!(stdout, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}
