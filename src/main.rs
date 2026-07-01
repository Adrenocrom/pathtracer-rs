use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, ClearType, Clear},
};
use rand::Rng;
use image::{RgbImage, Rgb};
use rayon::prelude::*;
use std::io::{self, Write};

// --- VEC3 UTILS ---
#[derive(Clone, Copy, Debug, PartialEq)]
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

    fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
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

const VEC_ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

// --- CONFIGURATION ---
const SAMPLES_PREVIEW: usize = 16; 
const SAMPLES_FHD: usize = 256;
const MAX_DEPTH: i32 = 4;

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
    lookat: Vec3,
    fov: f32,
}

impl Cam {
    fn new(origin: Vec3, lookat: Vec3, fov_deg: f32) -> Self {
        Self {
            origin,
            lookat,
            fov: fov_deg.to_radians(),
        }
    }

    fn render(&self, scene: &[Box<dyn Intersectable>], width: usize, height: usize, samples: usize) -> PixelBuffer {
        let aspect = width as f32 / height as f32;
        let scale = (self.fov * 0.5).tan();

        let forward = (self.lookat - self.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

        let pixels: Vec<Vec3> = (0..height).into_par_iter().flat_map(|y| {
            (0..width).into_iter().map(|x| {
                let mut color = VEC_ZERO;
                for _ in 0..samples {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;
                    
                    let px = (u * 2.0 - 1.0) * aspect * scale;
                    let py = -(v * 2.0 - 1.0) * scale;
                    
                    let dir = (right * px + up * py + forward).normalize();
                    let ray = Ray { origin: self.origin, direction: dir };
                    color = color + trace(&ray, scene, MAX_DEPTH);
                }
                color / samples as f32
            }).collect::<Vec<_>>()
        }).collect();

        PixelBuffer {
            width,
            height,
            pixels,
        }
    }

    fn move_forward(&mut self, dist: f32) {
        let forward = (self.lookat - self.origin).normalize();
        self.origin = self.origin + forward * dist;
        self.lookat = self.lookat + forward * dist;
    }

    fn _move_right(&mut self, dist: f32) {
        let forward = (self.lookat - self.origin).normalize();
        let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).normalize();
        self.origin = self.origin + right * dist;
        self.lookat = self.lookat + right * dist;
    }

    fn rotate(&mut self, angle_rad: f32) {
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        
        let rotate_vec = |v: Vec3| -> Vec3 {
            Vec3::new(
                v.x * cos_a - v.z * sin_a,
                v.y,
                v.x * sin_a + v.z * cos_a,
            )
        };

        let rel_lookat = self.lookat - self.origin;
        let rotated_lookat = rotate_vec(rel_lookat);
        self.lookat = self.origin + rotated_lookat;
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
        return VEC_ZERO;
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

    fn apply_filters(&mut self) {
        let width = self.width;
        let height = self.height;
        let original_pixels = self.pixels.clone();

        // Bilateral Filter Parameters
        let sigma_spatial = 1.0; 
        let sigma_range = 0.15;  

        let mut filtered = vec![VEC_ZERO; width * height];

        for y in 0..height {
            for x in 0..width {
                let center_color = original_pixels[y * width + x];
                let mut sum_color = VEC_ZERO;
                let mut sum_weight = 0.0;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;

                        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                            let neighbor_color = original_pixels[(ny as usize * width) + nx as usize];
                            
                            let dist_sq = (dx * dx + dy * dy) as f32;
                            let spatial_w = (-dist_sq / (2.0 * sigma_spatial * sigma_spatial)).exp();

                            let color_diff = neighbor_color - center_color;
                            let color_dist_sq = color_diff.length_squared();
                            let range_w = (-color_dist_sq / (2.0 * sigma_range * sigma_range)).exp();

                            let weight = spatial_w * range_w;
                            sum_color = sum_color + neighbor_color * weight;
                            sum_weight += weight;
                        }
                    }
                }
                filtered[y * width + x] = if sum_weight > 0.0 { sum_color / sum_weight } else { center_color };
            }
        }

        for i in 0..self.pixels.len() {
            let p = filtered[i];
            let mapped = Vec3::new(
                p.x / (1.0 + p.x),
                p.y / (1.0 + p.y),
                p.z / (1.0 + p.z),
            );
            self.pixels[i] = Vec3::new(
                mapped.x.powf(1.0 / 2.2),
                mapped.y.powf(1.0 / 2.2),
                mapped.z.powf(1.0 / 2.2),
            );
        }
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
                VEC_ZERO
            };

            let mut r_top = (top_pixel.x.clamp(0.0, 1.0) * 255.0) as u8;
            let mut g_top = (top_pixel.y.clamp(0.0, 1.0) * 255.0) as u8;
            let mut b_top = (top_pixel.z.clamp(0.0, 1.0) * 255.0) as u8;
            if r_top == 0 && g_top == 0 && b_top == 0 {
                r_top = 1;
                g_top = 1;
                b_top = 1;
            }

            let mut r_bot = (bottom_pixel.x.clamp(0.0, 1.0) * 255.0) as u8;
            let mut g_bot = (bottom_pixel.y.clamp(0.0, 1.0) * 255.0) as u8;
            let mut b_bot = (bottom_pixel.z.clamp(0.0, 1.0) * 255.0) as u8;
            if r_bot == 0 && g_bot == 0 && b_bot == 0 {
                r_bot = 1;
                g_bot = 1;
                b_bot = 1;
            }

            output.push_str(&format!(
                "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                r_top, g_top, b_top, r_bot, g_bot, b_bot
            ));
        }
        output.push_str("\x1b[0m\r\n");
    }
    if output.ends_with("\r\n") {
        output.truncate(output.len() - 2);
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

    let white = Material { albedo: Vec3::new(0.5, 0.5, 0.5), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let red = Material { albedo: Vec3::new(1.1, 0.1, 0.1), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let green = Material { albedo: Vec3::new(0.1, 0.5, 0.1), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let yellow = Material { albedo: Vec3::new(0.5, 0.5, 0.1), emission: VEC_ZERO, mat_type: MaterialType::Diffuse };
    let light = Material { albedo: VEC_ZERO, emission: Vec3::new(1.5, 1.5, 1.4), mat_type: MaterialType::Emissive };

    let scene: Vec<Box<dyn Intersectable>> = vec![
        Box::new(Plane { point: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(0.0, 1.0, 0.0), mat: white }),
        Box::new(Plane { point: Vec3::new(0.0, 2.0, 0.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: white }),
        Box::new(Plane { point: Vec3::new(-3.0, 1.0, 0.0), normal: Vec3::new(1.0, 0.0, 0.0), mat: red }),
        Box::new(Plane { point: Vec3::new(3.0, 1.0, 0.0), normal: Vec3::new(-1.0, 0.0, 0.0), mat: green }),
        Box::new(Plane { point: Vec3::new(0.0, 1.0, 2.0), normal: Vec3::new(0.0, 0.0, -1.0), mat: white }),
        Box::new(Sphere { center: Vec3::new(0.5, 0.5, 0.5), radius: 0.4, mat: yellow }),
        Box::new(Sphere { center: Vec3::new(-0.3, 0.5, 0.1), radius: 0.4, mat: white }),
        Box::new(Plane { point: Vec3::new(0.0, 2.0, 1.0), normal: Vec3::new(0.0, -1.0, 0.0), mat: white }),
        Box::new(Sphere { center: Vec3::new(0.0, 1.9, 1.0), radius: 0.1, mat: light }),
    ];

    let mut cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 1.0, 0.0), 90.0);
    let mut needs_render = true;
    
    loop {
        if needs_render {
            let mut buffer = cam.render(&scene, width, height, SAMPLES_PREVIEW);
            buffer.apply_filters();
            let frame_string = buffer_to_string(&buffer);
            execute!(stdout, cursor::MoveTo(0, 0))?;
            writeln!(stdout, "{}", frame_string).unwrap();
            stdout.flush()?;
            needs_render = false;
        }

        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('p') => {
                        write!(stdout, "\r\nRendering FHD screenshot... ").unwrap();
                        stdout.flush()?;
                        let fhd_buffer = cam.render(&scene, 1920, 1080, SAMPLES_FHD);
                        if let Err(e) = fhd_buffer.save_as_png("screenshot.png") {
                            write!(stdout, "Failed to save: {}", e).unwrap();
                        } else {
                            write!(stdout, "Saved to screenshot.png!").unwrap();
                        }
                        stdout.flush()?;
                        std::thread::sleep(std::time::Duration::from_secs(2));
                    }
                    KeyCode::Char('w') => {
                        cam.move_forward(0.1);
                        needs_render = true;
                    }
                    KeyCode::Char('s') => {
                        cam.move_forward(-0.1);
                        needs_render = true;
                    }
                    KeyCode::Char('a') => {
                        cam.rotate(0.05);
                        needs_render = true;
                    }
                    KeyCode::Char('d') => {
                        cam.rotate(-0.05);
                        needs_render = true;
                    }
                    _ => {}
                }
            }
        }
    }

    execute!(stdout, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}
