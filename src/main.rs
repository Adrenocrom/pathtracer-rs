mod gpu;
mod gpu;

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

// ... [rest of the file unchanged until main]

// --- VEC3 UTILS ---
#[derive(Clone, Copy, Debug, PartialEq, Default)]
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

    fn min(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x.min(other.x), self.y.min(other.y), self.z.min(other.z))
    }

    fn max(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x.max(other.x), self.y.max(other.y), self.z.max(other.z))
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
const SAMPLES_PREVIEW: usize = 32; 
const SAMPLES_FHD: usize = 6400;
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

// --- BOUNDING BOX ---
#[derive(Clone, Copy, Debug)]
struct BBox {
    min: Vec3,
    max: Vec3,
}

impl BBox {
    fn empty() -> Self {
        BBox {
            min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    fn surround(self, other: BBox) -> BBox {
        BBox {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn intersect(&self, ray: &Ray) -> bool {
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
    fn bounding_box(&self) -> BBox;
    fn get_emission(&self) -> Vec3 { VEC_ZERO }
    fn get_position(&self) -> Vec3 { VEC_ZERO }
    fn is_light(&self) -> bool { false }
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

    fn bounding_box(&self) -> BBox {
        BBox {
            min: self.center - Vec3::new(self.radius, self.radius, self.radius),
            max: self.center + Vec3::new(self.radius, self.radius, self.radius),
        }
    }

    fn get_emission(&self) -> Vec3 { self.mat.emission }
    fn get_position(&self) -> Vec3 { self.center }
    fn is_light(&self) -> bool { matches!(self.mat.mat_type, MaterialType::Emissive) }
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

    fn bounding_box(&self) -> BBox {
        // Planes are infinite, but for BVH we use a very large box
        BBox {
            min: Vec3::new(-1e6, -1e6, -1e6),
            max: Vec3::new(1e6, 1e6, 1e6),
        }
    }

    fn get_emission(&self) -> Vec3 { self.mat.emission }
    fn get_position(&self) -> Vec3 { self.point }
    fn is_light(&self) -> bool { matches!(self.mat.mat_type, MaterialType::Emissive) }
}

struct Cube {
    min: Vec3,
    max: Vec3,
    mat: Material,
}

impl Cube {
    fn new(center: Vec3, size: f32, mat: Material) -> Self {
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
            let t = if t_min < 0.001 { t_max } else { t_min };
            if t < 0.001 { return None; }

            let p = ray.at(t);
            
            // Calculate normal based on which face was hit
            let mut normal = VEC_ZERO;
            let eps = 0.001;
            if (p.x - self.min.x).abs() < eps { normal = Vec3::new(-1.0, 0.0, 0.0); }
            else if (p.x - self.max.x).abs() < eps { normal = Vec3::new(1.0, 0.0, 0.0); }
            else if (p.y - self.min.y).abs() < eps { normal = Vec3::new(0.0, -1.0, 0.0); }
            else if (p.y - self.max.y).abs() < eps { normal = Vec3::new(0.0, 1.0, 0.0); }
            else if (p.z - self.min.z).abs() < eps { normal = Vec3::new(0.0, 0.0, -1.0); }
            else if (p.z - self.max.z).abs() < eps { normal = Vec3::new(0.0, 0.0, 1.0); }

            Some(HitRecord { t, p, normal, mat: self.mat })
        } else {
            None
        }
    }

    fn bounding_box(&self) -> BBox {
        BBox { min: self.min, max: self.max }
    }

    fn get_emission(&self) -> Vec3 { self.mat.emission }
    fn get_position(&self) -> Vec3 { (self.min + self.max) * 0.5 }
    fn is_light(&self) -> bool { matches!(self.mat.mat_type, MaterialType::Emissive) }
}

struct BVHNode {
    bbox: BBox,
    left: Box<dyn Intersectable>,
    right: Box<dyn Intersectable>,
}

impl BVHNode {
    fn build(mut objects: Vec<Box<dyn Intersectable>>) -> Box<dyn Intersectable> {
        if objects.len() == 0 {
            panic!("BVH build with no objects");
        }
        if objects.len() == 1 {
            return objects.pop().unwrap();
        }

        // Calculate bounds of all objects in the node
        let mut total_bbox = BBox::empty();
        for obj in &objects {
            total_bbox = total_bbox.surround(obj.bounding_box());
        }

        // Split along the longest axis
        let dx = total_bbox.max.x - total_bbox.min.x;
        let dy = total_bbox.max.y - total_bbox.min.y;
        let dz = total_bbox.max.z - total_bbox.min.z;

        if dx > dy && dx > dz {
            objects.sort_by(|a, b| a.bounding_box().min.x.partial_cmp(&b.bounding_box().min.x).unwrap());
        } else if dy > dz {
            objects.sort_by(|a, b| a.bounding_box().min.y.partial_cmp(&b.bounding_box().min.y).unwrap());
        } else {
            objects.sort_by(|a, b| a.bounding_box().min.z.partial_cmp(&b.bounding_box().min.z).unwrap());
        }

        let mid = objects.len() / 2;
        let right_objs = objects.split_off(mid);
        
        let left = Self::build(objects);
        let right = Self::build(right_objs);

        Box::new(BVHNode {
            bbox: total_bbox,
            left,
            right,
        })
    }
}

impl Intersectable for BVHNode {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        if !self.bbox.intersect(ray) {
            return None;
        }

        let left_hit = self.left.intersect(ray);
        let right_hit = self.right.intersect(ray);

        match (left_hit, right_hit) {
            (Some(l), Some(r)) => if l.t < r.t { Some(l) } else { Some(r) },
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    fn bounding_box(&self) -> BBox {
        self.bbox
    }
}

// --- CAMERA ---
struct Cam {
    origin: Vec3,
    lookat: Vec3,
    fov_deg: f32, // Vertical Field of View
}

impl Cam {
    fn new(origin: Vec3, lookat: Vec3, fov_deg: f32) -> Self {
        Self {
            origin,
            lookat,
            fov_deg,
        }
    }

    fn render(&self, scene: &dyn Intersectable, width: usize, height: usize, samples: usize) -> PixelBuffer {
        let aspect = width as f32 / height as f32;
        let theta = self.fov_deg.to_radians();
        let h = (theta * 0.5).tan(); 
        let w = aspect * h;

        let forward = (self.lookat - self.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

        let pixels: Vec<Vec3> = (0..height).into_par_iter().flat_map(|y| {
            let mut rng = rand::thread_rng();
            (0..width).into_iter().map(move |x| {
                let mut color = VEC_ZERO;

                for _ in 0..samples {
                    let u_jitter = (x as f32 + rng.gen::<f32>()) / width as f32;
                    let v_jitter = (y as f32 + rng.gen::<f32>()) / height as f32;

                    let px = (u_jitter * 2.0 - 1.0) * w;
                    let py = -(v_jitter * 2.0 - 1.0) * h;

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
        let forward = (Vec3::new(self.lookat.x, 1.0, self.lookat.z) - self.origin).normalize();
        self.origin = self.origin + forward * dist;
        self.lookat = self.lookat + forward * dist;
    }

    fn rotate(&mut self, angle_rad: f32) {
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        
        let rel_lookat = self.lookat - self.origin;
        let rotated_lookat = Vec3::new(
            rel_lookat.x * cos_a - rel_lookat.z * sin_a,
            rel_lookat.y,
            rel_lookat.x * sin_a + rel_lookat.z * cos_a,
        );
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

fn trace(ray: &Ray, scene: &dyn Intersectable, depth: i32) -> Vec3 {
    if depth <= 0 {
        return VEC_ZERO;
    }

    if let Some(hit) = scene.intersect(ray) {
        let emission = hit.mat.emission;
        
        if let MaterialType::Emissive = hit.mat.mat_type {
            return emission;
        }

        // --- INDIRECT LIGHTING (Recursive Path Trace) ---
        let target = hit.normal + random_unit_vector();
        let scattered_ray = Ray {
            origin: hit.p + hit.normal * 0.001,
            direction: target.normalize(),
        };

        let indirect = trace(&scattered_ray, scene, depth - 1);
        
        // Final color = Emission + Albedo * Indirect
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

        let sigma_spatial = 1.0; 
        let sigma_range = 0.15;  

        let filtered: Vec<Vec3> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                (0..width).into_iter().map(|x| {
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
                    if sum_weight > 0.0 { sum_color / sum_weight } else { center_color }
                }).collect::<Vec<_>>()
            })
            .collect();

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

async fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, Clear(ClearType::All))?;

    let (term_w, term_h) = {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 40));
        (w as usize, h as usize)
    };

    let width = term_w;
    let height = term_h * 2;

    let mut cam = Cam::new(Vec3::new(0.0, 1.0, -1.5), Vec3::new(0.0, 0.5, 0.0), 90.0);
    let mut needs_render = true;
    let mut filter_enabled = true;
    
    loop {
        if needs_render {
            let mut buffer = gpu::render_gpu(width, height, SAMPLES_PREVIEW, &cam).await;
            if filter_enabled {
                buffer.apply_filters();
            }
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
                    KeyCode::Char('f') => {
                        filter_enabled = !filter_enabled;
                        needs_render = true;
                    },
                    KeyCode::Char('p') => {
                        write!(stdout, "\r\nRendering FHD screenshot... ").unwrap();
                        stdout.flush()?;
                        let mut fhd_buffer = gpu::render_gpu(1920, 1080, SAMPLES_FHD, &cam).await;
                        if filter_enabled {
                            fhd_buffer.apply_filters();
                        }
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
