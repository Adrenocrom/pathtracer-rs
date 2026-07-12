use crate::math::{Vec3, VEC_ZERO};
use rand::Rng;

// --- CAMERA ---
pub struct Cam {
    pub origin: Vec3,
    pub lookat: Vec3,
    pub fov_deg: f32, 
}

impl Cam {
    pub fn new(origin: Vec3, lookat: Vec3, fov_deg: f32) -> Self {
        Self {
            origin,
            lookat,
            fov_deg,
        }
    }

    /// Render the scene using BDPT. `lights` is a list of (position, area) pairs for light sampling.
    pub fn render(
        &self,
        scene: &dyn crate::geometry::Intersectable,
        width: usize,
        height: usize,
        samples: usize,
        max_depth: i32,
        lights: &[(Vec3, f32)],
    ) -> crate::output::PixelBuffer {
        let aspect = width as f32 / height as f32;
        let theta = self.fov_deg.to_radians();
        let h = (theta * 0.5).tan(); 
        let w = aspect * h;

        let forward = (self.lookat - self.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

        // One RNG per thread for performance
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
                    
                    let ray = crate::geometry::Ray { origin: self.origin, direction: dir };
                    color = color + crate::bdpt::bdpt_trace(&ray, scene, max_depth, lights, &mut rng);
                }
                color / samples as f32
            }).collect::<Vec<_>>()
        }).collect();

        crate::output::PixelBuffer {
            width,
            height,
            pixels,
        }
    }

    pub fn move_forward(&mut self, dist: f32) {
        let forward = (Vec3::new(self.lookat.x, 1.0, self.lookat.z) - self.origin).normalize();
        self.origin = self.origin + forward * dist;
        self.lookat = self.lookat + forward * dist;
    }

    pub fn rotate(&mut self, angle_rad: f32) {
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
