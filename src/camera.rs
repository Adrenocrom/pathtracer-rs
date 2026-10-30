use crate::math::{Vec3, VEC_ZERO};
use rand::Rng;
use rayon::prelude::*;

// --- CAMERA ---
#[derive(Clone, Copy)]
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

    /// Orthonormal camera basis (forward, right, up).
    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.lookat - self.origin).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();
        (forward, right, up)
    }

    /// Move along the camera's horizontal forward direction (the view
    /// direction projected onto the ground plane), keeping the lookat at
    /// the same relative offset. Walk-style: pitching up/down does not
    /// make the camera fly.
    ///
    /// The projection zeroes `forward.y` instead of the old hack that
    /// rebuilt the forward vector from a lookat with `y` forced to 1.0 —
    /// which only moved horizontally because the demo camera happened to
    /// start at height 1 and sank at any other height.
    pub fn move_forward(&mut self, dist: f32) {
        let rel = self.lookat - self.origin;
        let flat = Vec3::new(rel.x, 0.0, rel.z);
        if flat.length_squared() < 1e-12 {
            return; // Looking straight up/down: no horizontal direction.
        }
        let forward = flat.normalize();
        self.origin = self.origin + forward * dist;
        self.lookat = self.lookat + forward * dist;
    }

    /// Move sideways along the camera's right axis.
    pub fn strafe(&mut self, dist: f32) {
        let (_, right, _) = self.basis();
        self.origin = self.origin + right * dist;
        self.lookat = self.lookat + right * dist;
    }

    /// Yaw: rotate the lookat around the world up axis.
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

    /// Pitch: rotate the lookat around the camera's right axis. Clamped so
    /// the forward direction never becomes vertical (the world-up-based
    /// basis degenerates there).
    pub fn pitch(&mut self, angle_rad: f32) {
        let (_, right, _) = self.basis();
        let rel = self.lookat - self.origin;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        // Rodrigues rotation of `rel` around `right`. The sign is chosen
        // so positive angles look up: rotating `rel` by `+θ` around
        // `right` sends it toward `rel × right`, which points up when the
        // camera looks along -z with right = -x.
        let rotated =
            rel * cos_a + rel.cross(right) * sin_a + right * (right.dot(rel) * (1.0 - cos_a));

        // Clamp the elevation: keep the forward direction at least ~1°
        // away from straight up/down so `basis()` stays well-defined.
        let forward = rotated.normalize();
        let max_elev = (89.0f32).to_radians();
        let elev = forward.y.asin();
        let clamped = if elev > max_elev {
            let flat = Vec3::new(forward.x, 0.0, forward.z).normalize();
            Vec3::new(flat.x, max_elev.tan(), flat.z).normalize()
        } else if elev < -max_elev {
            let flat = Vec3::new(forward.x, 0.0, forward.z).normalize();
            Vec3::new(flat.x, -max_elev.tan(), flat.z).normalize()
        } else {
            forward
        };
        self.lookat = self.origin + clamped * rel.length();
    }

    /// Render the scene using BDPT. `lights` are the sampleable area lights.
    pub fn render(
        &self,
        scene: &dyn crate::geometry::Intersectable,
        width: usize,
        height: usize,
        samples: usize,
        max_depth: usize,
        lights: &[crate::geometry::AreaLight],
    ) -> crate::output::PixelBuffer {
        let aspect = width as f32 / height as f32;
        let theta = self.fov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let w = aspect * h;

        let (forward, right, up) = self.basis();

        // One RNG per thread for performance
        let pixels: Vec<Vec3> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                let mut rng = rand::thread_rng();
                (0..width)
                    .into_iter()
                    .map(move |x| {
                        let mut color = VEC_ZERO;

                        for _ in 0..samples {
                            let u_jitter = (x as f32 + rng.gen::<f32>()) / width as f32;
                            let v_jitter = (y as f32 + rng.gen::<f32>()) / height as f32;

                            let px = (u_jitter * 2.0 - 1.0) * w;
                            let py = -(v_jitter * 2.0 - 1.0) * h;

                            let dir = (right * px + up * py + forward).normalize();

                            let ray = crate::geometry::Ray {
                                origin: self.origin,
                                direction: dir,
                            };
                            color = color
                                + crate::bdpt::bdpt_trace(&ray, scene, max_depth, lights, &mut rng);
                        }
                        color / samples as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        crate::output::PixelBuffer {
            width,
            height,
            pixels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_forward_is_horizontal_at_any_height() {
        // Regression for the old `y = 1.0` hack: the horizontal forward
        // was computed from a lookat with y forced to 1.0, so a camera
        // at any other height drifted vertically while moving.
        let mut cam = Cam::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 2.0, -1.0), 60.0);
        cam.move_forward(0.5);
        assert!(
            (cam.origin.y - 2.0).abs() < 1e-5,
            "camera sank: {:?}",
            cam.origin
        );
        assert!((cam.origin.z + 0.5).abs() < 1e-5);

        // Walk-style: pitched up and moving forward must stay level.
        let mut cam = Cam::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 60.0);
        cam.move_forward(1.0);
        assert!(
            (cam.origin.y - 1.0).abs() < 1e-5,
            "walk move climbed: {:?}",
            cam.origin
        );
    }

    #[test]
    fn strafe_moves_along_the_right_axis() {
        let mut cam = Cam::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, -1.0), 60.0);
        cam.strafe(1.0);
        // Strafing must keep height and view direction, and move sideways.
        assert!((cam.origin.y - 1.0).abs() < 1e-5);
        let rel = cam.lookat - cam.origin;
        assert!(
            (rel.z + 1.0).abs() < 1e-5,
            "view direction changed: {rel:?}"
        );
        assert!(
            cam.origin.x.abs() > 0.9,
            "did not move sideways: {:?}",
            cam.origin
        );
    }

    #[test]
    fn pitch_rotates_and_clamps_near_vertical() {
        let mut cam = Cam::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, -1.0), 60.0);
        let dist_before = (cam.lookat - cam.origin).length();
        cam.pitch(0.5);
        let rel = cam.lookat - cam.origin;
        assert!(rel.y > 0.0, "pitch up did not raise the lookat: {rel:?}");
        // Pitch rotates the lookat; the distance to it must be preserved.
        assert!(
            (rel.length() - dist_before).abs() < 1e-4,
            "pitch changed the distance: {rel:?}"
        );

        // Hammering pitch up must never reach straight up (the basis
        // degenerates there) and must stay finite.
        for _ in 0..100 {
            cam.pitch(0.3);
        }
        let forward = (cam.lookat - cam.origin).normalize();
        assert!(forward.y < 0.9999, "pitch clamp failed: {forward:?}");
        assert!(forward.y.is_finite());
    }
}
