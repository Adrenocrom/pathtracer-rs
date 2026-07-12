use crate::math::{Vec3, VEC_ZERO};
use rand::Rng;

// --- VERTEX (for path storage) ---
#[derive(Clone)]
pub struct Vertex {
    pub p: Vec3,
    pub normal: Vec3,
    pub throughput: Vec3,
    pub emission: Vec3,
}

/// Generate a random unit vector using spherical coordinates (uniform on sphere).
fn random_unit_vector(rng: &mut impl Rng) -> Vec3 {
    let u = rng.gen::<f32>();
    let v = rng.gen::<f32>();
    let theta = 2.0 * std::f32::consts::PI * u;
    let phi = (2.0 * v - 1.0).acos();
    Vec3::new(
        phi.sin() * theta.cos(),
        phi.sin() * theta.sin(),
        phi.cos(),
    )
}

/// Generate a random direction in the hemisphere around `normal` (cosine-weighted).
fn random_hemisphere_direction(normal: Vec3, rng: &mut impl Rng) -> Vec3 {
    let mut dir = random_unit_vector(rng);
    // If the random vector points opposite to normal, flip it
    if dir.dot(normal) < 0.0 {
        dir = -dir;
    }
    dir.normalize()
}

/// Build an eye path from a ray through the scene.
fn build_eye_path(
    origin: Vec3,
    direction: Vec3,
    scene: &dyn crate::geometry::Intersectable,
    max_depth: i32,
) -> Vec<Vertex> {
    let mut path = vec![Vertex {
        p: origin,
        normal: -direction.normalize(), // Camera vertex has "normal" pointing back along ray
        throughput: Vec3::new(1.0, 1.0, 1.0),
        emission: VEC_ZERO,
    }];

    let mut current_ray = crate::geometry::Ray { origin, direction };
    let mut current_throughput = Vec3::new(1.0, 1.0, 1.0);

    for _ in 0..max_depth {
        if let Some(hit) = scene.intersect(&current_ray) {
            let vertex = Vertex {
                p: hit.p,
                normal: hit.normal,
                throughput: current_throughput,
                emission: hit.mat.emission,
            };
            path.push(vertex);

            // If we hit an emissive surface, stop (light found!)
            if hit.mat.mat_type == crate::geometry::MaterialType::Emissive {
                break;
            }

            let next_dir = random_hemisphere_direction(hit.normal, &mut rand::thread_rng());
            current_throughput = current_throughput * hit.mat.albedo;
            current_ray = crate::geometry::Ray {
                origin: hit.p + hit.normal * 0.001, // Offset to avoid self-intersection
                direction: next_dir,
            };
        } else {
            break;
        }
    }

    path
}

/// Build a light path from a given starting point and direction.
fn build_light_path(
    origin: Vec3,
    direction: Vec3,
    scene: &dyn crate::geometry::Intersectable,
    max_depth: i32,
) -> Vec<Vertex> {
    let mut path = vec![Vertex {
        p: origin,
        normal: -direction.normalize(),
        throughput: Vec3::new(1.0, 1.0, 1.0),
        emission: VEC_ZERO, // Light vertex starts with no emission (emission is added at connection)
    }];

    let mut current_ray = crate::geometry::Ray { origin, direction };
    let mut current_throughput = Vec3::new(1.0, 1.0, 1.0);

    for _ in 0..max_depth {
        if let Some(hit) = scene.intersect(&current_ray) {
            let vertex = Vertex {
                p: hit.p,
                normal: hit.normal,
                throughput: current_throughput,
                emission: hit.mat.emission,
            };
            path.push(vertex);

            // If we hit an emissive surface, this is a light connection point
            if hit.mat.mat_type == crate::geometry::MaterialType::Emissive {
                break;
            }

            let next_dir = random_hemisphere_direction(hit.normal, &mut rand::thread_rng());
            current_throughput = current_throughput * hit.mat.albedo;
            current_ray = crate::geometry::Ray {
                origin: hit.p + hit.normal * 0.001,
                direction: next_dir,
            };
        } else {
            break;
        }
    }

    path
}

/// Check if a segment between two points is visible (no occlusion).
fn is_visible(eye_p: Vec3, light_p: Vec3, scene: &dyn crate::geometry::Intersectable) -> bool {
    let dir = light_p - eye_p;
    let dist = dir.length();
    if dist < 0.001 { return true; } // Too close, consider visible

    let ray_dir = dir.normalize();
    let test_ray = crate::geometry::Ray {
        origin: eye_p + ray_dir * 0.001, // Small offset
        direction: ray_dir,
    };

    if let Some(hit) = scene.intersect(&test_ray) {
        // If something hits before the light point, it's occluded
        hit.t < dist - 0.001
    } else {
        true // No intersection means visible
    }
}

/// Compute the geometry term between two points (diffuse BRDF).
fn geometry_term(eye_normal: Vec3, eye_p: Vec3, light_normal: Vec3, light_p: Vec3) -> f32 {
    let dir = light_p - eye_p;
    let dist_sq = dir.length_squared();
    if dist_sq < 0.0001 { return 0.0; }

    let cos_theta = eye_normal.dot(dir.normalize()).max(0.0);
    let cos_phi = light_normal.dot(-dir.normalize()).max(0.0);

    (cos_theta * cos_phi) / dist_sq
}

/// Sample a random light from the list using area-weighted sampling.
fn sample_light(lights: &[(Vec3, f32)], rng: &mut impl Rng) -> (usize, Vec3) {
    let total_area: f32 = lights.iter().map(|(_, area)| *area).sum();
    let mut r = rng.gen::<f32>() * total_area;
    
    for (i, (_, area)) in lights.iter().enumerate() {
        if r < *area {
            return (i, lights[i].0);
        }
        r -= *area;
    }
    // Fallback to last light
    (lights.len() - 1, lights[lights.len() - 1].0)
}

/// Main BDPT function: trace a ray and compute lighting using bidirectional path tracing.
pub fn bdpt_trace(
    ray: &crate::geometry::Ray, 
    scene: &dyn crate::geometry::Intersectable, 
    max_depth: i32,
    lights: &[(Vec3, f32)],
    rng: &mut impl Rng,
) -> Vec3 {
    // 1. Build eye path from camera ray
    let eye_path = build_eye_path(ray.origin, ray.direction, scene, max_depth);

    // 2. Sample a light and build a light path from it
    let mut total_color = VEC_ZERO;

    if !lights.is_empty() {
        let (light_idx, _light_pos) = sample_light(lights, rng);
        let light_area = lights[light_idx].1;
        
        // Sample a random point on the light surface (simplified: use center + small offset)
        let light_center = lights[light_idx].0;
        let light_offset = Vec3::new(
            (rng.gen::<f32>() - 0.5) * 0.1,
            (rng.gen::<f32>() - 0.5) * 0.1,
            (rng.gen::<f32>() - 0.5) * 0.1,
        );
        let light_origin = light_center + light_offset;

        // Build light path from this sampled point
        let light_dir = random_unit_vector(rng);
        let light_path = build_light_path(light_origin, light_dir, scene, max_depth);

        // Add emission from the light vertex itself (if it's emissive)
        if let Some(first_v) = light_path.first() {
            if first_v.emission != VEC_ZERO {
                total_color = total_color + first_v.emission;
            }
        }

        // 3. Connect eye path vertices with light path vertices
        for eye_v in &eye_path {
            for light_v in &light_path {
                if light_v.emission == VEC_ZERO {
                    continue; // Only connect to actual lights
                }

                let dir = light_v.p - eye_v.p;
                let dist_sq = dir.length_squared();
                if dist_sq < 0.0001 { continue; }

                // Visibility check
                if !is_visible(eye_v.p, light_v.p, scene) {
                    continue;
                }

                // Geometry term (diffuse BRDF)
                let geo = geometry_term(eye_v.normal, eye_v.p, light_v.normal, light_v.p);
                
                // Contribution: throughput * emission * geometry / light_area
                let contribution = eye_v.throughput * light_v.emission * Vec3::new(geo, geo, geo) / light_area;
                total_color = total_color + contribution;
            }
        }
    }

    // If no paths connected and we only have the camera vertex, return background
    if total_color == VEC_ZERO && eye_path.len() == 1 {
        return Vec3::new(0.02, 0.02, 0.05); // Dark blue background
    }

    total_color
}
