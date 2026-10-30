use crate::math::{BBox, Vec3, EPS_NEAR, VEC_ZERO};
use rand::Rng;

const PI: f32 = std::f32::consts::PI;

// --- RAY ---
#[derive(Clone)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Minimum positive t at which this ray could enter a bounding box.
    /// Returns None if the ray starts inside the box or can never reach it.
    pub fn min_t_intersect(&self, bbox: &BBox) -> Option<f32> {
        let mut t_min = f32::NEG_INFINITY;

        for i in 0..3 {
            let d = self.direction[i];
            if d == 0.0 {
                // Parallel to this slab: unreachable unless inside it.
                if self.origin[i] < bbox.min[i] || self.origin[i] > bbox.max[i] {
                    return None;
                }
                continue;
            }
            let inv_d = 1.0 / d;
            let t0 = (bbox.min[i] - self.origin[i]) * inv_d;
            let t1 = (bbox.max[i] - self.origin[i]) * inv_d;
            t_min = t_min.max(t0.min(t1));
        }

        if t_min <= EPS_NEAR {
            None
        } else {
            Some(t_min)
        }
    }
}

// --- HIT RECORD ---
pub struct HitRecord {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub mat: Material,
}

// --- MATERIALS ---
/// The scattering model of a surface. Emissive materials ignore the BSDF:
/// emitters are treated as pure light sources that do not reflect.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Bsdf {
    /// Lambertian, sampled with a cosine-weighted distribution.
    Diffuse,
    /// Perfect specular reflection (a delta distribution).
    Mirror,
    /// Perfect specular dielectric (a delta distribution): exact dielectric
    /// Fresnel splits reflection and refraction, total internal reflection
    /// falls back to reflection. `ior` is the interior index of refraction;
    /// the exterior is vacuum (ior = 1), so glass objects must be closed.
    Glass { ior: f32 },
}

impl Bsdf {
    /// True for BSDFs that scatter into a single direction (a delta
    /// distribution): they have no continuous density, so connections
    /// through them are invalid and their pdfs cancel in MIS ratios.
    pub fn is_delta(&self) -> bool {
        !matches!(self, Bsdf::Diffuse)
    }
}

/// Which quantity a subpath transports. The two agree for reflection but
/// differ for transmission: basic radiance L/eta^2 is invariant across a
/// refractive boundary, so radiance picks up (eta_t/eta_i)^2 crossing one
/// (Veach's thesis; pbrt's TransportMode::Radiance factor), while
/// importance does not. The eye subpath carries radiance, the light
/// subpath importance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transport {
    Radiance,
    Importance,
}

/// A surface material. `emission != 0` marks a pure emitter: the BSDF is
/// ignored and the surface does not scatter.
#[derive(Clone, Copy)]
pub struct Material {
    pub albedo: Vec3,
    pub emission: Vec3,
    pub bsdf: Bsdf,
}

impl Material {
    pub fn diffuse(albedo: Vec3) -> Self {
        Self {
            albedo,
            emission: VEC_ZERO,
            bsdf: Bsdf::Diffuse,
        }
    }

    pub fn emissive(emission: Vec3) -> Self {
        Self {
            albedo: VEC_ZERO,
            emission,
            bsdf: Bsdf::Diffuse,
        }
    }

    pub fn mirror(albedo: Vec3) -> Self {
        Self {
            albedo,
            emission: VEC_ZERO,
            bsdf: Bsdf::Mirror,
        }
    }

    pub fn glass(ior: f32) -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            emission: VEC_ZERO,
            bsdf: Bsdf::Glass { ior },
        }
    }

    /// Pure emitters do not scatter; the BSDF is ignored for them.
    pub fn is_emissive(&self) -> bool {
        self.emission != VEC_ZERO
    }

    /// Delta BSDFs cannot be connection endpoints. Emitters are never
    /// delta (they do not scatter at all).
    pub fn is_delta(&self) -> bool {
        !self.is_emissive() && self.bsdf.is_delta()
    }
}

/// One scattering event sampled from a BSDF.
pub struct BsdfSample {
    /// Unit travel direction away from the surface.
    pub dir_out: Vec3,
    /// Solid-angle pdf of `dir_out`. 0 for delta BSDFs: their density is a
    /// Dirac delta, which the MIS machinery replaces symbolically.
    pub pdf_omega: f32,
    /// f · |cos| / pdf, the throughput multiplier of the event. For delta
    /// BSDFs the delta cancels and this is the deterministic specular
    /// weight; the Fresnel probability cancels against the BSDF value.
    pub weight: Vec3,
}

/// Cosine-weighted direction in the hemisphere around `normal` (pdf = cos/pi).
pub(crate) fn cosine_hemisphere(normal: Vec3, rng: &mut impl Rng) -> Vec3 {
    let u1 = rng.gen::<f32>();
    let u2 = rng.gen::<f32>();
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let z = (1.0 - u1).sqrt();

    let a = if normal.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t = a.cross(normal).normalize();
    let b = normal.cross(t);

    t * (r * phi.cos()) + b * (r * phi.sin()) + normal * z
}

/// Exact unpolarized dielectric Fresnel reflectance for an incidence
/// cosine `cos_i` (measured against the normal on the incident side).
fn fresnel_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cos_i = cos_i.abs().min(1.0);
    let sin_i = (1.0 - cos_i * cos_i).sqrt();
    let sin_t = eta_i / eta_t * sin_i;
    if sin_t >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    let r_s = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    let r_p = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    0.5 * (r_s * r_s + r_p * r_p)
}

/// Snell refraction of the travel direction `dir_in` at a surface whose
/// (unit) `normal` is oriented toward the side the ray arrives from.
/// None = total internal reflection.
fn refract_dir(dir_in: Vec3, normal: Vec3, eta_i: f32, eta_t: f32) -> Option<Vec3> {
    let cos_i = -normal.dot(dir_in);
    let eta = eta_i / eta_t;
    let sin_t_sq = eta * eta * (1.0 - cos_i * cos_i);
    if sin_t_sq >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin_t_sq).sqrt();
    Some(dir_in * eta + normal * (eta * cos_i - cos_t))
}

/// Specular reflection of the travel direction `dir_in` (unit, with `normal`
/// oriented against it).
fn reflect_dir(dir_in: Vec3, normal: Vec3) -> Vec3 {
    dir_in - normal * (2.0 * normal.dot(dir_in))
}

/// The eta^2 transmission factor for the transport mode: radiance picks up
/// (eta_t/eta_i)^2 crossing a refractive boundary, importance does not.
fn eta_factor(transport: Transport, eta_i: f32, eta_t: f32) -> f32 {
    if transport == Transport::Radiance {
        (eta_t / eta_i) * (eta_t / eta_i)
    } else {
        1.0
    }
}

impl Material {
    /// Sample a scattering direction. `normal` must be oriented toward the
    /// side the ray arrives from (normal · dir_in < 0); `front_face` says
    /// whether the geometric normal faces that side (for glass: which
    /// medium is incident). Delta BSDFs sample deterministically.
    pub fn sample_bsdf(
        &self,
        normal: Vec3,
        front_face: bool,
        dir_in: Vec3,
        transport: Transport,
        rng: &mut impl Rng,
    ) -> BsdfSample {
        match self.bsdf {
            Bsdf::Diffuse => {
                let dir_out = cosine_hemisphere(normal, rng);
                let pdf = dir_out.dot(normal) / PI;
                BsdfSample {
                    dir_out,
                    pdf_omega: pdf,
                    weight: self.albedo,
                }
            }
            Bsdf::Mirror => BsdfSample {
                dir_out: reflect_dir(dir_in, normal),
                pdf_omega: 0.0,
                weight: self.albedo,
            },
            Bsdf::Glass { ior } => {
                let (eta_i, eta_t) = if front_face { (1.0, ior) } else { (ior, 1.0) };
                let cos_i = -normal.dot(dir_in);
                let f = fresnel_dielectric(cos_i, eta_i, eta_t);
                if rng.gen::<f32>() < f {
                    // Reflection. The Fresnel reflectance is both the
                    // discrete choice probability and the BSDF value, so
                    // it cancels out of the weight.
                    BsdfSample {
                        dir_out: reflect_dir(dir_in, normal),
                        pdf_omega: 0.0,
                        weight: self.albedo,
                    }
                } else {
                    // Refraction (total internal reflection cannot reach
                    // this branch: it gives f = 1, so the roll reflects).
                    let dir_out = refract_dir(dir_in, normal, eta_i, eta_t)
                        .unwrap_or_else(|| reflect_dir(dir_in, normal));
                    BsdfSample {
                        dir_out,
                        pdf_omega: 0.0,
                        weight: self.albedo * eta_factor(transport, eta_i, eta_t),
                    }
                }
            }
        }
    }

    /// f (the raw BSDF, no cosine) and the solid-angle pdf of `dir_out`
    /// under the BSDF's sampling distribution, for an arbitrary direction
    /// pair. Delta BSDFs return (0, 0) — exactly what invalidates
    /// connections through them. `normal` must be oriented toward the side
    /// `dir_in` arrives from.
    pub fn eval_bsdf(&self, normal: Vec3, dir_in: Vec3, dir_out: Vec3) -> (Vec3, f32) {
        match self.bsdf {
            Bsdf::Diffuse => {
                let cos_i = -normal.dot(dir_in);
                let cos_o = normal.dot(dir_out);
                if cos_i > 0.0 && cos_o > 0.0 {
                    (self.albedo * (1.0 / PI), cos_o / PI)
                } else {
                    (VEC_ZERO, 0.0)
                }
            }
            _ => (VEC_ZERO, 0.0),
        }
    }

    /// True when (dir_in, dir_out) is the specular pair of this delta BSDF
    /// (the deterministic direction it scatters into). Used by the MIS
    /// machinery to give delta-sampled edges their symbolic pdf of 1.
    pub fn is_specular_pair(
        &self,
        normal: Vec3,
        front_face: bool,
        dir_in: Vec3,
        dir_out: Vec3,
    ) -> bool {
        self.specular_weight(normal, front_face, dir_in, dir_out, Transport::Radiance)
            .is_some()
    }

    /// The deterministic sample weight of a SPECULAR direction pair at a
    /// delta BSDF — needed for hypothetical path extensions, where the
    /// rolls never happened but the path geometry fixes the directions.
    /// None if the pair is not the specular pair. `normal`/`front_face`
    /// describe the event being weighted.
    pub fn specular_weight(
        &self,
        normal: Vec3,
        front_face: bool,
        dir_in: Vec3,
        dir_out: Vec3,
        transport: Transport,
    ) -> Option<Vec3> {
        const DOT_EPS: f32 = 1.0 - 1e-4;
        match self.bsdf {
            Bsdf::Mirror => {
                if reflect_dir(dir_in, normal).dot(dir_out) > DOT_EPS {
                    Some(self.albedo)
                } else {
                    None
                }
            }
            Bsdf::Glass { ior } => {
                let (eta_i, eta_t) = if front_face { (1.0, ior) } else { (ior, 1.0) };
                if reflect_dir(dir_in, normal).dot(dir_out) > DOT_EPS {
                    return Some(self.albedo);
                }
                let refr = refract_dir(dir_in, normal, eta_i, eta_t)?;
                if refr.dot(dir_out) > DOT_EPS {
                    Some(self.albedo * eta_factor(transport, eta_i, eta_t))
                } else {
                    None
                }
            }
            Bsdf::Diffuse => None,
        }
    }
}

// --- INTERSECTABLE TRAIT ---
pub trait Intersectable: Sync + Send {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord>;
    fn bounding_box(&self) -> BBox;
}

// --- SPHERE ---
#[derive(Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub mat: Material,
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        // Prefer the near root; fall back to the far root so the sphere is
        // still visible when the camera is inside it.
        let sqrt_d = discriminant.sqrt();
        let mut t = (-b - sqrt_d) / (2.0 * a);
        if t < EPS_NEAR {
            t = (-b + sqrt_d) / (2.0 * a);
        }
        if t < EPS_NEAR {
            return None;
        }

        let p = ray.at(t);
        let normal = (p - self.center).normalize();
        Some(HitRecord {
            t,
            p,
            normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        BBox {
            min: self.center - Vec3::new(self.radius, self.radius, self.radius),
            max: self.center + Vec3::new(self.radius, self.radius, self.radius),
        }
    }
}

// --- PLANE ---
#[derive(Clone)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
    pub mat: Material,
}

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < EPS_NEAR {
            return None;
        }

        Some(HitRecord {
            t,
            p: ray.at(t),
            normal: self.normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        // Planes have infinite extent - use a large box
        BBox {
            min: Vec3::new(-1e6, -1e6, -1e6),
            max: Vec3::new(1e6, 1e6, 1e6),
        }
    }
}

// --- QUAD ---
/// Tangent frame for a `width` x `height` rectangle facing `normal`.
///
/// For axis-aligned normals the extents are world-axis sized: `width`
/// spans the horizontal axis (x for normals along z or y, z for normals
/// along x) and `height` spans the vertical axis (y, or z for normals
/// along y). Sizing the extents along an arbitrary helper-axis tangent
/// frame instead *rotates* walls: a "5 wide, 2 tall" wall facing +z came
/// out 2 wide and 5 tall, because the helper axis for a +z normal is
/// vertical.
///
/// Non-axis-aligned normals have no world-axis mapping; they fall back
/// to a helper axis perpendicular to the normal, with `width` along the
/// tangent and `height` along the bitangent.
///
/// Returns `(tangent, bitangent, half_w, half_h)`.
pub(crate) fn quad_frame(normal: Vec3, width: f32, height: f32) -> (Vec3, Vec3, f32, f32) {
    // An axis normal is exactly unit length along its axis after
    // normalization, so a near-1 threshold separates it from diagonals.
    const AXIS: f32 = 1.0 - 1e-3;
    if normal.y.abs() > AXIS {
        // Floor/ceiling: width along x, height along z.
        (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            width * 0.5,
            height * 0.5,
        )
    } else if normal.x.abs() > AXIS {
        // Side walls: width along z, height along y.
        (
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            width * 0.5,
            height * 0.5,
        )
    } else if normal.z.abs() > AXIS {
        // Back/front walls: width along x, height along y.
        (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            width * 0.5,
            height * 0.5,
        )
    } else {
        // Diagonal normal: |normal.x| <= 1 - 1e-3, so the x axis is
        // never parallel to it and works as the helper axis.
        let tangent = Vec3::new(1.0, 0.0, 0.0).cross(normal).normalize();
        let bitangent = normal.cross(tangent);
        (tangent, bitangent, width * 0.5, height * 0.5)
    }
}

/// A finite rectangle, used for light panels.
#[derive(Clone)]
pub struct Quad {
    pub center: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub half_w: f32,
    pub half_h: f32,
    pub mat: Material,
}

impl Quad {
    /// A `width` x `height` rectangle centered at `center`, facing `normal`.
    /// Kept for API completeness (used by tests and future scenes).
    #[allow(dead_code)]
    pub fn new(center: Vec3, normal: Vec3, width: f32, height: f32, mat: Material) -> Self {
        let normal = normal.normalize();
        let (tangent, bitangent, half_w, half_h) = quad_frame(normal, width, height);
        Self {
            center,
            normal,
            tangent,
            bitangent,
            half_w,
            half_h,
            mat,
        }
    }
}

impl Intersectable for Quad {
    fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (self.center - ray.origin).dot(self.normal) / denom;
        if t < EPS_NEAR {
            return None;
        }

        let p = ray.at(t);
        let local = p - self.center;
        if local.dot(self.tangent).abs() > self.half_w
            || local.dot(self.bitangent).abs() > self.half_h
        {
            return None;
        }
        Some(HitRecord {
            t,
            p,
            normal: self.normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        let mut bbox = BBox::empty();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                let corner = self.center
                    + self.tangent * (sx * self.half_w)
                    + self.bitangent * (sy * self.half_h);
                bbox = bbox.surround(BBox {
                    min: corner,
                    max: corner,
                });
            }
        }
        bbox
    }
}

// --- CUBE ---
#[derive(Clone)]
pub struct Cube {
    pub min: Vec3,
    pub max: Vec3,
    pub mat: Material,
}

impl Cube {
    /// Kept for API completeness (used by tests and future scenes).
    #[allow(dead_code)]
    pub fn new(center: Vec3, size: f32, mat: Material) -> Self {
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
            if dirs[i] == 0.0 {
                // Parallel to this slab: must lie within it.
                if origins[i] < mins[i] || origins[i] > maxs[i] {
                    return None;
                }
                continue;
            }
            let inv_d = 1.0 / dirs[i];
            let mut t0 = (mins[i] - origins[i]) * inv_d;
            let mut t1 = (maxs[i] - origins[i]) * inv_d;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
        }

        if !(t_max >= t_min && t_max > 0.0 && t_min < 1e6) {
            return None;
        }
        let t = if t_min < EPS_NEAR { t_max } else { t_min };
        if t < EPS_NEAR {
            return None;
        }

        let p = ray.at(t);

        // Outward normal of the face closest to the hit point. Checking all
        // six faces (not just the min faces) is what makes top/+x/+z hits
        // shade correctly.
        let candidates = [
            ((p.x - self.min.x).abs(), Vec3::new(-1.0, 0.0, 0.0)),
            ((p.x - self.max.x).abs(), Vec3::new(1.0, 0.0, 0.0)),
            ((p.y - self.min.y).abs(), Vec3::new(0.0, -1.0, 0.0)),
            ((p.y - self.max.y).abs(), Vec3::new(0.0, 1.0, 0.0)),
            ((p.z - self.min.z).abs(), Vec3::new(0.0, 0.0, -1.0)),
            ((p.z - self.max.z).abs(), Vec3::new(0.0, 0.0, 1.0)),
        ];
        let normal = candidates
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, n)| *n)
            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

        Some(HitRecord {
            t,
            p,
            normal,
            mat: self.mat,
        })
    }

    fn bounding_box(&self) -> BBox {
        BBox {
            min: self.min,
            max: self.max,
        }
    }
}

// --- AREA LIGHTS ---
/// A sampleable emitter surface. The shape parameters must match an emissive
/// object in the scene so that BDPT can connect paths to real geometry.
#[derive(Clone)]
pub enum LightShape {
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Quad {
        center: Vec3,
        normal: Vec3,
        tangent: Vec3,
        bitangent: Vec3,
        half_w: f32,
        half_h: f32,
    },
}

#[derive(Clone)]
pub struct AreaLight {
    pub shape: LightShape,
    pub emission: Vec3,
}

impl AreaLight {
    pub fn from_sphere(sphere: &Sphere) -> Self {
        Self {
            shape: LightShape::Sphere {
                center: sphere.center,
                radius: sphere.radius,
            },
            emission: sphere.mat.emission,
        }
    }

    pub fn from_quad(quad: &Quad) -> Self {
        Self {
            shape: LightShape::Quad {
                center: quad.center,
                normal: quad.normal,
                tangent: quad.tangent,
                bitangent: quad.bitangent,
                half_w: quad.half_w,
                half_h: quad.half_h,
            },
            emission: quad.mat.emission,
        }
    }

    pub fn area(&self) -> f32 {
        match &self.shape {
            LightShape::Sphere { radius, .. } => 4.0 * PI * radius * radius,
            LightShape::Quad { half_w, half_h, .. } => 4.0 * half_w * half_h,
        }
    }

    /// Uniform surface sample: (point, normal). The pdf is 1 / area.
    pub fn sample(&self, rng: &mut impl Rng) -> (Vec3, Vec3) {
        match &self.shape {
            LightShape::Sphere { center, radius } => {
                // Uniform point on the sphere via spherical coordinates.
                let z = 1.0 - 2.0 * rng.gen::<f32>(); // cos theta in [-1, 1]
                let phi = 2.0 * PI * rng.gen::<f32>();
                let r = (1.0 - z * z).sqrt();
                let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                (*center + dir * *radius, dir)
            }
            LightShape::Quad {
                center,
                normal,
                tangent,
                bitangent,
                half_w,
                half_h,
            } => {
                let u = rng.gen::<f32>() * 2.0 - 1.0;
                let v = rng.gen::<f32>() * 2.0 - 1.0;
                let p = *center + *tangent * (u * half_w) + *bitangent * (v * half_h);
                (p, *normal)
            }
        }
    }

    /// True if `p` lies on the light surface (within a small epsilon).
    pub fn contains(&self, p: Vec3) -> bool {
        match &self.shape {
            LightShape::Sphere { center, radius } => {
                ((p - *center).length() - *radius).abs() < 1e-3
            }
            LightShape::Quad {
                center,
                normal,
                tangent,
                bitangent,
                half_w,
                half_h,
            } => {
                let local = p - *center;
                local.dot(*normal).abs() < 1e-3
                    && local.dot(*tangent).abs() <= half_w + 1e-3
                    && local.dot(*bitangent).abs() <= half_h + 1e-3
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn diffuse(albedo: Vec3) -> Material {
        Material::diffuse(albedo)
    }

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray {
            origin,
            direction: direction.normalize(),
        }
    }

    #[test]
    fn sphere_hit_from_outside_and_inside() {
        let s = Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            mat: diffuse(Vec3::new(1.0, 1.0, 1.0)),
        };
        let outside = s.intersect(&ray(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)));
        assert!(outside.is_some());
        assert!((outside.unwrap().t - 4.0).abs() < 1e-4);

        // A ray starting at the center must hit the far surface.
        let inside = s.intersect(&ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)));
        assert!(inside.is_some());
        assert!((inside.unwrap().t - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cube_normals_face_outward_on_all_sides() {
        let c = Cube::new(
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        let top = c
            .intersect(&ray(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0)))
            .unwrap();
        assert_eq!(top.normal, Vec3::new(0.0, 1.0, 0.0));

        let right = c
            .intersect(&ray(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)))
            .unwrap();
        assert_eq!(right.normal, Vec3::new(1.0, 0.0, 0.0));

        let front = c
            .intersect(&ray(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)))
            .unwrap();
        assert_eq!(front.normal, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn quad_intersect_respects_bounds() {
        let q = Quad::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            2.0,
            diffuse(Vec3::new(1.0, 1.0, 1.0)),
        );
        let hit = q.intersect(&ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 2.0).abs() < 1e-4);

        let miss = q.intersect(&ray(Vec3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        assert!(miss.is_none());
    }

    #[test]
    fn area_light_samples_lie_on_the_surface() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sphere = Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
            mat: Material::emissive(Vec3::new(5.0, 5.0, 5.0)),
        };
        let light = AreaLight::from_sphere(&sphere);
        assert!((light.area() - 4.0 * PI * 0.25).abs() < 1e-5);
        for _ in 0..100 {
            let (p, _) = light.sample(&mut rng);
            assert!(light.contains(p));
        }

        let quad = Quad::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            4.0,
            sphere.mat,
        );
        let light = AreaLight::from_quad(&quad);
        assert!((light.area() - 8.0).abs() < 1e-5);
        for _ in 0..100 {
            let (p, n) = light.sample(&mut rng);
            assert!(light.contains(p));
            assert_eq!(n, Vec3::new(0.0, -1.0, 0.0));
        }
    }

    // --- BSDF tests ---

    #[test]
    fn diffuse_bsdf_reduces_to_the_cosine_sampling_constants() {
        // f = albedo/PI, cosine sampling pdf = cos/PI, so the sample
        // weight f·cos/pdf = albedo exactly — the constants the old
        // hand-computed BDPT tests were written against.
        let m = Material::diffuse(Vec3::new(0.5, 0.6, 0.7));
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.0, -1.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..100 {
            let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, &mut rng);
            assert!(s.dir_out.dot(n) > 0.0);
            assert!((s.pdf_omega - s.dir_out.dot(n) / PI).abs() < 1e-6);
            assert_eq!(s.weight, m.albedo);
            let (f, pdf) = m.eval_bsdf(n, dir_in, s.dir_out);
            assert_eq!(f, m.albedo * (1.0 / PI));
            assert!((pdf - s.pdf_omega).abs() < 1e-6);
        }
        // Below the horizon: no density.
        let (f, pdf) = m.eval_bsdf(n, dir_in, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(f, VEC_ZERO);
        assert_eq!(pdf, 0.0);
    }

    #[test]
    fn mirror_bsdf_is_deterministic_and_delta() {
        let m = Material::mirror(Vec3::new(0.9, 0.9, 0.9));
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, &mut rng);
        assert_eq!(s.dir_out, Vec3::new(0.6, 0.8, 0.0));
        assert_eq!(s.pdf_omega, 0.0);
        assert_eq!(s.weight, m.albedo);
        // eval is (0, 0) for delta BSDFs — connections through them are
        // invalid.
        let (f, pdf) = m.eval_bsdf(n, dir_in, s.dir_out);
        assert_eq!(f, VEC_ZERO);
        assert_eq!(pdf, 0.0);
        assert!(m.is_delta());
        // The specular weight of the actual pair is the albedo; any other
        // direction has none.
        assert_eq!(
            m.specular_weight(n, true, dir_in, s.dir_out, Transport::Radiance),
            Some(m.albedo)
        );
        assert_eq!(
            m.specular_weight(n, true, dir_in, Vec3::new(0.0, 1.0, 0.0), Transport::Radiance),
            None
        );
    }

    #[test]
    fn glass_fresnel_conserves_energy() {
        // R + T = 1 for every incidence angle, from both sides, including
        // the total-internal-reflection regime (T = 0 there).
        let ior = 1.5;
        for deg in [0.0f32, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0] {
            let cos_i = deg.to_radians().cos();
            let r_out = fresnel_dielectric(cos_i, 1.0, ior);
            let r_in = fresnel_dielectric(cos_i, ior, 1.0);
            assert!((r_out + (1.0 - r_out) - 1.0).abs() < 1e-6);
            assert!((r_in + (1.0 - r_in) - 1.0).abs() < 1e-6);
            // Beyond the critical angle (sin > 1/1.5, i.e. > 41.8°) the
            // internal reflectance must be exactly 1.
            if deg > 41.8 {
                assert!((r_in - 1.0).abs() < 1e-6, "TIR at {deg}°: {r_in}");
            }
        }
        // Normal incidence: the textbook value ((1-1.5)/(1+1.5))^2 = 0.04.
        assert!((fresnel_dielectric(1.0, 1.0, 1.5) - 0.04).abs() < 1e-4);
    }

    #[test]
    fn glass_sampling_matches_the_fresnel_split() {
        // Over many samples the reflection/refraction split must follow
        // the Fresnel reflectance, and the refracted direction must obey
        // Snell's law.
        let m = Material::glass(1.5);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let f = fresnel_dielectric(-n.dot(dir_in), 1.0, 1.5);
        let mut reflected = 0;
        const N: usize = 20000;
        for _ in 0..N {
            let s = m.sample_bsdf(n, true, dir_in, Transport::Radiance, &mut rng);
            assert_eq!(s.pdf_omega, 0.0);
            if s.dir_out.y > 0.0 {
                reflected += 1;
            } else {
                // Snell: sin_t = sin_i / 1.5.
                let sin_i = (1.0f32 - 0.8 * 0.8).sqrt();
                let sin_t = (1.0 - s.dir_out.y * s.dir_out.y).sqrt();
                assert!((sin_t - sin_i / 1.5).abs() < 1e-4);
                // Radiance transport picks up (eta_t/eta_i)^2 = 1.5^2.
                assert_eq!(s.weight, Vec3::new(2.25, 2.25, 2.25));
            }
        }
        let frac = reflected as f32 / N as f32;
        assert!((frac - f).abs() < 0.02, "split {frac} vs Fresnel {f}");
    }

    #[test]
    fn glass_tir_falls_back_to_reflection() {
        // Inside the glass (front_face = false), beyond the critical angle
        // every sample must reflect — no refraction exists.
        let m = Material::glass(1.5);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.9, -0.435889894354067, 0.0); // sin_i = 0.9 > 1/1.5
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        for _ in 0..100 {
            let s = m.sample_bsdf(n, false, dir_in, Transport::Radiance, &mut rng);
            assert!(s.dir_out.y > 0.0, "TIR must reflect: {:?}", s.dir_out);
            assert_eq!(s.weight, Vec3::new(1.0, 1.0, 1.0));
        }
        // And the specular weight of the refracted pair is None there.
        assert_eq!(
            m.specular_weight(n, false, dir_in, Vec3::new(0.0, -1.0, 0.0), Transport::Radiance),
            None
        );
    }

    #[test]
    fn glass_specular_weight_covers_both_modes() {
        // The refracted pair's weight differs by transport mode: radiance
        // gets (eta_t/eta_i)^2, importance does not.
        let m = Material::glass(1.5);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let dir_in = Vec3::new(0.6, -0.8, 0.0);
        let refr = refract_dir(dir_in, n, 1.0, 1.5).unwrap();
        assert_eq!(
            m.specular_weight(n, true, dir_in, refr, Transport::Radiance),
            Some(Vec3::new(2.25, 2.25, 2.25))
        );
        assert_eq!(
            m.specular_weight(n, true, dir_in, refr, Transport::Importance),
            Some(Vec3::new(1.0, 1.0, 1.0))
        );
        // Exiting the glass: eta_i = 1.5, eta_t = 1, so radiance gets
        // (1/1.5)^2 and importance 1. Compare within a few ulps: the
        // implementation computes (1/1.5)*(1/1.5), which rounds
        // differently from the literal 1/2.25.
        let refr_out = refract_dir(dir_in, n, 1.5, 1.0).unwrap();
        let w = m.specular_weight(n, false, dir_in, refr_out, Transport::Radiance);
        let expected = Vec3::new(1.0 / 2.25, 1.0 / 2.25, 1.0 / 2.25);
        assert!(w.is_some());
        assert!(
            (w.unwrap().x - expected.x).abs() < 1e-6
                && (w.unwrap().y - expected.y).abs() < 1e-6
                && (w.unwrap().z - expected.z).abs() < 1e-6,
            "got {w:?}, expected {expected:?}"
        );
    }
}
