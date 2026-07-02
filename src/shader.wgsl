struct Material {
    albedo: vec4<f32>,
    emission: vec4<f32>,
    mat_type: u32,
    _pad: vec3<u32>,
}

struct Sphere {
    center: vec4<f32>,
    radius: f32,
    mat_id: u32,
    _pad: vec2<u32>,
}

struct Cube {
    min: vec4<f32>,
    max: vec4<f32>,
    mat_id: u32,
    _pad: vec3<u32>,
}

struct Plane {
    point: vec4<f32>,
    normal: vec4<f32>,
    mat_id: u32,
    _pad: vec3<u32>,
}

struct SceneInfo {
    sphere_count: u32,
    cube_count: u32,
    plane_count: u32,
    material_count: u32,
}

struct Camera {
    origin: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> materials: array<Material>;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read> cubes: array<Cube>;
@group(0) @binding(3) var<storage, read> planes: array<Plane>;
@group(0) @binding(4) var<storage, read> scene: SceneInfo>;
@group(0) @binding(5) var<uniform> cam: Camera>;
@group(0) @binding(6) var<storage, read_write> output: array<vec4<f32>>;

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn random_unit_vector(seed: ptr<function, f32>) -> vec3<f32> {
    let s = *seed;
    let r1 = hash(s) * 2.0 - 1.0;
    let r2 = hash(s + 1.23) * 2.0 - 1.0;
    let r3 = hash(s + 2.46) * 2.0 - 1.0;
    return normalize(vec3<f32>(r1, r2, r3));
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Hit {
    t: f32,
    p: vec3<f32>,
    normal: vec3<f32>,
    mat_id: u32,
}

fn intersect_sphere(ray: Ray, sphere: Sphere) -> Hit {
    let oc = ray.origin - sphere.center.xyz;
    let a = dot(ray.direction, ray.direction);
    let b = 2.0 * dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) { return Hit(1e10, vec3(0.0), vec3(0.0), 0u); }

    let t = (-b - sqrt(discriminant)) / (2.0 * a);
    if (t < 0.001) { return Hit(1e10, vec3(0.0), vec3(0.0), 0u); }

    let p = ray.origin + t * ray.direction;
    let normal = normalize(p - sphere.center.xyz);
    return Hit(t, p, normal, sphere.mat_id);
}

fn intersect_plane(ray: Ray, plane: Plane) -> Hit {
    let denom = dot(plane.normal.xyz, ray.direction);
    if (abs(denom) < 1e-6) { return Hit(1e10, vec3(0.0), vec3(0.0), 0u); }
    let t = dot(plane.point.xyz - ray.origin, plane.normal.xyz) / denom;
    if (t < 0.001) { return Hit(1e10, vec3(0.0), vec3(0.0), 0u); }
    return Hit(t, ray.origin + t * ray.direction, plane.normal.xyz, plane.mat_id);
}

fn intersect_cube(ray: Ray, cube: Cube) -> Hit {
    var t_min = -1e10;
    var t_max = 1e10;

    let directions = vec3<f32>(ray.direction.x, ray.direction.y, ray.direction.z);
    let origins = vec3<f32>(ray.origin.x, ray.origin.y, ray.origin.z);
    let mins = vec3<f32>(cube.min.x, cube.min.y, cube.min.z);
    let maxs = vec3<f32>(cube.max.x, cube.max.y, cube.max.z);

    for (var i = 0; i < 3; i++) {
        let inv_d = 1.0 / directions[i];
        var t0 = (mins[i] - origins[i]) * inv_d;
        var t1 = (maxs[i] - origins[i]) * inv_d;
        if (inv_d < 0.0) {
            let tmp = t0; t0 = t1; t1 = tmp;
        }
        t_min = max(t_min, t0);
        t_max = min(t_max, t1);
    }

    if (t_max >= t_min && t_max > 0.0) {
        let t = select(t_max, t_min, t_min > 0.001);
        if (t < 0.001) { return Hit(1e10, vec3(0.0), vec3(0.0), 0u); }
        
        let p = ray.origin + t * ray.direction;
        var normal = vec3<f32>(0.0);
        let eps = 0.001;
        if (abs(p.x - cube.min.x) < eps) { normal = vec3<f32>(-1.0, 0.0, 0.0); }
        else if (abs(p.x - cube.max.x) < eps) { normal = vec3<f32>(1.0, 0.0, 0.0); }
        else if (abs(p.y - cube.min.y) < eps) { normal = vec3<f32>(0.0, -1.0, 0.0); }
        else if (abs(p.y - cube.max.y) < eps) { normal = vec3<f32>(0.0, 1.0, 0.0); }
        else if (abs(p.z - cube.min.z) < eps) { normal = vec3<f32>(0.0, 0.0, -1.0); }
        else if (abs(p.z - cube.max.z) < eps) { normal = vec3<f32>(0.0, 0.0, 1.0); }
        
        return Hit(t, p, normal, cube.mat_id);
    }
    return Hit(1e10, vec3(0.0), vec3(0.0), 0u);
}

fn trace(ray: Ray, seed: ptr<function, f32>) -> vec3<f32> {
    var current_ray = ray;
    var color = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    
    for (var depth = 0; depth < 10; depth++) {
        var closest_hit = Hit(1e10, vec3(0.0), vec3(0.0), 0u);
        
        for (var i = 0u; i < scene.sphere_count; i++) {
            let h = intersect_sphere(current_ray, spheres[i]);
            if (h.t < closest_hit.t) { closest_hit = h; }
        }
        for (var i = 0u; i < scene.cube_count; i++) {
            let h = intersect_cube(current_ray, cubes[i]);
            if (h.t < closest_hit.t) { closest_hit = h; }
        }
        for (var i = 0u; i < scene.plane_count; i++) {
            let h = intersect_plane(current_ray, planes[i]);
            if (h.t < closest_hit.t) { closest_hit = h; }
        }
        
        if (closest_hit.t > 1e9) {
            color += throughput * vec3<f32>(0.02, 0.02, 0.05);
            break;
        }
        
        let mat = materials[closest_hit.mat_id];
        color += throughput * mat.emission.xyz;
        
        if (mat.mat_type == 1u) { // Emissive
            break;
        }
        
        let target = closest_hit.normal + random_unit_vector(seed);
        current_ray = Ray(closest_hit.p + closest_hit.normal * 0.001, normalize(target));
        throughput = throughput * mat.albedo.xyz;
        
        if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01) {
            break;
        }
    }
    return color;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = 1920u; // Should be passed in, but simplifying for now
    let height = 1080u;
    
    if (id.x >= width || id.y >= height) { return; }
    
    var seed = f32(id.x + id.y * width) * 0.12345;
    
    let u = (f32(id.x) + 0.5) / f32(width);
    let v = (f32(id.y) + 0.5) / f32(height);
    
    let px = (u * 2.0 - 1.0) * (1.77 * 0.84); // simplified cam aspect
    let py = -(v * 2.0 - 1.0) * 0.84;
    
    let dir = normalize(cam.right.xyz * px + cam.up.xyz * py + cam.forward.xyz);
    let ray = Ray(cam.origin.xyz, dir);
    
    let color = trace(ray, &seed);
    output[id.y * width + id.x] = vec4<f32>(color, 1.0);
}
