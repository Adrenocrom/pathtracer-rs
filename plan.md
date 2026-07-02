You have a solid foundation here for a basic Path Tracer. Mathematically, your implementation of **Lambertian (Diffuse) reflection** using cosine-weighted sampling is correct: because the $\cos(\theta)$ from the rendering equation and the $\pi$ from the BRDF cancel out with the PDF of the cosine-weighted hemisphere sample, you are left with just `albedo * indirect`.

However, there are several architectural and physical improvements you are likely missing. Here is a breakdown of what to improve:

### 1. Decouple Material Logic (The biggest issue)
Currently, your `trace` function "decides" how the light bounces (`hit.normal + random_unit_vector()`). This means **every single object in your world is a matte diffuse surface**.

If you want mirrors, glass, or metals, the `Material` should determine the next ray.

**Suggested Change:** Move the scattering logic into the Material struct.

```rust
// In your Material implementation
impl Material for MyMaterial {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterRecord> {
        // Logic for diffuse, mirror, or glass goes here
        // Return the new ray and the attenuation (albedo)
    }
}
```

### 2. The "Emissive" Logic Bug
Look at this block:
```rust
if let MaterialType::Emissive = hit.mat.mat_type {
    return emission;
}
```
This means that if a ray hits a light source, it returns the color and **stops**. This is correct for the *end* of a path. However, your `trace` function later does: `emission + hit.mat.albedo * indirect`. 

If you have a material that is both slightly emissive AND reflective (like a glowing neon plastic), your current logic skips the reflection entirely because of the early return. You should generally only stop recursion based on `depth`, not based on whether an object emits light.

### 3. Precision: The "Shadow Acne" Offset
You are using `hit.p + hit.normal * 0.001`. While this works, it can cause "light leaking" or "shadow acne" depending on the scale of your scene. A more robust way is to push the origin along the normal based on a small epsilon relative to the floating point precision.

### 4. Performance: Russian Roulette
Currently, you use a fixed `depth`. This means every single ray travels exactly $N$ bounces (unless it hits the background). This is inefficient.

**Russian Roulette** allows rays to terminate early based on their "throughput" (how much color is left). If a ray has hit three black walls, its contribution to the final pixel is nearly zero—why keep calculating?

---

### The "Improved" Version
Here is how I would rewrite this function to be professional and extensible:

```rust
fn trace(ray: &Ray, scene: &dyn Intersectable, depth: i32) -> Vec3 {
    // 1. Base case for recursion
    if depth <= 0 {
        return VEC_ZERO;
    }

    if let Some(hit) = scene.intersect(ray) {
        let material = &hit.mat;
        
        // The object emits light regardless of whether it reflects
        let emitted = material.emission;

        // Ask the material: "If a ray hits you, how does it bounce?"
        if let Some(scatter_record) = material.scatter(ray, &hit) {
            let indirect = trace(&scatter_record.ray, scene, depth - 1);
            
            // Rendering Equation: Outgoing = Emission + (Reflection * Incoming)
            return emitted + (material.albedo * indirect);
        }

        // If the material doesn't scatter (absorbed), only return emission
        return emitted;
    }

    // Background color / Skybox
    Vec3::new(0.02, 0.02, 0.05) 
}
```

### Summary of what you missed:
1.  **Abstraction:** The `trace` function should not know *how* a surface reflects; it should only ask the material for the resulting ray.
2.  **Material Variety:** By moving the logic to `material.scatter()`, you can now easily add:
    *   **Specular (Mirrors):** `reflect(ray.dir, hit.normal)`
    *   **Refractive (Glass):** Use Snell's Law for the new direction.
    *   **Roughness:** Interpolate between the normal and a random unit vector.
3.  **The "Emissive" flow:** Allow emissive materials to also reflect light if they have an albedo value.
4.  **Next Step (Advanced):** Look into **Importance Sampling**. Instead of picking a random direction, pick a direction toward known light sources. This will remove the "noise" (graininess) from your image much faster.
