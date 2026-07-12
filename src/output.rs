use image::{RgbImage, Rgb};
use rayon::prelude::*;

/// A buffer of pixels ready for output.
pub struct PixelBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Vec3>,
}

impl PixelBuffer {
    /// Get a single pixel by coordinates.
    pub fn get_pixel(&self, x: usize, y: usize) -> Vec3 {
        self.pixels[y * self.width + x]
    }

    /// Reinhard tone mapping with gamma correction (2.2 sRGB).
    fn tone_map(p: Vec3) -> Vec3 {
        // Reinhard local operator
        let mapped = Vec3::new(
            p.x / (1.0 + p.x),
            p.y / (1.0 + p.y),
            p.z / (1.0 + p.z),
        );
        // Gamma correction for sRGB display
        Vec3::new(
            mapped.x.powf(1.0 / 2.2),
            mapped.y.powf(1.0 / 2.2),
            mapped.z.powf(1.0 / 2.2),
        )
    }

    /// Save the buffer as a PNG file with proper tone mapping and gamma correction.
    pub fn save_as_png(&self, path: &str) -> Result<(), image::ImageError> {
        let mut img = RgbImage::new(self.width as u32, self.height as u32);

        for y in 0..self.height {
            for x in 0..self.width {
                let color = Self::tone_map(self.get_pixel(x, y));
                img.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([
                        (color.x.clamp(0.0, 1.0) * 255.0) as u8,
                        (color.y.clamp(0.0, 1.0) * 255.0) as u8,
                        (color.z.clamp(0.0, 1.0) * 255.0) as u8,
                    ]),
                );
            }
        }
        img.save(path)
    }

    /// Convert the buffer to a string for terminal display using true-color ANSI codes.
    pub fn to_string(&self) -> String {
        let mut output = String::new();
        for y in (0..self.height).step_by(2) {
            for x in 0..self.width {
                let top_pixel = Self::tone_map(self.get_pixel(x, y));
                let bottom_pixel = if y + 1 < self.height {
                    Self::tone_map(self.get_pixel(x, y + 1))
                } else {
                    VEC_ZERO
                };

                // Clamp and convert to 8-bit color values
                let r_top = (top_pixel.x.clamp(0.0, 1.0) * 255.0) as u8;
                let g_top = (top_pixel.y.clamp(0.0, 1.0) * 255.0) as u8;
                let b_top = (top_pixel.z.clamp(0.0, 1.0) * 255.0) as u8;

                let r_bot = (bottom_pixel.x.clamp(0.0, 1.0) * 255.0) as u8;
                let g_bot = (bottom_pixel.y.clamp(0.0, 1.0) * 255.0) as u8;
                let b_bot = (bottom_pixel.z.clamp(0.0, 1.0) * 255.0) as u8;

                // Avoid pure black characters which are invisible in terminals
                let r_top = if r_top == 0 { 1 } else { r_top };
                let g_top = if g_top == 0 { 1 } else { g_top };
                let b_top = if b_top == 0 { 1 } else { b_top };

                output.push_str(&format!(
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                    r_top, g_top, b_top, r_bot, g_bot, b_bot
                ));
            }
            output.push_str("\x1b[0m\r\n");
        }
        // Remove trailing newline for cleaner output
        if output.ends_with("\r\n") {
            output.truncate(output.len() - 2);
        }
        output
    }

    /// Render progress bar string.
    pub fn render_progress(current: usize, total: usize) -> String {
        let pct = current as f64 / total as f64;
        let filled = (pct * 30.0) as usize;
        let empty = 30 - filled;
        format!(
            "\rProgress: [{}{}] {:.1}% ({}/{})",
            "█".repeat(filled),
            "░".repeat(empty),
            pct * 100.0,
            current,
            total
        )
    }
}

/// Simple box blur denoiser (much cheaper than bilateral filter).
fn box_blur(pixels: &[Vec3], width: usize, height: usize, radius: i32) -> Vec<Vec3> {
    let mut result = vec![VEC_ZERO; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = VEC_ZERO;
            let mut count = 0u32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        sum = sum + pixels[ny as usize * width + nx as usize];
                        count += 1;
                    }
                }
            }

            result[y * width + x] = if count > 0 { sum / count as f32 } else { VEC_ZERO };
        }
    }

    result
}

/// Apply tone mapping and optional denoising to the pixel buffer.
pub fn apply_filters(buffer: &mut PixelBuffer) {
    let width = buffer.width;
    let height = buffer.height;

    // Step 1: Tone map all pixels first (Reinhard + gamma 2.2)
    for i in 0..buffer.pixels.len() {
        buffer.pixels[i] = PixelBuffer::tone_map(buffer.pixels[i]);
    }

    // Step 2: Light denoising via box blur (radius 1, very subtle)
    let blurred = box_blur(&buffer.pixels, width, height, 1);

    // Blend original and blurred for a mild smoothing effect
    let blend = 0.3;
    for i in 0..buffer.pixels.len() {
        buffer.pixels[i] = buffer.pixels[i] * (1.0 - blend) + blurred[i] * blend;
    }
}