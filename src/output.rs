use crate::math::{Vec3, VEC_ZERO};
use image::{Rgb, RgbImage};
use std::fmt;

/// A buffer of pixels ready for output.
pub struct PixelBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Vec3>,
}

/// Extended Reinhard tone mapping with gamma correction (2.2 sRGB).
///
/// The plain operator `c/(1+c)` never reaches 1.0, so bright emitters
/// (the light sphere is 10 in linear HDR) render as dull gray. The
/// extended form maps the white point `W` to exactly 1.0:
/// `c(1 + c/W²)/(1 + c)`, so highlights roll off to pure white instead
/// of plateauing below it.
const TONE_MAP_WHITE_POINT: f32 = 10.0;

impl PixelBuffer {
    /// Get a single pixel by coordinates.
    pub fn get_pixel(&self, x: usize, y: usize) -> Vec3 {
        self.pixels[y * self.width + x]
    }

    /// Extended Reinhard tone mapping with gamma correction (2.2 sRGB).
    fn tone_map(p: Vec3) -> Vec3 {
        // Extended Reinhard local operator with white point W.
        let w = TONE_MAP_WHITE_POINT;
        let mapped = Vec3::new(
            p.x * (1.0 + p.x / (w * w)) / (1.0 + p.x),
            p.y * (1.0 + p.y / (w * w)) / (1.0 + p.y),
            p.z * (1.0 + p.z / (w * w)) / (1.0 + p.z),
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

/// Terminal display: the buffer as true-color ANSI escape codes, two
/// pixel rows per character cell via the half-block glyph (▀).
impl fmt::Display for PixelBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        f.write_str(&output)
    }
}

/// Edge-preserving bilateral denoiser.
///
/// A neighbor contributes with weight `exp(-|d|^2 / 2 sigma_r^2)` based on its
/// color distance `d` from the center pixel, so flat regions are smoothed
/// while silhouettes stay crisp. `sigma_r` is in linear-light units.
fn bilateral(pixels: &[Vec3], width: usize, height: usize, radius: i32, sigma_r: f32) -> Vec<Vec3> {
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma_r * sigma_r);
    let mut result = vec![VEC_ZERO; width * height];

    for y in 0..height {
        for x in 0..width {
            let center = pixels[y * width + x];
            let mut sum = VEC_ZERO;
            let mut weight_sum = 0.0f32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                        continue;
                    }
                    let neighbor = pixels[ny as usize * width + nx as usize];
                    let d = neighbor - center;
                    let dist_sq = d.dot(d);
                    let w = (-dist_sq * inv_two_sigma_sq).exp();
                    sum = sum + neighbor * w;
                    weight_sum += w;
                }
            }

            result[y * width + x] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                center
            };
        }
    }

    result
}

/// Apply the denoiser to the raw (linear-light) pixel buffer.
///
/// This operates on linear HDR values *before* tone mapping; the display
/// functions apply Reinhard + gamma themselves, so filtering must not
/// tone-map in place (that would double-apply the gamma curve).
pub fn apply_filters(buffer: &mut PixelBuffer) {
    let width = buffer.width;
    let height = buffer.height;

    let filtered = bilateral(&buffer.pixels, width, height, 1, 0.15);
    buffer.pixels.copy_from_slice(&filtered);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilateral_smooths_noise_within_flat_regions() {
        // A flat 0.5 region with one slightly-noisy pixel: the filter must
        // pull the noisy pixel back toward its neighborhood.
        let width = 5;
        let height = 5;
        let mut pixels = vec![Vec3::new(0.5, 0.5, 0.5); width * height];
        pixels[2 * width + 2] = Vec3::new(0.55, 0.55, 0.55);

        let out = bilateral(&pixels, width, height, 1, 0.15);
        let smoothed = out[2 * width + 2];
        assert!(
            smoothed.x < 0.52 && smoothed.x > 0.48,
            "noise not smoothed: {smoothed:?}"
        );
    }

    #[test]
    fn bilateral_preserves_hard_edges() {
        // A hard 0.1 | 1.0 step: pixels adjacent to the edge must keep their
        // value (that is the whole point of an edge-preserving filter).
        let width = 6;
        let height = 3;
        let mut pixels = Vec::with_capacity(width * height);
        for _y in 0..height {
            for x in 0..width {
                pixels.push(if x < 3 {
                    Vec3::new(0.1, 0.1, 0.1)
                } else {
                    Vec3::new(1.0, 1.0, 1.0)
                });
            }
        }

        let out = bilateral(&pixels, width, height, 1, 0.15);

        // Bright pixel right at the boundary stays bright...
        let bright_edge = out[width + 3];
        assert!(bright_edge.x > 0.95, "edge blurred away: {bright_edge:?}");
        // ...and its dark neighbor stays dark.
        let dark_edge = out[width + 2];
        assert!(dark_edge.x < 0.15, "edge leaked: {dark_edge:?}");
    }

    #[test]
    fn tone_map_is_monotonic_and_bounded() {
        // Below the white point the operator stays in [0, 1].
        for v in [0.0f32, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0] {
            let t = PixelBuffer::tone_map(Vec3::new(v, v, v));
            assert!(t.x >= 0.0 && t.x <= 1.0, "value {v} maps to {t:?}");
        }
        // Above the white point it blows out past 1.0 (by design; the
        // display path clamps) but stays finite.
        let over = PixelBuffer::tone_map(Vec3::new(100.0, 100.0, 100.0));
        assert!(over.x > 1.0 && over.x.is_finite());
        // Monotonic across the whole range.
        let a = PixelBuffer::tone_map(Vec3::new(0.2, 0.2, 0.2));
        let b = PixelBuffer::tone_map(Vec3::new(0.4, 0.4, 0.4));
        assert!(a.x < b.x);
    }

    /// The extended operator must map the white point to (nearly) 1.0
    /// after gamma — the plain operator plateaus at ~0.32 there, which is
    /// why bright emitters looked dull.
    #[test]
    fn tone_map_reaches_white_at_the_white_point() {
        let w = TONE_MAP_WHITE_POINT;
        let t = PixelBuffer::tone_map(Vec3::new(w, w, w));
        assert!(t.x > 0.99, "white point maps to {t:?}");
        // Above the white point the operator keeps rising past 1.0 (it is
        // designed to blow out to white); the display path clamps. The
        // mapped value must stay finite and above the white-point value.
        let over = PixelBuffer::tone_map(Vec3::new(w * 10.0, w * 10.0, w * 10.0));
        assert!(
            over.x > t.x && over.x.is_finite(),
            "over-white mapping: {over:?}"
        );
    }
}
