//! Minimal 5×7 bitmap font for the window frontend's status line and help
//! overlay.
//!
//! The windowed frontend has no font-rendering dependency; instead the
//! overlay text is rasterized on the CPU into a small RGBA8 texture
//! (re-uploaded only when the text changes) and composited in the display
//! shader with `textureLoad`. Colors are authored in linear space so the
//! blend happens before gamma, uniformly for sRGB and linear surfaces.

/// Glyph cell: 5 wide, 7 tall, 1px advance padding.
pub const GLYPH_W: u32 = 5;
pub const GLYPH_H: u32 = 7;
/// Horizontal advance between glyphs (glyph + 1px spacing).
pub const ADVANCE: u32 = GLYPH_W + 1;

/// A 5×7 bitmap font covering uppercase letters, digits and the
/// punctuation used by the overlay strings. Rows are the low 5 bits,
/// top row first; bit 4 is the leftmost pixel.
const GLYPHS: &[(&str, [u8; 7])] = &[
    (
        "A",
        [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "B",
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        "C",
        [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
    ),
    (
        "D",
        [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        "E",
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        "F",
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        "G",
        [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
    ),
    (
        "H",
        [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "I",
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
    ),
    (
        "J",
        [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
    ),
    (
        "K",
        [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        "L",
        [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        "M",
        [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "N",
        [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "O",
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        "P",
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        "Q",
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
    ),
    (
        "R",
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        "S",
        [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    ),
    (
        "T",
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        "U",
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        "V",
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
    ),
    (
        "W",
        [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
    ),
    (
        "X",
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
    ),
    (
        "Y",
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        "Z",
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
    ),
    (
        "0",
        [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
    ),
    (
        "1",
        [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        "2",
        [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
    ),
    (
        "3",
        [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    ),
    (
        "4",
        [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
    ),
    (
        "5",
        [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        "6",
        [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        "7",
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
    ),
    (
        "8",
        [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        "9",
        [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
    ),
    (" ", [0, 0, 0, 0, 0, 0, 0]),
    (".", [0, 0, 0, 0, 0, 0b01100, 0b01100]),
    (":", [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0]),
    ("-", [0, 0, 0, 0b11111, 0, 0, 0]),
    (
        "/",
        [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
    ),
    (
        "%",
        [
            0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
        ],
    ),
    (
        "|",
        [
            0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        "(",
        [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
    ),
    (
        ")",
        [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
    ),
    (
        "<",
        [
            0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010,
        ],
    ),
    (
        ">",
        [
            0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000,
        ],
    ),
    ("+", [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0]),
];

/// Look up a glyph's rows. Unknown characters render as a blank cell
/// (the space glyph), so the overlay never panics on odd input.
fn glyph_rows(ch: char) -> [u8; 7] {
    let s = ch.to_ascii_uppercase().to_string();
    GLYPHS
        .iter()
        .find(|(name, _)| *name == s)
        .map(|(_, rows)| *rows)
        .unwrap_or([0; 7])
}

/// An RGBA8 canvas the overlay text is rasterized into.
pub struct TextCanvas {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, alpha = coverage (0 or 255).
    pub data: Vec<u8>,
}

impl TextCanvas {
    /// A canvas sized to fit `lines` lines of text.
    pub fn new(lines: usize) -> Self {
        let height = (lines as u32 * (GLYPH_H + 1)).max(1);
        Self {
            width: 1,
            height,
            data: vec![0; 4],
        }
    }

    /// Rasterize `lines` into the canvas, growing it to the widest line.
    /// Returns the final size. Alpha is 255 where a glyph pixel is set.
    pub fn draw(&mut self, lines: &[&str]) -> (u32, u32) {
        let text_w = lines
            .iter()
            .map(|l| l.chars().count() as u32 * ADVANCE)
            .max()
            .unwrap_or(ADVANCE);
        let text_h = lines.len() as u32 * (GLYPH_H + 1);
        self.width = text_w.max(1);
        self.height = text_h.max(1);
        self.data = vec![0u8; (self.width * self.height * 4) as usize];

        for (li, line) in lines.iter().enumerate() {
            for (ci, ch) in line.chars().enumerate() {
                let rows = glyph_rows(ch);
                let gx = ci as u32 * ADVANCE;
                let gy = li as u32 * (GLYPH_H + 1);
                for row in 0..GLYPH_H {
                    for col in 0..GLYPH_W {
                        let bit = rows[row as usize] >> (4 - col) & 1;
                        if bit == 1 {
                            let idx = ((gy + row) * self.width + gx + col) as usize * 4;
                            self.data[idx] = 255;
                            self.data[idx + 1] = 255;
                            self.data[idx + 2] = 255;
                            self.data[idx + 3] = 255;
                        }
                    }
                }
            }
        }
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_row_fits_five_bits() {
        for (_, rows) in GLYPHS {
            for r in rows {
                assert_eq!(r >> 5, 0, "glyph row wider than 5 bits: {r:08b}");
            }
        }
    }

    #[test]
    fn canvas_draws_text_with_expected_size() {
        let mut canvas = TextCanvas::new(2);
        let (w, h) = canvas.draw(&["AB", "C"]);
        // 2 glyphs + trailing advance = 12 px wide; 2 lines of 8 px tall.
        assert_eq!(w, 2 * ADVANCE);
        assert_eq!(h, 2 * (GLYPH_H + 1));
        assert_eq!(canvas.data.len(), (w * h * 4) as usize);
        // The first pixel of 'A' (row 0, col 1) must be set.
        let idx = (1) as usize * 4;
        assert_eq!(canvas.data[idx + 3], 255);
    }

    #[test]
    fn unknown_characters_draw_as_blank() {
        let mut canvas = TextCanvas::new(1);
        canvas.draw(&["\u{263A}"]); // Unicode smiley: not in the font.
        assert!(canvas.data.iter().all(|b| *b == 0));
    }
}
