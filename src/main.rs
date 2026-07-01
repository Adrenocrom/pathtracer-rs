use rayon::prelude::*;

use image::{RgbImage, ImageBuffer};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    execute,
    terminal::{self, ClearType},
};
use num_complex::Complex64;
use std::{io::{Write, stdout}};

use chrono::Local;

const MAX_ITER: u32 = 1000;

struct Camera {
    center: Complex64,
    zoom: f64, // Lower value = more zoomed in
}

impl Camera {
    fn new() -> Self {
        Self {
            center: Complex64::new(-0.5, 0.0),
            zoom: 1.0,
        }
    }

    // Scale movement based on current zoom level
    fn move_center(&mut self, dx: f64, dy: f64) {
        let step = 0.1 * self.zoom;
        self.center.re += dx * step;
        self.center.im += dy * step;
    }

    fn zoom_in(&mut self) { self.zoom *= 0.9; }
    fn zoom_out(&mut self) { self.zoom *= 1.1; }
}

fn fg(r: u8, g: u8, b: u8) -> String {
    format!("\x1b[38;2;{};{};{}m", r, g, b)
}

fn bg(r: u8, g: u8, b: u8) -> String {
    format!("\x1b[48;2;{};{};{}m", r, g, b)
}

fn timestamp() -> String {
    Local::now().format("%Y%m%d_%H%M%S").to_string()
}

struct ColorStop {
    position: f64, // 0.0 to 1.0
    color: (f64, f64, f64), // R, G, B as 0.0 to 1.0
}

fn get_palette() -> Vec<ColorStop> {
    vec![
        ColorStop { position: 0.0, color: (0.0, 0.0, 0.01) }, // Deep Blue
        ColorStop { position: 0.5, color: (0.0, 0.2, 0.5) },  // Red
        ColorStop { position: 0.75, color: (1.0, 1.0, 0.0) },  // Red
        ColorStop { position: 1.0, color: (1.0, 1.0, 1.0) },   // Dark Red (end)
    ]
}

pub fn iteration_to_rgb(iter: u32, max_iter: u32) -> (u8, u8, u8) {
    if iter >= max_iter {
        return (1, 1, 1); // The Mandelbrot set itself is black
    }

    // t is our position in the gradient (0.0 to 1.0)
    let t = iter as f64 / max_iter as f64;
    let palette = get_palette();

    // 1. Find which two color stops t falls between
    let mut lower = 0;
    let mut upper = palette.len() - 1;

    for i in 0..palette.len() - 1 {
        if t >= palette[i].position && t <= palette[i+1].position {
            lower = i;
            upper = i + 1;
            break;
        }
    }

    let stop_low = &palette[lower];
    let stop_high = &palette[upper];

    // 2. Calculate local interpolation factor (0.0 to 1.0) between these two stops
    let range = stop_high.position - stop_low.position;
    let local_t = if range == 0.0 { 0.0 } else { (t - stop_low.position) / range };

    // 3. Linearly interpolate RGB channels
    let r = stop_low.color.0 + local_t * (stop_high.color.0 - stop_low.color.0);
    let g = stop_low.color.1 + local_t * (stop_high.color.1 - stop_low.color.1);
    let b = stop_low.color.2 + local_t * (stop_high.color.2 - stop_low.color.2);

    // 4. Apply Gamma Correction for better visual contrast
    let gamma_correct = |x: f64| ((x.powf(0.5) * 255.0).clamp(0.0, 255.0)) as u8;

    (gamma_correct(r), gamma_correct(g), gamma_correct(b))
}

fn save_screenshot(cam: &Camera) -> std::io::Result<()> {
    let mut stdout = stdout();
    execute!( stdout, cursor::MoveToColumn(0), terminal::Clear(ClearType::CurrentLine))?;
    print!("Creating screenshot");
    stdout.flush()?;

    let width = 4096;
    let height = 2304;
    let timestamp = timestamp();                     // ← from the helper above
    let filename = format!("mandelbrot_{}.png", timestamp);

    let x_scale = cam.zoom * (3.5 / width as f64);
    let y_scale = cam.zoom * (2.0 / height as f64);

    // Store both the RGB and the iteration count to decide where to blur
    let mut pixel_data = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let re = cam.center.re + (x as f64 - width as f64 / 2.0) * x_scale;
            let im = cam.center.im + (y as f64 - height as f64 / 2.0) * y_scale;
            
            let i = mandelbrot_iter(Complex64::new(re, im), MAX_ITER);
            pixel_data.push((iteration_to_rgb(i, MAX_ITER), i));
        }
    }

    // Only blur "low-end" iterations. 
    // We'll define a threshold based on the local view's max potential iterations.
    // In this case, we'll use a fixed threshold of 20% of MAX_ITER as "low-end".
    let blur_threshold = MAX_ITER / 10; 
    let blur_radius = 2;
    let mut final_pixels = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let ((r, g, b), iter) = pixel_data[y * width + x];

            if false && iter > blur_threshold && iter != 0 {
                // This is a low-iteration area, apply blur
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for dy in -(blur_radius as i32)..=(blur_radius as i32) {
                    for dx in -(blur_radius as i32)..=(blur_radius as i32) {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let ((pr, pg, pb), _) = pixel_data[ny as usize * width + nx as usize];
                            r_sum += pr as u32;
                            g_sum += pg as u32;
                            b_sum += pb as u32;
                            count += 1;
                        }
                    }
                }
                final_pixels.push(((r_sum / count) as u8, (g_sum / count) as u8, (b_sum / count) as u8));
            } else {
                // High-end iterations or the set itself: keep sharp
                final_pixels.push((r, g, b));
            }
        }
    }

    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for (i, pixel) in final_pixels.iter().enumerate() {
        let x = (i % width) as u32;
        let y = (i / width) as u32;
        img.put_pixel(x, y, image::Rgb([pixel.0, pixel.1, pixel.2]));
    }
    let _ = img.save(&filename);
    print!(" - Screenshot saved to {}", &filename);
    stdout.flush()?;
    Ok(())
}

fn mandelbrot_iter(c: Complex64, max_iter: u32) -> u32 {
    let mut z = Complex64::new(0.0, 0.0);
    let mut i = 0;

    // We grow the escape radius exponentially with each step.
    // If |z| > R*2^k we can stop because it will never get back inside.
    while i < max_iter && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        i += 1;
    }
    i
}

fn render(cam: &Camera) -> String {
    let (cols, rows) = terminal::size().unwrap_or((80, 24));

    let img_rows = rows as f64 * 2.0;
    let img_cols = cols as f64;

    let x_scale = cam.zoom * (3.5 / img_cols);
    let y_scale = cam.zoom * (2.0 / img_rows);

    // parallel over terminal lines
    let lines: Vec<String> = (0..rows).into_par_iter().map(|term_y| {
        let y_up   = term_y as f64 * 2.0;
        let y_down = y_up + 1.0;

        let mut line = String::with_capacity((cols as usize) * 12);

        for term_x in 0..cols {
            let x      = term_x as f64;
            let re     = cam.center.re + ((x - img_cols / 2.0) * x_scale);
            let im_up   = cam.center.im + ((y_up   - img_rows / 2.0) * y_scale);
            let im_down = cam.center.im + ((y_down - img_rows / 2.0) * y_scale);

            let up_col    = iteration_to_rgb( mandelbrot_iter(Complex64::new(re, im_up), MAX_ITER), MAX_ITER,);
            let down_col  = iteration_to_rgb( mandelbrot_iter(Complex64::new(re, im_down), MAX_ITER), MAX_ITER,);

            line.push_str(&fg(up_col.0, up_col.1, up_col.2));
            line.push_str(&bg(down_col.0, down_col.1, down_col.2));
            line.push('▀');
        }

        //line.push_str(RESET);
        line
    })
    .collect();

    lines.join("\r\n") + "\r\n"
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = stdout();
    let mut cam = Camera::new();

    terminal::enable_raw_mode()?;
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;

    // We'll keep the last frame around so we can avoid an expensive
    // re‑render when nothing changed.
    let mut last_frame = String::new();
    let mut need_redraw = true;          // first time we always draw

    loop {
        if event::poll(std::time::Duration::from_millis(0))? {
            match event::read()? {
                Event::Key(key) => match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('+') | KeyCode::Char('=') => { cam.zoom_in(); need_redraw = true },
                    KeyCode::Char('-') | KeyCode::Char('_') => { cam.zoom_out(); need_redraw = true },
                    KeyCode::Up | KeyCode::Char('w') | KeyCode::Char('k') => { cam.move_center(0.0, -1.0); need_redraw = true },
                    KeyCode::Down | KeyCode::Char('s') | KeyCode::Char('j') => { cam.move_center(0.0,  1.0); need_redraw = true },
                    KeyCode::Left | KeyCode::Char('a') | KeyCode::Char('h') => { cam.move_center(-1.0, 0.0); need_redraw = true },
                    KeyCode::Right| KeyCode::Char('d') | KeyCode::Char('l') => { cam.move_center(1.0, 0.0); need_redraw = true },

                    KeyCode::Char('p') => {
                        save_screenshot(&cam)?;
                        need_redraw = true;
                    }
                    _ => {}
                },
                Event::Resize(_, _) => {
                    need_redraw = true;
                }
                _ => {} // ignore mouse, focus etc.
            }

            if !need_redraw { continue; } // skip rendering this loop
        }

        if need_redraw {
            let frame = render(&cam);   // or any of your rendering functions
            if frame != last_frame {
                execute!(
                    stdout,
                    cursor::MoveTo(0, 0),
                    terminal::Clear(ClearType::All)
                )?;
                write!(stdout, "{}", frame)?;
                write!(stdout, "\x1b[0m [WASD/Arrows]: Move | +/-: Zoom | Q: Quit | Zoom: {:.4}", cam.zoom)?;
                stdout.flush()?;

                last_frame = frame;
            }

            need_redraw = false; // reset the flag until we get another event
        }
    }

    execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show)?;
    terminal::disable_raw_mode()?;
    Ok(())
}
