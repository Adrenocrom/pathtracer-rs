# Rust Path Tracer (Terminal Edition)

A high-performance, interactive path tracer rendered directly in the terminal using Rust. This project demonstrates the intersection of physically based rendering (PBR) and low-resolution terminal output.

## 🚀 Features

- **Real-time Rendering:** Interactive preview rendered using terminal half-blocks (`▀`) to double vertical resolution.
- **Temporal Accumulation:** Progressive rendering that cleans up noise and sharpens the image the longer the camera remains stationary.
- **Interactive Exploration:** Full WASD controls for navigating the scene in 3D space.
- **Advanced Post-Processing:**
    - **Bilateral Filtering:** Edge-preserving denoising to remove path-tracing grain while keeping silhouettes crisp.
    - **Gamma Correction:** Linear-to-sRGB conversion for natural lighting.
    - **Reinhard Tone Mapping:** Preserves detail in high-intensity emissive areas.
- **High-Quality Export:** Trigger a full FHD (1920x1080) render with high sample counts, saved as a PNG.

## 🎮 Controls

| Key | Action |
| :--- | :--- |
| `W` | Move Forward |
| `S` | Move Backward |
| `A` | Rotate Left (Yaw) |
| `D` | Rotate Right (Yaw) |
| `P` | Render High-Quality FHD Screenshot (`screenshot.png`) |
| `Q` / `Esc` | Quit |

## 🛠️ Technical Details

- **Language:** Rust
- **Parallelism:** Powered by `rayon` for multi-core ray tracing.
- **Terminal Handling:** `crossterm` for raw mode, alternate screen, and ANSI 24-bit color sequences.
- **Image Processing:** `image` crate for PNG export.
- **Complexity:** Implements a full orthonormal camera basis and a recursive path-tracing loop with diffuse bouncing.

## 📦 Getting Started

### Prerequisites
- Rust (Cargo)
- A terminal that supports 24-bit TrueColor (most modern terminals like Alacritty, iTerm2, Windows Terminal, etc.)

### Running the project
```bash
# Build and run in release mode for maximum performance
cargo run --release
```

## 📈 Rendering Pipeline
`Ray Generation` $\rightarrow$ `Intersection Testing` $\rightarrow$ `Recursive Path Tracing` $\rightarrow$ `Temporal Accumulation` $\rightarrow$ `Bilateral Filtering` $\rightarrow$ `Tone Mapping` $\rightarrow$ `Gamma Correction` $\rightarrow$ `ANSI Terminal Output`
