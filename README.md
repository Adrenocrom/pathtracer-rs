# Rust Path Tracer (Terminal Edition)

A high-performance, interactive path tracer rendered directly in the terminal using Rust. This project demonstrates the intersection of physically based rendering (PBR) and low-resolution terminal output.

## 🚀 Features

- **Bidirectional Path Tracing:** Camera and light subpaths are traced simultaneously and connected at every vertex pair, with multiple-importance sampling (power heuristic) to combine all connection strategies without bias.
- **Real-time Rendering:** Interactive preview rendered using terminal half-blocks (`▀`) to double vertical resolution.
- **Temporal Accumulation:** Progressive rendering that averages up to 32 frames while the camera remains stationary, cleaning up noise over time.
- **Interactive Exploration:** Full WASD controls for navigating the scene in 3D space.
- **Advanced Post-Processing:**
    - **Bilateral Filtering:** Edge-preserving denoising to remove path-tracing grain while keeping silhouettes crisp.
    - **Gamma Correction:** Linear-to-sRGB conversion for natural lighting.
    - **Reinhard Tone Mapping:** Preserves detail in high-intensity emissive areas.
- **High-Quality Export:** Trigger a full FHD (1920x1080) render with high sample counts, saved as a PNG.
- **BVH Acceleration:** Scene geometry is organized in a bounding volume hierarchy for fast ray intersection.

## 🎮 Controls

| Key | Action |
| :--- | :--- |
| `W` | Move Forward |
| `S` | Move Backward |
| `A` | Rotate Left (Yaw) |
| `D` | Rotate Right (Yaw) |
| `F` | Toggle Denoiser |
| `H` | Show Help |
| `P` | Render High-Quality FHD Screenshot (`screenshot.png`) |
| `Q` / `Esc` | Quit |

## 🛠️ Technical Details

- **Language:** Rust
- **Parallelism:** Powered by `rayon` for multi-core ray tracing (preview and FHD renders).
- **Terminal Handling:** `crossterm` for raw mode, alternate screen, and ANSI 24-bit color sequences.
- **Image Processing:** `image` crate for PNG export.
- **Light Transport:** Bidirectional path tracing with cosine-weighted diffuse sampling, area-light sampling, and MIS power-heuristic weights. Every path edge sampled from either subpath has area pdf `G/π`, which keeps the MIS weights simple and exact.
- **Complexity:** Implements a full orthonormal camera basis and a bidirectional path-tracing loop with vertex connection.

## 📦 Getting Started

### Prerequisites
- Rust (Cargo)
- A terminal that supports 24-bit TrueColor (most modern terminals like Alacritty, iTerm2, Windows Terminal, etc.)

### Running the project
```bash
# Build and run in release mode for maximum performance
cargo run --release
```

### Running the tests
```bash
cargo test
```

## 📈 Rendering Pipeline
`Ray Generation` $\rightarrow$ `Intersection Testing (BVH)` $\rightarrow$ `Bidirectional Path Tracing` $\rightarrow$ `Temporal Accumulation` $\rightarrow$ `Bilateral Filtering` $\rightarrow$ `Tone Mapping` $\rightarrow$ `Gamma Correction` $\rightarrow$ `ANSI Terminal Output`