# Rust Path Tracer (Terminal Edition)

A high-performance, interactive path tracer rendered directly in the terminal using Rust. This project demonstrates the intersection of physically based rendering (PBR) and low-resolution terminal output.

![Example render of the demo scene: a Cornell-box style room with red and green side walls, a glass sphere under the hanging light focusing its light into a bright caustic on the floor, a chrome sphere on the cube stack, and diffuse spheres and cubes.](example.png)

*The demo scene rendered with bidirectional path tracing — a caustic focused through the glass sphere (a light path unidirectional tracing cannot find), reflections in the chrome sphere, indirect color bleeding from the red and green walls, and global illumination in the cube stack.*

## 🚀 Features

- **Bidirectional Path Tracing:** Camera and light subpaths are traced simultaneously and connected at every vertex pair, with multiple-importance sampling (power heuristic, β = 2) to combine all connection strategies without bias.
- **Two Frontends:**
    - **Terminal (`--terminal`):** Interactive preview rendered using terminal half-blocks (`▀`) to double vertical resolution.
    - **Windowed (default, or `--window`):** A native winit window with a wgpu surface. The HDR accumulation buffer is post-processed (denoise + tone map + gamma) entirely in a fragment shader, so pixels never round-trip through the CPU while the camera is stationary. The window title shows accumulated samples and fps; `H` overlays the controls.
- **Temporal Accumulation:** Progressive rendering that averages up to 128 frames while the camera remains stationary, cleaning up noise over time.
- **Interactive Exploration:** WASD-style navigation — forward/back, yaw, strafe and pitch (window frontend; the terminal frontend supports forward and yaw).
- **Advanced Post-Processing:**
    - **Bilateral Filtering:** Edge-preserving denoising on the linear HDR buffer to remove path-tracing grain while keeping silhouettes crisp.
    - **Extended Reinhard Tone Mapping:** Maps the scene white point to display white (`c(1+c/W²)/(1+c)`), so highlights roll off to pure white instead of plateauing below it.
    - **Gamma Correction:** Linear-to-sRGB conversion (γ = 2.2) for natural lighting.
- **High-Quality Export:** Trigger a full FHD (1920×1080) render at 1024 samples per pixel with up to 8 bounces, saved as a PNG with a progress bar (terminal) or in background batches (window).
- **BVH Acceleration:** Scene geometry is organized in a bounding volume hierarchy (median split along the widest axis, with a subtree early-exit test) for fast ray intersection. The room walls are finite quads, not infinite planes, so their bounding boxes are tight and the BVH can actually cull.

## 🎮 Controls

### Window frontend

| Key | Action |
| :--- | :--- |
| `W` / `S` | Move Forward / Backward |
| `A` / `D` | Turn Left / Right (Yaw) |
| `Q` / `E` | Strafe Left / Right |
| `↑` / `↓` | Pitch Up / Down |
| `F` | Toggle Denoiser |
| `P` | Render High-Quality FHD Screenshot (`screenshot.png`) |
| `H` | Toggle Help Overlay |
| `Esc` | Quit |

### Terminal frontend

| Key | Action |
| :--- | :--- |
| `W` / `S` | Move Forward / Backward |
| `A` / `D` | Turn Left / Right (Yaw) |
| `F` | Toggle Denoiser |
| `P` | Render High-Quality FHD Screenshot (`screenshot.png`) |
| `H` | Show Help |
| `Q` / `Esc` | Quit |

## 🛠️ Technical Details

- **Language:** Rust
- **Parallelism:** Powered by `rayon` for multi-core ray tracing (preview and FHD renders); the GPU backend uses one wgpu compute thread per pixel.
- **Terminal Handling:** `crossterm` for raw mode, alternate screen, and ANSI 24-bit color sequences.
- **Windowing:** `winit` 0.30 (`ApplicationHandler`) with a wgpu 30 surface; the display pass reads the accumulation buffer directly (no readback).
- **Image Processing:** `image` crate for PNG export.
- **Light Transport:** Bidirectional path tracing with BSDF-aware sampling (Lambertian diffuse, perfect mirror, and dielectric glass with exact Fresnel and Snell refraction), area-light sampling, and MIS power-heuristic weights computed by Veach's incremental-ratio walk over the actual per-edge pdfs. Delta (specular) BSDFs contribute a symbolic pdf that cancels between strategies, and connections through them are excluded from the MIS normalization — specular transport is handled purely by subpath sampling, which is what lets BDPT find caustics that unidirectional path tracing cannot.
- **Scene:** A Cornell-box style room — red and green side walls, a diffuse sphere, a stack of white cubes with a chrome sphere on top, and a hanging emissive sphere light above a glass sphere that focuses its light into a caustic on the floor.
- **Complexity:** Implements a full orthonormal camera basis and a bidirectional path-tracing loop with vertex connection.

## 📦 Getting Started

### Prerequisites
- Rust (Cargo)
- A terminal that supports 24-bit TrueColor (most modern terminals like Alacritty, iTerm2, Windows Terminal, etc.)
- For the windowed frontend: a GPU with Vulkan / Metal / DX12 support

### Running the project
```bash
# Build and run in release mode for maximum performance.
# The windowed frontend is preferred; it falls back to the terminal UI
# when no window system or GPU is available.
cargo run --release

# Force a specific frontend
cargo run --release -- --window     # native window (GPU only)
cargo run --release -- --terminal   # terminal UI (GPU or CPU)
```

### Running the tests
```bash
cargo test
```
The test suite covers the MIS weights (hand-computed values and sum-to-one), light-transport sanity checks (direct hits, shadows, energy bounds), geometry intersection, the bilateral denoiser, temporal accumulation, and offline WGSL validation of both shaders.

## 📈 Rendering Pipeline
`Ray Generation` $\rightarrow$ `Intersection Testing (BVH)` $\rightarrow$ `Bidirectional Path Tracing` $\rightarrow$ `Temporal Accumulation` $\rightarrow$ `Bilateral Filtering` $\rightarrow$ `Extended Reinhard Tone Mapping` $\rightarrow$ `Gamma Correction` $\rightarrow$ `ANSI Terminal Output / wgpu Surface`