# Path Tracing Algorithms: A Mathematical Overview

## Introduction
Path tracing is a rendering algorithm in computer graphics that simulates the physical behavior of light as it interacts with objects, surfaces, and participating media. It provides a unified framework for solving the Rendering Equation by employing Monte Carlo integration to estimate complex integrals over high-dimensional spaces.

## Core Principles

### The Rendering Equation
The fundamental equation governing path tracing is the **Rendering Equation**, introduced by Kajiya (1986). For a point $x$ and a direction $\omega_o$, the outgoing radiance $L_o(x, \omega_o)$ is defined as:

$$L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o) L_i(x, \omega_i) \cos(\theta_i) \, d\omega_i$$

Where:
- $L_e(x, \omega_o)$ is the **emitted radiance** at point $x$ in direction $\omega_o$.
- $L_i(x, \omega_i)$ is the **incoming radiance** at point $x$ from direction $\omega_i$.
- $f_r(x, \omega_i, \omega_o)$ is the **Bidirectional Reflectance Distribution Function (BRDF)**, describing the probability of light reflecting from $\omega_i$ to $\omega_o$.
- $\cos(\theta_i)$ is the Lambertian cosine factor ($\mathbf{n} \cdot \omega_i$).
- $\Omega$ is the hemisphere of all possible incoming directions.

### Monte Carlo Integration
To solve the integral in the Rendering Equation, path tracing uses **Monte Carlo integration**. The integral is approximated by a summation over $N$ samples:

$$\int_{\Omega} g(\omega) d\omega \approx \frac{1}{N} \sum_{i=1}^{N} \frac{g(\omega_i)}{p(\omega_i)}$$

Where $p(\omega)$ is the **Probability Density Function (PDF)**. By choosing a PDF that matches the shape of the integrand, we can significantly reduce variance in the resulting estimate.

## The Path Tracing Process
The algorithm approximates the integral by tracing paths from the camera into the scene:

1.  **Ray Generation**: For each pixel, a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ is cast.
2.  **Intersection**: The nearest intersection point $x$ and surface normal $\mathbf{n}$ are determined.
3.  **Sampling**: A new direction $\omega_i$ is sampled from the BRDF:
    $$\omega_i \sim f_r(x, \omega_i, \omega_o) \cos(\theta_i)$$
4.  **Path Accumulation**: The radiance is accumulated along the path. For a single bounce, the contribution is:
    $$C = \frac{f_r(x, \omega_i, \omega_o) L_i(x, \omega_i) \cos(\theta_i)}{p(\omega_i)}$$

## Key Techniques and Optimizations

### 1. Importance Sampling
Instead of uniform sampling, we use **Importance Sampling** to select $\omega_i$ based on the BRDF. For a Lambertian surface:
$$f_r = \frac{\rho}{\pi} \implies p(\omega) = \frac{\cos(\theta)}{\pi}$$
This cancels out the $\cos(\theta)$ term in the Rendering Equation, reducing variance for diffuse surfaces.

### 2. Next Event Estimation (NEE)
To handle small light sources efficiently, we use **Next Event Estimation**. The direct lighting component is calculated by sampling light sources directly at each hit point:
$$L_{direct} = \sum_{l \in Lights} \int_{\Omega_l} f_r(x, \omega_i, \omega_o) L_e(x, \omega_i) \frac{\cos(\theta_i)}{p(\omega_i)} d\omega_i$$
This allows the path to "find" light sources even if they are small relative to the scene.

### 3. Russian Roulette
To terminate paths without introducing bias, we use **Russian Roulette**. A path is continued with probability $P$ and its weight is scaled by $1/P$. If a random number $r \in [0,1]$ is greater than $P$, the path is terminated:
$$L_{new} = \frac{L_{old}}{P} \text{ if } r < P \text{ else } 0$$

### 4. Bidirectional Path Tracing (BDPT)
BDPT considers paths starting from both the camera and the light source. A path is formed by connecting two sub-paths $\mathbf{x}_1$ and $\mathbf{x}_2$ via a geometric connection:
$$L = \sum_{i=0}^{k} \frac{\prod_{j=0}^{i} f_r(x_j, \omega_j) \cos(\theta_j) \Delta \omega_j}{\prod_{j=0}^{i} p(\omega_j)}$$
This is particularly effective for "caustics" and scenes with complex geometry.

## Conclusion
Path tracing transforms the Rendering Equation into a solvable computational problem through Monte Carlo integration. By employing advanced sampling techniques like Importance Sampling, NEE, and Russian Roulette, it provides a mathematically rigorous way to simulate global illumination in 3D environments.
