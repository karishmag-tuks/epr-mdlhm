import math
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


# ============================================================================
# OPTIMIZED MATHEMATICAL FUNCTIONS (Parallelized & Pre-computed)
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def exp_array(x):
    """exponential function using parallelisation"""
    result = np.zeros_like(x)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        result[i, j] = math.exp(x[i, j])
    return result


@njit(fastmath=True, parallel=True, cache=True)
def sqrt_array(x):
    """Compute sqrt for array using math.sqrt - PARALLELIZED"""
    result = np.zeros_like(x)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        if x[i, j] >= 0:
            result[i, j] = math.sqrt(x[i, j])
    return result


@njit(fastmath=True, parallel=True, cache=True)
def sqr_array(x, power):
    """Compute sqrt for array using math.sqrt - PARALLELIZED"""
    result = np.zeros_like(x)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        if x[i, j] >= 0:
            result[i, j] = math.pow(x[i, j], power)
    return result


@njit(fastmath=True, parallel=True, cache=True)
def log10_array(x):
    """Compute log10 for array - PARALLELIZED"""
    result = np.zeros_like(x)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        if x[i, j] > 0:
            result[i, j] = math.log10(x[i, j])
        else:
            result[i, j] = -10.0
    return result


@njit(fastmath=True, parallel=True, cache=True)
def abs_complex(x):
    """Compute absolute value of complex array - PARALLELIZED"""
    result = np.zeros(x.shape, dtype=np.float32)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        real = x[i, j].real
        imag = x[i, j].imag
        result[i, j] = math.sqrt(real * real + imag * imag)
    return result


@njit(fastmath=True, parallel=True, cache=True)
def angle_complex(x):
    """Compute angle of complex array - PARALLELIZED"""
    result = np.zeros(x.shape, dtype=np.float32)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        result[i, j] = math.atan2(x[i, j].imag, x[i, j].real)
    return result


@njit(fastmath=True, parallel=True, cache=True)
def convolution_kernels(a, x, y):
    """convolution kernels computation - PARALLELIZED"""
    result = np.zeros_like(a)
    n = a.size
    for idx in prange(n):
        i = idx // a.shape[1]
        j = idx % a.shape[1]
        result[i, j] = a[i, j] * x[j] * y[i]
    return result


# ============================================================================
# IN-PLACE FFT/IFFT IMPLEMENTATION
# ============================================================================
@njit(cache=True, fastmath=True)
def bit_reverse(i, bits):
    """Bit-reverse an integer with given number of bits
      inputL i: integer to be bit-reversed
      bits: number of bits in the integer
      returns: bit-reversed integer"""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


@njit(cache=True, fastmath=True)
def bit_reverse_permute_inplace(U):
    """Bit-reverse permutation in-place for real and imaginary parts
        The original array elements are swapped in place using the butterfly
        transformation to reorder them according to bit-reversed indices.
        Ur: real part array
        Ui: imaginary part array"""
    n = U.shape[0]
    bits = 0
    tmp = n
    while tmp > 1:
        tmp >>= 1
        bits += 1
    for i in range(n):
        j = bit_reverse(i, bits)
        if j > i:
            t = U[i]
            U[i] = U[j]
            U[j] = t


@njit(cache=True, fastmath=True)
def fft1d_inplace(U, inverse=False):
    """1D FFT in-place with real and imaginary arrays
       weights are manually calculated to avoid extra memory allocation

    Ur: real part array
    Ui: imaginary part array
    inverse: boolean flag for inverse FFT or normal FFT"""
    n = U.shape[0]
    bit_reverse_permute_inplace(U)
    m = 2
    sign = 1.0 if inverse else -1.0
    while m <= n:
        half_m = m >> 1
        theta = sign * 2.0 * np.pi / m
        w_m_r = np.cos(theta)
        w_m_i = np.sin(theta)
        for k in range(0, n, m):
            w_r = 1.0
            w_i = 0.0
            for j in range(half_m):
                u_half = U[k + j + half_m]
                t_r = w_r * u_half.real - w_i * u_half.imag
                t_i = w_r * u_half.imag + w_i * u_half.real
                u_current = U[k + j]
                U[k + j] = u_current.real + t_r + 1j * (u_current.imag + t_i)
                U[k + j + half_m] = (u_current.real - t_r) + 1j * (u_current.imag - t_i)

                tmp_r = w_r * w_m_r - w_i * w_m_i
                tmp_i = w_r * w_m_i + w_i * w_m_r
                w_r = tmp_r
                w_i = tmp_i
        m <<= 1
    if inverse:
        inv_n = 1.0 / n
        for i in range(n):
            U[i] *= inv_n


@njit(cache=True, fastmath=True, parallel=True)
def fft2d_inplace(U, inverse=False):
    """2D FFT in-place with real and imaginary arrays
       U: 2D array of complex
       inverse: boolean flag for inverse FFT or normal FFT"""
    ny, nx = U.shape
    # FFT on rows
    for y in prange(ny):
        fft1d_inplace(U[y, :], inverse)
    # FFT on columns
    for x in prange(nx):
        col = np.empty(ny, dtype=np.complex64)
        for y in range(ny):
            col[y] = U[y, x]
        fft1d_inplace(col, inverse)
        for y in range(ny):
            U[y, x] = col[y]


# ============================================================================
# IMAGE LOADING AND PREPROCESSING (Functional)
# ================================== ==========================================
@njit(cache=True)
def pad_to_pow2_rect(a) -> tuple:
    """
    Zero-pad a 2D array to the next power-of-two sizes in Y and X,
    centered in the padded array.

    Returns:
        out : padded array
        (sy, sx) :  insert offsets
        (Ny, Nx) : original array size
    """
    Ny, Nx = a.shape
    Py = int(2 ** np.ceil(np.log2(Ny)))
    Px = int(2 ** np.ceil(np.log2(Nx)))
    out = np.zeros((Py, Px), dtype=a.dtype)
    sy = (Py - Ny) // 2
    sx = (Px - Nx) // 2
    out[sy:sy + Ny, sx:sx + Nx] = a.copy()
    return out, (sy, sx), (Ny, Nx)


@njit()
def crop_rect(a, offs, size):
    """Crop array to original size"""
    sy, sx = offs
    Ny, Nx = size
    return a[sy:sy + Ny, sx:sx + Nx]


@njit()
def crop_center(image, crop_size):
    """Crop image to crop_size x crop_size from center"""
    h, w = image.shape[:2]
    start_w = (w - crop_size) // 2
    cropped = image[:, start_w:start_w + crop_size]
    return cropped.astype(np.float32)


@njit()
def normalize_image(img: np.ndarray):
    """Normalize image to [0, 1] range as float32"""
    return img.astype(np.float32) / 255.0


def compute_contrast_hologram(object_image: np.ndarray,
                              reference_image: np.ndarray):
    """Compute contrast hologram"""

    contrast = object_image - reference_image

    mean_val = contrast.mean()
    std_val = contrast.std()
    min_val = contrast.min()
    max_val = contrast.max()

    print(f"Contrast hologram statistics:")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std: {std_val:.6f}")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")

    return contrast


# ============================================================================
# CO-ORDINATE TRANSFORMATION AND INTERPOLATION FUNCTIONS
# ============================================================================
@njit(fastmath=True, cache=True)
def bilinear_interpolate(image, x, y, x_grid, y_grid):
    """Bilinear interpolation at point (x, y)"""
    if len(x_grid) < 2 or len(y_grid) < 2:
        return 0.0

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    x_idx = (x - x_grid[0]) / dx
    y_idx = (y - y_grid[0]) / dy

    i = int(np.floor(x_idx))
    j = int(np.floor(y_idx))

    if i < 0 or i >= len(x_grid) - 1 or j < 0 or j >= len(y_grid) - 1:
        return 0.0

    fx = x_idx - i
    fy = y_idx - j

    # creates four quadrants for the images
    Q11 = image[j, i]
    Q21 = image[j, i + 1]
    Q12 = image[j + 1, i]
    Q22 = image[j + 1, i + 1]

    R1 = Q11 * (1 - fx) + Q21 * fx
    R2 = Q12 * (1 - fx) + Q22 * fx
    P = R1 * (1 - fy) + R2 * fy

    return P


@njit(fastmath=True, parallel=True, cache=True)
def interpolate_2d_grid(image, X_target, Y_target, x_grid, y_grid):
    """Interpolate image onto a new 2D grid"""
    M, N = X_target.shape
    result = np.zeros((M, N), dtype=np.float32)

    for i in prange(M):
        for j in range(N):
            result[i, j] = bilinear_interpolate(
                image, X_target[i, j], Y_target[i, j],
                x_grid, y_grid
            )

    return result


@njit(fastmath=True, parallel=True)
def prepare_transformed_hologram(hologram: np.ndarray, z: float, L: float, pixel_size: float, wavelength: float):
    M, N = hologram.shape
    k = 2.0 * np.pi / wavelength

    X0 = -(N // 2) * pixel_size
    Y0 = -(M // 2) * pixel_size

    j = np.arange(N, dtype=np.float32)
    i = np.arange(M, dtype=np.float32)

    X_grid = X0 + j * pixel_size
    Y_grid = Y0 + i * pixel_size

    X_max = X0 + (N - 1) * pixel_size
    Y_max = Y0 + (M - 1) * pixel_size

    X_prime_0 = X0 * L / np.sqrt(L ** 2 + X0 ** 2)
    Y_prime_0 = Y0 * L / np.sqrt(L ** 2 + Y0 ** 2)

    X_prime_max = X_max * L / np.sqrt(L ** 2 + X_max ** 2)
    Y_prime_max = Y_max * L / np.sqrt(L ** 2 + Y_max ** 2)

    Delta_x_prime = (X_prime_max - X_prime_0) / N
    Delta_y_prime = (Y_prime_max - Y_prime_0) / M

    X_prime_1d = X_prime_0 + j * Delta_x_prime
    Y_prime_1d = Y_prime_0 + i * Delta_y_prime

    X_prime_2d = np.zeros((M, N), dtype=np.float32)
    Y_prime_2d = np.zeros((M, N), dtype=np.float32)

    for row in prange(M):
        for col in range(N):
            X_prime_2d[row, col] = X_prime_1d[col]  # Column → X
            Y_prime_2d[row, col] = Y_prime_1d[row]  # Row → Y

    R_prime = np.sqrt(L ** 2 - X_prime_2d ** 2 - Y_prime_2d ** 2)

    X_orig = X_prime_2d * L / R_prime
    Y_orig = Y_prime_2d * L / R_prime

    I_interp = interpolate_2d_grid(hologram, X_orig, Y_orig, X_grid, Y_grid)

    Jacobian = (L / R_prime) ** 4

    phase_factor = np.zeros((M, N), dtype=np.complex64)
    phase_arg = k * z * R_prime / L

    for row in prange(M):
        for col in range(N):
            phase_factor[row, col] = np.cos(phase_arg[row, col]) + 1j * np.sin(phase_arg[row, col])

    I_prime = I_interp * Jacobian * phase_factor

    return I_prime, X_prime_0, Y_prime_0, Delta_x_prime, Delta_y_prime


# ============================================================================
# PHASE COMPUTATION FOR FFTs
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def compute_K(I_pad, x0, y0, Delta_x_prime, Delta_y_prime, delta_x, delta_y, k_over_L):
    M, N = I_pad.shape

    j = np.arange(N, dtype=np.float32)
    j_prime = np.arange(M, dtype=np.float32)

    phase = np.zeros((M, N), dtype=np.float32)

    for i in range(M):
        j_p = j_prime[i]
        y_term = j_p * y0 * Delta_y_prime + j_p * j_p * delta_y * Delta_y_prime / 2.0

        for j_idx in range(N):
            j_val = j[j_idx]
            x_term = j_val * x0 * Delta_x_prime + j_val * j_val * delta_x * Delta_x_prime / 2.0
            phase[i, j_idx] = k_over_L * (x_term + y_term)

    I_with_phase = np.zeros((M, N), dtype=np.complex64)
    for i in range(M):
        for j_idx in range(N):
            cos_p = np.cos(phase[i, j_idx])
            sin_p = np.sin(phase[i, j_idx])
            real = I_pad[i, j_idx].real * cos_p - I_pad[i, j_idx].imag * sin_p
            imaginary = I_pad[i, j_idx].real * sin_p + I_pad[i, j_idx].imag * cos_p
            I_with_phase[i, j_idx] = real + 1j * imaginary

    return I_with_phase


@njit(fastmath=True, parallel=True, cache=True)
def compute_phase_prefactor_result(convolution_result, N, M, x0, y0,
                                   X_prime_0, Y_prime_0, delta_x, delta_y,
                                   Delta_x_prime, Delta_y_prime, k, L):
    n = np.arange(N, dtype=np.float32)
    m = np.arange(M, dtype=np.float32)

    k_over_L = k / L
    k_over_2L = k / (2.0 * L)

    K = np.zeros((M, N), dtype=np.complex64)

    for i in prange(M):
        m_val = m[i]
        phase1_y = (y0 + m_val * delta_y) * Y_prime_0
        phase2_y = m_val * m_val * delta_y * Delta_y_prime

        for j_idx in range(N):
            n_val = n[j_idx]

            phase1_x = (x0 + n_val * delta_x) * X_prime_0
            phase2_x = n_val * n_val * delta_x * Delta_x_prime

            phase1 = k_over_L * (phase1_x + phase1_y)
            phase2 = k_over_2L * (phase2_x + phase2_y)

            total_phase = phase1 + phase2

            phase_prefix_real = Delta_x_prime * Delta_y_prime * np.cos(total_phase)
            phase_prefix_imag = Delta_x_prime * Delta_y_prime * np.sin(total_phase)

            conv_real = convolution_result[i, j_idx].real
            conv_imag = convolution_result[i, j_idx].imag

            K[i, j_idx] = complex(
                phase_prefix_real * conv_real - phase_prefix_imag * conv_imag,
                phase_prefix_real * conv_imag + phase_prefix_imag * conv_real
            )

    return K


# ============================================================================
# RECONSTRUCTION FUNCTION
# ============================================================================

@njit(fastmath=True, parallel=True)
def reconstruct(hologram: np.ndarray, z: float, wavelength: float, L: float,
                pixel_size: float) -> np.ndarray:
    """Main reconstruction using pure functional approach with optimized FFT"""

    M, N = hologram.shape

    # Pre-compute constants
    k = 2 * math.pi / wavelength
    k_over_L = k / L
    k_over_2L = k / (2 * L)
    I_prime, X_prime_0, Y_prime_0, Delta_x_prime, Delta_y_prime = prepare_transformed_hologram(hologram, z, L,
                                                                                               pixel_size, wavelength)
    delta_x = wavelength * L / (N * abs(Delta_x_prime))
    delta_y = wavelength * L / (M * abs(Delta_y_prime))
    x0 = -(N // 2) * delta_x
    y0 = -(M // 2) * delta_y

    I_pad, (sy, sx), (Ny, Nx) = pad_to_pow2_rect(I_prime)
    Py, Px = I_pad.shape

    print(f" computing first FFT ")
    A = compute_K(I_pad, x0, y0, Delta_x_prime, Delta_y_prime, delta_x, delta_y, k_over_L)
    # Convert complex to separate real/imaginary for in-place FFT
    print(f"    Computing FFT2 ({Py}x{Px}) with optimized in-place algorithm...")

    fft2d_inplace(A)

    jx = np.arange(Px, dtype=np.float32)
    jy = np.arange(Py, dtype=np.float32)

    print(f"    Computing convolution kernels...")
    R_x = np.zeros(Px, dtype=np.complex64)

    for i in prange(Px):
        phase_x = -1 * k_over_2L * jx[i] ** 2 * delta_x * Delta_x_prime
        R_x[i] = math.cos(phase_x) + 1j * math.sin(phase_x)

    fft1d_inplace(R_x, inverse=False)

    R_y = np.zeros(Py, dtype=np.complex64)

    for j in prange(Py):
        phase_y = -1 * k_over_2L * jy[j] ** 2 * delta_y * Delta_y_prime
        R_y[j] = math.cos(phase_y) + 1j * math.sin(phase_y)

    fft1d_inplace(R_y, inverse=False)

    c_results = convolution_kernels(A, R_x, R_y)

    fft2d_inplace(c_results, inverse=True)

    # Vectorized prefactor computation
    K_pad = compute_phase_prefactor_result(c_results, Px, Py, x0, y0,
                                           X_prime_0, Y_prime_0, delta_x, delta_y,
                                           Delta_x_prime, Delta_y_prime, k, L)
    K = crop_rect(K_pad, (sy, sx), (Ny, Nx))

    return K


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def demonstrate_reconstruction():
    """Main reconstruction demonstration - purely functional"""

    print("=" * 70)
    print("ULTRA-OPTIMIZED HOLOGRAM RECONSTRUCTION (Functional Programming)")
    print("(Optimized In-Place FFT + Parallelized + Vectorized phases)")
    print("=" * 70)

    object_image_path = "654_holo.jpg"
    reference_image_path = "654_ref.jpg"

    if not Path(object_image_path).exists():
        print("Error: Object image not found:", object_image_path)
        return
    elif not Path(reference_image_path).exists():
        print(f"Error: Reference image not found", reference_image_path)
        return

    wavelength = 654e-9  # 654 nm in meters
    L = 18.52e-3  # 18.52 mm in meters
    pixel_size = 3.8e-6  # 3.8 μm in meters
    crop_size = 4096  # Crop size for images

    print("\nLoading images...")
    object_image = cv2.imread(object_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    print(f"  Original image size: {object_image.shape}")

    print(f"\nCropping to {crop_size}x{crop_size} from center...")
    object_image = crop_center(object_image, crop_size)
    reference_image = crop_center(reference_image, crop_size)

    print(f"  Cropped image size: {object_image.shape}")

    print("-" * 70)
    print("\nComputing contrast hologram...")
    hologram = compute_contrast_hologram(object_image, reference_image)

    print("\nReconstruction parameters:")
    print(f"  Wavelength: {wavelength} μm ({wavelength} nm)")
    print(f"  Pinhole-to-CCD distance: {L} mm")
    print(f"  Pixel size: {pixel_size} μm")
    print(f"  Image size after crop: {crop_size}x{crop_size}")

    z_values = [6.2e-3]

    print(f"\nReconstruction parameters:")
    print(f"  Reconstruction depths: {z_values} μm")

    reconstructions = []

    for z in z_values:
        print(f"\nReconstructing at z = {z} μm from pinhole...")

        t0 = time.time()
        K = reconstruct(hologram, z, wavelength, L, pixel_size)
        t_elapsed = time.time() - t0

        print(f"  ✓ Reconstruction time: {t_elapsed:.3f} seconds")

        K_abs = abs_complex(K)
        print(f"  Wavefront amplitude range: [{K_abs.min():.6f}, {K_abs.max():.6f}]")

        reconstructions.append((z, K))
    num_depths = len(reconstructions)
    ncols = max(3, num_depths)

    fig, axes = plt.subplots(3, ncols, figsize=(3 * ncols, 8), constrained_layout=True)

    # Ensure axes is 2D even if ncols==1
    # --- Top row: input images ---
    axes[0, 0].imshow(object_image, cmap='gray')
    axes[0, 0].set_title('Object/Hologram Image', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reference_image, cmap='gray')
    axes[0, 1].set_title('Reference Wave Image', fontsize=10)
    axes[0, 1].axis('off')

    im0 = axes[0, 2].imshow(hologram, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('Contrast Hologram\n(Background Subtracted)', fontsize=10)
    axes[0, 2].axis('off')
    fig.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # If there are extra columns beyond 3 in the top row, turn them off
    for c in range(3, ncols):
        axes[0, c].axis('off')

    # --- Middle row: reconstructed intensity ---
    for i, (z, K) in enumerate(reconstructions):
        ax = axes[1, i]
        intensity = abs_complex(K) ** 2
        intensity_log = np.log10(intensity + 1e-10)
        im = ax.imshow(intensity_log, cmap='hot')
        ax.set_title(f'Intensity at z={z} m\n(log scale)', fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused intensity axes
    for i in range(num_depths, ncols):
        axes[1, i].axis('off')

    # --- Bottom row: reconstructed phase ---
    for i, (z, K) in enumerate(reconstructions):
        ax = axes[2, i]
        phase = angle_complex(K)
        im = ax.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax.set_title(f'Phase at z={z} m', fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused phase axes
    for i in range(num_depths, ncols):
        axes[2, i].axis('off')

    output_filename = 'hologram_reconstruction_from_images.png'
    fig.suptitle('Hologram Reconstruction', y=0.995, fontsize=12)
    fig.savefig(output_filename, dpi=150)
    print(f"\nVisualization saved as '{output_filename}'")
    plt.show()

    for z, K in reconstructions:
        intensity = np.log10(abs_complex(K) ** 2 + 1e-12)
        intensity_norm = (intensity_log - intensity_log.min()) / (intensity_log.max() - intensity_log.min() + 1e-10)
        intensity_8bit = np.clip(intensity_norm * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f'reconstruction_intensity_z{int(z * 1e3)}.png', intensity_8bit)
        print(f"  ✓ Saved reconstruction at z={z}m")


if __name__ == "__main__":
    demonstrate_reconstruction()
