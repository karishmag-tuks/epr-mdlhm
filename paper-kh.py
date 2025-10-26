import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time
from numba import jit, prange
import math


# ============================================================================
# OPTIMIZED MATHEMATICAL FUNCTIONS (Parallelized & Pre-computed)
# ============================================================================

@jit(nopython=True, fastmath=True, parallel=True)
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


@jit(nopython=True, fastmath=True, parallel=True)
def angle_complex(x):
    """Compute angle of complex array - PARALLELIZED"""
    result = np.zeros(x.shape, dtype=np.float32)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        result[i, j] = math.atan2(x[i, j].imag, x[i, j].real)
    return result


@jit(nopython=True, fastmath=True, parallel=True)
def conj_complex(x):
    """Compute conjugate of complex array - PARALLELIZED"""
    result = np.zeros_like(x)
    n = x.size
    for idx in prange(n):
        i = idx // x.shape[1]
        j = idx % x.shape[1]
        result[i, j] = x[i, j].conjugate()
    return result


# ============================================================================
# PRE-COMPUTED TWIDDLE FACTORS
# ============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def precompute_twiddle_factors(n):
    """Pre-compute all twiddle factors for FFT of size n"""
    twiddle = np.zeros((n,), dtype=np.complex64)
    two_pi_over_n = 2.0 * math.pi / n
    
    for k in range(n):
        angle = -two_pi_over_n * k
        twiddle[k] = complex(math.cos(angle), math.sin(angle))
    
    return twiddle


# ============================================================================
# OPTIMIZED FFT WITH TWIDDLE FACTORS
# ============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def bit_reverse_1d(x):
    """Bit-reverse permutation for 1D array"""
    n = x.shape[0]
    result = np.zeros_like(x)
    j = 0
    
    for i in range(n):
        result[j] = x[i]
        bit = n >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
    
    return result


@jit(nopython=True, fastmath=True, cache=True)
def fft_1d_with_twiddles(x, twiddle_factors):
    """1D FFT using pre-computed twiddle factors"""
    n = x.shape[0]
    
    if n & (n - 1) != 0:
        raise ValueError("FFT length must be power of 2")
    
    if n == 1:
        return x.copy()
    
    X = bit_reverse_1d(x)
    
    # Compute number of stages
    num_stages = 0
    temp_n = n
    while temp_n > 1:
        num_stages += 1
        temp_n >>= 1
    
    # Iterative FFT computation using pre-computed twiddles
    for s in range(1, num_stages + 1):
        m = 1 << s
        omega_m_idx_step = n // m
        
        for k in range(0, n, m):
            omega_idx = 0
            for j in range(m // 2):
                omega = twiddle_factors[omega_idx]
                
                # Butterfly operation
                t_real = omega.real * X[k + j + m // 2].real - omega.imag * X[k + j + m // 2].imag
                t_imag = omega.real * X[k + j + m // 2].imag + omega.imag * X[k + j + m // 2].real
                
                u_real = X[k + j].real
                u_imag = X[k + j].imag
                
                X[k + j] = complex(u_real + t_real, u_imag + t_imag)
                X[k + j + m // 2] = complex(u_real - t_real, u_imag - t_imag)
                
                omega_idx += omega_m_idx_step
    
    return X


@jit(nopython=True, fastmath=True, cache=True)
def ifft_1d_with_twiddles(X, twiddle_factors):
    """Inverse 1D FFT using pre-computed twiddles"""
    n = X.shape[0]
    X_conj = conj_complex(X)
    result = fft_1d_with_twiddles(X_conj, twiddle_factors)
    result_conj = conj_complex(result)
    
    for i in range(n):
        result_conj[i] /= n
    
    return result_conj


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def fft_2d_with_twiddles(x, twiddle_x, twiddle_y):
    """2D FFT using 1D FFT on rows then columns - PARALLELIZED"""
    M, N = x.shape
    
    # FFT on rows - PARALLELIZED
    X = np.zeros_like(x, dtype=np.complex64)
    for i in prange(M):
        X[i, :] = fft_1d_with_twiddles(x[i, :], twiddle_x)
    
    # FFT on columns - PARALLELIZED
    for j in prange(N):
        X[:, j] = fft_1d_with_twiddles(X[:, j], twiddle_y)
    
    return X


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def ifft_2d_with_twiddles(X, twiddle_x, twiddle_y):
    """2D inverse FFT - PARALLELIZED"""
    M, N = X.shape
    
    # IFFT on rows
    x = np.zeros_like(X, dtype=np.complex64)
    for i in prange(M):
        x[i, :] = ifft_1d_with_twiddles(X[i, :], twiddle_x)
    
    # IFFT on columns
    for j in prange(N):
        x[:, j] = ifft_1d_with_twiddles(x[:, j], twiddle_y)
    
    return x


# ============================================================================
# VECTORIZED PHASE COMPUTATION
# ============================================================================

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def compute_phase_A_vectorized(Py, Px, n, m, x_0, y_0, Delta_x_prime, Delta_y_prime,
                               delta_x, delta_y, k_over_L):
    """Vectorized phase computation"""
    phase_A = np.zeros((Py, Px), dtype=np.float32)
    
    # Pre-compute column-dependent terms
    col_term = np.zeros(Px, dtype=np.float32)
    for j in prange(Px):
        n_j = n[j]
        col_term[j] = n_j * x_0 * Delta_x_prime + n_j * n_j * delta_x * Delta_x_prime / 2.0
    
    # Pre-compute row-dependent terms (parallelized)
    row_term = np.zeros(Py, dtype=np.float32)
    for i in prange(Py):
        m_i = m[i]
        row_term[i] = m_i * y_0 * Delta_y_prime + m_i * m_i * delta_y * Delta_y_prime / 2.0
    
    # Combine
    for i in prange(Py):
        for j in range(Px):
            phase_A[i, j] = k_over_L * (col_term[j] + row_term[i])
    
    return phase_A


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def compute_phase_prefactor_vectorized(Py, Px, n, m, x_0, y_0, delta_x, delta_y,
                                      X_prime_0, Y_prime_0, k_over_L, k_over_2L):
    """Vectorized prefactor computation"""
    
    # Pre-compute column-dependent terms
    col_phase2 = np.zeros(Px, dtype=np.float32)
    col_phase3 = np.zeros(Px, dtype=np.float32)
    for j in prange(Px):
        n_j = n[j]
        col_phase2[j] = (x_0 + n_j * delta_x) * X_prime_0
        col_phase3[j] = n_j * n_j * delta_x
    
    # Pre-compute row-dependent terms (parallelized)
    row_phase2 = np.zeros(Py, dtype=np.float32)
    row_phase3 = np.zeros(Py, dtype=np.float32)
    for i in prange(Py):
        m_i = m[i]
        row_phase2[i] = (y_0 + m_i * delta_y) * Y_prime_0
        row_phase3[i] = m_i * m_i * delta_y
    
    # Combine phase factors
    phase_prefactor2 = np.zeros((Py, Px), dtype=np.float32)
    phase_prefactor3 = np.zeros((Py, Px), dtype=np.float32)
    
    for i in prange(Py):
        for j in range(Px):
            phase_prefactor2[i, j] = k_over_L * (col_phase2[j] + row_phase2[i])
            phase_prefactor3[i, j] = k_over_2L * (col_phase3[j] + row_phase3[i])
    
    return phase_prefactor2, phase_prefactor3


# ============================================================================
# IMAGE LOADING AND PREPROCESSING (Functional)
# ============================================================================

def next_pow2(n: int) -> int:
    """Smallest power of two >= n (n > 0)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 if n == 1 else 1 << ((n - 1).bit_length())

def pad_to_pow2_rect(a):
    """
    Zero-pad a 2D array to the next power-of-two sizes in Y and X,
    centered in the padded array.

    Returns:
        out : padded array
        (sy, sx) : top-left insert offsets
        (Ny, Nx) : original array size
    """
    Ny, Nx = a.shape
    Py = next_pow2(Ny)
    Px = next_pow2(Nx)

    out = np.zeros((Py, Px), dtype=a.dtype)
    sy = (Py - Ny) // 2
    sx = (Px - Nx) // 2
    out[sy:sy + Ny, sx:sx + Nx] = a
    return out, (sy, sx), (Ny, Nx)


def crop_rect(a, offs, size):
    """Crop array to original size"""
    sy, sx = offs
    Ny, Nx = size
    return a[sy:sy + Ny, sx:sx + Nx]


def normalize_image(img: np.ndarray):
    """Normalize image to [0, 1] range as float32"""
    return img.astype(np.float32) / 255.0


def compute_contrast_hologram(object_image: np.ndarray,
                              reference_image: np.ndarray):
    """Compute contrast hologram"""
    obj_norm = normalize_image(object_image)
    ref_norm = normalize_image(reference_image)
    contrast = obj_norm - ref_norm

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
# PRE-COMPUTED CONSTANTS (Pure Functional Parameters)
# ============================================================================

def create_reconstruction_config(wavelength: float, L: float, pixel_size: float) -> dict:
    """Create all pre-computed constants as a configuration dictionary"""
    k = 2 * math.pi / wavelength
    L_mm = L * 1000
    
    return {
        'wavelength': wavelength,
        'L': L_mm,
        'pixel_size': pixel_size,
        'k': k,
        'k_over_L': k / L_mm,
        'k_over_2L': k / (2 * L_mm),
        'L_squared': L_mm * L_mm,
        'two_pi': 2 * math.pi,
        'inv_wavelength': 1.0 / wavelength,
        'inv_L': 1.0 / L_mm,
        'inv_pixel_size': 1.0 / pixel_size,
    }


def create_twiddle_cache() -> dict:
    """Create empty twiddle factor cache"""
    return {}


def get_or_compute_twiddles(cache: dict, n: int, label: str = "") -> np.ndarray:
    """Get twiddle factors from cache or compute and cache them"""
    if n not in cache:
        if label:
            print(f"      Pre-computing twiddle factors for size {n} ({label})...")
        else:
            print(f"      Pre-computing twiddle factors for size {n}...")
        cache[n] = precompute_twiddle_factors(n)
    return cache[n]


# ============================================================================
# COORDINATE TRANSFORMATION (Pure Functional)
# ============================================================================

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def coordinate_transform(X, Y, L_squared, L):
    """Transform coordinates: X' = XL/R, Y' = YL/R"""
    R = np.sqrt(L_squared + X ** 2 + Y ** 2)
    L_over_R = L / R
    
    X_prime = X * L_over_R
    Y_prime = Y * L_over_R
    R_prime = L_squared / R
    
    return X_prime, Y_prime, R_prime

@jit(nopython=True, fastmath=True, parallel=True)
def prepare_coordinate_grids(M: int, N: int, pixel_size: float) -> tuple:
    """Prepare X and Y coordinate grids"""
    coords_n = np.arange(N, dtype=np.float32) - N // 2
    coords_m = np.arange(M, dtype=np.float32) - M // 2
    
    X = np.zeros((M, N), dtype=np.float32)
    Y = np.zeros((M, N), dtype=np.float32)
    
    for i in prange(M):
        for j in range(N):
            X[i, j] = coords_n[j] * pixel_size
            Y[i, j] = coords_m[i] * pixel_size
    
    return X, Y


# ============================================================================
# HOLOGRAM PREPARATION (Pure Functional)
# ============================================================================
@jit(nopython=True, fastmath=True, parallel=True)
def prepare_transformed_hologram(hologram: np.ndarray, z: float, config: dict) -> tuple:
    """Prepare I'(X', Y') with phase factors"""
    M, N = hologram.shape
    
    print(f"    Processing hologram {M}x{N}...")
    
    # Prepare coordinates
    X, Y = prepare_coordinate_grids(M, N, config['pixel_size'])
    
    # Transform coordinates
    X_prime, Y_prime, R_prime = coordinate_transform(X, Y, config['L_squared'], config['L'])
    
    M_center = M // 2
    N_center = N // 2
    X_prime_0 = X_prime[M_center, N_center]
    Y_prime_0 = Y_prime[M_center, N_center]
    

    
    # Compute phase factors
    jacobian_factor = (config['L'] / R_prime) ** 4
    
    # Compute phase
    phase_factor = np.zeros(R_prime.shape, dtype=np.complex64)
    phase_scale = config['k'] * z / config['L']
    for i in prange(R_prime.shape[0]):
        for j in prange(R_prime.shape[1]):
            phase = phase_scale * R_prime[i, j]
            phase_factor[i, j] = complex(math.cos(phase), math.sin(phase))
    
    I_prime = np.zeros((M, N), dtype=np.complex64)
    for i in prange(M):
        for j in prange(N):
            factor = hologram[i, j] * jacobian_factor[i, j]
            I_prime[i, j] = complex(factor * phase_factor[i, j].real,
                                   factor * phase_factor[i, j].imag)
    
    return I_prime, X_prime_0, Y_prime_0


# ============================================================================
# RECONSTRUCTION PARAMETERS (Pure Functional)
# ============================================================================

def compute_reconstruction_params(N: int, M: int, config: dict,
                                 magnification_x: float = 1.0,
                                 magnification_y: float = 1.0) -> tuple:
    """Compute reconstruction parameters"""
    Delta_x_prime = config['pixel_size']
    Delta_y_prime = config['pixel_size']

    delta_x = (config['wavelength'] * config['L']) / (N * Delta_x_prime) / magnification_x
    delta_y = (config['wavelength'] * config['L']) / (M * Delta_y_prime) / magnification_y

    return delta_x, delta_y, Delta_x_prime, Delta_y_prime


# ============================================================================
# MAIN RECONSTRUCTION FUNCTION (Pure Functional)
# ============================================================================
@jit(nopython=True, fastmath=True, parallel=True)
def reconstruct(hologram: np.ndarray, z: float, config: dict, twiddle_cache: dict,
               magnification_x: float = 1.0, magnification_y: float = 1.0) -> np.ndarray:
    """Main reconstruction using pure functional approach"""
    
    M, N = hologram.shape
    
    I_prime, X_prime_0, Y_prime_0 = prepare_transformed_hologram(hologram, z, config)
    
    I_pad, (sy, sx), (Ny, Nx) = pad_to_pow2_rect(I_prime)
    Py, Px = I_pad.shape

    delta_x, delta_y, Delta_x_prime, Delta_y_prime = \
        compute_reconstruction_params(N, M, config, magnification_x, magnification_y)

    n = np.arange(Px, dtype=np.float32)
    m = np.arange(Py, dtype=np.float32)
    x_0 = -(Px // 2) * delta_x
    y_0 = -(Py // 2) * delta_y

    # Compute phase modulation (vectorized)
    print(f"    Computing phase modulation (vectorized)...")
    phase_A = compute_phase_A_vectorized(Py, Px, n, m, x_0, y_0, Delta_x_prime,
                                        Delta_y_prime, delta_x, delta_y, config['k_over_L'])
    
    A = np.zeros_like(I_pad, dtype=np.complex64)
    for i in prange(Py):
        for j in prange(Px):
            cos_p = math.cos(phase_A[i, j])
            sin_p = math.sin(phase_A[i, j])
            real = I_pad[i, j].real * cos_p - I_pad[i, j].imag * sin_p
            imag = I_pad[i, j].real * sin_p + I_pad[i, j].imag * cos_p
            A[i, j] = complex(real, imag)
    
    # Get twiddle factors (cached)
    print(f"    Computing FFT2 ({Py}x{Px}) with pre-computed twiddles...")
    twiddle_x = get_or_compute_twiddles(twiddle_cache, Px, "rows")
    twiddle_y = get_or_compute_twiddles(twiddle_cache, Py, "columns")
    
    K_prime = fft_2d_with_twiddles(A, twiddle_x, twiddle_y)

    jx = np.arange(Px, dtype=np.float32)
    jy = np.arange(Py, dtype=np.float32)

    print(f"    Computing convolution kernels...")
    phase_R_x = np.zeros(Px, dtype=np.float32)
    for i in prange(Px):
        phase_R_x[i] = -(config['k'] * jx[i] ** 2 * delta_x * Delta_x_prime) / (2 * config['L'])
    
    R_x_input = np.zeros(Px, dtype=np.complex64)
    for i in prange(Px):
        R_x_input[i] = complex(math.cos(phase_R_x[i]), math.sin(phase_R_x[i]))
    
    R_nu_x = fft_1d_with_twiddles(R_x_input, twiddle_x)

    phase_R_y = np.zeros(Py, dtype=np.float32)
    for i in prange(Py):
        phase_R_y[i] = -(config['k'] * jy[i] ** 2 * delta_y * Delta_y_prime) / (2 * config['L'])
    
    R_y_input = np.zeros(Py, dtype=np.complex64)
    for i in prange(Py):
        R_y_input[i] = complex(math.cos(phase_R_y[i]), math.sin(phase_R_y[i]))
    
    R_nu_y = fft_1d_with_twiddles(R_y_input, twiddle_y)

    print(f"    Convolving in frequency domain...")
    Product = np.zeros_like(K_prime)
    for i in prange(Py):
        for j in range(Px):
            temp = K_prime[i, j] * R_nu_x[j]
            Product[i, j] = temp * R_nu_y[i]
    

    convolution_result = ifft_2d_with_twiddles(Product, twiddle_x, twiddle_y)

    print(f"    Computing prefactors (vectorized)...")
    prefactor1 = Delta_x_prime * Delta_y_prime
    
    # Vectorized prefactor computation
    phase_prefactor2, phase_prefactor3 = compute_phase_prefactor_vectorized(
        Py, Px, n, m, x_0, y_0, delta_x, delta_y,
        X_prime_0, Y_prime_0, config['k_over_L'], config['k_over_2L']
    )

    K_pad = np.zeros_like(convolution_result)
    for i in prange(Py):
        for j in range(Px):
            phase_total = phase_prefactor2[i, j] + phase_prefactor3[i, j]
            cos_p = math.cos(phase_total)
            sin_p = math.sin(phase_total)
            
            real = convolution_result[i, j].real * cos_p - convolution_result[i, j].imag * sin_p
            imag = convolution_result[i, j].real * sin_p + convolution_result[i, j].imag * cos_p
            
            K_pad[i, j] = complex(real * prefactor1, imag * prefactor1)


    K = crop_rect(K_pad, (sy, sx), (Ny, Nx))

    return K


# ============================================================================
# MAIN DEMONSTRATION (Pure Functional)
# ============================================================================

def demonstrate_reconstruction():
    """Main reconstruction demonstration - purely functional"""

    print("=" * 70)
    print("ULTRA-OPTIMIZED HOLOGRAM RECONSTRUCTION (Functional Programming)")
    print("(Pre-computed constants + Parallelized FFT + Vectorized phases)")
    print("=" * 70)

    object_image_path = "654_holo.jpg"
    reference_image_path = "654_ref.jpg"

    if not Path(object_image_path).exists():
        print(f"Error: Object image not found: {object_image_path}")
        return
    elif not Path(reference_image_path).exists():
        print(f"Error: Reference image not found: {reference_image_path}")
        return

    wavelength = 0.654
    L = 18.52
    pixel_size = 3.8
    target_size_x = 4640
    target_size_y = 3506

    print("\nLoading images...")
    object_image = cv2.imread(object_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    print("-" * 70)
    print("\nComputing contrast hologram...")
    hologram = compute_contrast_hologram(object_image, reference_image)

    print("\nCreating reconstruction configuration...")
    print(f"  Wavelength: {wavelength} μm ({wavelength * 1000:.0f} nm)")
    print(f"  Pinhole-to-CCD distance: {L} mm")
    print(f"  Pixel size: {pixel_size} μm")

    config = create_reconstruction_config(wavelength, L, pixel_size)
    twiddle_cache = create_twiddle_cache()

    z_values = [620]
    # magnification_x = (wavelength * L * 1000) / (target_size_x * pixel_size ** 2)
    # magnification_y = (wavelength * L * 1000) / (target_size_y * pixel_size ** 2)
    magnification_x = 1.0
    magnification_y = 1.0

    print(f"\nReconstruction parameters:")
    print(f"  Magnification x: {magnification_x}x")
    print(f"  Magnification y: {magnification_y}x")
    print(f"  Reconstruction depths: {z_values} μm")

    reconstructions = []

    for z in z_values:
        print(f"\nReconstructing at z = {z} μm from pinhole...")

        t0 = time.time()
        K = reconstruct(hologram, z, config, twiddle_cache,
                       magnification_x=magnification_x,
                       magnification_y=magnification_y)
        t_elapsed = time.time() - t0

        print(f"  ✓ Reconstruction time: {t_elapsed:.3f} seconds")
        
        K_abs = abs_complex(K)
        print(f"  Wavefront amplitude range: [{K_abs.min():.6f}, {K_abs.max():.6f}]")

        reconstructions.append((z, K))

    # Visualize
    print("\nGenerating visualization...")
    num_depths = len(reconstructions)
    plt.figure(figsize=(3, 3))

    ax1 = plt.subplot(3, 1, 1)
    ax1.imshow(object_image, cmap='gray')
    ax1.set_title('Object/Hologram Image', fontsize=10)
    ax1.axis('off')

    ax1 = plt.subplot(3, 1, 2)
    ax1.imshow(reference_image, cmap='gray')
    ax1.set_title('Reference Wave Image', fontsize=10)
    ax1.axis('off')

    ax1 = plt.subplot(3, 1, 3)
    ax1.imshow(hologram, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax1.set_title('Contrast Hologram\n(Background Subtracted)', fontsize=10)
    ax1.axis('off')

    for i, (z, K) in enumerate(reconstructions):
        ax = plt.subplot(3, num_depths, num_depths + i + 1)
        intensity = abs_complex(K) ** 2
        intensity_log = np.log10(intensity + 1e-10)

        im = ax.imshow(intensity_log, cmap='hot')
        ax.set_title(f'Intensity at z={z}μm\n(log scale)', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, (z, K) in enumerate(reconstructions):
        ax = plt.subplot(3, num_depths, 2 * num_depths + i + 1)
        phase = angle_complex(K)

        im = ax.imshow(phase, cmap='twilight', vmin=-math.pi, vmax=math.pi)
        ax.set_title(f'Phase at z={z}μm', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_filename = 'hologram_reconstruction_from_images.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\nVisualization saved as '{output_filename}'")
    plt.show()

    # Save results
    print("\nSaving reconstruction data...")
    for z, K in reconstructions:
        intensity = abs_complex(K) ** 2
        intensity_min = intensity.min()
        intensity_max = intensity.max()
        intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min + 1e-10)
        intensity_8bit = (intensity_norm * 255).astype(np.uint8)
        cv2.imwrite(f'reconstruction_intensity_z{int(z)}.png', intensity_8bit)
        print(f"  ✓ Saved reconstruction at z={z}μm")

    print("\n" + "=" * 70)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_reconstruction()