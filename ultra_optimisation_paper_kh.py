import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time
from numba import njit, prange
import math


# ============================================================================
# OPTIMIZED IN-PLACE FFT/IFFT IMPLEMENTATION
# ============================================================================

@njit(cache=True, fastmath=True)
def bit_reverse(i, bits):
    """Bit-reverse an integer with given number of bits"""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


@njit(cache=True, fastmath=True)
def bit_reverse_permute_inplace(Ur, Ui):
    """Bit-reverse permutation in-place for real and imaginary parts"""
    n = Ur.shape[0]
    bits = 0
    tmp = n
    while tmp > 1:
        tmp >>= 1
        bits += 1
    for i in range(n):
        j = bit_reverse(i, bits)
        if j > i:
            tr = Ur[i]
            ti = Ui[i]
            Ur[i] = Ur[j]
            Ui[i] = Ui[j]
            Ur[j] = tr
            Ui[j] = ti


@njit(cache=True, fastmath=True)
def fft1d_inplace(Ur, Ui, inverse):
    """1D FFT in-place with real and imaginary arrays"""
    n = Ur.shape[0]
    bit_reverse_permute_inplace(Ur, Ui)
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
                t_r = w_r * Ur[k + j + half_m] - w_i * Ui[k + j + half_m]
                t_i = w_r * Ui[k + j + half_m] + w_i * Ur[k + j + half_m]
                u_r = Ur[k + j]
                u_i = Ui[k + j]
                Ur[k + j] = u_r + t_r
                Ui[k + j] = u_i + t_i
                Ur[k + j + half_m] = u_r - t_r
                Ui[k + j + half_m] = u_i - t_i
                tmp_r = w_r * w_m_r - w_i * w_m_i
                tmp_i = w_r * w_m_i + w_i * w_m_r
                w_r = tmp_r
                w_i = tmp_i
        m <<= 1
    if inverse:
        inv_n = 1.0 / n
        for i in range(n):
            Ur[i] *= inv_n
            Ui[i] *= inv_n


@njit(cache=True, fastmath=True, parallel=True)
def fft2d_inplace(Ur, Ui, inverse=False):
    """2D FFT in-place with real and imaginary arrays - PARALLELIZED"""
    ny, nx = Ur.shape
    # FFT on rows - parallelized
    for y in prange(ny):
        fft1d_inplace(Ur[y, :], Ui[y, :], inverse)
    # FFT on columns - parallelized
    for x in prange(nx):
        colr = np.empty(ny, dtype=np.float32)
        coli = np.empty(ny, dtype=np.float32)
        for y in range(ny):
            colr[y] = Ur[y, x]
            coli[y] = Ui[y, x]
        fft1d_inplace(colr, coli, inverse)
        for y in range(ny):
            Ur[y, x] = colr[y]
            Ui[y, x] = coli[y]


# ============================================================================
# OPTIMIZED MATHEMATICAL FUNCTIONS (Parallelized & Pre-computed)
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def exp_array(x):
    """Compute exp for array using math.exp - PARALLELIZED"""
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
        (sy, sx) : top-left insert offsets
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
# COORDINATE TRANSFORMATION (Pure Functional)
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def coordinate_transform(X, Y, L):
    """Transform coordinates: X' = XL/R, Y' = YL/R"""
    R = sqrt_array(L ** 2 + X ** 2 + Y ** 2)

    X_prime = X * L / R
    Y_prime = Y * L / R
    R_prime = L ** 2 / R

    return X_prime, Y_prime, R_prime


@njit(fastmath=True, cache=True)
def prepare_coordinate_grids(M: int, N: int, pixel_size: float):
    coords_n = (np.arange(N, dtype=np.float32) - N // 2) * pixel_size
    coords_m = (np.arange(M, dtype=np.float32) - M // 2) * pixel_size

    X = np.zeros((M, N), dtype=np.float32)
    Y = np.zeros((M, N), dtype=np.float32)

    for i in range(M):
        X[i, :] = coords_n
    for j in range(N):
        Y[:, j] = coords_m[j]

    return X, Y


@njit()
def compute_reconstruction_params(N: int, M: int, wavelength: float, L: float,
                                  pixel_size: float) -> tuple:
    """Compute reconstruction parameters"""
    # Starting coordinates (center at origin)
    X0 = -(N // 2) * pixel_size
    Y0 = -(M // 2) * pixel_size
    # End coordinates
    X_max = X0 + (N - 1) * pixel_size
    Y_max = Y0 + (M - 1) * pixel_size

    # Apply equation (1.32): Non-uniform spacing in transformed coordinates

    Delta_x_prime = (L * X_max) / (N * math.sqrt(L ** 2 + X_max ** 2)) - (L * X0) / (N * math.sqrt(L ** 2 + X0 ** 2))
    Delta_y_prime = (L * Y_max) / (M * math.sqrt(L ** 2 + Y_max ** 2)) - (L * Y0) / (M * math.sqrt(L ** 2 + Y0 ** 2))

    # Reconstruction parameters for diffraction computation
    # These determine the spacing in the reconstruction plane
    delta_x = (wavelength * L) / (N * abs(Delta_x_prime)) if abs(Delta_x_prime) > 1e-10 else (wavelength * L) / N
    delta_y = (wavelength * L) / (M * abs(Delta_y_prime)) if abs(Delta_y_prime) > 1e-10 else (wavelength * L) / M

    return delta_x, delta_y, Delta_x_prime, Delta_y_prime


# ============================================================================
# VECTORIZED PHASE COMPUTATION
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def compute_a(I_pad, x_0, y_0, Delta_x_prime, Delta_y_prime, delta_x, delta_y, k_over_L):
    Py, Px = I_pad.shape

    phase_A = np.zeros((Py, Px), dtype=np.float32)

    # Pre-compute column-dependent terms
    col_term = np.zeros(Px, dtype=np.float32)
    for j in prange(Px):
        col_term[j] = j * x_0 * Delta_x_prime + j * j * delta_x * Delta_x_prime / 2.0

    # Pre-compute row-dependent terms (parallelized)
    row_term = np.zeros(Py, dtype=np.float32)
    for i in prange(Py):
        row_term[i] = i * y_0 * Delta_y_prime + i * i * delta_y * Delta_y_prime / 2.0

    # Combine
    for i in prange(Py):
        for j in range(Px):
            phase_A[i, j] = k_over_L * (col_term[j] + row_term[i])

    A = np.zeros_like(I_pad, dtype=np.complex64)
    for i in prange(Py):
        for j in range(Px):
            cos_p = math.cos(phase_A[i, j])
            sin_p = math.sin(phase_A[i, j])
            real = I_pad[i, j].real * cos_p - I_pad[i, j].imag * sin_p
            imag = I_pad[i, j].real * sin_p + I_pad[i, j].imag * cos_p
            A[i, j] = real + 1j * imag

    return A


@njit(fastmath=True, parallel=True, cache=True)
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
# HOLOGRAM PREPARATION (Pure Functional)
# ============================================================================
@njit(fastmath=True, parallel=True)
def prepare_transformed_hologram(hologram: np.ndarray, z: float, L: float, pixel_size: float, k: float):
    M, N = hologram.shape
    print(f"    Processing hologram {M}x{N}...")

    # Prepare coordinates
    X, Y = prepare_coordinate_grids(M, N, pixel_size)

    # Transform coordinates
    X_prime, Y_prime, R_prime = coordinate_transform(X, Y, L)

    M_center = M // 2
    N_center = N // 2
    X_prime_0 = X_prime[M_center, N_center]
    Y_prime_0 = Y_prime[M_center, N_center]

    # Compute phase factors
    jacobian_factor = (L / R_prime) ** 4

    # Compute phase
    phase_factor = np.zeros(R_prime.shape, dtype=np.complex64)
    phase_scale = k * z / L
    for i in prange(R_prime.shape[0]):
        for j in range(R_prime.shape[1]):
            phase = phase_scale * R_prime[i, j]
            phase_factor[i, j] = math.cos(phase) + 1j * math.sin(phase)

    I_prime = np.zeros((M, N), dtype=np.complex64)

    for i in prange(M):
        for j in range(N):
            # Use transformed indices to sample the hologram
            x_idx = int(X_prime[i, j])
            y_idx = int(Y_prime[i, j])
            # Bounds check
            if 0 <= x_idx < M and 0 <= y_idx < N:
                factor = hologram[x_idx, y_idx] * jacobian_factor[i, j]
                I_prime[i, j] = factor * (phase_factor[i, j])
    return I_prime, X_prime_0, Y_prime_0


@njit(fastmath=True, parallel=True)
def reconstruct(hologram: np.ndarray, z: float, wavelength: float, L: float,
                pixel_size: float) -> np.ndarray:
    """Main reconstruction using pure functional approach with optimized FFT"""

    M, N = hologram.shape

    # Pre-compute constants
    k = 2 * math.pi / wavelength
    k_over_L = k / L
    k_over_2L = k / (2 * L)
    I_prime, X_prime_0, Y_prime_0 = prepare_transformed_hologram(
        hologram, z, L, pixel_size, k)

    I_pad, (sy, sx), (Ny, Nx) = pad_to_pow2_rect(I_prime)
    Py, Px = I_pad.shape

    delta_x, delta_y, Delta_x_prime, Delta_y_prime = compute_reconstruction_params(N, M, wavelength, L, pixel_size)
    n = np.arange(Px, dtype=np.float32)
    m = np.arange(Py, dtype=np.float32)
    x_0 = -(Px // 2) * delta_x
    y_0 = -(Py // 2) * delta_y
    print(f" computing first FFT ")
    A = compute_a(I_pad, x_0, y_0, Delta_x_prime, Delta_y_prime, delta_x, delta_y, k_over_L)

    # Convert complex to separate real/imaginary for in-place FFT
    print(f"    Computing FFT2 ({Py}x{Px}) with optimized in-place algorithm...")
    A_r = np.ascontiguousarray(np.real(A).astype(np.float32))
    A_i = np.ascontiguousarray(np.imag(A).astype(np.float32))
    fft2d_inplace(A_r, A_i)

    jx = np.arange(Px, dtype=np.float32)
    jy = np.arange(Py, dtype=np.float32)

    print(f"    Computing convolution kernels...")
    phase_R_x = np.zeros(Px, dtype=np.float32)
    for i in prange(Px):
        phase_R_x[i] = -(k * jx[i] ** 2 * delta_x * Delta_x_prime) / (2 * L)

    R_x_r = np.zeros(Px, dtype=np.float32)
    R_x_i = np.zeros(Px, dtype=np.float32)
    for i in prange(Px):
        R_x_r[i] = math.cos(phase_R_x[i])
        R_x_i[i] = math.sin(phase_R_x[i])

    fft1d_inplace(R_x_r, R_x_i, inverse=False)

    phase_R_y = np.zeros(Py, dtype=np.float32)
    for i in prange(Py):
        phase_R_y[i] = -(k * jy[i] ** 2 * delta_y * Delta_y_prime) / (2 * L)

    R_y_r = np.zeros(Py, dtype=np.float32)
    R_y_i = np.zeros(Py, dtype=np.float32)
    for i in prange(Py):
        R_y_r[i] = math.cos(phase_R_y[i])
        R_y_i[i] = math.sin(phase_R_y[i])

    fft1d_inplace(R_y_r, R_y_i, inverse=False)

    print(f"    Convolving in frequency domain...")
    Product_r = np.zeros((Py, Px), dtype=np.float32)
    Product_i = np.zeros((Py, Px), dtype=np.float32)

    for i in prange(Py):
        for j in range(Px):
            # (A_r + i*A_i) * (R_x_r + i*R_x_i) * (R_y_r + i*R_y_i)
            temp_r = A_r[i, j] * R_x_r[j] - A_i[i, j] * R_x_i[j]
            temp_i = A_r[i, j] * R_x_i[j] + A_i[i, j] * R_x_r[j]

            Product_r[i, j] = temp_r * R_y_r[i] - temp_i * R_y_i[i]
            Product_i[i, j] = temp_r * R_y_i[i] + temp_i * R_y_r[i]

    fft2d_inplace(Product_r, Product_i, inverse=True)

    print(f"    Computing prefactors (vectorized)...")
    prefactor1 = Delta_x_prime * Delta_y_prime

    # Vectorized prefactor computation
    phase_prefactor2, phase_prefactor3 = compute_phase_prefactor_vectorized(
        Py, Px, n, m, x_0, y_0, delta_x, delta_y,
        X_prime_0, Y_prime_0, k_over_L, k_over_2L
    )

    K_pad_r = np.zeros((Py, Px), dtype=np.float32)
    K_pad_i = np.zeros((Py, Px), dtype=np.float32)

    for i in prange(Py):
        for j in range(Px):
            phase_total = phase_prefactor2[i, j] + phase_prefactor3[i, j]
            cos_p = math.cos(phase_total)
            sin_p = math.sin(phase_total)

            real = Product_r[i, j] * cos_p - Product_i[i, j] * sin_p
            imag = Product_r[i, j] * sin_p + Product_i[i, j] * cos_p

            K_pad_r[i, j] = real * prefactor1
            K_pad_i[i, j] = imag * prefactor1

    # Convert back to complex
    K_pad = K_pad_r + 1j * K_pad_i

    K = crop_rect(K_pad, (sy, sx), (Ny, Nx))

    return K


# ============================================================================
# MAIN DEMONSTRATION (Pure Functional)
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
        intensity = np.abs(K) ** 2
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
        phase = np.angle(K)
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
        intensity = np.abs(K) ** 2
        intensity_min = intensity.min()
        intensity_max = intensity.max()
        intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min + 1e-10)
        intensity_8bit = (intensity_norm * 255).astype(np.uint8)
        cv2.imwrite(f'reconstruction_intensity_z{int(z * 1e3)}.png', intensity_8bit)
        print(f"  ✓ Saved reconstruction at z={z}m")


if __name__ == "__main__":
    demonstrate_reconstruction()
