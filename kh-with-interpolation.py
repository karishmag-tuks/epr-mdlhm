import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time
from numba import njit, prange

"""
CORRECTED Kirchhoff-Helmholtz Reconstruction
Following the appendix equations EXACTLY
NO SCIPY DEPENDENCY - Custom bilinear interpolation
"""


# ============================================================================
# CUSTOM BILINEAR INTERPOLATION (No scipy)
# ============================================================================

@njit(fastmath=True, cache=True)
def bilinear_interpolate(image, x, y, x_grid, y_grid):
    """
    Bilinear interpolation at point (x, y)

    Parameters:
    -----------
    image : 2D array
    x, y : float - coordinates to interpolate at (in physical units)
    x_grid, y_grid : 1D arrays - the coordinate grids of the image

    Returns:
    --------
    float - interpolated value
    """
    # Find the grid cell containing (x, y)
    # x_grid and y_grid must be uniformly spaced

    if len(x_grid) < 2 or len(y_grid) < 2:
        return 0.0

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    # Convert physical coordinates to grid indices
    x_idx = (x - x_grid[0]) / dx
    y_idx = (y - y_grid[0]) / dy

    # Get integer parts (cell indices)
    i = int(np.floor(x_idx))
    j = int(np.floor(y_idx))

    # Check bounds
    if i < 0 or i >= len(x_grid) - 1 or j < 0 or j >= len(y_grid) - 1:
        return 0.0

    # Get fractional parts
    fx = x_idx - i
    fy = y_idx - j

    # Bilinear interpolation
    # Q11 = image[j, i], Q12 = image[j+1, i], Q21 = image[j, i+1], Q22 = image[j+1, i+1]
    Q11 = image[j, i]
    Q12 = image[j + 1, i]
    Q21 = image[j, i + 1]
    Q22 = image[j + 1, i + 1]

    # Interpolate in x direction
    R1 = Q11 * (1 - fx) + Q21 * fx
    R2 = Q12 * (1 - fx) + Q22 * fx

    # Interpolate in y direction
    P = R1 * (1 - fy) + R2 * fy

    return P


@njit(fastmath=True, parallel=True, cache=True)
def interpolate_2d_grid(image, X_target, Y_target, x_grid, y_grid):
    """
    Interpolate image onto a new 2D grid of target coordinates

    Parameters:
    -----------
    image : 2D array - original image
    X_target, Y_target : 2D arrays - target coordinates (physical units)
    x_grid, y_grid : 1D arrays - original coordinate grids

    Returns:
    --------
    2D array - interpolated image
    """
    M, N = X_target.shape
    result = np.zeros((M, N), dtype=np.float32)

    for i in prange(M):
        for j in range(N):
            result[i, j] = bilinear_interpolate(image, X_target[i, j], Y_target[i, j],
                                                x_grid, y_grid)

    return result


# ============================================================================
# STEP 1: COORDINATE TRANSFORMATION (Equations 1.23-1.27)
# ============================================================================

@njit(fastmath=True, cache=True)
def prepare_transformed_hologram_CORRECTED(hologram, z, L, pixel_size, wavelength):
    """
    Equation (1.27): I'(X', Y') = I(X, Y) * (L/R')^4 * exp(ikzR'/L)

    Key insight: We need to create an EQUIDISTANT grid in (X', Y') space,
    then interpolate the hologram at the corresponding (X, Y) points.
    """
    M, N = hologram.shape
    k = 2.0 * np.pi / wavelength

    # Original grid in (X, Y) - centered at origin
    X0 = -(N // 2) * pixel_size
    Y0 = -(M // 2) * pixel_size

    j = np.arange(N, dtype=np.float32)
    j_prime = np.arange(M, dtype=np.float32)

    X_grid = X0 + j * pixel_size
    Y_grid = Y0 + j_prime * pixel_size

    # Compute transformed grid spacing (Equation 1.32)
    X_max = X0 + (N - 1) * pixel_size
    Y_max = Y0 + (M - 1) * pixel_size

    X_prime_0 = X0 * L / np.sqrt(L ** 2 + X0 ** 2)
    Y_prime_0 = Y0 * L / np.sqrt(L ** 2 + Y0 ** 2)

    X_prime_max = X_max * L / np.sqrt(L ** 2 + X_max ** 2)
    Y_prime_max = Y_max * L / np.sqrt(L ** 2 + Y_max ** 2)

    Delta_x_prime = (X_prime_max - X_prime_0) / N
    Delta_y_prime = (Y_prime_max - Y_prime_0) / M

    # Create EQUIDISTANT grid in (X', Y') space
    X_prime_grid = X_prime_0 + j * Delta_x_prime
    Y_prime_grid = Y_prime_0 + j_prime * Delta_y_prime

    # Create 2D grids
    X_prime_2d = np.zeros((M, N), dtype=np.float32)
    Y_prime_2d = np.zeros((M, N), dtype=np.float32)

    for i in range(M):
        X_prime_2d[i, :] = X_prime_grid
    for j_idx in range(N):
        Y_prime_2d[:, j_idx] = Y_prime_grid[j_idx]

    # Compute R' from (X', Y') using equation (1.24)
    R_prime = np.sqrt(L ** 2 - X_prime_2d ** 2 - Y_prime_2d ** 2)

    # INVERSE TRANSFORM (Equation 1.24): Get (X, Y) from (X', Y')
    X_orig = X_prime_2d * L / R_prime
    Y_orig = Y_prime_2d * L / R_prime

    # Interpolate hologram at these (X, Y) points using custom interpolation
    I_interp = interpolate_2d_grid(hologram, X_orig, Y_orig, X_grid, Y_grid)

    # Apply Jacobian (L/R')^4 and phase factor (Equation 1.27)
    Jacobian = (L / R_prime) ** 4

    # Compute phase factor
    phase_factor = np.zeros((M, N), dtype=np.complex64)
    phase_arg = k * z * R_prime / L
    for i in range(M):
        for j_idx in range(N):
            phase_factor[i, j_idx] = np.cos(phase_arg[i, j_idx]) + 1j * np.sin(phase_arg[i, j_idx])

    I_prime = I_interp * Jacobian * phase_factor

    return I_prime, X_prime_0, Y_prime_0, Delta_x_prime, Delta_y_prime


# ============================================================================
# STEP 2: COMPUTE RECONSTRUCTION PARAMETERS (Equation 1.34)
# ============================================================================

@njit(cache=True)
def compute_reconstruction_params(N, M, wavelength, L, Delta_x_prime, Delta_y_prime):
    """
    Equation (1.34): Choose pixel size for reconstruction
    delta_x = λL / (N * Δx')
    """
    delta_x = wavelength * L / (N * abs(Delta_x_prime))
    delta_y = wavelength * L / (M * abs(Delta_y_prime))

    return delta_x, delta_y


# ============================================================================
# STEP 3: FIRST FFT (Equation 1.38)
# ============================================================================

@njit(fastmath=True, cache=True)
def compute_K_prime_fft(I_prime, N, M, x0, y0, Delta_x_prime, Delta_y_prime,
                        delta_x, delta_y, k, L):
    """
    Equation (1.38): K'_{νν'} = FFT[ I'_{jj'} * exp(ik(...)/L) ]

    The phase includes:
    - j*x0*Δx'
    - j'*y0*Δy'
    - j²*δx*Δx'/2
    - j'²*δy*Δy'/2
    """
    j = np.arange(N, dtype=np.float32)
    j_prime = np.arange(M, dtype=np.float32)

    # Compute phase array
    phase = np.zeros((M, N), dtype=np.float32)

    k_over_L = k / L

    for i in range(M):
        j_p = j_prime[i]
        y_term = j_p * y0 * Delta_y_prime + j_p * j_p * delta_y * Delta_y_prime / 2.0

        for j_idx in range(N):
            j_val = j[j_idx]
            x_term = j_val * x0 * Delta_x_prime + j_val * j_val * delta_x * Delta_x_prime / 2.0

            phase[i, j_idx] = k_over_L * (x_term + y_term)

    # Multiply I_prime by phase factor
    I_with_phase = np.zeros((M, N), dtype=np.complex64)
    for i in range(M):
        for j_idx in range(N):
            cos_p = np.cos(phase[i, j_idx])
            sin_p = np.sin(phase[i, j_idx])
            real = I_prime[i, j_idx].real * cos_p - I_prime[i, j_idx].imag * sin_p
            imag = I_prime[i, j_idx].real * sin_p + I_prime[i, j_idx].imag * cos_p
            I_with_phase[i, j_idx] = real + 1j * imag

    # First FFT (done in numpy since numba doesn't support fft directly)
    # We'll return the I_with_phase and do FFT outside
    return I_with_phase


# ============================================================================
# STEP 4: SECOND FFT - Chirp factors (Equation 1.39)
# ============================================================================

@njit(fastmath=True, cache=True)
def compute_chirp_factors(N, M, delta_x, delta_y, Delta_x_prime, Delta_y_prime, k, L):
    """
    Equation (1.39): R_ν = FFT[ exp(-ik*j²*δx*Δx'/(2L)) ]
    """
    j = np.arange(N, dtype=np.float32)
    j_prime = np.arange(M, dtype=np.float32)

    k_over_2L = k / (2.0 * L)

    # Chirp for x direction
    chirp_x = np.zeros(N, dtype=np.complex64)
    for j_idx in range(N):
        phase = -k_over_2L * j[j_idx] ** 2 * delta_x * Delta_x_prime
        chirp_x[j_idx] = np.cos(phase) + 1j * np.sin(phase)

    # Chirp for y direction
    chirp_y = np.zeros(M, dtype=np.complex64)
    for j_idx in range(M):
        phase = -k_over_2L * j_prime[j_idx] ** 2 * delta_y * Delta_y_prime
        chirp_y[j_idx] = np.cos(phase) + 1j * np.sin(phase)

    return chirp_x, chirp_y


# ============================================================================
# STEP 5: INVERSE FFT with phase postprocessing (Equations 1.40-1.41)
# ============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def compute_K_reconstruction(convolution_result, N, M, x0, y0,
                             X_prime_0, Y_prime_0, delta_x, delta_y,
                             Delta_x_prime, Delta_y_prime, k, L):
    """
    Equations (1.40-1.41):
    K_nm = Δx'*Δy' * exp(ik*phase1) * exp(ik*phase2) * IFFT[K'_{νν'} * R_ν * R_ν']

    phase1 = ((x0 + nδx)X'_0 + (y0 + mδy)Y'_0) / L
    phase2 = (n²δxΔx' + m²δyΔy') / (2L)
    """
    n = np.arange(N, dtype=np.float32)
    m = np.arange(M, dtype=np.float32)

    k_over_L = k / L
    k_over_2L = k / (2.0 * L)

    # Phase prefix (Equation 1.40)
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

            # Multiply complex numbers
            conv_real = convolution_result[i, j_idx].real
            conv_imag = convolution_result[i, j_idx].imag

            K[i, j_idx] = complex(
                phase_prefix_real * conv_real - phase_prefix_imag * conv_imag,
                phase_prefix_real * conv_imag + phase_prefix_imag * conv_real
            )

    return K


# ============================================================================
# MAIN RECONSTRUCTION FUNCTION
# ============================================================================

def reconstruct_KH(hologram, z, wavelength, L, pixel_size):
    """
    Complete Kirchhoff-Helmholtz reconstruction following the appendix.

    The 3 FFTs are:
    1. FFT of preprocessed hologram (eq 1.38)
    2. FFT of chirp factor for x (eq 1.39)
    3. FFT of chirp factor for y (eq 1.39)
    Plus 1 inverse FFT (eq 1.41)

    NO SCIPY DEPENDENCY - uses custom bilinear interpolation

    Parameters:
    -----------
    hologram : 2D array - contrast hologram
    z : float - reconstruction distance from pinhole
    wavelength : float - wavelength of light
    L : float - distance from pinhole to CCD
    pixel_size : float - CCD pixel size

    Returns:
    --------
    K : complex 2D array - reconstructed wavefront
    """
    M, N = hologram.shape
    k = 2.0 * np.pi / wavelength

    print(f"  Step 1: Coordinate transformation and interpolation...")
    # STEP 1: Prepare I'(X', Y') with coordinate transformation (Eq 1.27)
    I_prime, X_prime_0, Y_prime_0, Delta_x_prime, Delta_y_prime = \
        prepare_transformed_hologram_CORRECTED(hologram, z, L, pixel_size, wavelength)

    print(f"    Δx' = {Delta_x_prime:.6e} m")
    print(f"    Δy' = {Delta_y_prime:.6e} m")

    # STEP 2: Compute reconstruction pixel sizes (Eq 1.34)
    delta_x, delta_y = compute_reconstruction_params(
        N, M, wavelength, L, Delta_x_prime, Delta_y_prime
    )

    print(f"    δx = {delta_x:.6e} m (reconstruction pixel)")
    print(f"    δy = {delta_y:.6e} m (reconstruction pixel)")

    # Reconstruction grid origin
    x0 = -(N // 2) * delta_x
    y0 = -(M // 2) * delta_y

    print(f"  Step 2: First FFT (Equation 1.38)...")
    # STEP 3: First FFT with phase preprocessing (Eq 1.38)
    I_with_phase = compute_K_prime_fft(
        I_prime, N, M, x0, y0, Delta_x_prime, Delta_y_prime,
        delta_x, delta_y, k, L
    )

    # Perform FFT (numpy)
    K_prime = np.fft.fft2(I_with_phase)

    print(f"  Step 3: Computing chirp factors (Equation 1.39)...")
    # STEP 4: Second and third FFTs - chirp factors (Eq 1.39)
    chirp_x, chirp_y = compute_chirp_factors(
        N, M, delta_x, delta_y, Delta_x_prime, Delta_y_prime, k, L
    )

    # Perform FFTs of chirps
    R_x = np.fft.fft(chirp_x)
    R_y = np.fft.fft(chirp_y)

    print(f"  Step 4: Convolution in Fourier domain...")
    # Multiply in Fourier space (convolution in spatial domain)
    convolution_freq = K_prime * R_x[np.newaxis, :] * R_y[:, np.newaxis]

    print(f"  Step 5: Inverse FFT with phase postprocessing (Equations 1.40-1.41)...")
    # Inverse FFT
    convolution_result = np.fft.ifft2(convolution_freq)

    # STEP 5: Apply phase postprocessing (Eq 1.40-1.41)
    K = compute_K_reconstruction(
        convolution_result, N, M, x0, y0,
        X_prime_0, Y_prime_0, delta_x, delta_y,
        Delta_x_prime, Delta_y_prime, k, L
    )

    return K


# ============================================================================
# DEMO
# ============================================================================

def demo_reconstruction():
    """Demonstrate the corrected reconstruction"""

    print("=" * 70)
    print("CORRECTED Kirchhoff-Helmholtz Reconstruction")
    print("Following Appendix Equations Exactly")
    print("=" * 70)

    # Load images
    object_path = "654_holo.jpg"
    reference_path = "654_ref.jpg"

    if not Path(object_path).exists():
        print(f"\nCreating synthetic test hologram...")
        # Create simple test pattern
        N = 512
        hologram = np.random.randn(N, N).astype(np.float32) * 0.01

        # Add some features
        y, x = np.ogrid[-N // 2:N // 2, -N // 2:N // 2]
        r = np.sqrt(x ** 2 + y ** 2)
        hologram += 0.1 * np.cos(r * 0.1)

    else:
        print(f"\nLoading images...")
        obj_img = cv2.imread(object_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

        # Crop to manageable size
        crop_size = 4096
        h, w = obj_img.shape
        start = (w - crop_size) // 2
        obj_img = obj_img[:crop_size, start:start + crop_size]
        ref_img = ref_img[:crop_size, start:start + crop_size]

        # Compute contrast hologram
        hologram = (obj_img.astype(np.float32) - ref_img.astype(np.float32)) / 255.0

    # Parameters
    wavelength = 654e-9  # m
    L = 18.52e-3  # m
    pixel_size = 3.8e-6  # m
    z = 6.2e-3  # m

    print(f"\nParameters:")
    print(f"  Wavelength: {wavelength * 1e9:.1f} nm")
    print(f"  Pinhole-CCD distance L: {L * 1e3:.2f} mm")
    print(f"  Reconstruction distance z: {z * 1e3:.2f} mm")
    print(f"  CCD pixel size: {pixel_size * 1e6:.2f} μm")
    print(f"  Hologram size: {hologram.shape}")

    print(f"\n{'=' * 70}")
    print("RECONSTRUCTION")
    print(f"{'=' * 70}")

    t0 = time.time()
    K = reconstruct_KH(hologram, z, wavelength, L, pixel_size)
    t_elapsed = time.time() - t0

    print(f"\n  ✓ Total reconstruction time: {t_elapsed:.3f} seconds")

    # Compute intensity and phase
    intensity = np.abs(K) ** 2
    phase = np.angle(K)

    print(f"\nResults:")
    print(f"  Intensity range: [{intensity.min():.2e}, {intensity.max():.2e}]")
    print(f"  Phase range: [{phase.min():.2f}, {phase.max():.2f}]")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Hologram
    im0 = axes[0, 0].imshow(hologram, cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[0, 0].set_title('Contrast Hologram')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Intensity (linear)
    im1 = axes[0, 1].imshow(intensity, cmap='hot')
    axes[0, 1].set_title(f'Reconstructed Intensity at z={z * 1e3:.2f}mm')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Intensity (log)
    intensity_log = np.log10(intensity + 1e-10)
    im2 = axes[1, 0].imshow(intensity_log, cmap='hot')
    axes[1, 0].set_title('Intensity (log scale)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # Phase
    im3 = axes[1, 1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Reconstructed Phase')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    output_file = 'corrected_reconstruction_no_scipy.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved visualization to '{output_file}'")
    plt.show()


if __name__ == "__main__":
    demo_reconstruction()