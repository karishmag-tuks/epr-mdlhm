

import numpy as np
from numba import njit, prange

# -----------------------------
# Utilities: next power of two
# -----------------------------
@njit
def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p

# -----------------------------
# Contrast hologram (Sec. 1.3)
# -----------------------------
@njit
def make_contrast_hologram(I_raw, I_bg):
    # 𝑰̃(ξ) = I_raw(ξ) - I_bg(ξ)  (preferred implementation of Eq. (1.5))
    return I_raw - I_bg

# ------------------------------------------------------
# Bit-reversal permutation for radix-2 (iterative, 1-D)
# ------------------------------------------------------
@njit
def bit_reverse_indices(n):
    # n is power of two
    bits = 0
    t = n
    while t > 1:
        bits += 1
        t >>= 1
    out = np.empty(n, np.int64)
    for i in range(n):
        x = i
        y = 0
        for _ in range(bits):
            y = (y << 1) | (x & 1)
            x >>= 1
        out[i] = y
    return out

# ------------------------------------------------------
# 1-D in-place radix-2 Cooley–Tukey FFT (complex128)
# ------------------------------------------------------
@njit
def fft1d_inplace(a_real, a_imag, inverse=False):
    """
    In-place 1-D FFT on separate real/imag arrays (length N power of 2).
    Forward: X[k] = sum_n a[n] * exp(-j 2π kn / N)
    Inverse: x[n] = (1/N) * sum_k X[k] * exp(+j 2π kn / N)
    """
    n = a_real.shape[0]
    # Bit-reversal reordering
    rev = bit_reverse_indices(n)
    # scratch copies
    r = a_real.copy()
    im = a_imag.copy()
    for i in range(n):
        a_real[i] = r[rev[i]]
        a_imag[i] = im[rev[i]]

    # Iterative Danielson–Lanczos
    m = 2
    while m <= n:
        half = m // 2
        theta = (2.0 * np.pi / m) * (1.0 if inverse else -1.0)
        w_m_real = np.cos(theta)
        w_m_imag = np.sin(theta)
        for k in range(0, n, m):
            w_real = 1.0
            w_imag = 0.0
            for j in range(half):
                t_real = w_real * a_real[k + j + half] - w_imag * a_imag[k + j + half]
                t_imag = w_real * a_imag[k + j + half] + w_imag * a_real[k + j + half]
                u_real = a_real[k + j]
                u_imag = a_imag[k + j]
                a_real[k + j]       = u_real + t_real
                a_imag[k + j]       = u_imag + t_imag
                a_real[k + j + half] = u_real - t_real
                a_imag[k + j + half] = u_imag - t_imag
                # w *= w_m
                tmp = w_real * w_m_real - w_imag * w_m_imag
                w_imag = w_real * w_m_imag + w_imag * w_m_real
                w_real = tmp
        m <<= 1

    # Normalize inverse
    if inverse:
        inv_n = 1.0 / n
        for i in range(n):
            a_real[i] *= inv_n
            a_imag[i] *= inv_n

# ------------------------------------------------------
# 2-D FFT via separability (row-then-column)
# ------------------------------------------------------
@njit(parallel=True)
def fft2_radix2(data_real, data_imag, inverse=False):
    ny, nx = data_real.shape

    # Row-wise FFTs
    for y in prange(ny):
        fft1d_inplace(data_real[y], data_imag[y], inverse=inverse)

    # Column-wise FFTs
    # Work on columns by extracting views into temporary arrays
    col_r = np.empty(ny, dtype=np.float64)
    col_i = np.empty(ny, dtype=np.float64)
    for x in prange(nx):
        # copy column
        for y in range(ny):
            col_r[y] = data_real[y, x]
            col_i[y] = data_imag[y, x]
        fft1d_inplace(col_r, col_i, inverse=inverse)
        # write back
        for y in range(ny):
            data_real[y, x] = col_r[y]
            data_imag[y, x] = col_i[y]

# ------------------------------------------------------
# Build angular-spectrum transfer function H(fx,fy; z)
# ------------------------------------------------------
@njit
def build_transfer_kernel(nx, ny, pitch, wavelength, z):
    lam = wavelength
    k = 2.0 * np.pi / lam

    fx = np.empty(nx, dtype=np.float64)
    fy = np.empty(ny, dtype=np.float64)

    # manual fftfreq: [0,1,...,N/2-1, -N/2, ..., -1] / (N*dx)
    for i in range(nx):
        if i <= nx//2 - 1:
            fx[i] = i / (nx * pitch)
        else:
            fx[i] = (i - nx) / (nx * pitch)
    for j in range(ny):
        if j <= ny//2 - 1:
            fy[j] = j / (ny * pitch)
        else:
            fy[j] = (j - ny) / (ny * pitch)

    # H = exp(i k z sqrt(1 - (λ fx)^2 - (λ fy)^2)), clamp evanescent
    H_real = np.empty((ny, nx), dtype=np.float64)
    H_imag = np.empty((ny, nx), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            arg = 1.0 - (lam * fx[i])**2 - (lam * fy[j])**2
            if arg < 0.0:
                arg = 0.0  # discard evanescent for stability
            phase = k * z * np.sqrt(arg)
            H_real[j, i] = np.cos(phase)
            H_imag[j, i] = np.sin(phase)
    return H_real, H_imag

# ------------------------------------------------------
# Pointwise complex multiply: C = A * B
# ------------------------------------------------------
@njit(parallel=True)
def cmul2d(a_r, a_i, b_r, b_i, out_r, out_i):
    ny, nx = a_r.shape
    for j in prange(ny):
        for i in range(nx):
            out_r[j, i] = a_r[j, i] * b_r[j, i] - a_i[j, i] * b_i[j, i]
            out_i[j, i] = a_r[j, i] * b_i[j, i] + a_i[j, i] * b_r[j, i]

# ------------------------------------------------------
# Zero-pad to power-of-two canvas (Appendix recommends padding)
# ------------------------------------------------------
def pad_to_pow2_center(img):
    ny, nx = img.shape
    py, px = next_pow2(ny), next_pow2(nx)
    out = np.zeros((py, px), dtype=img.dtype)
    offy = (py - ny) // 2
    offx = (px - nx) // 2
    out[offy:offy+ny, offx:offx+nx] = img
    return out, offy, offx

def crop_center(img, ny, nx, offy, offx):
    return img[offy:offy+ny, offx:offx+nx]

# ------------------------------------------------------
# Three-FFT propagation using radix-2 FFTs
# ------------------------------------------------------
def propagate_fft_radix2(I_tilde, wavelength, pitch, z):
    """
    Uz = F^{-1}{ F{𝑰̃} * H }  (three FFTs total)
    Returns complex Uz (ny, nx) cropped to original size.
    """
    # Power-of-two padding (centered)
    I_pad, offy, offx = pad_to_pow2_center(I_tilde)
    py, px = I_pad.shape

    # Build transfer kernel on padded grid
    H_r, H_i = build_transfer_kernel(px, py, pitch, wavelength, z)

    # Forward FFT of 𝑰̃
    U_r = I_pad.copy()
    U_i = np.zeros_like(I_pad)
    fft2_radix2(U_r, U_i, inverse=False)

    # Multiply by H in frequency domain
    M_r = np.empty_like(U_r)
    M_i = np.empty_like(U_i)
    cmul2d(U_r, U_i, H_r, H_i, M_r, M_i)

    # Inverse FFT to get Uz
    fft2_radix2(M_r, M_i, inverse=True)

    # Crop back to original size
    Uz_r = crop_center(M_r, I_tilde.shape[0], I_tilde.shape[1], offy, offx)
    Uz_i = crop_center(M_i, I_tilde.shape[0], I_tilde.shape[1], offy, offx)

    return Uz_r + 1j * Uz_i

# ------------------------------------------------------
# End-to-end: multi-z Appendix reconstruction with radix-2 FFT
# ------------------------------------------------------
def reconstruct_dihm_fft_radix2(I_raw, I_bg, wavelength, pitch, z_list,
                                return_intensity=True):
    """
    1) 𝑰̃ = I_raw - I_bg   (contrast hologram, Sec. 1.3 Eq. (1.5))
    2) For each z: Uz = IFFT2( FFT2(𝑰̃) * H(fx,fy;z) )  (Appendix 3-FFT)
    """
    I_tilde = make_contrast_hologram(I_raw.astype(np.float64),
                                     I_bg.astype(np.float64))
    nz = len(z_list)
    ny, nx = I_tilde.shape
    if return_intensity:
        out = np.empty((nz, ny, nx), dtype=np.float64)
    else:
        out = np.empty((nz, ny, nx), dtype=np.complex128)

    for k in range(nz):
        Uz = propagate_fft_radix2(I_tilde, wavelength, pitch, z_list[k])
        if return_intensity:
            out[k] = (Uz.real * Uz.real + Uz.imag * Uz.imag)
        else:
            out[k] = Uz
    return out

# -----------------
# Example usage
# -----------------
if __name__ == "__main__":
    # Example parameters (meters)
    pitch = 1.12e-6     # detector pixel pitch
    lam   = 405e-9      # wavelength
    z0, z1, Nz = 0.6e-3, 1.0e-3, 41
    z_list = np.linspace(z0, z1, Nz)

    # Load your hologram/background (float64, shape ny x nx)
    # I_raw = np.load("holo.npy")
    # I_bg  = np.load("bg.npy")
    # For demonstration, use placeholders:
    ny, nx = 1536, 2048
    I_raw = np.random.rand(ny, nx).astype(np.float64)
    I_bg  = np.random.rand(ny, nx).astype(np.float64)

    Ivol = reconstruct_dihm_fft_radix2(I_raw, I_bg, lam, pitch, z_list,
                                       return_intensity=True)
    # Ivol[k] is intensity at z_list[k]; choose best z via your focus metric.
