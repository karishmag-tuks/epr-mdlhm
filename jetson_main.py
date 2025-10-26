import time
import numpy as np
from numba import njit, jit, prange, config
import matplotlib.pyplot as plt


# ==============================================
# 1. FFT IMPLEMENTATION (radix-2 iterative)
# ==============================================

# config.DISABLE_JIT = True
@njit(cache=True, fastmath=True)
def bit_reverse(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


@njit(cache=True, fastmath=True)
def bit_reverse_permute_inplace(Ur, Ui):
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
    ny, nx = Ur.shape
    for y in prange(ny):
        fft1d_inplace(Ur[y, :], Ui[y, :], inverse)
    for x in prange(nx):
        colr = np.empty(ny, dtype=np.float64)
        coli = np.empty(ny, dtype=np.float64)
        for y in range(ny):
            colr[y] = Ur[y, x]
            coli[y] = Ui[y, x]
        fft1d_inplace(colr, coli, inverse)
        for y in range(ny):
            Ur[y, x] = colr[y]
            Ui[y, x] = coli[y]


# ==============================================
# 2. FRESNEL PROPAGATION
# ==============================================
@njit(cache=True, fastmath=True, parallel=True)
def apply_fresnel_kernel_inplace(Ur, Ui, wavelength, z, dx):
    ny, nx = Ur.shape
    inv_dx = 1.0 / dx
    lam = wavelength
    for y in prange(ny):
        fy = ((y if y < ny // 2 else y - ny) / ny) * inv_dx
        for x in range(nx):
            fx = ((x if x < nx // 2 else x - nx) / nx) * inv_dx
            phase = np.pi * lam * z * (fx * fx + fy * fy)
            c = np.cos(phase)
            s = np.sin(phase)
            ur = Ur[y, x]
            ui = Ui[y, x]
            Ur[y, x] = ur * c - ui * s
            Ui[y, x] = ur * s + ui * c


def fresnel_propagate_inplace(Ur, Ui, wavelength, z, dx):
    fft2d_inplace(Ur, Ui, inverse=False)
    apply_fresnel_kernel_inplace(Ur, Ui, wavelength, z, dx)
    fft2d_inplace(Ur, Ui, inverse=True)


# ==============================================
# 3. CDLHM PIPELINE (FRESNEL ONLY)
# ==============================================
def reference_field(ny, nx, wavelength, L):
    k = 2.0 * np.pi / wavelength
    c = np.cos(k * L)
    s = np.sin(k * L)
    Rr = np.full((ny, nx), c, dtype=np.float64)
    Ri = np.full((ny, nx), s, dtype=np.float64)
    return Rr, Ri


def cdlhm_channel_fresnel(obj_r, obj_i, wavelength, dx, L, z):
    ny, nx = obj_r.shape
    # Propagate object wave to sensor
    Ur = obj_r.copy()
    Ui = obj_i.copy()
    fresnel_propagate_inplace(Ur, Ui, wavelength, z=L - z, dx=dx)
    # Add reference field
    Rr, Ri = reference_field(ny, nx, wavelength, L)
    Ur += Rr
    Ui += Ri
    Hr, Hi = Ur.copy(), Ui.copy()
    # Back-propagate to object plane
    fresnel_propagate_inplace(Ur, Ui, wavelength, z=-(L - z), dx=dx)
    # Amplitude reconstruction
    return np.sqrt(Ur ** 2 + Ui ** 2), np.sqrt(Hr ** 2 + Ui ** 2)


def cdlhm_channel_fresnel_with_subtraction(obj_r, obj_i, wavelength, dx, L, z):
    """
    Fresnel CDLHM reconstruction with background subtraction:
      1) Forward propagate sample field to sensor (distance L - z)
      2) Form hologram with reference: I = |U_obj + U_ref|^2
      3) Subtract reference-only frame: I_tilde = I - |U_ref|^2
      4) Multiply by U_ref* (conjugate reference)
      5) Back-propagate by -(L - z) to object plane (Fresnel)
      6) Return amplitude

    Inputs:
      obj_r, obj_i : float64 arrays, real and imag parts of object-plane field
      wavelength   : meters
      dx           : pixel pitch (meters)
      L            : source-to-sensor distance (meters)
      z            : source-to-sample distance (meters)
    """
    ny, nx = obj_r.shape

    # 1) Propagate object wave to sensor (distance L - z)
    Ur = obj_r.copy()
    Ui = obj_i.copy()
    fresnel_propagate_inplace(Ur, Ui, wavelength, z=L - z, dx=dx)

    # 2) Add reference field across the sensor
    Rr, Ri = reference_field(ny, nx, wavelength, L)
    Ur_tot = Ur + Rr
    Ui_tot = Ui + Ri

    # 3) Hologram intensity and reference-only intensity (background)
    I = Ur_tot * Ur_tot + Ui_tot * Ui_tot  # |U_ref + U_obj|^2
    Iref = Rr * Rr + Ri * Ri  # |U_ref|^2
    I_til = I - Iref  # modified hologram  (Eq. (4))

    # 4) Form complex field for back-propagation: I_tilde * U_ref*
    #    U_ref* = Rr - i Ri
    F_r = I_til * Rr
    F_i = -I_til * Ri

    # 5) Back-propagate to object plane (distance -(L - z)) using Fresnel kernel
    fresnel_propagate_inplace(F_r, F_i, wavelength, z=-(L - z), dx=dx)

    # 6) Reconstructed amplitude
    A = np.sqrt(F_r * F_r + F_i * F_i)
    return A


# ==============================================
# 4. RUN RECONSTRUCTION (DEMO)
# ==============================================
ny = nx = 1024
dx = 1.12e-6  # pixel pitch
L = 68e-3  # source–sensor
z = 60e-3  # source–sample
radius = 30e-6
pixel_radius = int(round(radius / dx))
wavelengths = {'R': 633e-9, 'G': 532e-9, 'B': 405e-9}

# Simple object
obj_r = np.zeros((ny, nx), dtype=np.float64)
obj_i = np.zeros_like(obj_r)
cy, cx = ny // 2, nx // 2
y, x = np.ogrid[:ny, :nx]
mask = (x - cx) ** 2 + (y - cy) ** 2 <= pixel_radius ** 2
obj_r[mask] = 1.0

print("\n=== Fresnel CDLHM reconstruction ===")
start = time.perf_counter()
A_R, H_R = cdlhm_channel_fresnel(obj_r, obj_i, wavelengths['R'], dx, L, z)
A_G, H_G = cdlhm_channel_fresnel(obj_r, obj_i, wavelengths['G'], dx, L, z)
A_B, H_B = cdlhm_channel_fresnel(obj_r, obj_i, wavelengths['B'], dx, L, z)
end = time.perf_counter()
print(f"Completed in {end - start:.3f} s")


def norm_channel(C):
    cmin, cmax = float(C.min()), float(C.max())
    rng = cmax - cmin if cmax > cmin else 1.0
    return ((C - cmin) / rng * 255.0).astype(np.uint8)


Rb, Gb, Bb = norm_channel(A_R), norm_channel(A_G), norm_channel(A_B)
Rh, Gh, Bh = norm_channel(H_R), norm_channel(H_G), norm_channel(H_B)

plt.figure(0)
plt.imshow(obj_r)
plt.figure(2)
plt.imshow(Rh, cmap='gray')
plt.figure(3)
plt.imshow(Gh, cmap='gray')
plt.figure(4)
plt.imshow(Bh, cmap='gray')

plt.figure(5)
plt.imshow(Rb, cmap='gray')
plt.figure(6)
plt.imshow(Gb, cmap='gray')
plt.figure(7)
plt.imshow(Bb, cmap='gray')


plt.show()
