import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, config

config.DISABLE_JIT = True


@njit(fastmath=True, parallel=True)
def sub2d(A, B):
    ny, nx = len(A), len(A[0])
    C = np.zeros((ny, nx), dtype=np.float32)
    for y in prange(ny):
        for x in range(nx):
            C[y][x] = A[y][x] - B[y][x]
    return C


@njit(cache=True, fastmath=True, parallel=True)
def multiply_conjugate_reference(Fr, Fi, Rr, Ri):
    ny, nx = Fr.shape
    for y in prange(ny):
        for x in range(nx):
            temp_r = Fr[y, x]
            temp_i = Fi[y, x]
            Fr[y, x] = temp_r * Rr[y, x]
            Fi[y, x] = temp_i * Ri[y, x]


@njit(cache=True, fastmath=True, parallel=True)
def cmul2d(a_r, a_i, b_r, b_i):
    ny, nx = a_r.shape
    for j in prange(ny):
        for i in range(nx):
            temp_r = a_r[j, i] * b_r[j, i] - a_i[j, i] * b_i[j, i]
            temp_i = a_r[j, i] * b_i[j, i] + a_i[j, i] * b_r[j, i]
            a_r[j, i] = temp_r
            a_i[j, i] = temp_i


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


@njit(cache=True, fastmath=True, parallel=True)
def reference_field_spherical(ny, nx, wavelength, L, dx):
    Ur = np.empty((ny, nx), dtype=np.float64)
    Ui = np.empty((ny, nx), dtype=np.float64)
    k = 2.0 * np.pi / wavelength
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    for i in prange(ny):
        y = (i - cy) * dx
        for j in range(nx):
            x = (j - cx) * dx
            R = np.sqrt(x * x + y * y + L * L)
            phase = k * R
            c = np.cos(phase)
            s = np.sin(phase)
            Ur[i, j] = c
            Ui[i, j] = s
    return Ur, Ui


def reference_field(ny, nx, wavelength, L):
    k = 2.0 * np.pi / wavelength
    c = np.cos(k * L)
    s = np.sin(k * L)
    Rr = np.full((ny, nx), c, dtype=np.float64)
    Ri = np.full((ny, nx), s, dtype=np.float64)
    return Rr, Ri


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


@njit(cache=True, fastmath=True, parallel=True)
def build_transfer_kernel(nx, ny, pitch, wavelength, z):
    lamda = wavelength
    k = 2.0 * np.pi / lamda

    fx = np.empty(nx, dtype=np.float64)
    fy = np.empty(ny, dtype=np.float64)

    for i in prange(nx):
        if i <= nx // 2 - 1:
            fx[i] = i / (nx * pitch)
        else:
            fx[i] = (i - nx) / (nx * pitch)
    for j in prange(ny):
        if j <= ny // 2 - 1:
            fy[j] = j / (ny * pitch)
        else:
            fy[j] = (j - ny) / (ny * pitch)

    # H = exp(i k z sqrt(1 - (λ fx)^2 - (λ fy)^2))
    H_real = np.empty((ny, nx), dtype=np.float64)
    H_imag = np.empty((ny, nx), dtype=np.float64)
    for j in prange(ny):
        for i in range(nx):
            arg = 1.0 - (lamda * fx[i]) ** 2 - (lamda * fy[j]) ** 2
            if arg < 0.0:
                arg = 0.0  # discard evanescent for stability
            phase = k * z * np.sqrt(arg)
            H_real[j, i] = np.cos(phase)
            H_imag[j, i] = np.sin(phase)
    return H_real, H_imag


def helmholtz_propagate_inplace(I_tilde, px, py, wavelength, pitch, z):
    # Build transfer kernel on padded grid
    H_r, H_i = build_transfer_kernel(px, py, pitch, wavelength, z)
    # Forward FFT of 𝑰̃
    U_r = I_tilde.copy()
    U_i = np.zeros_like(I_tilde)
    fft2d_inplace(U_r, U_i, inverse=False)

    cmul2d(U_r, U_i, H_r, H_i)
    # Inverse FFT to get Uz
    fft2d_inplace(U_r, U_i, inverse=True)

    return U_r, U_i


def fresnel_reconstruction_function(obj_m, ref_m, wavelength, dx, L, z):
    ny, nx = obj_m.shape
    fft_ny, fft_nx = int(2 ** np.ceil(np.log2(ny))), int(2 ** np.ceil(np.log2(nx)))
    fft_center_y, fft_center_x = int((fft_ny - ny) // 2), int((fft_nx - nx) // 2)
    # subtracts wave intensities and store it in the object magnitude function
    I_modified = sub2d(obj_m, ref_m)
    # modified image phase
    Rr, Ri = reference_field_spherical(fft_ny, fft_nx, wavelength, L, dx)
    # Rr, Ri = reference_field(fft_ny, fft_nx, wavelength, L)

    Fr = np.zeros((fft_ny, fft_nx), dtype=np.float64)
    Fi = np.zeros((fft_ny, fft_nx), dtype=np.float64)
    Fr[fft_center_y:fft_ny - fft_center_y, fft_center_x:fft_nx - fft_center_x] = I_modified

    Fi[fft_center_y:fft_ny - fft_center_y, fft_center_x:fft_nx - fft_center_x] = I_modified

    # creates the term IUref used in equation (5)
    multiply_conjugate_reference(Fr, Fi, Rr, Ri)

    fresnel_propagate_inplace(Fr, Fi, wavelength, z=-(L - z), dx=dx)
    A = np.sqrt(Fr ** 2 + Fi ** 2)
    return A[fft_center_y:fft_ny - fft_center_y, fft_center_x:fft_nx - fft_center_x], I_modified


def helmoholtz_reconstruction_function(obj_m, ref_m, wavelength, dx, z):
    ny, nx = obj_m.shape
    fft_ny, fft_nx = int(2 ** np.ceil(np.log2(ny))), int(2 ** np.ceil(np.log2(nx)))
    fft_center_y, fft_center_x = int((fft_ny - ny) // 2), int((fft_nx - nx) // 2)
    # subtracts wave intensities and store it in the object magnitude function
    I_modified = sub2d(obj_m, ref_m)
    I_m = np.zeros((fft_ny, fft_nx), dtype=np.float64)
    I_m[fft_center_y:fft_ny - fft_center_y, fft_center_x:fft_nx - fft_center_x] = I_modified
    U_r, U_i = helmholtz_propagate_inplace(I_m, fft_nx, fft_ny, wavelength, dx, z)
    A = U_r ** 2 + U_i ** 2
    return A[fft_center_y:fft_ny - fft_center_y, fft_center_x:fft_nx - fft_center_x], I_modified


dx = 3.8e-6  # pixel pitch
# L = 68e-3  # source–sensor
# z = 60e-3  # source–sample

wavelengths = {'R': 654e-9, 'G': 510e-9, 'B': 405e-9}
L = {'R': 18.52e-3, 'G': 18.33e-3, "B": 18.96e-3}
z = {'R': 0.62e-3, 'G': 0.555e-3, "B": 0.58e-3}

r_image = cv2.imread('654_holo.jpg', cv2.IMREAD_GRAYSCALE)
r_image = r_image.astype(dtype=np.float64)
r_holo = cv2.imread('654_ref.jpg', cv2.IMREAD_GRAYSCALE)
r_holo = r_holo.astype(dtype=np.float64) / r_holo.max()
g_image = cv2.imread('510_holo.jpg', cv2.IMREAD_GRAYSCALE)
g_image = g_image.astype(dtype=np.float64) / g_image.max()
g_holo = cv2.imread('510_ref.jpg', cv2.IMREAD_GRAYSCALE)
g_holo = g_holo.astype(dtype=np.float64) / g_holo.max()
b_image = cv2.imread('405_holo.jpg', cv2.IMREAD_GRAYSCALE)
b_image = b_image.astype(dtype=np.float64) / b_image.max()
b_holo = cv2.imread('405_ref.jpg', cv2.IMREAD_GRAYSCALE)
b_holo = b_holo.astype(dtype=np.float64) / b_holo.max()
print("\n=== Fresnel CDLHM reconstruction ===")
start = time.perf_counter()
# A_R, I_R = fresnel_reconstruction_function(r_image, r_holo, wavelengths['R'], dx, L['R'], z['R'])
# A_G, I_G = fresnel_reconstruction_function(g_image, g_holo, wavelengths['G'], dx, L['G'], z['G'])
# A_B, I_B = fresnel_reconstruction_function(b_image, b_holo, wavelengths['B'], dx, L['B'], z['B'])

A_R, I_R = helmoholtz_reconstruction_function(r_image, r_holo, wavelengths['R'], dx, z['R'])
A_G, I_G = helmoholtz_reconstruction_function(g_image, g_holo, wavelengths['G'], dx, z['G'])
A_B, I_B = helmoholtz_reconstruction_function(b_image, b_holo, wavelengths['B'], dx, z['B'])
end = time.perf_counter()

print(f"Completed in {end - start:.3f} s")


def norm_channel(C):
    cmin, cmax = float(C.min()), float(C.max())
    rng = cmax - cmin
    return ((C - cmin) / rng * 255.0).astype(np.uint8)


r_reconstruct = norm_channel(A_R)
g_reconstruct = norm_channel(A_G)
b_reconstruct = norm_channel(A_B)

plt.figure(0)
plt.imshow(r_image, cmap='gray')
plt.figure(1)
plt.imshow(r_holo, cmap='gray')
plt.figure(2)
plt.imshow(I_R, cmap='gray')
plt.figure(3)
plt.imshow(r_reconstruct, cmap='gray')

plt.figure(4)
plt.imshow(g_image, cmap='gray')
plt.figure(5)
plt.imshow(g_holo, cmap='gray')
plt.figure(6)
plt.imshow(I_G, cmap='gray')
plt.figure(7)
plt.imshow(g_reconstruct, cmap='gray')

plt.figure(8)
plt.imshow(b_image, cmap='gray')
plt.figure(9)
plt.imshow(b_holo, cmap='gray')
plt.figure(10)
plt.imshow(I_B, cmap='gray')
plt.figure(11)
plt.imshow(b_reconstruct, cmap='gray')

plt.show()
