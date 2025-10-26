# Retry the pure-Python FFT CDLHM demo with a smaller grid for reliability and speed.
import cmath
import math


def zeros(ny, nx, c=0 + 0j):
    return [[c for _ in range(nx)] for _ in range(ny)]


def transpose(A):
    ny, nx = len(A), len(A[0])
    T = [[0j] * ny for _ in range(nx)]
    for y in range(ny):
        for x in range(nx):
            T[x][y] = A[y][x]
    return T


def fft1d(x):
    N = len(x)
    if N == 1:
        return x[:]
    if N % 2 != 0:
        X = [0j] * N
        for k in range(N):
            s = 0j
            for n in range(N):
                s += x[n] * cmath.exp(-2j * math.pi * k * n / N)
            X[k] = s
        return X
    Xeven = fft1d(x[0::2])
    Xodd = fft1d(x[1::2])
    X = [0j] * N
    for k in range(N // 2):
        tw = cmath.exp(-2j * math.pi * k / N) * Xodd[k]
        X[k] = Xeven[k] + tw
        X[k + N // 2] = Xeven[k] - tw
    return X


def ifft1d(X):
    N = len(X)
    Xc = [z.conjugate() for z in X]
    xc = fft1d(Xc)
    return [z.conjugate() / N for z in xc]


def fft2d(a):
    ny, nx = len(a), len(a[0])
    rows_fft = [[0j] * nx for _ in range(ny)]
    for y in range(ny):
        rows_fft[y] = fft1d(a[y])
    rows_fft_T = transpose(rows_fft)
    cols_fft_T = [[0j] * ny for _ in range(nx)]
    for x in range(nx):
        cols_fft_T[x] = fft1d(rows_fft_T[x])
    return transpose(cols_fft_T)


def ifft2d(A):
    ny, nx = len(A), len(A[0])
    rows_ifft = [[0j] * nx for _ in range(ny)]
    for y in range(ny):
        rows_ifft[y] = ifft1d(A[y])
    rows_ifft_T = transpose(rows_ifft)
    cols_ifft_T = [[0j] * ny for _ in range(nx)]
    for x in range(nx):
        cols_ifft_T[x] = ifft1d(rows_ifft_T[x])
    return transpose(cols_ifft_T)


def freq_grids(nx, ny, dx):
    fx = [(k / nx) / dx if k < nx // 2 else ((k - nx) / nx) / dx for k in range(nx)]
    fy = [(k / ny) / dx if k < ny // 2 else ((k - ny) / ny) / dx for k in range(ny)]
    FX = [[0.0] * nx for _ in range(ny)]
    FY = [[0.0] * nx for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            FX[y][x] = fx[x]
            FY[y][x] = fy[y]
    return FX, FY


def propagator_AS(nx, ny, dx, wavelength, z):
    FX, FY = freq_grids(nx, ny, dx)
    H = [[0j] * nx for _ in range(ny)]
    inv_lam2 = 1.0 / (wavelength * wavelength)
    for y in range(ny):
        for x in range(nx):
            fx = FX[y][x]
            fy = FY[y][x]
            arg = inv_lam2 - fx * fx - fy * fy
            if arg <= 0:
                H[y][x] = 0 + 0j
            else:
                kz = 2 * math.pi * math.sqrt(arg)
                H[y][x] = cmath.exp(1j * z * kz)
    return H


def mul2d(A, B):
    ny, nx = len(A), len(A[0])
    C = [[0j] * nx for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            C[y][x] = A[y][x] * B[y][x]
    return C


def add2d(A, B):
    ny, nx = len(A), len(A[0])
    C = [[0j] * nx for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            C[y][x] = A[y][x] + B[y][x]
    return C


def sub2d(A, B):
    ny, nx = len(A), len(A[0])
    C = [[0j] * nx for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            C[y][x] = A[y][x] - B[y][x]
    return C


def scale2d(A, s):
    ny, nx = len(A), len(A[0])
    C = [[0j] * nx for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            C[y][x] = A[y][x] * s
    return C


def abs2d(A):
    ny, nx = len(A), len(A[0])
    B = [[0.0] * nx for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            v = A[y][x]
            B[y][x] = (v.real * v.real + v.imag * v.imag) ** 0.5
    return B


def propagate_AS(field, wavelength, z, dx):
    U1 = fft2d(field)
    H = propagator_AS(len(field[0]), len(field), dx, wavelength, z)
    U2 = mul2d(U1, H)
    return ifft2d(U2)


def to_ppm(path, R, G, B):
    ny, nx = len(R), len(R[0])

    def norm(A):
        m = min(min(row) for row in A)
        M = max(max(row) for row in A)
        rng = (M - m) if M > m else 1.0
        out = [[0] * nx for _ in range(ny)]
        for y in range(ny):
            for x in range(nx):
                v = (A[y][x] - m) / rng
                out[y][x] = max(0, min(255, int(round(v * 255))))
        return out

    Rb, Gb, Bb = norm(R), norm(G), norm(B)
    with open(path, 'wb') as f:
        f.write(f"P6 {nx} {ny} 255\n".encode('ascii'))
        for y in range(ny):
            row = bytearray()
            for x in range(nx):
                row += bytes([Rb[y][x], Gb[y][x], Bb[y][x]])
            f.write(row)


# Parameters
nx = ny = 64  # even smaller to ensure quick execution with pure Python FFT
dx = 1.12e-6
L = 16.8e-3
z = 8.0e-3
prop = L - z
wavelengths = {'R': 633e-9, 'G': 532e-9, 'B': 473e-9}

# Object
obj = [[0j] * nx for _ in range(ny)]
for (yy, xx) in [(ny // 2, nx // 2), (ny // 2 + 4, nx // 2 - 6), (ny // 2 - 8, nx // 2 + 3)]:
    obj[yy][xx] = 1.0 + 0j


def ref_wave(wl):
    k = 2 * math.pi / wl
    phase = cmath.exp(1j * k * L)
    return [[phase for _ in range(nx)] for _ in range(ny)]


recon_amp = {}
for ch, wl in wavelengths.items():
    U_sens = propagate_AS(obj, wl, prop, dx)
    U_ref = ref_wave(wl)
    I = [[0j] * nx for _ in range(ny)]
    Iref = [[0j] * nx for _ in range(ny)]
    for y in range(ny):
        for x in range(nx):
            u = U_sens[y][x] + U_ref[y][x]
            I[y][x] = (u.real * u.real + u.imag * u.imag) + 0j
            r = U_ref[y][x]
            Iref[y][x] = (r.real * r.real + r.imag * r.imag) + 0j
    Itilde = sub2d(I, Iref)
    k = 2 * math.pi / wl
    phase_conj = cmath.exp(-1j * k * L)
    ItildeUconj = [[Itilde[y][x] * phase_conj for x in range(nx)] for y in range(ny)]
    U_rec = propagate_AS(ItildeUconj, wl, -prop, dx)
    recon_amp[ch] = abs2d(U_rec)


