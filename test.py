import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit, prange
import math
import time
import os
from matplotlib import gridspec

# Configuration
PRECISION = np.float32
USE_CUDA = True


# ==================== RECONSTRUCTION CODE ====================

@cuda.jit
def fast_reconstruction_kernel(hologram, wavelength, L, z, pixel_size, output_real, output_imag):
    """Ultra-fast CUDA kernel for approximate reconstruction"""
    i, j = cuda.grid(2)

    if i < hologram.shape[0] and j < hologram.shape[1]:
        # Fast coordinate calculation
        x = (i - 1024) * pixel_size
        y = (j - 1024) * pixel_size
        r2 = x * x + y * y

        # Ultra-fast phase approximation
        k = 6.283185307179586 / wavelength  # 2*pi/lambda
        R_approx = L + r2 / (2 * L)  # First-order Taylor
        phase = k * (z - R_approx)

        # Output
        output_real[i, j] = hologram[i, j] * math.cos(phase)
        output_imag[i, j] = hologram[i, j] * math.sin(phase)


@jit(nopython=True, parallel=True, fastmath=True)
def fast_reconstruction_cpu(hologram, wavelength, L, z_planes, pixel_size):
    """Ultra-fast CPU version"""
    h, w = hologram.shape
    num_planes = len(z_planes)
    results_real = np.zeros((num_planes, h, w), dtype=PRECISION)
    results_imag = np.zeros((num_planes, h, w), dtype=PRECISION)

    k = 6.283185307179586 / wavelength
    L2 = 2 * L

    for plane_idx in prange(num_planes):
        z = z_planes[plane_idx]
        for i in prange(h):
            x = (i - 1024) * pixel_size
            x2 = x * x
            for j in range(w):
                y = (j - 1024) * pixel_size
                r2 = x2 + y * y

                R_approx = L + r2 / L2
                phase = k * (z - R_approx)

                results_real[plane_idx, i, j] = hologram[i, j] * math.cos(phase)
                results_imag[plane_idx, i, j] = hologram[i, j] * math.sin(phase)

    return results_real, results_imag


def reconstruct_ultra_fast(hologram, wavelength, L, pixel_size, z_planes):
    """Main reconstruction function - target <20 seconds"""
    start_time = time.time()
    h, w = hologram.shape
    num_planes = len(z_planes)

    print(f"Starting reconstruction: {h}x{w}, {num_planes} planes...")

    if USE_CUDA and cuda.is_available():
        print("Using CUDA acceleration...")
        # CUDA path
        hologram_gpu = cuda.to_device(hologram.astype(PRECISION))

        # Configure CUDA
        threadsperblock = (16, 16)
        blockspergrid_x = (h + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (w + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        results_real = np.zeros((num_planes, h, w), dtype=PRECISION)
        results_imag = np.zeros((num_planes, h, w), dtype=PRECISION)

        for plane_idx, z in enumerate(z_planes):
            output_real_gpu = cuda.device_array((h, w), dtype=PRECISION)
            output_imag_gpu = cuda.device_array((h, w), dtype=PRECISION)

            fast_reconstruction_kernel[blockspergrid, threadsperblock](
                hologram_gpu, PRECISION(wavelength), PRECISION(L),
                PRECISION(z), PRECISION(pixel_size),
                output_real_gpu, output_imag_gpu
            )
            cuda.synchronize()

            results_real[plane_idx] = output_real_gpu.copy_to_host()
            results_imag[plane_idx] = output_imag_gpu.copy_to_host()

            elapsed = time.time() - start_time
            print(f"  Plane {plane_idx + 1}/{num_planes} - {elapsed:.1f}s")

            if elapsed > 18:  # Time budget
                print("Time budget exceeded")
                break

    else:
        print("Using CPU optimization...")
        results_real, results_imag = fast_reconstruction_cpu(hologram, wavelength, L, z_planes, pixel_size)

    # Calculate intensity
    intensity = np.sqrt(results_real ** 2 + results_imag ** 2)

    elapsed = time.time() - start_time
    print(f"Reconstruction completed in {elapsed:.2f} seconds")

    return intensity, results_real, results_imag


def generate_test_hologram(size=2048):
    """Generate realistic test hologram"""
    hologram = np.zeros((size, size), dtype=PRECISION)

    # Create interference pattern with multiple features
    for i in range(size):
        for j in range(size):
            x = (i - size / 2) / size
            y = (j - size / 2) / size

            # Multiple spatial frequencies
            pattern1 = 0.4 * math.cos(50 * (x * x + y * y))
            pattern2 = 0.3 * math.cos(20 * (x + y))
            pattern3 = 0.2 * math.sin(80 * math.sqrt(x * x + y * y))

            hologram[i, j] = 0.5 + pattern1 + pattern2 + pattern3

    return hologram


# ==================== DISPLAY CODE ====================

def plot_reconstruction_live(intensity, z_planes, hologram_idx=0, show_plots=True):
    """
    Live plotting of reconstruction results
    """
    if len(intensity) == 0:
        print("No results to plot")
        return None

    # Get the specific hologram results
    hologram_intensity = intensity[hologram_idx]

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'DIHM Reconstruction - Hologram {hologram_idx + 1} (2048×2048)',
                 fontsize=16, fontweight='bold')

    # Create grid layout
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Individual reconstruction planes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Additional visualizations
    ax4 = fig.add_subplot(gs[1, 0])  # Best focus
    ax5 = fig.add_subplot(gs[1, 1])  # Max projection
    ax6 = fig.add_subplot(gs[1, 2])  # Focus analysis

    # Get data ranges for consistent colormapping
    vmin = np.min(hologram_intensity)
    vmax = np.max(hologram_intensity)

    # Plot individual reconstruction planes
    axes = [ax1, ax2, ax3]
    for idx, (ax, z) in enumerate(zip(axes, z_planes)):
        # Display downsampled version for speed
        display_data = hologram_intensity[idx][::4, ::4]  # 4x downsampling for display

        im = ax.imshow(display_data, cmap='hot',
                       vmin=vmin, vmax=vmax,
                       extent=[0, display_data.shape[1], display_data.shape[0], 0])
        ax.set_title(f'z = {z * 1000:.1f} mm', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Find and plot best focus plane
    variances = [np.var(plane) for plane in hologram_intensity]
    best_plane_idx = np.argmax(variances)
    best_plane = hologram_intensity[best_plane_idx][::4, ::4]

    im4 = ax4.imshow(best_plane, cmap='viridis')
    ax4.set_title(f'Best Focus\n(z = {z_planes[best_plane_idx] * 1000:.1f} mm)',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Maximum intensity projection
    max_projection = np.max(hologram_intensity, axis=0)[::4, ::4]
    im5 = ax5.imshow(max_projection, cmap='plasma')
    ax5.set_title('Max Intensity Projection', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Focus quality analysis
    ax6.plot([z * 1000 for z in z_planes], variances, 'bo-', linewidth=3, markersize=8)
    ax6.plot(z_planes[best_plane_idx] * 1000, variances[best_plane_idx], 'ro',
             markersize=12, label='Best Focus')
    ax6.set_xlabel('Reconstruction Distance (mm)')
    ax6.set_ylabel('Variance (Focus Quality)')
    ax6.set_title('Focus Quality vs Distance')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if show_plots:
        plt.show()

    return best_plane_idx, fig


def plot_multiple_holograms_comparison(all_intensity, z_planes, show_plots=True):
    """Compare all three holograms"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Multiple Hologram Reconstruction Comparison (2048×2048)',
                 fontsize=16, fontweight='bold')

    # Find global min/max for consistent colormapping
    global_min = min([np.min(intensity) for intensity in all_intensity])
    global_max = max([np.max(intensity) for intensity in all_intensity])

    for hologram_idx in range(3):
        hologram_intensity = all_intensity[hologram_idx]

        # Plot individual planes (downsampled for display)
        for plane_idx in range(3):
            ax = axes[hologram_idx, plane_idx]
            display_data = hologram_intensity[plane_idx][::4, ::4]

            im = ax.imshow(display_data, cmap='hot', vmin=global_min, vmax=global_max)
            ax.set_title(f'z={z_planes[plane_idx] * 1000:.1f}mm')
            ax.set_xticks([])
            ax.set_yticks([])

            if plane_idx == 0:
                ax.set_ylabel(f'Holo {hologram_idx + 1}', rotation=90, size='large')

        # Plot best focus
        ax = axes[hologram_idx, 3]
        variances = [np.var(plane) for plane in hologram_intensity]
        best_plane_idx = np.argmax(variances)
        best_plane = hologram_intensity[best_plane_idx][::4, ::4]

        im = ax.imshow(best_plane, cmap='viridis')
        ax.set_title(f'Best Focus\n(z={z_planes[best_plane_idx] * 1000:.1f}mm)')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Intensity')

    if show_plots:
        plt.show()

    return fig


# ==================== MAIN PIPELINE ====================

def run_complete_pipeline():
    """
    Complete reconstruction and display pipeline
    """
    print("=" * 60)
    print("DIHM Reconstruction Pipeline - 2048×2048 Holograms")
    print("=" * 60)

    # Parameters
    wavelength = 408e-9
    L = 20e-3
    pixel_size = 6.45e-6
    z_planes = [0.5e-3, 1.0e-3, 1.5e-3]  # 3 reconstruction planes

    # Generate test holograms
    print("\n1. Generating test holograms...")
    holograms = []
    for i in range(3):
        print(f"   Generating hologram {i + 1}/3...")
        hologram = generate_test_hologram(2048)
        holograms.append(hologram)

    # Reconstruct all holograms
    print("\n2. Starting reconstruction...")
    all_intensity = []
    all_real = []
    all_imag = []

    total_start_time = time.time()

    for i, hologram in enumerate(holograms):
        print(f"\n   Reconstructing hologram {i + 1}/3:")
        intensity, real, imag = reconstruct_ultra_fast(hologram, wavelength, L, pixel_size, z_planes)
        all_intensity.append(intensity)
        all_real.append(real)
        all_imag.append(imag)

    total_time = time.time() - total_start_time
    print(f"\nTotal reconstruction time: {total_time:.2f} seconds")
    print(f"Average per hologram: {total_time / 3:.2f} seconds")

    # Display results
    print("\n3. Displaying results...")

    # Individual hologram plots
    best_focus_distances = []
    for i in range(3):
        print(f"   Plotting hologram {i + 1}...")
        best_plane_idx, fig = plot_reconstruction_live(all_intensity, z_planes, i, show_plots=False)
        best_focus_distances.append(z_planes[best_plane_idx])

        # Save individual plots
        plt.savefig(f'reconstruction_hologram_{i + 1}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Comparison plot
    print("   Creating comparison plot...")
    comp_fig = plot_multiple_holograms_comparison(all_intensity, z_planes, show_plots=False)
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(comp_fig)

    # Print summary
    print("\n" + "=" * 60)
    print("RECONSTRUCTION SUMMARY")
    print("=" * 60)
    for i, z_best in enumerate(best_focus_distances):
        print(f"Hologram {i + 1}: Best focus at z = {z_best * 1000:.1f} mm")

    print(f"\nResults saved to:")
    print("  - reconstruction_hologram_1.png")
    print("  - reconstruction_hologram_2.png")
    print("  - reconstruction_hologram_3.png")
    print("  - reconstruction_comparison.png")

    # Show one plot interactively
    print("\nDisplaying interactive plot for hologram 1...")
    plot_reconstruction_live(all_intensity, z_planes, 0, show_plots=True)

    return all_intensity, all_real, all_imag


# ==================== REAL-TIME MONITORING ====================

def monitor_reconstruction_progress():
    """Display progress during reconstruction"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Progress (%)')
    ax.set_title('Reconstruction Progress')
    ax.grid(True, alpha=0.3)

    return fig, ax


def update_progress(ax, progress, text):
    """Update progress bar"""
    ax.clear()
    ax.barh(0, progress, height=0.6, color='blue', alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Progress (%)')
    ax.set_title(text)
    ax.grid(True, alpha=0.3)
    ax.text(progress / 2, 0, f'{progress:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    plt.pause(0.01)


# ==================== QUICK SINGLE HOLOGRAM ====================

def quick_single_reconstruction():
    """Quick reconstruction and display for one hologram"""
    print("Quick Single Hologram Reconstruction")

    wavelength = 408e-9
    L = 20e-3
    pixel_size = 6.45e-6
    z_planes = [0.5e-3, 1.0e-3, 1.5e-3]

    # Generate hologram
    hologram = generate_test_hologram(2048)

    # Reconstruct
    start_time = time.time()
    intensity, real, imag = reconstruct_ultra_fast(hologram, wavelength, L, pixel_size, z_planes)
    recon_time = time.time() - start_time

    print(f"Reconstruction time: {recon_time:.2f} seconds")

    # Display
    best_plane_idx, fig = plot_reconstruction_live([intensity], z_planes, 0, show_plots=True)

    print(f"Best focus at z = {z_planes[best_plane_idx] * 1000:.1f} mm")

    return intensity, real, imag


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("DIHM 2048×2048 Reconstruction with Live Display")
    print("Choose mode:")
    print("1. Complete pipeline (3 holograms)")
    print("2. Quick single hologram")

    try:
        choice = input("Enter choice (1 or 2, default=2): ").strip()
    except:
        choice = "2"

    if choice == "1":
        # Run complete pipeline
        all_intensity, all_real, all_imag = run_complete_pipeline()
    else:
        # Quick single reconstruction
        intensity, real, imag = quick_single_reconstruction()