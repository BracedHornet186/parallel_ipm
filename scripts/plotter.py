import matplotlib.pyplot as plt

# --- Data Preparation: Lane Detect ---
threads = [1, 2, 4, 8, 16]
lane_pixel_speedup = [1.02, 1.65, 2.13, 2.45, 2.36]
lane_batch_speedup = [1.00, 1.98, 3.55, 5.11, 5.23]
lane_hybrid_speedup = [1.03, 1.93, 2.90, 3.68, 5.12]

# Calculate Efficiency (Speedup / Threads)
lane_pixel_eff = [s / t for s, t in zip(lane_pixel_speedup, threads)]
lane_batch_eff = [s / t for s, t in zip(lane_batch_speedup, threads)]
lane_hybrid_eff = [s / t for s, t in zip(lane_hybrid_speedup, threads)]

# --- Data Preparation: IPM 3D ---
ipm_omp_speedup = [1.02, 1.72, 2.33, 2.53, 2.53]
ipm_acc_speedup = [0.85, 1.30, 2.00, 2.31, 2.10]

# Calculate Efficiency (Speedup / Threads)
ipm_omp_eff = [s / t for s, t in zip(ipm_omp_speedup, threads)]
ipm_acc_eff = [s / t for s, t in zip(ipm_acc_speedup, threads)]

# Create Figure with 2x2 layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# --- Plot 1: Lane Detect Speedup ---
ax1.plot(threads, lane_pixel_speedup, marker='o', label='Pixel-based')
ax1.plot(threads, lane_batch_speedup, marker='s', label='Batch-based')
ax1.plot(threads, lane_hybrid_speedup, marker='^', label='Hybrid')
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Lane Detect: Speedup', fontweight='bold')
ax1.set_ylabel('Speedup (x)')
ax1.set_xticks(threads)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# --- Plot 2: IPM 3D Speedup ---
ax2.plot(threads, ipm_omp_speedup, marker='o', color='tab:red', label='OpenMP (CPU)')
ax2.plot(threads, ipm_acc_speedup, marker='D', color='tab:green', label='OpenACC (GPU)')
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('IPM 3D: Speedup', fontweight='bold')
ax2.set_ylabel('Speedup (x)')
ax2.set_xticks(threads)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

# --- Plot 3: Lane Detect Efficiency ---
ax3.plot(threads, lane_pixel_eff, marker='o', label='Pixel-based')
ax3.plot(threads, lane_batch_eff, marker='s', label='Batch-based')
ax3.plot(threads, lane_hybrid_eff, marker='^', label='Hybrid')
ax3.axhline(y=1, color='black', linestyle='-', alpha=0.3, label='Ideal (1.0)')
ax3.set_title('Lane Detect: Parallel Efficiency', fontweight='bold')
ax3.set_xlabel('Number of Threads')
ax3.set_ylabel('Efficiency (Speedup/Threads)')
ax3.set_xticks(threads)
ax3.set_ylim(0, 1.1)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend()

# --- Plot 4: IPM 3D Efficiency ---
ax4.plot(threads, ipm_omp_eff, marker='o', color='tab:red', label='OpenMP (CPU)')
ax4.plot(threads, ipm_acc_eff, marker='D', color='tab:green', label='OpenACC (GPU)')
ax4.axhline(y=1, color='black', linestyle='-', alpha=0.3, label='Ideal (1.0)')
ax4.set_title('IPM 3D: Parallel Efficiency', fontweight='bold')
ax4.set_xlabel('Number of Threads')
ax4.set_ylabel('Efficiency (Speedup/Threads)')
ax4.set_xticks(threads)
ax4.set_ylim(0, 1.1)
ax4.grid(True, linestyle=':', alpha=0.6)
ax4.legend()

plt.tight_layout()
plt.savefig('plots/performance_plots.png')
plt.show()