# Parallel Lane Detection + Inverse Perspective Mapping

**ID5130 — Parallel Scientific Computing — Term Project**

End-to-end monocular vision pipeline that converts a sequence of dash-cam
frames into 3-D point clouds of road lane markings, parallelized with
OpenMP and OpenACC and visualized as an animated point-cloud playback in
OpenGL.

| Stage | Binary | Backend |
|---|---|---|
| 1. Lane detection (raw JPG → binary edge mask) | `bin/lane_detect_*` | OpenMP |
| 2. Inverse Perspective Mapping (mask → 3-D coords) | `bin/ipm_3d_*` | OpenACC + OpenMP |
| 3. Animated 3-D viewer | `bin/viewer` | OpenGL / GLUT |

```
data/raw_data/*.jpg
   │
   ▼  (OpenMP per-pixel + per-image)
[lane_detect]  ───►  data/lane_masks/*.png        binary edge masks
                       │
                       ▼  (OpenMP outer + OpenACC inner)
                   [ipm_3d]  ───►  data/points_3d/*.txt   3-D camera-frame coords
                                     │
                                     ▼
                                  [viewer]  ───►  interactive multi-frame animation
```

## Repo layout

```
Project/
├── src/
│   ├── lane_detect.cpp     OpenMP lane detector (hand-rolled per-pixel kernels)
│   ├── ipm_3d.cpp          OpenACC IPM kernel + OpenMP outer batch loop
│   ├── 3d_viewer.cpp       OpenGL animated viewer
│   └── timing.h            Tiny chrono-based Timer used by all binaries
├── include/stb_image.h     Image-loader header (used by ipm_3d only)
├── data/
│   ├── raw_data/           248 JPG dash-cam frames (1920×1080)
│   ├── masked_images/      Reference Python output (Hough overlay)
│   ├── lane_masks/         (generated) C++ output: binary edge PNGs
│   └── points_3d/          (generated) per-frame 3-D coordinate TXTs
├── scripts/
│   ├── bench.sh            Speedup benchmark (runs every binary × thread sweep)
│   └── plot_bench.py       Renders matplotlib speedup / wall-time charts
├── plots/                  (generated) bench-result PNGs
├── Makefile                One-shot build for all six binaries
└── README.md
```

## Parallelization strategy

The project exposes parallelism along **two independent axes**:

* **per-pixel** — each kernel (HLS conversion, threshold, blur, Sobel,
  ROI test, IPM ray-cast) runs `H × W ≈ 2 M` independent ops per frame.
* **per-image** — the 248 frames are independent; they can be processed
  concurrently.

Different combinations of these two axes give four distinct parallel
modes, summarised below. The lane-detection binary supports the first
three; the IPM binary supports the OpenMP-batch and OpenACC modes.

| Mode | Outer (frames) | Inner (pixels) | Active levels | Used by |
|---|---|---|:-:|---|
| `pixel`  | sequential  | OpenMP, `N` threads     | 1 | `lane_detect_omp` |
| `batch`  | OpenMP, `N` threads | sequential           | 1 | `lane_detect_omp`, `ipm_3d_omp` |
| `hybrid` | OpenMP, `N_o` threads | OpenMP, `N_i` threads | 2 | `lane_detect_omp` |
| ACC      | OpenMP, `N` threads | OpenACC GPU (gang+vector) | 2 | `ipm_3d_acc` |

`N` is the total OpenMP thread budget (`--threads N`).
In hybrid mode `N_i = ⌈√N⌉`, `N_o = N / N_i`.

### `pixel` mode — pure per-pixel parallelism

Outer loop over frames is sequential; every per-pixel kernel inside one
frame is a flat `#pragma omp parallel for collapse(2)` over the
(row, col) grid. There are six such kernels per frame, so
fork–join overhead × 6 caps the achievable speedup at ~2.5×.

```cpp
#pragma omp parallel for num_threads(g_inner_threads) schedule(static)
for (int i = 0; i < n_pixels; ++i) { /* per-pixel work */ }
```

### `batch` mode — pure per-image parallelism

Outer loop over frames is parallelized; each thread owns a private
`ImageBuffers` scratch struct and processes one whole image end-to-end
with the inner kernels degenerating to sequential code (`omp_set_max_active_levels(1)`
disables nesting). Best speedup observed (5.2× at N=16) because there is
no inner synchronization and per-image working sets fit in private
caches.

```cpp
#pragma omp parallel num_threads(N) {
    ImageBuffers buf;                       // per-thread scratch
    #pragma omp for schedule(dynamic, 1)
    for (int i = 0; i < num_frames; ++i)
        process_one(frames[i], buf);
}
```

### `hybrid` mode — nested OpenMP

Both axes parallel. Nested OpenMP enabled (`omp_set_max_active_levels(2)`).
Thread budget is split as `N_o · N_i ≈ N`; outer team has `N_o` threads
each spawning inner teams of `N_i` threads on the per-pixel kernels.
Approaches batch performance only when `√N` is integer (e.g. N=16 →
4×4); at non-square N some threads sit idle due to integer rounding.

### OpenACC GPU offload (IPM only)

The IPM ray-cast is wrapped with both pragmas; `nvc++ -acc -mp` enables
both:

```cpp
#pragma omp parallel for schedule(dynamic, 1)        // outer: over images
for (int i = 0; i < num_frames; ++i) {
    #pragma acc parallel loop gang vector            // inner: over pixels (GPU)
    for (int idx = 0; idx < N_pixels; ++idx)
        /* compute (X_c, Y_c, Z_c) = λ · K^{-1} · (u,v,1)ᵀ */
}
```

The inner loop runs as a single GPU kernel of millions of gang/vector
workers; the outer loop multiplexes work across host CPU threads,
overlapping per-frame I/O with concurrent kernel launches and host-↔-device
transfers.

### Single-source serial baselines

Every parallel binary has a serial twin built from the same `.cpp` file
**without** `-fopenmp` / `-acc`. The pragmas silently fall through
(`-Wno-unknown-pragmas` suppresses the diagnostic), so the serial
binary contains zero OpenMP runtime — a fair baseline for speedup.

## Build

### Dependencies

| Tool | Purpose | Tested with |
|---|---|---|
| `g++` ≥ 9 | C++17, OpenMP | GCC 13.3 |
| `nvc++` (NVHPC) | OpenACC GPU build (optional) | NVHPC 25 |
| `pkg-config` + `libopencv-dev` | image I/O for `lane_detect` | OpenCV 4.6 |
| `freeglut3-dev`, `libgl1-mesa-dev` | viewer | — |
| Python ≥ 3.8 + `matplotlib`, `numpy`, `Pillow` | bench plots / comparison figs | Python 3.12 |

On Ubuntu/Debian:

```bash
sudo apt install build-essential libopencv-dev freeglut3-dev libglu1-mesa-dev
pip install matplotlib numpy pillow
```

### Make targets

```bash
make             # build all 5 CPU binaries (lane_detect_serial/omp, ipm_3d_serial/omp, viewer)
make ipm_3d_acc  # build the OpenACC GPU binary (requires nvc++)
make bench       # run scripts/bench.sh — full speedup sweep
make run         # end-to-end demo: lane_detect → ipm_3d → viewer
make clean
make help
```

The Makefile produces six binaries in total:

```
bin/lane_detect_serial   g++           — sequential baseline
bin/lane_detect_omp      g++ -fopenmp  — pixel/batch/hybrid modes
bin/ipm_3d_serial        g++           — sequential baseline
bin/ipm_3d_omp           g++ -fopenmp  — OpenMP batch over images
bin/ipm_3d_acc           nvc++ -acc -mp — OpenACC GPU + OpenMP batch
bin/viewer               g++ + GLUT    — animated point-cloud viewer
```

## Usage

### 1. Lane detection

```bash
# parallel build, three modes:
./bin/lane_detect_omp data/raw_data data/lane_masks --mode pixel  --threads 8 --time
./bin/lane_detect_omp data/raw_data data/lane_masks --mode batch  --threads 8 --time
./bin/lane_detect_omp data/raw_data data/lane_masks --mode hybrid --threads 8 --time

# serial baseline:
./bin/lane_detect_serial data/raw_data data/lane_masks --time

# limit to N frames for quick tests:
./bin/lane_detect_omp data/raw_data data/lane_masks --mode batch --threads 8 --limit 30
```

Outputs: one `frameNNNNNN.png` per input — a black image with white
edge pixels along the lane markings, suitable as input to `ipm_3d`.

### 2. IPM (mask → 3-D points)

```bash
# CPU OpenMP batch:
./bin/ipm_3d_omp --input-dir data/lane_masks --output-dir data/points_3d \
                 --threads 8 --time

# GPU OpenACC inner + OpenMP outer:
./bin/ipm_3d_acc --input-dir data/lane_masks --output-dir data/points_3d \
                 --threads 4 --time

# Sequential baseline:
./bin/ipm_3d_serial --input-dir data/lane_masks --output-dir data/points_3d --time

# Legacy single-image mode (backward compatible):
./bin/ipm_3d_omp <one_mask.png> <output.txt>
```

Each output TXT lists, one line per surviving white pixel:
```
pixel_u  pixel_v  Xc(m)  Yc(m)  Zc(m)
```

### 3. Animated 3-D viewer

```bash
./bin/viewer data/points_3d            # animate all frames
./bin/viewer data/points_3d/frame000050.txt   # single static frame
```

Controls:

| Key | Action |
|---|---|
| `Space` | play / pause |
| `←` `→` | step one frame back / forward (pauses) |
| `+` `-` | increase / decrease playback FPS (5 – 120) |
| `r` | reset to frame 0 |
| `0` | reset orbit + zoom |
| `w` `s` | zoom in / out (also mouse wheel) |
| Mouse drag | orbit |
| `Esc` | quit |

The HUD shows current frame, FPS, point count, play/pause state.

## Benchmarking & plots

```bash
make bench            # sweeps threads ∈ {1, 2, 4, 8, nproc} for every binary
LIMIT=100 make bench  # use 100 frames per measurement (default)
LIMIT=248 make bench  # full dataset
```

`scripts/bench.sh` parses each binary's `--time` line and prints a
side-by-side speedup table. Sample output (LIMIT=30):

```
[1/2] lane_detect
  threads    pixel(s)  speedup  batch(s)  speedup hybrid(s)  speedup
  1            2.1873    1.02x    2.2286    1.00x    2.1517    1.03x
  4            1.0414    2.13x    0.6266    3.55x    0.7675    2.90x
  8            0.9078    2.45x    0.4350    5.11x    0.6048    3.68x
  16           0.9410    2.36x    0.4248    5.23x    0.4339    5.12x

[2/2] ipm_3d
  threads       omp(s)    speedup     acc(s)    speedup
  4             0.6484    2.33x       0.7549    2.00x
  8             0.5990    2.53x       0.6548    2.31x
  16            0.5981    2.53x       0.7217    2.10x
```


## End-to-end one-liner

```bash
make            # build
make run        # lane_detect_omp → ipm_3d_omp → viewer
```

`make run` writes 248 edge masks to `data/lane_masks/`, 248 point-cloud
TXTs to `data/points_3d/`, then launches the animated viewer at 30 FPS.

## Acknowledgements

* Thomas Fermi, *Algorithms for Automated Driving — Inverse Perspective
  Mapping*, [thomasfermi.github.io](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/LaneDetection/InversePerspectiveMapping.html)
* OpenCV 4 — image I/O
* `stb_image.h` (public domain) by Sean Barrett
* OpenMP 5.2 and OpenACC 3.3 specifications

— Yash Purswani (ME22B214) · Praveen Joseph Thomas (ME22B180) ·
Department of Mechanical Engineering, IIT Madras
