/*
 * ipm_to_3d.cpp
 * -------------
 * Inverse Perspective Mapping (IPM) — Pixels → 3-D camera-frame coordinates
 *
 * Algorithm from:
 *   Thomas Fermi, "Algorithms for Automated Driving",
 *   https://thomasfermi.github.io/Algorithms-for-Automated-Driving/
 *   LaneDetection/InversePerspectiveMapping.html
 *
 * Camera assumptions
 * ------------------
 *   • Pitch  : 15° downward  (camera looks slightly toward the ground)
 *   • Height : h = 1.70 m above the flat road plane
 *   • No yaw, no roll
 *
 * The core formula (Eq. 8 from the page):
 *
 *          h
 *   λ  =  ──────────────────────────────
 *          n_c^T · K^{-1} · (u, v, 1)^T
 *
 *   (Xc, Yc, Zc)^T = λ · K^{-1} · (u, v, 1)^T
 *
 * where n_c = R_cr · (0,1,0)^T  is the road-plane normal in camera frame,
 * and R_cr is the rotation matrix that maps road-frame vectors into camera frame.
 *
 * Coordinate conventions (camera frame, standard computer-vision):
 *   Xc → right,  Yc → down,  Zc → forward (optical axis)
 *
 * Road-frame normal in road frame: n_r = (0,1,0)^T  (Y is up in road frame)
 * A pure pitch-down rotation of angle θ about the X-axis gives:
 *
 *         ⎡ 1    0        0    ⎤
 *  R_cr = ⎢ 0  cos(θ)  -sin(θ) ⎥   (θ = 15° = π/12 rad)
 *         ⎣ 0  sin(θ)   cos(θ) ⎦
 *
 * So n_c = R_cr · (0,1,0)^T = (0, cos θ, sin θ)^T
 *
 * Image reading: single-channel (greyscale) or RGB treated as greyscale.
 * Only WHITE pixels (value == 255 for all channels) are mapped.
 * (The image is described as "b/w pixels" — black background, white markings.)
 *
 * Build (PGI/NVHPC with OpenACC + stb_image header):
 *   nvc++ -O2 -acc -Minfo=accel -o ipm_to_3d ipm_to_3d.cpp -lm
 *
 * Build (host-only fallback, no GPU):
 *   g++ -O2 -o ipm_to_3d ipm_to_3d.cpp -lm
 *
 * Usage:
 *   ./ipm_to_3d dummy_lanes.png [output.txt]
 *
 * Output format (one line per white pixel):
 *   pixel_u  pixel_v  Xc  Yc  Zc
 *
 * Dependencies: stb_image (single-header, public domain)
 *   Download stb_image.h from https://github.com/nothings/stb
 *   and place it next to this file, OR embed it as shown below.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// ─── tuneable camera intrinsics ──────────────────────────────────────────────
// These are reasonable defaults for a 1920x1080 automotive camera.
// Adjust to match your actual calibration if available.
static constexpr double FOCAL_LENGTH_PX = 1080.0;   // f_x
static constexpr double FOCAL_LENGTH_PY = 1080.0;   // f_y
static constexpr double CX              = 960.0;     // principal point as fraction of width
static constexpr double CY              = 540.0;     // principal point as fraction of height

// ─── extrinsic parameters ────────────────────────────────────────────────────
static constexpr double CAMERA_HEIGHT_M = 1.70;    // h (metres)
static constexpr double PITCH_DEG       = 15.0;    // downward pitch (degrees)

// ─── white-pixel threshold ───────────────────────────────────────────────────
// A pixel is considered "white" (i.e. a lane marking) when every channel ≥ this.
static constexpr int WHITE_THRESHOLD = 200;

// ─── small helpers ────────────────────────────────────────────────────────────
static inline double deg2rad(double d) { return d * M_PI / 180.0; }

// ─── result record ────────────────────────────────────────────────────────────
struct Point3D {
    int    u, v;
    double Xc, Yc, Zc;
};

// =============================================================================
int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <image.png> [output.txt]\n"
            "  Reads a b/w image, applies IPM, writes 3-D camera-frame\n"
            "  coordinates of all white pixels to output.txt.\n",
            argv[0]);
        return 1;
    }

    const char* img_path = argv[1];
    const char* out_path = (argc >= 3) ? argv[2] : "output_3d.txt";

    // ── 1. Load image ─────────────────────────────────────────────────────────
    int img_w = 0, img_h = 0, channels = 0;
    unsigned char* raw = stbi_load(img_path, &img_w, &img_h, &channels, 0);
    if (!raw) {
        fprintf(stderr, "ERROR: could not load image '%s'\n", img_path);
        return 1;
    }
    printf("Loaded image: %d × %d, %d channel(s)\n", img_w, img_h, channels);

    const int N = img_w * img_h;  // total pixel count

    // ── 2. Build a flat boolean mask: white = 1, everything else = 0 ──────────
    // We treat a pixel as white when ALL channels exceed WHITE_THRESHOLD.
    std::vector<int> is_white(N, 0);

    // Build mask on host (I/O is not parallelisable on GPU anyway)
    for (int idx = 0; idx < N; ++idx) {
        bool white = true;
        for (int c = 0; c < channels; ++c) {
            if (raw[idx * channels + c] < WHITE_THRESHOLD) {
                white = false;
                break;
            }
        }
        is_white[idx] = white ? 1 : 0;
    }
    stbi_image_free(raw);

    // ── 3. Pre-compute camera parameters (scalars, stays on host/GPU constant) ─

    // Intrinsic matrix K:
    //   K = | fx   0   cx |
    //       |  0  fy   cy |
    //       |  0   0    1 |
    // with fx = fy (square pixels assumed).
    const double fx = FOCAL_LENGTH_PX;
    const double fy = FOCAL_LENGTH_PY;
    const double cx = CX;
    const double cy = CY;

    // K^{-1} (analytical inverse of the above 3×3):
    //   K_inv = | 1/fx    0   -cx/fx |
    //           |  0    1/fy  -cy/fy |
    //           |  0      0      1   |
    const double k_inv[3][3] = {
        { 1.0/fx,    0.0,   -cx/fx },
        {    0.0,  1.0/fy,  -cy/fy },
        {    0.0,    0.0,     1.0  }
    };

    // Road-plane normal in camera frame:
    //   n_c = R_cr · (0, 1, 0)^T
    // For a pure pitch-down rotation by θ about the camera's X-axis:
    //   R_cr = Rx(θ)  →  column 1 (Y column) = (0, cos θ, sin θ)^T
    // So n_c = (0, cos θ, sin θ)^T
    const double theta = deg2rad(PITCH_DEG);
    const double nc[3] = { 0.0, std::cos(theta), std::sin(theta) };

    const double h = CAMERA_HEIGHT_M;

    // Flatten K_inv into 9 doubles for easy OpenACC data clauses
    const double ki00 = k_inv[0][0], ki01 = k_inv[0][1], ki02 = k_inv[0][2];
    const double ki10 = k_inv[1][0], ki11 = k_inv[1][1], ki12 = k_inv[1][2];
    // Third row is (0, 0, 1) — hard-coded below for speed

    const double nc0 = nc[0], nc1 = nc[1], nc2 = nc[2];

    // ── 4. Allocate flat arrays for results ────────────────────────────────────
    // We compute (Xc, Yc, Zc) for EVERY pixel; non-white pixels get NaN.
    // This avoids any data-dependent control flow inside the parallel kernel.
    std::vector<double> Xc_arr(N, 0.0);
    std::vector<double> Yc_arr(N, 0.0);
    std::vector<double> Zc_arr(N, 0.0);

    // Raw pointers for OpenACC data movement
    const int*    iw = is_white.data();
    double* pXc = Xc_arr.data();
    double* pYc = Yc_arr.data();
    double* pZc = Zc_arr.data();

    // ── 5. Parallel IPM kernel ─────────────────────────────────────────────────
    //
    // Each thread handles one pixel independently — ideal for SIMD / GPU.
    //
    // The algorithm (Eq. 8):
    //   d = K^{-1} · (u, v, 1)^T          (direction vector)
    //   λ = h / (n_c^T · d)
    //   P = λ · d                          (3-D point in camera frame)
    //
    // Pixels whose ray is parallel to the road plane (n_c^T·d ≤ 0) are skipped.

    printf("Running IPM kernel over %d pixels (OpenACC parallel)...\n", N);

    #pragma acc data \
        copyin(iw[0:N]) \
        copyout(pXc[0:N], pYc[0:N], pZc[0:N])
    {
        #pragma acc parallel loop \
            gang vector \
            firstprivate(ki00,ki01,ki02, ki10,ki11,ki12, nc0,nc1,nc2, h, img_w)
        for (int idx = 0; idx < N; ++idx) {

            if (!iw[idx]) {
                // Non-white pixel: store a sentinel value
                pXc[idx] = 0.0;
                pYc[idx] = 0.0;
                pZc[idx] = 0.0;
                continue;
            }

            // Pixel coordinates from flat index
            const int u = idx % img_w;
            const int v = idx / img_w;

            // d = K^{-1} · (u, v, 1)^T
            double d0 = ki00 * u + ki01 * v + ki02;   // Xc direction
            double d1 = ki10 * u + ki11 * v + ki12;   // Yc direction
            double d2 = /* 0*u + 0*v + */ 1.0;        // Zc direction (= 1)

            // denominator: n_c^T · d
            double denom = nc0 * d0 + nc1 * d1 + nc2 * d2;

            if (denom <= 1e-9) {
                // Ray nearly parallel to road plane — no valid intersection
                pXc[idx] = 0.0;
                pYc[idx] = 0.0;
                pZc[idx] = 0.0;
                // Mark as invalid by zeroing is_white would race; use Zc < 0 flag instead
                pZc[idx] = -1.0;   // negative Z = behind camera → invalid
                continue;
            }

            double lambda = h / denom;

            pXc[idx] = lambda * d0;
            pYc[idx] = lambda * d1;
            pZc[idx] = lambda * d2;
        }
    }  // end #pragma acc data

    // ── 6. Collect valid results and write output ──────────────────────────────
    printf("Writing results to '%s'...\n", out_path);

    FILE* fout = fopen(out_path, "w");
    if (!fout) {
        fprintf(stderr, "ERROR: cannot open output file '%s'\n", out_path);
        return 1;
    }

    // Header
    fprintf(fout,
        "# Inverse Perspective Mapping — 3-D camera-frame coordinates\n"
        "# Camera: height=%.2f m, pitch=%.1f deg (down)\n"
        "# Intrinsics: fx=fy=%.1f px, cx=%.1f, cy=%.1f\n"
        "# Image: %d x %d\n"
        "# Format: pixel_u  pixel_v  Xc(m)  Yc(m)  Zc(m)\n"
        "#\n",
        h, PITCH_DEG, fx, cx, cy, img_w, img_h);

    long valid_count   = 0;
    long invalid_count = 0;

    for (int idx = 0; idx < N; ++idx) {
        if (!iw[idx]) continue;                    // non-white pixel
        if (pZc[idx] < 0.0) { ++invalid_count; continue; }  // bad geometry

        const int u = idx % img_w;
        const int v = idx / img_w;

        fprintf(fout, "%5d  %5d  %10.5f  %10.5f  %10.5f\n",
                u, v, pXc[idx], pYc[idx], pZc[idx]);
        ++valid_count;
    }

    fclose(fout);

    printf("Done.\n");
    printf("  White pixels found   : %ld\n", valid_count + invalid_count);
    printf("  Valid 3-D points     : %ld\n", valid_count);
    printf("  Skipped (bad ray)    : %ld\n", invalid_count);
    printf("  Output written to    : %s\n",  out_path);

    return 0;
}