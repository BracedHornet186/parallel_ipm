#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "timing.h"

namespace fs = std::filesystem;

// ---- intrinsics (1920x1080 automotive camera) -------------------------------
static constexpr double FOCAL_LENGTH_PX = 1080.0;
static constexpr double FOCAL_LENGTH_PY = 1080.0;
static constexpr double CX              =  960.0;
static constexpr double CY              =  540.0;

// ---- extrinsics -------------------------------------------------------------
static constexpr double CAMERA_HEIGHT_M = 1.70;
static constexpr double PITCH_DEG       = 15.0;

// ---- white threshold --------------------------------------------------------
static constexpr int    WHITE_THRESHOLD = 200;

// --- output filtering --------------------------------------------------------
static constexpr double MAX_DISTANCE_M  = 200.0;

static inline double deg2rad(double d) { return d * M_PI / 180.0; }

static int process_image(const std::string& img_path,
                         const std::string& out_path)
{
    int img_w = 0, img_h = 0, channels = 0;
    unsigned char* raw = stbi_load(img_path.c_str(),
                                   &img_w, &img_h, &channels, 0);
    if (!raw) {
        std::fprintf(stderr, "  WARN: could not load %s\n", img_path.c_str());
        return 1;
    }

    const int N = img_w * img_h;

    // --- white-pixel mask ----------------------------------------------------
    std::vector<int> is_white(N, 0);
    for (int idx = 0; idx < N; ++idx) {
        bool white = true;
        for (int c = 0; c < channels; ++c) {
            if (raw[idx*channels + c] < WHITE_THRESHOLD) { white = false; break; }
        }
        is_white[idx] = white ? 1 : 0;
    }
    stbi_image_free(raw);

    // --- camera maths --------------------------------------------------------
    const double fx = FOCAL_LENGTH_PX, fy = FOCAL_LENGTH_PY;
    const double cx = CX,              cy = CY;

    const double ki00 = 1.0/fx,  ki01 = 0.0,    ki02 = -cx/fx;
    const double ki10 = 0.0,     ki11 = 1.0/fy, ki12 = -cy/fy;

    const double theta = deg2rad(PITCH_DEG);
    const double nc0 = 0.0, nc1 = std::cos(theta), nc2 = std::sin(theta);
    const double h   = CAMERA_HEIGHT_M;

    std::vector<double> Xc_arr(N, 0.0), Yc_arr(N, 0.0), Zc_arr(N, 0.0);

    const int*    iw  = is_white.data();
    double*       pXc = Xc_arr.data();
    double*       pYc = Yc_arr.data();
    double*       pZc = Zc_arr.data();

    #pragma acc data \
        copyin(iw[0:N]) \
        copyout(pXc[0:N], pYc[0:N], pZc[0:N])
    {
        #pragma acc parallel loop \
            gang vector \
            firstprivate(ki00,ki01,ki02, ki10,ki11,ki12, nc0,nc1,nc2, h, img_w)
        for (int idx = 0; idx < N; ++idx) {
            if (!iw[idx]) {
                pXc[idx] = 0.0; pYc[idx] = 0.0; pZc[idx] = 0.0;
                continue;
            }
            const int u = idx % img_w;
            const int v = idx / img_w;

            double d0 = ki00*u + ki01*v + ki02;
            double d1 = ki10*u + ki11*v + ki12;
            double d2 = 1.0;

            double denom = nc0*d0 + nc1*d1 + nc2*d2;
            if (denom <= 1e-9) {
                pXc[idx] = 0.0; pYc[idx] = 0.0; pZc[idx] = -1.0;
                continue;
            }
            double lambda = h / denom;
            pXc[idx] = lambda * d0;
            pYc[idx] = lambda * d1;
            pZc[idx] = lambda * d2;
        }
    }

    // --- write output --------------------------------------------------------
    FILE* fout = std::fopen(out_path.c_str(), "w");
    if (!fout) {
        std::fprintf(stderr, "  ERROR: cannot open %s for writing\n",
                     out_path.c_str());
        return 1;
    }
    std::fprintf(fout,
        "# Inverse Perspective Mapping -- 3-D camera-frame coordinates\n"
        "# Camera: height=%.2f m, pitch=%.1f deg (down)\n"
        "# Intrinsics: fx=fy=%.1f px, cx=%.1f, cy=%.1f\n"
        "# Image: %d x %d\n"
        "# Format: pixel_u  pixel_v  Xc(m)  Yc(m)  Zc(m)\n#\n",
        h, PITCH_DEG, fx, cx, cy, img_w, img_h);

    long valid = 0, invalid = 0;
    for (int idx = 0; idx < N; ++idx) {
        if (!iw[idx]) continue;
        if (pZc[idx] < 0.0) { ++invalid; continue; }
        if (pZc[idx] > MAX_DISTANCE_M) { ++invalid; continue; }
        const int u = idx % img_w;
        const int v = idx / img_w;
        std::fprintf(fout, "%5d  %5d  %10.5f  %10.5f  %10.5f\n",
                     u, v, pXc[idx], pYc[idx], pZc[idx]);
        ++valid;
    }
    std::fclose(fout);
    (void)valid; (void)invalid;
    return 0;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s <image.png> [output.txt]              (single-image legacy mode)\n"
        "  %s --input-dir DIR --output-dir DIR \\\n"
        "         [--threads N] [--time] [--limit N]   (batch mode)\n",
        prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // -------- batch mode --------
    bool batch = false;
    std::string in_dir, out_dir;
    int n_threads = 0;
    bool time_flag = false;
    int limit = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--input-dir"  && i+1 < argc) { in_dir  = argv[++i]; batch = true; }
        else if (a == "--output-dir" && i+1 < argc) { out_dir = argv[++i]; batch = true; }
        else if (a == "--threads"    && i+1 < argc) n_threads = std::atoi(argv[++i]);
        else if (a == "--time")                     time_flag = true;
        else if (a == "--limit"      && i+1 < argc) limit     = std::atoi(argv[++i]);
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "Unknown flag: %s\n", a.c_str());
            return 1;
        }
    }

    if (batch) {
        if (in_dir.empty() || out_dir.empty()) {
            std::fprintf(stderr,
                "Batch mode requires both --input-dir and --output-dir.\n");
            return 1;
        }

#ifdef _OPENMP
        if (n_threads > 0) omp_set_num_threads(n_threads);
        n_threads = omp_get_max_threads();
#else
        n_threads = 1;
#endif

        fs::create_directories(out_dir);

        std::vector<fs::path> frames;
        for (auto& e : fs::directory_iterator(in_dir)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                frames.push_back(e.path());
            }
        }
        std::sort(frames.begin(), frames.end());
        if (limit > 0 && (int)frames.size() > limit) frames.resize(limit);

        if (frames.empty()) {
            std::fprintf(stderr, "No images in %s\n", in_dir.c_str());
            return 1;
        }

        std::fprintf(stderr,
                     "ipm_3d batch: %zu frames, threads=%d\n",
                     frames.size(), n_threads);

        Timer total;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < (int)frames.size(); ++i) {
            const fs::path& p = frames[i];
            fs::path op = fs::path(out_dir) / (p.stem().string() + ".txt");
            process_image(p.string(), op.string());
        }

        const double t = total.seconds();
        if (time_flag) {
            print_timing("ipm_3d", t, n_threads,
                         "frames=" + std::to_string(frames.size()));
        }
        std::fprintf(stderr,
                     "Done in %.3f s (%.1f ms/frame).\n",
                     t, 1000.0 * t / frames.size());
        return 0;
    }

    // -------- single-image legacy mode --------
    const std::string img_path = argv[1];
    const std::string out_path = (argc >= 3 && argv[2][0] != '-')
                                  ? argv[2] : "output_3d.txt";
    Timer t0;
    int rc = process_image(img_path, out_path);
    if (time_flag) {
        print_timing("ipm_3d", t0.seconds(), 1, "single-image");
    }
    return rc;
}
