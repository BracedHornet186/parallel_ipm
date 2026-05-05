#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "timing.h"

namespace fs = std::filesystem;

static const uint8_t LOWER_WHITE[3]  = { 50, 200,   0};
static const uint8_t UPPER_WHITE[3]  = {100, 255, 180};
static const uint8_t LOWER_YELLOW[3] = { 30, 150,  30};
static const uint8_t UPPER_YELLOW[3] = { 90, 200,  50};

static constexpr float SOBEL_THRESHOLD = 100.0f;

struct Pt { int x, y; };
static const Pt ROI_POLY[6] = {
    {   0, 980}, {   0, 600}, { 650, 120},
    {1400, 120}, {1920, 300}, {1920, 980}
};
static constexpr int ROI_POLY_N = 6;

static constexpr int   GBLUR_R = 3;
static constexpr int   GBLUR_K = 2*GBLUR_R + 1;
static constexpr float GBLUR_SIGMA = 1.4f;

static int g_inner_threads = 1;


struct ImageBuffers {
    int W = 0, H = 0;
    std::vector<uint8_t> hls, mask, gray, edge;
    std::vector<float>   tmp, blur;
    void resize(int w, int h) {
        if (W == w && H == h) return;
        W = w; H = h;
        const int n = w * h;
        hls.assign(n*3, 0);
        mask.assign(n, 0);
        gray.assign(n, 0);
        tmp.assign(n, 0.0f);
        blur.assign(n, 0.0f);
        edge.assign(n, 0);
    }
};

static void bgr_to_hls(const uint8_t* bgr, uint8_t* hls, int W, int H) {
    const int n = W * H;
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int i = 0; i < n; ++i) {
        const float b = bgr[i*3 + 0] / 255.0f;
        const float g = bgr[i*3 + 1] / 255.0f;
        const float r = bgr[i*3 + 2] / 255.0f;

        const float cmax = std::max(r, std::max(g, b));
        const float cmin = std::min(r, std::min(g, b));
        const float delta = cmax - cmin;

        const float L = 0.5f * (cmax + cmin);
        float S = 0.0f, Hh = 0.0f;
        if (delta > 1e-6f) {
            S = (L < 0.5f) ? delta / (cmax + cmin)
                           : delta / (2.0f - cmax - cmin);
            if      (cmax == r) Hh = std::fmod((g - b) / delta + 6.0f, 6.0f);
            else if (cmax == g) Hh = (b - r) / delta + 2.0f;
            else                Hh = (r - g) / delta + 4.0f;
            Hh *= 60.0f;  // degrees [0, 360)
        }

        const int H8 = (int)(Hh * 0.5f + 0.5f);
        const int L8 = (int)(L  * 255.0f + 0.5f);
        const int S8 = (int)(S  * 255.0f + 0.5f);
        hls[i*3 + 0] = (uint8_t)std::min(180, std::max(0, H8));
        hls[i*3 + 1] = (uint8_t)std::min(255, std::max(0, L8));
        hls[i*3 + 2] = (uint8_t)std::min(255, std::max(0, S8));
    }
}

static void color_threshold(const uint8_t* hls, uint8_t* mask, int W, int H) {
    const int n = W * H;
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int i = 0; i < n; ++i) {
        const uint8_t h = hls[i*3 + 0], l = hls[i*3 + 1], s = hls[i*3 + 2];
        const bool white  = (h >= LOWER_WHITE[0]  && h <= UPPER_WHITE[0])  &&
                            (l >= LOWER_WHITE[1]  && l <= UPPER_WHITE[1])  &&
                            (s >= LOWER_WHITE[2]  && s <= UPPER_WHITE[2]);
        const bool yellow = (h >= LOWER_YELLOW[0] && h <= UPPER_YELLOW[0]) &&
                            (l >= LOWER_YELLOW[1] && l <= UPPER_YELLOW[1]) &&
                            (s >= LOWER_YELLOW[2] && s <= UPPER_YELLOW[2]);
        mask[i] = (white | yellow) ? 255 : 0;
    }
}

static void masked_gray(const uint8_t* bgr, const uint8_t* mask,
                        uint8_t* gray, int W, int H)
{
    const int n = W * H;
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int i = 0; i < n; ++i) {
        if (mask[i] == 0) { gray[i] = 0; continue; }
        const int g = (int)(0.114f * bgr[i*3 + 0] +
                            0.587f * bgr[i*3 + 1] +
                            0.299f * bgr[i*3 + 2] + 0.5f);
        gray[i] = (uint8_t)std::min(255, std::max(0, g));
    }
}

static void make_gaussian_1d(float* k) {
    float sum = 0.0f;
    for (int i = -GBLUR_R; i <= GBLUR_R; ++i) {
        const float v = std::exp(-(i*i) / (2.0f * GBLUR_SIGMA * GBLUR_SIGMA));
        k[i + GBLUR_R] = v;
        sum += v;
    }
    for (int i = 0; i < GBLUR_K; ++i) k[i] /= sum;
}

static void gaussian_blur(const uint8_t* in, float* tmp, float* out,
                          int W, int H, const float* k)
{
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int t = -GBLUR_R; t <= GBLUR_R; ++t) {
                const int xx = std::min(W-1, std::max(0, x + t));
                s += k[t + GBLUR_R] * in[y*W + xx];
            }
            tmp[y*W + x] = s;
        }
    }
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int t = -GBLUR_R; t <= GBLUR_R; ++t) {
                const int yy = std::min(H-1, std::max(0, y + t));
                s += k[t + GBLUR_R] * tmp[yy*W + x];
            }
            out[y*W + x] = s;
        }
    }
}

static void sobel_edge(const float* in, uint8_t* edge, int W, int H,
                       float threshold)
{
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (x == 0 || y == 0 || x == W-1 || y == H-1) {
                edge[y*W + x] = 0;
                continue;
            }
            const float gx = -1.0f * in[(y-1)*W + (x-1)]
                           +  1.0f * in[(y-1)*W + (x+1)]
                           + -2.0f * in[(y  )*W + (x-1)]
                           +  2.0f * in[(y  )*W + (x+1)]
                           + -1.0f * in[(y+1)*W + (x-1)]
                           +  1.0f * in[(y+1)*W + (x+1)];
            const float gy = -1.0f * in[(y-1)*W + (x-1)]
                           + -2.0f * in[(y-1)*W + (x  )]
                           + -1.0f * in[(y-1)*W + (x+1)]
                           +  1.0f * in[(y+1)*W + (x-1)]
                           +  2.0f * in[(y+1)*W + (x  )]
                           +  1.0f * in[(y+1)*W + (x+1)];
            const float mag = std::sqrt(gx*gx + gy*gy);
            edge[y*W + x] = (mag > threshold) ? 255 : 0;
        }
    }
}

static std::vector<uint8_t> build_roi_mask(int W, int H) {
    std::vector<uint8_t> mask(W*H, 0);
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bool inside = false;
            for (int i = 0, j = ROI_POLY_N - 1; i < ROI_POLY_N; j = i++) {
                const Pt& A = ROI_POLY[i];
                const Pt& B = ROI_POLY[j];
                if (((A.y > y) != (B.y > y)) &&
                    ((double)x < (double)(B.x - A.x) * (y - A.y) /
                                  (double)(B.y - A.y) + A.x))
                    inside = !inside;
            }
            mask[y*W + x] = inside ? 255 : 0;
        }
    }
    return mask;
}

static void apply_roi(uint8_t* edge, const uint8_t* roi, int W, int H) {
    const int n = W * H;
    #pragma omp parallel for num_threads(g_inner_threads) schedule(static)
    for (int i = 0; i < n; ++i) {
        edge[i] = (roi[i] && edge[i]) ? 255 : 0;
    }
}

static void process_one(const cv::Mat& bgr,
                        const std::vector<uint8_t>& roi_mask,
                        const float* gauss_k,
                        ImageBuffers& buf,
                        cv::Mat& edge_out)
{
    const int W = bgr.cols, H = bgr.rows;
    buf.resize(W, H);
    bgr_to_hls(bgr.data, buf.hls.data(), W, H);
    color_threshold(buf.hls.data(), buf.mask.data(), W, H);
    masked_gray(bgr.data, buf.mask.data(), buf.gray.data(), W, H);
    gaussian_blur(buf.gray.data(), buf.tmp.data(), buf.blur.data(),
                  W, H, gauss_k);
    sobel_edge(buf.blur.data(), buf.edge.data(), W, H, SOBEL_THRESHOLD);
    apply_roi(buf.edge.data(), roi_mask.data(), W, H);
    edge_out = cv::Mat(H, W, CV_8UC1, buf.edge.data()).clone();
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <input_dir> <output_dir> [options]\n"
        "Options:\n"
        "  --mode pixel|batch|hybrid   parallelism level (parallel build)\n"
        "                pixel  : sequential outer, per-pixel parallel inner\n"
        "                batch  : per-image parallel outer, sequential inner\n"
        "                hybrid : both -- nested OpenMP, outer ~= sqrt(N),\n"
        "                         inner ~= sqrt(N)\n"
        "  --threads N           total OpenMP thread budget\n"
        "  --time                print summary timing line\n"
        "  --limit N             only process first N frames (debug)\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    const std::string in_dir  = argv[1];
    const std::string out_dir = argv[2];

    std::string mode = "pixel";
    int  n_threads   = 0;
    bool time_flag   = false;
    int  limit       = -1;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--mode"    && i+1 < argc) mode      = argv[++i];
        else if (a == "--threads" && i+1 < argc) n_threads = std::atoi(argv[++i]);
        else if (a == "--time")                  time_flag = true;
        else if (a == "--limit"   && i+1 < argc) limit     = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "Unknown arg: %s\n", a.c_str()); return 1; }
    }

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    n_threads = omp_get_max_threads();
    omp_set_max_active_levels(mode == "hybrid" ? 2 : 1);
#else
    n_threads = 1;
    mode = "serial";
#endif

    // Configure inner-team thread budget for the per-pixel kernels.
    int outer_threads = 1;       // outer team size for parallel image loop
    if      (mode == "pixel")  { g_inner_threads = n_threads; outer_threads = 1; }
    else if (mode == "batch")  { g_inner_threads = 1;         outer_threads = n_threads; }
    else if (mode == "hybrid") {
        // Split N total into outer * inner, balanced.
        int inner = std::max(1, (int)std::round(std::sqrt((double)n_threads)));
        int outer = std::max(1, n_threads / inner);
        g_inner_threads = inner;
        outer_threads   = outer;
    }
    else                       { g_inner_threads = 1;         outer_threads = 1; }

    fs::create_directories(out_dir);

    std::vector<fs::path> frames;
    for (auto& e : fs::directory_iterator(in_dir)) {
        if (!e.is_regular_file()) continue;
        const std::string ext = e.path().extension().string();
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
            frames.push_back(e.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    if (limit > 0 && (int)frames.size() > limit) frames.resize(limit);

    if (frames.empty()) {
        std::fprintf(stderr, "No images found in %s\n", in_dir.c_str());
        return 1;
    }

    cv::Mat first = cv::imread(frames[0].string(), cv::IMREAD_COLOR);
    if (first.empty()) {
        std::fprintf(stderr, "Could not read %s\n", frames[0].string().c_str());
        return 1;
    }
    const int W = first.cols, H = first.rows;

    float gauss_k[GBLUR_K];
    make_gaussian_1d(gauss_k);

    auto roi_mask = build_roi_mask(W, H);

    std::fprintf(stderr,
                 "lane_detect: %zu frames, %dx%d, mode=%s, threads=%d "
                 "(outer=%d, inner=%d)\n",
                 frames.size(), W, H, mode.c_str(), n_threads,
                 outer_threads, g_inner_threads);

    Timer total;

#ifdef _OPENMP
    if (mode == "batch" || mode == "hybrid") {
        #pragma omp parallel num_threads(outer_threads)
        {
            ImageBuffers buf;
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < (int)frames.size(); ++i) {
                cv::Mat bgr = cv::imread(frames[i].string(), cv::IMREAD_COLOR);
                if (bgr.empty()) continue;
                cv::Mat edge_out;
                process_one(bgr, roi_mask, gauss_k, buf, edge_out);
                fs::path op = fs::path(out_dir) /
                              (frames[i].stem().string() + ".png");
                cv::imwrite(op.string(), edge_out);
            }
        }
    } else
#endif
    {
        ImageBuffers buf;
        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat bgr = cv::imread(frames[i].string(), cv::IMREAD_COLOR);
            if (bgr.empty()) continue;
            cv::Mat edge_out;
            process_one(bgr, roi_mask, gauss_k, buf, edge_out);
            fs::path op = fs::path(out_dir) /
                          (frames[i].stem().string() + ".png");
            cv::imwrite(op.string(), edge_out);
        }
    }

    const double t = total.seconds();
    if (time_flag) {
        print_timing("lane_detect", t, n_threads,
                     "mode=" + mode +
                     " frames=" + std::to_string(frames.size()));
    }
    std::fprintf(stderr, "Done in %.3f s (%.1f ms/frame).\n",
                 t, 1000.0 * t / frames.size());

    return 0;
}
