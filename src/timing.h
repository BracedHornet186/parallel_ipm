#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <cstdio>
#include <string>

class Timer {
public:
    Timer() { reset(); }
    void reset() { t0_ = std::chrono::steady_clock::now(); }
    double seconds() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

inline void print_timing(const std::string& label,
                         double seconds,
                         int threads,
                         const std::string& extra = "")
{
    std::fprintf(stderr,
                 "[TIME] %-20s threads=%d  %.4f s%s%s\n",
                 label.c_str(), threads, seconds,
                 extra.empty() ? "" : "  ",
                 extra.c_str());
}

#endif
