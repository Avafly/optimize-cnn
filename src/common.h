#ifndef COMMON_H_
#define COMMON_H_

#include "3rdparty/fmt/base.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

// root of the exported data/ directory
#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

namespace optcnn
{

inline constexpr int IMG_C = 1;
inline constexpr int IMG_H = 28;
inline constexpr int IMG_W = 28;
inline constexpr int IMG_SIZE = IMG_C * IMG_H * IMG_W;
inline constexpr int NUM_CLASSES = 10;
inline constexpr int NUM_IMAGES = 70000;
inline constexpr int BATCH = 32;
inline constexpr int ALIGN = 64;
inline constexpr int MAX_THREADS = 8;
inline constexpr int DEFAULT_THREADS = 1;
inline constexpr int DEFAULT_RUNS = 1;

struct Args
{
    std::string data_dir = DATA_DIR;
    int threads = DEFAULT_THREADS;
    int runs = DEFAULT_RUNS;

    std::string path(const std::string &rel) const
    {
        return data_dir + "/" + rel;
    }

    std::string images_path() const
    {
        return path("images.bin");
    }

    std::string ref_path() const
    {
        return path("ref_logits.bin");
    }
};

struct Timing
{
    double best, mean, worst;
};

template <typename T>
void load_data(const std::string &path, T *buf, std::size_t size)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        fmt::print(stderr, "error: cannot open {}\n", path);
        std::exit(1);
    }
    const std::streamsize bytes = f.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) % sizeof(T) != 0)
    {
        fmt::print(stderr, "error: {} size is not a multiple of {} bytes\n", path, sizeof(T));
        std::exit(1);
    }

    if (static_cast<std::size_t>(bytes) != size)
    {
        fmt::print(stderr, "error: {} is {} bytes, expected {}\n", path, bytes, size);
        std::exit(1);
    }

    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(buf), bytes))
    {
        fmt::print(stderr, "error: failed to read {}\n", path);
        std::exit(1);
    }
}

inline void print_usage(const char *exe)
{
    fmt::print(stderr,
               "Usage: {0} [data_dir] [--threads T] [--runs N]\n"
               "  data_dir   dir with images.bin/ref_logits.bin (default: {1})\n"
               "  --threads  worker threads (default: {2})\n"
               "  --runs     timed runs, best/mean/worst reported (default: {3})\n"
               "\n"
               "Examples:\n"
               "  {0} --threads 2\n"
               "  {0} /path/to/data --threads 4 --runs 30\n",
               exe, DATA_DIR, DEFAULT_THREADS, DEFAULT_RUNS);
}

inline Args parse_args(int argc, char **argv)
{
    Args a;
    if (argc == 1)
    {
        print_usage(argv[0]);
        std::exit(0);
    }
    bool got_dir = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string s = argv[i];
        if (s == "-h" || s == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (s == "--threads" && i + 1 < argc)
            a.threads = std::atoi(argv[++i]);
        else if (s == "--runs" && i + 1 < argc)
            a.runs = std::atoi(argv[++i]);
        else if (!got_dir && !s.empty() && s[0] != '-')
        {
            a.data_dir = s;
            got_dir = true;
        }
        else
        {
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    if (a.threads < 1)
        a.threads = 1;
    if (a.runs < 1)
        a.runs = 1;
    return a;
}

inline int argmax(const float *row)
{
    int best = 0;
    for (int j = 1; j < NUM_CLASSES; ++j)
        if (row[j] > row[best])
            best = j;
    return best;
}

inline bool verify(const float *logits, const float *ref, const int count, const float tol)
{
    fmt::print("------\n");

    int agree = 0;
    double max_diff = 0.0, sum_diff = 0.0;
    std::vector<std::pair<float, int>> miss;

    for (int i = 0; i < count; ++i)
    {
        const float *a = logits + static_cast<std::size_t>(i) * NUM_CLASSES;
        const float *r = ref + static_cast<std::size_t>(i) * NUM_CLASSES;

        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            double d = static_cast<double>(a[j]) - r[j];
            d = std::abs(d);
            max_diff = std::max(d, max_diff);
            sum_diff += d;
        }

        const int pr = argmax(r);
        if (argmax(a) == pr)
        {
            ++agree;
            continue;
        }
        float top2 = -1e30f;
        for (int j = 0; j < NUM_CLASSES; ++j)
            if (j != pr && r[j] > top2)
                top2 = r[j];
        miss.emplace_back(r[pr] - top2, i);
    }

    const double mean_diff = sum_diff / (static_cast<double>(count) * NUM_CLASSES);
    fmt::print("verify: argmax {}/{} ({:.4f})  max|d|={:.2e}  mean|d|={:.2e}\n", agree, count,
               static_cast<double>(agree) / count, max_diff, mean_diff);

    if (!miss.empty())
    {
        std::sort(miss.begin(), miss.end(),
                  [](const auto &x, const auto &y) { return x.first > y.first; });
        const int show = miss.size() < 10 ? static_cast<int>(miss.size()) : 10;
        fmt::print("  {} mismatch, top {} by ref margin (large = suspicious):\n", miss.size(),
                   show);
        for (int k = 0; k < show; ++k)
            fmt::print("    img {:6d}  ref margin {:.4f}\n", miss[k].second, miss[k].first);
    }

    const bool pass = tol > 0.0f ? max_diff <= tol : true;
    if (tol > 0.0f)
        fmt::print("  tol {:.1e} -> {}\n", tol, pass ? "PASS" : "FAIL");
    return pass;
}

template <typename F>
Timing run_timed(int runs, F &&body)
{
    body();
    double best = 0.0, worst = 0.0, sum = 0.0;
    for (int r = 0; r < runs; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        body();
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        if (r == 0 || ms < best)
            best = ms;
        if (ms > worst)
            worst = ms;
        sum += ms;
    }
    return {best, sum / runs, worst};
}

} // namespace optcnn

#endif // COMMON_H_
