// benchmarks/benchmark.cpp
//
// Fair, like-for-like comparison of this library's AVX2 kernels against:
//   1. a scalar baseline  (same -O3, auto-vectorisation disabled)  → "what SIMD buys"
//   2. Eigen 3.4 fixed-size types                                   → an established,
//      header-only, fully-inlined, compile-time-sized C++ library - the closest
//      possible apples-to-apples reference (no runtime dispatch, no framework).
//
// All three implementations run in the same process, on the same hardware, through
// the same harness (benchmarks/bench_common.h), on byte-identical input data that
// is reset (untimed) before every measurement batch. Outputs are anchored with
// optimisation barriers so no implementation can have its work eliminated.
//
// Sizes match the actual MNIST network:
//   Flatten<28,28> → Dense<784,32> → Sigmoid → Dense<32,10> → Sigmoid
//
// Build:  cd build && cmake .. && make benchmark
// Run:    ./benchmark

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <cstring>
#include <chrono>      // also pulled in transitively by network_utils.h
#include <immintrin.h>
#include <cmath>

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

#include "tensor/tensor.h"
#include "network/network_parameters.h"
#include "bench_common.h"

// ============================================================
// Scalar baselines - auto-vectorisation explicitly disabled
// ============================================================

#pragma GCC push_options
#pragma GCC optimize("O3,no-tree-vectorize")

__attribute__((noinline))
static void naive_add_ip(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
}

__attribute__((noinline))
static void naive_mul_ip(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] *= b[i];
}

__attribute__((noinline))
static void naive_matvec(const float* __restrict__ W, const float* __restrict__ v,
                          float* __restrict__ out, size_t cols, size_t rows) {
    for (size_t r = 0; r < rows; ++r) {
        float s = 0.f;
        const float* row = W + r * cols;
        for (size_t c = 0; c < cols; ++c) s += row[c] * v[c];
        out[r] = s;
    }
}

__attribute__((noinline))
static void naive_outer(const float* __restrict__ a, size_t m,
                         const float* __restrict__ b, size_t n,
                         float* __restrict__ out) {
    for (size_t i = 0; i < m; ++i) {
        const float ai = a[i];
        for (size_t j = 0; j < n; ++j) out[i * n + j] = ai * b[j];
    }
}

__attribute__((noinline))
static void naive_relu(float* __restrict__ a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = a[i] > 0.f ? a[i] : 0.f;
}

__attribute__((noinline))
static void naive_sigmoid(float* __restrict__ a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = 1.f / (1.f + expf(-a[i]));
}

#pragma GCC pop_options

// ============================================================
// Helpers
// ============================================================

static std::mt19937 g_rng(20240618);

static float* alloc(size_t n) {
    return static_cast<float*>(_mm_malloc(n * sizeof(float), 32));
}
static float* alloc_const(size_t n, float v) {
    float* p = alloc(n);
    for (size_t i = 0; i < n; ++i) p[i] = v;
    return p;
}
static float* alloc_normal(size_t n) {
    float* p = alloc(n);
    std::normal_distribution<float> d(0.f, 1.f);
    for (size_t i = 0; i < n; ++i) p[i] = d(g_rng);
    return p;
}

static void print_header() {
    std::cout << "\n"
              << std::left  << std::setw(34) << "Operation"
              << std::setw(24) << "Size"
              << std::right
              << std::setw(11) << "Scalar"
              << std::setw(13) << "This (AVX2)"
              << std::setw(11) << "Eigen"
              << std::setw(13) << "vs scalar"
              << std::setw(12) << "vs Eigen"
              << "\n" << std::string(118, '-') << "\n";
}

// ratio "vs Eigen": >1 means this library is faster than Eigen.
static void print_row(const char* op, const char* size,
                      double scalar_us, double lib_us, double eigen_us) {
    std::cout << std::left  << std::setw(34) << op
              << std::setw(24) << size
              << std::right << std::fixed
              << std::setw(8)  << std::setprecision(3) << scalar_us << " us"
              << std::setw(10) << std::setprecision(3) << lib_us    << " us"
              << std::setw(8)  << std::setprecision(3) << eigen_us  << " us"
              << std::setw(11) << std::setprecision(2) << (scalar_us / lib_us) << "x"
              << std::setw(11) << std::setprecision(2) << (eigen_us  / lib_us) << "x"
              << "\n";
}

// ============================================================
// Main
// ============================================================

int main()
{
    Eigen::setNbThreads(1);

    std::cout
        << "======================================================================\n"
        << "  Deep Library C++ - Kernel Benchmark (fair, single-thread)\n"
        << "  CPU    : Intel Core i7-14700KF  (5.5 GHz max)\n"
        << "  SIMD   : AVX2 + FMA  (8 x float32/register)\n"
        << "  Build  : -O3 -mavx2 -mfma  (C++23)\n"
        << "  Scalar : same -O3, auto-vectorisation disabled (no-tree-vectorize)\n"
        << "  Eigen  : " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
                          << EIGEN_MINOR_VERSION << ", fixed-size, 1 thread\n"
        << "  Method : min of " << bench_cfg::REPEATS << " batches x "
                                 << bench_cfg::ITERS << " calls; identical data; barriers on every result\n"
        << "======================================================================\n";

    // ────────────────────────────────────────────────────────────────────────
    // Section 1 - Element-wise, in-place (+=, *=). No allocation on any side.
    // ────────────────────────────────────────────────────────────────────────
    print_header();
    std::cout << "  [Element-wise - in-place]\n";

    auto bench_elementwise = [&](const char* op, const char* size, size_t N,
                                 float a0, float b0, bool mul)
    {
        float* golden = alloc_const(N, a0);
        float* b      = alloc_const(N, b0);

        float* as = alloc(N);          // scalar operand
        float* ae = alloc(N);          // eigen  operand

        auto reset_buf = [&](float* dst){ std::memcpy(dst, golden, N * sizeof(float)); };

        double scalar_us = bench_us([&]{ reset_buf(as); },
            [&]{ if (mul) naive_mul_ip(as, b, N); else naive_add_ip(as, b, N);
                 ClobberMemory(); });

        Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>, Eigen::Aligned32> emA(ae, (Eigen::Index)N);
        Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>, Eigen::Aligned32> emB(b, (Eigen::Index)N);
        double eigen_us = bench_us([&]{ reset_buf(ae); },
            [&]{ if (mul) emA *= emB; else emA += emB; ClobberMemory(); });

        _mm_free(golden); _mm_free(b); _mm_free(as); _mm_free(ae);
        return std::pair{scalar_us, eigen_us};
    };

    // Library element-wise needs the compile-time-sized Tensor type, so each size
    // is spelled out explicitly.
    {
        constexpr size_t N = 784;
        float* golden = alloc_const(N, 0.5f);
        float* b      = alloc_const(N, 0.01f);
        Tensor<N> ta(golden), tb(b);
        double lib_us = bench_us([&]{ ta = Tensor<N>(golden); },
            [&]{ ta += tb; ClobberMemory(); });
        auto [s, e] = bench_elementwise("add (+=)", "784 floats", N, 0.5f, 0.01f, false);
        print_row("Element-wise add (+=)", "784 floats", s, lib_us, e);
        _mm_free(golden); _mm_free(b);
    }
    {
        constexpr size_t N = 784 * 32;
        float* golden = alloc_const(N, 0.5f);
        float* b      = alloc_const(N, 0.01f);
        Tensor<N> ta(golden), tb(b);
        double lib_us = bench_us([&]{ ta = Tensor<N>(golden); },
            [&]{ ta += tb; ClobberMemory(); });
        auto [s, e] = bench_elementwise("add", "784x32 (98 KB)", N, 0.5f, 0.01f, false);
        print_row("Element-wise add (+=)", "784x32 (98 KB)", s, lib_us, e);
        _mm_free(golden); _mm_free(b);
    }
    {
        constexpr size_t N = 784;
        float* golden = alloc_const(N, 0.999f);
        float* b      = alloc_const(N, 0.999f);
        Tensor<N> ta(golden), tb(b);
        double lib_us = bench_us([&]{ ta = Tensor<N>(golden); },
            [&]{ ta *= tb; ClobberMemory(); });
        auto [s, e] = bench_elementwise("mul", "784 floats", N, 0.999f, 0.999f, true);
        print_row("Element-wise mul (*=)", "784 floats", s, lib_us, e);
        _mm_free(golden); _mm_free(b);
    }
    {
        constexpr size_t N = 784 * 32;
        float* golden = alloc_const(N, 0.999f);
        float* b      = alloc_const(N, 0.999f);
        Tensor<N> ta(golden), tb(b);
        double lib_us = bench_us([&]{ ta = Tensor<N>(golden); },
            [&]{ ta *= tb; ClobberMemory(); });
        auto [s, e] = bench_elementwise("mul", "784x32 (98 KB)", N, 0.999f, 0.999f, true);
        print_row("Element-wise mul (*=)", "784x32 (98 KB)", s, lib_us, e);
        _mm_free(golden); _mm_free(b);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Section 2 - Dense forward: W·x. Output allocated per call on BOTH sides
    // (the library returns a fresh Tensor; Eigen mallocs/frees a matching buffer)
    // so the comparison includes the same allocation cost.
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "\n  [Dense forward - W (rows x cols) * v(cols), output allocated per call]\n";

    auto bench_matvec = [&](const char* size, auto rows_c, auto cols_c)
    {
        constexpr size_t ROWS = rows_c;
        constexpr size_t COLS = cols_c;
        float* W = alloc_normal(ROWS * COLS);
        float* v = alloc_normal(COLS);

        float* outn = alloc(ROWS);
        double scalar_us = bench_us([&]{
            naive_matvec(W, v, outn, COLS, ROWS); DoNotOptimize(outn[0]); });
        _mm_free(outn);

        Tensor<COLS, ROWS> tW(W);
        Tensor<COLS>       tv(v);
        double lib_us = bench_us([&]{
            auto r = mul_b_transposed_scalar(tW, tv);
            float s = r(0); DoNotOptimize(s); });

        using MatW = Eigen::Matrix<float, ROWS, COLS, Eigen::RowMajor>;
        Eigen::Map<const MatW, Eigen::Aligned32> eW(W);
        Eigen::Map<const Eigen::Matrix<float, COLS, 1>, Eigen::Aligned32> ev(v);
        double eigen_us = bench_us([&]{
            float* o = alloc(ROWS);
            Eigen::Map<Eigen::Matrix<float, ROWS, 1>> eo(o);
            eo.noalias() = eW * ev;
            DoNotOptimize(o[0]);
            _mm_free(o); });

        _mm_free(W); _mm_free(v);
        char buf[48]; std::snprintf(buf, sizeof(buf), "%s", size);
        print_row("Dense forward (mat-vec)", buf, scalar_us, lib_us, eigen_us);
    };
    bench_matvec("784x32 -> 32",  std::integral_constant<size_t,32>{}, std::integral_constant<size_t,784>{});
    bench_matvec("32x10 -> 10",   std::integral_constant<size_t,10>{}, std::integral_constant<size_t,32>{});

    // ────────────────────────────────────────────────────────────────────────
    // Section 3 - Dense backward: outer product grad ⊗ input → ΔW.
    // Output allocated per call on both sides (same as forward).
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "\n  [Dense backward - vec(m) (x) vec(n) -> m x n, output allocated per call]\n";

    auto bench_outer = [&](const char* size, auto m_c, auto n_c)
    {
        constexpr size_t M = m_c;
        constexpr size_t N = n_c;
        float* a = alloc_normal(M);
        float* b = alloc_normal(N);

        float* outn = alloc(M * N);
        double scalar_us = bench_us([&]{
            naive_outer(a, M, b, N, outn); DoNotOptimize(outn[0]); });
        _mm_free(outn);

        Tensor<M> ta(a);
        Tensor<N> tb(b);
        double lib_us = bench_us([&]{
            auto r = mul_transposed_scalar(ta, tb);
            float s = r(0); DoNotOptimize(s); });

        Eigen::Map<const Eigen::Matrix<float, M, 1>, Eigen::Aligned32> ea(a);
        Eigen::Map<const Eigen::Matrix<float, N, 1>, Eigen::Aligned32> eb(b);
        double eigen_us = bench_us([&]{
            float* o = alloc(M * N);
            Eigen::Map<Eigen::Matrix<float, M, N, Eigen::RowMajor>> eo(o);
            eo.noalias() = ea * eb.transpose();
            DoNotOptimize(o[0]);
            _mm_free(o); });

        _mm_free(a); _mm_free(b);
        print_row("Dense backward (outer)", size, scalar_us, lib_us, eigen_us);
    };
    bench_outer("vec(32) (x) vec(784)", std::integral_constant<size_t,32>{}, std::integral_constant<size_t,784>{});
    bench_outer("vec(10) (x) vec(32)",  std::integral_constant<size_t,10>{}, std::integral_constant<size_t,32>{});

    // ────────────────────────────────────────────────────────────────────────
    // Section 4 - Activations, in-place. Identical mixed-sign random input.
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "\n  [Activations - in-place, mixed-sign random input]\n";

    auto bench_relu = [&](const char* size, auto n_c)
    {
        constexpr size_t N = n_c;
        float* golden = alloc_normal(N);
        float* as = alloc(N);
        float* ae = alloc(N);
        auto cpy = [&](float* d){ std::memcpy(d, golden, N * sizeof(float)); };

        double scalar_us = bench_us([&]{ cpy(as); },
            [&]{ naive_relu(as, N); ClobberMemory(); });

        Tensor<N> ta(golden);
        double lib_us = bench_us([&]{ ta = Tensor<N>(golden); },
            [&]{ ta.apply_ReLU(); ClobberMemory(); });

        Eigen::Map<Eigen::Array<float, N, 1>, Eigen::Aligned32> em(ae);
        double eigen_us = bench_us([&]{ cpy(ae); },
            [&]{ em = em.max(0.f); ClobberMemory(); });

        _mm_free(golden); _mm_free(as); _mm_free(ae);
        print_row("ReLU", size, scalar_us, lib_us, eigen_us);
    };
    bench_relu("784 values", std::integral_constant<size_t,784>{});
    bench_relu("32 values",  std::integral_constant<size_t,32>{});

    {
        constexpr size_t N = 784;
        float* golden = alloc_normal(N);
        float* as = alloc(N);
        float* ae = alloc(N);
        auto cpy = [&](float* d){ std::memcpy(d, golden, N * sizeof(float)); };

        double scalar_us = bench_us([&]{ cpy(as); },
            [&]{ naive_sigmoid(as, N); ClobberMemory(); });

        Tensor<N> ta(golden);
        double lib_us = bench_us([&]{ ta = Tensor<N>(golden); },
            [&]{ ta.apply_sigmoid(); ClobberMemory(); });

        Eigen::Map<Eigen::Array<float, N, 1>, Eigen::Aligned32> em(ae);
        double eigen_us = bench_us([&]{ cpy(ae); },
            [&]{ em = 1.f / (1.f + (-em).exp()); ClobberMemory(); });

        _mm_free(golden); _mm_free(as); _mm_free(ae);
        print_row("Sigmoid", "784 values", scalar_us, lib_us, eigen_us);
    }

    std::cout << "\n  Note: 'This (AVX2)' uses a fast polynomial exp; Eigen uses an\n"
                 "  accurate vectorised exp - the Sigmoid row is speed-vs-accuracy.\n\n";
    return 0;
}
