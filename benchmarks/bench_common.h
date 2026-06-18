// benchmarks/bench_common.h
//
// Shared, deliberately-fair micro-benchmark harness used by every benchmark
// target (this library, the scalar baseline, Eigen). Keeping the
// timing methodology in one place guarantees an apples-to-apples comparison.
//
// Design choices (and why they matter for an unbiased result):
//
//   * Optimisation barriers (DoNotOptimize / ClobberMemory) are applied to
//     EVERY implementation, so none of them can have its work eliminated as
//     dead code. Without this a discarded result lets the compiler delete the
//     whole computation, producing absurd "speedups".
//
//   * The reported time is the MINIMUM over many repeat batches, not the mean.
//     The minimum is the run least perturbed by interrupts, context switches
//     and frequency scaling - the standard estimator for short kernels (it is
//     what nanobench / Google Benchmark surface as the headline number). A mean
//     is dominated by outliers and over-reports latency for sub-microsecond ops.
//
//   * steady_clock is used (monotonic, never adjusted) rather than
//     high_resolution_clock, whose steadiness is implementation-defined.
//
//   * A caller-supplied reset() runs UNTIMED before each batch, so in-place
//     operators (a *= b, a += b) cannot drift their operands into the denormal
//     range over millions of iterations - denormals are handled in microcode and
//     would silently slow one implementation but not another.

#pragma once

#include <chrono>
#include <algorithm>
#include <limits>
#include <utility>

// ── Optimisation barriers (Google-Benchmark style) ──────────────────────────
#if defined(__clang__) || defined(__GNUC__)
template <typename T>
inline void DoNotOptimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}
template <typename T>
inline void DoNotOptimize(T& value) {
    asm volatile("" : "+r,m"(value) : : "memory");
}
inline void ClobberMemory() {
    asm volatile("" : : : "memory");
}
#else
template <typename T> inline void DoNotOptimize(const T& value) {
    volatile auto sink = value; (void)sink;
}
inline void ClobberMemory() {}
#endif

// ── Tunables (identical for every implementation) ───────────────────────────
namespace bench_cfg {
    inline constexpr int WARMUP  = 2000;
    inline constexpr int ITERS   = 20000;  // 0.999^20000 ≈ 2e-9 - stays normal
    inline constexpr int REPEATS = 25;
}

// Minimum µs/call over `repeats` batches of `iters` calls. `reset` runs untimed
// before every batch (use a no-op lambda when the kernel doesn't mutate state).
template <typename Reset, typename F>
double bench_us(Reset&& reset, F&& fn,
                int iters   = bench_cfg::ITERS,
                int repeats = bench_cfg::REPEATS,
                int warmup  = bench_cfg::WARMUP)
{
    reset();
    for (int i = 0; i < warmup; ++i) fn();

    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < repeats; ++r) {
        reset();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double us =
            std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        best = std::min(best, us);
    }
    return best;
}

// Convenience overload for kernels that need no reset.
template <typename F>
double bench_us(F&& fn) {
    return bench_us([]{}, std::forward<F>(fn));
}
