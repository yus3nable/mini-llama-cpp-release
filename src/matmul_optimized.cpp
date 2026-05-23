#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

namespace mini_llama {

// ---------------------------------------------------------------------------
// Naive implementations (kept for readability and fallback)
// ---------------------------------------------------------------------------

static Tensor matmul_naive(const Tensor& A, const Tensor& B) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    Tensor C({M, N}, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
    return C;
}

static Tensor linear_naive(const Tensor& x, const Tensor& W) {
    int in_features = W.shape[1];
    int out_features = W.shape[0];
    bool is_1d = (x.ndim() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features}, 0.0f);
    for (int j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += x.data[k] * W.data[j * in_features + k];
        }
        result.data[j] = sum;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Threaded implementations
// ---------------------------------------------------------------------------

static Tensor matmul_threaded(const Tensor& A, const Tensor& B) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    Tensor C({M, N}, 0.0f);
    parallel_for(M, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A.data[i * K + k] * B.data[k * N + j];
                }
                C.data[i * N + j] = sum;
            }
        }
    });
    return C;
}

static Tensor linear_threaded(const Tensor& x, const Tensor& W) {
    int in_features = W.shape[1];
    int out_features = W.shape[0];
    bool is_1d = (x.ndim() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features}, 0.0f);
    parallel_for(out_features, [&](int begin, int end) {
        for (int j = begin; j < end; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += x.data[k] * W.data[j * in_features + k];
            }
            result.data[j] = sum;
        }
    });
    return result;
}

// ---------------------------------------------------------------------------
// SIMD dot-product helpers (platform-specific)
// ---------------------------------------------------------------------------

#ifdef MINI_LLAMA_USE_NEON
static float dot_simd_neon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float32x2_t r = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    r = vpadd_f32(r, r);
    float result = vget_lane_f32(r, 0);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}
#endif

#ifdef MINI_LLAMA_USE_AVX2
static float dot_simd_avx2(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // Horizontal reduce
    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    sum_lo = _mm_add_ps(sum_lo, sum_hi);
    sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
    sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
    float result = _mm_cvtss_f32(sum_lo);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}
#endif

static float dot_simd(const float* a, const float* b, int n) {
#if defined(MINI_LLAMA_USE_AVX2)
    return dot_simd_avx2(a, b, n);
#elif defined(MINI_LLAMA_USE_NEON)
    return dot_simd_neon(a, b, n);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

// ---------------------------------------------------------------------------
// SIMD implementations
// ---------------------------------------------------------------------------

static Tensor linear_simd(const Tensor& x, const Tensor& W) {
    int in_features = W.shape[1];
    int out_features = W.shape[0];
    bool is_1d = (x.ndim() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features}, 0.0f);
    for (int j = 0; j < out_features; ++j) {
        result.data[j] = dot_simd(x.data.data(), W.data.data() + j * in_features, in_features);
    }
    return result;
}

static Tensor linear_threaded_simd(const Tensor& x, const Tensor& W) {
    int in_features = W.shape[1];
    int out_features = W.shape[0];
    bool is_1d = (x.ndim() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features}, 0.0f);
    parallel_for(out_features, [&](int begin, int end) {
        for (int j = begin; j < end; ++j) {
            result.data[j] = dot_simd(x.data.data(), W.data.data() + j * in_features, in_features);
        }
    });
    return result;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

MatmulMode default_matmul_mode() {
    return MatmulMode::ThreadedSimd;
}

Tensor matmul_dispatch(const Tensor& A, const Tensor& B, MatmulMode mode) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("matmul_dispatch: expected 2D tensors");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::runtime_error("matmul_dispatch: dimension mismatch");
    }
    switch (mode) {
        case MatmulMode::Naive:
            return matmul_naive(A, B);
        case MatmulMode::Threaded:
            return matmul_threaded(A, B);
        case MatmulMode::Simd:
        case MatmulMode::ThreadedSimd:
            // For general matmul, B column access is not contiguous,
            // so SIMD is complex. Fall back to threaded for now.
            return matmul_threaded(A, B);
    }
    return matmul_threaded(A, B);
}

Tensor linear_dispatch(const Tensor& x, const Tensor& W, MatmulMode mode) {
    if (W.ndim() != 2) {
        throw std::runtime_error("linear_dispatch: expected W shape [out_features, in_features]");
    }
    int in_features;
    if (x.ndim() == 1) {
        in_features = x.shape[0];
    } else if (x.ndim() == 2 && x.shape[0] == 1) {
        in_features = x.shape[1];
    } else {
        throw std::runtime_error("linear_dispatch: expected x shape [in_features] or [1, in_features]");
    }
    if (W.shape[1] != in_features) {
        throw std::runtime_error("linear_dispatch: dimension mismatch");
    }
    switch (mode) {
        case MatmulMode::Naive:
            return linear_naive(x, W);
        case MatmulMode::Threaded:
            return linear_threaded(x, W);
        case MatmulMode::Simd:
            return linear_simd(x, W);
        case MatmulMode::ThreadedSimd:
            return linear_threaded_simd(x, W);
    }
    return linear_threaded_simd(x, W);
}

} // namespace mini_llama
