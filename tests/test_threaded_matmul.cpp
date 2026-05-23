#include "test_main.h"
#include "mini_llama/tensor.h"
#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/thread_pool.h"
#include <cmath>
#include <stdexcept>

using namespace mini_llama;

static bool test_matmul_naive_vs_threaded() {
    Tensor A({4, 8}, 0.0f);
    Tensor B({8, 4}, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A.data[i] = static_cast<float>(i) * 0.1f - 0.5f;
    for (size_t i = 0; i < B.size(); ++i) B.data[i] = static_cast<float>(i) * 0.05f - 0.3f;

    Tensor naive = matmul_dispatch(A, B, MatmulMode::Naive);
    Tensor threaded = matmul_dispatch(A, B, MatmulMode::Threaded);

    ASSERT_EQ(naive.shape.size(), threaded.shape.size());
    for (size_t i = 0; i < naive.shape.size(); ++i) {
        ASSERT_EQ(naive.shape[i], threaded.shape[i]);
    }
    for (size_t i = 0; i < naive.size(); ++i) {
        ASSERT_NEAR(naive.data[i], threaded.data[i], 1e-6f);
    }
    return true;
}

static bool test_linear_all_modes_match() {
    Tensor W({8, 16}, 0.0f);
    Tensor x({16}, 0.0f);
    for (size_t i = 0; i < W.size(); ++i) W.data[i] = static_cast<float>(i) * 0.05f - 0.4f;
    for (size_t i = 0; i < x.size(); ++i) x.data[i] = static_cast<float>(i) * 0.1f - 0.5f;

    Tensor naive = linear_dispatch(x, W, MatmulMode::Naive);
    Tensor threaded = linear_dispatch(x, W, MatmulMode::Threaded);
    Tensor simd = linear_dispatch(x, W, MatmulMode::Simd);
    Tensor threaded_simd = linear_dispatch(x, W, MatmulMode::ThreadedSimd);

    ASSERT_EQ(naive.shape.size(), threaded.shape.size());
    ASSERT_EQ(naive.shape.size(), simd.shape.size());
    ASSERT_EQ(naive.shape.size(), threaded_simd.shape.size());
    for (size_t i = 0; i < naive.shape.size(); ++i) {
        ASSERT_EQ(naive.shape[i], threaded.shape[i]);
        ASSERT_EQ(naive.shape[i], simd.shape[i]);
        ASSERT_EQ(naive.shape[i], threaded_simd.shape[i]);
    }
    for (size_t i = 0; i < naive.size(); ++i) {
        ASSERT_NEAR(naive.data[i], threaded.data[i], 1e-5f);
        ASSERT_NEAR(naive.data[i], simd.data[i], 1e-5f);
        ASSERT_NEAR(naive.data[i], threaded_simd.data[i], 1e-5f);
    }
    return true;
}

static bool test_linear_2d_input_all_modes_match() {
    Tensor W({6, 12}, 0.0f);
    Tensor x({1, 12}, 0.0f);
    for (size_t i = 0; i < W.size(); ++i) W.data[i] = static_cast<float>(i) * 0.03f - 0.2f;
    for (size_t i = 0; i < x.size(); ++i) x.data[i] = static_cast<float>(i) * 0.07f - 0.3f;

    Tensor naive = linear_dispatch(x, W, MatmulMode::Naive);
    Tensor threaded_simd = linear_dispatch(x, W, MatmulMode::ThreadedSimd);

    ASSERT_EQ(naive.shape.size(), threaded_simd.shape.size());
    for (size_t i = 0; i < naive.shape.size(); ++i) {
        ASSERT_EQ(naive.shape[i], threaded_simd.shape[i]);
    }
    for (size_t i = 0; i < naive.size(); ++i) {
        ASSERT_NEAR(naive.data[i], threaded_simd.data[i], 1e-5f);
    }
    return true;
}

static bool test_different_thread_counts_same_output() {
    Tensor W({10, 32}, 0.0f);
    Tensor x({32}, 0.0f);
    for (size_t i = 0; i < W.size(); ++i) W.data[i] = static_cast<float>(i) * 0.02f - 0.3f;
    for (size_t i = 0; i < x.size(); ++i) x.data[i] = static_cast<float>(i) * 0.05f - 0.4f;

    Tensor baseline = linear_dispatch(x, W, MatmulMode::Naive);

    int saved = get_thread_count();
    int thread_counts[] = {1, 2, 4};
    for (int tc : thread_counts) {
        set_thread_count(tc);
        Tensor result = linear_dispatch(x, W, MatmulMode::ThreadedSimd);
        ASSERT_EQ(baseline.shape.size(), result.shape.size());
        for (size_t i = 0; i < baseline.shape.size(); ++i) {
            ASSERT_EQ(baseline.shape[i], result.shape[i]);
        }
        for (size_t i = 0; i < baseline.size(); ++i) {
            ASSERT_NEAR(baseline.data[i], result.data[i], 1e-5f);
        }
    }

    set_thread_count(saved);
    return true;
}

static bool test_thread_count_api() {
    set_thread_count(0);
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 0) {
        ASSERT_EQ(get_thread_count(), hw);
    } else {
        ASSERT_EQ(get_thread_count(), 4);
    }

    set_thread_count(2);
    ASSERT_EQ(get_thread_count(), 2);
    set_thread_count(8);
    ASSERT_EQ(get_thread_count(), 8);
    set_thread_count(0);
    return true;
}

static bool test_parallel_for_propagates_exception() {
    int saved = get_thread_count();
    set_thread_count(4);
    try {
        parallel_for(128, [](int begin, int end) {
            if (begin <= 64 && 64 < end) {
                throw std::runtime_error("parallel worker failed");
            }
        });
        ASSERT_FAIL("expected worker exception to propagate");
    } catch (const std::runtime_error& e) {
        ASSERT_TRUE(std::string(e.what()).find("parallel worker failed") != std::string::npos);
    }
    set_thread_count(saved);
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct ThreadedMatmulTestRegistrar {
    ThreadedMatmulTestRegistrar() {
        register_test("matmul_naive_vs_threaded", test_matmul_naive_vs_threaded);
        register_test("linear_all_modes_match", test_linear_all_modes_match);
        register_test("linear_2d_input_all_modes_match", test_linear_2d_input_all_modes_match);
        register_test("different_thread_counts_same_output", test_different_thread_counts_same_output);
        register_test("thread_count_api", test_thread_count_api);
        register_test("parallel_for_propagates_exception", test_parallel_for_propagates_exception);
    }
} threaded_matmul_test_registrar;
