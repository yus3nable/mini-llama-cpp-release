#include "test_main.h"
#include "mini_llama/quant.h"
#include "mini_llama/tensor.h"
#include "mini_llama/ops.h"

using namespace mini_llama;

static bool test_q8_0_block_layout() {
    ASSERT_EQ(Q8_0_BLOCK_SIZE, 32);
    ASSERT_EQ(sizeof(BlockQ8_0), 34);
    return true;
}

// ---------------------------------------------------------------------------
// Q8_0 quantization / dequantization
// ---------------------------------------------------------------------------
static bool test_q8_0_roundtrip_identity() {
    Tensor src({32}, 0.0f);
    for (int i = 0; i < 32; ++i) {
        src.data[i] = static_cast<float>(i) * 0.1f;
    }

    auto blocks = quantize_to_q8_0(src);
    ASSERT_EQ(blocks.size(), 1);

    Tensor dst = dequantize_from_q8_0(blocks, src.shape);
    ASSERT_EQ(dst.shape.size(), 1);
    ASSERT_EQ(dst.shape[0], 32);

    // Should be close but not exact due to quantization.
    // Q8_0 roundtrip error can be up to ~0.5 * scale per element.
    float max_err = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float err = std::abs(src.data[i] - dst.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    ASSERT_TRUE(max_err < 6e-2f);
    return true;
}

static bool test_q8_0_all_zeros() {
    Tensor src({64}, 0.0f);
    auto blocks = quantize_to_q8_0(src);
    ASSERT_EQ(blocks.size(), 2);

    Tensor dst = dequantize_from_q8_0(blocks, src.shape);
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_NEAR(dst.data[i], 0.0f, 1e-6f);
    }
    return true;
}

static bool test_q8_0_multi_block() {
    Tensor src({100}, 0.0f);
    for (int i = 0; i < 100; ++i) {
        src.data[i] = static_cast<float>(i) * 0.05f;
    }

    auto blocks = quantize_to_q8_0(src);
    ASSERT_EQ(blocks.size(), 4); // ceil(100 / 32) = 4

    Tensor dst = dequantize_from_q8_0(blocks, src.shape);
    ASSERT_EQ(dst.shape[0], 100);

    float max_err = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float err = std::abs(src.data[i] - dst.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    ASSERT_TRUE(max_err < 6e-2f);
    return true;
}

static bool test_q8_0_empty_tensor() {
    try {
        Tensor src({0}, 0.0f);
        (void)src;
        ASSERT_FAIL("expected exception for zero-dimension tensor");
    } catch (const std::runtime_error&) {
        // expected: Tensor rejects zero dimension
    }
    return true;
}

// ---------------------------------------------------------------------------
// Q8_0 matmul
// ---------------------------------------------------------------------------
static bool test_matmul_q8_0_matches_f32() {
    // Small exact test
    Tensor A({2, 3}, 0.0f);
    A.data[0] = 1.0f; A.data[1] = 2.0f; A.data[2] = 3.0f;
    A.data[3] = 4.0f; A.data[4] = 5.0f; A.data[5] = 6.0f;

    Tensor B({3, 2}, 0.0f);
    B.data[0] = 1.0f; B.data[1] = 2.0f;
    B.data[2] = 3.0f; B.data[3] = 4.0f;
    B.data[4] = 5.0f; B.data[5] = 6.0f;

    auto qA = quantize_to_q8_0(A);
    Tensor C_f32 = matmul(A, B);
    Tensor C_q8 = matmul_q8_0(qA, B, A.shape);

    ASSERT_EQ(C_f32.shape.size(), 2);
    ASSERT_EQ(C_q8.shape.size(), 2);
    ASSERT_EQ(C_f32.shape[0], C_q8.shape[0]);
    ASSERT_EQ(C_f32.shape[1], C_q8.shape[1]);

    for (size_t i = 0; i < C_f32.size(); ++i) {
        // Q8_0 matmul accumulates per-element quantization error;
        // tolerance must be looser than raw roundtrip.
        ASSERT_NEAR(C_f32.data[i], C_q8.data[i], 2e-1f);
    }
    return true;
}

static bool test_compare_matmul_error() {
    Tensor A({16, 16}, 0.0f);
    Tensor B({16, 16}, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) {
        A.data[i] = static_cast<float>(i) * 0.01f - 1.0f;
    }
    for (size_t i = 0; i < B.size(); ++i) {
        B.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
    }

    float err = compare_matmul_error(A, B);
    ASSERT_TRUE(err >= 0.0f);
    ASSERT_TRUE(err < 1e-1f);  // Q8_0 error should be small
    return true;
}

static bool test_dequantize_rejects_bad_shape() {
    Tensor src({32}, 1.0f);
    auto blocks = quantize_to_q8_0(src);

    try {
        (void)dequantize_from_q8_0(blocks, {-1});
        ASSERT_FAIL("expected exception for negative shape");
    } catch (const std::runtime_error&) {
        // expected
    }

    try {
        (void)dequantize_from_q8_0(blocks, {0});
        ASSERT_FAIL("expected exception for zero shape");
    } catch (const std::runtime_error&) {
        // expected
    }

    return true;
}

static bool test_dequantize_rejects_block_count_mismatch() {
    Tensor src({64}, 1.0f);
    auto blocks = quantize_to_q8_0(src);
    blocks.pop_back();

    try {
        (void)dequantize_from_q8_0(blocks, src.shape);
        ASSERT_FAIL("expected exception for block count mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ---------------------------------------------------------------------------
// Q4_0 quantization / dequantization
// ---------------------------------------------------------------------------
static bool test_q4_0_block_layout() {
    ASSERT_EQ(Q4_0_BLOCK_SIZE, 32);
    ASSERT_EQ(sizeof(BlockQ4_0), 18);
    return true;
}

static bool test_q4_1_block_layout() {
    ASSERT_EQ(Q4_1_BLOCK_SIZE, 32);
    ASSERT_EQ(sizeof(BlockQ4_1), 20);
    return true;
}

static bool test_q4_0_roundtrip_identity() {
    Tensor src({32}, 0.0f);
    for (int i = 0; i < 32; ++i) {
        src.data[i] = static_cast<float>(i) * 0.1f;
    }

    auto blocks = quantize_to_q4_0(src);
    ASSERT_EQ(blocks.size(), 1);

    Tensor dst = dequantize_from_q4_0(blocks, src.shape);
    ASSERT_EQ(dst.shape[0], 32);

    float max_err = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        float err = std::abs(src.data[i] - dst.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    ASSERT_TRUE(max_err < 3e-1f);
    return true;
}

static bool test_q4_0_all_zeros() {
    Tensor src({64}, 0.0f);
    auto blocks = quantize_to_q4_0(src);
    ASSERT_EQ(blocks.size(), 2);

    Tensor dst = dequantize_from_q4_0(blocks, src.shape);
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_NEAR(dst.data[i], 0.0f, 1e-6f);
    }
    return true;
}

static bool test_q4_1_dequantize_known_values() {
    BlockQ4_1 block{};
    block.d = 0x3800; // fp16(0.5)
    block.m = 0xbc00; // fp16(-1.0)
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = static_cast<uint8_t>(i | ((15 - i) << 4));
    }

    Tensor dst = dequantize_from_q4_1({block}, {32});
    ASSERT_EQ(dst.shape[0], 32);
    for (int i = 0; i < 16; ++i) {
        ASSERT_NEAR(dst.data[i], -1.0f + 0.5f * static_cast<float>(i), 1e-6f);
        ASSERT_NEAR(dst.data[i + 16], -1.0f + 0.5f * static_cast<float>(15 - i), 1e-6f);
    }
    return true;
}

static bool test_q4_1_linear_known_values() {
    BlockQ4_1 block{};
    block.d = 0x3c00; // fp16(1.0)
    block.m = 0x0000; // fp16(0.0)
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = 0x11; // q0=1, q1=1
    }

    Tensor x({32}, 2.0f);
    Tensor y = linear_q4_1(x, {block}, {1, 32});
    ASSERT_EQ(y.shape[0], 1);
    ASSERT_NEAR(y.data[0], 64.0f, 1e-6f);
    return true;
}

static bool test_q4_0_linear_matches_f32() {
    // Use in_features divisible by Q4_0_BLOCK_SIZE (32)
    Tensor W({4, 32}, 0.0f);
    for (size_t i = 0; i < W.size(); ++i) {
        W.data[i] = static_cast<float>(i) * 0.05f - 1.0f;
    }

    Tensor x({32}, 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        x.data[i] = static_cast<float>(i) * 0.1f - 0.4f;
    }

    auto qW = quantize_to_q4_0(W);
    Tensor y_f32 = linear(x, W);
    Tensor y_q4 = linear_q4_0(x, qW, W.shape);

    ASSERT_EQ(y_f32.shape, y_q4.shape);
    for (size_t i = 0; i < y_f32.size(); ++i) {
        ASSERT_NEAR(y_f32.data[i], y_q4.data[i], 2.0f);
    }
    return true;
}

static bool test_compare_q4_0_error() {
    // Use dimensions divisible by Q4_0_BLOCK_SIZE (32)
    Tensor W({8, 32}, 0.0f);
    Tensor x({32}, 0.0f);
    for (size_t i = 0; i < W.size(); ++i) {
        W.data[i] = static_cast<float>(i) * 0.01f - 1.0f;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
    }

    float err = compare_q4_0_error(W, x);
    ASSERT_TRUE(err >= 0.0f);
    ASSERT_TRUE(err < 2.0f);
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct QuantTestRegistrar {
    QuantTestRegistrar() {
        register_test("q8_0_block_layout", test_q8_0_block_layout);
        register_test("q8_0_roundtrip_identity", test_q8_0_roundtrip_identity);
        register_test("q8_0_all_zeros", test_q8_0_all_zeros);
        register_test("q8_0_multi_block", test_q8_0_multi_block);
        register_test("q8_0_empty_tensor", test_q8_0_empty_tensor);
        register_test("matmul_q8_0_matches_f32", test_matmul_q8_0_matches_f32);
        register_test("compare_matmul_error", test_compare_matmul_error);
        register_test("dequantize_rejects_bad_shape", test_dequantize_rejects_bad_shape);
        register_test("dequantize_rejects_block_count_mismatch", test_dequantize_rejects_block_count_mismatch);
        register_test("q4_0_block_layout", test_q4_0_block_layout);
        register_test("q4_1_block_layout", test_q4_1_block_layout);
        register_test("q4_0_roundtrip_identity", test_q4_0_roundtrip_identity);
        register_test("q4_0_all_zeros", test_q4_0_all_zeros);
        register_test("q4_1_dequantize_known_values", test_q4_1_dequantize_known_values);
        register_test("q4_1_linear_known_values", test_q4_1_linear_known_values);
        register_test("q4_0_linear_matches_f32", test_q4_0_linear_matches_f32);
        register_test("compare_q4_0_error", test_compare_q4_0_error);
    }
} quant_test_registrar;
