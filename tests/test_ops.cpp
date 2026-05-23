#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"
#include "test_main.h"

using namespace mini_llama;

// ==========================================================================
// matmul
// ==========================================================================
static bool test_matmul_identity() {
    Tensor A({2, 2}, 0.0f);
    A[0] = 1.0f; A[1] = 0.0f;
    A[2] = 0.0f; A[3] = 1.0f;
    Tensor B({2, 2}, 0.0f);
    B[0] = 1.0f; B[1] = 2.0f;
    B[2] = 3.0f; B[3] = 4.0f;
    Tensor C = matmul(A, B);
    ASSERT_EQ(C.shape[0], 2);
    ASSERT_EQ(C.shape[1], 2);
    ASSERT_NEAR(C[0], 1.0f, 1e-5f);
    ASSERT_NEAR(C[1], 2.0f, 1e-5f);
    ASSERT_NEAR(C[2], 3.0f, 1e-5f);
    ASSERT_NEAR(C[3], 4.0f, 1e-5f);
    return true;
}

static bool test_matmul_simple() {
    Tensor A({2, 2}, 0.0f);
    A[0] = 1.0f; A[1] = 2.0f;
    A[2] = 3.0f; A[3] = 4.0f;
    Tensor B({2, 2}, 0.0f);
    B[0] = 5.0f; B[1] = 6.0f;
    B[2] = 7.0f; B[3] = 8.0f;
    Tensor C = matmul(A, B);
    ASSERT_NEAR(C[0], 19.0f, 1e-5f);
    ASSERT_NEAR(C[1], 22.0f, 1e-5f);
    ASSERT_NEAR(C[2], 43.0f, 1e-5f);
    ASSERT_NEAR(C[3], 50.0f, 1e-5f);
    return true;
}

static bool test_matmul_rectangular() {
    // A: [2, 3], B: [3, 4] -> C: [2, 4]
    Tensor A({2, 3}, 0.0f);
    A[0] = 1.0f; A[1] = 2.0f; A[2] = 3.0f;
    A[3] = 4.0f; A[4] = 5.0f; A[5] = 6.0f;
    Tensor B({3, 4}, 0.0f);
    for (int i = 0; i < 12; ++i) {
        B[i] = static_cast<float>(i);
    }
    Tensor C = matmul(A, B);
    ASSERT_EQ(C.shape[0], 2);
    ASSERT_EQ(C.shape[1], 4);
    // C[0,0] = 1*0 + 2*4 + 3*8 = 0 + 8 + 24 = 32
    ASSERT_NEAR(C.at2(0, 0), 32.0f, 1e-5f);
    // C[1,0] = 4*0 + 5*4 + 6*8 = 0 + 20 + 48 = 68
    ASSERT_NEAR(C.at2(1, 0), 68.0f, 1e-5f);
    return true;
}

static bool test_matmul_shape_mismatch() {
    Tensor A({2, 3}, 0.0f);
    Tensor B({2, 3}, 0.0f);
    try {
        matmul(A, B);
        ASSERT_FAIL("expected exception for matmul shape mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ==========================================================================
// linear
// ==========================================================================
static bool test_linear_1d() {
    Tensor x({3}, 0.0f);
    x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
    Tensor W({3, 3}, 0.0f);
    W[0] = 1.0f; W[4] = 1.0f; W[8] = 1.0f;
    Tensor y = linear(x, W);
    ASSERT_EQ(y.ndim(), 1);
    ASSERT_EQ(y.shape[0], 3);
    ASSERT_NEAR(y[0], 1.0f, 1e-5f);
    ASSERT_NEAR(y[1], 2.0f, 1e-5f);
    ASSERT_NEAR(y[2], 3.0f, 1e-5f);
    return true;
}

static bool test_linear_2d() {
    Tensor x({1, 3}, 0.0f);
    x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
    Tensor W({2, 3}, 0.0f);
    W[0] = 1.0f; W[1] = 0.0f; W[2] = 0.0f;
    W[3] = 0.0f; W[4] = 1.0f; W[5] = 0.0f;
    Tensor y = linear(x, W);
    ASSERT_EQ(y.ndim(), 2);
    ASSERT_EQ(y.shape[0], 1);
    ASSERT_EQ(y.shape[1], 2);
    ASSERT_NEAR(y[0], 1.0f, 1e-5f);
    ASSERT_NEAR(y[1], 2.0f, 1e-5f);
    return true;
}

static bool test_linear_shape_mismatch() {
    Tensor x({3}, 0.0f);
    Tensor W({2, 4}, 0.0f);
    try {
        linear(x, W);
        ASSERT_FAIL("expected exception for linear shape mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ==========================================================================
// rmsnorm
// ==========================================================================
static bool test_rmsnorm_unit() {
    Tensor x({4}, 1.0f);
    Tensor w({4}, 1.0f);
    Tensor y = rmsnorm(x, w, 1e-5f);
    ASSERT_EQ(y.shape[0], 4);
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(y[i], 1.0f, 1e-4f);
    }
    return true;
}

static bool test_rmsnorm_scaled() {
    Tensor x({4}, 0.0f);
    x[0] = 2.0f;
    Tensor w({4}, 1.0f);
    Tensor y = rmsnorm(x, w, 1e-5f);
    ASSERT_NEAR(y[0], 2.0f, 1e-4f);
    ASSERT_NEAR(y[1], 0.0f, 1e-4f);
    return true;
}

static bool test_rmsnorm_with_weight() {
    // x = [1, 1, 1, 1], weight = [2, 2, 2, 2]
    // rms = 1, scale = 1, y = x * scale * weight = [2, 2, 2, 2]
    Tensor x({4}, 1.0f);
    Tensor w({4}, 2.0f);
    Tensor y = rmsnorm(x, w, 1e-5f);
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(y[i], 2.0f, 1e-4f);
    }
    return true;
}

static bool test_rmsnorm_shape_mismatch() {
    Tensor x({4}, 0.0f);
    Tensor w({3}, 0.0f);
    try {
        rmsnorm(x, w, 1e-5f);
        ASSERT_FAIL("expected exception for rmsnorm shape mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ==========================================================================
// softmax
// ==========================================================================
static bool test_softmax() {
    Tensor x({3}, 0.0f);
    x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
    Tensor y = softmax(x);
    ASSERT_EQ(y.shape[0], 3);
    ASSERT_NEAR(y[0], 0.09003057f, 1e-5f);
    ASSERT_NEAR(y[1], 0.24472847f, 1e-5f);
    ASSERT_NEAR(y[2], 0.66524096f, 1e-5f);
    float sum = y[0] + y[1] + y[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    return true;
}

static bool test_softmax_uniform() {
    // softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
    Tensor x({3}, 0.0f);
    Tensor y = softmax(x);
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(y[i], 1.0f / 3.0f, 1e-5f);
    }
    return true;
}

static bool test_softmax_wrong_dim() {
    Tensor x({2, 3}, 0.0f);
    try {
        softmax(x);
        ASSERT_FAIL("expected exception for softmax on 2D tensor");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_softmax_empty() {
    Tensor x;
    x.shape = {0};
    try {
        softmax(x);
        ASSERT_FAIL("expected exception for softmax on empty tensor");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ==========================================================================
// silu
// ==========================================================================
static bool test_silu() {
    Tensor x({1}, 0.0f);
    Tensor y = silu(x);
    ASSERT_NEAR(y[0], 0.0f, 1e-5f);

    x[0] = 1.0f;
    y = silu(x);
    ASSERT_NEAR(y[0], 0.7310586f, 1e-5f);
    return true;
}

// ==========================================================================
// swiglu
// ==========================================================================
static bool test_swiglu() {
    // swiglu(gate, up) = silu(gate) * up
    Tensor gate({3}, 0.0f);
    gate[0] = 0.0f; gate[1] = 1.0f; gate[2] = 2.0f;
    Tensor up({3}, 0.0f);
    up[0] = 1.0f; up[1] = 2.0f; up[2] = 3.0f;
    Tensor y = swiglu(gate, up);

    // silu(0) = 0, so swiglu(0, 1) = 0
    ASSERT_NEAR(y[0], 0.0f, 1e-5f);
    // silu(1) ≈ 0.731, so swiglu(1, 2) ≈ 0.731 * 2 ≈ 1.462
    ASSERT_NEAR(y[1], 0.7310586f * 2.0f, 1e-5f);
    // silu(2) ≈ 1.761, so swiglu(2, 3) ≈ 1.761 * 3 ≈ 5.284
    ASSERT_NEAR(y[2], 1.7615942f * 3.0f, 1e-4f);
    return true;
}

static bool test_swiglu_shape_mismatch() {
    Tensor a({3}, 0.0f);
    Tensor b({4}, 0.0f);
    try {
        swiglu(a, b);
        ASSERT_FAIL("expected exception for swiglu shape mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ==========================================================================
// elementwise_mul
// ==========================================================================
static bool test_elementwise_mul() {
    Tensor a({3}, 0.0f);
    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f;
    Tensor b({3}, 0.0f);
    b[0] = 4.0f; b[1] = 5.0f; b[2] = 6.0f;
    Tensor y = elementwise_mul(a, b);
    ASSERT_NEAR(y[0], 4.0f, 1e-5f);
    ASSERT_NEAR(y[1], 10.0f, 1e-5f);
    ASSERT_NEAR(y[2], 18.0f, 1e-5f);
    return true;
}

static bool test_elementwise_mul_shape_mismatch() {
    Tensor a({3}, 0.0f);
    Tensor b({4}, 0.0f);
    try {
        elementwise_mul(a, b);
        ASSERT_FAIL("expected exception for elementwise_mul shape mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ==========================================================================
// argmax
// ==========================================================================
static bool test_argmax() {
    Tensor x({5}, 0.0f);
    x[0] = 1.0f; x[1] = 5.0f; x[2] = 3.0f; x[3] = 5.0f; x[4] = 2.0f;
    int idx = argmax(x);
    ASSERT_EQ(idx, 1);
    return true;
}

static bool test_argmax_empty() {
    Tensor x;
    try {
        argmax(x);
        ASSERT_FAIL("expected exception for argmax on empty tensor");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ==========================================================================
// rope
// ==========================================================================
static bool test_rope_preserves_norm() {
    int n_heads = 2;
    int head_dim = 4;
    Tensor q({n_heads, head_dim}, 1.0f);
    Tensor k({n_heads, head_dim}, 1.0f);
    rope(q, k, 5, 10000.0f);
    for (int h = 0; h < n_heads; ++h) {
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            ASSERT_TRUE(std::isfinite(q.at2(h, d)));
            ASSERT_TRUE(std::isfinite(k.at2(h, d)));
            q_norm += q.at2(h, d) * q.at2(h, d);
            k_norm += k.at2(h, d) * k.at2(h, d);
        }
        ASSERT_NEAR(q_norm, 4.0f, 1e-5f);
        ASSERT_NEAR(k_norm, 4.0f, 1e-5f);
    }
    return true;
}

static bool test_rope_neox_matches_split_half_pairs() {
    Tensor q({1, 4}, 0.0f);
    Tensor k({1, 4}, 0.0f);
    q[0] = 1.0f;
    q[1] = 2.0f;
    q[2] = 3.0f;
    q[3] = 4.0f;
    k = q;

    rope(q, k, 1, 10000.0f, RopeType::NeoX);

    float c0 = std::cos(1.0f);
    float s0 = std::sin(1.0f);
    float c1 = std::cos(0.01f);
    float s1 = std::sin(0.01f);

    ASSERT_NEAR(q[0], 1.0f * c0 - 3.0f * s0, 1e-5f);
    ASSERT_NEAR(q[2], 1.0f * s0 + 3.0f * c0, 1e-5f);
    ASSERT_NEAR(q[1], 2.0f * c1 - 4.0f * s1, 1e-5f);
    ASSERT_NEAR(q[3], 2.0f * s1 + 4.0f * c1, 1e-5f);
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(k[i], q[i], 1e-5f);
    }
    return true;
}

static bool test_rope_wrong_dim() {
    Tensor q({4}, 0.0f);
    Tensor k({2, 4}, 0.0f);
    try {
        rope(q, k, 0, 10000.0f);
        ASSERT_FAIL("expected exception for rope with 1D q");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_rope_rejects_mismatched_head_dim() {
    Tensor q({2, 4}, 0.0f);
    Tensor k({2, 6}, 0.0f);
    try {
        rope(q, k, 0, 10000.0f);
        ASSERT_FAIL("expected exception for mismatched RoPE head_dim");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_rope_rejects_odd_head_dim() {
    Tensor q({2, 3}, 0.0f);
    Tensor k({2, 3}, 0.0f);
    try {
        rope(q, k, 0, 10000.0f);
        ASSERT_FAIL("expected exception for odd RoPE head_dim");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_rope_rejects_negative_position() {
    Tensor q({2, 4}, 0.0f);
    Tensor k({2, 4}, 0.0f);
    try {
        rope(q, k, -1, 10000.0f);
        ASSERT_FAIL("expected exception for negative RoPE position");
    } catch (const std::out_of_range&) {
        // expected
    }
    return true;
}

static bool test_rope_rejects_nonpositive_theta() {
    Tensor q({2, 4}, 0.0f);
    Tensor k({2, 4}, 0.0f);
    try {
        rope(q, k, 0, 0.0f);
        ASSERT_FAIL("expected exception for nonpositive RoPE theta");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static struct OpsTestRegistrar {
    OpsTestRegistrar() {
        register_test("matmul_identity", test_matmul_identity);
        register_test("matmul_simple", test_matmul_simple);
        register_test("matmul_rectangular", test_matmul_rectangular);
        register_test("matmul_shape_mismatch", test_matmul_shape_mismatch);
        register_test("linear_1d", test_linear_1d);
        register_test("linear_2d", test_linear_2d);
        register_test("linear_shape_mismatch", test_linear_shape_mismatch);
        register_test("rmsnorm_unit", test_rmsnorm_unit);
        register_test("rmsnorm_scaled", test_rmsnorm_scaled);
        register_test("rmsnorm_with_weight", test_rmsnorm_with_weight);
        register_test("rmsnorm_shape_mismatch", test_rmsnorm_shape_mismatch);
        register_test("softmax", test_softmax);
        register_test("softmax_uniform", test_softmax_uniform);
        register_test("softmax_wrong_dim", test_softmax_wrong_dim);
        register_test("softmax_empty", test_softmax_empty);
        register_test("silu", test_silu);
        register_test("swiglu", test_swiglu);
        register_test("swiglu_shape_mismatch", test_swiglu_shape_mismatch);
        register_test("elementwise_mul", test_elementwise_mul);
        register_test("elementwise_mul_shape_mismatch", test_elementwise_mul_shape_mismatch);
        register_test("argmax", test_argmax);
        register_test("argmax_empty", test_argmax_empty);
        register_test("rope_preserves_norm", test_rope_preserves_norm);
        register_test("rope_neox_matches_split_half_pairs", test_rope_neox_matches_split_half_pairs);
        register_test("rope_wrong_dim", test_rope_wrong_dim);
        register_test("rope_rejects_mismatched_head_dim", test_rope_rejects_mismatched_head_dim);
        register_test("rope_rejects_odd_head_dim", test_rope_rejects_odd_head_dim);
        register_test("rope_rejects_negative_position", test_rope_rejects_negative_position);
        register_test("rope_rejects_nonpositive_theta", test_rope_rejects_nonpositive_theta);
    }
} ops_test_registrar;
