#include "mini_llama/tensor.h"
#include "test_main.h"

using namespace mini_llama;

// ---------------------------------------------------------------------------
// Basic construction and properties
// ---------------------------------------------------------------------------
static bool test_tensor_shape_and_size() {
    Tensor t1({3, 4, 5}, 0.0f);
    ASSERT_EQ(t1.ndim(), 3);
    ASSERT_EQ(t1.size(), 60);
    ASSERT_EQ(t1.numel(), 60);
    ASSERT_EQ(t1.shape[0], 3);
    ASSERT_EQ(t1.shape[1], 4);
    ASSERT_EQ(t1.shape[2], 5);
    return true;
}

static bool test_tensor_fill() {
    Tensor t({10}, 3.14f);
    ASSERT_EQ(t.size(), 10);
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_NEAR(t.data[i], 3.14f, 1e-6f);
    }
    return true;
}

static bool test_tensor_rejects_zero_dimension() {
    try {
        Tensor t({2, 0}, 0.0f);
        ASSERT_FAIL("expected exception for zero tensor dimension");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_tensor_rejects_negative_dimension() {
    try {
        Tensor t({2, -3}, 0.0f);
        ASSERT_FAIL("expected exception for negative tensor dimension");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_tensor_indexing() {
    Tensor t({2, 3}, 0.0f);
    t.at({0, 0}) = 1.0f;
    t.at({0, 1}) = 2.0f;
    t.at({1, 2}) = 6.0f;
    ASSERT_NEAR(t.at({0, 0}), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at({0, 1}), 2.0f, 1e-6f);
    ASSERT_NEAR(t.at({1, 2}), 6.0f, 1e-6f);
    return true;
}

static bool test_tensor_index_wrong_rank() {
    Tensor t({2, 3}, 0.0f);
    try {
        t.at({1});
        ASSERT_FAIL("expected exception for wrong index rank");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_tensor_index_negative() {
    Tensor t({2, 3}, 0.0f);
    try {
        t.at({-1, 0});
        ASSERT_FAIL("expected exception for negative index");
    } catch (const std::out_of_range&) {
        // expected
    }
    return true;
}

static bool test_tensor_index_too_large() {
    Tensor t({2, 3}, 0.0f);
    try {
        t.at({0, 3});
        ASSERT_FAIL("expected exception for index above shape bound");
    } catch (const std::out_of_range&) {
        // expected
    }
    return true;
}

static bool test_tensor_1d_access() {
    Tensor t({5}, 0.0f);
    t[2] = 42.0f;
    ASSERT_NEAR(t[2], 42.0f, 1e-6f);
    return true;
}

static bool test_make_tensor_helpers() {
    auto t1 = make_tensor_1d(10, 1.0f);
    ASSERT_EQ(t1.ndim(), 1);
    ASSERT_EQ(t1.shape[0], 10);

    auto t2 = make_tensor_2d(3, 4, 1.0f);
    ASSERT_EQ(t2.ndim(), 2);
    ASSERT_EQ(t2.shape[0], 3);
    ASSERT_EQ(t2.shape[1], 4);

    auto t3 = make_tensor_3d(2, 3, 4, 1.0f);
    ASSERT_EQ(t3.ndim(), 3);
    ASSERT_EQ(t3.size(), 24);

    auto t4 = make_tensor_4d(2, 3, 4, 5, 1.0f);
    ASSERT_EQ(t4.ndim(), 4);
    ASSERT_EQ(t4.size(), 120);
    return true;
}

// ---------------------------------------------------------------------------
// at1 / at2 / at3 / at4 convenience accessors
// ---------------------------------------------------------------------------
static bool test_tensor_at1() {
    Tensor t({5}, 0.0f);
    t.at1(2) = 7.0f;
    ASSERT_NEAR(t.at1(2), 7.0f, 1e-6f);
    return true;
}

static bool test_tensor_at1_wrong_dim() {
    Tensor t({2, 3}, 0.0f);
    try {
        t.at1(0);
        ASSERT_FAIL("expected exception for at1 on 2D tensor");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_tensor_at2() {
    Tensor t({2, 3}, 0.0f);
    t.at2(0, 1) = 5.0f;
    t.at2(1, 2) = 9.0f;
    ASSERT_NEAR(t.at2(0, 1), 5.0f, 1e-6f);
    ASSERT_NEAR(t.at2(1, 2), 9.0f, 1e-6f);
    return true;
}

static bool test_tensor_at2_out_of_range() {
    Tensor t({2, 3}, 0.0f);
    try {
        t.at2(2, 0);
        ASSERT_FAIL("expected exception for at2 row out of range");
    } catch (const std::out_of_range&) {
        // expected
    }
    return true;
}

static bool test_tensor_at3() {
    Tensor t({2, 2, 2}, 0.0f);
    t.at3(1, 0, 1) = 3.0f;
    ASSERT_NEAR(t.at3(1, 0, 1), 3.0f, 1e-6f);
    return true;
}

static bool test_tensor_at4() {
    Tensor t({2, 2, 2, 2}, 0.0f);
    t.at4(1, 1, 0, 0) = 4.0f;
    ASSERT_NEAR(t.at4(1, 1, 0, 0), 4.0f, 1e-6f);
    return true;
}

// ---------------------------------------------------------------------------
// row_ptr
// ---------------------------------------------------------------------------
static bool test_tensor_row_ptr() {
    Tensor t({3, 4}, 0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            t.at2(i, j) = static_cast<float>(i * 10 + j);
        }
    }
    const float* row1 = t.row_ptr(1);
    ASSERT_NEAR(row1[0], 10.0f, 1e-6f);
    ASSERT_NEAR(row1[3], 13.0f, 1e-6f);
    return true;
}

static bool test_tensor_row_ptr_wrong_dim() {
    Tensor t({5}, 0.0f);
    try {
        t.row_ptr(0);
        ASSERT_FAIL("expected exception for row_ptr on 1D tensor");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_tensor_row_ptr_out_of_range() {
    Tensor t({3, 4}, 0.0f);
    try {
        t.row_ptr(3);
        ASSERT_FAIL("expected exception for row_ptr row out of range");
    } catch (const std::out_of_range&) {
        // expected
    }
    return true;
}

// ---------------------------------------------------------------------------
// assert_shape
// ---------------------------------------------------------------------------
static bool test_tensor_assert_shape_pass() {
    Tensor t({2, 3, 4}, 0.0f);
    t.assert_shape({2, 3, 4}, "test_pass");
    return true;
}

static bool test_tensor_assert_shape_fail() {
    Tensor t({2, 3, 4}, 0.0f);
    try {
        t.assert_shape({2, 3, 5}, "test_fail");
        ASSERT_FAIL("expected exception for shape mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ---------------------------------------------------------------------------
// reshape_checked
// ---------------------------------------------------------------------------
static bool test_tensor_reshape_checked_pass() {
    Tensor t({2, 3}, 0.0f);
    for (int i = 0; i < 6; ++i) {
        t[i] = static_cast<float>(i);
    }
    Tensor r = t.reshape_checked({3, 2}, "test_reshape");
    ASSERT_EQ(r.ndim(), 2);
    ASSERT_EQ(r.shape[0], 3);
    ASSERT_EQ(r.shape[1], 2);
    ASSERT_NEAR(r.at2(0, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(r.at2(2, 1), 5.0f, 1e-6f);
    return true;
}

static bool test_tensor_reshape_checked_fail() {
    Tensor t({2, 3}, 0.0f);
    try {
        t.reshape_checked({4, 2}, "test_reshape_fail");
        ASSERT_FAIL("expected exception for incompatible reshape");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_tensor_reshape_checked_negative_dim() {
    Tensor t({2, 3}, 0.0f);
    try {
        t.reshape_checked({-1, 6}, "test_reshape_neg");
        ASSERT_FAIL("expected exception for negative dimension");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ---------------------------------------------------------------------------
// shape_string alias
// ---------------------------------------------------------------------------
static bool test_tensor_shape_string_alias() {
    Tensor t({2, 3, 4}, 0.0f);
    ASSERT_TRUE(t.shape_str() == t.shape_string());
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct TensorTestRegistrar {
    TensorTestRegistrar() {
        register_test("tensor_shape_and_size", test_tensor_shape_and_size);
        register_test("tensor_fill", test_tensor_fill);
        register_test("tensor_rejects_zero_dimension", test_tensor_rejects_zero_dimension);
        register_test("tensor_rejects_negative_dimension", test_tensor_rejects_negative_dimension);
        register_test("tensor_indexing", test_tensor_indexing);
        register_test("tensor_index_wrong_rank", test_tensor_index_wrong_rank);
        register_test("tensor_index_negative", test_tensor_index_negative);
        register_test("tensor_index_too_large", test_tensor_index_too_large);
        register_test("tensor_1d_access", test_tensor_1d_access);
        register_test("make_tensor_helpers", test_make_tensor_helpers);
        register_test("tensor_at1", test_tensor_at1);
        register_test("tensor_at1_wrong_dim", test_tensor_at1_wrong_dim);
        register_test("tensor_at2", test_tensor_at2);
        register_test("tensor_at2_out_of_range", test_tensor_at2_out_of_range);
        register_test("tensor_at3", test_tensor_at3);
        register_test("tensor_at4", test_tensor_at4);
        register_test("tensor_row_ptr", test_tensor_row_ptr);
        register_test("tensor_row_ptr_wrong_dim", test_tensor_row_ptr_wrong_dim);
        register_test("tensor_row_ptr_out_of_range", test_tensor_row_ptr_out_of_range);
        register_test("tensor_assert_shape_pass", test_tensor_assert_shape_pass);
        register_test("tensor_assert_shape_fail", test_tensor_assert_shape_fail);
        register_test("tensor_reshape_checked_pass", test_tensor_reshape_checked_pass);
        register_test("tensor_reshape_checked_fail", test_tensor_reshape_checked_fail);
        register_test("tensor_reshape_checked_negative_dim", test_tensor_reshape_checked_negative_dim);
        register_test("tensor_shape_string_alias", test_tensor_shape_string_alias);
    }
} tensor_test_registrar;
