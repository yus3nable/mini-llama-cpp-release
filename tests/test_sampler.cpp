#include "mini_llama/sampler.h"
#include "mini_llama/tensor.h"
#include "test_main.h"
#include <limits>
#include <stdexcept>

using namespace mini_llama;

// ---------------------------------------------------------------------------
// Greedy sampling
// ---------------------------------------------------------------------------
static bool test_sampler_greedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 5.0f;
    logits[2] = 3.0f;
    logits[3] = 5.0f;  // same max as index 1
    logits[4] = 2.0f;

    int idx = MiniSampler::sample_greedy(logits);
    ASSERT_EQ(idx, 1);  // first occurrence of max
    return true;
}

// ---------------------------------------------------------------------------
// temperature = 0 is equivalent to greedy
// ---------------------------------------------------------------------------
static bool test_sampler_temperature_zero_is_greedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 0.0f;
    params.top_k = 0;

    // Run many times; should always pick the same token (greedy).
    int expected = MiniSampler::sample_greedy(logits);
    for (int i = 0; i < 20; ++i) {
        int idx = sampler.sample(logits, params);
        ASSERT_EQ(idx, expected);
    }
    return true;
}

// ---------------------------------------------------------------------------
// top_k = 1 is equivalent to greedy
// ---------------------------------------------------------------------------
static bool test_sampler_top_k_one_is_greedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 1;

    int expected = MiniSampler::sample_greedy(logits);
    for (int i = 0; i < 20; ++i) {
        int idx = sampler.sample(logits, params);
        ASSERT_EQ(idx, expected);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fixed seed gives reproducible temperature sampling
// ---------------------------------------------------------------------------
static bool test_sampler_temperature_reproducible() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.5f;
    }

    MiniSampler sampler1(123);
    MiniSampler sampler2(123);
    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 0;

    // Same seed should produce identical sequences.
    for (int i = 0; i < 10; ++i) {
        int a = sampler1.sample(logits, params);
        int b = sampler2.sample(logits, params);
        ASSERT_EQ(a, b);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fixed seed gives reproducible top-k sampling
// ---------------------------------------------------------------------------
static bool test_sampler_top_k_reproducible() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.5f;
    }

    MiniSampler sampler1(456);
    MiniSampler sampler2(456);
    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 3;

    for (int i = 0; i < 10; ++i) {
        int a = sampler1.sample(logits, params);
        int b = sampler2.sample(logits, params);
        ASSERT_EQ(a, b);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Different seeds give different sequences (with high probability)
// ---------------------------------------------------------------------------
static bool test_sampler_different_seeds_differ() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i) * 0.3f;
    }

    MiniSampler sampler1(1);
    MiniSampler sampler2(2);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 0;

    int same_count = 0;
    for (int i = 0; i < 20; ++i) {
        int a = sampler1.sample(logits, params);
        int b = sampler2.sample(logits, params);
        if (a == b) {
            ++same_count;
        }
    }

    // With different seeds and temperature=1.0, it's extremely unlikely
    // that all 20 samples match. Allow up to 18 matches as a loose bound.
    ASSERT_TRUE(same_count < 18);
    return true;
}

// ---------------------------------------------------------------------------
// Top-k only samples from the top-k set
// ---------------------------------------------------------------------------
static bool test_sampler_top_k_limits_candidates() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits[i] = static_cast<float>(i);
    }

    MiniSampler sampler(789);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 2;

    // With top_k=2 on logits [0..9], only tokens 8 and 9 should ever be sampled.
    for (int i = 0; i < 50; ++i) {
        int idx = sampler.sample(logits, params);
        ASSERT_TRUE(idx == 8 || idx == 9);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Legacy free function still works
// ---------------------------------------------------------------------------
static bool test_sampler_legacy_greedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 5.0f;
    logits[2] = 3.0f;
    logits[3] = 4.0f;
    logits[4] = 2.0f;

    int idx = sample_greedy(logits);
    ASSERT_EQ(idx, 1);
    return true;
}

// ---------------------------------------------------------------------------
// Default params preserve greedy behavior
// ---------------------------------------------------------------------------
static bool test_sampler_default_params_are_greedy() {
    Tensor logits({5}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 2.0f;
    logits[2] = 1.0f;
    logits[3] = 0.5f;
    logits[4] = 0.2f;

    MiniSampler sampler(42);
    SamplingParams params;

    int idx = sampler.sample(logits, params);
    ASSERT_EQ(idx, MiniSampler::sample_greedy(logits));
    return true;
}

// ---------------------------------------------------------------------------
// Invalid parameters fail clearly
// ---------------------------------------------------------------------------
static bool test_sampler_rejects_negative_top_k() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 0.2f;
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = -1;

    try {
        sampler.sample(logits, params);
        ASSERT_FAIL("expected exception for negative top_k");
    } catch (const std::runtime_error&) {
        return true;
    }
}

static bool test_sampler_rejects_nan_temperature() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 0.2f;
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = std::numeric_limits<float>::quiet_NaN();
    params.top_k = 0;

    try {
        sampler.sample(logits, params);
        ASSERT_FAIL("expected exception for NaN temperature");
    } catch (const std::runtime_error&) {
        return true;
    }
}

static bool test_sampler_rejects_empty_logits() {
    Tensor logits;
    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 0.0f;

    try {
        sampler.sample(logits, params);
        ASSERT_FAIL("expected exception for empty logits");
    } catch (const std::runtime_error&) {
        return true;
    }
}

static bool test_sampler_rejects_nonfinite_logits() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = std::numeric_limits<float>::infinity();
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    SamplingParams params;
    params.temperature = 1.0f;

    try {
        sampler.sample(logits, params);
        ASSERT_FAIL("expected exception for non-finite logits");
    } catch (const std::runtime_error&) {
        return true;
    }
}

static bool test_sampler_top_k_larger_than_vocab_is_clamped() {
    Tensor logits({4}, 0.0f);
    logits[0] = 0.0f;
    logits[1] = 1.0f;
    logits[2] = 2.0f;
    logits[3] = 3.0f;

    MiniSampler sampler(42);
    for (int i = 0; i < 20; ++i) {
        int idx = sampler.sample_top_k(logits, 1.0f, 100);
        ASSERT_TRUE(idx >= 0 && idx < 4);
    }
    return true;
}

static bool test_sampler_direct_top_k_zero_rejected() {
    Tensor logits({3}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 0.2f;
    logits[2] = 0.3f;

    MiniSampler sampler(42);
    try {
        sampler.sample_top_k(logits, 1.0f, 0);
        ASSERT_FAIL("expected exception for direct top_k zero");
    } catch (const std::runtime_error&) {
        return true;
    }
}

// ---------------------------------------------------------------------------
// Top-k keeps exactly k candidates even when logits tie
// ---------------------------------------------------------------------------
static bool test_sampler_top_k_ties_keep_exact_k() {
    Tensor logits({5}, 0.0f);
    logits[0] = 5.0f;
    logits[1] = 5.0f;
    logits[2] = 5.0f;
    logits[3] = 1.0f;
    logits[4] = 0.0f;

    MiniSampler sampler(321);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 2;

    for (int i = 0; i < 40; ++i) {
        int idx = sampler.sample(logits, params);
        ASSERT_TRUE(idx == 0 || idx == 1);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct SamplerTestRegistrar {
    SamplerTestRegistrar() {
        register_test("sampler_greedy", test_sampler_greedy);
        register_test("sampler_temperature_zero_is_greedy", test_sampler_temperature_zero_is_greedy);
        register_test("sampler_top_k_one_is_greedy", test_sampler_top_k_one_is_greedy);
        register_test("sampler_temperature_reproducible", test_sampler_temperature_reproducible);
        register_test("sampler_top_k_reproducible", test_sampler_top_k_reproducible);
        register_test("sampler_different_seeds_differ", test_sampler_different_seeds_differ);
        register_test("sampler_top_k_limits_candidates", test_sampler_top_k_limits_candidates);
        register_test("sampler_legacy_greedy", test_sampler_legacy_greedy);
        register_test("sampler_default_params_are_greedy", test_sampler_default_params_are_greedy);
        register_test("sampler_rejects_negative_top_k", test_sampler_rejects_negative_top_k);
        register_test("sampler_rejects_nan_temperature", test_sampler_rejects_nan_temperature);
        register_test("sampler_rejects_empty_logits", test_sampler_rejects_empty_logits);
        register_test("sampler_rejects_nonfinite_logits", test_sampler_rejects_nonfinite_logits);
        register_test("sampler_top_k_larger_than_vocab_is_clamped", test_sampler_top_k_larger_than_vocab_is_clamped);
        register_test("sampler_direct_top_k_zero_rejected", test_sampler_direct_top_k_zero_rejected);
        register_test("sampler_top_k_ties_keep_exact_k", test_sampler_top_k_ties_keep_exact_k);
    }
} sampler_test_registrar;
