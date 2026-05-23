#include "test_main.h"
#include "mini_llama/debug.h"
#include "mini_llama/tensor.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include "mini_llama/loader.h"
#include "mini_llama/tokenizer.h"

using namespace mini_llama;

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------
static bool test_benchmark_result_zero() {
    BenchmarkResult r;
    ASSERT_EQ(r.n_prompt_tokens, 0);
    ASSERT_EQ(r.n_generated_tokens, 0);
    ASSERT_EQ(r.n_decode_tokens, 0);
    ASSERT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
    ASSERT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
    return true;
}

static bool test_benchmark_result_tokens_per_sec() {
    BenchmarkResult r;
    r.n_prompt_tokens = 10;
    r.n_generated_tokens = 100;
    r.n_decode_tokens = 100;
    r.prefill_ms = 50.0;
    r.decode_ms = 450.0;
    ASSERT_NEAR(r.tokens_per_sec(), 200.0, 1e-6);  // 100 / 0.5
    ASSERT_NEAR(r.decode_tokens_per_sec(), 1000.0 / 4.5, 1e-6);
    return true;
}

static bool test_benchmark_result_no_decode() {
    BenchmarkResult r;
    r.n_generated_tokens = 0;
    r.n_decode_tokens = 0;
    r.prefill_ms = 100.0;
    r.decode_ms = 0.0;
    ASSERT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
    ASSERT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
    return true;
}

// ---------------------------------------------------------------------------
// dump_tensor_shape (smoke test: just verify no crash)
// ---------------------------------------------------------------------------
static bool test_dump_tensor_shape() {
    Tensor t({2, 3, 4}, 1.0f);
    dump_tensor_shape(t, "test_tensor");
    return true;
}

// ---------------------------------------------------------------------------
// dump_logits_topk (smoke test)
// ---------------------------------------------------------------------------
static bool test_dump_logits_topk() {
    Tensor logits({10}, 0.0f);
    for (int i = 0; i < 10; ++i) {
        logits.data[i] = static_cast<float>(i);
    }
    dump_logits_topk(logits, 3);
    return true;
}

static bool test_dump_logits_topk_k_larger_than_vocab() {
    Tensor logits({5}, 0.0f);
    for (int i = 0; i < 5; ++i) {
        logits.data[i] = static_cast<float>(i);
    }
    dump_logits_topk(logits, 10);  // k > vocab_size
    return true;
}

// ---------------------------------------------------------------------------
// dump_kv_cache_info (smoke test)
// ---------------------------------------------------------------------------
static bool test_dump_kv_cache_info() {
    KVCache cache(2, 16, 4, 8);
    dump_kv_cache_info(cache);
    return true;
}

// ---------------------------------------------------------------------------
// run_benchmark (integration smoke test)
// ---------------------------------------------------------------------------
static bool test_run_benchmark_smoke() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    AsciiTokenizer tokenizer;
    std::vector<int> tokens = tokenizer.encode("hi");
    BenchmarkResult result = run_benchmark(model, tokens, 4, 42, false);

    ASSERT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
    ASSERT_EQ(result.n_generated_tokens, 4);
    ASSERT_EQ(result.n_decode_tokens, 3);
    ASSERT_TRUE(result.prefill_ms >= 0.0);
    ASSERT_TRUE(result.decode_ms >= 0.0);
    ASSERT_TRUE(result.tokens_per_sec() >= 0.0);
    return true;
}

static bool test_run_benchmark_zero_predict() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    AsciiTokenizer tokenizer;
    std::vector<int> tokens = tokenizer.encode("hi");
    BenchmarkResult result = run_benchmark(model, tokens, 0, 42, false);

    ASSERT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
    ASSERT_EQ(result.n_generated_tokens, 0);
    ASSERT_EQ(result.n_decode_tokens, 0);
    ASSERT_TRUE(result.prefill_ms >= 0.0);
    ASSERT_TRUE(result.decode_ms == 0.0);
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct DebugTestRegistrar {
    DebugTestRegistrar() {
        register_test("benchmark_result_zero", test_benchmark_result_zero);
        register_test("benchmark_result_tokens_per_sec", test_benchmark_result_tokens_per_sec);
        register_test("benchmark_result_no_decode", test_benchmark_result_no_decode);
        register_test("dump_tensor_shape", test_dump_tensor_shape);
        register_test("dump_logits_topk", test_dump_logits_topk);
        register_test("dump_logits_topk_k_larger_than_vocab", test_dump_logits_topk_k_larger_than_vocab);
        register_test("dump_kv_cache_info", test_dump_kv_cache_info);
        register_test("run_benchmark_smoke", test_run_benchmark_smoke);
        register_test("run_benchmark_zero_predict", test_run_benchmark_zero_predict);
    }
} debug_test_registrar;
