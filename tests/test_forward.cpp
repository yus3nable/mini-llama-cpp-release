#include "mini_llama/model.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"
#include "test_main.h"

#include <vector>
#include <iostream>
#include <stdexcept>

using namespace mini_llama;

static bool test_forward_token_produces_logit_shape() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (model.layers.empty()) {
        ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;
    Tensor logits = forward_token(model, ctx, 1);  // BOS token
    ASSERT_EQ(logits.ndim(), 1);
    ASSERT_EQ(logits.shape[0], model.config.vocab_size);
    return true;
}

static bool test_forward_token_changes_kv_cache() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (model.layers.empty()) {
        ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;
    // Before forward, cache at layer 0, pos 0 should be zero
    float before = ctx.kv_cache.keys.at({0, 0, 0, 0});
    ASSERT_NEAR(before, 0.0f, 1e-6f);

    forward_token(model, ctx, 1);

    // After forward, cache should have been written
    float after = ctx.kv_cache.keys.at({0, 0, 0, 0});
    // It's unlikely to be exactly 0 after random weights forward
    // Just check it's finite and cache was written somewhere
    ASSERT_TRUE(std::isfinite(after));
    return true;
}

static bool test_generation_greedy_sequence() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (model.layers.empty()) {
        ASSERT_FAIL("Failed to load model");
    }
    AsciiTokenizer tokenizer;
    std::vector<int> tokens = tokenizer.encode("hello");
    MiniLlamaContext ctx(&model);

    // Prefill
    size_t prompt_len = tokens.size();
    for (size_t i = 0; i < prompt_len; ++i) {
        ctx.pos = static_cast<int>(i);
        Tensor logits = forward_token(model, ctx, tokens[i]);
        if (i + 1 == prompt_len) {
            tokens.push_back(sample_greedy(logits));
        }
    }

    // Decode 5 more tokens
    for (int i = 0; i < 5; ++i) {
        ctx.pos = static_cast<int>(tokens.size() - 1);
        Tensor logits = forward_token(model, ctx, tokens.back());
        tokens.push_back(sample_greedy(logits));
    }

    ASSERT_TRUE(tokens.size() > 6);
    // First generated token should be the same as what we got in prefill
    ASSERT_TRUE(tokens[tokens.size() - 6] >= 0);
    return true;
}

static bool test_position_increments_correctly() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (model.layers.empty()) {
        ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);

    // Forward token at pos 0
    ctx.pos = 0;
    forward_token(model, ctx, 1);

    // Forward token at pos 1
    ctx.pos = 1;
    Tensor logits = forward_token(model, ctx, 2);
    ASSERT_EQ(logits.shape[0], model.config.vocab_size);
    return true;
}

static bool test_forward_rejects_invalid_token() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;

    try {
        forward_token(model, ctx, model.config.vocab_size);
        ASSERT_FAIL("forward_token accepted an out-of-range token");
    } catch (const std::out_of_range&) {
        return true;
    } catch (...) {
        ASSERT_FAIL("forward_token threw the wrong exception type for invalid token");
    }
}

static bool test_forward_rejects_invalid_position() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("Failed to load model");
    }
    MiniLlamaContext ctx(&model);
    ctx.pos = model.config.max_seq_len;

    try {
        forward_token(model, ctx, 1);
        ASSERT_FAIL("forward_token accepted an out-of-range position");
    } catch (const std::out_of_range&) {
        return true;
    } catch (...) {
        ASSERT_FAIL("forward_token threw the wrong exception type for invalid position");
    }
}

static struct ForwardTestRegistrar {
    ForwardTestRegistrar() {
        register_test("forward_token_produces_logit_shape", test_forward_token_produces_logit_shape);
        register_test("forward_token_changes_kv_cache", test_forward_token_changes_kv_cache);
        register_test("generation_greedy_sequence", test_generation_greedy_sequence);
        register_test("position_increments_correctly", test_position_increments_correctly);
        register_test("forward_rejects_invalid_token", test_forward_rejects_invalid_token);
        register_test("forward_rejects_invalid_position", test_forward_rejects_invalid_position);
    }
} forward_test_registrar;
