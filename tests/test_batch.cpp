#include "test_main.h"
#include "mini_llama/batch.h"
#include "mini_llama/model.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"

using namespace mini_llama;

// ---------------------------------------------------------------------------
// MiniBatch construction
// ---------------------------------------------------------------------------
static bool test_batch_single() {
    MiniBatch b = MiniBatch::single(42, 5);
    ASSERT_EQ(b.n_tokens(), 1);
    ASSERT_EQ(b.tokens[0], 42);
    ASSERT_EQ(b.positions[0], 5);
    return true;
}

static bool test_batch_from_tokens() {
    std::vector<int> toks = {10, 20, 30};
    MiniBatch b = MiniBatch::from_tokens(toks, 0);
    ASSERT_EQ(b.n_tokens(), 3);
    ASSERT_EQ(b.tokens[0], 10);
    ASSERT_EQ(b.tokens[1], 20);
    ASSERT_EQ(b.tokens[2], 30);
    ASSERT_EQ(b.positions[0], 0);
    ASSERT_EQ(b.positions[1], 1);
    ASSERT_EQ(b.positions[2], 2);
    return true;
}

static bool test_batch_from_tokens_with_offset() {
    std::vector<int> toks = {5, 6};
    MiniBatch b = MiniBatch::from_tokens(toks, 10);
    ASSERT_EQ(b.positions[0], 10);
    ASSERT_EQ(b.positions[1], 11);
    return true;
}

static bool test_batch_empty() {
    MiniBatch b;
    ASSERT_EQ(b.n_tokens(), 0);
    return true;
}

// ---------------------------------------------------------------------------
// forward_batch
// ---------------------------------------------------------------------------
static bool test_forward_batch_prefill_matches_individual() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    std::vector<int> tokens = {1, 104, 101, 108, 108, 111};

    // Individual forward
    MiniLlamaContext ctx1(&model);
    Tensor logits1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        ctx1.pos = static_cast<int>(i);
        logits1 = forward_token(model, ctx1, tokens[i]);
    }

    // Batch forward
    MiniLlamaContext ctx2(&model);
    MiniBatch batch = MiniBatch::from_tokens(tokens, 0);
    Tensor logits2 = forward_batch(model, ctx2, batch);

    ASSERT_EQ(logits1.ndim(), 1);
    ASSERT_EQ(logits2.ndim(), 1);
    ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
    for (size_t i = 0; i < logits1.size(); ++i) {
        ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
    }
    return true;
}

static bool test_forward_batch_single_matches_individual() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    // Individual forward
    MiniLlamaContext ctx1(&model);
    ctx1.pos = 3;
    Tensor logits1 = forward_token(model, ctx1, 42);

    // Batch forward with single token
    MiniLlamaContext ctx2(&model);
    MiniBatch batch = MiniBatch::single(42, 3);
    Tensor logits2 = forward_batch(model, ctx2, batch);

    ASSERT_EQ(logits1.ndim(), 1);
    ASSERT_EQ(logits2.ndim(), 1);
    ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
    for (size_t i = 0; i < logits1.size(); ++i) {
        ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
    }
    return true;
}

static bool test_forward_batch_updates_context() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    std::vector<int> tokens = {1, 104, 101};
    MiniLlamaContext ctx(&model);
    MiniBatch batch = MiniBatch::from_tokens(tokens, 0);
    forward_batch(model, ctx, batch);

    ASSERT_EQ(ctx.token_history.size(), tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        ASSERT_EQ(ctx.token_history[i], tokens[i]);
    }
    ASSERT_EQ(ctx.pos, static_cast<int>(tokens.size()) - 1);
    return true;
}

static bool test_forward_batch_appends_decode_history() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch prefill = MiniBatch::from_tokens({1, 104}, 0);
    forward_batch(model, ctx, prefill);

    MiniBatch decode = MiniBatch::single(101, 2);
    forward_batch(model, ctx, decode);

    ASSERT_EQ(ctx.token_history.size(), 3);
    ASSERT_EQ(ctx.token_history[0], 1);
    ASSERT_EQ(ctx.token_history[1], 104);
    ASSERT_EQ(ctx.token_history[2], 101);
    ASSERT_EQ(ctx.pos, 2);
    return true;
}

static bool test_forward_batch_empty_rejected() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch batch;
    try {
        forward_batch(model, ctx, batch);
        ASSERT_FAIL("expected exception for empty batch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool test_forward_batch_mismatched_sizes_rejected() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch batch;
    batch.tokens = {1, 2};
    batch.positions = {0};
    try {
        forward_batch(model, ctx, batch);
        ASSERT_FAIL("expected exception for size mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct BatchTestRegistrar {
    BatchTestRegistrar() {
        register_test("batch_single", test_batch_single);
        register_test("batch_from_tokens", test_batch_from_tokens);
        register_test("batch_from_tokens_with_offset", test_batch_from_tokens_with_offset);
        register_test("batch_empty", test_batch_empty);
        register_test("forward_batch_prefill_matches_individual", test_forward_batch_prefill_matches_individual);
        register_test("forward_batch_single_matches_individual", test_forward_batch_single_matches_individual);
        register_test("forward_batch_updates_context", test_forward_batch_updates_context);
        register_test("forward_batch_appends_decode_history", test_forward_batch_appends_decode_history);
        register_test("forward_batch_empty_rejected", test_forward_batch_empty_rejected);
        register_test("forward_batch_mismatched_sizes_rejected", test_forward_batch_mismatched_sizes_rejected);
    }
} batch_test_registrar;
