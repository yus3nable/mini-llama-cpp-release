#include "mini_llama/gguf_loader.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "test_main.h"
#include "gguf_loader_fixture.h"

#include <filesystem>

using namespace mini_llama;

// ---------------------------------------------------------------------------
// Load a generated tiny GGUF model and verify config + tensor shapes.
// ---------------------------------------------------------------------------
static bool test_gguf_loader_loads_tiny_fixture() {
    std::filesystem::path path = test::write_tiny_llama_gguf_fixture("tiny_loader_fixture.gguf");
    MiniLlamaModel model = load_gguf_model(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(model.loaded);
    ASSERT_TRUE(model.load_error.empty());

    ASSERT_EQ(model.config.vocab_size, 128);
    ASSERT_EQ(model.config.dim, 32);
    ASSERT_EQ(model.config.n_layers, 2);
    ASSERT_EQ(model.config.n_heads, 4);
    ASSERT_EQ(model.config.n_kv_heads, 4);
    ASSERT_EQ(model.config.head_dim, 8);
    ASSERT_EQ(model.config.hidden_dim, 86);
    ASSERT_EQ(model.config.max_seq_len, 128);
    ASSERT_NEAR(model.config.rms_norm_eps, 1e-5f, 1e-7f);

    ASSERT_TRUE(model.token_embedding.data.size() > 0);
    ASSERT_EQ(model.token_embedding.shape.size(), 2);
    ASSERT_EQ(model.token_embedding.shape[0], 128);
    ASSERT_EQ(model.token_embedding.shape[1], 32);

    ASSERT_EQ(static_cast<int>(model.layers.size()), 2);
    for (int i = 0; i < 2; ++i) {
        const LayerWeights& lw = model.layers[i];
        ASSERT_TRUE(lw.attention_norm.data.size() > 0);
        ASSERT_EQ(lw.attention_norm.shape[0], 32);

        ASSERT_TRUE(lw.wq.f32_data.size() > 0);
        ASSERT_EQ(lw.wq.shape, std::vector<int>({32, 32}));

        ASSERT_TRUE(lw.wk.f32_data.size() > 0);
        ASSERT_EQ(lw.wk.shape, std::vector<int>({32, 32}));
        ASSERT_TRUE(lw.wv.f32_data.size() > 0);
        ASSERT_EQ(lw.wv.shape, std::vector<int>({32, 32}));
        ASSERT_TRUE(lw.wo.f32_data.size() > 0);
        ASSERT_EQ(lw.wo.shape, std::vector<int>({32, 32}));

        ASSERT_TRUE(lw.ffn_norm.data.size() > 0);
        ASSERT_EQ(lw.ffn_norm.shape[0], 32);

        ASSERT_TRUE(lw.w_gate.f32_data.size() > 0);
        ASSERT_EQ(lw.w_gate.shape, std::vector<int>({86, 32}));
        ASSERT_TRUE(lw.w_up.f32_data.size() > 0);
        ASSERT_EQ(lw.w_up.shape, std::vector<int>({86, 32}));
        ASSERT_TRUE(lw.w_down.f32_data.size() > 0);
        ASSERT_EQ(lw.w_down.shape, std::vector<int>({32, 86}));
    }

    ASSERT_TRUE(model.final_norm.data.size() > 0);
    ASSERT_EQ(model.final_norm.shape[0], 32);

    ASSERT_TRUE(model.lm_head.f32_data.size() > 0);
    ASSERT_EQ(model.lm_head.shape.size(), 2);
    ASSERT_EQ(model.lm_head.shape[0], 128);
    ASSERT_EQ(model.lm_head.shape[1], 32);

    return true;
}

// ---------------------------------------------------------------------------
// The generated GGUF model must enter the normal forward path.
// ---------------------------------------------------------------------------
static bool test_gguf_loader_fixture_enters_forward_path() {
    std::filesystem::path path = test::write_tiny_llama_gguf_fixture("tiny_loader_forward.gguf");
    MiniLlamaModel model = load_gguf_model(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(model.loaded);
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;
    Tensor logits = forward_token(model, ctx, 1);
    ASSERT_EQ(logits.shape, std::vector<int>({128}));
    return true;
}

// ---------------------------------------------------------------------------
// Load the local demo GGUF when it exists. This keeps the large-file check
// useful on the development machine while the tiny fixture covers clean clones.
// ---------------------------------------------------------------------------
static bool test_gguf_loader_loads_real_model_when_available() {
    const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        return true;
    }

    MiniLlamaModel model = load_gguf_model(path);
    ASSERT_TRUE(model.loaded);
    ASSERT_TRUE(model.load_error.empty());
    ASSERT_EQ(model.config.vocab_size, 151936);
    ASSERT_EQ(model.config.dim, 896);
    ASSERT_EQ(model.config.n_layers, 24);
    ASSERT_EQ(model.config.n_heads, 14);
    ASSERT_EQ(model.config.n_kv_heads, 2);
    ASSERT_EQ(model.config.head_dim, 64);
    ASSERT_EQ(model.config.hidden_dim, 4864);
    ASSERT_EQ(model.config.max_seq_len, 32768);
    ASSERT_NEAR(model.config.rms_norm_eps, 1e-6f, 1e-7f);
    return true;
}

// ---------------------------------------------------------------------------
// Error handling: missing file.
// ---------------------------------------------------------------------------
static bool test_gguf_loader_missing_file() {
    MiniLlamaModel model = load_gguf_model("models/chat/nonexistent.gguf");
    ASSERT_TRUE(!model.loaded);
    ASSERT_TRUE(!model.load_error.empty());
    return true;
}

// ---------------------------------------------------------------------------
// Error handling: non-GGUF file.
// ---------------------------------------------------------------------------
static bool test_gguf_loader_invalid_file() {
    MiniLlamaModel model = load_gguf_model("models/tiny/model.json");
    ASSERT_TRUE(!model.loaded);
    ASSERT_TRUE(!model.load_error.empty());
    return true;
}

// ---------------------------------------------------------------------------
// The tiny test.gguf lacks config metadata, so load_gguf_model should fail
// gracefully during config validation.
// ---------------------------------------------------------------------------
static bool test_gguf_loader_rejects_incomplete_config() {
    MiniLlamaModel model = load_gguf_model("models/tiny/test.gguf");
    ASSERT_TRUE(!model.loaded);
    ASSERT_TRUE(!model.load_error.empty());
    return true;
}

static struct GGUFLloaderTestRegistrar {
    GGUFLloaderTestRegistrar() {
        register_test("gguf_loader_loads_tiny_fixture", test_gguf_loader_loads_tiny_fixture);
        register_test("gguf_loader_fixture_enters_forward_path", test_gguf_loader_fixture_enters_forward_path);
        register_test("gguf_loader_loads_real_model_when_available", test_gguf_loader_loads_real_model_when_available);
        register_test("gguf_loader_missing_file", test_gguf_loader_missing_file);
        register_test("gguf_loader_invalid_file", test_gguf_loader_invalid_file);
        register_test("gguf_loader_rejects_incomplete_config", test_gguf_loader_rejects_incomplete_config);
    }
} gguf_loader_test_registrar;
