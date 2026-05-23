#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/chat.h"
#include "test_main.h"
#include "gguf_tokenizer_fixture.h"

#include <cstdint>
#include <filesystem>
#include <iostream>

using namespace mini_llama;

// ---------------------------------------------------------------------------
// GgufTokenizer tests
// ---------------------------------------------------------------------------

static bool test_gguf_tokenizer_loads_from_gguf() {
    std::filesystem::path path = test::write_tiny_gguf_tokenizer_fixture("tiny_tokenizer_load.gguf");
    auto tok = create_gguf_tokenizer(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(tok != nullptr);
    ASSERT_EQ(tok->vocab_size(), 14);
    ASSERT_EQ(tok->bos_id(), 3);
    ASSERT_EQ(tok->eos_id(), 4);
    return true;
}

static bool test_gguf_tokenizer_encode_decode_roundtrip() {
    std::filesystem::path path = test::write_tiny_gguf_tokenizer_fixture("tiny_tokenizer_roundtrip.gguf");
    auto tok = create_gguf_tokenizer(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(tok != nullptr);

    std::string text = "ab";
    std::vector<int> ids = tok->encode(text);
    ASSERT_EQ(ids.size(), 1u);
    ASSERT_EQ(ids[0], 2);

    std::string decoded = tok->decode(ids);
    ASSERT_TRUE(decoded == text);
    return true;
}

static bool test_gguf_tokenizer_special_tokens() {
    std::filesystem::path path = test::write_tiny_gguf_tokenizer_fixture("tiny_tokenizer_special.gguf");
    auto tok = create_gguf_tokenizer(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(tok != nullptr);

    // <|im_start|> should be a single token
    std::vector<int> ids = tok->encode("<|im_start|>");
    ASSERT_EQ(ids.size(), 1u);
    ASSERT_EQ(ids[0], 5);

    // <|im_end|> should be a single token
    ids = tok->encode("<|im_end|>");
    ASSERT_EQ(ids.size(), 1u);
    ASSERT_EQ(ids[0], 6);

    // eos_id
    ASSERT_EQ(tok->eos_id(), 4);
    ASSERT_EQ(tok->decode_token(4), "</s>");

    return true;
}

static bool test_gguf_tokenizer_loads_real_model_when_available() {
    const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        return true;
    }

    auto tok = create_gguf_tokenizer(path);
    ASSERT_TRUE(tok != nullptr);
    ASSERT_EQ(tok->vocab_size(), 151936);
    ASSERT_EQ(tok->bos_id(), 151643);
    ASSERT_EQ(tok->eos_id(), 151645);
    ASSERT_EQ(tok->encode("<|im_start|>")[0], 151644);
    return true;
}

static bool test_gguf_tokenizer_matches_bpe_tokenizer() {
    if (!std::filesystem::exists("models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf") ||
        !std::filesystem::exists("models/chat/vocab.json") ||
        !std::filesystem::exists("models/chat/merges.txt")) {
        return true;
    }

    // Load both tokenizers and compare outputs
    auto gguf_tok = create_gguf_tokenizer("models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf");
    ASSERT_TRUE(gguf_tok != nullptr);

    auto bpe_tok = create_bpe_tokenizer(
        "models/chat/vocab.json",
        "models/chat/merges.txt",
        "models/chat/special_tokens.json"
    );
    ASSERT_TRUE(bpe_tok != nullptr);

    std::vector<std::string> test_strings = {
        "Hello",
        "Hello world!",
        "你好",
        "<|im_start|>user\nHello<|im_end|>\n",
        "The quick brown fox jumps over the lazy dog.",
    };

    for (const auto& text : test_strings) {
        std::vector<int> gguf_ids = gguf_tok->encode(text);
        std::vector<int> bpe_ids = bpe_tok->encode(text);
        if (gguf_ids != bpe_ids) {
            std::cerr << "Mismatch for: " << text << std::endl;
            std::cerr << "  GGUF: ";
            for (int id : gguf_ids) {
                std::cerr << id << " ";
            }
            std::cerr << std::endl;
            std::cerr << "  BPE:  ";
            for (int id : bpe_ids) {
                std::cerr << id << " ";
            }
            std::cerr << std::endl;
            ASSERT_FAIL("encode mismatch");
        }
    }

    return true;
}

static bool test_gguf_tokenizer_rejects_missing_file() {
    auto tok = create_gguf_tokenizer("models/chat/nonexistent.gguf");
    ASSERT_TRUE(tok == nullptr);
    return true;
}

// ---------------------------------------------------------------------------
// Chat template tests
// ---------------------------------------------------------------------------

static bool test_chat_template_from_gguf_qwen2() {
    std::filesystem::path path = test::write_tiny_gguf_tokenizer_fixture("tiny_template_qwen2.gguf");
    std::string tmpl = load_chat_template_from_gguf(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(!tmpl.empty());
    // Should contain Jinja2 tags
    ASSERT_TRUE(tmpl.find("{% for message in messages %}") != std::string::npos);

    // Build a prompt
    PromptBuilder builder;
    builder.set_chat_template(tmpl);

    std::vector<ChatMessage> messages = {
        {"user", "Hello"}
    };
    std::string prompt = builder.build(messages);

    // Should contain system message (auto-injected because no system msg)
    ASSERT_TRUE(prompt.find("<|im_start|>system") != std::string::npos);
    ASSERT_TRUE(prompt.find("<|im_start|>user") != std::string::npos);
    ASSERT_TRUE(prompt.find("Hello") != std::string::npos);
    ASSERT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);

    return true;
}

static bool test_chat_template_with_system_message() {
    std::filesystem::path path = test::write_tiny_gguf_tokenizer_fixture("tiny_template_system.gguf");
    std::string tmpl = load_chat_template_from_gguf(path.string());
    std::filesystem::remove(path);

    PromptBuilder builder;
    builder.set_chat_template(tmpl);

    std::vector<ChatMessage> messages = {
        {"system", "You are a coding assistant."},
        {"user", "Write a hello world program."}
    };
    std::string prompt = builder.build(messages);

    // Should contain the custom system message
    ASSERT_TRUE(prompt.find("coding assistant") != std::string::npos);
    // Should NOT contain the default system message
    ASSERT_TRUE(prompt.find("You are a helpful assistant") == std::string::npos);
    ASSERT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);

    return true;
}

static bool test_chat_template_missing_template_uses_arch_fallback_only() {
    std::filesystem::path qwen_path = test::write_tiny_gguf_tokenizer_fixture(
        "tiny_template_qwen2_fallback.gguf",
        "qwen2",
        false
    );
    std::string qwen_tmpl = load_chat_template_from_gguf(qwen_path.string());
    std::filesystem::remove(qwen_path);
    ASSERT_TRUE(qwen_tmpl == "qwen2");

    std::filesystem::path llama_path = test::write_tiny_gguf_tokenizer_fixture(
        "tiny_template_llama_empty.gguf",
        "llama",
        false
    );
    std::string llama_tmpl = load_chat_template_from_gguf(llama_path.string());
    std::filesystem::remove(llama_path);
    ASSERT_TRUE(llama_tmpl.empty());
    return true;
}

static bool test_chat_template_malformed_template_is_safe() {
    PromptBuilder builder;
    builder.set_chat_template("{% endif %}{{ 'ok' }}");
    std::string prompt = builder.build({});
    ASSERT_TRUE(prompt == "ok");

    builder.set_chat_template("{% if missing %}bad");
    prompt = builder.build({});
    ASSERT_TRUE(prompt.empty());

    builder.set_chat_template("{% for message in messages %}bad");
    prompt = builder.build({});
    ASSERT_TRUE(prompt.empty());
    return true;
}

static bool test_chat_template_supports_jinja_trim_markers() {
    PromptBuilder builder;
    builder.set_chat_template("{%- for message in messages -%}{{- '<|im_start|>' + message['role'] + '\n' + message['content'] -}}{%- endfor -%}{{- '<|im_start|>assistant\n' -}}");
    std::string prompt = builder.build({{"user", "Hello"}});

    ASSERT_TRUE(prompt.find("<|im_start|>user") != std::string::npos);
    ASSERT_TRUE(prompt.find("Hello") != std::string::npos);
    ASSERT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);
    return true;
}

static bool test_chat_template_builtin_qwen2_fallback() {
    PromptBuilder builder;
    builder.set_chat_template("qwen2");

    std::vector<ChatMessage> messages = {
        {"user", "Hi"}
    };
    std::string prompt = builder.build(messages);

    ASSERT_TRUE(prompt.find("<|im_start|>system") != std::string::npos);
    ASSERT_TRUE(prompt.find("<|im_start|>user") != std::string::npos);
    ASSERT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);

    return true;
}

static bool test_chat_template_plain_text() {
    PromptBuilder builder;
    // Empty template = plain text mode
    std::vector<ChatMessage> messages = {
        {"user", "Hello"}
    };
    std::string prompt = builder.build(messages);

    ASSERT_TRUE(prompt.find("User: Hello") != std::string::npos);
    ASSERT_TRUE(prompt.find("Assistant:") != std::string::npos);

    return true;
}

static struct GgufTokenizerTestRegistrar {
    GgufTokenizerTestRegistrar() {
        register_test("gguf_tokenizer_loads_from_gguf", test_gguf_tokenizer_loads_from_gguf);
        register_test("gguf_tokenizer_encode_decode_roundtrip", test_gguf_tokenizer_encode_decode_roundtrip);
        register_test("gguf_tokenizer_special_tokens", test_gguf_tokenizer_special_tokens);
        register_test("gguf_tokenizer_loads_real_model_when_available", test_gguf_tokenizer_loads_real_model_when_available);
        register_test("gguf_tokenizer_matches_bpe_tokenizer", test_gguf_tokenizer_matches_bpe_tokenizer);
        register_test("gguf_tokenizer_rejects_missing_file", test_gguf_tokenizer_rejects_missing_file);
        register_test("chat_template_from_gguf_qwen2", test_chat_template_from_gguf_qwen2);
        register_test("chat_template_with_system_message", test_chat_template_with_system_message);
        register_test("chat_template_missing_template_uses_arch_fallback_only", test_chat_template_missing_template_uses_arch_fallback_only);
        register_test("chat_template_malformed_template_is_safe", test_chat_template_malformed_template_is_safe);
        register_test("chat_template_supports_jinja_trim_markers", test_chat_template_supports_jinja_trim_markers);
        register_test("chat_template_builtin_qwen2_fallback", test_chat_template_builtin_qwen2_fallback);
        register_test("chat_template_plain_text", test_chat_template_plain_text);
    }
} gguf_tokenizer_test_registrar;
