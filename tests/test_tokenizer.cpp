#include "test_main.h"
#include "mini_llama/tokenizer.h"
#include <chrono>
#include <filesystem>
#include <fstream>

using namespace mini_llama;

namespace {

struct TempBpeFiles {
    std::filesystem::path dir;
    std::filesystem::path vocab;
    std::filesystem::path merges;
    std::filesystem::path special;
};

static TempBpeFiles write_temp_bpe_files() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    TempBpeFiles files;
    files.dir = std::filesystem::temp_directory_path() /
                ("mini_llama_bpe_tokenizer_" + std::to_string(stamp));
    std::filesystem::create_directories(files.dir);
    files.vocab = files.dir / "vocab.json";
    files.merges = files.dir / "merges.txt";
    files.special = files.dir / "special_tokens.json";

    {
        std::ofstream out(files.vocab);
        out << "{\n"
            << "  \"h\": 0,\n"
            << "  \"e\": 1,\n"
            << "  \"l\": 2,\n"
            << "  \"o\": 3,\n"
            << "  \"he\": 4,\n"
            << "  \"hel\": 5,\n"
            << "  \"hell\": 6,\n"
            << "  \"hello\": 7,\n"
            << "  \"<|im_start|>\": 8,\n"
            << "  \"<unk>\": 9,\n"
            << "  \"<bos>\": 10,\n"
            << "  \"<eos>\": 11\n"
            << "}\n";
    }
    {
        std::ofstream out(files.merges);
        out << "h e\n"
            << "he l\n"
            << "hel l\n"
            << "hell o\n";
    }
    {
        std::ofstream out(files.special);
        out << "{ \"bos_id\": 10, \"eos_id\": 11, \"unk_id\": 9 }\n";
    }
    return files;
}

static void remove_temp_bpe_files(const TempBpeFiles& files) {
    std::error_code ec;
    std::filesystem::remove_all(files.dir, ec);
}

} // namespace

// ---------------------------------------------------------------------------
// AsciiTokenizer
// ---------------------------------------------------------------------------
static bool test_ascii_encode_basic() {
    AsciiTokenizer tok;
    auto tokens = tok.encode("hi");
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0], tok.bos_id());  // 1
    ASSERT_EQ(tokens[1], static_cast<int>('h'));
    ASSERT_EQ(tokens[2], static_cast<int>('i'));
    return true;
}

static bool test_ascii_encode_empty() {
    AsciiTokenizer tok;
    auto tokens = tok.encode("");
    ASSERT_EQ(tokens.size(), 1);
    ASSERT_EQ(tokens[0], tok.bos_id());
    return true;
}

static bool test_ascii_decode_token() {
    AsciiTokenizer tok;
    ASSERT_TRUE(tok.decode_token(tok.bos_id()) == "<bos>");
    ASSERT_TRUE(tok.decode_token(tok.eos_id()) == "<eos>");
    ASSERT_TRUE(tok.decode_token(tok.unk_id()) == "<unk>");
    ASSERT_TRUE(tok.decode_token(static_cast<int>('a')) == "a");
    ASSERT_TRUE(tok.decode_token(32) == " ");
    ASSERT_TRUE(tok.decode_token(10) == "");  // control char
    ASSERT_TRUE(tok.decode_token(127) == ""); // DEL
    ASSERT_TRUE(tok.decode_token(200) == "<unk>"); // out of range
    return true;
}

static bool test_ascii_decode_sequence() {
    AsciiTokenizer tok;
    std::vector<int> tokens = {tok.bos_id(), static_cast<int>('h'), static_cast<int>('i')};
    std::string text = tok.decode(tokens);
    ASSERT_TRUE(text == "<bos>hi");
    return true;
}

static bool test_ascii_vocab_size() {
    AsciiTokenizer tok;
    ASSERT_EQ(tok.vocab_size(), 128);
    ASSERT_EQ(tok.bos_id(), 1);
    ASSERT_EQ(tok.eos_id(), 2);
    ASSERT_EQ(tok.unk_id(), 0);
    return true;
}

// ---------------------------------------------------------------------------
// JsonVocabTokenizer
// ---------------------------------------------------------------------------
static bool test_jsonvocab_loads() {
    try {
        JsonVocabTokenizer tok("models/tiny/vocab.json");
        ASSERT_EQ(tok.vocab_size(), 128);
        ASSERT_EQ(tok.bos_id(), 1);
        ASSERT_EQ(tok.eos_id(), 2);
        ASSERT_EQ(tok.unk_id(), 0);
    } catch (const std::runtime_error& e) {
        ASSERT_FAIL(std::string("failed to load vocab.json: ") + e.what());
    }
    return true;
}

static bool test_jsonvocab_encode_basic() {
    try {
        JsonVocabTokenizer tok("models/tiny/vocab.json");
        auto tokens = tok.encode("hi");
        ASSERT_EQ(tokens.size(), 3);
        ASSERT_EQ(tokens[0], tok.bos_id());
        ASSERT_EQ(tokens[1], static_cast<int>('h'));
        ASSERT_EQ(tokens[2], static_cast<int>('i'));
    } catch (const std::runtime_error& e) {
        ASSERT_FAIL(std::string("failed to load vocab.json: ") + e.what());
    }
    return true;
}

static bool test_jsonvocab_decode_token() {
    try {
        JsonVocabTokenizer tok("models/tiny/vocab.json");
        ASSERT_TRUE(tok.decode_token(0) == "<unk>");
        ASSERT_TRUE(tok.decode_token(1) == "<bos>");
        ASSERT_TRUE(tok.decode_token(2) == "<eos>");
        ASSERT_TRUE(tok.decode_token(static_cast<int>('a')) == "a");
        ASSERT_TRUE(tok.decode_token(32) == " ");
        ASSERT_TRUE(tok.decode_token(10) == "");  // control char maps to empty
        ASSERT_TRUE(tok.decode_token(127) == ""); // DEL
        ASSERT_TRUE(tok.decode_token(200) == "<unk>"); // out of range
    } catch (const std::runtime_error& e) {
        ASSERT_FAIL(std::string("failed to load vocab.json: ") + e.what());
    }
    return true;
}

static bool test_jsonvocab_decode_sequence() {
    try {
        JsonVocabTokenizer tok("models/tiny/vocab.json");
        std::vector<int> tokens = {1, static_cast<int>('h'), static_cast<int>('i')};
        std::string text = tok.decode(tokens);
        ASSERT_TRUE(text == "<bos>hi");
    } catch (const std::runtime_error& e) {
        ASSERT_FAIL(std::string("failed to load vocab.json: ") + e.what());
    }
    return true;
}

static bool test_jsonvocab_matches_ascii() {
    try {
        AsciiTokenizer ascii_tok;
        JsonVocabTokenizer json_tok("models/tiny/vocab.json");

        std::string prompt = "hello world!";
        auto ascii_tokens = ascii_tok.encode(prompt);
        auto json_tokens = json_tok.encode(prompt);

        ASSERT_EQ(ascii_tokens.size(), json_tokens.size());
        for (size_t i = 0; i < ascii_tokens.size(); ++i) {
            ASSERT_EQ(ascii_tokens[i], json_tokens[i]);
        }

        std::string ascii_decoded = ascii_tok.decode(ascii_tokens);
        std::string json_decoded = json_tok.decode(json_tokens);
        ASSERT_TRUE(ascii_decoded == json_decoded);
    } catch (const std::runtime_error& e) {
        ASSERT_FAIL(std::string("failed to load vocab.json: ") + e.what());
    }
    return true;
}

static bool test_jsonvocab_missing_file() {
    try {
        JsonVocabTokenizer tok("models/tiny/nonexistent_vocab.json");
        ASSERT_FAIL("expected exception for missing vocab file");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
static bool test_factory_uses_json_when_exists() {
    auto tok = create_tokenizer("models/tiny/vocab.json");
    ASSERT_TRUE(tok != nullptr);
    ASSERT_EQ(tok->vocab_size(), 128);
    return true;
}

static bool test_factory_falls_back_to_ascii() {
    auto tok = create_tokenizer("");
    ASSERT_TRUE(tok != nullptr);
    ASSERT_EQ(tok->vocab_size(), 128); // AsciiTokenizer
    return true;
}

static bool test_factory_falls_back_when_missing() {
    auto tok = create_tokenizer("models/tiny/does_not_exist.json");
    ASSERT_TRUE(tok != nullptr);
    ASSERT_EQ(tok->vocab_size(), 128); // AsciiTokenizer fallback
    return true;
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------
static bool test_bpe_encode_decode_and_special_token() {
    TempBpeFiles files = write_temp_bpe_files();
    auto tok = create_bpe_tokenizer(
        files.vocab.string(),
        files.merges.string(),
        files.special.string()
    );
    ASSERT_TRUE(tok != nullptr);
    ASSERT_EQ(tok->bos_id(), 10);
    ASSERT_EQ(tok->eos_id(), 11);
    ASSERT_EQ(tok->unk_id(), 9);

    std::vector<int> hello = tok->encode("hello");
    ASSERT_EQ(hello.size(), 1u);
    ASSERT_EQ(hello[0], 7);
    ASSERT_TRUE(tok->decode(hello) == "hello");

    std::vector<int> special = tok->encode("<|im_start|>");
    ASSERT_EQ(special.size(), 1u);
    ASSERT_EQ(special[0], 8);
    ASSERT_TRUE(tok->decode(special) == "<|im_start|>");

    remove_temp_bpe_files(files);
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct TokenizerTestRegistrar {
    TokenizerTestRegistrar() {
        register_test("ascii_encode_basic", test_ascii_encode_basic);
        register_test("ascii_encode_empty", test_ascii_encode_empty);
        register_test("ascii_decode_token", test_ascii_decode_token);
        register_test("ascii_decode_sequence", test_ascii_decode_sequence);
        register_test("ascii_vocab_size", test_ascii_vocab_size);
        register_test("jsonvocab_loads", test_jsonvocab_loads);
        register_test("jsonvocab_encode_basic", test_jsonvocab_encode_basic);
        register_test("jsonvocab_decode_token", test_jsonvocab_decode_token);
        register_test("jsonvocab_decode_sequence", test_jsonvocab_decode_sequence);
        register_test("jsonvocab_matches_ascii", test_jsonvocab_matches_ascii);
        register_test("jsonvocab_missing_file", test_jsonvocab_missing_file);
        register_test("factory_uses_json_when_exists", test_factory_uses_json_when_exists);
        register_test("factory_falls_back_to_ascii", test_factory_falls_back_to_ascii);
        register_test("factory_falls_back_when_missing", test_factory_falls_back_when_missing);
        register_test("bpe_encode_decode_and_special_token", test_bpe_encode_decode_and_special_token);
    }
} tokenizer_test_registrar;
