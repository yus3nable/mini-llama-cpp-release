#include "mini_llama/loader.h"
#include "test_main.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace mini_llama;

namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::filesystem::path test_temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
}

bool replace_once(std::string& text, const std::string& from, const std::string& to) {
    size_t pos = text.find(from);
    if (pos == std::string::npos) {
        return false;
    }
    text.replace(pos, from.size(), to);
    return true;
}

bool replace_after(std::string& text, const std::string& marker, const std::string& from, const std::string& to) {
    size_t marker_pos = text.find(marker);
    if (marker_pos == std::string::npos) {
        return false;
    }
    size_t pos = text.find(from, marker_pos);
    if (pos == std::string::npos) {
        return false;
    }
    text.replace(pos, from.size(), to);
    return true;
}

bool parse_manifest_fails(const std::string& json, const std::string& filename) {
    std::filesystem::path path = test_temp_path(filename);
    write_text_file(path, json);
    try {
        parse_manifest(path.string());
        std::filesystem::remove(path);
        return false;
    } catch (const std::runtime_error&) {
        std::filesystem::remove(path);
        return true;
    }
}

std::string replace_tokenizer_type(const std::string& json, const std::string& tokenizer_type) {
    std::string updated = json;
    if (!replace_once(updated, "\"type\": \"json_vocab\"", "\"type\": \"" + tokenizer_type + "\"")) {
        throw std::runtime_error("failed to replace tokenizer type");
    }
    return updated;
}

std::string remove_tokenizer_block(const std::string& json) {
    const std::string tokenizer_block =
        "  \"tokenizer\": {\n"
        "    \"type\": \"json_vocab\",\n"
        "    \"path\": \"vocab.json\",\n"
        "    \"bos_id\": 1,\n"
        "    \"eos_id\": 2,\n"
        "    \"unk_id\": 0\n"
        "  },\n";
    std::string updated = json;
    if (!replace_once(updated, tokenizer_block, "")) {
        throw std::runtime_error("failed to remove tokenizer block");
    }
    return updated;
}

} // namespace

static bool test_loader_manifest_parses_metadata() {
    ModelManifest manifest = parse_manifest("models/tiny/model.json");
    ASSERT_EQ(manifest.config.vocab_size, 128);
    ASSERT_EQ(manifest.config.n_layers, 2);
    ASSERT_EQ(manifest.tensors.size(), 21);
    ASSERT_TRUE(manifest.tensors[0].name == "token_embedding");
    ASSERT_TRUE(manifest.tensors[0].dtype == "float32");
    ASSERT_EQ(manifest.tensors[0].offset, 0);
    ASSERT_EQ(manifest.tensors[0].byte_size, 16384);
    ASSERT_TRUE(manifest.tokenizer.type == "json_vocab");
    ASSERT_TRUE(manifest.tokenizer.path == "vocab.json");
    ASSERT_EQ(manifest.tokenizer.bos_id, 1);
    ASSERT_EQ(manifest.tokenizer.eos_id, 2);
    ASSERT_EQ(manifest.tokenizer.unk_id, 0);
    return true;
}

static bool test_loader_manifest_defaults_ascii_without_tokenizer_block() {
    std::string json = read_text_file("models/tiny/model.json");
    std::string without_tokenizer = remove_tokenizer_block(json);
    std::filesystem::path path = test_temp_path("tokenizer_manifest.json");
    write_text_file(path, without_tokenizer);

    ModelManifest manifest = parse_manifest(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(manifest.tokenizer.type == "ascii");
    ASSERT_TRUE(manifest.tokenizer.path.empty());
    ASSERT_EQ(manifest.tokenizer.bos_id, 1);
    ASSERT_EQ(manifest.tokenizer.eos_id, 2);
    ASSERT_EQ(manifest.tokenizer.unk_id, 0);
    return true;
}

static bool test_loader_rejects_invalid_tokenizer_type() {
    std::string json = read_text_file("models/tiny/model.json");
    std::string with_tokenizer = replace_tokenizer_type(json, "bpe");
    ASSERT_TRUE(parse_manifest_fails(with_tokenizer, "invalid_tokenizer_type.json"));
    return true;
}

static bool test_loader_rejects_duplicate_tensor_name() {
    std::string json = read_text_file("models/tiny/model.json");
    ASSERT_TRUE(replace_once(json, "\"name\": \"final_norm\"", "\"name\": \"token_embedding\""));
    ASSERT_TRUE(parse_manifest_fails(json, "duplicate_tensor.json"));
    return true;
}

static bool test_loader_rejects_required_shape_mismatch() {
    std::string json = read_text_file("models/tiny/model.json");
    ASSERT_TRUE(replace_after(
        json,
        "\"name\": \"lm_head\"",
        "\"shape\": [\n        128,\n        32\n      ]",
        "\"shape\": [\n        64,\n        64\n      ]"
    ));
    ASSERT_TRUE(parse_manifest_fails(json, "shape_mismatch.json"));
    return true;
}

static bool test_loader_rejects_overlapping_tensor_ranges() {
    std::string json = read_text_file("models/tiny/model.json");
    ASSERT_TRUE(replace_after(json, "\"name\": \"lm_head\"", "\"offset\": 115840", "\"offset\": 115712"));
    ASSERT_TRUE(parse_manifest_fails(json, "overlap.json"));
    return true;
}

static bool test_loader_rejects_missing_dtype() {
    std::string json = read_text_file("models/tiny/model.json");
    ASSERT_TRUE(replace_after(json, "\"name\": \"lm_head\"", "      \"dtype\": \"float32\",\n", ""));
    ASSERT_TRUE(parse_manifest_fails(json, "missing_dtype.json"));
    return true;
}

static bool test_loader_rejects_fractional_integer_config() {
    std::string json = read_text_file("models/tiny/model.json");
    ASSERT_TRUE(replace_once(json, "\"n_heads\": 4", "\"n_heads\": 4.5"));
    ASSERT_TRUE(parse_manifest_fails(json, "fractional_integer_config.json"));
    return true;
}

static bool test_loader_rejects_nonfinite_float_config() {
    std::string json = read_text_file("models/tiny/model.json");
    ASSERT_TRUE(replace_once(json, "\"rope_theta\": 10000.0", "\"rope_theta\": NaN"));
    ASSERT_TRUE(parse_manifest_fails(json, "nonfinite_float_config.json"));
    return true;
}

static bool test_loader_rejects_trailing_weight_bytes() {
    std::filesystem::path weights_path = test_temp_path("trailing_model.bin");
    {
        std::ifstream in("models/tiny/model.bin", std::ios::binary);
        std::ofstream out(weights_path, std::ios::binary);
        out << in.rdbuf();
        char extra = 0;
        out.write(&extra, 1);
    }

    MiniLlamaModel model = load_model("models/tiny/model.json", weights_path.string());
    std::filesystem::remove(weights_path);
    ASSERT_TRUE(!model.loaded);
    ASSERT_TRUE(model.load_error.find("trailing bytes") != std::string::npos);
    return true;
}

static size_t cuda_resident_weight_bytes(const MiniLlamaModel& model) {
    auto bytes_for = [](const QuantizedTensor& weight) -> size_t {
        if (weight.type != QuantType::F32) {
            return 0;
        }
        return weight.f32_data.size() * sizeof(float);
    };
    auto tensor_bytes = [](const Tensor& tensor) -> size_t {
        return tensor.data.size() * sizeof(float);
    };

    size_t bytes = tensor_bytes(model.token_embedding) + tensor_bytes(model.final_norm) + bytes_for(model.lm_head);
    for (const auto& layer : model.layers) {
        bytes += tensor_bytes(layer.attention_norm);
        bytes += bytes_for(layer.wq);
        bytes += bytes_for(layer.wk);
        bytes += bytes_for(layer.wv);
        bytes += tensor_bytes(layer.bq);
        bytes += tensor_bytes(layer.bk);
        bytes += tensor_bytes(layer.bv);
        bytes += bytes_for(layer.wo);
        bytes += tensor_bytes(layer.ffn_norm);
        bytes += bytes_for(layer.w_gate);
        bytes += bytes_for(layer.w_up);
        bytes += bytes_for(layer.w_down);
    }
    return bytes;
}

static bool test_model_cuda_weight_storage_state() {
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    ASSERT_TRUE(model.loaded);
    ASSERT_TRUE(!model_has_cuda_weights(model));
    ASSERT_EQ(model_cuda_uploaded_weight_count(model), 0);
    ASSERT_EQ(model_cuda_memory_bytes(model), 0);

    if (!cuda_runtime_built()) {
        bool threw = false;
        try {
            upload_model_weights_to_cuda(model, 0);
        } catch (const std::runtime_error& e) {
            threw = true;
            ASSERT_TRUE(std::string(e.what()).find("CUDA backend was not built") != std::string::npos);
        }
        ASSERT_TRUE(threw);
        ASSERT_TRUE(!model_has_cuda_weights(model));
        return true;
    }

    if (cuda_device_count() == 0) {
        return true;
    }

    upload_model_weights_to_cuda(model, 0);
    ASSERT_TRUE(model_has_cuda_weights(model));
    ASSERT_EQ(model_cuda_uploaded_weight_count(model), 21);
    ASSERT_EQ(model_cuda_memory_bytes(model), cuda_resident_weight_bytes(model));

    clear_model_cuda_weights(model);
    ASSERT_TRUE(!model_has_cuda_weights(model));
    ASSERT_EQ(model_cuda_uploaded_weight_count(model), 0);
    ASSERT_EQ(model_cuda_memory_bytes(model), 0);
    return true;
}

static struct LoaderTestRegistrar {
    LoaderTestRegistrar() {
        register_test("loader_manifest_parses_metadata", test_loader_manifest_parses_metadata);
        register_test("loader_manifest_defaults_ascii_without_tokenizer_block", test_loader_manifest_defaults_ascii_without_tokenizer_block);
        register_test("loader_rejects_invalid_tokenizer_type", test_loader_rejects_invalid_tokenizer_type);
        register_test("loader_rejects_duplicate_tensor_name", test_loader_rejects_duplicate_tensor_name);
        register_test("loader_rejects_required_shape_mismatch", test_loader_rejects_required_shape_mismatch);
        register_test("loader_rejects_overlapping_tensor_ranges", test_loader_rejects_overlapping_tensor_ranges);
        register_test("loader_rejects_missing_dtype", test_loader_rejects_missing_dtype);
        register_test("loader_rejects_fractional_integer_config", test_loader_rejects_fractional_integer_config);
        register_test("loader_rejects_nonfinite_float_config", test_loader_rejects_nonfinite_float_config);
        register_test("loader_rejects_trailing_weight_bytes", test_loader_rejects_trailing_weight_bytes);
        register_test("model_cuda_weight_storage_state", test_model_cuda_weight_storage_state);
    }
} loader_test_registrar;
