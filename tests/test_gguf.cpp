#include "test_main.h"
#include "mini_llama/gguf.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

using namespace mini_llama;

namespace {

std::filesystem::path test_temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>()
    );
}

void write_binary_file(const std::filesystem::path& path, const std::vector<char>& data) {
    std::ofstream out(path, std::ios::binary);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void append_u32_le(std::vector<char>& data, uint32_t value) {
    data.push_back(static_cast<char>(value & 0xff));
    data.push_back(static_cast<char>((value >> 8) & 0xff));
    data.push_back(static_cast<char>((value >> 16) & 0xff));
    data.push_back(static_cast<char>((value >> 24) & 0xff));
}

void append_u64_le(std::vector<char>& data, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
    }
}

void append_string(std::vector<char>& data, const std::string& value) {
    append_u64_le(data, value.size());
    data.insert(data.end(), value.begin(), value.end());
}

void write_u32_le(std::vector<char>& data, size_t offset, uint32_t value) {
    data[offset + 0] = static_cast<char>(value & 0xff);
    data[offset + 1] = static_cast<char>((value >> 8) & 0xff);
    data[offset + 2] = static_cast<char>((value >> 16) & 0xff);
    data[offset + 3] = static_cast<char>((value >> 24) & 0xff);
}

void write_u64_le(std::vector<char>& data, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        data[offset + i] = static_cast<char>((value >> (i * 8)) & 0xff);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// GGUF reader tests
// ---------------------------------------------------------------------------
static bool test_gguf_load_test_file() {
    GGUFReader reader;
    bool ok = reader.load("models/tiny/test.gguf");
    if (!ok) {
        ASSERT_FAIL(std::string("failed to load test.gguf: ") + reader.load_error);
    }
    ASSERT_EQ(reader.version, 3);
    ASSERT_EQ(reader.n_tensors, 1);
    ASSERT_EQ(reader.n_metadata, 3);
    return true;
}

static bool test_gguf_metadata() {
    GGUFReader reader;
    if (!reader.load("models/tiny/test.gguf")) {
        ASSERT_FAIL(reader.load_error);
    }

    ASSERT_EQ(reader.metadata.size(), 3);
    ASSERT_TRUE(reader.metadata[0].key == "general.architecture");
    ASSERT_TRUE(reader.metadata[0].value_str == "llama");
    ASSERT_TRUE(reader.metadata[1].key == "general.name");
    ASSERT_TRUE(reader.metadata[1].value_str == "tiny-test");
    ASSERT_TRUE(reader.metadata[2].key == "general.alignment");
    ASSERT_TRUE(reader.metadata[2].value_str == "64");
    return true;
}

static bool test_gguf_tensor_info() {
    GGUFReader reader;
    if (!reader.load("models/tiny/test.gguf")) {
        ASSERT_FAIL(reader.load_error);
    }

    ASSERT_EQ(reader.tensors.size(), 1);
    ASSERT_TRUE(reader.tensors[0].name == "token_embd.weight");
    ASSERT_EQ(reader.tensors[0].shape.size(), 2);
    ASSERT_EQ(reader.tensors[0].shape[0], 128);
    ASSERT_EQ(reader.tensors[0].shape[1], 32);
    ASSERT_EQ(reader.tensors[0].type, GGML_TYPE_F32);
    ASSERT_EQ(reader.tensors[0].offset, 0);
    return true;
}

static bool test_gguf_data_offset_computed() {
    GGUFReader reader;
    if (!reader.load("models/tiny/test.gguf")) {
        ASSERT_FAIL(reader.load_error);
    }

    // Data offset should be > 0 and honor general.alignment.
    ASSERT_TRUE(reader.data_offset > 0);
    ASSERT_EQ(reader.data_offset % 64, 0);
    return true;
}

static bool test_gguf_repeated_load_resets_state() {
    GGUFReader reader;
    if (!reader.load("models/tiny/test.gguf")) {
        ASSERT_FAIL(reader.load_error);
    }

    bool ok = reader.load("models/tiny/model.json");
    ASSERT_TRUE(!ok);
    ASSERT_EQ(reader.version, 0);
    ASSERT_EQ(reader.n_tensors, 0);
    ASSERT_EQ(reader.n_metadata, 0);
    ASSERT_TRUE(reader.metadata.empty());
    ASSERT_TRUE(reader.tensors.empty());
    ASSERT_EQ(reader.data_offset, 0);
    ASSERT_TRUE(reader.load_error == "invalid GGUF magic");
    return true;
}

static bool test_gguf_unsupported_version_rejected() {
    std::vector<char> data = read_binary_file("models/tiny/test.gguf");
    write_u32_le(data, 4, 2);

    std::filesystem::path path = test_temp_path("bad_version.gguf");
    write_binary_file(path, data);

    GGUFReader reader;
    bool ok = reader.load(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(!ok);
    ASSERT_TRUE(reader.load_error == "unsupported GGUF version: 2");
    return true;
}

static bool test_gguf_truncated_file_rejected() {
    std::vector<char> data = read_binary_file("models/tiny/test.gguf");
    data.resize(12);

    std::filesystem::path path = test_temp_path("truncated.gguf");
    write_binary_file(path, data);

    GGUFReader reader;
    bool ok = reader.load(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(!ok);
    ASSERT_TRUE(!reader.load_error.empty());
    return true;
}

static bool test_gguf_missing_file() {
    GGUFReader reader;
    bool ok = reader.load("models/tiny/nonexistent.gguf");
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(!reader.load_error.empty());
    return true;
}

static bool test_gguf_invalid_magic() {
    GGUFReader reader;
    bool ok = reader.load("models/tiny/model.json"); // not a GGUF file
    ASSERT_TRUE(!ok);
    ASSERT_TRUE(reader.load_error == "invalid GGUF magic");
    return true;
}

static bool test_gguf_invalid_alignment_rejected() {
    std::vector<char> data = read_binary_file("models/tiny/test.gguf");

    // general.alignment is the third metadata value in the generated file.
    const std::string needle = "general.alignment";
    auto it = std::search(data.begin(), data.end(), needle.begin(), needle.end());
    ASSERT_TRUE(it != data.end());
    size_t alignment_value_offset = static_cast<size_t>(it - data.begin()) + needle.size() + 4;
    write_u32_le(data, alignment_value_offset, 48);

    std::filesystem::path path = test_temp_path("bad_alignment.gguf");
    write_binary_file(path, data);

    GGUFReader reader;
    bool ok = reader.load(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(!ok);
    ASSERT_TRUE(reader.load_error == "invalid general.alignment");
    return true;
}

static bool test_gguf_duplicate_metadata_key_rejected() {
    std::vector<char> data;
    data.insert(data.end(), {'G', 'G', 'U', 'F'});
    append_u32_le(data, 3);
    append_u64_le(data, 0);
    append_u64_le(data, 2);

    append_string(data, "general.name");
    append_u32_le(data, GGUF_TYPE_STRING);
    append_string(data, "first");

    append_string(data, "general.name");
    append_u32_le(data, GGUF_TYPE_STRING);
    append_string(data, "second");

    std::filesystem::path path = test_temp_path("duplicate_metadata.gguf");
    write_binary_file(path, data);

    GGUFReader reader;
    bool ok = reader.load(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(!ok);
    ASSERT_TRUE(reader.load_error == "duplicate metadata key: general.name");
    return true;
}

static bool test_gguf_zero_tensor_dimension_rejected() {
    std::vector<char> data = read_binary_file("models/tiny/test.gguf");

    const std::string tensor_name = "token_embd.weight";
    auto it = std::search(data.begin(), data.end(), tensor_name.begin(), tensor_name.end());
    ASSERT_TRUE(it != data.end());
    size_t first_dim_offset = static_cast<size_t>(it - data.begin()) + tensor_name.size() + 4;
    write_u64_le(data, first_dim_offset, 0);

    std::filesystem::path path = test_temp_path("zero_dim.gguf");
    write_binary_file(path, data);

    GGUFReader reader;
    bool ok = reader.load(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(!ok);
    ASSERT_TRUE(reader.load_error == "invalid zero tensor dimension for: token_embd.weight");
    return true;
}

static bool test_gguf_tensor_data_bounds_checked() {
    std::vector<char> data = read_binary_file("models/tiny/test.gguf");

    const std::string tensor_name = "token_embd.weight";
    auto it = std::search(data.begin(), data.end(), tensor_name.begin(), tensor_name.end());
    ASSERT_TRUE(it != data.end());
    size_t offset_field = static_cast<size_t>(it - data.begin()) + tensor_name.size() + 4 + 8 + 8 + 4;
    write_u64_le(data, offset_field, static_cast<uint64_t>(data.size()));

    std::filesystem::path path = test_temp_path("bad_tensor_bounds.gguf");
    write_binary_file(path, data);

    GGUFReader reader;
    bool ok = reader.load(path.string());
    std::filesystem::remove(path);

    ASSERT_TRUE(!ok);
    ASSERT_TRUE(reader.load_error == "tensor data outside file bounds: token_embd.weight");
    return true;
}

// ---------------------------------------------------------------------------
// Type name helpers
// ---------------------------------------------------------------------------
static bool test_ggml_type_name() {
    ASSERT_TRUE(std::string(ggml_type_name(GGML_TYPE_F32)) == "F32");
    ASSERT_TRUE(std::string(ggml_type_name(GGML_TYPE_F16)) == "F16");
    ASSERT_TRUE(std::string(ggml_type_name(GGML_TYPE_Q4_0)) == "Q4_0");
    ASSERT_TRUE(std::string(ggml_type_name(GGML_TYPE_Q8_0)) == "Q8_0");
    ASSERT_TRUE(std::string(ggml_type_name(999)) == "UNKNOWN");
    return true;
}

static bool test_gguf_value_type_name() {
    ASSERT_TRUE(std::string(gguf_value_type_name(GGUF_TYPE_UINT32)) == "uint32");
    ASSERT_TRUE(std::string(gguf_value_type_name(GGUF_TYPE_STRING)) == "string");
    ASSERT_TRUE(std::string(gguf_value_type_name(GGUF_TYPE_ARRAY)) == "array");
    ASSERT_TRUE(std::string(gguf_value_type_name(999)) == "unknown");
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct GGUFTestRegistrar {
    GGUFTestRegistrar() {
        register_test("gguf_load_test_file", test_gguf_load_test_file);
        register_test("gguf_metadata", test_gguf_metadata);
        register_test("gguf_tensor_info", test_gguf_tensor_info);
        register_test("gguf_data_offset_computed", test_gguf_data_offset_computed);
        register_test("gguf_repeated_load_resets_state", test_gguf_repeated_load_resets_state);
        register_test("gguf_unsupported_version_rejected", test_gguf_unsupported_version_rejected);
        register_test("gguf_truncated_file_rejected", test_gguf_truncated_file_rejected);
        register_test("gguf_missing_file", test_gguf_missing_file);
        register_test("gguf_invalid_magic", test_gguf_invalid_magic);
        register_test("gguf_invalid_alignment_rejected", test_gguf_invalid_alignment_rejected);
        register_test("gguf_duplicate_metadata_key_rejected", test_gguf_duplicate_metadata_key_rejected);
        register_test("gguf_zero_tensor_dimension_rejected", test_gguf_zero_tensor_dimension_rejected);
        register_test("gguf_tensor_data_bounds_checked", test_gguf_tensor_data_bounds_checked);
        register_test("ggml_type_name", test_ggml_type_name);
        register_test("gguf_value_type_name", test_gguf_value_type_name);
    }
} gguf_test_registrar;
