#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// GGML types (subset)
// ---------------------------------------------------------------------------
enum GGMLType : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q8_0 = 8,
};

const char* ggml_type_name(uint32_t type);

// ---------------------------------------------------------------------------
// GGUF value types
// ---------------------------------------------------------------------------
enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

const char* gguf_value_type_name(uint32_t type);

// ---------------------------------------------------------------------------
// GGUF tensor info
// ---------------------------------------------------------------------------
struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = 0;     // GGML type
    uint64_t offset = 0;   // offset from start of data section
};

// ---------------------------------------------------------------------------
// GGUF metadata KV
// ---------------------------------------------------------------------------
struct GGUFMetadataKV {
    std::string key;
    uint32_t type = 0;     // GGUF value type
    std::string value_str; // human-readable representation
};

// ---------------------------------------------------------------------------
// Minimal GGUF reader (v3)
// ---------------------------------------------------------------------------
struct GGUFReader {
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_metadata = 0;
    std::vector<GGUFMetadataKV> metadata;
    std::vector<GGUFTensorInfo> tensors;
    uint64_t data_offset = 0;  // absolute file offset where tensor data begins

    // Load and parse a GGUF file. Returns false on error (message in load_error).
    bool load(const std::string& path);

    std::string load_error;
};

// Print GGUF contents to stdout (for inspect-gguf).
void inspect_gguf(const GGUFReader& reader);

// ---------------------------------------------------------------------------
// Tensor data reading
// ---------------------------------------------------------------------------

// Read raw tensor bytes from the GGUF file data section.
// Returns empty vector on error.
std::vector<uint8_t> read_gguf_tensor_raw(
    const std::string& path,
    const GGUFTensorInfo& info,
    uint64_t data_offset
);

// Compute the byte size of a tensor's data given its GGML type and shape.
bool gguf_tensor_data_size(const GGUFTensorInfo& info, uint64_t& out_size);

// ---------------------------------------------------------------------------
// Typed metadata reading (re-opens file and seeks to the key)
// Returns true if found and type matches.
// ---------------------------------------------------------------------------
bool gguf_get_metadata_int(const std::string& path, const std::string& key, int64_t& out);
bool gguf_get_metadata_float(const std::string& path, const std::string& key, double& out);
bool gguf_get_metadata_string(const std::string& path, const std::string& key, std::string& out);
bool gguf_get_metadata_string_array(const std::string& path, const std::string& key, std::vector<std::string>& out);
bool gguf_get_metadata_int_array(const std::string& path, const std::string& key, std::vector<int32_t>& out);

} // namespace mini_llama
