#include "mini_llama/gguf.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Type name helpers
// ---------------------------------------------------------------------------
const char* ggml_type_name(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        default:             return "UNKNOWN";
    }
}

const char* gguf_value_type_name(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return "uint8";
        case GGUF_TYPE_INT8:    return "int8";
        case GGUF_TYPE_UINT16:  return "uint16";
        case GGUF_TYPE_INT16:   return "int16";
        case GGUF_TYPE_UINT32:  return "uint32";
        case GGUF_TYPE_INT32:   return "int32";
        case GGUF_TYPE_FLOAT32: return "float32";
        case GGUF_TYPE_BOOL:    return "bool";
        case GGUF_TYPE_STRING:  return "string";
        case GGUF_TYPE_ARRAY:   return "array";
        case GGUF_TYPE_UINT64:  return "uint64";
        case GGUF_TYPE_INT64:   return "int64";
        case GGUF_TYPE_FLOAT64: return "float64";
        default:                return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Internal little-endian readers
// ---------------------------------------------------------------------------
namespace {

constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;

template<typename T>
bool read_le(std::ifstream& f, T& out) {
    char buf[sizeof(T)];
    if (!f.read(buf, sizeof(T))) {
        return false;
    }
    std::memcpy(&out, buf, sizeof(T));
    return true;
}

bool read_u32(std::ifstream& f, uint32_t& out) {
    return read_le(f, out);
}

bool read_u64(std::ifstream& f, uint64_t& out) {
    return read_le(f, out);
}

bool read_string(std::ifstream& f, std::string& out) {
    uint64_t len = 0;
    if (!read_u64(f, len)) {
        return false;
    }
    if (len > kMaxStringBytes ||
        len > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    out.resize(len);
    if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
        return false;
    }
    return true;
}

bool skip_value(std::ifstream& f, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8: {
            char dummy;
            return f.read(&dummy, 1).good();
        }
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: {
            char dummy[2];
            return f.read(dummy, 2).good();
        }
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: {
            char dummy[4];
            return f.read(dummy, 4).good();
        }
        case GGUF_TYPE_BOOL: {
            char dummy;
            return f.read(&dummy, 1).good();
        }
        case GGUF_TYPE_STRING: {
            std::string dummy;
            return read_string(f, dummy);
        }
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: {
            char dummy[8];
            return f.read(dummy, 8).good();
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!read_u32(f, elem_type) || !read_u64(f, len)) {
                return false;
            }
            if (len > kMaxArrayElements) {
                return false;
            }
            for (uint64_t i = 0; i < len; ++i) {
                if (!skip_value(f, elem_type)) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool multiply_checked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool tensor_numel(const GGUFTensorInfo& tensor, uint64_t& out) {
    out = 1;
    for (uint64_t dim : tensor.shape) {
        if (dim == 0) {
            return false;
        }
        if (!multiply_checked(out, dim, out)) {
            return false;
        }
    }
    return true;
}

bool tensor_data_size(const GGUFTensorInfo& tensor, uint64_t& out) {
    uint64_t numel = 0;
    if (!tensor_numel(tensor, numel)) {
        return false;
    }

    switch (tensor.type) {
        case GGML_TYPE_F32:
            return multiply_checked(numel, 4, out);
        case GGML_TYPE_F16:
            return multiply_checked(numel, 2, out);
        case GGML_TYPE_Q4_0: {
            if (numel % kQuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = numel / kQuantBlockSize;
            return multiply_checked(n_blocks, 18, out);
        }
        case GGML_TYPE_Q4_1: {
            if (numel % kQuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = numel / kQuantBlockSize;
            return multiply_checked(n_blocks, 20, out);
        }
        case GGML_TYPE_Q8_0: {
            if (numel % kQuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = numel / kQuantBlockSize;
            return multiply_checked(n_blocks, 34, out);
        }
        default:
            out = 0;
            return true;
    }
}

bool add_checked(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

// Read a scalar value and convert to string representation.
bool read_value_as_string(std::ifstream& f, uint32_t type, std::string& out) {
    std::ostringstream oss;
    switch (type) {
        case GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << static_cast<unsigned>(v);
            break;
        }
        case GGUF_TYPE_INT8: {
            int8_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << static_cast<int>(v);
            break;
        }
        case GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_INT16: {
            int16_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (!read_u32(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_INT32: {
            int32_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_FLOAT32: {
            float v = 0.0f;
            if (!read_le(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << (v ? "true" : "false");
            break;
        }
        case GGUF_TYPE_STRING: {
            if (!read_string(f, out)) {
                return false;
            }
            return true;
        }
        case GGUF_TYPE_UINT64: {
            uint64_t v = 0;
            if (!read_u64(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_INT64: {
            int64_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_FLOAT64: {
            double v = 0.0;
            if (!read_le(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!read_u32(f, elem_type) || !read_u64(f, len)) {
                return false;
            }
            if (len > kMaxArrayElements) {
                return false;
            }
            oss << "[" << len << " x " << gguf_value_type_name(elem_type) << "]";
            for (uint64_t i = 0; i < len; ++i) {
                if (!skip_value(f, elem_type)) {
                    return false;
                }
            }
            break;
        }
        default:
            return false;
    }
    out = oss.str();
    return true;
}

uint64_t parse_alignment(const std::vector<GGUFMetadataKV>& metadata) {
    uint64_t alignment = kDefaultAlignment;
    for (const auto& kv : metadata) {
        if (kv.key != "general.alignment") {
            continue;
        }
        try {
            size_t parsed = 0;
            alignment = std::stoull(kv.value_str, &parsed);
            if (parsed != kv.value_str.size()) {
                return 0;
            }
        } catch (const std::exception&) {
            return 0;
        }
    }
    if (alignment == 0 || alignment > kMaxAlignment) {
        return 0;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return 0;
    }
    return alignment;
}

} // namespace

// ---------------------------------------------------------------------------
// GGUFReader::load
// ---------------------------------------------------------------------------
bool GGUFReader::load(const std::string& path) {
    version = 0;
    n_tensors = 0;
    n_metadata = 0;
    metadata.clear();
    tensors.clear();
    data_offset = 0;
    load_error.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        load_error = "failed to open: " + path;
        return false;
    }

    // Magic
    char magic[4];
    if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        load_error = "invalid GGUF magic";
        return false;
    }

    // Version
    if (!read_u32(f, version)) {
        load_error = "failed to read version";
        return false;
    }
    if (version != 3) {
        load_error = "unsupported GGUF version: " + std::to_string(version);
        return false;
    }

    // n_tensors, n_metadata
    if (!read_u64(f, n_tensors) || !read_u64(f, n_metadata)) {
        load_error = "failed to read header counts";
        return false;
    }
    if (n_tensors > kMaxTensorEntries) {
        load_error = "too many tensors: " + std::to_string(n_tensors);
        return false;
    }
    if (n_metadata > kMaxMetadataEntries) {
        load_error = "too many metadata entries: " + std::to_string(n_metadata);
        return false;
    }

    // Metadata
    metadata.reserve(n_metadata);
    std::unordered_set<std::string> metadata_keys;
    for (uint64_t i = 0; i < n_metadata; ++i) {
        GGUFMetadataKV kv;
        if (!read_string(f, kv.key)) {
            load_error = "failed to read metadata key at index " + std::to_string(i);
            return false;
        }
        if (!metadata_keys.insert(kv.key).second) {
            load_error = "duplicate metadata key: " + kv.key;
            return false;
        }
        if (!read_u32(f, kv.type)) {
            load_error = "failed to read metadata type for key: " + kv.key;
            return false;
        }
        if (!read_value_as_string(f, kv.type, kv.value_str)) {
            load_error = "failed to read metadata value for key: " + kv.key;
            return false;
        }
        metadata.push_back(std::move(kv));
    }

    // Tensor info
    tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GGUFTensorInfo info;
        if (!read_string(f, info.name)) {
            load_error = "failed to read tensor name at index " + std::to_string(i);
            return false;
        }
        uint32_t n_dims = 0;
        if (!read_u32(f, n_dims)) {
            load_error = "failed to read tensor n_dims for: " + info.name;
            return false;
        }
        if (n_dims == 0 || n_dims > kMaxTensorDims) {
            load_error = "invalid tensor n_dims for: " + info.name;
            return false;
        }
        info.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!read_u64(f, info.shape[d])) {
                load_error = "failed to read tensor shape for: " + info.name;
                return false;
            }
            if (info.shape[d] == 0) {
                load_error = "invalid zero tensor dimension for: " + info.name;
                return false;
            }
        }
        if (!read_u32(f, info.type)) {
            load_error = "failed to read tensor type for: " + info.name;
            return false;
        }
        if (!read_u64(f, info.offset)) {
            load_error = "failed to read tensor offset for: " + info.name;
            return false;
        }
        tensors.push_back(std::move(info));
    }

    std::unordered_set<std::string> names;
    for (const auto& tensor : tensors) {
        if (!names.insert(tensor.name).second) {
            load_error = "duplicate tensor name: " + tensor.name;
            return false;
        }
    }

    // Compute data offset (aligned to general.alignment, default 32).
    std::streampos current = f.tellg();
    if (current < 0) {
        load_error = "failed to determine GGUF data offset";
        return false;
    }
    data_offset = static_cast<uint64_t>(current);
    uint64_t alignment = parse_alignment(metadata);
    if (alignment == 0) {
        load_error = "invalid general.alignment";
        return false;
    }
    uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
    data_offset += padding;

    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();
    if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
        load_error = "GGUF file ended before tensor data section";
        return false;
    }
    uint64_t file_size = static_cast<uint64_t>(end);

    for (const auto& tensor : tensors) {
        uint64_t tensor_size = 0;
        if (!tensor_data_size(tensor, tensor_size)) {
            load_error = "invalid tensor data size for: " + tensor.name;
            return false;
        }
        if (tensor_size == 0) {
            continue;
        }

        uint64_t tensor_start = 0;
        if (!add_checked(data_offset, tensor.offset, tensor_start)) {
            load_error = "tensor offset overflow for: " + tensor.name;
            return false;
        }

        uint64_t tensor_end = 0;
        if (!add_checked(tensor_start, tensor_size, tensor_end)) {
            load_error = "tensor data range overflow for: " + tensor.name;
            return false;
        }

        if (tensor_end > file_size) {
            load_error = "tensor data outside file bounds: " + tensor.name;
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// inspect_gguf
// ---------------------------------------------------------------------------
void inspect_gguf(const GGUFReader& reader) {
    std::cout << "=== GGUF Info ===" << std::endl;
    std::cout << "  version:    " << reader.version << std::endl;
    std::cout << "  n_tensors:  " << reader.n_tensors << std::endl;
    std::cout << "  n_metadata: " << reader.n_metadata << std::endl;
    std::cout << "  data_offset: " << reader.data_offset << std::endl;

    if (!reader.metadata.empty()) {
        std::cout << std::endl;
        std::cout << "=== Metadata ===" << std::endl;
        for (const auto& kv : reader.metadata) {
            std::cout << "  " << kv.key << " (" << gguf_value_type_name(kv.type) << "): "
                      << kv.value_str << std::endl;
        }
    }

    if (!reader.tensors.empty()) {
        std::cout << std::endl;
        std::cout << "=== Tensors ===" << std::endl;
        for (const auto& t : reader.tensors) {
            std::cout << "  " << t.name << std::endl;
            std::cout << "    shape:  [";
            for (size_t i = 0; i < t.shape.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << t.shape[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "    dtype:  " << ggml_type_name(t.type) << " (" << t.type << ")" << std::endl;
            std::cout << "    offset: " << t.offset << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor data reading (inside anonymous namespace for helper access)
// ---------------------------------------------------------------------------

bool gguf_tensor_data_size(const GGUFTensorInfo& tensor, uint64_t& out_size) {
    return tensor_data_size(tensor, out_size);
}

std::vector<uint8_t> read_gguf_tensor_raw(
    const std::string& path,
    const GGUFTensorInfo& info,
    uint64_t data_offset
) {
    std::vector<uint8_t> result;

    uint64_t tensor_size = 0;
    if (!tensor_data_size(info, tensor_size)) {
        return result;
    }
    if (tensor_size == 0) {
        return result;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return result;
    }

    uint64_t tensor_start = 0;
    if (!add_checked(data_offset, info.offset, tensor_start)) {
        return result;
    }

    if (tensor_start > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return result;
    }

    f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
    if (!f.good()) {
        return result;
    }

    result.resize(tensor_size);
    f.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(tensor_size));
    if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
        result.clear();
        return result;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Internal: open GGUF, skip to metadata section, find key by name.
// On success, f is positioned after the key string, before the type uint32.
// ---------------------------------------------------------------------------

bool open_and_find_metadata_key(
    std::ifstream& f,
    const std::string& path,
    const std::string& key,
    uint32_t& out_type
) {
    f.open(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    // Skip magic (4) + version (4)
    f.seekg(8, std::ios::beg);
    if (!f.good()) {
        return false;
    }

    uint64_t n_tensors = 0;
    uint64_t n_metadata = 0;
    if (!read_u64(f, n_tensors) || !read_u64(f, n_metadata)) {
        return false;
    }

    for (uint64_t i = 0; i < n_metadata; ++i) {
        std::string read_key;
        if (!read_string(f, read_key)) {
            return false;
        }
        uint32_t type = 0;
        if (!read_u32(f, type)) {
            return false;
        }
        if (read_key == key) {
            out_type = type;
            return true;
        }
        // Skip value
        if (!skip_value(f, type)) {
            return false;
        }
    }
    return false; // key not found
}

// ---------------------------------------------------------------------------
// Typed metadata reading
// ---------------------------------------------------------------------------

bool gguf_get_metadata_int(const std::string& path, const std::string& key, int64_t& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!open_and_find_metadata_key(f, path, key, type)) {
        return false;
    }
    switch (type) {
        case GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case GGUF_TYPE_INT8: {
            int8_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case GGUF_TYPE_INT16: {
            int16_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (!read_u32(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case GGUF_TYPE_INT32: {
            int32_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case GGUF_TYPE_UINT64: {
            uint64_t v = 0;
            if (!read_u64(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case GGUF_TYPE_INT64: {
            int64_t v = 0;
            if (!read_le(f, v)) {
                return false;
            }
            out = v;
            return true;
        }
        default:
            return false;
    }
}

bool gguf_get_metadata_float(const std::string& path, const std::string& key, double& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!open_and_find_metadata_key(f, path, key, type)) {
        return false;
    }
    switch (type) {
        case GGUF_TYPE_FLOAT32: {
            float v = 0.0f;
            if (!read_le(f, v)) {
                return false;
            }
            out = static_cast<double>(v);
            return true;
        }
        case GGUF_TYPE_FLOAT64: {
            double v = 0.0;
            if (!read_le(f, v)) {
                return false;
            }
            out = v;
            return true;
        }
        default:
            return false;
    }
}

bool gguf_get_metadata_string(const std::string& path, const std::string& key, std::string& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!open_and_find_metadata_key(f, path, key, type)) {
        return false;
    }
    if (type != GGUF_TYPE_STRING) {
        return false;
    }
    return read_string(f, out);
}

bool gguf_get_metadata_string_array(const std::string& path, const std::string& key, std::vector<std::string>& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!open_and_find_metadata_key(f, path, key, type)) {
        return false;
    }
    if (type != GGUF_TYPE_ARRAY) {
        return false;
    }
    uint32_t elem_type = 0;
    uint64_t len = 0;
    if (!read_u32(f, elem_type) || !read_u64(f, len)) {
        return false;
    }
    if (elem_type != GGUF_TYPE_STRING) {
        return false;
    }
    if (len > kMaxArrayElements) {
        return false;
    }
    out.clear();
    out.reserve(len);
    for (uint64_t i = 0; i < len; ++i) {
        std::string s;
        if (!read_string(f, s)) {
            return false;
        }
        out.push_back(std::move(s));
    }
    return true;
}

bool gguf_get_metadata_int_array(const std::string& path, const std::string& key, std::vector<int32_t>& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!open_and_find_metadata_key(f, path, key, type)) {
        return false;
    }
    if (type != GGUF_TYPE_ARRAY) {
        return false;
    }
    uint32_t elem_type = 0;
    uint64_t len = 0;
    if (!read_u32(f, elem_type) || !read_u64(f, len)) {
        return false;
    }
    if (elem_type != GGUF_TYPE_INT32) {
        return false;
    }
    if (len > kMaxArrayElements) {
        return false;
    }
    out.clear();
    out.reserve(len);
    for (uint64_t i = 0; i < len; ++i) {
        int32_t v = 0;
        if (!read_le(f, v)) {
            return false;
        }
        out.push_back(v);
    }
    return true;
}

} // namespace mini_llama
