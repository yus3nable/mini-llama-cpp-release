#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf.h"
#include "mini_llama/quant.h"
#include "mini_llama/model.h"
#include <fstream>
#include <iostream>
#include <map>
#include <cstring>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Tensor name mapping: GGUF (llama.cpp style) -> our naming
// ---------------------------------------------------------------------------

static std::string map_tensor_name(const std::string& gguf_name) {
    // Embedding
    if (gguf_name == "token_embd.weight") {
        return "token_embedding";
    }
    // Output norm
    if (gguf_name == "output_norm.weight") {
        return "final_norm";
    }
    // Output projection (lm_head) — may be tied with token_embd
    if (gguf_name == "output.weight") {
        return "lm_head";
    }

    // Layer tensors: blk.{L}.{name}.weight
    // Parse "blk.N.xxx.weight"
    if (gguf_name.rfind("blk.", 0) == 0) {
        size_t dot1 = gguf_name.find('.', 4);
        if (dot1 == std::string::npos) {
            return "";
        }
        std::string layer_str = gguf_name.substr(4, dot1 - 4);
        size_t dot2 = gguf_name.find('.', dot1 + 1);
        if (dot2 == std::string::npos) {
            return "";
        }
        std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 - 1);
        std::string suffix = gguf_name.substr(dot2 + 1);
        std::string mapped_op;
        if (suffix == "bias") {
            if (op == "attn_q") {
                mapped_op = "bq";
            } else if (op == "attn_k") {
                mapped_op = "bk";
            } else if (op == "attn_v") {
                mapped_op = "bv";
            } else {
                return "";
            }
        } else if (suffix != "weight") {
            return "";
        } else if (op == "attn_norm") {
            mapped_op = "attention_norm";
        } else if (op == "attn_q") {
            mapped_op = "wq";
        } else if (op == "attn_k") {
            mapped_op = "wk";
        } else if (op == "attn_v") {
            mapped_op = "wv";
        } else if (op == "attn_output") {
            mapped_op = "wo";
        } else if (op == "ffn_norm") {
            mapped_op = "ffn_norm";
        } else if (op == "ffn_gate") {
            mapped_op = "w_gate";
        } else if (op == "ffn_up") {
            mapped_op = "w_up";
        } else if (op == "ffn_down") {
            mapped_op = "w_down";
        } else {
            return "";
        }

        return "layers." + layer_str + "." + mapped_op;
    }

    return "";
}

// ---------------------------------------------------------------------------
// Convert GGUF shape to our row-major Tensor view.
// GGUF uses ne0 as the fastest-changing dimension. For a 2D weight advertised
// as [in_features, out_features], the raw bytes are already laid out as
// out_features rows of in_features values. Our Tensor wants [out, in].
// ---------------------------------------------------------------------------

static std::vector<int> map_shape(const std::vector<uint64_t>& gguf_shape) {
    std::vector<int> shape;
    shape.reserve(gguf_shape.size());
    if (gguf_shape.size() == 2) {
        shape.push_back(static_cast<int>(gguf_shape[1]));
        shape.push_back(static_cast<int>(gguf_shape[0]));
        return shape;
    }
    for (uint64_t dim : gguf_shape) {
        shape.push_back(static_cast<int>(dim));
    }
    return shape;
}

// ---------------------------------------------------------------------------
// Load quantized tensor from raw bytes (F32, Q8_0, Q4_0, Q4_1)
// ---------------------------------------------------------------------------

static QuantizedTensor load_quantized_tensor(const std::vector<uint8_t>& raw,
                                              const std::vector<int>& shape,
                                              int ggml_type) {
    size_t numel = 1;
    for (int d : shape) {
        numel *= static_cast<size_t>(d);
    }

    QuantizedTensor q;
    q.shape = shape;

    switch (ggml_type) {
        case GGML_TYPE_F32: {
            q.type = QuantType::F32;
            size_t expected_bytes = numel * sizeof(float);
            if (raw.size() != expected_bytes) {
                throw std::runtime_error(
                    "F32 tensor size mismatch: expected " + std::to_string(expected_bytes) +
                    " bytes, got " + std::to_string(raw.size())
                );
            }
            q.f32_data.resize(numel);
            std::memcpy(q.f32_data.data(), raw.data(), expected_bytes);
            break;
        }
        case GGML_TYPE_Q8_0: {
            q.type = QuantType::Q8_0;
            size_t n_blocks = (numel + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
            size_t expected_bytes = n_blocks * sizeof(BlockQ8_0);
            if (raw.size() != expected_bytes) {
                throw std::runtime_error(
                    "Q8_0 tensor size mismatch: expected " + std::to_string(expected_bytes) +
                    " bytes, got " + std::to_string(raw.size())
                );
            }
            q.q8_0_data.resize(n_blocks);
            std::memcpy(q.q8_0_data.data(), raw.data(), expected_bytes);
            break;
        }
        case GGML_TYPE_Q4_0: {
            q.type = QuantType::Q4_0;
            size_t n_blocks = (numel + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
            size_t expected_bytes = n_blocks * sizeof(BlockQ4_0);
            if (raw.size() != expected_bytes) {
                throw std::runtime_error(
                    "Q4_0 tensor size mismatch: expected " + std::to_string(expected_bytes) +
                    " bytes, got " + std::to_string(raw.size())
                );
            }
            q.q4_0_data.resize(n_blocks);
            std::memcpy(q.q4_0_data.data(), raw.data(), expected_bytes);
            break;
        }
        case GGML_TYPE_Q4_1: {
            q.type = QuantType::Q4_1;
            size_t n_blocks = (numel + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE;
            size_t expected_bytes = n_blocks * sizeof(BlockQ4_1);
            if (raw.size() != expected_bytes) {
                throw std::runtime_error(
                    "Q4_1 tensor size mismatch: expected " + std::to_string(expected_bytes) +
                    " bytes, got " + std::to_string(raw.size())
                );
            }
            q.q4_1_data.resize(n_blocks);
            std::memcpy(q.q4_1_data.data(), raw.data(), expected_bytes);
            break;
        }
        default:
            throw std::runtime_error(
                "Unsupported GGUF tensor type: " + std::to_string(ggml_type)
            );
    }

    return q;
}

// ---------------------------------------------------------------------------
// Shape validation helpers
// ---------------------------------------------------------------------------

static void require_shape(const Tensor& tensor, const std::vector<int>& expected, const std::string& name) {
    tensor.assert_shape(expected, name.c_str());
}

static void require_shape(const QuantizedTensor& tensor, const std::vector<int>& expected, const std::string& name) {
    tensor.assert_shape(expected, name.c_str());
}

static void require_optional_shape(const Tensor& tensor, const std::vector<int>& expected, const std::string& name) {
    if (tensor.data.empty()) {
        return;
    }
    require_shape(tensor, expected, name);
}

static void validate_loaded_shapes(const MiniLlamaModel& model) {
    const ModelConfig& c = model.config;
    const int q_dim = c.n_heads * c.head_dim;
    const int kv_dim = c.n_kv_heads * c.head_dim;

    require_shape(model.token_embedding, {c.vocab_size, c.dim}, "GGUF token_embedding");
    require_shape(model.final_norm, {c.dim}, "GGUF final_norm");
    require_shape(model.lm_head, {c.vocab_size, c.dim}, "GGUF lm_head");

    if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
        throw std::runtime_error(
            "GGUF layer count mismatch: expected " + std::to_string(c.n_layers) +
            ", got " + std::to_string(model.layers.size())
        );
    }

    for (int layer = 0; layer < c.n_layers; ++layer) {
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "GGUF layers." + std::to_string(layer) + ".";
        require_shape(lw.attention_norm, {c.dim}, prefix + "attention_norm");
        require_shape(lw.wq, {q_dim, c.dim}, prefix + "wq");
        require_shape(lw.wk, {kv_dim, c.dim}, prefix + "wk");
        require_shape(lw.wv, {kv_dim, c.dim}, prefix + "wv");
        require_optional_shape(lw.bq, {q_dim}, prefix + "bq");
        require_optional_shape(lw.bk, {kv_dim}, prefix + "bk");
        require_optional_shape(lw.bv, {kv_dim}, prefix + "bv");
        require_shape(lw.wo, {c.dim, q_dim}, prefix + "wo");
        require_shape(lw.ffn_norm, {c.dim}, prefix + "ffn_norm");
        require_shape(lw.w_gate, {c.hidden_dim, c.dim}, prefix + "w_gate");
        require_shape(lw.w_up, {c.hidden_dim, c.dim}, prefix + "w_up");
        require_shape(lw.w_down, {c.dim, c.hidden_dim}, prefix + "w_down");
    }
}

// ---------------------------------------------------------------------------
// Build ModelConfig from GGUF metadata
// ---------------------------------------------------------------------------

static ModelConfig build_config(const std::string& path) {
    ModelConfig config;

    auto get_u32 = [&](const std::string& key, int fallback) -> int {
        int64_t v = 0;
        if (gguf_get_metadata_int(path, key, v)) {
            return static_cast<int>(v);
        }
        return fallback;
    };

    auto get_f32 = [&](const std::string& key, float fallback) -> float {
        double v = 0.0;
        if (gguf_get_metadata_float(path, key, v)) {
            return static_cast<float>(v);
        }
        return fallback;
    };

    // Try qwen2.* keys first, then llama.* keys as fallback
    std::string arch;
    if (gguf_get_metadata_string(path, "general.architecture", arch) && arch == "qwen2") {
        config.rope_type = RopeType::NeoX;
    }

    config.n_layers    = get_u32("qwen2.block_count", get_u32("llama.block_count", 0));
    config.dim         = get_u32("qwen2.embedding_length", get_u32("llama.embedding_length", 0));
    config.hidden_dim  = get_u32("qwen2.feed_forward_length", get_u32("llama.feed_forward_length", 0));
    config.n_heads     = get_u32("qwen2.attention.head_count", get_u32("llama.attention.head_count", 0));
    config.n_kv_heads  = get_u32("qwen2.attention.head_count_kv", get_u32("llama.attention.head_count_kv", 0));
    config.max_seq_len = get_u32("qwen2.context_length", get_u32("llama.context_length", 0));
    config.rope_theta  = get_f32("qwen2.rope.freq_base", get_f32("llama.rope.freq_base", 10000.0f));
    config.rms_norm_eps = get_f32("qwen2.attention.layer_norm_rms_epsilon",
                                   get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f));

    // vocab_size from tokenizer.ggml.tokens array length
    std::vector<std::string> tokens;
    if (gguf_get_metadata_string_array(path, "tokenizer.ggml.tokens", tokens)) {
        config.vocab_size = static_cast<int>(tokens.size());
    }

    // Validate
    if (config.n_layers <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.n_heads <= 0 || config.n_kv_heads <= 0 || config.max_seq_len <= 0 ||
        config.vocab_size <= 0) {
        throw std::runtime_error(
            "Invalid model config from GGUF: dim=" + std::to_string(config.dim) +
            " layers=" + std::to_string(config.n_layers) +
            " heads=" + std::to_string(config.n_heads) +
            " vocab=" + std::to_string(config.vocab_size)
        );
    }
    if (config.dim % config.n_heads != 0) {
        throw std::runtime_error("dim not divisible by n_heads");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads not divisible by n_kv_heads");
    }

    config.head_dim = config.dim / config.n_heads;

    std::cout << "GGUF config: vocab=" << config.vocab_size
              << " dim=" << config.dim
              << " layers=" << config.n_layers
              << " heads=" << config.n_heads
              << " kv_heads=" << config.n_kv_heads
              << " head_dim=" << config.head_dim
              << " hidden_dim=" << config.hidden_dim
              << " max_seq_len=" << config.max_seq_len
              << " rope_theta=" << config.rope_theta
              << " rms_norm_eps=" << config.rms_norm_eps
              << std::endl;

    return config;
}

// ---------------------------------------------------------------------------
// load_gguf_model
// ---------------------------------------------------------------------------

MiniLlamaModel load_gguf_model(const std::string& gguf_path) {
    MiniLlamaModel model;

    // Parse header
    GGUFReader reader;
    if (!reader.load(gguf_path)) {
        model.load_error = "GGUF parse error: " + reader.load_error;
        return model;
    }

    // Build config
    try {
        model.config = build_config(gguf_path);
    } catch (const std::exception& e) {
        model.load_error = std::string("Failed to build config: ") + e.what();
        return model;
    }

    const ModelConfig& c = model.config;
    std::cout << "GGUF config: vocab=" << c.vocab_size
              << " dim=" << c.dim
              << " layers=" << c.n_layers
              << " heads=" << c.n_heads
              << " kv_heads=" << c.n_kv_heads
              << " head_dim=" << c.head_dim
              << " hidden_dim=" << c.hidden_dim
              << " max_seq_len=" << c.max_seq_len
              << std::endl;

    // Index tensors by mapped name
    std::map<std::string, const GGUFTensorInfo*> tensor_map;
    for (const auto& t : reader.tensors) {
        std::string mapped = map_tensor_name(t.name);
        if (mapped.empty()) {
            continue;
        }
        tensor_map[mapped] = &t;
    }

    // Helper to load a weight tensor (returns QuantizedTensor)
    auto load_weight = [&](const std::string& mapped_name) -> QuantizedTensor {
        auto it = tensor_map.find(mapped_name);
        if (it == tensor_map.end()) {
            throw std::runtime_error("Missing tensor: " + mapped_name);
        }
        const GGUFTensorInfo* info = it->second;
        std::vector<int> shape = map_shape(info->shape);

        std::vector<uint8_t> raw = read_gguf_tensor_raw(gguf_path, *info, reader.data_offset);
        if (raw.empty() && info->type != GGML_TYPE_F32) {
            throw std::runtime_error("Failed to read tensor: " + mapped_name);
        }

        return load_quantized_tensor(raw, shape, info->type);
    };

    auto load_optional_weight = [&](const std::string& mapped_name) -> QuantizedTensor {
        if (tensor_map.find(mapped_name) == tensor_map.end()) {
            return QuantizedTensor();
        }
        return load_weight(mapped_name);
    };

    // Helper to load an F32 tensor (norm, embedding, bias).
    // If the source is quantized, dequantize to F32 on-the-fly.
    auto load_f32 = [&](const std::string& mapped_name) -> Tensor {
        QuantizedTensor q = load_weight(mapped_name);
        switch (q.type) {
            case QuantType::F32:
                return to_tensor(q);
            case QuantType::Q8_0:
                return dequantize_from_q8_0(q.q8_0_data, q.shape);
            case QuantType::Q4_0:
                return dequantize_from_q4_0(q.q4_0_data, q.shape);
            case QuantType::Q4_1:
                return dequantize_from_q4_1(q.q4_1_data, q.shape);
        }
        throw std::runtime_error("Cannot convert tensor to F32: " + mapped_name);
    };

    auto load_optional_f32 = [&](const std::string& mapped_name) -> Tensor {
        if (tensor_map.find(mapped_name) == tensor_map.end()) {
            return Tensor();
        }
        return load_f32(mapped_name);
    };

    try {
        // Token embedding (F32)
        model.token_embedding = load_f32("token_embedding");

        // Layers
        model.layers.resize(c.n_layers);
        for (int layer = 0; layer < c.n_layers; ++layer) {
            LayerWeights& lw = model.layers[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            lw.attention_norm = load_f32(prefix + "attention_norm");
            lw.wq = load_weight(prefix + "wq");
            lw.wk = load_weight(prefix + "wk");
            lw.wv = load_weight(prefix + "wv");
            lw.bq = load_optional_f32(prefix + "bq");
            lw.bk = load_optional_f32(prefix + "bk");
            lw.bv = load_optional_f32(prefix + "bv");
            lw.wo = load_weight(prefix + "wo");
            lw.ffn_norm = load_f32(prefix + "ffn_norm");
            lw.w_gate = load_weight(prefix + "w_gate");
            lw.w_up = load_weight(prefix + "w_up");
            lw.w_down = load_weight(prefix + "w_down");
        }

        // Final norm (F32)
        model.final_norm = load_f32("final_norm");

        // LM head: use output.weight if present, otherwise tie with token_embedding
        auto it = tensor_map.find("lm_head");
        if (it != tensor_map.end()) {
            model.lm_head = load_weight("lm_head");
        } else {
            // Copy token_embedding for tied weights
            model.lm_head = to_quantized_tensor(model.token_embedding);
        }

        validate_loaded_shapes(model);
    } catch (const std::exception& e) {
        model.load_error = std::string("Failed to load GGUF weights: ") + e.what();
        return model;
    }

    model.loaded = true;
    std::cout << "GGUF model loaded successfully from " << gguf_path << std::endl;
    return model;
}

} // namespace mini_llama
