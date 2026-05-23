#pragma once

#include "mini_llama/gguf.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama::test {

inline std::filesystem::path gguf_tokenizer_temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

inline void tok_append_u32_le(std::vector<char>& data, uint32_t value) {
    data.push_back(static_cast<char>(value & 0xff));
    data.push_back(static_cast<char>((value >> 8) & 0xff));
    data.push_back(static_cast<char>((value >> 16) & 0xff));
    data.push_back(static_cast<char>((value >> 24) & 0xff));
}

inline void tok_append_u64_le(std::vector<char>& data, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
    }
}

inline void tok_append_f32_le(std::vector<char>& data, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    tok_append_u32_le(data, bits);
}

inline void tok_append_string(std::vector<char>& data, const std::string& value) {
    tok_append_u64_le(data, value.size());
    data.insert(data.end(), value.begin(), value.end());
}

inline void tok_append_metadata_string(std::vector<char>& data, const std::string& key, const std::string& value) {
    tok_append_string(data, key);
    tok_append_u32_le(data, GGUF_TYPE_STRING);
    tok_append_string(data, value);
}

inline void tok_append_metadata_u32(std::vector<char>& data, const std::string& key, uint32_t value) {
    tok_append_string(data, key);
    tok_append_u32_le(data, GGUF_TYPE_UINT32);
    tok_append_u32_le(data, value);
}

inline void tok_append_metadata_f32(std::vector<char>& data, const std::string& key, float value) {
    tok_append_string(data, key);
    tok_append_u32_le(data, GGUF_TYPE_FLOAT32);
    tok_append_f32_le(data, value);
}

inline void tok_append_string_array(std::vector<char>& data, const std::string& key, const std::vector<std::string>& values) {
    tok_append_string(data, key);
    tok_append_u32_le(data, GGUF_TYPE_ARRAY);
    tok_append_u32_le(data, GGUF_TYPE_STRING);
    tok_append_u64_le(data, static_cast<uint64_t>(values.size()));
    for (const std::string& value : values) {
        tok_append_string(data, value);
    }
}

inline void tok_append_i32_array(std::vector<char>& data, const std::string& key, const std::vector<int32_t>& values) {
    tok_append_string(data, key);
    tok_append_u32_le(data, GGUF_TYPE_ARRAY);
    tok_append_u32_le(data, GGUF_TYPE_INT32);
    tok_append_u64_le(data, static_cast<uint64_t>(values.size()));
    for (int32_t value : values) {
        tok_append_u32_le(data, static_cast<uint32_t>(value));
    }
}

inline std::string tiny_qwen2_template() {
    return "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
}

inline std::filesystem::path write_tiny_gguf_tokenizer_fixture(
    const std::string& name,
    const std::string& architecture = "qwen2",
    bool include_chat_template = true
) {
    const std::vector<std::string> tokens = {
        "a",
        "b",
        "ab",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>",
        "u",
        "s",
        "e",
        "r",
        "system",
        "assistant",
        "\n",
    };
    const std::vector<int32_t> token_types = {
        1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1,
    };
    const std::vector<std::string> merges = {
        "a b",
    };

    const uint64_t metadata_count = include_chat_template ? 16 : 15;
    std::vector<char> data;
    data.insert(data.end(), {'G', 'G', 'U', 'F'});
    tok_append_u32_le(data, 3);
    tok_append_u64_le(data, 0);
    tok_append_u64_le(data, metadata_count);

    tok_append_metadata_string(data, "general.architecture", architecture);
    tok_append_metadata_string(data, "general.name", "tiny-tokenizer-test");
    tok_append_metadata_u32(data, "general.alignment", 32);
    tok_append_metadata_u32(data, architecture + ".block_count", 1);
    tok_append_metadata_u32(data, architecture + ".context_length", 16);
    tok_append_metadata_u32(data, architecture + ".embedding_length", 8);
    tok_append_metadata_u32(data, architecture + ".feed_forward_length", 16);
    tok_append_metadata_u32(data, architecture + ".attention.head_count", 1);
    tok_append_metadata_u32(data, architecture + ".attention.head_count_kv", 1);
    tok_append_metadata_f32(data, architecture + ".attention.layer_norm_rms_epsilon", 1e-5f);
    tok_append_string_array(data, "tokenizer.ggml.tokens", tokens);
    tok_append_i32_array(data, "tokenizer.ggml.token_type", token_types);
    tok_append_string_array(data, "tokenizer.ggml.merges", merges);
    tok_append_metadata_u32(data, "tokenizer.ggml.bos_token_id", 3);
    tok_append_metadata_u32(data, "tokenizer.ggml.eos_token_id", 4);
    if (include_chat_template) {
        tok_append_metadata_string(data, "tokenizer.chat_template", tiny_qwen2_template());
    }

    size_t padding = (32 - (data.size() % 32)) % 32;
    data.insert(data.end(), padding, '\0');

    std::filesystem::path path = gguf_tokenizer_temp_path(name);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to create tokenizer fixture: " + path.string());
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    return path;
}

} // namespace mini_llama::test
