#pragma once

#include "mini_llama/model.h"
#include <string>
#include <map>
#include <vector>

namespace mini_llama {

struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t offset = 0;
    size_t byte_size = 0;
};

struct TokenizerInfo {
    std::string type = "ascii";
    std::string path;
    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;
};

struct ModelManifest {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensors;
};

// Parse model.json into a manifest (config + tensor metadata).
// Throws std::runtime_error when the manifest is incomplete or inconsistent.
ModelManifest parse_manifest(const std::string& config_path);

// Load model weights by looking up tensors in the manifest by name.
MiniLlamaModel load_model(const std::string& config_path, const std::string& weights_path);

// Inspect and print model metadata without loading weights.
bool inspect_model(const std::string& config_path);

} // namespace mini_llama
