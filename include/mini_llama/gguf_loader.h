#pragma once

#include "mini_llama/model.h"
#include <string>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Load a model from a GGUF file.
// Supports F32 and Q8_0 tensors. Dequantizes Q8_0 to F32 at load time.
// ---------------------------------------------------------------------------
MiniLlamaModel load_gguf_model(const std::string& gguf_path);

} // namespace mini_llama
