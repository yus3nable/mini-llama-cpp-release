#pragma once

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include <vector>

namespace mini_llama {

// Inference context: holds KV cache, current position, token history, and stats.
struct MiniLlamaContext {
    const MiniLlamaModel* model = nullptr;
    KVCache kv_cache;
    CudaKVCache cuda_kv_cache;
    int pos = 0;

    // All tokens that have been fed through this context (including prefill).
    std::vector<int> token_history;

    // Stats
    int n_prefill_tokens = 0;
    int n_decode_tokens = 0;

    MiniLlamaContext() = default;
    MiniLlamaContext(const MiniLlamaContext&) = delete;
    MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
    MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
    MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
    explicit MiniLlamaContext(const MiniLlamaModel* model);
};

} // namespace mini_llama
