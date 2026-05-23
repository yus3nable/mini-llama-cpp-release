#pragma once

#include "mini_llama/model.h"
#include "mini_llama/context.h"
#include "mini_llama/batch.h"

namespace mini_llama {

// Forward pass for a single token
// Returns logits: [vocab_size]
Tensor forward_token(
    const MiniLlamaModel& model,
    MiniLlamaContext& ctx,
    int token
);

// Forward pass for a batch of tokens.
// Internally processes tokens sequentially and returns logits for the last token.
// This unifies prefill (multi-token) and decode (single-token) paths.
Tensor forward_batch(
    const MiniLlamaModel& model,
    MiniLlamaContext& ctx,
    const MiniBatch& batch
);

} // namespace mini_llama
