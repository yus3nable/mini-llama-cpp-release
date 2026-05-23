#include "mini_llama/batch.h"
#include "mini_llama/forward.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include <stdexcept>

namespace mini_llama {

MiniBatch MiniBatch::single(int token, int pos) {
    MiniBatch batch;
    batch.tokens.push_back(token);
    batch.positions.push_back(pos);
    return batch;
}

MiniBatch MiniBatch::from_tokens(const std::vector<int>& toks, int start_pos) {
    MiniBatch batch;
    batch.tokens = toks;
    batch.positions.reserve(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.positions.push_back(start_pos + static_cast<int>(i));
    }
    return batch;
}

// ---------------------------------------------------------------------------
// forward_batch
// ---------------------------------------------------------------------------
// Processes all tokens in the batch sequentially.
// Returns logits for the *last* token in the batch.
// This unifies prefill (multi-token) and decode (single-token) paths.
// ---------------------------------------------------------------------------
Tensor forward_batch(
    const MiniLlamaModel& model,
    MiniLlamaContext& ctx,
    const MiniBatch& batch
) {
    if (batch.n_tokens() == 0) {
        throw std::runtime_error("forward_batch: empty batch");
    }
    if (batch.tokens.size() != batch.positions.size()) {
        throw std::runtime_error("forward_batch: tokens and positions size mismatch");
    }

    Tensor logits;
    for (int i = 0; i < batch.n_tokens(); ++i) {
        ctx.pos = batch.positions[i];
        logits = forward_token(model, ctx, batch.tokens[i]);
        ctx.token_history.push_back(batch.tokens[i]);
    }
    return logits;
}

} // namespace mini_llama
