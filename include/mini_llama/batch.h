#pragma once

#include <vector>

namespace mini_llama {

// MiniBatch mirrors llama_batch: a collection of tokens with positions.
// For this educational implementation, each token maps to one position.
struct MiniBatch {
    std::vector<int> tokens;    // token ids
    std::vector<int> positions; // position for each token

    int n_tokens() const {
        return static_cast<int>(tokens.size());
    }

    // Build a single-token batch (decode step).
    static MiniBatch single(int token, int pos);

    // Build a multi-token batch from a sequence (prefill step).
    // positions start at start_pos and increment by 1.
    static MiniBatch from_tokens(const std::vector<int>& toks, int start_pos = 0);
};

} // namespace mini_llama
