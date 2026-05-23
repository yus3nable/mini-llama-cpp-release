#include "mini_llama/debug.h"
#include "mini_llama/forward.h"
#include "mini_llama/context.h"
#include "mini_llama/sampler.h"
#include "mini_llama/ops.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// dump_tensor_shape
// ---------------------------------------------------------------------------
void dump_tensor_shape(const Tensor& t, const std::string& name) {
    std::cout << "  tensor: " << name;
    std::cout << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << t.shape[i];
    }
    std::cout << "] dtype=f32 size=" << t.size() << "\n";
}

// ---------------------------------------------------------------------------
// dump_logits_topk
// ---------------------------------------------------------------------------
void dump_logits_topk(const Tensor& logits, int k) {
    if (logits.ndim() != 1) {
        std::cout << "  logits: invalid shape for top-k dump\n";
        return;
    }
    int vocab_size = logits.shape[0];
    if (k <= 0 || k > vocab_size) {
        k = vocab_size;
    }

    // Collect (value, index) pairs
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        pairs.push_back({logits.data[i], i});
    }

    // Sort descending by value
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });

    std::cout << "  logits top-" << k << ":\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "    [" << std::setw(3) << i << "] id=" << std::setw(4) << pairs[i].second
                  << " value=" << std::fixed << std::setprecision(6) << pairs[i].first << "\n";
    }
}

// ---------------------------------------------------------------------------
// dump_kv_cache_info
// ---------------------------------------------------------------------------
void dump_kv_cache_info(const KVCache& cache, int current_pos) {
    int n_layers = cache.keys.shape[0];
    int seq_len = cache.keys.shape[1];
    int n_kv_heads = cache.keys.shape[2];
    int head_dim = cache.keys.shape[3];
    std::cout << "  KV cache: layers=" << n_layers
              << " seq_len=" << seq_len
              << " n_kv_heads=" << n_kv_heads
              << " head_dim=" << head_dim
              << "\n";
    if (current_pos >= 0) {
        std::cout << "    current_pos: " << current_pos << "\n";
    }
    std::cout << "    keys shape: [" << cache.keys.shape[0] << ", " << cache.keys.shape[1]
              << ", " << cache.keys.shape[2] << ", " << cache.keys.shape[3] << "]\n";
    std::cout << "    values shape: [" << cache.values.shape[0] << ", " << cache.values.shape[1]
              << ", " << cache.values.shape[2] << ", " << cache.values.shape[3] << "]\n";
}

// ---------------------------------------------------------------------------
// run_benchmark
// ---------------------------------------------------------------------------
BenchmarkResult run_benchmark(
    const MiniLlamaModel& model,
    const std::vector<int>& prompt_tokens,
    int n_predict,
    unsigned int seed,
    bool verbose
) {
    BenchmarkResult result;
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

    MiniLlamaContext ctx(&model);
    MiniSampler sampler(seed);
    SamplingParams params;
    params.seed = seed;

    // Prefill
    if (verbose) {
        std::cout << "[verbose] prefill " << prompt_tokens.size() << " tokens\n";
    }

    auto prefill_start = std::chrono::steady_clock::now();
    MiniBatch prefill = MiniBatch::from_tokens(prompt_tokens, 0);
    Tensor logits = forward_batch(model, ctx, prefill);
    auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

    if (verbose) {
        dump_tensor_shape(logits, "logits");
        dump_logits_topk(logits, 5);
        dump_kv_cache_info(ctx.kv_cache, ctx.pos);
    }

    // Decode
    if (n_predict > 0 && result.n_prompt_tokens < model.config.max_seq_len) {
        std::vector<int> tokens = prompt_tokens;
        int next_token = sampler.sample(logits, params);
        tokens.push_back(next_token);
        ++result.n_generated_tokens;

        if (verbose) {
            std::cout << "[verbose] decode step 0: token=" << next_token << "\n";
        }

        auto decode_start = std::chrono::steady_clock::now();
        for (int i = 1; i < n_predict && static_cast<int>(tokens.size()) < model.config.max_seq_len; ++i) {
            MiniBatch decode = MiniBatch::single(tokens.back(), static_cast<int>(tokens.size() - 1));
            logits = forward_batch(model, ctx, decode);
            ++result.n_decode_tokens;
            next_token = sampler.sample(logits, params);
            tokens.push_back(next_token);
            ++result.n_generated_tokens;

            if (verbose) {
                std::cout << "[verbose] decode step " << i << ": token=" << next_token << "\n";
                dump_logits_topk(logits, 5);
            }
        }

        auto decode_end = std::chrono::steady_clock::now();
        if (result.n_decode_tokens > 0) {
            result.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
        }
    }

    if (verbose) {
        dump_kv_cache_info(ctx.kv_cache, ctx.pos);
    }

    return result;
}

} // namespace mini_llama
