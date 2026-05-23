#pragma once

#include "mini_llama/tensor.h"
#include <random>

namespace mini_llama {

// Sampling parameters.
struct SamplingParams {
    float temperature = 0.0f;
    int top_k = 0;          // 0 means disabled
    unsigned int seed = 0;  // 0 means use random device
};

// MiniSampler supports greedy, temperature, and top-k sampling.
class MiniSampler {
public:
    explicit MiniSampler(unsigned int seed = 0);
    explicit MiniSampler(const SamplingParams& params);

    // Sample a token from logits using the given parameters.
    int sample(const Tensor& logits, const SamplingParams& params);

    // Greedy sampling (always returns argmax).
    static int sample_greedy(const Tensor& logits);

    // Temperature sampling.
    int sample_temperature(const Tensor& logits, float temperature);

    // Top-k sampling (temperature applied within top-k).
    int sample_top_k(const Tensor& logits, float temperature, int top_k);

private:
    std::mt19937 rng_;
};

// Legacy free function for backward compatibility.
int sample_greedy(const Tensor& logits);

} // namespace mini_llama
