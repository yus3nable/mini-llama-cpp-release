#pragma once

#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"
#include <cmath>

namespace mini_llama {

// Matrix multiplication: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
Tensor matmul(const Tensor& A, const Tensor& B);

// Linear projection: y = x * W^T
// x: [in_features] or [1, in_features]
// W: [out_features, in_features]
// result keeps x rank: [out_features] or [1, out_features]
Tensor linear(const Tensor& x, const Tensor& W);

// Linear projection with quantized weight.
// Dispatches to F32, Q8_0, or Q4_0 path based on W.type.
Tensor linear(const Tensor& x, const QuantizedTensor& W);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// x: [dim], weight: [dim], result: [dim]
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over a 1D tensor
// x: [N], result: [N]
Tensor softmax(const Tensor& x);

// SiLU activation: silu(x) = x * sigmoid(x)
Tensor silu(const Tensor& x);

// Element-wise multiplication
Tensor elementwise_mul(const Tensor& a, const Tensor& b);

// SwiGLU: swiglu(gate, up) = silu(gate) * up
Tensor swiglu(const Tensor& gate, const Tensor& up);

// RoPE (Rotary Position Embedding) applied to Q and K
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
// pos: current token position
void rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type = RopeType::Normal);

// Argmax: return index of maximum value
int argmax(const Tensor& x);

} // namespace mini_llama
