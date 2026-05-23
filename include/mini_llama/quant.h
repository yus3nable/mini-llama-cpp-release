#pragma once

#include "mini_llama/quantized_tensor.h"
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Quantize / dequantize (Q8_0)
// ---------------------------------------------------------------------------
std::vector<BlockQ8_0> quantize_to_q8_0(const Tensor& src);

// Dequantize Q8_0 blocks back to F32 tensor.
// `shape` must match the original source tensor shape.
Tensor dequantize_from_q8_0(const std::vector<BlockQ8_0>& blocks, const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// Quantize / dequantize (Q4_0)
// ---------------------------------------------------------------------------
std::vector<BlockQ4_0> quantize_to_q4_0(const Tensor& src);

// Dequantize Q4_0 blocks back to F32 tensor.
// `shape` must match the original source tensor shape.
Tensor dequantize_from_q4_0(const std::vector<BlockQ4_0>& blocks, const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// Dequantize / linear (Q4_1)
// ---------------------------------------------------------------------------
Tensor dequantize_from_q4_1(const std::vector<BlockQ4_1>& blocks, const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// True quantized linear (block-level on-the-fly)
// weight is stored in quantized format; input and output are F32.
// No temporary F32 weight array is allocated.
// ---------------------------------------------------------------------------
Tensor linear_q8_0(const Tensor& x, const std::vector<BlockQ8_0>& weight, const std::vector<int>& weight_shape);
Tensor linear_q4_0(const Tensor& x, const std::vector<BlockQ4_0>& weight, const std::vector<int>& weight_shape);
Tensor linear_q4_1(const Tensor& x, const std::vector<BlockQ4_1>& weight, const std::vector<int>& weight_shape);

// ---------------------------------------------------------------------------
// Legacy pseudo-quantized matmul (dequantizes to F32 then calls matmul)
// Kept for backward compatibility with existing tests.
// ---------------------------------------------------------------------------
Tensor matmul_q8_0(const std::vector<BlockQ8_0>& weight, const Tensor& input, const std::vector<int>& weight_shape);
Tensor matmul_q4_0(const std::vector<BlockQ4_0>& weight, const Tensor& input, const std::vector<int>& weight_shape);

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
// Compare F32 matmul vs Q8_0 matmul for the same weight + input.
// Returns max absolute error.
float compare_matmul_error(const Tensor& weight, const Tensor& input);

// Compare F32 linear vs Q4_0 linear for the same weight + input.
// Returns max absolute error.
float compare_q4_0_error(const Tensor& weight, const Tensor& input);

} // namespace mini_llama
