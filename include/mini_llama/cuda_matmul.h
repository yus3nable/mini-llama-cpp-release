#pragma once

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"
#include <vector>

namespace mini_llama {

bool cuda_matmul_built();

// F32 matrix multiplication on CUDA using cuBLAS.
// A: [M, K], B: [K, N], result: [M, N]
Tensor cuda_matmul(const Tensor& A, const Tensor& B, int device_id = 0);

// F32 linear projection on CUDA using cuBLAS.
// x: [in_features] or [batch, in_features]
// W: [out_features, in_features]
// bias: optional [out_features], applied after the cuBLAS matmul
// result: [out_features] for 1D input, [batch, out_features] for 2D input
Tensor cuda_linear(const Tensor& x, const Tensor& W, const Tensor* bias = nullptr, int device_id = 0);

// F32 linear projection using a weight matrix that already lives in CUDA device
// memory. The input activation is copied host->device, and the result is copied
// device->host.
Tensor cuda_linear_device_weight(
    const Tensor& x,
    const void* w_device,
    const std::vector<int>& w_shape,
    const Tensor* bias = nullptr,
    int device_id = 0
);

CudaTensor cuda_linear_device_input(
    const CudaTensor& x,
    const void* w_device,
    const std::vector<int>& w_shape,
    int device_id = 0
);

} // namespace mini_llama
