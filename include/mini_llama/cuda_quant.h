#pragma once

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

#include <vector>

namespace mini_llama {

bool cuda_quant_built();

Tensor cuda_q8_0_linear(
    const Tensor& x,
    const std::vector<BlockQ8_0>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias = nullptr,
    int device_id = 0
);

Tensor cuda_q8_0_linear_device_weight(
    const Tensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias = nullptr,
    int device_id = 0
);

CudaTensor cuda_q8_0_linear_device_input(
    const CudaTensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id = 0
);

Tensor cuda_q4_0_linear(
    const Tensor& x,
    const std::vector<BlockQ4_0>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias = nullptr,
    int device_id = 0
);

Tensor cuda_q4_0_linear_device_weight(
    const Tensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias = nullptr,
    int device_id = 0
);

CudaTensor cuda_q4_0_linear_device_input(
    const CudaTensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id = 0
);

Tensor cuda_q4_1_linear(
    const Tensor& x,
    const std::vector<BlockQ4_1>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias = nullptr,
    int device_id = 0
);

Tensor cuda_q4_1_linear_device_weight(
    const Tensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias = nullptr,
    int device_id = 0
);

CudaTensor cuda_q4_1_linear_device_input(
    const CudaTensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    int device_id = 0
);

} // namespace mini_llama
