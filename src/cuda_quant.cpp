#include "mini_llama/cuda_quant.h"

#include <stdexcept>

namespace mini_llama {

namespace {

std::runtime_error cuda_quant_not_built_error() {
    return std::runtime_error(
        "CUDA quant kernels were not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine."
    );
}

} // namespace

bool cuda_quant_built() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA

Tensor cuda_q8_0_linear(
    const Tensor& x,
    const std::vector<BlockQ8_0>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

Tensor cuda_q8_0_linear_device_weight(
    const Tensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    (void)x;
    (void)q8_0_device;
    (void)q8_0_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

CudaTensor cuda_q8_0_linear_device_input(
    const CudaTensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    (void)x;
    (void)q8_0_device;
    (void)q8_0_block_count;
    (void)weight_shape;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

Tensor cuda_q4_0_linear(
    const Tensor& x,
    const std::vector<BlockQ4_0>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

Tensor cuda_q4_0_linear_device_weight(
    const Tensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    (void)x;
    (void)q4_0_device;
    (void)q4_0_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

CudaTensor cuda_q4_0_linear_device_input(
    const CudaTensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    (void)x;
    (void)q4_0_device;
    (void)q4_0_block_count;
    (void)weight_shape;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

Tensor cuda_q4_1_linear(
    const Tensor& x,
    const std::vector<BlockQ4_1>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

Tensor cuda_q4_1_linear_device_weight(
    const Tensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    (void)x;
    (void)q4_1_device;
    (void)q4_1_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

CudaTensor cuda_q4_1_linear_device_input(
    const CudaTensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    (void)x;
    (void)q4_1_device;
    (void)q4_1_block_count;
    (void)weight_shape;
    (void)device_id;
    throw cuda_quant_not_built_error();
}

#endif

} // namespace mini_llama
