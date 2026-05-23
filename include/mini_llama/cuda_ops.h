#pragma once

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool cuda_ops_built();

Tensor cuda_rmsnorm(const Tensor& x, const Tensor& weight, float eps, int device_id = 0);

Tensor cuda_silu(const Tensor& x, int device_id = 0);

Tensor cuda_elementwise_mul(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor cuda_elementwise_add(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor cuda_softmax(const Tensor& x, int device_id = 0);

CudaTensor cuda_embedding_lookup_device_weight(
    const void* embedding_data,
    const std::vector<int>& embedding_shape,
    int token_id,
    int device_id = 0
);

CudaTensor cuda_rmsnorm_device_input(
    const CudaTensor& x,
    const Tensor& weight,
    float eps,
    int device_id = 0
);

CudaTensor cuda_rmsnorm_device_weight(
    const CudaTensor& x,
    const void* weight_data,
    const std::vector<int>& weight_shape,
    float eps,
    int device_id = 0
);

CudaTensor cuda_silu_device_input(const CudaTensor& x, int device_id = 0);

CudaTensor cuda_elementwise_mul_device_input(const CudaTensor& a, const CudaTensor& b, int device_id = 0);

CudaTensor cuda_elementwise_add_device_input(const CudaTensor& a, const CudaTensor& b, int device_id = 0);

CudaTensor cuda_elementwise_add_device_weight(
    const CudaTensor& a,
    const void* b_data,
    const std::vector<int>& b_shape,
    int device_id = 0
);

void cuda_rope(
    Tensor& q,
    Tensor& k,
    int pos,
    float theta,
    RopeType rope_type = RopeType::Normal,
    int device_id = 0
);

void cuda_rope_device_input(
    CudaTensor& q,
    CudaTensor& k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float theta,
    RopeType rope_type = RopeType::Normal,
    int device_id = 0
);

} // namespace mini_llama
