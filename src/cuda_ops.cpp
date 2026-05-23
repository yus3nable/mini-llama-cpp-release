#include "mini_llama/cuda_ops.h"

#include <stdexcept>

namespace mini_llama {

namespace {

std::runtime_error cuda_ops_not_built_error() {
    return std::runtime_error(
        "CUDA ops were not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine."
    );
}

} // namespace

bool cuda_ops_built() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA

Tensor cuda_rmsnorm(const Tensor& x, const Tensor& weight, float eps, int device_id) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

Tensor cuda_silu(const Tensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

Tensor cuda_elementwise_mul(const Tensor& a, const Tensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

CudaTensor cuda_embedding_lookup_device_weight(
    const void* embedding_data,
    const std::vector<int>& embedding_shape,
    int token_id,
    int device_id
) {
    (void)embedding_data;
    (void)embedding_shape;
    (void)token_id;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

CudaTensor cuda_rmsnorm_device_input(const CudaTensor& x, const Tensor& weight, float eps, int device_id) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

CudaTensor cuda_rmsnorm_device_weight(
    const CudaTensor& x,
    const void* weight_data,
    const std::vector<int>& weight_shape,
    float eps,
    int device_id
) {
    (void)x;
    (void)weight_data;
    (void)weight_shape;
    (void)eps;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

CudaTensor cuda_silu_device_input(const CudaTensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

CudaTensor cuda_elementwise_mul_device_input(const CudaTensor& a, const CudaTensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

CudaTensor cuda_elementwise_add_device_input(const CudaTensor& a, const CudaTensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

CudaTensor cuda_elementwise_add_device_weight(
    const CudaTensor& a,
    const void* b_data,
    const std::vector<int>& b_shape,
    int device_id
) {
    (void)a;
    (void)b_data;
    (void)b_shape;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

Tensor cuda_elementwise_add(const Tensor& a, const Tensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

Tensor cuda_softmax(const Tensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

void cuda_rope(
    Tensor& q,
    Tensor& k,
    int pos,
    float theta,
    RopeType rope_type,
    int device_id
) {
    (void)q;
    (void)k;
    (void)pos;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

void cuda_rope_device_input(
    CudaTensor& q,
    CudaTensor& k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float theta,
    RopeType rope_type,
    int device_id
) {
    (void)q;
    (void)k;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)pos;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw cuda_ops_not_built_error();
}

#endif

} // namespace mini_llama
