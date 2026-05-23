#include "mini_llama/cuda_attention.h"

#include <stdexcept>

namespace mini_llama {

namespace {

std::runtime_error cuda_attention_not_built_error() {
    return std::runtime_error(
        "CUDA attention was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine."
    );
}

} // namespace

bool cuda_attention_built() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA
Tensor cuda_attention_decode(
    const Tensor& q,
    const CudaKVCache& kv_cache,
    int layer,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int device_id
) {
    (void)q;
    (void)kv_cache;
    (void)layer;
    (void)pos;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)device_id;
    throw cuda_attention_not_built_error();
}

CudaTensor cuda_attention_decode_device_input(
    const CudaTensor& q,
    const CudaKVCache& kv_cache,
    int layer,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int device_id
) {
    (void)q;
    (void)kv_cache;
    (void)layer;
    (void)pos;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)device_id;
    throw cuda_attention_not_built_error();
}
#endif

} // namespace mini_llama
