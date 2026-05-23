#pragma once

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool cuda_attention_built();

Tensor cuda_attention_decode(
    const Tensor& q,
    const CudaKVCache& kv_cache,
    int layer,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int device_id = 0
);

CudaTensor cuda_attention_decode_device_input(
    const CudaTensor& q,
    const CudaKVCache& kv_cache,
    int layer,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int device_id = 0
);

} // namespace mini_llama
