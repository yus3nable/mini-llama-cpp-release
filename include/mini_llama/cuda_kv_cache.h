#pragma once

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

#include <cstddef>

namespace mini_llama {

bool cuda_kv_cache_built();

class CudaKVCache {
public:
    CudaKVCache() = default;
    CudaKVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);

    CudaKVCache(const CudaKVCache&) = delete;
    CudaKVCache& operator=(const CudaKVCache&) = delete;
    CudaKVCache(CudaKVCache&&) noexcept = default;
    CudaKVCache& operator=(CudaKVCache&&) noexcept = default;

    void reset();
    void reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);
    void clear();

    void write(int layer, int pos, const Tensor& k, const Tensor& v);
    void write_device(int layer, int pos, const CudaTensor& k, const CudaTensor& v);

    Tensor read_key(int layer, int pos) const;
    Tensor read_value(int layer, int pos) const;
    Tensor read_key_head(int layer, int pos, int kv_head) const;
    Tensor read_value_head(int layer, int pos, int kv_head) const;

    const void* keys_data() const;
    const void* values_data() const;

    bool empty() const;
    size_t bytes() const;

    int n_layers() const {
        return n_layers_;
    }

    int max_seq_len() const {
        return max_seq_len_;
    }

    int n_kv_heads() const {
        return n_kv_heads_;
    }

    int head_dim() const {
        return head_dim_;
    }

    int device_id() const {
        return device_id_;
    }

private:
    void validate_indices(int layer, int pos) const;
    void validate_head_index(int kv_head) const;
    void validate_write_tensors(const Tensor& k, const Tensor& v) const;
    size_t slot_offset_bytes(int layer, int pos) const;
    size_t head_offset_bytes(int layer, int pos, int kv_head) const;

    int n_layers_ = 0;
    int max_seq_len_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int device_id_ = 0;
    CudaDeviceBuffer keys_;
    CudaDeviceBuffer values_;
};

} // namespace mini_llama
