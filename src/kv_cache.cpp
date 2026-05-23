#include "mini_llama/kv_cache.h"
#include <cstring>
#include <stdexcept>
#include <string>

namespace mini_llama {

static void require_4d_cache_shape(const Tensor& t, const char* name) {
    if (t.ndim() != 4) {
        throw std::runtime_error(std::string(name) + " must be a 4D tensor");
    }
}

KVCache::KVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    keys = make_tensor_4d(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
    values = make_tensor_4d(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

void KVCache::write(int layer, int pos, const Tensor& k, const Tensor& v) {
    // k: [n_kv_heads, head_dim]
    // v: [n_kv_heads, head_dim]
    require_4d_cache_shape(keys, "keys");
    require_4d_cache_shape(values, "values");
    if (k.ndim() != 2 || v.ndim() != 2 || k.shape != v.shape) {
        throw std::out_of_range("KVCache::write expected matching [n_kv_heads, head_dim] tensors");
    }
    int n_kv_heads = k.shape[0];
    int head_dim = k.shape[1];
    if (layer < 0 || layer >= keys.shape[0]) {
        throw std::out_of_range("KVCache::write layer out of range");
    }
    if (pos < 0 || pos >= keys.shape[1]) {
        throw std::out_of_range("KVCache::write position out of range");
    }
    if (n_kv_heads != keys.shape[2] || head_dim != keys.shape[3]) {
        throw std::out_of_range("KVCache::write tensor shape does not match cache shape");
    }

    // keys shape: [n_layers, max_seq_len, n_kv_heads, head_dim]
    // We need to write at [layer, pos, :, :]
    size_t layer_stride = static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t head_stride = static_cast<size_t>(head_dim);

    size_t base = layer * layer_stride + pos * pos_stride;

    for (int h = 0; h < n_kv_heads; ++h) {
        size_t offset = base + h * head_stride;
        std::memcpy(&keys.data[offset], &k.data[h * head_dim], head_dim * sizeof(float));
        std::memcpy(&values.data[offset], &v.data[h * head_dim], head_dim * sizeof(float));
    }
}

const float* KVCache::key_ptr(int layer, int pos, int kv_head) const {
    require_4d_cache_shape(keys, "keys");
    if (layer < 0 || layer >= keys.shape[0] ||
        pos < 0 || pos >= keys.shape[1] ||
        kv_head < 0 || kv_head >= keys.shape[2]) {
        throw std::out_of_range("KVCache::key_ptr index out of range");
    }
    size_t layer_stride = static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t head_stride = static_cast<size_t>(keys.shape[3]);
    return &keys.data[layer * layer_stride + pos * pos_stride + kv_head * head_stride];
}

const float* KVCache::value_ptr(int layer, int pos, int kv_head) const {
    require_4d_cache_shape(values, "values");
    if (layer < 0 || layer >= values.shape[0] ||
        pos < 0 || pos >= values.shape[1] ||
        kv_head < 0 || kv_head >= values.shape[2]) {
        throw std::out_of_range("KVCache::value_ptr index out of range");
    }
    size_t layer_stride = static_cast<size_t>(values.shape[1] * values.shape[2] * values.shape[3]);
    size_t pos_stride = static_cast<size_t>(values.shape[2] * values.shape[3]);
    size_t head_stride = static_cast<size_t>(values.shape[3]);
    return &values.data[layer * layer_stride + pos * pos_stride + kv_head * head_stride];
}

} // namespace mini_llama
