#include "mini_llama/cuda_kv_cache.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

namespace {

std::runtime_error cuda_kv_cache_not_built_error() {
    return std::runtime_error(
        "CUDA KV cache was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine."
    );
}

void require_positive(int value, const char* name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("CudaKVCache: ") + name + " must be positive");
    }
}

size_t checked_mul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error(std::string("CudaKVCache: size overflow for ") + label);
    }
    return a * b;
}

} // namespace

bool cuda_kv_cache_built() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

CudaKVCache::CudaKVCache(
    int n_layers,
    int max_seq_len,
    int n_kv_heads,
    int head_dim,
    int device_id
) {
    reset(n_layers, max_seq_len, n_kv_heads, head_dim, device_id);
}

void CudaKVCache::reset() {
    keys_.reset();
    values_.reset();
    n_layers_ = 0;
    max_seq_len_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    device_id_ = 0;
}

void CudaKVCache::reset(
    int n_layers,
    int max_seq_len,
    int n_kv_heads,
    int head_dim,
    int device_id
) {
#ifdef MINI_LLAMA_USE_CUDA
    require_positive(n_layers, "n_layers");
    require_positive(max_seq_len, "max_seq_len");
    require_positive(n_kv_heads, "n_kv_heads");
    require_positive(head_dim, "head_dim");

    size_t elements = static_cast<size_t>(n_layers);
    elements = checked_mul(elements, static_cast<size_t>(max_seq_len), "layers * seq");
    elements = checked_mul(elements, static_cast<size_t>(n_kv_heads), "layers * seq * heads");
    elements = checked_mul(elements, static_cast<size_t>(head_dim), "layers * seq * heads * dim");
    size_t bytes_per_cache = checked_mul(elements, sizeof(float), "cache bytes");

    n_layers_ = n_layers;
    max_seq_len_ = max_seq_len;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    device_id_ = device_id;
    keys_.reset(bytes_per_cache, device_id);
    values_.reset(bytes_per_cache, device_id);
    clear();
#else
    (void)n_layers;
    (void)max_seq_len;
    (void)n_kv_heads;
    (void)head_dim;
    (void)device_id;
    throw cuda_kv_cache_not_built_error();
#endif
}

void CudaKVCache::clear() {
#ifdef MINI_LLAMA_USE_CUDA
    if (empty()) {
        return;
    }
    cuda_set_device(device_id_);
    std::vector<float> zeros(keys_.bytes() / sizeof(float), 0.0f);
    keys_.upload(zeros.data(), keys_.bytes());
    values_.upload(zeros.data(), values_.bytes());
#else
    throw cuda_kv_cache_not_built_error();
#endif
}

void CudaKVCache::write(int layer, int pos, const Tensor& k, const Tensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
    validate_indices(layer, pos);
    validate_write_tensors(k, v);
    size_t offset = slot_offset_bytes(layer, pos);
    size_t bytes_to_copy = k.size() * sizeof(float);
    char* key_dst = static_cast<char*>(keys_.data()) + offset;
    char* value_dst = static_cast<char*>(values_.data()) + offset;
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(key_dst, k.data.data(), bytes_to_copy, CudaMemcpyKind::HostToDevice);
    cuda_memcpy_bytes(value_dst, v.data.data(), bytes_to_copy, CudaMemcpyKind::HostToDevice);
#else
    (void)layer;
    (void)pos;
    (void)k;
    (void)v;
    throw cuda_kv_cache_not_built_error();
#endif
}

void CudaKVCache::write_device(int layer, int pos, const CudaTensor& k, const CudaTensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
    validate_indices(layer, pos);
    if (k.device_id() != device_id_ || v.device_id() != device_id_) {
        throw std::runtime_error("CudaKVCache::write_device tensor is on a different CUDA device");
    }
    const size_t expected = static_cast<size_t>(n_kv_heads_) * static_cast<size_t>(head_dim_);
    if (k.size() != expected || v.size() != expected) {
        throw std::out_of_range("CudaKVCache::write_device tensor shape does not match cache shape");
    }
    size_t offset = slot_offset_bytes(layer, pos);
    size_t bytes_to_copy = expected * sizeof(float);
    char* key_dst = static_cast<char*>(keys_.data()) + offset;
    char* value_dst = static_cast<char*>(values_.data()) + offset;
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(key_dst, k.data(), bytes_to_copy, CudaMemcpyKind::DeviceToDevice);
    cuda_memcpy_bytes(value_dst, v.data(), bytes_to_copy, CudaMemcpyKind::DeviceToDevice);
#else
    (void)layer;
    (void)pos;
    (void)k;
    (void)v;
    throw cuda_kv_cache_not_built_error();
#endif
}

Tensor CudaKVCache::read_key(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
    validate_indices(layer, pos);
    Tensor out({n_kv_heads_, head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(keys_.data()) + slot_offset_bytes(layer, pos);
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::DeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    throw cuda_kv_cache_not_built_error();
#endif
}

Tensor CudaKVCache::read_value(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
    validate_indices(layer, pos);
    Tensor out({n_kv_heads_, head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(values_.data()) + slot_offset_bytes(layer, pos);
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::DeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    throw cuda_kv_cache_not_built_error();
#endif
}

Tensor CudaKVCache::read_key_head(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
    validate_indices(layer, pos);
    validate_head_index(kv_head);
    Tensor out({head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(keys_.data()) + head_offset_bytes(layer, pos, kv_head);
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::DeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    (void)kv_head;
    throw cuda_kv_cache_not_built_error();
#endif
}

Tensor CudaKVCache::read_value_head(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
    validate_indices(layer, pos);
    validate_head_index(kv_head);
    Tensor out({head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(values_.data()) + head_offset_bytes(layer, pos, kv_head);
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::DeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    (void)kv_head;
    throw cuda_kv_cache_not_built_error();
#endif
}

const void* CudaKVCache::keys_data() const {
    if (empty()) {
        throw std::runtime_error("CudaKVCache: cache is empty");
    }
    return keys_.data();
}

const void* CudaKVCache::values_data() const {
    if (empty()) {
        throw std::runtime_error("CudaKVCache: cache is empty");
    }
    return values_.data();
}

bool CudaKVCache::empty() const {
    return keys_.empty() || values_.empty();
}

size_t CudaKVCache::bytes() const {
    return keys_.bytes() + values_.bytes();
}

void CudaKVCache::validate_indices(int layer, int pos) const {
    if (empty()) {
        throw std::runtime_error("CudaKVCache: cache is empty");
    }
    if (layer < 0 || layer >= n_layers_) {
        throw std::out_of_range("CudaKVCache layer out of range");
    }
    if (pos < 0 || pos >= max_seq_len_) {
        throw std::out_of_range("CudaKVCache position out of range");
    }
}

void CudaKVCache::validate_head_index(int kv_head) const {
    if (kv_head < 0 || kv_head >= n_kv_heads_) {
        throw std::out_of_range("CudaKVCache head out of range");
    }
}

void CudaKVCache::validate_write_tensors(const Tensor& k, const Tensor& v) const {
    if (k.ndim() != 2 || v.ndim() != 2 || k.shape != v.shape) {
        throw std::out_of_range("CudaKVCache::write expected matching [n_kv_heads, head_dim] tensors");
    }
    if (k.shape[0] != n_kv_heads_ || k.shape[1] != head_dim_) {
        throw std::out_of_range("CudaKVCache::write tensor shape does not match cache shape");
    }
}

size_t CudaKVCache::slot_offset_bytes(int layer, int pos) const {
    size_t slot = static_cast<size_t>(layer);
    slot = slot * static_cast<size_t>(max_seq_len_) + static_cast<size_t>(pos);
    slot = slot * static_cast<size_t>(n_kv_heads_) * static_cast<size_t>(head_dim_);
    return slot * sizeof(float);
}

size_t CudaKVCache::head_offset_bytes(int layer, int pos, int kv_head) const {
    size_t offset = slot_offset_bytes(layer, pos);
    offset += static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim_) * sizeof(float);
    return offset;
}

} // namespace mini_llama
