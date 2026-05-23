#pragma once

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace mini_llama {

class CudaTensor {
public:
    CudaTensor() = default;
    CudaTensor(const std::vector<int>& shape, int device_id = 0);

    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    CudaTensor(CudaTensor&&) noexcept = default;
    CudaTensor& operator=(CudaTensor&&) noexcept = default;

    void reset();
    void reset(const std::vector<int>& shape, int device_id = 0);
    void upload_from(const Tensor& src);
    Tensor download() const;
    void download_to(Tensor& dst) const;

    void* data();
    const void* data() const;

    const std::vector<int>& shape() const {
        return shape_;
    }

    int ndim() const {
        return static_cast<int>(shape_.size());
    }

    size_t size() const {
        return numel_;
    }

    size_t bytes() const {
        return buffer_.bytes();
    }

    int device_id() const {
        return device_id_;
    }

    bool empty() const {
        return buffer_.empty();
    }

    std::string shape_str() const;

private:
    std::vector<int> shape_;
    size_t numel_ = 0;
    int device_id_ = 0;
    CudaDeviceBuffer buffer_;
};

CudaTensor cuda_tensor_from_host(const Tensor& src, int device_id = 0);

} // namespace mini_llama
