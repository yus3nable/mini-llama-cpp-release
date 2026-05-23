#include "mini_llama/cuda_tensor.h"

#include <limits>
#include <stdexcept>

namespace mini_llama {

namespace {

size_t checked_numel(const std::vector<int>& shape, const char* caller) {
    if (shape.empty()) {
        return 0;
    }
    size_t total = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        int dim = shape[axis];
        if (dim <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": dimension at axis " + std::to_string(axis) +
                " must be positive, got " + std::to_string(dim)
            );
        }
        size_t dim_size = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error(std::string(caller) + ": shape element count overflow");
        }
        total *= dim_size;
    }
    return total;
}

std::string shape_to_string(const std::vector<int>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

} // namespace

CudaTensor::CudaTensor(const std::vector<int>& shape, int device_id) {
    reset(shape, device_id);
}

void CudaTensor::reset() {
    buffer_.reset();
    shape_.clear();
    numel_ = 0;
    device_id_ = 0;
}

void CudaTensor::reset(const std::vector<int>& shape, int device_id) {
    size_t numel = checked_numel(shape, "CudaTensor");
    shape_ = shape;
    numel_ = numel;
    device_id_ = device_id;
    buffer_.reset(numel * sizeof(float), device_id);
}

void CudaTensor::upload_from(const Tensor& src) {
    if (src.shape != shape_) {
        throw std::runtime_error(
            "CudaTensor::upload_from shape mismatch: device=" + shape_str() +
            " host=" + src.shape_str()
        );
    }
    buffer_.upload(src.data.data(), src.size() * sizeof(float));
}

Tensor CudaTensor::download() const {
    if (shape_.empty()) {
        return Tensor();
    }
    Tensor dst(shape_, 0.0f);
    download_to(dst);
    return dst;
}

void CudaTensor::download_to(Tensor& dst) const {
    if (dst.shape != shape_) {
        throw std::runtime_error(
            "CudaTensor::download_to shape mismatch: device=" + shape_str() +
            " host=" + dst.shape_str()
        );
    }
    buffer_.download(dst.data.data(), dst.size() * sizeof(float));
}

void* CudaTensor::data() {
    return buffer_.data();
}

const void* CudaTensor::data() const {
    return buffer_.data();
}

std::string CudaTensor::shape_str() const {
    return shape_to_string(shape_);
}

CudaTensor cuda_tensor_from_host(const Tensor& src, int device_id) {
    CudaTensor dst(src.shape, device_id);
    dst.upload_from(src);
    return dst;
}

} // namespace mini_llama
