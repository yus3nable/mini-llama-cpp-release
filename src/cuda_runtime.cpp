#include "mini_llama/cuda_runtime.h"

#include <sstream>
#include <stdexcept>

#ifdef MINI_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace mini_llama {

namespace {

std::runtime_error cuda_not_built_error() {
    return std::runtime_error(
        "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine."
    );
}

#ifdef MINI_LLAMA_USE_CUDA
void check_cuda(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "CUDA runtime error in " + std::string(expr) + ": " + cudaGetErrorString(err)
        );
    }
}

cudaMemcpyKind to_cuda_memcpy_kind(CudaMemcpyKind kind) {
    switch (kind) {
        case CudaMemcpyKind::HostToDevice:
            return cudaMemcpyHostToDevice;
        case CudaMemcpyKind::DeviceToHost:
            return cudaMemcpyDeviceToHost;
        case CudaMemcpyKind::DeviceToDevice:
            return cudaMemcpyDeviceToDevice;
    }
    throw std::runtime_error("unknown CUDA memcpy kind");
}
#endif

} // namespace

bool cuda_runtime_built() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

int cuda_device_count() {
#ifdef MINI_LLAMA_USE_CUDA
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
#else
    throw cuda_not_built_error();
#endif
}

void cuda_set_device(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
#else
    (void)device_id;
    throw cuda_not_built_error();
#endif
}

CudaDeviceInfo cuda_get_device_info(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    int count = cuda_device_count();
    if (device_id < 0 || device_id >= count) {
        throw std::runtime_error(
            "CUDA device " + std::to_string(device_id) + " is not available; detected " +
            std::to_string(count) + " device(s)"
        );
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    CudaDeviceInfo info;
    info.id = device_id;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_memory_bytes = prop.totalGlobalMem;
    check_cuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
    return info;
#else
    (void)device_id;
    throw cuda_not_built_error();
#endif
}

std::string cuda_format_device_info(const CudaDeviceInfo& info) {
    double total_gb = static_cast<double>(info.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream out;
    out << "device " << info.id << ": " << info.name
        << ", compute capability " << info.compute_major << "." << info.compute_minor
        << ", total memory " << total_gb << " GB"
        << ", cuda runtime " << info.runtime_version
        << ", driver " << info.driver_version;
    return out.str();
}

void* cuda_malloc_bytes(size_t bytes) {
#ifdef MINI_LLAMA_USE_CUDA
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
#else
    (void)bytes;
    throw cuda_not_built_error();
#endif
}

void cuda_free_bytes(void* ptr) {
#ifdef MINI_LLAMA_USE_CUDA
    if (ptr == nullptr) {
        return;
    }
    check_cuda(cudaFree(ptr), "cudaFree");
#else
    (void)ptr;
    throw cuda_not_built_error();
#endif
}

void cuda_memcpy_bytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind) {
#ifdef MINI_LLAMA_USE_CUDA
    if (bytes == 0) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error("cudaMemcpy requires non-null src and dst for non-empty copies");
    }
    check_cuda(cudaMemcpy(dst, src, bytes, to_cuda_memcpy_kind(kind)), "cudaMemcpy");
#else
    (void)dst;
    (void)src;
    (void)bytes;
    (void)kind;
    throw cuda_not_built_error();
#endif
}

CudaDeviceBuffer::CudaDeviceBuffer(size_t bytes, int device_id) {
    reset(bytes, device_id);
}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    release_noexcept();
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_), device_id_(other.device_id_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release_noexcept();
    data_ = other.data_;
    bytes_ = other.bytes_;
    device_id_ = other.device_id_;
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
    return *this;
}

void CudaDeviceBuffer::reset() {
    if (data_ == nullptr) {
        return;
    }
    cuda_set_device(device_id_);
    cuda_free_bytes(data_);
    data_ = nullptr;
    bytes_ = 0;
    device_id_ = 0;
}

void CudaDeviceBuffer::reset(size_t bytes, int device_id) {
    reset();
    if (bytes == 0) {
        device_id_ = device_id;
        return;
    }
    cuda_set_device(device_id);
    data_ = cuda_malloc_bytes(bytes);
    bytes_ = bytes;
    device_id_ = device_id;
}

void CudaDeviceBuffer::upload(const void* src, size_t bytes) {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA upload exceeds device buffer size");
    }
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(data_, src, bytes, CudaMemcpyKind::HostToDevice);
}

void CudaDeviceBuffer::download(void* dst, size_t bytes) const {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA download exceeds device buffer size");
    }
    cuda_set_device(device_id_);
    cuda_memcpy_bytes(dst, data_, bytes, CudaMemcpyKind::DeviceToHost);
}

void CudaDeviceBuffer::release_noexcept() noexcept {
    try {
        reset();
    } catch (...) {
    }
}

} // namespace mini_llama
