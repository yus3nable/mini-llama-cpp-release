#pragma once

#include <cstddef>
#include <string>

namespace mini_llama {

struct CudaDeviceInfo {
    int id = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    size_t total_memory_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

enum class CudaMemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
};

bool cuda_runtime_built();
int cuda_device_count();
void cuda_set_device(int device_id);
CudaDeviceInfo cuda_get_device_info(int device_id);
std::string cuda_format_device_info(const CudaDeviceInfo& info);

void* cuda_malloc_bytes(size_t bytes);
void cuda_free_bytes(void* ptr);
void cuda_memcpy_bytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind);

class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
    ~CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    void reset();
    void reset(size_t bytes, int device_id = 0);
    void upload(const void* src, size_t bytes);
    void download(void* dst, size_t bytes) const;

    void* data() {
        return data_;
    }

    const void* data() const {
        return data_;
    }

    size_t bytes() const {
        return bytes_;
    }

    int device_id() const {
        return device_id_;
    }

    bool empty() const {
        return data_ == nullptr;
    }

private:
    void release_noexcept() noexcept;

    void* data_ = nullptr;
    size_t bytes_ = 0;
    int device_id_ = 0;
};

} // namespace mini_llama
