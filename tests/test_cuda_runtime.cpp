#include "mini_llama/cuda_runtime.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mini_llama;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool message_contains(const std::exception& e, const std::string& text) {
    return std::string(e.what()).find(text) != std::string::npos;
}

void test_cpu_build_reports_missing_cuda() {
    require(!cuda_runtime_built(), "cuda_runtime_built should be false in CPU build");

    bool threw = false;
    try {
        (void)cuda_device_count();
    } catch (const std::exception& e) {
        threw = true;
        require(message_contains(e, "CUDA backend was not built"), "CPU build error message should name CUDA build flag");
    }
    require(threw, "cuda_device_count should throw in CPU build");
}

void test_cuda_build_device_info_and_buffer() {
    require(cuda_runtime_built(), "cuda_runtime_built should be true in CUDA build");

    int count = cuda_device_count();
    require(count >= 1, "CUDA build should see at least one device");

    CudaDeviceInfo info = cuda_get_device_info(0);
    require(info.id == 0, "device id should round trip");
    require(!info.name.empty(), "device name should be non-empty");
    require(info.compute_major > 0, "compute major should be positive");
    require(info.total_memory_bytes > 0, "total memory should be positive");
    require(info.driver_version > 0, "driver version should be positive");
    require(info.runtime_version > 0, "runtime version should be positive");

    std::string formatted = cuda_format_device_info(info);
    require(formatted.find("compute capability") != std::string::npos, "formatted info should include compute capability");
    require(formatted.find("total memory") != std::string::npos, "formatted info should include total memory");

    std::vector<float> host = {1.0f, 2.0f, 3.5f, -4.0f};
    std::vector<float> out(host.size(), 0.0f);
    CudaDeviceBuffer buffer(host.size() * sizeof(float), 0);
    require(!buffer.empty(), "device buffer should own memory");
    require(buffer.bytes() == host.size() * sizeof(float), "device buffer should store byte size");
    require(buffer.device_id() == 0, "device buffer should store device id");
    buffer.upload(host.data(), host.size() * sizeof(float));
    buffer.download(out.data(), out.size() * sizeof(float));
    require(out == host, "device buffer upload/download should round trip");

    CudaDeviceBuffer moved = std::move(buffer);
    require(buffer.empty(), "moved-from buffer should be empty");
    require(!moved.empty(), "moved-to buffer should own memory");
    moved.reset();
    require(moved.empty(), "reset should release memory");
}

} // namespace

int main() {
    try {
#ifdef MINI_LLAMA_USE_CUDA
        test_cuda_build_device_info_and_buffer();
#else
        test_cpu_build_reports_missing_cuda();
#endif
        std::cout << "PASS cuda_runtime\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL cuda_runtime: " << e.what() << "\n";
        return 1;
    }
}
