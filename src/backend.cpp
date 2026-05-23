#include "mini_llama/backend.h"
#include "mini_llama/cuda_runtime.h"

#include <stdexcept>

namespace mini_llama {

bool parse_backend_kind(const std::string& text, BackendKind& kind) {
    if (text == "cpu") {
        kind = BackendKind::Cpu;
        return true;
    }
    if (text == "cuda") {
        kind = BackendKind::Cuda;
        return true;
    }
    return false;
}

std::string backend_kind_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu:
            return "cpu";
        case BackendKind::Cuda:
            return "cuda";
    }
    return "unknown";
}

bool cuda_backend_built() {
    return cuda_runtime_built();
}

bool cuda_backend_available(int device_id) {
    try {
        int count = cuda_device_count();
        return device_id >= 0 && device_id < count;
    } catch (const std::exception&) {
        return false;
    }
}

std::string cuda_device_summary(int device_id) {
    try {
        return cuda_format_device_info(cuda_get_device_info(device_id));
    } catch (const std::exception& e) {
        return e.what();
    }
}

void validate_backend(const BackendConfig& config) {
    if (config.kind == BackendKind::Cpu) {
        if (config.device_id_set) {
            throw std::runtime_error("--device can only be used with --backend cuda.");
        }
        return;
    }

    if (config.kind == BackendKind::Cuda) {
        if (!cuda_backend_built()) {
            throw std::runtime_error(
                "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine."
            );
        }
        if (!cuda_backend_available(config.device_id)) {
            throw std::runtime_error(cuda_device_summary(config.device_id));
        }
        return;
    }

    throw std::runtime_error("unknown backend");
}

std::string backend_execution_note(const BackendConfig& config) {
    if (config.kind == BackendKind::Cpu) {
        return "compute: cpu";
    }
    return "compute: cuda linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA attention over GPU KV cache + device-resident decode path; CPU sampler";
}

} // namespace mini_llama
