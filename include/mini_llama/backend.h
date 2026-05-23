#pragma once

#include <string>

namespace mini_llama {

enum class BackendKind {
    Cpu,
    Cuda,
};

struct BackendConfig {
    BackendKind kind = BackendKind::Cpu;
    int device_id = 0;
    bool device_id_set = false;
};

bool parse_backend_kind(const std::string& text, BackendKind& kind);
std::string backend_kind_name(BackendKind kind);

bool cuda_backend_built();
bool cuda_backend_available(int device_id = 0);
std::string cuda_device_summary(int device_id = 0);

// Throws when the requested backend cannot run in this build or on this host.
void validate_backend(const BackendConfig& config);

// Explains which execution path is active.
std::string backend_execution_note(const BackendConfig& config);

} // namespace mini_llama
