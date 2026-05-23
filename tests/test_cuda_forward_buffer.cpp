#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace mini_llama;

namespace {

constexpr float kAbsTol = 2e-3f;
constexpr float kRelTol = 2e-3f;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close_enough(float actual, float expected) {
    float abs_err = std::abs(actual - expected);
    float scale = std::max(1.0f, std::abs(expected));
    return abs_err <= kAbsTol || abs_err / scale <= kRelTol;
}

void require_close_tensor(const Tensor& actual, const Tensor& expected, const std::string& label) {
    require(actual.shape == expected.shape, label + ": shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual.data[i], expected.data[i])) {
            throw std::runtime_error(
                label + ": value mismatch at " + std::to_string(i) +
                ", actual=" + std::to_string(actual.data[i]) +
                ", expected=" + std::to_string(expected.data[i])
            );
        }
    }
}

void require_cpu_build_path() {
    require(!cuda_runtime_built(), "cuda_runtime_built should be false in CPU build");
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(model.loaded, "tiny model should load");
    require(model_cuda_host_to_device_copies(model) == 0, "CPU model should report zero host->device copies");
}

void run_tiny_forward_buffer_case(bool quantize_q8_0) {
    require(cuda_runtime_built(), "cuda_runtime_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");

    MiniLlamaModel cpu_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    MiniLlamaModel cuda_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(cpu_model.loaded, "CPU tiny model should load");
    require(cuda_model.loaded, "CUDA tiny model should load");

    if (quantize_q8_0) {
        quantize_model_to_q8_0(cpu_model);
        quantize_model_to_q8_0(cuda_model);
    }

    MiniLlamaContext cpu_ctx(&cpu_model);
    MiniLlamaContext cuda_ctx(&cuda_model);
    MiniBatch batch = MiniBatch::single(1, 0);
    Tensor cpu_logits = forward_batch(cpu_model, cpu_ctx, batch);

    upload_model_weights_to_cuda(cuda_model, 0);
    reset_model_cuda_runtime_stats(cuda_model);
    Tensor cuda_logits = forward_batch(cuda_model, cuda_ctx, batch);

    require_close_tensor(cuda_logits, cpu_logits, quantize_q8_0 ? "cuda_forward_buffer q8 logits" : "cuda_forward_buffer f32 logits");
    require(model_cuda_linear_calls(cuda_model) == 15, "one tiny token should still run 15 CUDA linear calls");
    require(model_cuda_activation_calls(cuda_model) == 15, "one tiny token should run 15 CUDA activation calls");
    require(model_cuda_attention_calls(cuda_model) == 2, "one tiny token should run 2 CUDA attention calls");
    require(model_cuda_host_to_device_copies(cuda_model) == 0, "device-resident path should keep host->device copies at zero");
    require(model_cuda_device_to_host_copies(cuda_model) == 1, "device-resident path should download only logits");
    require(model_cuda_host_to_device_bytes(cuda_model) == 0, "tiny device-resident host->device bytes should stay zero");
    require(model_cuda_device_to_host_bytes(cuda_model) == 512, "tiny device-resident device->host bytes should match logits");
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = argc >= 2 ? argv[1] : "all";
    try {
#ifdef MINI_LLAMA_USE_CUDA
        if (mode == "f32" || mode == "all") {
            run_tiny_forward_buffer_case(false);
            std::cout << "PASS cuda_forward_buffer\n";
        }
        if (mode == "q8_0" || mode == "all") {
            run_tiny_forward_buffer_case(true);
            std::cout << "PASS cuda_forward_device_resident\n";
        }
#else
        require_cpu_build_path();
        std::cout << "PASS cuda_forward_buffer_cpu_build\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << mode << ": " << e.what() << "\n";
        return 1;
    }
}
