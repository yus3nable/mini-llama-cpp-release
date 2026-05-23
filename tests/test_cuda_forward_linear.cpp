#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mini_llama;

namespace {

constexpr float kAbsTol = 1e-3f;
constexpr float kRelTol = 1e-3f;

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

void require_cuda_not_built() {
    require(!cuda_runtime_built(), "cuda_runtime_built should be false in CPU build");
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(model.loaded, "tiny model should load");

    bool threw = false;
    try {
        upload_model_weights_to_cuda(model, 0);
    } catch (const std::exception& e) {
        threw = true;
        require(
            std::string(e.what()).find("CUDA backend was not built") != std::string::npos,
            "CPU build should report missing CUDA backend"
        );
    }
    require(threw, "CUDA weight upload should throw in CPU build");
}

void test_cuda_forward_linear() {
    require(cuda_runtime_built(), "cuda_runtime_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");

    MiniLlamaModel cpu_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    MiniLlamaModel cuda_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(cpu_model.loaded, "CPU tiny model should load");
    require(cuda_model.loaded, "CUDA tiny model should load");

    MiniLlamaContext cpu_ctx(&cpu_model);
    MiniLlamaContext cuda_ctx(&cuda_model);
    MiniBatch batch = MiniBatch::single(1, 0);

    Tensor cpu_logits = forward_batch(cpu_model, cpu_ctx, batch);

    upload_model_weights_to_cuda(cuda_model, 0);
    require(model_cuda_uploaded_weight_count(cuda_model) == 21, "tiny model should upload 21 CUDA resident weights");
    require(model_cuda_uploaded_f32_weight_count(cuda_model) == 15, "tiny model should upload 15 F32 linear weights");
    require(model_cuda_memory_bytes(cuda_model) == 132224, "tiny model uploaded byte count should match resident weights");

    Tensor cuda_logits = forward_batch(cuda_model, cuda_ctx, batch);
    require_close_tensor(cuda_logits, cpu_logits, "cuda_forward_linear logits");

    require(model_cuda_linear_calls(cuda_model) == 15, "one tiny token should run 15 CUDA linear calls");
    require(model_cuda_attention_calls(cuda_model) == 2, "one tiny token should run 2 CUDA attention calls");
    require(model_cuda_host_to_device_copies(cuda_model) == 0, "one tiny token should keep host->device copies at zero");
    require(model_cuda_device_to_host_copies(cuda_model) == 1, "one tiny token should download only logits");
    require(model_cuda_host_to_device_bytes(cuda_model) == 0, "one tiny token should keep host->device bytes at zero");
    require(model_cuda_device_to_host_bytes(cuda_model) > 0, "device->host byte count should be recorded");
}

} // namespace

int main() {
    try {
#ifdef MINI_LLAMA_USE_CUDA
        test_cuda_forward_linear();
        std::cout << "PASS cuda_forward_linear\n";
#else
        require_cuda_not_built();
        std::cout << "PASS cuda_forward_linear_cpu_build\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL cuda_forward_linear: " << e.what() << "\n";
        return 1;
    }
}
