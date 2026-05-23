#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_runtime.h"
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

void require_cuda_not_built() {
    require(!cuda_runtime_built(), "cuda_runtime_built should be false in CPU build");
    require(!cuda_ops_built(), "cuda_ops_built should be false in CPU build");
    MiniLlamaModel model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(model.loaded, "tiny model should load");
    require(model_cuda_activation_calls(model) == 0, "CPU model should report zero CUDA activation calls");
}

void test_cuda_forward_full_tiny_path() {
    require(cuda_runtime_built(), "cuda_runtime_built should be true in CUDA build");
    require(cuda_ops_built(), "cuda_ops_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");

    MiniLlamaModel cpu_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    MiniLlamaModel cuda_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(cpu_model.loaded, "CPU tiny model should load");
    require(cuda_model.loaded, "CUDA tiny model should load");

    MiniLlamaContext cpu_ctx(&cpu_model);
    MiniLlamaContext cuda_ctx(&cuda_model);
    MiniBatch batch = MiniBatch::from_tokens({1, 2, 3}, 0);

    Tensor cpu_logits = forward_batch(cpu_model, cpu_ctx, batch);

    upload_model_weights_to_cuda(cuda_model, 0);
    reset_model_cuda_runtime_stats(cuda_model);
    Tensor cuda_logits = forward_batch(cuda_model, cuda_ctx, batch);

    require_close_tensor(cuda_logits, cpu_logits, "cuda_forward logits");
    require(model_cuda_linear_calls(cuda_model) == 45, "three tiny tokens should run 45 CUDA linear calls");
    require(model_cuda_activation_calls(cuda_model) == 45, "three tiny tokens should run 45 CUDA activation calls");
    require(model_cuda_attention_calls(cuda_model) == 6, "three tiny tokens should run 6 CUDA attention calls");
    require(!cuda_ctx.cuda_kv_cache.empty(), "CUDA forward should allocate GPU KV cache");
}

} // namespace

int main() {
    try {
#ifdef MINI_LLAMA_USE_CUDA
        test_cuda_forward_full_tiny_path();
        std::cout << "PASS cuda_forward\n";
#else
        require_cuda_not_built();
        std::cout << "PASS cuda_forward_cpu_build\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL cuda_forward: " << e.what() << "\n";
        return 1;
    }
}
