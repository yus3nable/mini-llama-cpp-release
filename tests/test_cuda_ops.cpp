#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/ops.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mini_llama;

namespace {

constexpr float kAbsTol = 1e-4f;
constexpr float kRelTol = 1e-4f;

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

Tensor make_pattern_tensor(const std::vector<int>& shape, float scale, float shift) {
    Tensor t(shape, 0.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        int bucket = static_cast<int>((i * 17 + 5) % 29);
        t.data[i] = static_cast<float>(bucket - 14) * scale + shift;
    }
    return t;
}

Tensor cpu_add_reference(const Tensor& a, const Tensor& b) {
    require(a.shape == b.shape, "cpu_add_reference shape mismatch");
    Tensor y(a.shape, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        y.data[i] = a.data[i] + b.data[i];
    }
    return y;
}

void require_cuda_not_built() {
    require(!cuda_ops_built(), "cuda_ops_built should be false in CPU build");
    Tensor x({4}, 1.0f);
    bool threw = false;
    try {
        (void)cuda_silu(x);
    } catch (const std::exception& e) {
        threw = true;
        require(
            std::string(e.what()).find("CUDA ops were not built") != std::string::npos,
            "CPU build should report missing CUDA ops"
        );
    }
    require(threw, "cuda_silu should throw in CPU build");
}

void test_cuda_rmsnorm() {
    Tensor x = make_pattern_tensor({32}, 0.071f, -0.13f);
    Tensor weight = make_pattern_tensor({32}, 0.019f, 1.0f);
    require_close_tensor(cuda_rmsnorm(x, weight, 1e-5f), rmsnorm(x, weight, 1e-5f), "cuda_rmsnorm");
}

void test_cuda_embedding_lookup_device_weight() {
    Tensor embedding = make_pattern_tensor({5, 7}, 0.031f, -0.2f);
    CudaDeviceBuffer embedding_dev(embedding.size() * sizeof(float), 0);
    embedding_dev.upload(embedding.data.data(), embedding.size() * sizeof(float));

    Tensor expected({7}, 0.0f);
    int token_id = 3;
    for (int i = 0; i < 7; ++i) {
        expected.data[i] = embedding.data[static_cast<size_t>(token_id) * 7 + static_cast<size_t>(i)];
    }

    CudaTensor actual_dev = cuda_embedding_lookup_device_weight(
        embedding_dev.data(),
        embedding.shape,
        token_id
    );
    require_close_tensor(actual_dev.download(), expected, "cuda_embedding_lookup_device_weight");
}

void test_cuda_rmsnorm_device_weight() {
    Tensor x = make_pattern_tensor({32}, 0.071f, -0.13f);
    Tensor weight = make_pattern_tensor({32}, 0.019f, 1.0f);
    CudaTensor x_dev = cuda_tensor_from_host(x);
    CudaDeviceBuffer weight_dev(weight.size() * sizeof(float), 0);
    weight_dev.upload(weight.data.data(), weight.size() * sizeof(float));

    CudaTensor actual_dev = cuda_rmsnorm_device_weight(
        x_dev,
        weight_dev.data(),
        weight.shape,
        1e-5f
    );
    require_close_tensor(actual_dev.download(), rmsnorm(x, weight, 1e-5f), "cuda_rmsnorm_device_weight");
}

void test_cuda_silu() {
    Tensor x = make_pattern_tensor({65}, 0.083f, -0.2f);
    require_close_tensor(cuda_silu(x), silu(x), "cuda_silu");
}

void test_cuda_elementwise() {
    Tensor a = make_pattern_tensor({3, 7}, 0.041f, -0.2f);
    Tensor b = make_pattern_tensor({3, 7}, 0.023f, 0.13f);
    require_close_tensor(cuda_elementwise_mul(a, b), elementwise_mul(a, b), "cuda_elementwise_mul");
    require_close_tensor(cuda_elementwise_add(a, b), cpu_add_reference(a, b), "cuda_elementwise_add");

    CudaTensor a_dev = cuda_tensor_from_host(a);
    CudaDeviceBuffer b_dev(b.size() * sizeof(float), 0);
    b_dev.upload(b.data.data(), b.size() * sizeof(float));
    CudaTensor actual_dev = cuda_elementwise_add_device_weight(a_dev, b_dev.data(), b.shape);
    require_close_tensor(actual_dev.download(), cpu_add_reference(a, b), "cuda_elementwise_add_device_weight");
}

void test_cuda_softmax() {
    Tensor x = make_pattern_tensor({37}, 0.11f, -0.3f);
    require_close_tensor(cuda_softmax(x), softmax(x), "cuda_softmax");
}

void test_cuda_rope_normal() {
    Tensor q = make_pattern_tensor({4, 8}, 0.037f, -0.1f);
    Tensor k = make_pattern_tensor({2, 8}, 0.029f, 0.07f);
    Tensor expected_q = q;
    Tensor expected_k = k;

    rope(expected_q, expected_k, 5, 10000.0f, RopeType::Normal);
    cuda_rope(q, k, 5, 10000.0f, RopeType::Normal);

    require_close_tensor(q, expected_q, "cuda_rope normal q");
    require_close_tensor(k, expected_k, "cuda_rope normal k");
}

void test_cuda_rope_neox() {
    Tensor q = make_pattern_tensor({3, 8}, 0.031f, -0.04f);
    Tensor k = make_pattern_tensor({1, 8}, 0.027f, 0.09f);
    Tensor expected_q = q;
    Tensor expected_k = k;

    rope(expected_q, expected_k, 7, 10000.0f, RopeType::NeoX);
    cuda_rope(q, k, 7, 10000.0f, RopeType::NeoX);

    require_close_tensor(q, expected_q, "cuda_rope neox q");
    require_close_tensor(k, expected_k, "cuda_rope neox k");
}

void test_cuda_ops_cases() {
    require(cuda_ops_built(), "cuda_ops_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");
    test_cuda_rmsnorm();
    test_cuda_embedding_lookup_device_weight();
    test_cuda_rmsnorm_device_weight();
    test_cuda_silu();
    test_cuda_elementwise();
    test_cuda_softmax();
    test_cuda_rope_normal();
    test_cuda_rope_neox();
}

} // namespace

int main() {
    try {
#ifdef MINI_LLAMA_USE_CUDA
        test_cuda_ops_cases();
        std::cout << "PASS cuda_ops\n";
#else
        require_cuda_not_built();
        std::cout << "PASS cuda_ops_cpu_build\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL cuda_ops: " << e.what() << "\n";
        return 1;
    }
}
