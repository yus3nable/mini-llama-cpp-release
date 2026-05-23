#include "mini_llama/cuda_matmul.h"

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
        int bucket = static_cast<int>((i * 17 + 11) % 23);
        t.data[i] = static_cast<float>(bucket - 9) * scale + shift;
    }
    return t;
}

Tensor cpu_matmul_reference(const Tensor& A, const Tensor& B) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    Tensor C({M, N}, 0.0f);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.data[row * K + k] * B.data[k * N + col];
            }
            C.data[row * N + col] = sum;
        }
    }
    return C;
}

Tensor cpu_linear_reference(const Tensor& x, const Tensor& W, const Tensor* bias) {
    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = W.shape[0];
    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < out_features; ++out) {
            float sum = bias == nullptr ? 0.0f : bias->data[out];
            for (int k = 0; k < in_features; ++k) {
                sum += x.data[row * in_features + k] * W.data[out * in_features + k];
            }
            y.data[row * out_features + out] = sum;
        }
    }
    return y;
}

void require_cuda_not_built() {
    require(!cuda_matmul_built(), "cuda_matmul_built should be false in CPU build");
    Tensor A({1, 4}, 1.0f);
    Tensor B({4, 3}, 1.0f);
    bool threw = false;
    try {
        (void)cuda_matmul(A, B);
    } catch (const std::exception& e) {
        threw = true;
        require(
            std::string(e.what()).find("CUDA matmul was not built") != std::string::npos,
            "CPU build should report missing CUDA matmul"
        );
    }
    require(threw, "cuda_matmul should throw in CPU build");
}

void test_cuda_matmul_cases() {
    require(cuda_matmul_built(), "cuda_matmul_built should be true in CUDA build");
    const std::vector<std::pair<int, int>> shapes = {
        {1, 4},
        {2, 4},
        {8, 16},
    };
    const std::vector<int> out_cols = {3, 3, 32};
    for (size_t i = 0; i < shapes.size(); ++i) {
        int rows = shapes[i].first;
        int inner = shapes[i].second;
        int cols = out_cols[i];
        Tensor A = make_pattern_tensor({rows, inner}, 0.071f, -0.13f);
        Tensor B = make_pattern_tensor({inner, cols}, 0.037f, 0.08f);
        Tensor expected = cpu_matmul_reference(A, B);
        Tensor actual = cuda_matmul(A, B);
        require_close_tensor(actual, expected, "cuda_matmul " + A.shape_str() + " x " + B.shape_str());
    }
}

void test_cuda_linear_cases() {
    require(cuda_matmul_built(), "cuda_matmul_built should be true in CUDA build");

    Tensor x1 = make_pattern_tensor({4}, 0.11f, -0.2f);
    Tensor w1 = make_pattern_tensor({3, 4}, 0.043f, 0.03f);
    Tensor b1 = make_pattern_tensor({3}, 0.019f, -0.01f);
    require_close_tensor(cuda_linear(x1, w1), cpu_linear_reference(x1, w1, nullptr), "cuda_linear 1D no bias");
    require_close_tensor(cuda_linear(x1, w1, &b1), cpu_linear_reference(x1, w1, &b1), "cuda_linear 1D bias");

    Tensor x2 = make_pattern_tensor({2, 4}, 0.083f, 0.17f);
    Tensor w2 = make_pattern_tensor({3, 4}, 0.029f, -0.04f);
    Tensor b2 = make_pattern_tensor({3}, 0.017f, 0.05f);
    require_close_tensor(cuda_linear(x2, w2), cpu_linear_reference(x2, w2, nullptr), "cuda_linear 2D no bias");
    require_close_tensor(cuda_linear(x2, w2, &b2), cpu_linear_reference(x2, w2, &b2), "cuda_linear 2D bias");

    Tensor x3 = make_pattern_tensor({8, 16}, 0.031f, -0.09f);
    Tensor w3 = make_pattern_tensor({32, 16}, 0.023f, 0.07f);
    Tensor b3 = make_pattern_tensor({32}, 0.013f, -0.03f);
    require_close_tensor(cuda_linear(x3, w3), cpu_linear_reference(x3, w3, nullptr), "cuda_linear 8x16 no bias");
    require_close_tensor(cuda_linear(x3, w3, &b3), cpu_linear_reference(x3, w3, &b3), "cuda_linear 8x16 bias");
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = argc >= 2 ? argv[1] : "all";
    try {
#ifdef MINI_LLAMA_USE_CUDA
        if (mode == "matmul" || mode == "all") {
            test_cuda_matmul_cases();
            std::cout << "PASS cuda_matmul\n";
        }
        if (mode == "linear" || mode == "all") {
            test_cuda_linear_cases();
            std::cout << "PASS cuda_linear\n";
        }
#else
        require_cuda_not_built();
        std::cout << "PASS cuda_matmul_cpu_build\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << mode << ": " << e.what() << "\n";
        return 1;
    }
}
