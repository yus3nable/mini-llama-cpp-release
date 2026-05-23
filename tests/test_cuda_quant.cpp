#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
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

Tensor make_pattern_tensor(const std::vector<int>& shape, float scale, float shift) {
    Tensor t(shape, 0.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        int bucket = static_cast<int>((i * 19 + 7) % 31);
        t.data[i] = static_cast<float>(bucket - 15) * scale + shift;
    }
    return t;
}

uint16_t float_to_fp16_for_test(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        uint32_t shifted = mantissa >> (1 - exponent);
        if ((shifted & 0x00001000u) != 0) {
            shifted += 0x00002000u;
        }
        return static_cast<uint16_t>(sign | (shifted >> 13));
    }

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if ((mantissa & 0x00001000u) != 0) {
        mantissa += 0x00002000u;
        if ((mantissa & 0x00800000u) != 0) {
            mantissa = 0;
            ++exponent;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

float fp16_to_float_for_test(uint16_t value) {
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            uint32_t exp32 = exponent + (127 - 15);
            bits = sign | (exp32 << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        uint32_t exp32 = exponent + (127 - 15);
        bits = sign | (exp32 << 23) | (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<BlockQ4_1> quantize_to_q4_1_for_test(const Tensor& src) {
    int row_size = src.ndim() >= 2 ? src.shape.back() : static_cast<int>(src.size());
    int n_rows = src.ndim() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
    int row_blocks = (row_size + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE;

    std::vector<BlockQ4_1> blocks;
    blocks.reserve(static_cast<size_t>(n_rows) * row_blocks);
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * Q4_1_BLOCK_SIZE;
            int k_end = std::min(base + Q4_1_BLOCK_SIZE, row_offset + row_size);
            float min_value = std::numeric_limits<float>::infinity();
            float max_value = -std::numeric_limits<float>::infinity();
            for (int k = base; k < k_end; ++k) {
                min_value = std::min(min_value, src.data[k]);
                max_value = std::max(max_value, src.data[k]);
            }

            BlockQ4_1 block{};
            std::memset(&block, 0, sizeof(block));
            float range = max_value - min_value;
            if (range > 0.0f) {
                block.d = float_to_fp16_for_test(range / 15.0f);
                block.m = float_to_fp16_for_test(min_value);
                float stored_d = fp16_to_float_for_test(block.d);
                float stored_m = fp16_to_float_for_test(block.m);
                float inv_d = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;
                for (int j = 0; j < Q4_1_BLOCK_SIZE / 2; ++j) {
                    int idx0 = base + j;
                    int idx1 = base + j + Q4_1_BLOCK_SIZE / 2;
                    int q0 = 0;
                    int q1 = 0;
                    if (idx0 < k_end) {
                        q0 = static_cast<int>(std::round((src.data[idx0] - stored_m) * inv_d));
                    }
                    if (idx1 < k_end) {
                        q1 = static_cast<int>(std::round((src.data[idx1] - stored_m) * inv_d));
                    }
                    q0 = std::clamp(q0, 0, 15);
                    q1 = std::clamp(q1, 0, 15);
                    block.qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
                }
            } else {
                block.d = 0;
                block.m = float_to_fp16_for_test(min_value == std::numeric_limits<float>::infinity() ? 0.0f : min_value);
            }
            blocks.push_back(block);
        }
    }
    return blocks;
}

void quantize_qt_to_q4_1_for_test(QuantizedTensor& qt) {
    Tensor t = to_tensor(qt);
    qt.q4_1_data = quantize_to_q4_1_for_test(t);
    qt.type = QuantType::Q4_1;
    qt.f32_data.clear();
    qt.f32_data.shrink_to_fit();
    qt.q8_0_data.clear();
    qt.q8_0_data.shrink_to_fit();
    qt.q4_0_data.clear();
    qt.q4_0_data.shrink_to_fit();
}

void quantize_model_to_q4_1_for_test(MiniLlamaModel& model) {
    quantize_qt_to_q4_1_for_test(model.lm_head);
    for (auto& lw : model.layers) {
        quantize_qt_to_q4_1_for_test(lw.wq);
        quantize_qt_to_q4_1_for_test(lw.wk);
        quantize_qt_to_q4_1_for_test(lw.wv);
        quantize_qt_to_q4_1_for_test(lw.wo);
        quantize_qt_to_q4_1_for_test(lw.w_gate);
        quantize_qt_to_q4_1_for_test(lw.w_up);
        quantize_qt_to_q4_1_for_test(lw.w_down);
    }
}

size_t quant_linear_weight_bytes(const MiniLlamaModel& model, QuantType type) {
    auto bytes_for = [type](const QuantizedTensor& weight) -> size_t {
        if (type == QuantType::Q8_0) {
            return weight.q8_0_data.size() * sizeof(BlockQ8_0);
        }
        if (type == QuantType::Q4_0) {
            return weight.q4_0_data.size() * sizeof(BlockQ4_0);
        }
        if (type == QuantType::Q4_1) {
            return weight.q4_1_data.size() * sizeof(BlockQ4_1);
        }
        return 0;
    };
    size_t bytes = bytes_for(model.lm_head);
    for (const auto& lw : model.layers) {
        bytes += bytes_for(lw.wq);
        bytes += bytes_for(lw.wk);
        bytes += bytes_for(lw.wv);
        bytes += bytes_for(lw.wo);
        bytes += bytes_for(lw.w_gate);
        bytes += bytes_for(lw.w_up);
        bytes += bytes_for(lw.w_down);
    }
    return bytes;
}

size_t resident_f32_tensor_bytes(const MiniLlamaModel& model) {
    auto tensor_bytes = [](const Tensor& tensor) -> size_t {
        return tensor.data.size() * sizeof(float);
    };
    size_t bytes = tensor_bytes(model.token_embedding) + tensor_bytes(model.final_norm);
    for (const auto& lw : model.layers) {
        bytes += tensor_bytes(lw.attention_norm);
        bytes += tensor_bytes(lw.bq);
        bytes += tensor_bytes(lw.bk);
        bytes += tensor_bytes(lw.bv);
        bytes += tensor_bytes(lw.ffn_norm);
    }
    return bytes;
}

void require_cuda_quant_not_built() {
    require(!cuda_quant_built(), "cuda_quant_built should be false in CPU build");
    Tensor x({4}, 1.0f);
    Tensor w = make_pattern_tensor({3, 4}, 0.05f, 0.0f);
    std::vector<BlockQ8_0> q8 = quantize_to_q8_0(w);
    std::vector<BlockQ4_0> q4_0 = quantize_to_q4_0(w);
    std::vector<BlockQ4_1> q4_1 = quantize_to_q4_1_for_test(w);

    bool threw = false;
    try {
        (void)cuda_q8_0_linear(x, q8, w.shape);
    } catch (const std::exception& e) {
        threw = true;
        require(
            std::string(e.what()).find("CUDA quant kernels were not built") != std::string::npos,
            "CPU build should report missing CUDA quant kernels"
        );
    }
    require(threw, "cuda_q8_0_linear should throw in CPU build");

    threw = false;
    try {
        (void)cuda_q4_0_linear(x, q4_0, w.shape);
    } catch (const std::exception& e) {
        threw = true;
        require(
            std::string(e.what()).find("CUDA quant kernels were not built") != std::string::npos,
            "CPU build should report missing CUDA quant kernels for Q4_0"
        );
    }
    require(threw, "cuda_q4_0_linear should throw in CPU build");

    threw = false;
    try {
        (void)cuda_q4_1_linear(x, q4_1, w.shape);
    } catch (const std::exception& e) {
        threw = true;
        require(
            std::string(e.what()).find("CUDA quant kernels were not built") != std::string::npos,
            "CPU build should report missing CUDA quant kernels for Q4_1"
        );
    }
    require(threw, "cuda_q4_1_linear should throw in CPU build");
}

void test_cuda_q8_0_linear_cases() {
    require(cuda_quant_built(), "cuda_quant_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");

    Tensor w1 = make_pattern_tensor({5, 13}, 0.037f, -0.11f);
    Tensor x1 = make_pattern_tensor({13}, 0.071f, 0.03f);
    Tensor b1 = make_pattern_tensor({5}, 0.013f, -0.02f);
    std::vector<BlockQ8_0> q8_1 = quantize_to_q8_0(w1);
    Tensor expected1 = linear_q8_0(x1, q8_1, w1.shape);
    Tensor actual1 = cuda_q8_0_linear(x1, q8_1, w1.shape);
    require_close_tensor(actual1, expected1, "cuda_q8_0_linear 1D no bias");

    Tensor expected1_bias = expected1;
    for (int i = 0; i < b1.shape[0]; ++i) {
        expected1_bias.data[i] += b1.data[i];
    }
    Tensor actual1_bias = cuda_q8_0_linear(x1, q8_1, w1.shape, &b1);
    require_close_tensor(actual1_bias, expected1_bias, "cuda_q8_0_linear 1D bias");

    Tensor w2 = make_pattern_tensor({32, 64}, 0.019f, 0.04f);
    Tensor x2 = make_pattern_tensor({3, 64}, 0.029f, -0.06f);
    std::vector<BlockQ8_0> q8_2 = quantize_to_q8_0(w2);
    Tensor expected2 = linear_q8_0(x2, q8_2, w2.shape);
    Tensor actual2 = cuda_q8_0_linear(x2, q8_2, w2.shape);
    require_close_tensor(actual2, expected2, "cuda_q8_0_linear 2D batch");

    CudaDeviceBuffer w_dev(q8_2.size() * sizeof(BlockQ8_0), 0);
    w_dev.upload(q8_2.data(), q8_2.size() * sizeof(BlockQ8_0));
    Tensor actual_device = cuda_q8_0_linear_device_weight(
        x2,
        w_dev.data(),
        q8_2.size(),
        w2.shape
    );
    require_close_tensor(actual_device, expected2, "cuda_q8_0_linear device weight");
}

void test_cuda_q4_0_linear_cases() {
    require(cuda_quant_built(), "cuda_quant_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");

    Tensor w1 = make_pattern_tensor({5, 13}, 0.037f, -0.11f);
    Tensor x1 = make_pattern_tensor({13}, 0.071f, 0.03f);
    Tensor b1 = make_pattern_tensor({5}, 0.013f, -0.02f);
    std::vector<BlockQ4_0> q4_1 = quantize_to_q4_0(w1);
    Tensor expected1 = linear_q4_0(x1, q4_1, w1.shape);
    Tensor actual1 = cuda_q4_0_linear(x1, q4_1, w1.shape);
    require_close_tensor(actual1, expected1, "cuda_q4_0_linear 1D no bias");

    Tensor expected1_bias = expected1;
    for (int i = 0; i < b1.shape[0]; ++i) {
        expected1_bias.data[i] += b1.data[i];
    }
    Tensor actual1_bias = cuda_q4_0_linear(x1, q4_1, w1.shape, &b1);
    require_close_tensor(actual1_bias, expected1_bias, "cuda_q4_0_linear 1D bias");

    Tensor w2 = make_pattern_tensor({32, 64}, 0.019f, 0.04f);
    Tensor x2 = make_pattern_tensor({3, 64}, 0.029f, -0.06f);
    std::vector<BlockQ4_0> q4_2 = quantize_to_q4_0(w2);
    Tensor expected2 = linear_q4_0(x2, q4_2, w2.shape);
    Tensor actual2 = cuda_q4_0_linear(x2, q4_2, w2.shape);
    require_close_tensor(actual2, expected2, "cuda_q4_0_linear 2D batch");

    CudaDeviceBuffer w_dev(q4_2.size() * sizeof(BlockQ4_0), 0);
    w_dev.upload(q4_2.data(), q4_2.size() * sizeof(BlockQ4_0));
    Tensor actual_device = cuda_q4_0_linear_device_weight(
        x2,
        w_dev.data(),
        q4_2.size(),
        w2.shape
    );
    require_close_tensor(actual_device, expected2, "cuda_q4_0_linear device weight");

    CudaTensor x_dev = cuda_tensor_from_host(x2, 0);
    CudaTensor y_dev = cuda_q4_0_linear_device_input(
        x_dev,
        w_dev.data(),
        q4_2.size(),
        w2.shape
    );
    require_close_tensor(y_dev.download(), expected2, "cuda_q4_0_linear device input");
}

void test_cuda_q4_1_linear_cases() {
    require(cuda_quant_built(), "cuda_quant_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");

    Tensor w1 = make_pattern_tensor({5, 13}, 0.037f, -0.11f);
    Tensor x1 = make_pattern_tensor({13}, 0.071f, 0.03f);
    Tensor b1 = make_pattern_tensor({5}, 0.013f, -0.02f);
    std::vector<BlockQ4_1> q4_1 = quantize_to_q4_1_for_test(w1);
    Tensor expected1 = linear_q4_1(x1, q4_1, w1.shape);
    Tensor actual1 = cuda_q4_1_linear(x1, q4_1, w1.shape);
    require_close_tensor(actual1, expected1, "cuda_q4_1_linear 1D no bias");

    Tensor expected1_bias = expected1;
    for (int i = 0; i < b1.shape[0]; ++i) {
        expected1_bias.data[i] += b1.data[i];
    }
    Tensor actual1_bias = cuda_q4_1_linear(x1, q4_1, w1.shape, &b1);
    require_close_tensor(actual1_bias, expected1_bias, "cuda_q4_1_linear 1D bias");

    Tensor w2 = make_pattern_tensor({17, 65}, 0.019f, 0.04f);
    Tensor x2 = make_pattern_tensor({3, 65}, 0.029f, -0.06f);
    std::vector<BlockQ4_1> q4_2 = quantize_to_q4_1_for_test(w2);
    Tensor expected2 = linear_q4_1(x2, q4_2, w2.shape);
    Tensor actual2 = cuda_q4_1_linear(x2, q4_2, w2.shape);
    require_close_tensor(actual2, expected2, "cuda_q4_1_linear 2D batch");

    CudaDeviceBuffer w_dev(q4_2.size() * sizeof(BlockQ4_1), 0);
    w_dev.upload(q4_2.data(), q4_2.size() * sizeof(BlockQ4_1));
    Tensor actual_device = cuda_q4_1_linear_device_weight(
        x2,
        w_dev.data(),
        q4_2.size(),
        w2.shape
    );
    require_close_tensor(actual_device, expected2, "cuda_q4_1_linear device weight");

    CudaTensor x_dev = cuda_tensor_from_host(x2, 0);
    CudaTensor y_dev = cuda_q4_1_linear_device_input(
        x_dev,
        w_dev.data(),
        q4_2.size(),
        w2.shape
    );
    require_close_tensor(y_dev.download(), expected2, "cuda_q4_1_linear device input");
}

void test_cuda_forward_quant(QuantType quant_type) {
    require(cuda_quant_built(), "cuda_quant_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");

    MiniLlamaModel cpu_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    MiniLlamaModel cuda_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(cpu_model.loaded, "CPU tiny model should load");
    require(cuda_model.loaded, "CUDA tiny model should load");
    if (quant_type == QuantType::Q8_0) {
        quantize_model_to_q8_0(cpu_model);
        quantize_model_to_q8_0(cuda_model);
    } else if (quant_type == QuantType::Q4_0) {
        quantize_model_to_q4_0(cpu_model);
        quantize_model_to_q4_0(cuda_model);
    } else if (quant_type == QuantType::Q4_1) {
        quantize_model_to_q4_1_for_test(cpu_model);
        quantize_model_to_q4_1_for_test(cuda_model);
    } else {
        throw std::runtime_error("test_cuda_forward_quant: unsupported quant type");
    }

    MiniLlamaContext cpu_ctx(&cpu_model);
    MiniLlamaContext cuda_ctx(&cuda_model);
    MiniBatch batch = MiniBatch::single(1, 0);
    Tensor cpu_logits = forward_batch(cpu_model, cpu_ctx, batch);

    upload_model_weights_to_cuda(cuda_model, 0);
    require(model_cuda_uploaded_weight_count(cuda_model) == 21, "tiny quant model should upload 21 CUDA resident weights");
    require(model_cuda_uploaded_f32_weight_count(cuda_model) == 0, "tiny quant model should upload no F32 linear weights");
    size_t uploaded_quant = 0;
    if (quant_type == QuantType::Q8_0) {
        uploaded_quant = model_cuda_uploaded_q8_0_weight_count(cuda_model);
    } else if (quant_type == QuantType::Q4_0) {
        uploaded_quant = model_cuda_uploaded_q4_0_weight_count(cuda_model);
    } else {
        uploaded_quant = model_cuda_uploaded_q4_1_weight_count(cuda_model);
    }
    require(uploaded_quant == 15, "tiny quant model should upload 15 quantized linear weights");
    require(
        model_cuda_memory_bytes(cuda_model) == quant_linear_weight_bytes(cuda_model, quant_type) + resident_f32_tensor_bytes(cuda_model),
        "quant uploaded byte count should match resident weights"
    );

    Tensor cuda_logits = forward_batch(cuda_model, cuda_ctx, batch);
    require_close_tensor(cuda_logits, cpu_logits, "cuda_forward_quant logits");
    require(model_cuda_linear_calls(cuda_model) == 15, "one tiny token should run 15 CUDA quant linear calls");
    require(model_cuda_attention_calls(cuda_model) == 2, "one tiny token should run 2 CUDA attention calls");
    require(model_cuda_host_to_device_copies(cuda_model) == 0, "one tiny token should keep host->device copies at zero");
    require(model_cuda_device_to_host_copies(cuda_model) == 1, "one tiny token should download only logits");
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = argc >= 2 ? argv[1] : "all";
    try {
#ifdef MINI_LLAMA_USE_CUDA
        if (mode == "q8_0" || mode == "all") {
            test_cuda_q8_0_linear_cases();
            std::cout << "PASS cuda_quant_q8_0\n";
        }
        if (mode == "q4" || mode == "all") {
            test_cuda_q4_0_linear_cases();
            test_cuda_q4_1_linear_cases();
            std::cout << "PASS cuda_quant_q4\n";
        }
        if (mode == "forward" || mode == "all") {
            test_cuda_forward_quant(QuantType::Q8_0);
            std::cout << "PASS cuda_forward_quant\n";
        }
        if (mode == "forward_q4" || mode == "all") {
            test_cuda_forward_quant(QuantType::Q4_0);
            test_cuda_forward_quant(QuantType::Q4_1);
            std::cout << "PASS cuda_forward_q4\n";
        }
#else
        require_cuda_quant_not_built();
        std::cout << "PASS cuda_quant_cpu_build\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << mode << ": " << e.what() << "\n";
        return 1;
    }
}
