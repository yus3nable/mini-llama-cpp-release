#pragma once

#include "mini_llama/tensor.h"
#include <cstdint>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Quantization types
// ---------------------------------------------------------------------------

enum class QuantType {
    F32,
    Q8_0,
    Q4_0,
    Q4_1,
};

// ---------------------------------------------------------------------------
// Q8_0 block layout (ggml / llama.cpp compatible)
// Each block covers 32 elements.
// ---------------------------------------------------------------------------
static constexpr int Q8_0_BLOCK_SIZE = 32;

struct BlockQ8_0 {
    uint16_t d;        // FP16 delta (scale)
    int8_t qs[32];     // quantized values
};

static_assert(sizeof(BlockQ8_0) == sizeof(uint16_t) + Q8_0_BLOCK_SIZE, "wrong Q8_0 block size");

// ---------------------------------------------------------------------------
// Q4_0 block layout (ggml / llama.cpp compatible)
// Each block covers 32 elements, packed into 16 bytes.
// ---------------------------------------------------------------------------
static constexpr int Q4_0_BLOCK_SIZE = 32;

struct BlockQ4_0 {
    uint16_t d;        // FP16 delta (scale)
    uint8_t qs[16];    // 32 x 4-bit values, packed (2 per byte)
};

static_assert(sizeof(BlockQ4_0) == sizeof(uint16_t) + Q4_0_BLOCK_SIZE / 2, "wrong Q4_0 block size");

// ---------------------------------------------------------------------------
// Q4_1 block layout (ggml / llama.cpp compatible)
// Each block covers 32 elements, packed into 16 bytes.
// Value formula: x = d * q + m, q in [0, 15].
// ---------------------------------------------------------------------------
static constexpr int Q4_1_BLOCK_SIZE = 32;

struct BlockQ4_1 {
    uint16_t d;        // FP16 delta (scale)
    uint16_t m;        // FP16 minimum/offset
    uint8_t qs[16];    // 32 x 4-bit values, packed (2 per byte)
};

static_assert(sizeof(BlockQ4_1) == sizeof(uint16_t) * 2 + Q4_1_BLOCK_SIZE / 2, "wrong Q4_1 block size");

// ---------------------------------------------------------------------------
// QuantizedTensor: tagged struct holding quantized or F32 data.
//
// For teaching clarity, uses a tagged union-like layout rather than
// std::variant.  Only one of {f32_data, q8_0_data, q4_0_data, q4_1_data} is valid
// depending on `type`.
// ---------------------------------------------------------------------------
struct QuantizedTensor {
    QuantType type = QuantType::F32;
    std::vector<int> shape;

    // Valid when type == F32
    std::vector<float> f32_data;

    // Valid when type == Q8_0
    std::vector<BlockQ8_0> q8_0_data;

    // Valid when type == Q4_0
    std::vector<BlockQ4_0> q4_0_data;

    // Valid when type == Q4_1
    std::vector<BlockQ4_1> q4_1_data;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    std::string shape_str() const {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size()) s += ", ";
        }
        s += "]";
        return s;
    }

    void assert_shape(const std::vector<int>& expected, const char* caller) const {
        if (shape != expected) {
            throw std::runtime_error(
                std::string(caller) + ": shape mismatch, expected " +
                QuantizedTensor{QuantType::F32, expected}.shape_str() +
                ", got " + shape_str()
            );
        }
    }

    int ndim() const { return static_cast<int>(shape.size()); }
};

// Convert a QuantizedTensor (type==F32) to a plain Tensor.
Tensor to_tensor(const QuantizedTensor& q);

// Convert a plain Tensor to a QuantizedTensor (type==F32).
QuantizedTensor to_quantized_tensor(const Tensor& t);

} // namespace mini_llama
