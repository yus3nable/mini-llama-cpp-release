#include "mini_llama/quant.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

namespace mini_llama {

namespace {

uint16_t float_to_fp16(float value) {
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

float fp16_to_float(uint16_t value) {
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

size_t checked_numel(const std::vector<int>& shape, const char* caller) {
    size_t total = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        int dim = shape[axis];
        if (dim <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": dimension at axis " + std::to_string(axis) +
                " must be positive, got " + std::to_string(dim)
            );
        }
        size_t dim_size = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error(std::string(caller) + ": shape element count overflow");
        }
        total *= dim_size;
    }
    return total;
}

} // namespace

// ---------------------------------------------------------------------------
// Q8_0 quantize / dequantize
// ---------------------------------------------------------------------------
std::vector<BlockQ8_0> quantize_to_q8_0(const Tensor& src) {
    if (src.size() == 0) {
        return {};
    }

    int row_size = src.ndim() >= 2 ? src.shape.back() : static_cast<int>(src.size());
    int n_rows = src.ndim() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
    int row_blocks = (row_size + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

    std::vector<BlockQ8_0> blocks;
    blocks.reserve(total_blocks);

    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * Q8_0_BLOCK_SIZE;
            int k_end = std::min(base + Q8_0_BLOCK_SIZE, row_offset + row_size);
            int block_len = k_end - base;

            float max_abs = 0.0f;
            for (int k = base; k < k_end; ++k) {
                float abs_val = std::abs(src.data[k]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }

            BlockQ8_0 block;
            std::memset(&block, 0, sizeof(block));
            if (max_abs > 0.0f) {
                float d = max_abs / 127.0f;
                block.d = float_to_fp16(d);
                float stored_d = fp16_to_float(block.d);
                float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;
                for (int i = 0; i < block_len; ++i) {
                    float q = src.data[base + i] * id;
                    int qi = static_cast<int>(std::round(q));
                    if (qi > 127) {
                        qi = 127;
                    } else if (qi < -127) {
                        qi = -127;
                    }
                    block.qs[i] = static_cast<int8_t>(qi);
                }
            } else {
                block.d = 0;
            }
            // Pad remainder with 0
            for (int i = block_len; i < Q8_0_BLOCK_SIZE; ++i) {
                block.qs[i] = 0;
            }

            blocks.push_back(block);
        }
    }

    return blocks;
}

Tensor dequantize_from_q8_0(const std::vector<BlockQ8_0>& blocks, const std::vector<int>& shape) {
    size_t total = checked_numel(shape, "dequantize_from_q8_0");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
    if (blocks.size() != expected_blocks) {
        throw std::runtime_error(
            "dequantize_from_q8_0: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(blocks.size())
        );
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * Q8_0_BLOCK_SIZE;
            int k_end = std::min(base + Q8_0_BLOCK_SIZE, row_offset + row_size);
            const BlockQ8_0& block = blocks[block_idx++];
            float d = fp16_to_float(block.d);
            for (int k = base; k < k_end; ++k) {
                dst.data[k] = d * static_cast<float>(block.qs[k - base]);
            }
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// Q4_0 quantize / dequantize
// ---------------------------------------------------------------------------
std::vector<BlockQ4_0> quantize_to_q4_0(const Tensor& src) {
    if (src.size() == 0) {
        return {};
    }

    int row_size = src.ndim() >= 2 ? src.shape.back() : static_cast<int>(src.size());
    int n_rows = src.ndim() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
    int row_blocks = (row_size + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
    size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

    std::vector<BlockQ4_0> blocks;
    blocks.reserve(total_blocks);

    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * Q4_0_BLOCK_SIZE;
            int k_end = std::min(base + Q4_0_BLOCK_SIZE, row_offset + row_size);
            int block_len = k_end - base;

            float max_abs = 0.0f;
            for (int k = base; k < k_end; ++k) {
                float abs_val = std::abs(src.data[k]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }

            BlockQ4_0 block;
            std::memset(&block, 0, sizeof(block));
            if (max_abs > 0.0f) {
                float d = max_abs / 7.0f;  // max representable is 7 (q=15 -> 15-8=7)
                block.d = float_to_fp16(d);
                float stored_d = fp16_to_float(block.d);
                float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;

                for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; ++j) {
                    float x0 = 0.0f, x1 = 0.0f;
                    int idx0 = base + j;
                    int idx1 = base + j + Q4_0_BLOCK_SIZE / 2;
                    if (idx0 < k_end) {
                        x0 = src.data[idx0] * id;
                    }
                    if (idx1 < k_end) {
                        x1 = src.data[idx1] * id;
                    }
                    int qi0 = static_cast<int>(std::round(x0 + 8.0f));
                    int qi1 = static_cast<int>(std::round(x1 + 8.0f));
                    if (qi0 < 0) qi0 = 0;
                    if (qi0 > 15) qi0 = 15;
                    if (qi1 < 0) qi1 = 0;
                    if (qi1 > 15) qi1 = 15;
                    block.qs[j] = static_cast<uint8_t>(qi0 | (qi1 << 4));
                }
            } else {
                block.d = 0;
            }

            blocks.push_back(block);
        }
    }

    return blocks;
}

Tensor dequantize_from_q4_0(const std::vector<BlockQ4_0>& blocks, const std::vector<int>& shape) {
    size_t total = checked_numel(shape, "dequantize_from_q4_0");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
    size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
    if (blocks.size() != expected_blocks) {
        throw std::runtime_error(
            "dequantize_from_q4_0: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(blocks.size())
        );
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * Q4_0_BLOCK_SIZE;
            int k_end = std::min(base + Q4_0_BLOCK_SIZE, row_offset + row_size);
            const BlockQ4_0& block = blocks[block_idx++];
            float d = fp16_to_float(block.d);

            for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; ++j) {
                int idx0 = base + j;
                int idx1 = base + j + Q4_0_BLOCK_SIZE / 2;
                int q0 = static_cast<int>(block.qs[j] & 0x0F) - 8;
                int q1 = static_cast<int>(block.qs[j] >> 4) - 8;
                if (idx0 < k_end) {
                    dst.data[idx0] = d * static_cast<float>(q0);
                }
                if (idx1 < k_end) {
                    dst.data[idx1] = d * static_cast<float>(q1);
                }
            }
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// Q4_1 dequantize
// ---------------------------------------------------------------------------
Tensor dequantize_from_q4_1(const std::vector<BlockQ4_1>& blocks, const std::vector<int>& shape) {
    size_t total = checked_numel(shape, "dequantize_from_q4_1");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE;
    size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
    if (blocks.size() != expected_blocks) {
        throw std::runtime_error(
            "dequantize_from_q4_1: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(blocks.size())
        );
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * Q4_1_BLOCK_SIZE;
            int k_end = std::min(base + Q4_1_BLOCK_SIZE, row_offset + row_size);
            const BlockQ4_1& block = blocks[block_idx++];
            float d = fp16_to_float(block.d);
            float m = fp16_to_float(block.m);

            for (int j = 0; j < Q4_1_BLOCK_SIZE / 2; ++j) {
                int idx0 = base + j;
                int idx1 = base + j + Q4_1_BLOCK_SIZE / 2;
                int q0 = static_cast<int>(block.qs[j] & 0x0F);
                int q1 = static_cast<int>(block.qs[j] >> 4);
                if (idx0 < k_end) {
                    dst.data[idx0] = d * static_cast<float>(q0) + m;
                }
                if (idx1 < k_end) {
                    dst.data[idx1] = d * static_cast<float>(q1) + m;
                }
            }
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// True quantized linear (block-level on-the-fly)
// ---------------------------------------------------------------------------

// Shared helper: linear with quantized 2D weight [out_features, in_features]
// and 1D input [in_features] or 2D input [batch, in_features].
// result rank matches input rank.
template <typename BlockType, int BLOCK_SIZE>
static Tensor linear_quantized_impl(
    const Tensor& x,
    const std::vector<BlockType>& blocks,
    const std::vector<int>& weight_shape,
    float (*dequant_fn)(const BlockType&, int)
) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error("linear_quantized: expected 2D weight shape");
    }

    int in_features;
    int rows = 1;
    bool is_1d = false;
    if (x.ndim() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.ndim() == 2) {
        rows = x.shape[0];
        in_features = x.shape[1];
    } else {
        throw std::runtime_error(
            "linear_quantized: expected x shape [in_features] or [batch, in_features], got " + x.shape_str()
        );
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        throw std::runtime_error(
            "linear_quantized: dimension mismatch x=" + x.shape_str() + " W=" +
            QuantizedTensor{QuantType::F32, weight_shape}.shape_str()
        );
    }

    int n_blocks_per_row = (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t expected_blocks = static_cast<size_t>(out_features) * n_blocks_per_row;
    if (blocks.size() != expected_blocks) {
        throw std::runtime_error(
            "linear_quantized: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(blocks.size())
        );
    }

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);

    parallel_for(rows * out_features, [&](int begin, int end) {
        for (int index = begin; index < end; ++index) {
            int row = index / out_features;
            int j = index % out_features;
            const float* x_row = x.data.data() + static_cast<size_t>(row) * in_features;
            float sum = 0.0f;
            int block_base = j * n_blocks_per_row;
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockType& block = blocks[block_base + b];
                int base_k = b * BLOCK_SIZE;
                int k_end = std::min(base_k + BLOCK_SIZE, in_features);
                for (int k = base_k; k < k_end; ++k) {
                    float w = dequant_fn(block, k - base_k);
                    sum += w * x_row[k];
                }
            }
            result.data[static_cast<size_t>(row) * out_features + j] = sum;
        }
    });

    return result;
}

static float dequant_q8_0(const BlockQ8_0& block, int idx) {
    float d = fp16_to_float(block.d);
    return d * static_cast<float>(block.qs[idx]);
}

static float dequant_q4_0(const BlockQ4_0& block, int idx) {
    float d = fp16_to_float(block.d);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
    }
    return d * static_cast<float>(q);
}

static float dequant_q4_1(const BlockQ4_1& block, int idx) {
    float d = fp16_to_float(block.d);
    float m = fp16_to_float(block.m);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0F);
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4);
    }
    return d * static_cast<float>(q) + m;
}

#ifdef MINI_LLAMA_USE_NEON
// NEON-optimized Q8_0 linear: process 8 quantized values per SIMD iteration.
Tensor linear_q8_0(const Tensor& x, const std::vector<BlockQ8_0>& weight, const std::vector<int>& weight_shape) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error("linear_q8_0: expected 2D weight shape");
    }

    int in_features;
    int rows = 1;
    bool is_1d = false;
    if (x.ndim() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.ndim() == 2) {
        rows = x.shape[0];
        in_features = x.shape[1];
    } else {
        throw std::runtime_error(
            "linear_q8_0: expected x shape [in_features] or [batch, in_features], got " + x.shape_str()
        );
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        throw std::runtime_error(
            "linear_q8_0: dimension mismatch x=" + x.shape_str() + " W=" +
            QuantizedTensor{QuantType::F32, weight_shape}.shape_str()
        );
    }

    int n_blocks_per_row = (in_features + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    size_t expected_blocks = static_cast<size_t>(out_features) * n_blocks_per_row;
    if (weight.size() != expected_blocks) {
        throw std::runtime_error(
            "linear_q8_0: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(weight.size())
        );
    }

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);

    parallel_for(rows * out_features, [&](int begin, int end) {
        for (int index = begin; index < end; ++index) {
            int row = index / out_features;
            int j = index % out_features;
            const float* x_row = x.data.data() + static_cast<size_t>(row) * in_features;
            int block_base = j * n_blocks_per_row;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            float sum_scalar = 0.0f;

            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockQ8_0& block = weight[block_base + b];
                float d = fp16_to_float(block.d);
                int base_k = b * Q8_0_BLOCK_SIZE;
                int k_end = std::min(base_k + Q8_0_BLOCK_SIZE, in_features);
                // Process 8 elements at a time (2x float32x4_t)
                int k = base_k;
                for (; k + 8 <= k_end; k += 8) {
                    int8x8_t q8 = vld1_s8(block.qs + (k - base_k));
                    int16x8_t q16 = vmovl_s8(q8);

                    int32x4_t q32_lo = vmovl_s16(vget_low_s16(q16));
                    int32x4_t q32_hi = vmovl_s16(vget_high_s16(q16));

                    float32x4_t qf_lo = vcvtq_f32_s32(q32_lo);
                    float32x4_t qf_hi = vcvtq_f32_s32(q32_hi);

                    qf_lo = vmulq_n_f32(qf_lo, d);
                    qf_hi = vmulq_n_f32(qf_hi, d);

                    float32x4_t xf_lo = vld1q_f32(x_row + k);
                    float32x4_t xf_hi = vld1q_f32(x_row + k + 4);

                    sum_vec = vfmaq_f32(sum_vec, qf_lo, xf_lo);
                    sum_vec = vfmaq_f32(sum_vec, qf_hi, xf_hi);
                }

                // Horizontal reduce the 4-lane accumulator
                float32x2_t sum_lo = vget_low_f32(sum_vec);
                float32x2_t sum_hi = vget_high_f32(sum_vec);
                sum_lo = vadd_f32(sum_lo, sum_hi);
                float32x2_t sum_fp = vpadd_f32(sum_lo, sum_lo);
                sum_scalar += vget_lane_f32(sum_fp, 0);
                sum_vec = vdupq_n_f32(0.0f);

                // Tail: remaining elements (< 8)
                for (; k < k_end; ++k) {
                    float w = d * static_cast<float>(block.qs[k - base_k]);
                    sum_scalar += w * x_row[k];
                }
            }

            result.data[static_cast<size_t>(row) * out_features + j] = sum_scalar;
        }
    });

    return result;
}
#else
Tensor linear_q8_0(const Tensor& x, const std::vector<BlockQ8_0>& weight, const std::vector<int>& weight_shape) {
    return linear_quantized_impl<BlockQ8_0, Q8_0_BLOCK_SIZE>(x, weight, weight_shape, dequant_q8_0);
}
#endif

#ifdef MINI_LLAMA_USE_NEON
// NEON-optimized Q4_0 linear: unpack 4-bit nibbles and process 16 at a time.
Tensor linear_q4_0(const Tensor& x, const std::vector<BlockQ4_0>& weight, const std::vector<int>& weight_shape) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error("linear_q4_0: expected 2D weight shape");
    }

    int in_features;
    bool is_1d = false;
    if (x.ndim() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.ndim() == 2 && x.shape[0] == 1) {
        in_features = x.shape[1];
    } else {
        throw std::runtime_error(
            "linear_q4_0: expected x shape [in_features] or [1, in_features], got " + x.shape_str()
        );
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        throw std::runtime_error(
            "linear_q4_0: dimension mismatch x=" + x.shape_str() + " W=" +
            QuantizedTensor{QuantType::F32, weight_shape}.shape_str()
        );
    }

    int n_blocks_per_row = (in_features + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
    size_t expected_blocks = static_cast<size_t>(out_features) * n_blocks_per_row;
    if (weight.size() != expected_blocks) {
        throw std::runtime_error(
            "linear_q4_0: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(weight.size())
        );
    }

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features}, 0.0f);

    const uint8x16_t mask_low = vdupq_n_u8(0x0F);
    const int8x16_t zp = vdupq_n_s8(8);

    parallel_for(out_features, [&](int begin, int end) {
        for (int j = begin; j < end; ++j) {
            int block_base = j * n_blocks_per_row;
            float sum_scalar = 0.0f;

            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockQ4_0& block = weight[block_base + b];
                float d = fp16_to_float(block.d);
                int base_k = b * Q4_0_BLOCK_SIZE;
                int k_end = std::min(base_k + Q4_0_BLOCK_SIZE, in_features);
                int len = k_end - base_k;

                // Process full 16-value half-blocks with NEON. Partial tails
                // stay scalar so loads never cross the valid input range.
                int half = 0;
                for (; half < 2 && base_k + (half + 1) * 16 <= k_end; ++half) {
                    uint8x16_t packed = vld1q_u8(block.qs);
                    uint8x16_t nibble8;
                    if (half == 0) {
                        nibble8 = vandq_u8(packed, mask_low);
                    } else {
                        nibble8 = vshrq_n_u8(packed, 4);
                    }
                    int8x16_t q8 = vsubq_s8(vreinterpretq_s8_u8(nibble8), zp);

                    int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
                    int16x8_t q16_hi = vmovl_s8(vget_high_s8(q8));

                    int32x4_t q32_0 = vmovl_s16(vget_low_s16(q16_lo));
                    int32x4_t q32_1 = vmovl_s16(vget_high_s16(q16_lo));
                    int32x4_t q32_2 = vmovl_s16(vget_low_s16(q16_hi));
                    int32x4_t q32_3 = vmovl_s16(vget_high_s16(q16_hi));

                    float32x4_t qf_0 = vmulq_n_f32(vcvtq_f32_s32(q32_0), d);
                    float32x4_t qf_1 = vmulq_n_f32(vcvtq_f32_s32(q32_1), d);
                    float32x4_t qf_2 = vmulq_n_f32(vcvtq_f32_s32(q32_2), d);
                    float32x4_t qf_3 = vmulq_n_f32(vcvtq_f32_s32(q32_3), d);

                    int k_off = base_k + half * 16;
                    float32x4_t xf_0 = vld1q_f32(x.data.data() + k_off);
                    float32x4_t xf_1 = vld1q_f32(x.data.data() + k_off + 4);
                    float32x4_t xf_2 = vld1q_f32(x.data.data() + k_off + 8);
                    float32x4_t xf_3 = vld1q_f32(x.data.data() + k_off + 12);

                    float32x4_t sum0 = vmulq_f32(qf_0, xf_0);
                    float32x4_t sum1 = vmlaq_f32(sum0, qf_1, xf_1);
                    float32x4_t sum2 = vmlaq_f32(sum1, qf_2, xf_2);
                    float32x4_t sum3 = vmlaq_f32(sum2, qf_3, xf_3);

                    // Horizontal reduce
                    float32x2_t r_lo = vget_low_f32(sum3);
                    float32x2_t r_hi = vget_high_f32(sum3);
                    r_lo = vadd_f32(r_lo, r_hi);
                    float32x2_t r = vpadd_f32(r_lo, r_lo);
                    sum_scalar += vget_lane_f32(r, 0);
                }

                // Tail: should not happen for Q4_0_BLOCK_SIZE=32 unless in_features is odd
                int k = base_k + half * 16;
                for (; k < k_end; ++k) {
                    int idx = k - base_k;
                    int q;
                    if (idx < 16) {
                        q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
                    } else {
                        q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
                    }
                    float w = d * static_cast<float>(q);
                    sum_scalar += w * x.data[k];
                }
            }

            result.data[j] = sum_scalar;
        }
    });

    return result;
}
#else
Tensor linear_q4_0(const Tensor& x, const std::vector<BlockQ4_0>& weight, const std::vector<int>& weight_shape) {
    return linear_quantized_impl<BlockQ4_0, Q4_0_BLOCK_SIZE>(x, weight, weight_shape, dequant_q4_0);
}
#endif

Tensor linear_q4_1(const Tensor& x, const std::vector<BlockQ4_1>& weight, const std::vector<int>& weight_shape) {
    return linear_quantized_impl<BlockQ4_1, Q4_1_BLOCK_SIZE>(x, weight, weight_shape, dequant_q4_1);
}

// ---------------------------------------------------------------------------
// Legacy pseudo-quantized matmul (dequantizes to F32 then calls matmul)
// ---------------------------------------------------------------------------
Tensor matmul_q8_0(const std::vector<BlockQ8_0>& weight, const Tensor& input, const std::vector<int>& weight_shape) {
    Tensor weight_f32 = dequantize_from_q8_0(weight, weight_shape);
    return matmul(weight_f32, input);
}

Tensor matmul_q4_0(const std::vector<BlockQ4_0>& weight, const Tensor& input, const std::vector<int>& weight_shape) {
    Tensor weight_f32 = dequantize_from_q4_0(weight, weight_shape);
    return matmul(weight_f32, input);
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
float compare_matmul_error(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = matmul(weight, input);
    std::vector<BlockQ8_0> qweight = quantize_to_q8_0(weight);
    Tensor q8_result = matmul_q8_0(qweight, input, weight.shape);

    if (f32_result.shape != q8_result.shape) {
        throw std::runtime_error("compare_matmul_error: shape mismatch");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.size(); ++i) {
        float err = std::abs(f32_result.data[i] - q8_result.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

float compare_q4_0_error(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = linear(input, weight);
    std::vector<BlockQ4_0> qweight = quantize_to_q4_0(weight);
    Tensor q4_result = linear_q4_0(input, qweight, weight.shape);

    if (f32_result.shape != q4_result.shape) {
        throw std::runtime_error("compare_q4_0_error: shape mismatch");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.size(); ++i) {
        float err = std::abs(f32_result.data[i] - q4_result.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

} // namespace mini_llama
