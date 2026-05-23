#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
#include "mini_llama/thread_pool.h"
#include "mini_llama/matmul_dispatch.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mini_llama {

Tensor matmul(const Tensor& A, const Tensor& B) {
    return matmul_dispatch(A, B, default_matmul_mode());
}

Tensor linear(const Tensor& x, const Tensor& W) {
    return linear_dispatch(x, W, default_matmul_mode());
}

Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    // x: [dim], weight: [dim]
    if (x.ndim() != 1 || weight.ndim() != 1) {
        throw std::runtime_error("rmsnorm: expected 1D tensors");
    }
    int dim = x.shape[0];
    if (weight.shape[0] != dim) {
        throw std::runtime_error("rmsnorm: dimension mismatch");
    }

    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) {
        ss += x.data[i] * x.data[i];
    }
    ss /= static_cast<float>(dim);
    float scale = 1.0f / std::sqrt(ss + eps);

    Tensor y({dim}, 0.0f);
    for (int i = 0; i < dim; ++i) {
        y.data[i] = x.data[i] * scale * weight.data[i];
    }
    return y;
}

Tensor softmax(const Tensor& x) {
    if (x.ndim() != 1) {
        throw std::runtime_error(
            "softmax: expected 1D tensor, got " + x.shape_str()
        );
    }
    if (x.size() == 0) {
        throw std::runtime_error("softmax: cannot compute softmax of empty tensor");
    }
    int n = x.shape[0];
    float max_val = x.data[0];
    for (int i = 1; i < n; ++i) {
        if (x.data[i] > max_val) {
            max_val = x.data[i];
        }
    }

    float sum = 0.0f;
    Tensor y({n}, 0.0f);
    for (int i = 0; i < n; ++i) {
        y.data[i] = std::exp(x.data[i] - max_val);
        sum += y.data[i];
    }
    for (int i = 0; i < n; ++i) {
        y.data[i] /= sum;
    }
    return y;
}

Tensor silu(const Tensor& x) {
    Tensor y(x.shape, 0.0f);
    for (size_t i = 0; i < x.data.size(); ++i) {
        float val = x.data[i];
        y.data[i] = val / (1.0f + std::exp(-val));
    }
    return y;
}

Tensor elementwise_mul(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error(
            "elementwise_mul: shape mismatch " + a.shape_str() + " vs " + b.shape_str()
        );
    }
    Tensor y(a.shape, 0.0f);
    for (size_t i = 0; i < a.data.size(); ++i) {
        y.data[i] = a.data[i] * b.data[i];
    }
    return y;
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    if (gate.shape != up.shape) {
        throw std::runtime_error(
            "swiglu: shape mismatch " + gate.shape_str() + " vs " + up.shape_str()
        );
    }
    Tensor y(gate.shape, 0.0f);
    for (size_t i = 0; i < gate.data.size(); ++i) {
        float val = gate.data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        y.data[i] = val * sigmoid * up.data[i];
    }
    return y;
}

static void apply_rope_normal_to_heads(Tensor& x, int pos, float theta) {
    int n_heads = x.shape[0];
    int head_dim = x.shape[1];
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(head_dim));
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x.data[h * head_dim + i];
            float x1 = x.data[h * head_dim + i + 1];
            x.data[h * head_dim + i] = x0 * cos_val - x1 * sin_val;
            x.data[h * head_dim + i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

static void apply_rope_neox_to_heads(Tensor& x, int pos, float theta) {
    int n_heads = x.shape[0];
    int head_dim = x.shape[1];
    int half_dim = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        int base = h * head_dim;
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x.data[base + i];
            float x1 = x.data[base + half_dim + i];
            x.data[base + i] = x0 * cos_val - x1 * sin_val;
            x.data[base + half_dim + i] = x0 * sin_val + x1 * cos_val;
        }
    }
}

void rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
    // q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
    if (q.ndim() != 2 || k.ndim() != 2) {
        throw std::runtime_error("rope: expected 2D tensors");
    }
    if (pos < 0) {
        throw std::out_of_range("rope: position must be non-negative");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("rope: theta must be finite and positive");
    }

    int n_heads = q.shape[0];
    int head_dim = q.shape[1];
    int n_kv_heads = k.shape[0];
    int k_head_dim = k.shape[1];
    if (head_dim != k_head_dim) {
        throw std::runtime_error(
            "rope: q and k head_dim mismatch " + q.shape_str() + " vs " + k.shape_str()
        );
    }
    if (head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error(
            "rope: head_dim must be positive and even, got " + std::to_string(head_dim)
        );
    }

    if (rope_type == RopeType::NeoX) {
        apply_rope_neox_to_heads(q, pos, theta);
        apply_rope_neox_to_heads(k, pos, theta);
    } else {
        apply_rope_normal_to_heads(q, pos, theta);
        apply_rope_normal_to_heads(k, pos, theta);
    }
}

int argmax(const Tensor& x) {
    if (x.size() == 0) {
        throw std::runtime_error("argmax: cannot compute argmax of empty tensor");
    }
    int best = 0;
    float best_val = x.data[0];
    for (size_t i = 1; i < x.data.size(); ++i) {
        if (x.data[i] > best_val) {
            best_val = x.data[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Quantized linear dispatch
// ---------------------------------------------------------------------------
Tensor linear(const Tensor& x, const QuantizedTensor& W) {
    switch (W.type) {
        case QuantType::F32:
            return linear(x, to_tensor(W));
        case QuantType::Q8_0:
            return linear_q8_0(x, W.q8_0_data, W.shape);
        case QuantType::Q4_0:
            return linear_q4_0(x, W.q4_0_data, W.shape);
        case QuantType::Q4_1:
            return linear_q4_1(x, W.q4_1_data, W.shape);
    }
    throw std::runtime_error("linear: unknown quant type");
}

} // namespace mini_llama
