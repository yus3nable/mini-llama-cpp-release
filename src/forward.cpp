#include "mini_llama/forward.h"
#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_llama {

static void validate_forward_inputs(const MiniLlamaModel& model, const MiniLlamaContext& ctx, int token) {
    const ModelConfig& c = model.config;
    if (!model.loaded) {
        throw std::runtime_error("forward_token called with an unloaded model");
    }
    if (token < 0 || token >= c.vocab_size) {
        throw std::out_of_range("forward_token token id out of range");
    }
    if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
        throw std::out_of_range("forward_token position out of range");
    }
    if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
        throw std::runtime_error("forward_token layer count does not match model config");
    }
    if (model.token_embedding.data.size() < static_cast<size_t>(c.vocab_size * c.dim)) {
        throw std::runtime_error("forward_token token embedding tensor is smaller than config");
    }
}

static void require_shape(const Tensor& t, const std::vector<int>& expected, const char* caller) {
    t.assert_shape(expected, caller);
}

static Tensor add_optional_bias(const Tensor& x, const Tensor& bias, const char* caller) {
    if (bias.data.empty()) {
        return x;
    }
    if (x.ndim() != 1 || bias.ndim() != 1 || x.shape[0] != bias.shape[0]) {
        throw std::runtime_error(
            std::string(caller) + ": bias shape mismatch x=" + x.shape_str() +
            " bias=" + bias.shape_str()
        );
    }

    Tensor y = x;
    for (int i = 0; i < x.shape[0]; ++i) {
        y.data[i] += bias.data[i];
    }
    return y;
}

static const CudaUploadedWeight* find_cuda_weight(
    const MiniLlamaModel& model,
    const std::string& name,
    QuantType type
) {
    if (!model.cuda_weights) {
        return nullptr;
    }
    for (const auto& weight : model.cuda_weights->weights) {
        if (weight.name == name && weight.type == type) {
            return &weight;
        }
    }
    return nullptr;
}

static void record_cuda_linear_copy(const MiniLlamaModel& model, const Tensor& x, const Tensor& y) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls += 1;
    model.cuda_weights->host_to_device_copies += 1;
    model.cuda_weights->device_to_host_copies += 1;
    model.cuda_weights->host_to_device_bytes += x.size() * sizeof(float);
    model.cuda_weights->device_to_host_bytes += y.size() * sizeof(float);
}

static void record_cuda_linear_call(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls += 1;
}

static void record_cuda_host_to_device_copy(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->host_to_device_copies += 1;
    model.cuda_weights->host_to_device_bytes += bytes;
}

static void record_cuda_device_to_host_copy(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->device_to_host_copies += 1;
    model.cuda_weights->device_to_host_bytes += bytes;
}

static void record_cuda_activation_call(const MiniLlamaModel& model, size_t calls = 1) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->activation_calls += calls;
}

static void record_cuda_attention_call(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->attention_calls += 1;
}

static void record_cuda_attention_cpu_fallback(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->attention_cpu_fallbacks += 1;
}

static void record_cuda_kv_cache_write(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->kv_cache_write_bytes += bytes;
}

static void record_cuda_kv_cache_read(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->kv_cache_read_bytes += bytes;
}

static int cuda_device_id(const MiniLlamaModel& model) {
    return model.cuda_weights ? model.cuda_weights->device_id : 0;
}

static bool cuda_linear_supported(const QuantizedTensor& weight) {
    return weight.type == QuantType::F32 ||
        weight.type == QuantType::Q8_0 ||
        weight.type == QuantType::Q4_0 ||
        weight.type == QuantType::Q4_1;
}

static bool cuda_linear_ready(
    const MiniLlamaModel& model,
    const std::string& name,
    const QuantizedTensor& weight
) {
    return model.cuda_weights && cuda_linear_supported(weight) && find_cuda_weight(model, name, weight.type) != nullptr;
}

static CudaTensor upload_cuda_tensor(const MiniLlamaModel& model, const Tensor& x) {
    CudaTensor x_dev = cuda_tensor_from_host(x, cuda_device_id(model));
    record_cuda_host_to_device_copy(model, x.size() * sizeof(float));
    return x_dev;
}

static Tensor download_cuda_tensor(const MiniLlamaModel& model, const CudaTensor& x) {
    Tensor y = x.download();
    record_cuda_device_to_host_copy(model, y.size() * sizeof(float));
    return y;
}

static CudaTensor forward_linear_device(
    const MiniLlamaModel& model,
    const std::string& name,
    const CudaTensor& x,
    const QuantizedTensor& weight
) {
    if (!cuda_linear_supported(weight)) {
        throw std::runtime_error("CUDA forward linear unsupported weight type: " + name);
    }

    const CudaUploadedWeight* cuda_weight = find_cuda_weight(model, name, weight.type);
    if (cuda_weight == nullptr) {
        throw std::runtime_error("CUDA forward linear missing uploaded weight: " + name);
    }

    CudaTensor y;
    switch (weight.type) {
        case QuantType::F32:
            y = cuda_linear_device_input(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->shape,
                cuda_device_id(model)
            );
            break;
        case QuantType::Q8_0:
            y = cuda_q8_0_linear_device_input(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->block_count,
                cuda_weight->shape,
                cuda_device_id(model)
            );
            break;
        case QuantType::Q4_0:
            y = cuda_q4_0_linear_device_input(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->block_count,
                cuda_weight->shape,
                cuda_device_id(model)
            );
            break;
        case QuantType::Q4_1:
            y = cuda_q4_1_linear_device_input(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->block_count,
                cuda_weight->shape,
                cuda_device_id(model)
            );
            break;
    }
    record_cuda_linear_call(model);
    return y;
}

static Tensor forward_linear(
    const MiniLlamaModel& model,
    const std::string& name,
    const Tensor& x,
    const QuantizedTensor& weight
) {
    if (!model.cuda_weights) {
        return linear(x, weight);
    }

    if (!cuda_linear_supported(weight)) {
        return linear(x, weight);
    }

    const CudaUploadedWeight* cuda_weight = find_cuda_weight(model, name, weight.type);
    if (cuda_weight == nullptr) {
        throw std::runtime_error("CUDA forward linear missing uploaded weight: " + name);
    }
    Tensor y;
    switch (weight.type) {
        case QuantType::F32:
            y = cuda_linear_device_weight(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->shape,
                nullptr,
                model.cuda_weights->device_id
            );
            break;
        case QuantType::Q8_0:
            y = cuda_q8_0_linear_device_weight(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->block_count,
                cuda_weight->shape,
                nullptr,
                model.cuda_weights->device_id
            );
            break;
        case QuantType::Q4_0:
            y = cuda_q4_0_linear_device_weight(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->block_count,
                cuda_weight->shape,
                nullptr,
                model.cuda_weights->device_id
            );
            break;
        case QuantType::Q4_1:
            y = cuda_q4_1_linear_device_weight(
                x,
                cuda_weight->buffer.data(),
                cuda_weight->block_count,
                cuda_weight->shape,
                nullptr,
                model.cuda_weights->device_id
            );
            break;
    }
    record_cuda_linear_copy(model, x, y);
    return y;
}

static void forward_qkv_projection(
    const MiniLlamaModel& model,
    const std::string& layer_prefix,
    const Tensor& h,
    const LayerWeights& lw,
    Tensor& q_flat,
    Tensor& k_flat,
    Tensor& v_flat
) {
    const std::string q_name = layer_prefix + "wq";
    const std::string k_name = layer_prefix + "wk";
    const std::string v_name = layer_prefix + "wv";
    if (
        cuda_linear_ready(model, q_name, lw.wq) &&
        cuda_linear_ready(model, k_name, lw.wk) &&
        cuda_linear_ready(model, v_name, lw.wv)
    ) {
        CudaTensor h_dev = upload_cuda_tensor(model, h);
        CudaTensor q_dev = forward_linear_device(model, q_name, h_dev, lw.wq);
        CudaTensor k_dev = forward_linear_device(model, k_name, h_dev, lw.wk);
        CudaTensor v_dev = forward_linear_device(model, v_name, h_dev, lw.wv);
        q_flat = add_optional_bias(download_cuda_tensor(model, q_dev), lw.bq, "forward_layer q");
        k_flat = add_optional_bias(download_cuda_tensor(model, k_dev), lw.bk, "forward_layer k");
        v_flat = add_optional_bias(download_cuda_tensor(model, v_dev), lw.bv, "forward_layer v");
        return;
    }

    q_flat = add_optional_bias(
        forward_linear(model, q_name, h, lw.wq),
        lw.bq,
        "forward_layer q"
    );
    k_flat = add_optional_bias(
        forward_linear(model, k_name, h, lw.wk),
        lw.bk,
        "forward_layer k"
    );
    v_flat = add_optional_bias(
        forward_linear(model, v_name, h, lw.wv),
        lw.bv,
        "forward_layer v"
    );
}

static CudaTensor add_optional_bias_device(
    const MiniLlamaModel& model,
    const std::string& name,
    CudaTensor x,
    const Tensor& bias
) {
    if (bias.data.empty()) {
        return x;
    }
    const CudaUploadedWeight* cuda_weight = find_cuda_weight(model, name, QuantType::F32);
    if (cuda_weight == nullptr || cuda_weight->linear) {
        throw std::runtime_error("CUDA forward bias missing uploaded weight: " + name);
    }
    if (cuda_weight->shape != bias.shape) {
        throw std::runtime_error("CUDA forward bias uploaded weight shape mismatch: " + name);
    }
    CudaTensor y = cuda_elementwise_add_device_weight(
        x,
        cuda_weight->buffer.data(),
        cuda_weight->shape,
        cuda_device_id(model)
    );
    record_cuda_activation_call(model);
    return y;
}

static void forward_qkv_projection_device(
    const MiniLlamaModel& model,
    const std::string& layer_prefix,
    const CudaTensor& h,
    const LayerWeights& lw,
    CudaTensor& q,
    CudaTensor& k,
    CudaTensor& v
) {
    q = forward_linear_device(model, layer_prefix + "wq", h, lw.wq);
    k = forward_linear_device(model, layer_prefix + "wk", h, lw.wk);
    v = forward_linear_device(model, layer_prefix + "wv", h, lw.wv);
    q = add_optional_bias_device(model, layer_prefix + "bq", std::move(q), lw.bq);
    k = add_optional_bias_device(model, layer_prefix + "bk", std::move(k), lw.bk);
    v = add_optional_bias_device(model, layer_prefix + "bv", std::move(v), lw.bv);
}

static Tensor forward_rmsnorm(
    const MiniLlamaModel& model,
    const Tensor& x,
    const Tensor& weight,
    float eps
) {
    if (!model.cuda_weights) {
        return rmsnorm(x, weight, eps);
    }
    Tensor y = cuda_rmsnorm(x, weight, eps, cuda_device_id(model));
    record_cuda_activation_call(model);
    return y;
}

static CudaTensor forward_rmsnorm_device(
    const MiniLlamaModel& model,
    const std::string& name,
    const CudaTensor& x,
    const Tensor& weight,
    float eps
) {
    const CudaUploadedWeight* cuda_weight = find_cuda_weight(model, name, QuantType::F32);
    if (cuda_weight == nullptr || cuda_weight->linear) {
        throw std::runtime_error("CUDA forward RMSNorm missing uploaded weight: " + name);
    }
    if (cuda_weight->shape != weight.shape) {
        throw std::runtime_error("CUDA forward RMSNorm uploaded weight shape mismatch: " + name);
    }
    CudaTensor y = cuda_rmsnorm_device_weight(
        x,
        cuda_weight->buffer.data(),
        cuda_weight->shape,
        eps,
        cuda_device_id(model)
    );
    record_cuda_activation_call(model);
    return y;
}

static Tensor forward_swiglu(const MiniLlamaModel& model, const Tensor& gate, const Tensor& up) {
    if (!model.cuda_weights) {
        return swiglu(gate, up);
    }
    Tensor silu_gate = cuda_silu(gate, cuda_device_id(model));
    Tensor y = cuda_elementwise_mul(silu_gate, up, cuda_device_id(model));
    record_cuda_activation_call(model, 2);
    return y;
}

static CudaTensor forward_add_device(const MiniLlamaModel& model, const CudaTensor& a, const CudaTensor& b) {
    CudaTensor y = cuda_elementwise_add_device_input(a, b, cuda_device_id(model));
    record_cuda_activation_call(model);
    return y;
}

static Tensor forward_add(const MiniLlamaModel& model, const Tensor& a, const Tensor& b) {
    if (!model.cuda_weights) {
        if (a.shape != b.shape) {
            throw std::runtime_error(
                "forward_add: shape mismatch " + a.shape_str() + " vs " + b.shape_str()
            );
        }
        Tensor y(a.shape, 0.0f);
        for (size_t i = 0; i < a.size(); ++i) {
            y.data[i] = a.data[i] + b.data[i];
        }
        return y;
    }
    Tensor y = cuda_elementwise_add(a, b, cuda_device_id(model));
    record_cuda_activation_call(model);
    return y;
}

static Tensor forward_softmax(const MiniLlamaModel& model, const Tensor& x) {
    if (!model.cuda_weights) {
        return softmax(x);
    }
    Tensor y = cuda_softmax(x, cuda_device_id(model));
    record_cuda_activation_call(model);
    return y;
}

static void forward_rope(
    const MiniLlamaModel& model,
    Tensor& q,
    Tensor& k,
    int pos,
    float theta,
    RopeType rope_type
) {
    if (!model.cuda_weights) {
        rope(q, k, pos, theta, rope_type);
        return;
    }
    cuda_rope(q, k, pos, theta, rope_type, cuda_device_id(model));
    record_cuda_activation_call(model);
}

static void forward_rope_device(
    const MiniLlamaModel& model,
    CudaTensor& q,
    CudaTensor& k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float theta,
    RopeType rope_type
) {
    cuda_rope_device_input(q, k, n_heads, n_kv_heads, head_dim, pos, theta, rope_type, cuda_device_id(model));
    record_cuda_activation_call(model);
}

// ---------------------------------------------------------------------------
// Embedding lookup: token_id -> x[dim]
// ---------------------------------------------------------------------------
static Tensor embed_token(const MiniLlamaModel& model, int token_id) {
    int dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    for (int i = 0; i < dim; ++i) {
        x.data[i] = model.token_embedding.data[token_id * dim + i];
    }
    return x;
}

static CudaTensor embed_token_device(const MiniLlamaModel& model, int token_id) {
    const CudaUploadedWeight* cuda_weight = find_cuda_weight(model, "token_embedding", QuantType::F32);
    if (cuda_weight == nullptr || cuda_weight->linear) {
        throw std::runtime_error("CUDA forward embedding missing uploaded weight: token_embedding");
    }
    if (cuda_weight->shape != model.token_embedding.shape) {
        throw std::runtime_error("CUDA forward embedding uploaded weight shape mismatch");
    }
    return cuda_embedding_lookup_device_weight(
        cuda_weight->buffer.data(),
        cuda_weight->shape,
        token_id,
        cuda_device_id(model)
    );
}

// ---------------------------------------------------------------------------
// GQA head mapping: given a query head, return the corresponding kv head
// ---------------------------------------------------------------------------
static int map_q_head_to_kv_head(int q_head, int n_heads, int n_kv_heads) {
    return q_head / (n_heads / n_kv_heads);
}

// ---------------------------------------------------------------------------
// Attention forward for a single layer
//   q: [n_heads, head_dim]
//   k: [n_kv_heads, head_dim]
//   v: [n_kv_heads, head_dim]
//   pos: current position
//   layer: layer index
//   kv_cache: global kv cache
//   n_heads, n_kv_heads, head_dim
// Returns attention output: [n_heads, head_dim]
// ---------------------------------------------------------------------------
static Tensor attention_forward(
    const MiniLlamaModel& model,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    int pos,
    int layer,
    KVCache& kv_cache,
    CudaKVCache* cuda_kv_cache,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    // Write current k, v into cache
    kv_cache.write(layer, pos, k, v);
    if (cuda_kv_cache != nullptr && !cuda_kv_cache->empty()) {
        cuda_kv_cache->write(layer, pos, k, v);
        record_cuda_kv_cache_write(model, (k.size() + v.size()) * sizeof(float));
        Tensor attn_out = cuda_attention_decode(
            q,
            *cuda_kv_cache,
            layer,
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            cuda_device_id(model)
        );
        record_cuda_attention_call(model);
        record_cuda_kv_cache_read(
            model,
            static_cast<size_t>(pos + 1) *
                static_cast<size_t>(n_heads) *
                static_cast<size_t>(head_dim) *
                sizeof(float) *
                2
        );
        record_cuda_host_to_device_copy(model, q.size() * sizeof(float));
        record_cuda_device_to_host_copy(model, attn_out.size() * sizeof(float));
        return attn_out;
    }
    if (model.cuda_weights) {
        record_cuda_attention_cpu_fallback(model);
    }

    Tensor attn_out({n_heads, head_dim}, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto compute_head = [&](int h) {
        int kv_head = map_q_head_to_kv_head(h, n_heads, n_kv_heads);

        std::vector<float> scores_data(pos + 1, 0.0f);
        for (int t = 0; t <= pos; ++t) {
            const float* k_ptr = kv_cache.key_ptr(layer, t, kv_head);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q.data[h * head_dim + d] * k_ptr[d];
            }
            scores_data[t] = dot * scale;
        }

        Tensor scores({pos + 1}, 0.0f);
        scores.data = std::move(scores_data);
        Tensor probs = forward_softmax(model, scores);

        for (int d = 0; d < head_dim; ++d) {
            float val = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                const float* v_ptr = kv_cache.value_ptr(layer, t, kv_head);
                val += probs.data[t] * v_ptr[d];
            }
            attn_out.data[h * head_dim + d] = val;
        }
    };

    if (model.cuda_weights) {
        for (int h = 0; h < n_heads; ++h) {
            compute_head(h);
        }
        return attn_out;
    }

    parallel_for(n_heads, [&](int begin, int end) {
        for (int h = begin; h < end; ++h) {
            compute_head(h);
        }
    });

    return attn_out;
}

static CudaTensor attention_forward_device(
    const MiniLlamaModel& model,
    const CudaTensor& q,
    const CudaTensor& k,
    const CudaTensor& v,
    int pos,
    int layer,
    CudaKVCache& cuda_kv_cache,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    cuda_kv_cache.write_device(layer, pos, k, v);
    record_cuda_kv_cache_write(model, (k.size() + v.size()) * sizeof(float));
    CudaTensor attn_out = cuda_attention_decode_device_input(
        q,
        cuda_kv_cache,
        layer,
        pos,
        n_heads,
        n_kv_heads,
        head_dim,
        cuda_device_id(model)
    );
    record_cuda_attention_call(model);
    record_cuda_kv_cache_read(
        model,
        static_cast<size_t>(pos + 1) *
            static_cast<size_t>(n_heads) *
            static_cast<size_t>(head_dim) *
            sizeof(float) *
            2
    );
    return attn_out;
}

// ---------------------------------------------------------------------------
// FFN forward (SwiGLU) for a single layer
//   h: [dim]  (already normalized)
//   lw: layer weights
// Returns FFN output: [dim]
// ---------------------------------------------------------------------------
static Tensor ffn_forward(
    const MiniLlamaModel& model,
    const std::string& layer_prefix,
    const Tensor& h,
    const LayerWeights& lw
) {
    const std::string gate_name = layer_prefix + "w_gate";
    const std::string up_name = layer_prefix + "w_up";
    const std::string down_name = layer_prefix + "w_down";
    if (
        cuda_linear_ready(model, gate_name, lw.w_gate) &&
        cuda_linear_ready(model, up_name, lw.w_up) &&
        cuda_linear_ready(model, down_name, lw.w_down)
    ) {
        CudaTensor h_dev = upload_cuda_tensor(model, h);
        CudaTensor gate_dev = forward_linear_device(model, gate_name, h_dev, lw.w_gate);
        CudaTensor up_dev = forward_linear_device(model, up_name, h_dev, lw.w_up);
        CudaTensor silu_gate_dev = cuda_silu_device_input(gate_dev, cuda_device_id(model));
        CudaTensor ff_dev = cuda_elementwise_mul_device_input(silu_gate_dev, up_dev, cuda_device_id(model));
        record_cuda_activation_call(model, 2);
        CudaTensor out_dev = forward_linear_device(model, down_name, ff_dev, lw.w_down);
        return download_cuda_tensor(model, out_dev);
    }

    Tensor gate = forward_linear(model, layer_prefix + "w_gate", h, lw.w_gate);  // [hidden_dim]
    Tensor up   = forward_linear(model, layer_prefix + "w_up", h, lw.w_up);      // [hidden_dim]

    Tensor ff = forward_linear(model, layer_prefix + "w_down", forward_swiglu(model, gate, up), lw.w_down);  // [dim]

    return ff;
}

static CudaTensor ffn_forward_device(
    const MiniLlamaModel& model,
    const std::string& layer_prefix,
    const CudaTensor& h,
    const LayerWeights& lw
) {
    CudaTensor gate = forward_linear_device(model, layer_prefix + "w_gate", h, lw.w_gate);
    CudaTensor up = forward_linear_device(model, layer_prefix + "w_up", h, lw.w_up);
    CudaTensor silu_gate = cuda_silu_device_input(gate, cuda_device_id(model));
    CudaTensor ff = cuda_elementwise_mul_device_input(silu_gate, up, cuda_device_id(model));
    record_cuda_activation_call(model, 2);
    return forward_linear_device(model, layer_prefix + "w_down", ff, lw.w_down);
}

// ---------------------------------------------------------------------------
// Single layer forward
//   x: input hidden state [dim]
//   layer: layer index
//   ctx: inference context (for pos and kv_cache)
//   lw: layer weights
// Returns updated hidden state [dim]
// ---------------------------------------------------------------------------
static Tensor forward_layer(
    const MiniLlamaModel& model,
    const Tensor& x,
    int layer,
    MiniLlamaContext& ctx,
    const LayerWeights& lw,
    const ModelConfig& c
) {
    int dim = c.dim;
    int n_heads = c.n_heads;
    int n_kv_heads = c.n_kv_heads;
    int head_dim = c.head_dim;
    int pos = ctx.pos;
    const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

    // ---- Attention sublayer ----
    Tensor h = forward_rmsnorm(model, x, lw.attention_norm, c.rms_norm_eps);

    Tensor q_flat;
    Tensor k_flat;
    Tensor v_flat;
    forward_qkv_projection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);

    Tensor q = q_flat.reshape_checked({n_heads, head_dim}, "forward_layer q");
    Tensor k = k_flat.reshape_checked({n_kv_heads, head_dim}, "forward_layer k");
    Tensor v = v_flat.reshape_checked({n_kv_heads, head_dim}, "forward_layer v");

    // Apply RoPE to q and k
    forward_rope(model, q, k, pos, c.rope_theta, c.rope_type);

    // Attention: compute + read/write KV cache
    Tensor attn_out = attention_forward(
        model, q, k, v, pos, layer, ctx.kv_cache,
        model.cuda_weights ? &ctx.cuda_kv_cache : nullptr,
        n_heads, n_kv_heads, head_dim
    );

    // Project and residual
    Tensor attn_out_flat = attn_out.reshape_checked(
        {1, n_heads * head_dim},
        "forward_layer attn_out"
    );
    Tensor attn_proj = forward_linear(model, layer_prefix + "wo", attn_out_flat, lw.wo);  // [dim]
    require_shape(attn_proj, {1, dim}, "forward_layer attn_proj");
    Tensor attn_proj_1d = attn_proj.reshape_checked({dim}, "forward_layer attn_proj 1d");

    Tensor x_attn = forward_add(model, x, attn_proj_1d);

    // ---- FFN sublayer ----
    Tensor h2 = forward_rmsnorm(model, x_attn, lw.ffn_norm, c.rms_norm_eps);
    Tensor ff = ffn_forward(model, layer_prefix, h2, lw);
    require_shape(ff, {dim}, "forward_layer ffn");

    return forward_add(model, x_attn, ff);
}

static CudaTensor forward_layer_device(
    const MiniLlamaModel& model,
    const CudaTensor& x,
    int layer,
    MiniLlamaContext& ctx,
    const LayerWeights& lw,
    const ModelConfig& c
) {
    int dim = c.dim;
    int n_heads = c.n_heads;
    int n_kv_heads = c.n_kv_heads;
    int head_dim = c.head_dim;
    int pos = ctx.pos;
    const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

    CudaTensor h = forward_rmsnorm_device(
        model,
        layer_prefix + "attention_norm",
        x,
        lw.attention_norm,
        c.rms_norm_eps
    );

    CudaTensor q;
    CudaTensor k;
    CudaTensor v;
    forward_qkv_projection_device(model, layer_prefix, h, lw, q, k, v);
    forward_rope_device(model, q, k, n_heads, n_kv_heads, head_dim, pos, c.rope_theta, c.rope_type);

    CudaTensor attn_out = attention_forward_device(
        model,
        q,
        k,
        v,
        pos,
        layer,
        ctx.cuda_kv_cache,
        n_heads,
        n_kv_heads,
        head_dim
    );

    CudaTensor attn_proj = forward_linear_device(model, layer_prefix + "wo", attn_out, lw.wo);
    if (attn_proj.size() != static_cast<size_t>(dim)) {
        throw std::runtime_error("forward_layer_device: attention projection shape mismatch");
    }
    CudaTensor x_attn = forward_add_device(model, x, attn_proj);

    CudaTensor h2 = forward_rmsnorm_device(
        model,
        layer_prefix + "ffn_norm",
        x_attn,
        lw.ffn_norm,
        c.rms_norm_eps
    );
    CudaTensor ff = ffn_forward_device(model, layer_prefix, h2, lw);
    if (ff.size() != static_cast<size_t>(dim)) {
        throw std::runtime_error("forward_layer_device: FFN output shape mismatch");
    }
    return forward_add_device(model, x_attn, ff);
}

// ---------------------------------------------------------------------------
// Compute logits from final hidden state
//   x: [dim]
// Returns logits: [vocab_size]
// ---------------------------------------------------------------------------
static Tensor compute_logits(const Tensor& x, const MiniLlamaModel& model, const ModelConfig& c) {
    Tensor normed = forward_rmsnorm(model, x, model.final_norm, c.rms_norm_eps);
    Tensor logits_flat = forward_linear(model, "lm_head", normed, model.lm_head);  // [vocab_size]
    return logits_flat.reshape_checked({c.vocab_size}, "compute_logits");
}

static Tensor compute_logits_device(const CudaTensor& x, const MiniLlamaModel& model, const ModelConfig& c) {
    CudaTensor normed = forward_rmsnorm_device(model, "final_norm", x, model.final_norm, c.rms_norm_eps);
    CudaTensor logits_dev = forward_linear_device(model, "lm_head", normed, model.lm_head);
    Tensor logits = download_cuda_tensor(model, logits_dev);
    return logits.reshape_checked({c.vocab_size}, "compute_logits_device");
}

// ---------------------------------------------------------------------------
// Forward pass for a single token
// ---------------------------------------------------------------------------
Tensor forward_token(
    const MiniLlamaModel& model,
    MiniLlamaContext& ctx,
    int token
) {
    validate_forward_inputs(model, ctx, token);

    const ModelConfig& c = model.config;
    int n_layers = c.n_layers;
    if (model.cuda_weights && ctx.cuda_kv_cache.empty()) {
        ctx.cuda_kv_cache.reset(
            c.n_layers,
            c.max_seq_len,
            c.n_kv_heads,
            c.head_dim,
            model.cuda_weights->device_id
        );
    }

    // 1. Embedding lookup
    if (model.cuda_weights) {
        CudaTensor x_dev = embed_token_device(model, token);
        for (int layer = 0; layer < n_layers; ++layer) {
            x_dev = forward_layer_device(model, x_dev, layer, ctx, model.layers[layer], c);
        }
        return compute_logits_device(x_dev, model, c);
    }

    Tensor x = embed_token(model, token);

    // 2. Transformer layers
    for (int layer = 0; layer < n_layers; ++layer) {
        x = forward_layer(model, x, layer, ctx, model.layers[layer], c);
    }

    // 3. Final norm + logits
    Tensor logits = compute_logits(x, model, c);

    return logits;
}

} // namespace mini_llama
