#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/cuda_runtime.h"
#include <memory>
#include <string>
#include <vector>

namespace mini_llama {

enum class RopeType {
    Normal,
    NeoX,
};

struct ModelConfig {
    int vocab_size = 128;
    int dim = 32;
    int hidden_dim = 86;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 4;
    int head_dim = 8;
    int max_seq_len = 128;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    RopeType rope_type = RopeType::Normal;
};

struct LayerWeights {
    Tensor attention_norm;      // [dim] F32
    QuantizedTensor wq;         // [n_heads * head_dim, dim]  F32/Q8_0/Q4_0
    QuantizedTensor wk;         // [n_kv_heads * head_dim, dim]
    QuantizedTensor wv;         // [n_kv_heads * head_dim, dim]
    Tensor bq;                  // optional [n_heads * head_dim] F32
    Tensor bk;                  // optional [n_kv_heads * head_dim] F32
    Tensor bv;                  // optional [n_kv_heads * head_dim] F32
    QuantizedTensor wo;         // [dim, n_heads * head_dim]
    Tensor ffn_norm;            // [dim] F32
    QuantizedTensor w_gate;     // [hidden_dim, dim]
    QuantizedTensor w_up;       // [hidden_dim, dim]
    QuantizedTensor w_down;     // [dim, hidden_dim]
};

struct CudaUploadedWeight {
    std::string name;
    QuantType type = QuantType::F32;
    bool linear = true;
    std::vector<int> shape;
    size_t block_count = 0;
    CudaDeviceBuffer buffer;
};

struct CudaModelWeights {
    int device_id = 0;
    size_t uploaded_weight_count = 0;
    size_t uploaded_bytes = 0;
    size_t linear_calls = 0;
    size_t activation_calls = 0;
    size_t attention_calls = 0;
    size_t attention_cpu_fallbacks = 0;
    size_t kv_cache_write_bytes = 0;
    size_t kv_cache_read_bytes = 0;
    size_t host_to_device_copies = 0;
    size_t device_to_host_copies = 0;
    size_t host_to_device_bytes = 0;
    size_t device_to_host_bytes = 0;
    std::vector<CudaUploadedWeight> weights;
};

struct MiniLlamaModel {
    ModelConfig config;
    bool loaded = false;
    std::string load_error;
    Tensor token_embedding;           // [vocab_size, dim] F32
    std::vector<LayerWeights> layers; // [n_layers]
    Tensor final_norm;                // [dim] F32
    QuantizedTensor lm_head;          // [vocab_size, dim] F32/Q8_0/Q4_0
    std::shared_ptr<CudaModelWeights> cuda_weights;
};

// Return the actual bytes consumed by model weights in their current format.
size_t model_weight_bytes(const MiniLlamaModel& model);

// Return the bytes the same weights would consume if stored as F32.
size_t model_weight_bytes_f32(const MiniLlamaModel& model);

// Upload CUDA-supported resident weights and keep CPU weights intact.
void upload_model_weights_to_cuda(MiniLlamaModel& model, int device_id = 0);

void clear_model_cuda_weights(MiniLlamaModel& model);

bool model_has_cuda_weights(const MiniLlamaModel& model);

size_t model_cuda_uploaded_weight_count(const MiniLlamaModel& model);

size_t model_cuda_uploaded_f32_weight_count(const MiniLlamaModel& model);

size_t model_cuda_uploaded_q8_0_weight_count(const MiniLlamaModel& model);

size_t model_cuda_uploaded_q4_0_weight_count(const MiniLlamaModel& model);

size_t model_cuda_uploaded_q4_1_weight_count(const MiniLlamaModel& model);

size_t model_cuda_memory_bytes(const MiniLlamaModel& model);

void reset_model_cuda_runtime_stats(MiniLlamaModel& model);

size_t model_cuda_linear_calls(const MiniLlamaModel& model);

size_t model_cuda_activation_calls(const MiniLlamaModel& model);

size_t model_cuda_attention_calls(const MiniLlamaModel& model);

size_t model_cuda_attention_cpu_fallbacks(const MiniLlamaModel& model);

size_t model_cuda_kv_cache_write_bytes(const MiniLlamaModel& model);

size_t model_cuda_kv_cache_read_bytes(const MiniLlamaModel& model);

size_t model_cuda_host_to_device_copies(const MiniLlamaModel& model);

size_t model_cuda_device_to_host_copies(const MiniLlamaModel& model);

size_t model_cuda_host_to_device_bytes(const MiniLlamaModel& model);

size_t model_cuda_device_to_host_bytes(const MiniLlamaModel& model);

// Convert all linear weight QuantizedTensors from F32 to Q8_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void quantize_model_to_q8_0(MiniLlamaModel& model);

// Convert all linear weight QuantizedTensors from F32 to Q4_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void quantize_model_to_q4_0(MiniLlamaModel& model);

} // namespace mini_llama
