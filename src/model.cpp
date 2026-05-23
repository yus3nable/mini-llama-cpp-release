#include "mini_llama/model.h"
#include "mini_llama/quant.h"

#include <stdexcept>
#include <utility>

namespace mini_llama {

static size_t tensor_bytes(const Tensor& t) {
    return t.data.size() * sizeof(float);
}

static size_t quantized_tensor_bytes(const QuantizedTensor& qt) {
    switch (qt.type) {
        case QuantType::F32:
            return qt.f32_data.size() * sizeof(float);
        case QuantType::Q8_0:
            return qt.q8_0_data.size() * sizeof(BlockQ8_0);
        case QuantType::Q4_0:
            return qt.q4_0_data.size() * sizeof(BlockQ4_0);
        case QuantType::Q4_1:
            return qt.q4_1_data.size() * sizeof(BlockQ4_1);
    }
    return 0;
}

static size_t quantized_tensor_bytes_f32(const QuantizedTensor& qt) {
    return qt.numel() * sizeof(float);
}

size_t model_weight_bytes(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += tensor_bytes(model.token_embedding);
    bytes += tensor_bytes(model.final_norm);
    bytes += quantized_tensor_bytes(model.lm_head);
    for (const auto& lw : model.layers) {
        bytes += tensor_bytes(lw.attention_norm);
        bytes += quantized_tensor_bytes(lw.wq);
        bytes += quantized_tensor_bytes(lw.wk);
        bytes += quantized_tensor_bytes(lw.wv);
        bytes += tensor_bytes(lw.bq);
        bytes += tensor_bytes(lw.bk);
        bytes += tensor_bytes(lw.bv);
        bytes += quantized_tensor_bytes(lw.wo);
        bytes += tensor_bytes(lw.ffn_norm);
        bytes += quantized_tensor_bytes(lw.w_gate);
        bytes += quantized_tensor_bytes(lw.w_up);
        bytes += quantized_tensor_bytes(lw.w_down);
    }
    return bytes;
}

size_t model_weight_bytes_f32(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += tensor_bytes(model.token_embedding);
    bytes += tensor_bytes(model.final_norm);
    bytes += quantized_tensor_bytes_f32(model.lm_head);
    for (const auto& lw : model.layers) {
        bytes += tensor_bytes(lw.attention_norm);
        bytes += quantized_tensor_bytes_f32(lw.wq);
        bytes += quantized_tensor_bytes_f32(lw.wk);
        bytes += quantized_tensor_bytes_f32(lw.wv);
        bytes += tensor_bytes(lw.bq);
        bytes += tensor_bytes(lw.bk);
        bytes += tensor_bytes(lw.bv);
        bytes += quantized_tensor_bytes_f32(lw.wo);
        bytes += tensor_bytes(lw.ffn_norm);
        bytes += quantized_tensor_bytes_f32(lw.w_gate);
        bytes += quantized_tensor_bytes_f32(lw.w_up);
        bytes += quantized_tensor_bytes_f32(lw.w_down);
    }
    return bytes;
}

static void upload_f32_linear_weight(
    CudaModelWeights& storage,
    const QuantizedTensor& weight,
    const std::string& name,
    int device_id
) {
    if (weight.type != QuantType::F32) {
        return;
    }
    if (weight.f32_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty F32 data for " + name);
    }

    const size_t expected_numel = weight.numel();
    if (weight.f32_data.size() != expected_numel) {
        throw std::runtime_error(
            "CUDA weight upload: data size mismatch for " + name +
            ", expected " + std::to_string(expected_numel) +
            ", got " + std::to_string(weight.f32_data.size())
        );
    }

    const size_t bytes = weight.f32_data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::F32;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.f32_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void upload_f32_tensor_weight(
    CudaModelWeights& storage,
    const Tensor& weight,
    const std::string& name,
    int device_id
) {
    if (weight.data.empty()) {
        return;
    }
    const size_t expected_numel = weight.size();
    if (weight.data.size() != expected_numel) {
        throw std::runtime_error(
            "CUDA weight upload: tensor data size mismatch for " + name +
            ", expected " + std::to_string(expected_numel) +
            ", got " + std::to_string(weight.data.size())
        );
    }

    const size_t bytes = weight.data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::F32;
    uploaded.linear = false;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void upload_q8_0_linear_weight(
    CudaModelWeights& storage,
    const QuantizedTensor& weight,
    const std::string& name,
    int device_id
) {
    if (weight.type != QuantType::Q8_0) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q8_0 weight for " + name);
    }
    if (weight.q8_0_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q8_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q8_0_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q8_0 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) +
            ", got " + std::to_string(weight.q8_0_data.size())
        );
    }

    const size_t bytes = weight.q8_0_data.size() * sizeof(BlockQ8_0);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::Q8_0;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q8_0_data.size();
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.q8_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void upload_q4_0_linear_weight(
    CudaModelWeights& storage,
    const QuantizedTensor& weight,
    const std::string& name,
    int device_id
) {
    if (weight.type != QuantType::Q4_0) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q4_0 weight for " + name);
    }
    if (weight.q4_0_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q4_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q4_0_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q4_0 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) +
            ", got " + std::to_string(weight.q4_0_data.size())
        );
    }

    const size_t bytes = weight.q4_0_data.size() * sizeof(BlockQ4_0);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::Q4_0;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_0_data.size();
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.q4_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void upload_q4_1_linear_weight(
    CudaModelWeights& storage,
    const QuantizedTensor& weight,
    const std::string& name,
    int device_id
) {
    if (weight.type != QuantType::Q4_1) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q4_1 weight for " + name);
    }
    if (weight.q4_1_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q4_1 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q4_1_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q4_1 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) +
            ", got " + std::to_string(weight.q4_1_data.size())
        );
    }

    const size_t bytes = weight.q4_1_data.size() * sizeof(BlockQ4_1);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::Q4_1;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_1_data.size();
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.q4_1_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void upload_linear_weight(
    CudaModelWeights& storage,
    const QuantizedTensor& weight,
    const std::string& name,
    int device_id
) {
    upload_f32_linear_weight(storage, weight, name, device_id);
    upload_q8_0_linear_weight(storage, weight, name, device_id);
    upload_q4_0_linear_weight(storage, weight, name, device_id);
    upload_q4_1_linear_weight(storage, weight, name, device_id);
}

void upload_model_weights_to_cuda(MiniLlamaModel& model, int device_id) {
    if (!model.loaded) {
        throw std::runtime_error("CUDA weight upload requires a loaded model");
    }

    cuda_set_device(device_id);
    auto storage = std::make_shared<CudaModelWeights>();
    storage->device_id = device_id;

    upload_f32_tensor_weight(*storage, model.token_embedding, "token_embedding", device_id);
    upload_f32_tensor_weight(*storage, model.final_norm, "final_norm", device_id);
    upload_linear_weight(*storage, model.lm_head, "lm_head", device_id);
    for (size_t layer = 0; layer < model.layers.size(); ++layer) {
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        upload_f32_tensor_weight(*storage, lw.attention_norm, prefix + "attention_norm", device_id);
        upload_linear_weight(*storage, lw.wq, prefix + "wq", device_id);
        upload_linear_weight(*storage, lw.wk, prefix + "wk", device_id);
        upload_linear_weight(*storage, lw.wv, prefix + "wv", device_id);
        upload_f32_tensor_weight(*storage, lw.bq, prefix + "bq", device_id);
        upload_f32_tensor_weight(*storage, lw.bk, prefix + "bk", device_id);
        upload_f32_tensor_weight(*storage, lw.bv, prefix + "bv", device_id);
        upload_linear_weight(*storage, lw.wo, prefix + "wo", device_id);
        upload_f32_tensor_weight(*storage, lw.ffn_norm, prefix + "ffn_norm", device_id);
        upload_linear_weight(*storage, lw.w_gate, prefix + "w_gate", device_id);
        upload_linear_weight(*storage, lw.w_up, prefix + "w_up", device_id);
        upload_linear_weight(*storage, lw.w_down, prefix + "w_down", device_id);
    }

    model.cuda_weights = std::move(storage);
}

void clear_model_cuda_weights(MiniLlamaModel& model) {
    model.cuda_weights.reset();
}

bool model_has_cuda_weights(const MiniLlamaModel& model) {
    return model.cuda_weights != nullptr;
}

size_t model_cuda_uploaded_weight_count(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->uploaded_weight_count;
}

static size_t model_cuda_uploaded_weight_count_by_type(const MiniLlamaModel& model, QuantType type) {
    if (!model.cuda_weights) {
        return 0;
    }
    size_t count = 0;
    for (const auto& weight : model.cuda_weights->weights) {
        if (weight.linear && weight.type == type) {
            count += 1;
        }
    }
    return count;
}

size_t model_cuda_uploaded_f32_weight_count(const MiniLlamaModel& model) {
    return model_cuda_uploaded_weight_count_by_type(model, QuantType::F32);
}

size_t model_cuda_uploaded_q8_0_weight_count(const MiniLlamaModel& model) {
    return model_cuda_uploaded_weight_count_by_type(model, QuantType::Q8_0);
}

size_t model_cuda_uploaded_q4_0_weight_count(const MiniLlamaModel& model) {
    return model_cuda_uploaded_weight_count_by_type(model, QuantType::Q4_0);
}

size_t model_cuda_uploaded_q4_1_weight_count(const MiniLlamaModel& model) {
    return model_cuda_uploaded_weight_count_by_type(model, QuantType::Q4_1);
}

size_t model_cuda_memory_bytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->uploaded_bytes;
}

void reset_model_cuda_runtime_stats(MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls = 0;
    model.cuda_weights->activation_calls = 0;
    model.cuda_weights->attention_calls = 0;
    model.cuda_weights->attention_cpu_fallbacks = 0;
    model.cuda_weights->kv_cache_write_bytes = 0;
    model.cuda_weights->kv_cache_read_bytes = 0;
    model.cuda_weights->host_to_device_copies = 0;
    model.cuda_weights->device_to_host_copies = 0;
    model.cuda_weights->host_to_device_bytes = 0;
    model.cuda_weights->device_to_host_bytes = 0;
}

size_t model_cuda_linear_calls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->linear_calls;
}

size_t model_cuda_activation_calls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->activation_calls;
}

size_t model_cuda_attention_calls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->attention_calls;
}

size_t model_cuda_attention_cpu_fallbacks(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->attention_cpu_fallbacks;
}

size_t model_cuda_kv_cache_write_bytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->kv_cache_write_bytes;
}

size_t model_cuda_kv_cache_read_bytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->kv_cache_read_bytes;
}

size_t model_cuda_host_to_device_copies(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->host_to_device_copies;
}

size_t model_cuda_device_to_host_copies(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->device_to_host_copies;
}

size_t model_cuda_host_to_device_bytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->host_to_device_bytes;
}

size_t model_cuda_device_to_host_bytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->device_to_host_bytes;
}

static Tensor quantized_tensor_to_f32(const QuantizedTensor& qt) {
    switch (qt.type) {
        case QuantType::F32:
            return to_tensor(qt);
        case QuantType::Q8_0:
            return dequantize_from_q8_0(qt.q8_0_data, qt.shape);
        case QuantType::Q4_0:
            return dequantize_from_q4_0(qt.q4_0_data, qt.shape);
        case QuantType::Q4_1:
            return dequantize_from_q4_1(qt.q4_1_data, qt.shape);
    }
    throw std::runtime_error("quantized_tensor_to_f32: unknown quant type");
}

static void quantize_qt_to_q8_0(QuantizedTensor& qt) {
    if (qt.type == QuantType::Q8_0) return;
    Tensor t = quantized_tensor_to_f32(qt);
    qt.q8_0_data = quantize_to_q8_0(t);
    qt.type = QuantType::Q8_0;
    qt.f32_data.clear();
    qt.f32_data.shrink_to_fit();
    qt.q4_0_data.clear();
    qt.q4_0_data.shrink_to_fit();
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
}

void quantize_model_to_q8_0(MiniLlamaModel& model) {
    quantize_qt_to_q8_0(model.lm_head);
    for (auto& lw : model.layers) {
        quantize_qt_to_q8_0(lw.wq);
        quantize_qt_to_q8_0(lw.wk);
        quantize_qt_to_q8_0(lw.wv);
        quantize_qt_to_q8_0(lw.wo);
        quantize_qt_to_q8_0(lw.w_gate);
        quantize_qt_to_q8_0(lw.w_up);
        quantize_qt_to_q8_0(lw.w_down);
    }
}

static void quantize_qt_to_q4_0(QuantizedTensor& qt) {
    if (qt.type == QuantType::Q4_0) return;
    Tensor t = quantized_tensor_to_f32(qt);
    qt.q4_0_data = quantize_to_q4_0(t);
    qt.type = QuantType::Q4_0;
    qt.f32_data.clear();
    qt.f32_data.shrink_to_fit();
    qt.q8_0_data.clear();
    qt.q8_0_data.shrink_to_fit();
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
}

void quantize_model_to_q4_0(MiniLlamaModel& model) {
    quantize_qt_to_q4_0(model.lm_head);
    for (auto& lw : model.layers) {
        quantize_qt_to_q4_0(lw.wq);
        quantize_qt_to_q4_0(lw.wk);
        quantize_qt_to_q4_0(lw.wv);
        quantize_qt_to_q4_0(lw.wo);
        quantize_qt_to_q4_0(lw.w_gate);
        quantize_qt_to_q4_0(lw.w_up);
        quantize_qt_to_q4_0(lw.w_down);
    }
}

} // namespace mini_llama
