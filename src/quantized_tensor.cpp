#include "mini_llama/quantized_tensor.h"
#include <stdexcept>

namespace mini_llama {

Tensor to_tensor(const QuantizedTensor& q) {
    if (q.type != QuantType::F32) {
        throw std::runtime_error("to_tensor: expected F32 QuantizedTensor, got quantized type");
    }
    Tensor t(q.shape, 0.0f);
    if (q.f32_data.size() == t.data.size()) {
        t.data = q.f32_data;
    }
    return t;
}

QuantizedTensor to_quantized_tensor(const Tensor& t) {
    QuantizedTensor q;
    q.type = QuantType::F32;
    q.shape = t.shape;
    q.f32_data = t.data;
    return q;
}

} // namespace mini_llama
