#include "mini_llama/cuda_quant.h"

#include "mini_llama/cuda_runtime.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mini_llama {

namespace {

constexpr int kBlockSize = 128;

void check_cuda_quant(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "CUDA quant error in " + std::string(expr) + ": " + cudaGetErrorString(err)
        );
    }
}

void check_last_kernel(const char* name) {
    check_cuda_quant(cudaGetLastError(), name);
}

__device__ float half_bits_to_float(uint16_t bits) {
    __half_raw raw;
    raw.x = bits;
    return __half2float(raw);
}

__global__ void q8_0_linear_kernel(
    const float* x,
    const BlockQ8_0* weight,
    float* y,
    int rows,
    int in_features,
    int out_features,
    int blocks_per_row
) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (out >= out_features || row >= rows) {
        return;
    }

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ8_0* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const BlockQ8_0& block = w_row[block_index];
        float d = half_bits_to_float(block.d);
        int base_k = block_index * Q8_0_BLOCK_SIZE;
        int k_end = base_k + Q8_0_BLOCK_SIZE < in_features ? base_k + Q8_0_BLOCK_SIZE : in_features;
        for (int k = base_k; k < k_end; ++k) {
            sum += d * static_cast<float>(block.qs[k - base_k]) * x_row[k];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}

__device__ float q4_0_value(const BlockQ4_0& block, int index) {
    int q = index < 16
        ? static_cast<int>(block.qs[index] & 0x0F)
        : static_cast<int>(block.qs[index - 16] >> 4);
    return half_bits_to_float(block.d) * static_cast<float>(q - 8);
}

__device__ float q4_1_value(const BlockQ4_1& block, int index) {
    int q = index < 16
        ? static_cast<int>(block.qs[index] & 0x0F)
        : static_cast<int>(block.qs[index - 16] >> 4);
    return half_bits_to_float(block.d) * static_cast<float>(q) + half_bits_to_float(block.m);
}

__global__ void q4_0_linear_kernel(
    const float* x,
    const BlockQ4_0* weight,
    float* y,
    int rows,
    int in_features,
    int out_features,
    int blocks_per_row
) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (out >= out_features || row >= rows) {
        return;
    }

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ4_0* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const BlockQ4_0& block = w_row[block_index];
        int base_k = block_index * Q4_0_BLOCK_SIZE;
        int k_end = base_k + Q4_0_BLOCK_SIZE < in_features ? base_k + Q4_0_BLOCK_SIZE : in_features;
        for (int k = base_k; k < k_end; ++k) {
            sum += q4_0_value(block, k - base_k) * x_row[k];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}

__global__ void q4_1_linear_kernel(
    const float* x,
    const BlockQ4_1* weight,
    float* y,
    int rows,
    int in_features,
    int out_features,
    int blocks_per_row
) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (out >= out_features || row >= rows) {
        return;
    }

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ4_1* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const BlockQ4_1& block = w_row[block_index];
        int base_k = block_index * Q4_1_BLOCK_SIZE;
        int k_end = base_k + Q4_1_BLOCK_SIZE < in_features ? base_k + Q4_1_BLOCK_SIZE : in_features;
        for (int k = base_k; k < k_end; ++k) {
            sum += q4_1_value(block, k - base_k) * x_row[k];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}

void add_bias_in_place(Tensor& y, const Tensor& bias, int rows, int cols, const char* caller) {
    if (bias.ndim() != 1 || bias.shape[0] != cols) {
        throw std::runtime_error(std::string(caller) + ": expected bias shape [out_features]");
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
        }
    }
}

int q8_0_in_features_from_shape(
    const std::vector<int>& x_shape,
    const std::string& x_shape_str
) {
    if (x_shape.size() == 1) {
        return x_shape[0];
    }
    if (x_shape.size() == 2) {
        return x_shape[1];
    }
    throw std::runtime_error("cuda_q8_0_linear: expected x shape [in_features] or [batch, in_features], got " + x_shape_str);
}

void validate_q8_0_linear_shape(
    const std::vector<int>& x_shape,
    const std::string& x_shape_str,
    const std::vector<int>& weight_shape,
    size_t block_count
) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error("cuda_q8_0_linear: expected W shape [out_features, in_features]");
    }
    int in_features = q8_0_in_features_from_shape(x_shape, x_shape_str);
    if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
        throw std::runtime_error("cuda_q8_0_linear: empty tensors are not supported");
    }
    if (weight_shape[1] != in_features) {
        throw std::runtime_error(
            "cuda_q8_0_linear: dimension mismatch x=" + x_shape_str +
            " W=[" + std::to_string(weight_shape[0]) + ", " + std::to_string(weight_shape[1]) + "]"
        );
    }
    int blocks_per_row = (in_features + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
    if (block_count != expected_blocks) {
        throw std::runtime_error(
            "cuda_q8_0_linear: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(block_count)
        );
    }
}

void validate_q8_0_linear_inputs(
    const Tensor& x,
    const std::vector<int>& weight_shape,
    size_t block_count
) {
    validate_q8_0_linear_shape(x.shape, x.shape_str(), weight_shape, block_count);
}

int quant_in_features_from_shape(
    const std::vector<int>& x_shape,
    const std::string& x_shape_str,
    const char* caller
) {
    if (x_shape.size() == 1) {
        return x_shape[0];
    }
    if (x_shape.size() == 2) {
        return x_shape[1];
    }
    throw std::runtime_error(std::string(caller) + ": expected x shape [in_features] or [batch, in_features], got " + x_shape_str);
}

void validate_quant_linear_shape(
    const std::vector<int>& x_shape,
    const std::string& x_shape_str,
    const std::vector<int>& weight_shape,
    size_t block_count,
    int block_size,
    const char* caller
) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error(std::string(caller) + ": expected W shape [out_features, in_features]");
    }
    int in_features = quant_in_features_from_shape(x_shape, x_shape_str, caller);
    if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
        throw std::runtime_error(std::string(caller) + ": empty tensors are not supported");
    }
    if (weight_shape[1] != in_features) {
        throw std::runtime_error(
            std::string(caller) + ": dimension mismatch x=" + x_shape_str +
            " W=[" + std::to_string(weight_shape[0]) + ", " + std::to_string(weight_shape[1]) + "]"
        );
    }
    int blocks_per_row = (in_features + block_size - 1) / block_size;
    size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
    if (block_count != expected_blocks) {
        throw std::runtime_error(
            std::string(caller) + ": block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " + std::to_string(block_count)
        );
    }
}

Tensor cuda_q8_0_linear_with_device_pointer(
    const Tensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    if (q8_0_device == nullptr) {
        throw std::runtime_error("cuda_q8_0_linear: device weight pointer is null");
    }
    validate_q8_0_linear_inputs(x, weight_shape, q8_0_block_count);

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = weight_shape[0];
    int blocks_per_row = (in_features + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.size() * sizeof(float));

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    q8_0_linear_kernel<<<grid, block>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<const BlockQ8_0*>(q8_0_device),
        static_cast<float*>(y_dev.data()),
        rows,
        in_features,
        out_features,
        blocks_per_row
    );
    check_last_kernel("q8_0_linear_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    if (bias != nullptr) {
        add_bias_in_place(y, *bias, rows, out_features, "cuda_q8_0_linear");
    }
    return y;
}

CudaTensor cuda_q8_0_linear_with_device_input(
    const CudaTensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    if (q8_0_device == nullptr) {
        throw std::runtime_error("cuda_q8_0_linear: device weight pointer is null");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error("cuda_q8_0_linear: input tensor is on a different CUDA device");
    }
    validate_q8_0_linear_shape(x.shape(), x.shape_str(), weight_shape, q8_0_block_count);

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = weight_shape[0];
    int blocks_per_row = (in_features + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

    CudaTensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    q8_0_linear_kernel<<<grid, block>>>(
        static_cast<const float*>(x.data()),
        static_cast<const BlockQ8_0*>(q8_0_device),
        static_cast<float*>(y.data()),
        rows,
        in_features,
        out_features,
        blocks_per_row
    );
    check_last_kernel("q8_0_linear_kernel");

    return y;
}

Tensor cuda_q4_0_linear_with_device_pointer(
    const Tensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    static constexpr const char* kCaller = "cuda_q4_0_linear";
    if (q4_0_device == nullptr) {
        throw std::runtime_error(std::string(kCaller) + ": device weight pointer is null");
    }
    validate_quant_linear_shape(x.shape, x.shape_str(), weight_shape, q4_0_block_count, Q4_0_BLOCK_SIZE, kCaller);

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = weight_shape[0];
    int blocks_per_row = (in_features + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;

    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.size() * sizeof(float));

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    q4_0_linear_kernel<<<grid, block>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<const BlockQ4_0*>(q4_0_device),
        static_cast<float*>(y_dev.data()),
        rows,
        in_features,
        out_features,
        blocks_per_row
    );
    check_last_kernel("q4_0_linear_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    if (bias != nullptr) {
        add_bias_in_place(y, *bias, rows, out_features, kCaller);
    }
    return y;
}

CudaTensor cuda_q4_0_linear_with_device_input(
    const CudaTensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    static constexpr const char* kCaller = "cuda_q4_0_linear";
    if (q4_0_device == nullptr) {
        throw std::runtime_error(std::string(kCaller) + ": device weight pointer is null");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error(std::string(kCaller) + ": input tensor is on a different CUDA device");
    }
    validate_quant_linear_shape(x.shape(), x.shape_str(), weight_shape, q4_0_block_count, Q4_0_BLOCK_SIZE, kCaller);

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = weight_shape[0];
    int blocks_per_row = (in_features + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;

    CudaTensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    q4_0_linear_kernel<<<grid, block>>>(
        static_cast<const float*>(x.data()),
        static_cast<const BlockQ4_0*>(q4_0_device),
        static_cast<float*>(y.data()),
        rows,
        in_features,
        out_features,
        blocks_per_row
    );
    check_last_kernel("q4_0_linear_kernel");

    return y;
}

Tensor cuda_q4_1_linear_with_device_pointer(
    const Tensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    static constexpr const char* kCaller = "cuda_q4_1_linear";
    if (q4_1_device == nullptr) {
        throw std::runtime_error(std::string(kCaller) + ": device weight pointer is null");
    }
    validate_quant_linear_shape(x.shape, x.shape_str(), weight_shape, q4_1_block_count, Q4_1_BLOCK_SIZE, kCaller);

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = weight_shape[0];
    int blocks_per_row = (in_features + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE;

    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.size() * sizeof(float));

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    q4_1_linear_kernel<<<grid, block>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<const BlockQ4_1*>(q4_1_device),
        static_cast<float*>(y_dev.data()),
        rows,
        in_features,
        out_features,
        blocks_per_row
    );
    check_last_kernel("q4_1_linear_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    if (bias != nullptr) {
        add_bias_in_place(y, *bias, rows, out_features, kCaller);
    }
    return y;
}

CudaTensor cuda_q4_1_linear_with_device_input(
    const CudaTensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    static constexpr const char* kCaller = "cuda_q4_1_linear";
    if (q4_1_device == nullptr) {
        throw std::runtime_error(std::string(kCaller) + ": device weight pointer is null");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error(std::string(kCaller) + ": input tensor is on a different CUDA device");
    }
    validate_quant_linear_shape(x.shape(), x.shape_str(), weight_shape, q4_1_block_count, Q4_1_BLOCK_SIZE, kCaller);

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = weight_shape[0];
    int blocks_per_row = (in_features + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE;

    CudaTensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    q4_1_linear_kernel<<<grid, block>>>(
        static_cast<const float*>(x.data()),
        static_cast<const BlockQ4_1*>(q4_1_device),
        static_cast<float*>(y.data()),
        rows,
        in_features,
        out_features,
        blocks_per_row
    );
    check_last_kernel("q4_1_linear_kernel");

    return y;
}

} // namespace

Tensor cuda_q8_0_linear(
    const Tensor& x,
    const std::vector<BlockQ8_0>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    validate_q8_0_linear_inputs(x, weight_shape, weight.size());
    CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ8_0), device_id);
    w_dev.upload(weight.data(), weight.size() * sizeof(BlockQ8_0));
    return cuda_q8_0_linear_with_device_pointer(
        x,
        w_dev.data(),
        weight.size(),
        weight_shape,
        bias,
        device_id
    );
}

Tensor cuda_q8_0_linear_device_weight(
    const Tensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    return cuda_q8_0_linear_with_device_pointer(
        x,
        q8_0_device,
        q8_0_block_count,
        weight_shape,
        bias,
        device_id
    );
}

CudaTensor cuda_q8_0_linear_device_input(
    const CudaTensor& x,
    const void* q8_0_device,
    size_t q8_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    return cuda_q8_0_linear_with_device_input(
        x,
        q8_0_device,
        q8_0_block_count,
        weight_shape,
        device_id
    );
}

Tensor cuda_q4_0_linear(
    const Tensor& x,
    const std::vector<BlockQ4_0>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    validate_quant_linear_shape(x.shape, x.shape_str(), weight_shape, weight.size(), Q4_0_BLOCK_SIZE, "cuda_q4_0_linear");
    CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ4_0), device_id);
    w_dev.upload(weight.data(), weight.size() * sizeof(BlockQ4_0));
    return cuda_q4_0_linear_with_device_pointer(
        x,
        w_dev.data(),
        weight.size(),
        weight_shape,
        bias,
        device_id
    );
}

Tensor cuda_q4_0_linear_device_weight(
    const Tensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    return cuda_q4_0_linear_with_device_pointer(
        x,
        q4_0_device,
        q4_0_block_count,
        weight_shape,
        bias,
        device_id
    );
}

CudaTensor cuda_q4_0_linear_device_input(
    const CudaTensor& x,
    const void* q4_0_device,
    size_t q4_0_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    return cuda_q4_0_linear_with_device_input(
        x,
        q4_0_device,
        q4_0_block_count,
        weight_shape,
        device_id
    );
}

Tensor cuda_q4_1_linear(
    const Tensor& x,
    const std::vector<BlockQ4_1>& weight,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    validate_quant_linear_shape(x.shape, x.shape_str(), weight_shape, weight.size(), Q4_1_BLOCK_SIZE, "cuda_q4_1_linear");
    CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ4_1), device_id);
    w_dev.upload(weight.data(), weight.size() * sizeof(BlockQ4_1));
    return cuda_q4_1_linear_with_device_pointer(
        x,
        w_dev.data(),
        weight.size(),
        weight_shape,
        bias,
        device_id
    );
}

Tensor cuda_q4_1_linear_device_weight(
    const Tensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    const Tensor* bias,
    int device_id
) {
    return cuda_q4_1_linear_with_device_pointer(
        x,
        q4_1_device,
        q4_1_block_count,
        weight_shape,
        bias,
        device_id
    );
}

CudaTensor cuda_q4_1_linear_device_input(
    const CudaTensor& x,
    const void* q4_1_device,
    size_t q4_1_block_count,
    const std::vector<int>& weight_shape,
    int device_id
) {
    return cuda_q4_1_linear_with_device_input(
        x,
        q4_1_device,
        q4_1_block_count,
        weight_shape,
        device_id
    );
}

} // namespace mini_llama
