#include "mini_llama/cuda_ops.h"

#include "mini_llama/cuda_runtime.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace mini_llama {

namespace {

constexpr int kBlockSize = 256;

void check_cuda_ops(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "CUDA ops error in " + std::string(expr) + ": " + cudaGetErrorString(err)
        );
    }
}

void check_last_kernel(const char* name) {
    check_cuda_ops(cudaGetLastError(), name);
}

__global__ void sum_squares_kernel(const float* x, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, x[i] * x[i]);
    }
}

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y, const float* sum, int n, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float scale = rsqrtf((*sum / static_cast<float>(n)) + eps);
        y[i] = x[i] * scale * weight[i];
    }
}

__global__ void silu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

__global__ void elementwise_mul_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] * b[i];
    }
}

__global__ void elementwise_add_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}

__global__ void embedding_lookup_kernel(const float* embedding, float* y, int token_id, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        y[i] = embedding[token_id * dim + i];
    }
}

__global__ void softmax_max_kernel(const float* x, float* max_out, int n) {
    __shared__ float shared[kBlockSize];
    int tid = threadIdx.x;
    float local_max = -3.402823466e+38F;
    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        *max_out = shared[0];
    }
}

__global__ void softmax_exp_sum_kernel(const float* x, float* y, const float* max_value, float* sum_out, int n) {
    __shared__ float shared[kBlockSize];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float max_v = *max_value;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(x[i] - max_v);
        y[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *sum_out = shared[0];
    }
}

__global__ void softmax_norm_kernel(float* y, const float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] /= *sum;
    }
}

__global__ void rope_normal_kernel(float* x, int n_heads, int head_dim, int pos, float theta) {
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs_per_head = head_dim / 2;
    int total_pairs = n_heads * pairs_per_head;
    if (pair_index >= total_pairs) {
        return;
    }

    int head = pair_index / pairs_per_head;
    int pair = pair_index % pairs_per_head;
    int dim = pair * 2;
    int base = head * head_dim + dim;
    float freq = 1.0f / powf(theta, static_cast<float>(dim) / static_cast<float>(head_dim));
    float cos_val = cosf(static_cast<float>(pos) * freq);
    float sin_val = sinf(static_cast<float>(pos) * freq);
    float x0 = x[base];
    float x1 = x[base + 1];
    x[base] = x0 * cos_val - x1 * sin_val;
    x[base + 1] = x0 * sin_val + x1 * cos_val;
}

__global__ void rope_neox_kernel(float* x, int n_heads, int head_dim, int pos, float theta) {
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = n_heads * half_dim;
    if (pair_index >= total_pairs) {
        return;
    }

    int head = pair_index / half_dim;
    int pair = pair_index % half_dim;
    int base = head * head_dim;
    float freq = 1.0f / powf(theta, static_cast<float>(2 * pair) / static_cast<float>(head_dim));
    float cos_val = cosf(static_cast<float>(pos) * freq);
    float sin_val = sinf(static_cast<float>(pos) * freq);
    float x0 = x[base + pair];
    float x1 = x[base + half_dim + pair];
    x[base + pair] = x0 * cos_val - x1 * sin_val;
    x[base + half_dim + pair] = x0 * sin_val + x1 * cos_val;
}

int grid_for(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* caller) {
    if (a.shape != b.shape) {
        throw std::runtime_error(
            std::string(caller) + ": shape mismatch " + a.shape_str() + " vs " + b.shape_str()
        );
    }
}

void require_same_shape(const CudaTensor& a, const CudaTensor& b, const char* caller) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(
            std::string(caller) + ": shape mismatch " + a.shape_str() + " vs " + b.shape_str()
        );
    }
    if (a.device_id() != b.device_id()) {
        throw std::runtime_error(std::string(caller) + ": tensors are on different CUDA devices");
    }
}

void require_1d(const Tensor& x, const char* caller) {
    if (x.ndim() != 1) {
        throw std::runtime_error(std::string(caller) + ": expected 1D tensor, got " + x.shape_str());
    }
    if (x.size() == 0) {
        throw std::runtime_error(std::string(caller) + ": empty tensor");
    }
}

void require_1d(const CudaTensor& x, const char* caller) {
    if (x.ndim() != 1) {
        throw std::runtime_error(std::string(caller) + ": expected 1D tensor, got " + x.shape_str());
    }
    if (x.size() == 0) {
        throw std::runtime_error(std::string(caller) + ": empty tensor");
    }
}

void validate_rope_inputs(const Tensor& q, const Tensor& k, int pos, float theta) {
    if (q.ndim() != 2 || k.ndim() != 2) {
        throw std::runtime_error("cuda_rope: expected 2D tensors");
    }
    if (pos < 0) {
        throw std::out_of_range("cuda_rope: position must be non-negative");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("cuda_rope: theta must be finite and positive");
    }
    if (q.shape[1] != k.shape[1]) {
        throw std::runtime_error(
            "cuda_rope: q and k head_dim mismatch " + q.shape_str() + " vs " + k.shape_str()
        );
    }
    if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
        throw std::runtime_error("cuda_rope: head_dim must be positive and even");
    }
}

void validate_rope_device_inputs(
    const CudaTensor& q,
    const CudaTensor& k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float theta,
    int device_id
) {
    if (q.device_id() != device_id || k.device_id() != device_id) {
        throw std::runtime_error("cuda_rope_device_input: input tensor is on a different CUDA device");
    }
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error("cuda_rope_device_input: invalid head shape");
    }
    if (q.size() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error("cuda_rope_device_input: q shape mismatch");
    }
    if (k.size() != static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error("cuda_rope_device_input: k shape mismatch");
    }
    if (pos < 0) {
        throw std::out_of_range("cuda_rope_device_input: position must be non-negative");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("cuda_rope_device_input: theta must be finite and positive");
    }
}

void validate_embedding_device_weight_inputs(
    const void* embedding_data,
    const std::vector<int>& embedding_shape,
    int token_id
) {
    if (embedding_data == nullptr) {
        throw std::runtime_error("cuda_embedding_lookup_device_weight: embedding data is null");
    }
    if (embedding_shape.size() != 2) {
        throw std::runtime_error("cuda_embedding_lookup_device_weight: expected 2D embedding weight");
    }
    int vocab_size = embedding_shape[0];
    int dim = embedding_shape[1];
    if (vocab_size <= 0 || dim <= 0) {
        throw std::runtime_error("cuda_embedding_lookup_device_weight: embedding shape must be positive");
    }
    if (token_id < 0 || token_id >= vocab_size) {
        throw std::out_of_range("cuda_embedding_lookup_device_weight: token id out of range");
    }
}

void validate_rmsnorm_device_weight_inputs(
    const CudaTensor& x,
    const void* weight_data,
    const std::vector<int>& weight_shape,
    float eps,
    int device_id
) {
    require_1d(x, "cuda_rmsnorm_device_weight");
    if (weight_data == nullptr) {
        throw std::runtime_error("cuda_rmsnorm_device_weight: weight data is null");
    }
    if (x.shape() != weight_shape) {
        throw std::runtime_error("cuda_rmsnorm_device_weight: x and weight shape mismatch");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error("cuda_rmsnorm_device_weight: input tensor is on a different CUDA device");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("cuda_rmsnorm_device_weight: eps must be finite and positive");
    }
}

void validate_add_device_weight_inputs(
    const CudaTensor& a,
    const void* b_data,
    const std::vector<int>& b_shape,
    int device_id
) {
    if (a.device_id() != device_id) {
        throw std::runtime_error("cuda_elementwise_add_device_weight: input tensor is on a different CUDA device");
    }
    if (b_data == nullptr) {
        throw std::runtime_error("cuda_elementwise_add_device_weight: weight data is null");
    }
    if (a.shape() != b_shape) {
        throw std::runtime_error("cuda_elementwise_add_device_weight: input and weight shape mismatch");
    }
}

} // namespace

Tensor cuda_rmsnorm(const Tensor& x, const Tensor& weight, float eps, int device_id) {
    require_1d(x, "cuda_rmsnorm");
    require_1d(weight, "cuda_rmsnorm");
    if (x.shape != weight.shape) {
        throw std::runtime_error("cuda_rmsnorm: x and weight shape mismatch");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("cuda_rmsnorm: eps must be finite and positive");
    }

    cuda_set_device(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.size() * sizeof(float));
    w_dev.upload(weight.data.data(), weight.size() * sizeof(float));
    check_cuda_ops(cudaMemset(sum_dev.data(), 0, sizeof(float)), "cudaMemset(rmsnorm sum)");

    int n = static_cast<int>(x.size());
    sum_squares_kernel<<<grid_for(n), kBlockSize>>>(static_cast<const float*>(x_dev.data()), static_cast<float*>(sum_dev.data()), n);
    check_last_kernel("sum_squares_kernel");
    rmsnorm_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<const float*>(w_dev.data()),
        static_cast<float*>(y_dev.data()),
        static_cast<const float*>(sum_dev.data()),
        n,
        eps
    );
    check_last_kernel("rmsnorm_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    return y;
}

Tensor cuda_silu(const Tensor& x, int device_id) {
    cuda_set_device(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.size() * sizeof(float));

    int n = static_cast<int>(x.size());
    silu_kernel<<<grid_for(n), kBlockSize>>>(static_cast<const float*>(x_dev.data()), static_cast<float*>(y_dev.data()), n);
    check_last_kernel("silu_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    return y;
}

Tensor cuda_elementwise_mul(const Tensor& a, const Tensor& b, int device_id) {
    require_same_shape(a, b, "cuda_elementwise_mul");
    cuda_set_device(device_id);
    Tensor y(a.shape, 0.0f);
    CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    a_dev.upload(a.data.data(), a.size() * sizeof(float));
    b_dev.upload(b.data.data(), b.size() * sizeof(float));

    int n = static_cast<int>(a.size());
    elementwise_mul_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(a_dev.data()),
        static_cast<const float*>(b_dev.data()),
        static_cast<float*>(y_dev.data()),
        n
    );
    check_last_kernel("elementwise_mul_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    return y;
}

CudaTensor cuda_embedding_lookup_device_weight(
    const void* embedding_data,
    const std::vector<int>& embedding_shape,
    int token_id,
    int device_id
) {
    validate_embedding_device_weight_inputs(embedding_data, embedding_shape, token_id);
    cuda_set_device(device_id);

    int dim = embedding_shape[1];
    CudaTensor y({dim}, device_id);
    embedding_lookup_kernel<<<grid_for(dim), kBlockSize>>>(
        static_cast<const float*>(embedding_data),
        static_cast<float*>(y.data()),
        token_id,
        dim
    );
    check_last_kernel("embedding_lookup_kernel");
    return y;
}

CudaTensor cuda_rmsnorm_device_input(const CudaTensor& x, const Tensor& weight, float eps, int device_id) {
    require_1d(x, "cuda_rmsnorm_device_input");
    require_1d(weight, "cuda_rmsnorm_device_input");
    if (x.shape() != weight.shape) {
        throw std::runtime_error("cuda_rmsnorm_device_input: x and weight shape mismatch");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error("cuda_rmsnorm_device_input: input tensor is on a different CUDA device");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("cuda_rmsnorm_device_input: eps must be finite and positive");
    }

    cuda_set_device(device_id);
    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
    w_dev.upload(weight.data.data(), weight.size() * sizeof(float));
    return cuda_rmsnorm_device_weight(x, w_dev.data(), weight.shape, eps, device_id);
}

CudaTensor cuda_rmsnorm_device_weight(
    const CudaTensor& x,
    const void* weight_data,
    const std::vector<int>& weight_shape,
    float eps,
    int device_id
) {
    validate_rmsnorm_device_weight_inputs(x, weight_data, weight_shape, eps, device_id);
    cuda_set_device(device_id);

    CudaTensor y(x.shape(), device_id);
    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    check_cuda_ops(cudaMemset(sum_dev.data(), 0, sizeof(float)), "cudaMemset(rmsnorm sum)");

    int n = static_cast<int>(x.size());
    sum_squares_kernel<<<grid_for(n), kBlockSize>>>(static_cast<const float*>(x.data()), static_cast<float*>(sum_dev.data()), n);
    check_last_kernel("sum_squares_kernel");
    rmsnorm_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(x.data()),
        static_cast<const float*>(weight_data),
        static_cast<float*>(y.data()),
        static_cast<const float*>(sum_dev.data()),
        n,
        eps
    );
    check_last_kernel("rmsnorm_kernel");

    return y;
}

CudaTensor cuda_silu_device_input(const CudaTensor& x, int device_id) {
    if (x.device_id() != device_id) {
        throw std::runtime_error("cuda_silu_device_input: input tensor is on a different CUDA device");
    }
    cuda_set_device(device_id);
    CudaTensor y(x.shape(), device_id);

    int n = static_cast<int>(x.size());
    silu_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(x.data()),
        static_cast<float*>(y.data()),
        n
    );
    check_last_kernel("silu_kernel");

    return y;
}

CudaTensor cuda_elementwise_mul_device_input(const CudaTensor& a, const CudaTensor& b, int device_id) {
    require_same_shape(a, b, "cuda_elementwise_mul_device_input");
    if (a.device_id() != device_id) {
        throw std::runtime_error("cuda_elementwise_mul_device_input: input tensor is on a different CUDA device");
    }
    cuda_set_device(device_id);
    CudaTensor y(a.shape(), device_id);

    int n = static_cast<int>(a.size());
    elementwise_mul_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(y.data()),
        n
    );
    check_last_kernel("elementwise_mul_kernel");

    return y;
}

CudaTensor cuda_elementwise_add_device_input(const CudaTensor& a, const CudaTensor& b, int device_id) {
    require_same_shape(a, b, "cuda_elementwise_add_device_input");
    if (a.device_id() != device_id) {
        throw std::runtime_error("cuda_elementwise_add_device_input: input tensor is on a different CUDA device");
    }
    cuda_set_device(device_id);
    CudaTensor y(a.shape(), device_id);

    int n = static_cast<int>(a.size());
    elementwise_add_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(y.data()),
        n
    );
    check_last_kernel("elementwise_add_kernel");

    return y;
}

CudaTensor cuda_elementwise_add_device_weight(
    const CudaTensor& a,
    const void* b_data,
    const std::vector<int>& b_shape,
    int device_id
) {
    validate_add_device_weight_inputs(a, b_data, b_shape, device_id);
    cuda_set_device(device_id);
    CudaTensor y(a.shape(), device_id);

    int n = static_cast<int>(a.size());
    elementwise_add_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b_data),
        static_cast<float*>(y.data()),
        n
    );
    check_last_kernel("elementwise_add_kernel");

    return y;
}

Tensor cuda_elementwise_add(const Tensor& a, const Tensor& b, int device_id) {
    require_same_shape(a, b, "cuda_elementwise_add");
    cuda_set_device(device_id);
    Tensor y(a.shape, 0.0f);
    CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    a_dev.upload(a.data.data(), a.size() * sizeof(float));
    b_dev.upload(b.data.data(), b.size() * sizeof(float));

    int n = static_cast<int>(a.size());
    elementwise_add_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<const float*>(a_dev.data()),
        static_cast<const float*>(b_dev.data()),
        static_cast<float*>(y_dev.data()),
        n
    );
    check_last_kernel("elementwise_add_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    return y;
}

void cuda_rope_device_input(
    CudaTensor& q,
    CudaTensor& k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float theta,
    RopeType rope_type,
    int device_id
) {
    validate_rope_device_inputs(q, k, n_heads, n_kv_heads, head_dim, pos, theta, device_id);
    cuda_set_device(device_id);

    int q_pairs = n_heads * (head_dim / 2);
    int k_pairs = n_kv_heads * (head_dim / 2);
    if (rope_type == RopeType::NeoX) {
        rope_neox_kernel<<<grid_for(q_pairs), kBlockSize>>>(
            static_cast<float*>(q.data()),
            n_heads,
            head_dim,
            pos,
            theta
        );
        check_last_kernel("rope_neox_kernel(q)");
        rope_neox_kernel<<<grid_for(k_pairs), kBlockSize>>>(
            static_cast<float*>(k.data()),
            n_kv_heads,
            head_dim,
            pos,
            theta
        );
        check_last_kernel("rope_neox_kernel(k)");
    } else {
        rope_normal_kernel<<<grid_for(q_pairs), kBlockSize>>>(
            static_cast<float*>(q.data()),
            n_heads,
            head_dim,
            pos,
            theta
        );
        check_last_kernel("rope_normal_kernel(q)");
        rope_normal_kernel<<<grid_for(k_pairs), kBlockSize>>>(
            static_cast<float*>(k.data()),
            n_kv_heads,
            head_dim,
            pos,
            theta
        );
        check_last_kernel("rope_normal_kernel(k)");
    }
}

Tensor cuda_softmax(const Tensor& x, int device_id) {
    require_1d(x, "cuda_softmax");
    cuda_set_device(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    CudaDeviceBuffer max_dev(sizeof(float), device_id);
    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.size() * sizeof(float));

    int n = static_cast<int>(x.size());
    softmax_max_kernel<<<1, kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<float*>(max_dev.data()),
        n
    );
    check_last_kernel("softmax_max_kernel");
    softmax_exp_sum_kernel<<<1, kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<float*>(y_dev.data()),
        static_cast<const float*>(max_dev.data()),
        static_cast<float*>(sum_dev.data()),
        n
    );
    check_last_kernel("softmax_exp_sum_kernel");
    softmax_norm_kernel<<<grid_for(n), kBlockSize>>>(
        static_cast<float*>(y_dev.data()),
        static_cast<const float*>(sum_dev.data()),
        n
    );
    check_last_kernel("softmax_norm_kernel");

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    return y;
}

void cuda_rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type, int device_id) {
    validate_rope_inputs(q, k, pos, theta);
    cuda_set_device(device_id);
    CudaDeviceBuffer q_dev(q.size() * sizeof(float), device_id);
    CudaDeviceBuffer k_dev(k.size() * sizeof(float), device_id);
    q_dev.upload(q.data.data(), q.size() * sizeof(float));
    k_dev.upload(k.data.data(), k.size() * sizeof(float));

    int q_pairs = q.shape[0] * (q.shape[1] / 2);
    int k_pairs = k.shape[0] * (k.shape[1] / 2);
    if (rope_type == RopeType::NeoX) {
        rope_neox_kernel<<<grid_for(q_pairs), kBlockSize>>>(
            static_cast<float*>(q_dev.data()),
            q.shape[0],
            q.shape[1],
            pos,
            theta
        );
        check_last_kernel("rope_neox_kernel(q)");
        rope_neox_kernel<<<grid_for(k_pairs), kBlockSize>>>(
            static_cast<float*>(k_dev.data()),
            k.shape[0],
            k.shape[1],
            pos,
            theta
        );
        check_last_kernel("rope_neox_kernel(k)");
    } else {
        rope_normal_kernel<<<grid_for(q_pairs), kBlockSize>>>(
            static_cast<float*>(q_dev.data()),
            q.shape[0],
            q.shape[1],
            pos,
            theta
        );
        check_last_kernel("rope_normal_kernel(q)");
        rope_normal_kernel<<<grid_for(k_pairs), kBlockSize>>>(
            static_cast<float*>(k_dev.data()),
            k.shape[0],
            k.shape[1],
            pos,
            theta
        );
        check_last_kernel("rope_normal_kernel(k)");
    }

    q_dev.download(q.data.data(), q.size() * sizeof(float));
    k_dev.download(k.data.data(), k.size() * sizeof(float));
}

} // namespace mini_llama
