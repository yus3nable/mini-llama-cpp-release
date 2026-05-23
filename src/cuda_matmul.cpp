#include "mini_llama/cuda_matmul.h"

#include "mini_llama/cuda_runtime.h"

#include <stdexcept>
#include <string>

#ifdef MINI_LLAMA_USE_CUDA
#include <cublas_v2.h>
#endif

namespace mini_llama {

namespace {

std::runtime_error cuda_matmul_not_built_error() {
    return std::runtime_error(
        "CUDA matmul was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine."
    );
}

#ifdef MINI_LLAMA_USE_CUDA
const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

void check_cublas(cublasStatus_t status, const char* expr) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error in " + std::string(expr) + ": " + cublas_status_name(status)
        );
    }
}

class CublasHandle {
public:
    explicit CublasHandle(int device_id) {
        cuda_set_device(device_id);
        check_cublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle_;
    }

private:
    cublasHandle_t handle_ = nullptr;
};

void validate_matmul_shapes(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("cuda_matmul: expected A and B to be 2D tensors");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::runtime_error(
            "cuda_matmul: dimension mismatch " + A.shape_str() + " vs " + B.shape_str()
        );
    }
}

void add_bias_in_place(Tensor& y, const Tensor& bias, int rows, int cols) {
    if (bias.ndim() != 1 || bias.shape[0] != cols) {
        throw std::runtime_error("cuda_linear: expected bias shape [out_features]");
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y.data[row * cols + col] += bias.data[col];
        }
    }
}

int linear_in_features_from_shape(const std::vector<int>& x_shape, const std::string& x_shape_str, const char* caller) {
    if (x_shape.size() == 1) {
        return x_shape[0];
    }
    if (x_shape.size() == 2) {
        return x_shape[1];
    }
    throw std::runtime_error(std::string(caller) + ": expected x shape [in_features] or [batch, in_features], got " + x_shape_str);
}

void validate_linear_shape(
    const std::vector<int>& x_shape,
    const std::string& x_shape_str,
    const std::vector<int>& w_shape,
    const char* caller
) {
    if (w_shape.size() != 2) {
        throw std::runtime_error(std::string(caller) + ": expected W shape [out_features, in_features]");
    }
    int in_features = linear_in_features_from_shape(x_shape, x_shape_str, caller);
    if (w_shape[1] != in_features) {
        throw std::runtime_error(
            std::string(caller) + ": dimension mismatch x=" + x_shape_str +
            " W=[" + std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + "]"
        );
    }
}

void validate_linear_inputs(const Tensor& x, const std::vector<int>& w_shape) {
    validate_linear_shape(x.shape, x.shape_str(), w_shape, "cuda_linear");
}

Tensor cuda_linear_with_device_pointer(
    const Tensor& x,
    const void* w_device,
    const std::vector<int>& w_shape,
    const Tensor* bias,
    int device_id
) {
    if (w_device == nullptr) {
        throw std::runtime_error("cuda_linear: device weight pointer is null");
    }
    validate_linear_inputs(x, w_shape);

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = w_shape[0];

    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.size() * sizeof(float));

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // y = x[rows,K] * W[out,K]^T. In column-major terms, the output buffer is
    // y^T[out,rows] = W[out,K] * x^T[K,rows].
    check_cublas(
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_features,
            rows,
            in_features,
            &alpha,
            static_cast<const float*>(w_device),
            in_features,
            static_cast<const float*>(x_dev.data()),
            in_features,
            &beta,
            static_cast<float*>(y_dev.data()),
            out_features
        ),
        "cublasSgemm(cuda_linear)"
    );

    y_dev.download(y.data.data(), y.size() * sizeof(float));
    if (bias != nullptr) {
        add_bias_in_place(y, *bias, rows, out_features);
    }
    return y;
}

CudaTensor cuda_linear_device_input_with_device_pointer(
    const CudaTensor& x,
    const void* w_device,
    const std::vector<int>& w_shape,
    int device_id
) {
    if (w_device == nullptr) {
        throw std::runtime_error("cuda_linear_device_input: device weight pointer is null");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error("cuda_linear_device_input: input tensor is on a different CUDA device");
    }
    validate_linear_shape(x.shape(), x.shape_str(), w_shape, "cuda_linear_device_input");

    bool input_is_1d = x.ndim() == 1;
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = w_shape[0];

    CudaTensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_cublas(
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_features,
            rows,
            in_features,
            &alpha,
            static_cast<const float*>(w_device),
            in_features,
            static_cast<const float*>(x.data()),
            in_features,
            &beta,
            static_cast<float*>(y.data()),
            out_features
        ),
        "cublasSgemm(cuda_linear_device_input)"
    );

    return y;
}
#endif

} // namespace

bool cuda_matmul_built() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

Tensor cuda_matmul(const Tensor& A, const Tensor& B, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    validate_matmul_shapes(A, B);

    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    Tensor C({M, N}, 0.0f);

    CudaDeviceBuffer a_dev(A.size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(B.size() * sizeof(float), device_id);
    CudaDeviceBuffer c_dev(C.size() * sizeof(float), device_id);
    a_dev.upload(A.data.data(), A.size() * sizeof(float));
    b_dev.upload(B.data.data(), B.size() * sizeof(float));

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C = A[M,K] * B[K,N] is equivalent to column-major
    // C^T[N,M] = B^T[N,K] * A^T[K,M]. The row-major output buffer stores C
    // with the same byte layout as column-major C^T.
    check_cublas(
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            static_cast<const float*>(b_dev.data()),
            N,
            static_cast<const float*>(a_dev.data()),
            K,
            &beta,
            static_cast<float*>(c_dev.data()),
            N
        ),
        "cublasSgemm(cuda_matmul)"
    );

    c_dev.download(C.data.data(), C.size() * sizeof(float));
    return C;
#else
    (void)A;
    (void)B;
    (void)device_id;
    throw cuda_matmul_not_built_error();
#endif
}

Tensor cuda_linear(const Tensor& x, const Tensor& W, const Tensor* bias, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (W.ndim() != 2) {
        throw std::runtime_error("cuda_linear: expected W shape [out_features, in_features]");
    }
    CudaDeviceBuffer w_dev(W.size() * sizeof(float), device_id);
    w_dev.upload(W.data.data(), W.size() * sizeof(float));
    return cuda_linear_with_device_pointer(x, w_dev.data(), W.shape, bias, device_id);
#else
    (void)x;
    (void)W;
    (void)bias;
    (void)device_id;
    throw cuda_matmul_not_built_error();
#endif
}

Tensor cuda_linear_device_weight(
    const Tensor& x,
    const void* w_device,
    const std::vector<int>& w_shape,
    const Tensor* bias,
    int device_id
) {
#ifdef MINI_LLAMA_USE_CUDA
    return cuda_linear_with_device_pointer(x, w_device, w_shape, bias, device_id);
#else
    (void)x;
    (void)w_device;
    (void)w_shape;
    (void)bias;
    (void)device_id;
    throw cuda_matmul_not_built_error();
#endif
}

CudaTensor cuda_linear_device_input(
    const CudaTensor& x,
    const void* w_device,
    const std::vector<int>& w_shape,
    int device_id
) {
#ifdef MINI_LLAMA_USE_CUDA
    return cuda_linear_device_input_with_device_pointer(x, w_device, w_shape, device_id);
#else
    (void)x;
    (void)w_device;
    (void)w_shape;
    (void)device_id;
    throw cuda_matmul_not_built_error();
#endif
}

} // namespace mini_llama
