#pragma once

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class MatmulMode {
    Naive,
    Threaded,
    Simd,
    ThreadedSimd,
};

// Default mode used by the main inference path.
MatmulMode default_matmul_mode();

// F32 matrix multiplication [M,K] x [K,N] -> [M,N]
Tensor matmul_dispatch(const Tensor& A, const Tensor& B, MatmulMode mode);

// F32 linear: x @ W^T
// Supports x: [in_features] or [1, in_features]
// W: [out_features, in_features]
// result rank matches input rank.
Tensor linear_dispatch(const Tensor& x, const Tensor& W, MatmulMode mode);

} // namespace mini_llama
