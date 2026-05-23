#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <stdexcept>

namespace mini_llama {

// Minimal dense tensor for F32 data
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;
    Tensor(const std::vector<int>& shape_, float fill = 0.0f);

    // Total number of elements
    size_t size() const { return data.size(); }
    size_t numel() const { return data.size(); }

    // Number of dimensions
    int ndim() const { return static_cast<int>(shape.size()); }

    // Compute flat index from multi-dimensional indices
    size_t index(const std::vector<int>& indices) const;

    // Get element at multi-dimensional indices
    float& at(const std::vector<int>& indices);
    float at(const std::vector<int>& indices) const;

    // Convenience 1D/2D/3D/4D accessors
    float& at1(int i);
    float at1(int i) const;
    float& at2(int i, int j);
    float at2(int i, int j) const;
    float& at3(int i, int j, int k);
    float at3(int i, int j, int k) const;
    float& at4(int i, int j, int k, int l);
    float at4(int i, int j, int k, int l) const;

    // Get a pointer to a row in a 2D tensor
    // t: [rows, cols], row_ptr(r) -> &data[r * cols]
    float* row_ptr(int row);
    const float* row_ptr(int row) const;

    // Assert shape matches expected; throws on mismatch
    void assert_shape(const std::vector<int>& expected, const char* caller) const;

    // Reshape after verifying element count stays unchanged
    Tensor reshape_checked(const std::vector<int>& new_shape, const char* caller) const;

    // Helper for 1D access
    float& operator[](size_t i) { return data[i]; }
    float operator[](size_t i) const { return data[i]; }

    // Print shape and optionally data
    std::string shape_str() const;
    std::string shape_string() const { return shape_str(); }
    void print(const std::string& name = "", bool print_data = false) const;
};

// Create a 1D tensor
Tensor make_tensor_1d(int d0, float fill = 0.0f);

// Create a 2D tensor
Tensor make_tensor_2d(int d0, int d1, float fill = 0.0f);

// Create a 3D tensor
Tensor make_tensor_3d(int d0, int d1, int d2, float fill = 0.0f);

// Create a 4D tensor
Tensor make_tensor_4d(int d0, int d1, int d2, int d3, float fill = 0.0f);

} // namespace mini_llama
