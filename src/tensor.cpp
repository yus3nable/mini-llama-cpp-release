#include "mini_llama/tensor.h"
#include <numeric>
#include <iostream>
#include <iomanip>
#include <limits>

namespace mini_llama {

namespace {

const char* caller_name(const char* caller) {
    return caller == nullptr ? "Tensor" : caller;
}

std::string shape_to_string(const std::vector<int>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

size_t checked_numel(const std::vector<int>& shape, const char* caller) {
    size_t total = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        const int dim = shape[axis];
        if (dim <= 0) {
            throw std::runtime_error(
                std::string(caller_name(caller)) + ": dimension at axis " +
                std::to_string(axis) + " must be positive, got " +
                std::to_string(dim) + " in shape " + shape_to_string(shape)
            );
        }
        const size_t dim_size = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error(
                std::string(caller_name(caller)) + ": shape element count overflow for " +
                shape_to_string(shape)
            );
        }
        total *= dim_size;
    }
    return total;
}

void check_rank(const Tensor& t, int expected, const char* caller) {
    if (t.ndim() != expected) {
        throw std::runtime_error(
            std::string(caller) + " expected " + std::to_string(expected) +
            "D tensor, got " + t.shape_str()
        );
    }
}

void check_axis_index(const Tensor& t, int axis, int index, const char* caller) {
    const int dim = t.shape[axis];
    if (index < 0 || index >= dim) {
        throw std::out_of_range(
            std::string(caller) + ": index " + std::to_string(index) +
            " out of range for axis " + std::to_string(axis) +
            " with size " + std::to_string(dim) +
            " in tensor " + t.shape_str()
        );
    }
}

} // namespace

Tensor::Tensor(const std::vector<int>& shape_, float fill) : shape(shape_) {
    const size_t total = checked_numel(shape, "Tensor constructor");
    data.resize(total, fill);
}

size_t Tensor::index(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::runtime_error(
            "Tensor::index expected " + std::to_string(shape.size()) +
            " indices for tensor " + shape_str() + ", got " +
            std::to_string(indices.size())
        );
    }
    for (size_t axis = 0; axis < indices.size(); ++axis) {
        check_axis_index(*this, static_cast<int>(axis), indices[axis], "Tensor::index");
    }

    size_t flat = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        flat += static_cast<size_t>(indices[i]) * stride;
        stride *= static_cast<size_t>(shape[i]);
    }
    return flat;
}

float& Tensor::at(const std::vector<int>& indices) {
    return data[index(indices)];
}

float Tensor::at(const std::vector<int>& indices) const {
    return data[index(indices)];
}

// ---------------------------------------------------------------------------
// Convenience 1D accessor
// ---------------------------------------------------------------------------
float& Tensor::at1(int i) {
    check_rank(*this, 1, "at1");
    check_axis_index(*this, 0, i, "at1");
    return data[i];
}

float Tensor::at1(int i) const {
    check_rank(*this, 1, "at1");
    check_axis_index(*this, 0, i, "at1");
    return data[i];
}

// ---------------------------------------------------------------------------
// Convenience 2D accessor
// ---------------------------------------------------------------------------
float& Tensor::at2(int i, int j) {
    check_rank(*this, 2, "at2");
    check_axis_index(*this, 0, i, "at2");
    check_axis_index(*this, 1, j, "at2");
    return data[i * shape[1] + j];
}

float Tensor::at2(int i, int j) const {
    check_rank(*this, 2, "at2");
    check_axis_index(*this, 0, i, "at2");
    check_axis_index(*this, 1, j, "at2");
    return data[i * shape[1] + j];
}

// ---------------------------------------------------------------------------
// Convenience 3D accessor
// ---------------------------------------------------------------------------
float& Tensor::at3(int i, int j, int k) {
    check_rank(*this, 3, "at3");
    check_axis_index(*this, 0, i, "at3");
    check_axis_index(*this, 1, j, "at3");
    check_axis_index(*this, 2, k, "at3");
    return data[(i * shape[1] + j) * shape[2] + k];
}

float Tensor::at3(int i, int j, int k) const {
    check_rank(*this, 3, "at3");
    check_axis_index(*this, 0, i, "at3");
    check_axis_index(*this, 1, j, "at3");
    check_axis_index(*this, 2, k, "at3");
    return data[(i * shape[1] + j) * shape[2] + k];
}

// ---------------------------------------------------------------------------
// Convenience 4D accessor
// ---------------------------------------------------------------------------
float& Tensor::at4(int i, int j, int k, int l) {
    check_rank(*this, 4, "at4");
    check_axis_index(*this, 0, i, "at4");
    check_axis_index(*this, 1, j, "at4");
    check_axis_index(*this, 2, k, "at4");
    check_axis_index(*this, 3, l, "at4");
    return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

float Tensor::at4(int i, int j, int k, int l) const {
    check_rank(*this, 4, "at4");
    check_axis_index(*this, 0, i, "at4");
    check_axis_index(*this, 1, j, "at4");
    check_axis_index(*this, 2, k, "at4");
    check_axis_index(*this, 3, l, "at4");
    return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

// ---------------------------------------------------------------------------
// Row pointer for 2D tensors
// ---------------------------------------------------------------------------
float* Tensor::row_ptr(int row) {
    check_rank(*this, 2, "row_ptr");
    check_axis_index(*this, 0, row, "row_ptr");
    return data.data() + row * shape[1];
}

const float* Tensor::row_ptr(int row) const {
    check_rank(*this, 2, "row_ptr");
    check_axis_index(*this, 0, row, "row_ptr");
    return data.data() + row * shape[1];
}

// ---------------------------------------------------------------------------
// Shape assertion
// ---------------------------------------------------------------------------
void Tensor::assert_shape(const std::vector<int>& expected, const char* caller) const {
    if (shape != expected) {
        throw std::runtime_error(
            std::string(caller_name(caller)) + ": shape mismatch. expected " +
            shape_to_string(expected) + ", got " + shape_str()
        );
    }
}

// ---------------------------------------------------------------------------
// Reshape with element count validation
// ---------------------------------------------------------------------------
Tensor Tensor::reshape_checked(const std::vector<int>& new_shape, const char* caller) const {
    const size_t new_total = checked_numel(new_shape, caller_name(caller));
    if (new_total != data.size()) {
        throw std::runtime_error(
            std::string(caller_name(caller)) + ": cannot reshape tensor with " +
            std::to_string(data.size()) + " elements into shape with " +
            std::to_string(new_total) + " elements"
        );
    }
    Tensor r = *this;
    r.shape = new_shape;
    return r;
}

std::string Tensor::shape_str() const {
    return shape_to_string(shape);
}

void Tensor::print(const std::string& name, bool print_data) const {
    if (!name.empty()) {
        std::cout << name << " ";
    }
    std::cout << "shape=" << shape_str() << " size=" << size() << std::endl;
    if (print_data) {
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << data[i];
            if (i + 1 < data.size()) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}

Tensor make_tensor_1d(int d0, float fill) {
    return Tensor({d0}, fill);
}

Tensor make_tensor_2d(int d0, int d1, float fill) {
    return Tensor({d0, d1}, fill);
}

Tensor make_tensor_3d(int d0, int d1, int d2, float fill) {
    return Tensor({d0, d1, d2}, fill);
}

Tensor make_tensor_4d(int d0, int d1, int d2, int d3, float fill) {
    return Tensor({d0, d1, d2, d3}, fill);
}

} // namespace mini_llama
