#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mini_llama;

namespace {

constexpr float kAbsTol = 1e-6f;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(float actual, float expected, const std::string& label) {
    if (std::fabs(actual - expected) > kAbsTol) {
        throw std::runtime_error(
            label + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected)
        );
    }
}

void require_close_tensor(const Tensor& actual, const Tensor& expected, const std::string& label) {
    require(actual.shape == expected.shape, label + ": shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        require_close(actual.data[i], expected.data[i], label + " at " + std::to_string(i));
    }
}

void require_throws_with(const std::function<void()>& fn, const std::string& needle, const std::string& label) {
    try {
        fn();
    } catch (const std::exception& e) {
        require(
            std::string(e.what()).find(needle) != std::string::npos,
            label + ": unexpected error message: " + e.what()
        );
        return;
    }
    throw std::runtime_error(label + ": expected exception");
}

Tensor make_pattern_tensor(const std::vector<int>& shape, float base) {
    Tensor t(shape, 0.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data[i] = base + static_cast<float>(i) * 0.25f;
    }
    return t;
}

Tensor read_cpu_slot(const KVCache& cache, bool key, int layer, int pos) {
    int n_kv_heads = cache.keys.shape[2];
    int head_dim = cache.keys.shape[3];
    Tensor out({n_kv_heads, head_dim}, 0.0f);
    for (int h = 0; h < n_kv_heads; ++h) {
        const float* src = key ? cache.key_ptr(layer, pos, h) : cache.value_ptr(layer, pos, h);
        for (int d = 0; d < head_dim; ++d) {
            out.data[static_cast<size_t>(h * head_dim + d)] = src[d];
        }
    }
    return out;
}

void require_cuda_not_built() {
    require(!cuda_kv_cache_built(), "cuda_kv_cache_built should be false in CPU build");
    require_throws_with(
        []() {
            CudaKVCache cache(1, 2, 1, 2);
            (void)cache;
        },
        "CUDA KV cache was not built",
        "CPU build constructor"
    );
    require_throws_with(
        []() {
            CudaKVCache cache;
            cache.clear();
        },
        "CUDA KV cache was not built",
        "CPU build clear"
    );
}

void test_construct_write_read_clear() {
    CudaKVCache cache(2, 4, 2, 3);
    require(!cache.empty(), "cache should allocate device buffers");
    require(cache.bytes() == 2 * 2 * 4 * 2 * 3 * sizeof(float), "cache byte count mismatch");
    require(cache.n_layers() == 2, "n_layers mismatch");
    require(cache.max_seq_len() == 4, "max_seq_len mismatch");
    require(cache.n_kv_heads() == 2, "n_kv_heads mismatch");
    require(cache.head_dim() == 3, "head_dim mismatch");

    Tensor k = make_pattern_tensor({2, 3}, 1.0f);
    Tensor v = make_pattern_tensor({2, 3}, 10.0f);
    cache.write(1, 2, k, v);

    require_close_tensor(cache.read_key(1, 2), k, "read_key");
    require_close_tensor(cache.read_value(1, 2), v, "read_value");

    Tensor expected_key_head_0({3}, 0.0f);
    Tensor expected_key_head_1({3}, 0.0f);
    Tensor expected_value_head_0({3}, 0.0f);
    Tensor expected_value_head_1({3}, 0.0f);
    for (int d = 0; d < 3; ++d) {
        expected_key_head_0.data[d] = k.data[d];
        expected_key_head_1.data[d] = k.data[3 + d];
        expected_value_head_0.data[d] = v.data[d];
        expected_value_head_1.data[d] = v.data[3 + d];
    }
    require_close_tensor(cache.read_key_head(1, 2, 0), expected_key_head_0, "read_key_head 0");
    require_close_tensor(cache.read_key_head(1, 2, 1), expected_key_head_1, "read_key_head 1");
    require_close_tensor(cache.read_value_head(1, 2, 0), expected_value_head_0, "read_value_head 0");
    require_close_tensor(cache.read_value_head(1, 2, 1), expected_value_head_1, "read_value_head 1");

    cache.clear();
    Tensor zeros({2, 3}, 0.0f);
    require_close_tensor(cache.read_key(1, 2), zeros, "clear key");
    require_close_tensor(cache.read_value(1, 2), zeros, "clear value");
}

void test_rejects_bad_inputs() {
    CudaKVCache cache(1, 2, 1, 2);
    Tensor good({1, 2}, 0.0f);
    Tensor rank_3({1, 1, 2}, 0.0f);
    Tensor wrong_heads({2, 2}, 0.0f);
    Tensor wrong_dim({1, 3}, 0.0f);

    require_throws_with([&]() { cache.write(-1, 0, good, good); }, "layer out of range", "layer underflow");
    require_throws_with([&]() { cache.write(0, 2, good, good); }, "position out of range", "position overflow");
    require_throws_with([&]() { (void)cache.read_key_head(0, 0, 1); }, "head out of range", "head overflow");
    require_throws_with([&]() { cache.write(0, 0, rank_3, good); }, "expected matching", "rank mismatch");
    require_throws_with([&]() { cache.write(0, 0, wrong_heads, wrong_heads); }, "does not match", "head mismatch");
    require_throws_with([&]() { cache.write(0, 0, wrong_dim, wrong_dim); }, "does not match", "dim mismatch");
}

void test_cpu_gpu_cache_alignment() {
    KVCache cpu_cache(2, 4, 2, 3);
    CudaKVCache cuda_cache(2, 4, 2, 3);

    Tensor k = make_pattern_tensor({2, 3}, -2.0f);
    Tensor v = make_pattern_tensor({2, 3}, 5.0f);
    cpu_cache.write(1, 3, k, v);
    cuda_cache.write(1, 3, k, v);

    require_close_tensor(cuda_cache.read_key(1, 3), read_cpu_slot(cpu_cache, true, 1, 3), "CPU/GPU key slot");
    require_close_tensor(cuda_cache.read_value(1, 3), read_cpu_slot(cpu_cache, false, 1, 3), "CPU/GPU value slot");
}

void test_forward_syncs_cuda_kv_cache() {
    MiniLlamaModel cpu_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    MiniLlamaModel cuda_model = load_model("models/tiny/model.json", "models/tiny/model.bin");
    require(cpu_model.loaded, "CPU tiny model should load");
    require(cuda_model.loaded, "CUDA tiny model should load");

    MiniLlamaContext cpu_ctx(&cpu_model);
    MiniLlamaContext cuda_ctx(&cuda_model);
    require(cuda_ctx.cuda_kv_cache.empty(), "context should start without CUDA cache before upload");

    MiniBatch batch = MiniBatch::from_tokens({1, 2}, 0);
    (void)forward_batch(cpu_model, cpu_ctx, batch);

    upload_model_weights_to_cuda(cuda_model, 0);
    (void)forward_batch(cuda_model, cuda_ctx, batch);

    require(!cuda_ctx.cuda_kv_cache.empty(), "forward should allocate CUDA KV cache lazily");
    require(cuda_ctx.cuda_kv_cache.bytes() > 0, "CUDA KV cache should report bytes");

    require_close_tensor(cuda_ctx.cuda_kv_cache.read_key(0, 0), read_cpu_slot(cpu_ctx.kv_cache, true, 0, 0), "forward key layer0 pos0");
    require_close_tensor(cuda_ctx.cuda_kv_cache.read_value(0, 0), read_cpu_slot(cpu_ctx.kv_cache, false, 0, 0), "forward value layer0 pos0");
    require_close_tensor(cuda_ctx.cuda_kv_cache.read_key(1, 1), read_cpu_slot(cpu_ctx.kv_cache, true, 1, 1), "forward key layer1 pos1");
    require_close_tensor(cuda_ctx.cuda_kv_cache.read_value(1, 1), read_cpu_slot(cpu_ctx.kv_cache, false, 1, 1), "forward value layer1 pos1");
}

void test_cuda_kv_cache_cases() {
    require(cuda_kv_cache_built(), "cuda_kv_cache_built should be true in CUDA build");
    require(cuda_device_count() > 0, "CUDA build should see at least one device");
    test_construct_write_read_clear();
    test_rejects_bad_inputs();
    test_cpu_gpu_cache_alignment();
    test_forward_syncs_cuda_kv_cache();
}

} // namespace

int main() {
    try {
#ifdef MINI_LLAMA_USE_CUDA
        test_cuda_kv_cache_cases();
        std::cout << "PASS cuda_kv_cache\n";
#else
        require_cuda_not_built();
        std::cout << "PASS cuda_kv_cache_cpu_build\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL cuda_kv_cache: " << e.what() << "\n";
        return 1;
    }
}
