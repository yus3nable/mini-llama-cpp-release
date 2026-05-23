#include <iostream>
#include <string>
#include <vector>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <limits>
#include <functional>
#include <algorithm>

#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/sampler.h"
#include "mini_llama/loader.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/chat.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/terminal.h"
#include "mini_llama/debug.h"
#include "mini_llama/gguf.h"
#include "mini_llama/quant.h"
#include "mini_llama/thread_pool.h"
#include "mini_llama/ops.h"
#include "mini_llama/backend.h"
#include "mini_llama/cuda_runtime.h"

using namespace mini_llama;

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------
static bool parse_int_arg(const char* text, int& value) {
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (parsed < 0 || parsed > 1000000) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_float_arg(const char* text, float& value) {
    char* end = nullptr;
    errno = 0;
    double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    if (errno == ERANGE || !std::isfinite(parsed) || parsed < 0.0 || parsed > 10000.0) {
        return false;
    }
    value = static_cast<float>(parsed);
    return true;
}

static bool parse_uint_arg(const char* text, unsigned int& value) {
    if (text[0] == '-' || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max()) {
        return false;
    }
    value = static_cast<unsigned int>(parsed);
    return true;
}

static bool parse_backend_arg(const char* text, BackendConfig& config) {
    BackendKind kind;
    if (!parse_backend_kind(text, kind)) {
        return false;
    }
    config.kind = kind;
    return true;
}

static bool parse_device_arg(const char* text, BackendConfig& config) {
    int device_id = 0;
    if (!parse_int_arg(text, device_id) || device_id < 0) {
        return false;
    }
    config.device_id = device_id;
    config.device_id_set = true;
    return true;
}

static bool validate_backend_or_print(const BackendConfig& config) {
    try {
        validate_backend(config);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Backend setup failed: " << e.what() << "\n";
        return false;
    }
}

static void print_backend_info(const BackendConfig& config) {
    std::cout << "backend: " << backend_kind_name(config.kind) << "\n";
    if (config.kind == BackendKind::Cuda) {
        std::cout << "cuda: " << cuda_device_summary(config.device_id) << "\n";
    }
    std::cout << backend_execution_note(config) << "\n";
}

static std::string format_mb(size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    return out.str();
}

static bool prepare_cuda_weights_or_print(MiniLlamaModel& model, const BackendConfig& config) {
    if (config.kind != BackendKind::Cuda) {
        return true;
    }
    try {
        upload_model_weights_to_cuda(model, config.device_id);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CUDA weight upload failed: " << e.what() << "\n";
        return false;
    }
}

static void print_cuda_weight_summary(const MiniLlamaModel& model, const std::string& indent = "") {
    std::cout << indent << "uploaded weights: " << model_cuda_uploaded_weight_count(model) << "\n";
    std::cout << indent << "gpu memory used: " << model_cuda_memory_bytes(model)
              << " bytes (" << format_mb(model_cuda_memory_bytes(model)) << ")\n";
}

struct LinearWeightTypeCounts {
    size_t f32 = 0;
    size_t q8_0 = 0;
    size_t q4_0 = 0;
    size_t q4_1 = 0;

    size_t total() const {
        return f32 + q8_0 + q4_0 + q4_1;
    }

    size_t quantized() const {
        return q8_0 + q4_0 + q4_1;
    }
};

static void count_linear_weight(const QuantizedTensor& weight, LinearWeightTypeCounts& counts) {
    switch (weight.type) {
        case QuantType::F32:
            counts.f32 += 1;
            return;
        case QuantType::Q8_0:
            counts.q8_0 += 1;
            return;
        case QuantType::Q4_0:
            counts.q4_0 += 1;
            return;
        case QuantType::Q4_1:
            counts.q4_1 += 1;
            return;
    }
}

static LinearWeightTypeCounts model_linear_weight_type_counts(const MiniLlamaModel& model) {
    LinearWeightTypeCounts counts;
    count_linear_weight(model.lm_head, counts);
    for (const auto& lw : model.layers) {
        count_linear_weight(lw.wq, counts);
        count_linear_weight(lw.wk, counts);
        count_linear_weight(lw.wv, counts);
        count_linear_weight(lw.wo, counts);
        count_linear_weight(lw.w_gate, counts);
        count_linear_weight(lw.w_up, counts);
        count_linear_weight(lw.w_down, counts);
    }
    return counts;
}

static void print_cuda_execution_summary(const MiniLlamaModel& model, const std::string& indent = "") {
    print_cuda_weight_summary(model, indent);
    LinearWeightTypeCounts counts = model_linear_weight_type_counts(model);
    size_t uploaded_f32 = model_cuda_uploaded_f32_weight_count(model);
    size_t uploaded_q8_0 = model_cuda_uploaded_q8_0_weight_count(model);
    size_t uploaded_q4_0 = model_cuda_uploaded_q4_0_weight_count(model);
    size_t uploaded_q4_1 = model_cuda_uploaded_q4_1_weight_count(model);
    std::cout << indent << "cuda f32 linear weights: " << uploaded_f32 << "/" << counts.total() << "\n";
    if (counts.q8_0 > 0) {
        std::cout << indent << "cuda q8_0 linear weights: " << uploaded_q8_0 << "/" << counts.q8_0 << "\n";
    }
    if (counts.q4_0 > 0) {
        std::cout << indent << "cuda q4_0 linear weights: " << uploaded_q4_0 << "/" << counts.q4_0 << "\n";
    }
    if (counts.q4_1 > 0) {
        std::cout << indent << "cuda q4_1 linear weights: " << uploaded_q4_1 << "/" << counts.q4_1 << "\n";
    }
    size_t q8_0_fallback = counts.q8_0 >= uploaded_q8_0 ? counts.q8_0 - uploaded_q8_0 : 0;
    size_t q4_0_fallback = counts.q4_0 >= uploaded_q4_0 ? counts.q4_0 - uploaded_q4_0 : 0;
    size_t q4_1_fallback = counts.q4_1 >= uploaded_q4_1 ? counts.q4_1 - uploaded_q4_1 : 0;
    if (q8_0_fallback + q4_0_fallback + q4_1_fallback > 0) {
        std::cout << indent << "cuda fallback: unsupported or missing quantized linear weights use CPU path"
                  << " (Q8_0=" << q8_0_fallback
                  << ", Q4_0=" << q4_0_fallback
                  << ", Q4_1=" << q4_1_fallback << ")\n";
    }
}

static void print_cuda_runtime_summary(const MiniLlamaModel& model, const std::string& indent = "") {
    std::cout << indent << "cuda linear calls: " << model_cuda_linear_calls(model) << "\n";
    std::cout << indent << "cuda activation calls: " << model_cuda_activation_calls(model) << "\n";
    std::cout << indent << "cuda attention calls: " << model_cuda_attention_calls(model) << "\n";
    std::cout << indent << "cpu attention fallback calls: " << model_cuda_attention_cpu_fallbacks(model) << "\n";
    std::cout << indent << "cuda kv write bytes: " << model_cuda_kv_cache_write_bytes(model) << "\n";
    std::cout << indent << "cuda kv read bytes: " << model_cuda_kv_cache_read_bytes(model) << "\n";
    std::cout << indent << "host->device copies: " << model_cuda_host_to_device_copies(model)
              << " (" << model_cuda_host_to_device_bytes(model) << " bytes)\n";
    std::cout << indent << "device->host copies: " << model_cuda_device_to_host_copies(model)
              << " (" << model_cuda_device_to_host_bytes(model) << " bytes)\n";
}

static void apply_quant_override(MiniLlamaModel& model, const std::string& quant_type) {
    if (quant_type.empty()) {
        return;
    }
    if (quant_type == "q8_0") {
        quantize_model_to_q8_0(model);
        return;
    }
    if (quant_type == "q4_0") {
        quantize_model_to_q4_0(model);
        return;
    }
    throw std::runtime_error("unsupported quant type: " + quant_type);
}

static Tensor run_logits_for_tokens(const MiniLlamaModel& model, const std::vector<int>& tokens) {
    MiniLlamaContext ctx(&model);
    MiniBatch batch = MiniBatch::from_tokens(tokens, 0);
    return forward_batch(model, ctx, batch);
}

static std::pair<float, float> logits_error(const Tensor& baseline, const Tensor& candidate) {
    if (baseline.shape != candidate.shape) {
        throw std::runtime_error(
            "logits_error: shape mismatch, baseline=" + baseline.shape_str() +
            ", candidate=" + candidate.shape_str()
        );
    }
    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (size_t i = 0; i < baseline.size(); ++i) {
        float err = std::abs(baseline.data[i] - candidate.data[i]);
        max_err = std::max(max_err, err);
        sum_err += err;
    }
    return {max_err, sum_err / static_cast<float>(baseline.size())};
}

// ---------------------------------------------------------------------------
// Generate mode
// ---------------------------------------------------------------------------
static void print_generate_usage(const char* prog) {
    std::cout << "Usage: " << prog << " generate [options]\n"
              << "Options:\n"
              << "  --model <path|dir>   Path to model weights binary or model directory (default: models/tiny/model.bin)\n"
              << "  --config <path>      Path to model config JSON (default: models/tiny/model.json)\n"
              << "  -p, --prompt <str>   Input prompt text (default: \"hello\")\n"
              << "  -n, --n-predict <N>  Number of tokens to generate (default: 16)\n"
              << "  --temperature <T>    Sampling temperature (default: 0.0 = greedy)\n"
              << "  --top-k <K>          Top-k sampling (default: 0 = disabled)\n"
              << "  --seed <S>           Random seed for reproducible sampling (default: 0 = random)\n"
              << "  --tokenizer <path>   Path to vocab.json tokenizer file\n"
              << "  --quant q8_0|q4_0    Quantize loaded linear weights before generation\n"
              << "  --threads <N>        Number of threads for parallel ops (0 = auto)\n"
              << "  --backend cpu|cuda   Execution backend (default: cpu; cuda requires -DMINI_LLAMA_CUDA=ON)\n"
              << "  --device <N>         CUDA device id for --backend cuda (default: 0)\n"
              << "  --dump-logits <dir>  Dump logits for each step to directory\n"
              << "  -h, --help           Show this help\n";
}

static void dump_logits(const Tensor& logits, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open logits dump file: " + path);
    }
    out.write(reinterpret_cast<const char*>(logits.data.data()),
              static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
    if (!out.good()) {
        throw std::runtime_error("failed to write logits to: " + path);
    }
}

static void ensure_dump_directory(const std::string& path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        throw std::runtime_error(
            "failed to create logits dump directory: " + path + ": " + error.message()
        );
    }
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error("logits dump path is not a directory: " + path);
    }
}

static void dump_generated_tokens(const std::vector<int>& generated, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open generation token dump file: " + path);
    }
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
            out << " ";
        }
        out << generated[i];
    }
    out << "\n";
    if (!out.good()) {
        throw std::runtime_error("failed to write generation tokens to: " + path);
    }
}

// ---------------------------------------------------------------------------
// Tokenizer path resolution
// ---------------------------------------------------------------------------

static std::string resolve_manifest_tokenizer_path(const std::string& config_path) {
    try {
        ModelManifest manifest = parse_manifest(config_path);
        if (manifest.tokenizer.type != "json_vocab" || manifest.tokenizer.path.empty()) {
            return "";
        }
        std::filesystem::path tokenizer_path(manifest.tokenizer.path);
        if (tokenizer_path.is_relative()) {
            std::filesystem::path config_file(config_path);
            tokenizer_path = config_file.parent_path() / tokenizer_path;
        }
        return tokenizer_path.string();
    } catch (const std::exception&) {
        return "";
    }
}

// ---------------------------------------------------------------------------
// Model loading helper: supports both JSON+BIN and GGUF
// ---------------------------------------------------------------------------

struct LoadedModel {
    MiniLlamaModel model;
    std::unique_ptr<ITokenizer> tokenizer;
    std::string chat_template;
};

static bool is_gguf_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    char magic[4];
    return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

static std::string find_gguf_in_directory(const std::string& path) {
    std::vector<std::filesystem::path> candidates;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            candidates.push_back(entry.path());
        }
    }
    if (candidates.empty()) {
        return "";
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.front().string();
}

static std::unique_ptr<ITokenizer> create_tokenizer_from_vocab_hint(const std::string& vocab_path) {
    std::filesystem::path vocab(vocab_path);
    std::filesystem::path dir = vocab.parent_path();
    std::filesystem::path merges = dir / "merges.txt";
    std::filesystem::path special = dir / "special_tokens.json";
    if (std::filesystem::exists(merges)) {
        return create_bpe_tokenizer(vocab.string(), merges.string(), special.string());
    }
    return create_tokenizer(vocab.string());
}

static LoadedModel load_model_and_tokenizer(
    const std::string& path,
    const std::string& explicit_config_path = "",
    const std::string& explicit_tokenizer_path = ""
) {
    LoadedModel result;

    std::string model_path = path;
    std::string config_path;
    std::string tokenizer_path;
    if (std::filesystem::is_directory(path)) {
        std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
        std::filesystem::path json_path = std::filesystem::path(path) / "model.json";
        std::string gguf_path = find_gguf_in_directory(path);
        if ((!std::filesystem::exists(bin_path) || !std::filesystem::exists(json_path)) && !gguf_path.empty()) {
            model_path = gguf_path;
        }
    }

    bool is_gguf = is_gguf_file(model_path);

    if (is_gguf) {
        result.model = load_gguf_model(model_path);
        if (!result.model.loaded) {
            return result;
        }

        // 1. Try to load tokenizer from GGUF metadata.
        if (!explicit_tokenizer_path.empty()) {
            result.tokenizer = create_tokenizer_from_vocab_hint(explicit_tokenizer_path);
        } else {
            result.tokenizer = create_gguf_tokenizer(model_path);
        }

        // 2. Fallback to external vocab.json + merges.txt if GGUF has no tokenizer metadata
        if (!result.tokenizer) {
            std::filesystem::path gguf_dir = std::filesystem::path(model_path).parent_path();
            std::string vocab_path = (gguf_dir / "vocab.json").string();
            std::string merges_path = (gguf_dir / "merges.txt").string();
            std::string special_path = (gguf_dir / "special_tokens.json").string();
            if (std::filesystem::exists(vocab_path) && std::filesystem::exists(merges_path)) {
                result.tokenizer = create_bpe_tokenizer(vocab_path, merges_path, special_path);
            }
        }

        // 3. Load chat template from GGUF metadata.
        result.chat_template = load_chat_template_from_gguf(model_path);
    } else {
        // Directory-based JSON+BIN format
        if (std::filesystem::is_directory(path)) {
            model_path = path + "/model.bin";
            config_path = path + "/model.json";
        } else {
            model_path = path;
            if (!explicit_config_path.empty()) {
                config_path = explicit_config_path;
            } else {
                config_path = std::filesystem::path(path).parent_path().string() + "/model.json";
            }
        }
        result.model = load_model(config_path, model_path);
        if (!result.model.loaded) {
            return result;
        }
        if (!explicit_tokenizer_path.empty()) {
            tokenizer_path = explicit_tokenizer_path;
        }
        if (tokenizer_path.empty()) {
            std::filesystem::path dir = std::filesystem::is_directory(path)
                ? std::filesystem::path(path)
                : std::filesystem::path(path).parent_path();
            std::filesystem::path auto_vocab = dir / "vocab.json";
            if (std::filesystem::exists(auto_vocab)) {
                tokenizer_path = auto_vocab.string();
            }
        }
        if (tokenizer_path.empty()) {
            tokenizer_path = resolve_manifest_tokenizer_path(config_path);
        }
        result.tokenizer = create_tokenizer(tokenizer_path);
    }

    if (!result.tokenizer) {
        result.model.load_error = "Failed to load tokenizer";
        result.model.loaded = false;
        return result;
    }

    return result;
}

static size_t common_prefix_length(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}


static int run_generate(int argc, char** argv) {
    std::string model_path = "models/tiny/model.bin";
    std::string config_path = "models/tiny/model.json";
    bool config_path_set = false;
    std::string prompt = "hello";
    int n_predict = 16;
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;
    std::string dump_logits_dir;
    std::string tokenizer_path;
    std::string quant_type;
    int n_threads = 0;
    BackendConfig backend_config;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
            config_path_set = true;
        } else if ((std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], n_predict)) {
                std::cerr << "Invalid --n-predict value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], temperature)) {
                std::cerr << "Invalid --temperature value. Expected a non-negative float.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], top_k)) {
                std::cerr << "Invalid --top-k value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_uint_arg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            dump_logits_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            quant_type = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], n_threads) || n_threads < 0) {
                std::cerr << "Invalid --threads value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            if (!parse_backend_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_device_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_generate_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            print_generate_usage(argv[0]);
            return 1;
        }
    }

    if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value: " << quant_type << ". Supported values: q8_0, q4_0.\n";
        return 1;
    }

    if (!validate_backend_or_print(backend_config)) {
        return 1;
    }

    std::cout << "mini-llama.cpp\n";
    std::cout << "==============\n\n";
    print_backend_info(backend_config);

    if (!dump_logits_dir.empty()) {
        try {
            ensure_dump_directory(dump_logits_dir);
        } catch (const std::exception& e) {
            std::cerr << "Logits dump setup failed: " << e.what() << "\n";
            return 1;
        }
    }

    // For backward compat, if --config was explicitly set but --model is the default,
    // override model_path to be the config's directory.
    if (config_path_set && model_path == "models/tiny/model.bin") {
        model_path = std::filesystem::path(config_path).parent_path().string() + "/model.bin";
    }

    LoadedModel lm = load_model_and_tokenizer(
        model_path,
        config_path_set ? config_path : "",
        tokenizer_path
    );
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        std::cerr << "Model vocab_size must be at least " << lm.tokenizer->vocab_size()
                  << " for the tokenizer.\n";
        return 1;
    }
    MiniLlamaModel& model = lm.model;
    try {
        apply_quant_override(model, quant_type);
    } catch (const std::exception& e) {
        std::cerr << "Quantization failed: " << e.what() << "\n";
        return 1;
    }
    if (!prepare_cuda_weights_or_print(model, backend_config)) {
        return 1;
    }
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
    std::vector<int> tokens = tokenizer->encode(prompt);
    std::cout << "prompt: " << prompt << "\n";
    std::cout << "tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << tokens[i];
    }
    std::cout << "]\n";
    set_thread_count(n_threads);
    std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k << ", seed=" << seed << "\n";
    std::cout << "quant: " << (quant_type.empty() ? "model-native" : quant_type) << "\n";
    std::cout << "threads: " << get_thread_count() << "\n\n";
    if (backend_config.kind == BackendKind::Cuda) {
        print_cuda_execution_summary(model);
        std::cout << "\n";
    }

    if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
        std::cerr << "Prompt is too long for max_seq_len=" << model.config.max_seq_len << ".\n";
        return 1;
    }
    if (tokens.size() + static_cast<size_t>(n_predict) > static_cast<size_t>(model.config.max_seq_len)) {
        std::cerr << "Requested tokens exceed context window.\n";
        return 1;
    }

    MiniLlamaContext ctx(&model);
    SamplingParams sampling_params;
    sampling_params.temperature = temperature;
    sampling_params.top_k = top_k;
    sampling_params.seed = seed;
    MiniSampler sampler(sampling_params);

    size_t prompt_len = tokens.size();
    int step = 0;
    try {
        Tensor logits;
        std::cout << "prefill...\n";
        if (!dump_logits_dir.empty()) {
            for (size_t i = 0; i < prompt_len; ++i) {
                MiniBatch prefill_step = MiniBatch::from_tokens({tokens[i]}, static_cast<int>(i));
                logits = forward_batch(model, ctx, prefill_step);
                ++ctx.n_prefill_tokens;
                dump_logits(logits, dump_logits_dir + "/logits_step" + std::to_string(step) + ".bin");
                ++step;
            }
        } else {
            MiniBatch prefill = MiniBatch::from_tokens(tokens, 0);
            logits = forward_batch(model, ctx, prefill);
            ctx.n_prefill_tokens += static_cast<int>(tokens.size());
        }

        if (n_predict > 0) {
            int next_token = sampler.sample(logits, sampling_params);
            tokens.push_back(next_token);

            std::cout << "decode loop...\n";
            for (int i = 1; i < n_predict; ++i) {
                MiniBatch decode = MiniBatch::single(tokens.back(), static_cast<int>(tokens.size() - 1));
                logits = forward_batch(model, ctx, decode);
                ++ctx.n_decode_tokens;
                if (!dump_logits_dir.empty()) {
                    dump_logits(logits, dump_logits_dir + "/logits_step" + std::to_string(step) + ".bin");
                    ++step;
                }
                next_token = sampler.sample(logits, sampling_params);
                tokens.push_back(next_token);
                if (next_token == tokenizer->eos_id()) {
                    break;
                }
            }
        } else {
            std::cout << "decode loop skipped.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return 1;
    }

    if (!dump_logits_dir.empty()) {
        std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
        try {
            dump_generated_tokens(generated, dump_logits_dir + "/generation_tokens.txt");
        } catch (const std::exception& e) {
            std::cerr << "Logits dump failed: " << e.what() << "\n";
            return 1;
        }
    }

    std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
    std::cout << "\ngenerated tokens: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << generated[i];
    }
    std::cout << "]\n";

    std::string generated_text = tokenizer->decode(generated);
    std::cout << "generated text: \"" << generated_text << "\"\n";
    if (backend_config.kind == BackendKind::Cuda) {
        std::cout << "\n";
        print_cuda_runtime_summary(model);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Run mode (interactive chat)
// ---------------------------------------------------------------------------
static void print_run_usage(const char* prog) {
    std::cout << "Usage: " << prog << " run <model-path|dir> [options]\n"
              << "Options:\n"
              << "  --temperature <T>  Sampling temperature (default: 0.0 = greedy)\n"
              << "  --top-k <K>        Top-k sampling (default: 0 = disabled)\n"
              << "  --seed <S>         Random seed (default: 0 = random)\n"
              << "  -n, --n-predict <N> Maximum response tokens per turn (default: 64)\n"
              << "  --tokenizer <path> Path to vocab.json tokenizer file\n"
              << "  --backend cpu|cuda Execution backend (default: cpu; cuda requires -DMINI_LLAMA_CUDA=ON)\n"
              << "  --device <N>       CUDA device id for --backend cuda (default: 0)\n"
              << "  -h, --help         Show this help\n";
}

static int run_chat(int argc, char** argv) {
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0)) {
        print_run_usage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing model directory.\n";
        print_run_usage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[2];
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;
    int max_response_tokens = 64;
    std::string tokenizer_path;
    BackendConfig backend_config;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parse_float_arg(argv[++i], temperature)) {
                std::cerr << "Invalid --temperature value. Expected a non-negative float.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], top_k)) {
                std::cerr << "Invalid --top-k value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_uint_arg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], max_response_tokens) || max_response_tokens <= 0) {
                std::cerr << "Invalid --n-predict value. Expected a positive integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            if (!parse_backend_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_device_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_run_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_run_usage(argv[0]);
            return 1;
        }
    }

    if (!validate_backend_or_print(backend_config)) {
        return 1;
    }

    // Load model
    LoadedModel lm = load_model_and_tokenizer(model_dir, "", tokenizer_path);
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        std::cerr << "Model vocab_size must be at least " << lm.tokenizer->vocab_size()
                  << " for the tokenizer.\n";
        return 1;
    }

    MiniLlamaModel& model = lm.model;
    if (!prepare_cuda_weights_or_print(model, backend_config)) {
        return 1;
    }
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

    PromptBuilder builder;
    if (!lm.chat_template.empty()) {
        builder.set_chat_template(lm.chat_template);
    }
    Terminal term;
    ChatSession session;
    session.sampling_params.temperature = temperature;
    session.sampling_params.top_k = top_k;
    session.sampling_params.seed = seed;

    // Plain text mode keeps the default system message in session state.
    if (lm.chat_template.empty()) {
        session.add_message("system", "You are a helpful assistant.");
    }

    term.print_message("mini-llama.cpp chat");
    term.print_message("backend: " + backend_kind_name(backend_config.kind));
    if (backend_config.kind == BackendKind::Cuda) {
        LinearWeightTypeCounts counts = model_linear_weight_type_counts(model);
        size_t uploaded_f32 = model_cuda_uploaded_f32_weight_count(model);
        size_t uploaded_q8_0 = model_cuda_uploaded_q8_0_weight_count(model);
        size_t uploaded_q4_0 = model_cuda_uploaded_q4_0_weight_count(model);
        size_t uploaded_q4_1 = model_cuda_uploaded_q4_1_weight_count(model);
        size_t q8_0_fallback = counts.q8_0 >= uploaded_q8_0 ? counts.q8_0 - uploaded_q8_0 : 0;
        size_t q4_0_fallback = counts.q4_0 >= uploaded_q4_0 ? counts.q4_0 - uploaded_q4_0 : 0;
        size_t q4_1_fallback = counts.q4_1 >= uploaded_q4_1 ? counts.q4_1 - uploaded_q4_1 : 0;
        term.print_message("cuda: " + cuda_device_summary(backend_config.device_id));
        term.print_message("uploaded weights: " + std::to_string(model_cuda_uploaded_weight_count(model)));
        term.print_message("gpu memory used: " + std::to_string(model_cuda_memory_bytes(model)) + " bytes (" + format_mb(model_cuda_memory_bytes(model)) + ")");
        term.print_message("cuda f32 linear weights: " + std::to_string(uploaded_f32) + "/" + std::to_string(counts.total()));
        if (counts.q8_0 > 0) {
            term.print_message("cuda q8_0 linear weights: " + std::to_string(uploaded_q8_0) + "/" + std::to_string(counts.q8_0));
        }
        if (counts.q4_0 > 0) {
            term.print_message("cuda q4_0 linear weights: " + std::to_string(uploaded_q4_0) + "/" + std::to_string(counts.q4_0));
        }
        if (counts.q4_1 > 0) {
            term.print_message("cuda q4_1 linear weights: " + std::to_string(uploaded_q4_1) + "/" + std::to_string(counts.q4_1));
        }
        if (q8_0_fallback + q4_0_fallback + q4_1_fallback > 0) {
            term.print_message(
                "cuda fallback: unsupported or missing quantized linear weights use CPU path" +
                std::string(" (Q8_0=") + std::to_string(q8_0_fallback) +
                ", Q4_0=" + std::to_string(q4_0_fallback) +
                ", Q4_1=" + std::to_string(q4_1_fallback) + ")"
            );
        }
    }
    term.print_message(backend_execution_note(backend_config));
    term.print_message("Type /help for commands, /exit to quit.\n");
    if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
        term.print_message("Tiny teaching model: random weights, small context window, smoke-test output.");
        term.print_message("Real chat demo: ./build/mini-llama run models/chat -n 8\n");
    }

    MiniLlamaContext ctx(&model);

    while (true) {
        term.print_user_prompt();
        std::string input = term.read_line();
        if (input.empty() && std::cin.eof()) {
            break;
        }

        // Handle commands
        if (input == "/help") {
            term.print_help();
            continue;
        }
        if (input == "/exit") {
            break;
        }
        if (input == "/clear") {
            session.clear();
            ctx = MiniLlamaContext(&model);
            if (lm.chat_template.empty()) {
                session.add_message("system", "You are a helpful assistant.");
            }
            term.print_message("Chat history cleared.\n");
            continue;
        }
        if (input == "/stats") {
            term.print_stats(session);
            continue;
        }
        if (input == "/params") {
            term.print_params(session.sampling_params);
            continue;
        }
        if (!input.empty() && input[0] == '/') {
            term.print_message("Unknown command: " + input + "\n");
            continue;
        }
        if (input.empty()) {
            continue;
        }

        std::vector<ChatMessage> candidate_messages = session.messages;
        candidate_messages.push_back({"user", input});

        std::string prompt_text = builder.build(candidate_messages);
        std::vector<int> tokens = tokenizer->encode(prompt_text);

        if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
            term.print_message(
                "Error: prompt uses " + std::to_string(tokens.size()) +
                " tokens, context window is " + std::to_string(model.config.max_seq_len) +
                ". Use /clear or a shorter prompt.\n"
            );
            continue;
        }

        int max_response = model.config.max_seq_len - static_cast<int>(tokens.size());
        if (max_response > max_response_tokens) {
            max_response = max_response_tokens;
        }

        session.messages = candidate_messages;
        const std::vector<int> previous_tokens = session.token_history;

        MiniSampler sampler(session.sampling_params);
        Tensor logits;

        auto start = std::chrono::steady_clock::now();

        size_t prefix_len = common_prefix_length(previous_tokens, tokens);
        if (prefix_len != previous_tokens.size() || prefix_len >= tokens.size()) {
            ctx = MiniLlamaContext(&model);
            prefix_len = 0;
        }
        std::vector<int> new_prompt_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
        session.set_token_history(tokens);

        // Prefill only the context suffix that is missing from KV cache.
        {
            MiniBatch prefill = MiniBatch::from_tokens(new_prompt_tokens, static_cast<int>(prefix_len));
            logits = forward_batch(model, ctx, prefill);
            ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
        }

        // Generate response
        std::vector<int> generated_ids;
        std::string streamed_reply;
        int generated_count = 0;

        term.print_assistant_prefix();
        try {
            for (int i = 0; i < max_response; ++i) {
                int next_token = sampler.sample(logits, session.sampling_params);
                tokens.push_back(next_token);
                session.append_token(next_token);
                ++generated_count;

                if (next_token != tokenizer->eos_id()) {
                    generated_ids.push_back(next_token);
                    std::string current_reply = tokenizer->decode(generated_ids);
                    if (current_reply.size() > streamed_reply.size()) {
                        term.print_token_text(current_reply.substr(streamed_reply.size()));
                        term.flush();
                        streamed_reply = current_reply;
                    }
                }

                MiniBatch decode = MiniBatch::single(next_token, static_cast<int>(tokens.size() - 1));
                logits = forward_batch(model, ctx, decode);
                ++ctx.n_decode_tokens;
                if (next_token == tokenizer->eos_id()) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            term.new_line();
            term.print_message("Inference error: " + std::string(e.what()) + "\n");
            continue;
        }

        // Decode and output
        std::string assistant_reply = tokenizer->decode(generated_ids);
        if (assistant_reply.size() > streamed_reply.size()) {
            term.print_token_text(assistant_reply.substr(streamed_reply.size()));
        }
        term.new_line();
        term.new_line();

        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        session.add_message("assistant", assistant_reply);
        session.record_turn(static_cast<int>(new_prompt_tokens.size()), generated_count, elapsed_ms);
    }

    term.print_message("Goodbye.\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
static void print_global_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <command> [options]\n"
              << "Commands:\n"
              << "  generate      One-shot text generation (default)\n"
              << "  run           Interactive chat mode\n"
              << "  inspect       Inspect model metadata\n"
              << "  inspect-gguf  Inspect GGUF file\n"
              << "  bench         Benchmark inference performance\n"
              << "  --cuda-info   Print CUDA device information for the selected CUDA build\n"
              << "\n"
              << "Run \"" << prog << " generate --help\", \"" << prog << " run --help\", or \""
              << prog << " bench --help\" for details.\n";
}

static int run_cuda_info() {
    try {
        int count = cuda_device_count();
        std::cout << "CUDA devices: " << count << "\n";
        for (int device_id = 0; device_id < count; ++device_id) {
            CudaDeviceInfo info = cuda_get_device_info(device_id);
            double total_gb = static_cast<double>(info.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "device " << info.id << ": " << info.name << "\n"
                      << "  device name: " << info.name << "\n"
                      << "  compute capability: " << info.compute_major << "." << info.compute_minor << "\n"
                      << "  total memory: " << info.total_memory_bytes << " bytes ("
                      << std::fixed << std::setprecision(2) << total_gb << " GB)\n"
                      << "  driver version: " << info.driver_version << "\n"
                      << "  runtime version: " << info.runtime_version << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUDA info failed: " << e.what() << "\n";
        return 1;
    }
}

// ---------------------------------------------------------------------------
// Inspect mode
// ---------------------------------------------------------------------------
static void print_inspect_usage(const char* prog) {
    std::cout << "Usage: " << prog << " inspect <model-path|dir> [options]\n"
              << "Options:\n"
              << "  --backend cpu|cuda   Execution backend (default: cpu; cuda requires -DMINI_LLAMA_CUDA=ON)\n"
              << "  --device <N>         CUDA device id for --backend cuda (default: 0)\n"
              << "  -h, --help           Show this help\n";
}

static MiniLlamaModel load_model_for_inspect(const std::string& path) {
    std::string model_path = path;
    if (std::filesystem::is_directory(path)) {
        std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
        std::filesystem::path json_path = std::filesystem::path(path) / "model.json";
        if (std::filesystem::exists(bin_path) && std::filesystem::exists(json_path)) {
            return load_model(json_path.string(), bin_path.string());
        }
        std::string gguf_path = find_gguf_in_directory(path);
        if (!gguf_path.empty()) {
            model_path = gguf_path;
        } else {
            return load_model(json_path.string(), bin_path.string());
        }
    }

    if (is_gguf_file(model_path)) {
        return load_gguf_model(model_path);
    }

    std::filesystem::path model_file(model_path);
    std::string config_path = (model_file.parent_path() / "model.json").string();
    return load_model(config_path, model_path);
}

static int run_inspect(int argc, char** argv) {
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0)) {
        print_inspect_usage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing model path.\n";
        print_inspect_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[2];
    BackendConfig backend_config;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            if (!parse_backend_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_device_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_inspect_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_inspect_usage(argv[0]);
            return 1;
        }
    }

    if (!validate_backend_or_print(backend_config)) {
        return 1;
    }

    if (std::filesystem::is_directory(model_path)) {
        std::filesystem::path config_path = std::filesystem::path(model_path) / "model.json";
        if (std::filesystem::exists(config_path) && !mini_llama::inspect_model(config_path.string())) {
            return 1;
        }
    }

    MiniLlamaModel model = load_model_for_inspect(model_path);
    if (!model.loaded) {
        std::cerr << "Failed to load model: " << model.load_error << "\n";
        return 1;
    }

    std::cout << "backend: " << backend_kind_name(backend_config.kind) << "\n";
    if (backend_config.kind == BackendKind::Cuda) {
        std::cout << "cuda: " << cuda_device_summary(backend_config.device_id) << "\n";
        if (!prepare_cuda_weights_or_print(model, backend_config)) {
            return 1;
        }
        print_cuda_execution_summary(model);
    }

    return 0;
}

static int run_inspect_gguf(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
        return 1;
    }
    GGUFReader reader;
    if (!reader.load(argv[2])) {
        std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
        return 1;
    }
    inspect_gguf(reader);
    return 0;
}

// ---------------------------------------------------------------------------
// Bench mode
// ---------------------------------------------------------------------------
static void print_bench_usage(const char* prog) {
    std::cout << "Usage: " << prog << " bench <model-path|dir> [options]\n"
              << "Options:\n"
              << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
              << "  -n, --n-predict <N>   Number of tokens to generate (default: 64)\n"
              << "  --seed <S>            Random seed (default: 0 = random)\n"
              << "  --tokenizer <path>    Path to vocab.json tokenizer file\n"
              << "  --quant q8_0|q4_0     Quantize loaded linear weights before benchmark\n"
              << "  --threads <N>         Number of threads for parallel ops (0 = auto)\n"
              << "  --backend cpu|cuda    Execution backend (default: cpu; cuda requires -DMINI_LLAMA_CUDA=ON)\n"
              << "  --device <N>          CUDA device id for --backend cuda (default: 0)\n"
              << "  --verbose             Print debug dumps after each step\n"
              << "  -h, --help            Show this help\n"
              << "\n"
              << "CUDA benchmark metrics:\n"
              << "  uploaded weights: CUDA-resident linear weights loaded before inference\n"
              << "  cuda linear/activation/attention calls: GPU kernel coverage during forward\n"
              << "  cpu attention fallback calls: attention steps that returned to CPU\n"
              << "  host->device / device->host copies: runtime transfer count and bytes\n"
              << "  cuda fallback: unsupported or missing quantized linear weights running on CPU\n";
}

static double benchmark_repeated(int warmup, int iterations, const std::function<void()>& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / static_cast<double>(iterations);
}

static int run_bench(int argc, char** argv) {
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0)) {
        print_bench_usage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing model directory.\n";
        print_bench_usage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[2];
    std::string prompt = "hello";
    int n_predict = 64;
    unsigned int seed = 0;
    std::string tokenizer_path;
    bool verbose = false;
    std::string quant_type;
    int n_threads = 0;
    BackendConfig backend_config;

    for (int i = 3; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], n_predict)) {
                std::cerr << "Invalid --n-predict value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_uint_arg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            quant_type = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], n_threads) || n_threads < 0) {
                std::cerr << "Invalid --threads value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            if (!parse_backend_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_device_arg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_bench_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_bench_usage(argv[0]);
            return 1;
        }
    }

    if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value: " << quant_type << ". Supported values: q8_0, q4_0.\n";
        return 1;
    }

    if (!validate_backend_or_print(backend_config)) {
        return 1;
    }

    LoadedModel lm = load_model_and_tokenizer(model_dir, "", tokenizer_path);
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        std::cerr << "Model vocab_size must be at least " << lm.tokenizer->vocab_size()
                  << " for the tokenizer.\n";
        return 1;
    }

    MiniLlamaModel baseline_model = lm.model;
    MiniLlamaModel& model = lm.model;
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
    std::vector<int> tokens = tokenizer->encode(prompt);
    if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
        std::cerr << "Prompt too long.\n";
        return 1;
    }
    if (tokens.size() + static_cast<size_t>(n_predict) > static_cast<size_t>(model.config.max_seq_len)) {
        n_predict = model.config.max_seq_len - static_cast<int>(tokens.size());
    }

    try {
        apply_quant_override(model, quant_type);
    } catch (const std::exception& e) {
        std::cerr << "Quantization failed: " << e.what() << "\n";
        return 1;
    }
    if (!prepare_cuda_weights_or_print(model, backend_config)) {
        return 1;
    }

    std::cout << "Benchmark: " << model_dir << "\n";
    std::cout << "  backend: " << backend_kind_name(backend_config.kind) << "\n";
    if (backend_config.kind == BackendKind::Cuda) {
        std::cout << "  cuda: " << cuda_device_summary(backend_config.device_id) << "\n";
    }
    std::cout << "  " << backend_execution_note(backend_config) << "\n";
    std::cout << "  prompt: \"" << prompt << "\" (" << tokens.size() << " tokens)\n";
    set_thread_count(n_threads);
    std::cout << "  n_predict: " << n_predict << "\n";
    std::cout << "  seed: " << seed << "\n";
    std::cout << "  quant: " << (quant_type.empty() ? "model-native" : quant_type) << "\n";
    std::cout << "  threads: " << get_thread_count() << "\n";
    std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n\n";
    if (backend_config.kind == BackendKind::Cuda) {
        print_cuda_execution_summary(model, "  ");
        std::cout << "\n";
    }

    BenchmarkResult result = run_benchmark(model, tokens, n_predict, seed, verbose);

    std::cout << "Results:\n";
    std::cout << "  prompt tokens:     " << result.n_prompt_tokens << "\n";
    std::cout << "  generated tokens:  " << result.n_generated_tokens << "\n";
    std::cout << "  decode tokens:     " << result.n_decode_tokens << "\n";
    std::cout << "  prefill time:      " << std::fixed << std::setprecision(2) << result.prefill_ms << " ms\n";
    std::cout << "  decode time:       " << std::fixed << std::setprecision(2) << result.decode_ms << " ms\n";
    std::cout << "  total time:        " << std::fixed << std::setprecision(2) << (result.prefill_ms + result.decode_ms) << " ms\n";
    std::cout << "  tokens/s (total):  " << std::fixed << std::setprecision(2) << result.tokens_per_sec() << "\n";
    std::cout << "  tokens/s (decode): " << std::fixed << std::setprecision(2) << result.decode_tokens_per_sec() << "\n";

    // Memory footprint
    size_t actual_bytes = model_weight_bytes(model);
    size_t f32_bytes = model_weight_bytes_f32(model);
    std::cout << "\n  weight memory:\n";
    std::cout << "    actual:    " << actual_bytes << " bytes (" << std::fixed << std::setprecision(2)
              << (actual_bytes / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "    f32 equiv: " << f32_bytes << " bytes (" << std::fixed << std::setprecision(2)
              << (f32_bytes / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "    savings:   " << std::fixed << std::setprecision(2)
              << (static_cast<double>(f32_bytes) / actual_bytes) << "x compression\n";
    if (backend_config.kind == BackendKind::Cuda) {
        std::cout << "\n  cuda runtime:\n";
        print_cuda_runtime_summary(model, "    ");
    }

    if (!quant_type.empty()) {
        try {
            Tensor baseline_logits = run_logits_for_tokens(baseline_model, tokens);
            Tensor quant_logits = run_logits_for_tokens(model, tokens);
            auto [max_err, mean_err] = logits_error(baseline_logits, quant_logits);

            std::cout << "\n  logits error vs model-native:\n";
            std::cout << "    max:  " << std::scientific << std::setprecision(3) << max_err << "\n";
            std::cout << "    mean: " << std::scientific << std::setprecision(3) << mean_err << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to compute logits error: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_global_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];
    if (command == "generate") {
        return run_generate(argc, argv);
    } else if (command == "run") {
        return run_chat(argc, argv);
    } else if (command == "inspect") {
        return run_inspect(argc, argv);
    } else if (command == "bench") {
        return run_bench(argc, argv);
    } else if (command == "inspect-gguf") {
        return run_inspect_gguf(argc, argv);
    } else if (command == "--cuda-info") {
        return run_cuda_info();
    } else if (command == "-h" || command == "--help") {
        print_global_usage(argv[0]);
        return 0;
    } else if (!command.empty() && command[0] == '-') {
        // Backward compatibility: default to generate mode when first arg is a flag.
        // Shift argv[1..] down by inserting "generate" at position 1.
        std::vector<char*> shifted_argv;
        shifted_argv.reserve(argc + 1);
        shifted_argv.push_back(argv[0]);
        shifted_argv.push_back(const_cast<char*>("generate"));
        for (int i = 1; i < argc; ++i) {
            shifted_argv.push_back(argv[i]);
        }
        return run_generate(argc + 1, shifted_argv.data());
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        print_global_usage(argv[0]);
        return 1;
    }
}
