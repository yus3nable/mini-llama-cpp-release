# mini-llama.cpp

一个可读型的 LLaMA-style C++ 推理内核。从 tiny model 出发，逐步构建成一个可读、可测、可解释的推理引擎，并提供终端聊天体验。

## 快速开始

```bash
# 编译
mkdir -p build && cd build
cmake .. && make -j4
cd ..

# GGUF chat demo（先准备 models/chat）
python3 scripts/download_demo_model.py
python3 scripts/export_gguf_tokenizer.py models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf --out-dir models/chat
./build/mini-llama run models/chat -n 8

# tiny model 工程 smoke
./build/mini-llama generate --config models/tiny/model.json --model models/tiny/model.bin -p "hello" -n 16

# 查看模型 manifest
./build/mini-llama inspect models/tiny

# 查看 GGUF metadata
./build/mini-llama inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf

# 运行测试
ctest --test-dir build --output-on-failure

# CUDA v2 构建和设备信息（需要 NVIDIA CUDA 机器）
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
./build-cuda/mini-llama --cuda-info
./build-cuda/mini-llama bench models/chat --backend cuda -p hello -n 32 --seed 1
python3 scripts/run_cuda_benchmarks.py --build-dir build-cuda --out cuda-v2-benchmark-$(date +%F).md

# ASan 测试
rm -rf build-asan && mkdir build-asan && cd build-asan
cmake -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' \
      -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address' ..
make -j4
cd ..
ctest --test-dir build-asan --output-on-failure
```

## GGUF 支持范围

当前 loader 支持 llama.cpp 常见 LLaMA/Qwen2 tensor 命名，支持 `F32`、`Q8_0`、`Q4_0`、`Q4_1` 四类 tensor。已实测：

- `Qwen2-0.5B-Instruct-Q8_0.gguf`：项目内置模型。
- 其他 LLaMA/Qwen2 风格的 Q4_0 / Q4_1 GGUF 模型：可自行下载后用 `inspect-gguf` 和 `generate -n 1` 做最小验证。

`Q4_K_M`、`Q5_K_M`、`Q6_K`、`IQ4_XS` 这类 K-quants / IQ quants 属于后续扩展任务。

## CUDA v2 当前状态

CUDA backend 是 v2 的可选执行路径，默认 CPU 构建保持独立。macOS 本地开发继续使用普通 `build/`，NVIDIA CUDA 机器使用 `build-cuda/`。

当前 CUDA 路径覆盖：

- F32 linear：CUDA / cuBLAS。
- Q8_0 / Q4_0 / Q4_1 linear：已上传权重时走 CUDA kernel。
- RMSNorm、RoPE、softmax、SiLU、elementwise add/mul：CUDA kernel。
- KV cache：GPU KV cache 写入和读取。
- attention decode：CUDA attention over GPU KV cache。
- decode 主链路：QKV、attention、WO、FFN、lm_head 尽量保持 device-resident。
- sampler：CPU greedy / temperature / top-k，logits 边界回传 CPU。

当前支持单机单卡 NVIDIA CUDA。多 GPU、NCCL、TensorRT、FlashAttention、Metal、ROCm、K-quants / IQ-quants 属于后续扩展项。

CUDA benchmark 输出会显示：

- `uploaded weights`：进入 GPU 的 linear 权重数量和显存占用。
- `cuda linear/activation/attention calls`：forward 期间 GPU kernel 覆盖数。
- `cpu attention fallback calls`：attention 回到 CPU 的次数。
- `host->device copies` / `device->host copies`：运行期传输次数和字节数。
- `cuda fallback`：暂未上传或暂未支持的量化 linear 权重会走 CPU 路径，并打印 Q8_0 / Q4_0 / Q4_1 数量。

云端常见排查：

```bash
# 确认 GPU、驱动和 CUDA toolkit
nvidia-smi
nvcc --version

# 常见 CUDA 路径
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

# 重新配置 CUDA build
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4

# 确认模型文件
ls -lh models/chat

# SSH 断连保护
nohup python3 scripts/run_cuda_benchmarks.py \
  --build-dir build-cuda \
  --out cuda-v2-benchmark-$(date +%F).md \
  > /tmp/mini-llama-bench.log 2>&1 &
tail -f /tmp/mini-llama-bench.log
```

## 命令行参数

```text
./mini-llama <command> [options]

Commands:
  generate             单次生成，兼容旧的 --prompt / --model 参数调用
  run <model-path|dir>   进入终端聊天模式
  inspect <model-dir>    查看模型配置和 tensor metadata
  inspect-gguf <path>    查看 GGUF metadata 和 tensor 列表
  bench <model-path|dir> 跑 prefill / decode 性能统计
  --cuda-info            查看 CUDA 设备信息

generate options:
  --model <path|dir>   权重二进制文件路径、GGUF 文件或模型目录 (默认: models/tiny/model.bin)
  --config <path>      配置 JSON 路径 (默认: models/tiny/model.json)
  -p, --prompt <str>   输入提示 (默认: "hello")
  -n, --n-predict <N>  生成 token 数量 (默认: 16)
  --temperature <T>    采样温度 (默认: 0.0 = greedy)
  --top-k <K>          Top-k 采样候选数 (默认: 0 = 关闭)
  --seed <S>           随机种子，用于复现采样输出 (默认: 0 = 随机)
  --tokenizer <path>   指定 vocab.json tokenizer 文件
  --quant q8_0|q4_0    运行前把 linear 权重量化到指定格式
  --threads <N>        并行算子线程数 (默认: 0 = auto)
  --backend cpu|cuda   执行后端 (默认: cpu；cuda 需要 -DMINI_LLAMA_CUDA=ON)
  --device <N>         CUDA device id，用于 --backend cuda
  --dump-logits <dir>  把每一步 logits 写入目录，用于 Python 数值对齐

run options:
  --temperature <T>    采样温度 (默认: 0.0 = greedy)
  --top-k <K>          Top-k 采样候选数 (默认: 0 = 关闭)
  --seed <S>           随机种子，用于复现采样输出 (默认: 0 = 随机)
  -n, --n-predict <N>  每轮最多生成 token 数量 (默认: 64)
  --tokenizer <path>   指定 vocab.json tokenizer 文件
  --backend cpu|cuda   执行后端 (默认: cpu；cuda 需要 -DMINI_LLAMA_CUDA=ON)
  --device <N>         CUDA device id，用于 --backend cuda

bench options:
  -p, --prompt <str>   输入提示 (默认: "hello")
  -n, --n-predict <N>  生成 token 数量 (默认: 64)
  --seed <S>           随机种子，用于复现采样输出 (默认: 0 = 随机)
  --tokenizer <path>   指定 vocab.json tokenizer 文件
  --quant q8_0|q4_0    运行前把 linear 权重量化到指定格式
  --threads <N>        并行算子线程数 (默认: 0 = auto)
  --backend cpu|cuda   执行后端 (默认: cpu；cuda 需要 -DMINI_LLAMA_CUDA=ON)
  --device <N>         CUDA device id，用于 --backend cuda
  --verbose            输出 tensor shape、logits top-k、KV cache position

bench 的 CUDA 指标包括 uploaded weights、cuda linear / activation / attention calls、CPU attention fallback、H2D / D2H copy bytes 和量化 linear fallback 数量。
```

## 项目结构

```text
mini-llama.cpp/
  CMakeLists.txt
  README.md
  include/mini_llama/
    tensor.h        # 最小张量结构
    ops.h           # 基础算子: matmul, linear, rmsnorm, softmax, rope, silu, swiglu, elementwise_mul, argmax
    model.h         # 模型配置、CPU 权重和可选 CUDA weight view
    tokenizer.h     # Tokenizer 接口 + ASCII / JSON vocab 实现
    kv_cache.h      # KV cache 读写（带边界检查）
    context.h       # 推理上下文、KV cache、token history、stats
    batch.h         # MiniBatch，统一 prefill / decode 输入
    forward.h       # forward_batch + 单 token forward pass
    sampler.h       # Greedy / temperature / top-k sampler
    chat.h          # 聊天消息和会话状态
    prompt_builder.h# 多轮消息转 prompt
    terminal.h      # 终端输入输出和内置命令
    loader.h        # 模型加载器
    debug.h         # Benchmark 结果和 debug dump
    gguf.h          # GGUF reader
    gguf_loader.h   # GGUF -> MiniLlamaModel
    gguf_tokenizer.h# GGUF/BPE tokenizer
    quantized_tensor.h # F32 / Q8_0 / Q4_0 / Q4_1 权重存储
    thread_pool.h   # 并行 row executor
    matmul_dispatch.h # naive / threaded / SIMD dispatch
    backend.h      # CPU / CUDA backend 选择
    cuda_runtime.h # CUDA runtime wrapper 和 RAII device buffer
    cuda_matmul.h  # CUDA cuBLAS matmul / linear / device-weight linear
    cuda_quant.h   # CUDA Q8_0 / Q4_0 / Q4_1 quantized linear
    cuda_ops.h     # CUDA RMSNorm / RoPE / SiLU / elementwise / softmax kernels
    cuda_kv_cache.h # CUDA KV cache 分配、写入、读回和清零
    cuda_attention.h # CUDA attention over GPU KV cache
    cuda_tensor.h  # CUDA device-resident tensor view
  src/
    tensor.cpp
    ops.cpp
    tokenizer.cpp
    bpe_tokenizer.cpp
    json_vocab_tokenizer.cpp
    kv_cache.cpp
    context.cpp
    batch.cpp
    forward.cpp     # Transformer block 核心实现（已拆分为子函数）
    sampler.cpp
    chat.cpp
    prompt_builder.cpp
    terminal.cpp
    loader.cpp
    debug.cpp
    main.cpp        # CLI: generate / run / inspect / bench
    gguf.cpp        # GGUF metadata / tensor reader
    gguf_loader.cpp # GGUF -> MiniLlamaModel
    gguf_tokenizer.cpp
    quant.cpp       # Q8_0 / Q4_0 / Q4_1 量化和线性层
    quantized_tensor.cpp
    thread_pool.cpp
    matmul_optimized.cpp
    backend.cpp
    cuda_runtime.cpp
    cuda_matmul.cpp
    cuda_quant.cpp / cuda_quant.cu
    cuda_ops.cpp / cuda_ops.cu
    cuda_kv_cache.cpp
    cuda_attention.cpp / cuda_attention.cu
    cuda_tensor.cpp
  tests/
    test_main.h     # 极简测试框架头文件
    test_main.cpp   # 测试 runner
    test_tensor.cpp # Tensor 单元测试
    test_ops.cpp    # 算子单元测试
    test_forward.cpp# Forward + 生成测试
    test_sampler.cpp# Sampler 单元测试
    test_chat.cpp   # Chat / PromptBuilder / Terminal 单元测试
    test_loader.cpp # Loader / manifest 单元测试
    test_tokenizer.cpp # Tokenizer 单元测试
    test_batch.cpp  # MiniBatch / context 状态测试
    test_debug.cpp  # Benchmark / debug dump 测试
    test_gguf.cpp
    test_gguf_loader.cpp
    test_gguf_tokenizer.cpp
    test_quant.cpp
    test_threaded_matmul.cpp
    test_cuda_runtime.cpp
    test_cuda_matmul.cpp
  scripts/
    export_tiny_model.py    # 生成 tiny 权重
    upgrade_model_json.py   # 给旧 model.json 补 manifest 字段
    reference_forward.py    # Python 数值参考实现
    make_golden.py          # 生成 Python golden logits
    compare_logits.py       # 对齐 C++ / Python logits
    test_golden.py          # CTest golden logits 回归
    test_quant_cli.py       # Q8_0 / Q4_0 CLI smoke
    test_threaded_cli.py    # 线程一致性 CLI smoke
    run_cuda_benchmarks.py  # CUDA v2 correctness + benchmark report
  models/tiny/
    model.json              # 模型配置 + tensor metadata
    model.bin               # F32 权重二进制
  models/chat/
    Qwen2-0.5B-Instruct-Q8_0.gguf
    vocab.json
    merges.txt
    special_tokens.json
```

## 核心推理链路

```text
prompt -> tokenizer.encode() -> tokens
  -> prefill: MiniBatch::from_tokens() -> forward_batch()
  -> decode loop:
       MiniSampler.sample(logits, params) -> next_token
       tokenizer.decode() -> text
       MiniBatch::single() -> forward_batch() -> logits
```

`forward_token()` 内部已拆分为清晰子函数：

```text
embed_token()
  -> for each layer:
       forward_layer()
         -> rmsnorm -> q/k/v projection -> rope()
         -> attention_forward() (含 KV cache 读写)
         -> residual
         -> ffn_forward() (SwiGLU)
         -> residual
  -> compute_logits()
```

## 测试

使用内置极简测试框架（零外部依赖）：

```bash
# 普通测试
ctest --test-dir build --output-on-failure

# ASan 内存安全测试
ctest --test-dir build-asan --output-on-failure
```

当前覆盖 C++ 内部测试，并通过 2 个 Python 回归测试：

- Tensor: shape, fill, indexing, bounds checks, 1D/2D/3D/4D accessors, row pointers, shape checks, reshape
- Ops: matmul, linear, rmsnorm, softmax, silu, swiglu, elementwise_mul, argmax, normal RoPE, NeoX RoPE
- Forward: logit shape, KV cache 写入, greedy 生成序列, position 递增, 非法 token / position 拒绝
- Sampler: greedy, temperature, top-k, seed 复现、非法参数拒绝
- Chat: message/session 状态、plain prompt、Qwen2 chat template、terminal 命令输出
- Loader: manifest 解析、重复 tensor 拒绝、shape 校验、offset overlap 拒绝、trailing bytes 拒绝
- Tokenizer: ASCII / JSON vocab encode/decode、special token、factory fallback
- Batch: MiniBatch 构造、prefill 与单 token decode 等价、context token history
- Debug: benchmark result、tensor shape dump、logits top-k、KV cache position
- Python: MVP token 序列回归、golden logits 数值对齐

## Python 参考对齐

```bash
python3 scripts/reference_forward.py --config models/tiny/model.json --weights models/tiny/model.bin --prompt "hello" -n 16
```

生成并对齐 golden logits：

```bash
./build/mini-llama generate --prompt hello --n-predict 16 --dump-logits artifacts/golden
python3 scripts/make_golden.py --config models/tiny/model.json --weights models/tiny/model.bin --prompt hello -n 16 --output-dir artifacts/golden
python3 scripts/compare_logits.py --cpp-dir artifacts/golden --py-dir artifacts/golden --vocab-size 128
```

C++ 和 Python 的 greedy token 序列完全一致，logits 误差保持在 `1e-4` 以内。

## 与 llama.cpp simple.cpp 的对应

| llama.cpp `simple.cpp` | mini-llama.cpp | 作用 |
|---|---|---|
| `llama_model_load_from_file()` | `load_model()` | 加载权重和配置 |
| `llama_model_get_vocab()` | `Tokenizer` | token / 文本互转 |
| `llama_tokenize()` | `tokenizer.encode()` | prompt 转 token ids |
| `llama_init_from_model()` | `MiniLlamaContext` | 创建推理状态和 KV cache |
| `llama_batch_get_one()` | `MiniBatch::single()` | 准备单 token decode 输入 |
| `llama_decode(ctx, batch)` | `forward_batch()` | 执行 batch 推理，内部顺序调用 `forward_token()` |
| `llama_sampler_sample()` | `MiniSampler.sample()` | 从 logits 选下一个 token |
| `llama_token_to_piece()` | `tokenizer.decode_token()` | token 转文本 |

## 功能状态

当前代码支持 JSON+BIN tiny model、Qwen2 GGUF demo 模型、BPE tokenizer、Qwen2 chat template、终端聊天、benchmark、Q8_0 / Q4_0 / Q4_1 量化、多线程 matmul、SIMD 优化和可选 CUDA backend。`generate` / `bench` 支持 `--quant q8_0|q4_0` 和 `--threads <N>`；`run models/chat -n 8` 使用 BPE tokenizer 和 Qwen2 chat template。

当前验收命令：

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure

cmake -S . -B build-asan \
  -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address'
cmake --build build-asan -j4
ctest --test-dir build-asan --output-on-failure

./build/mini-llama generate --config models/tiny/model.json --model models/tiny/model.bin --tokenizer models/tiny/vocab.json -p hello -n 16
./build/mini-llama bench models/tiny --tokenizer models/tiny/vocab.json -p hello -n 4 --quant q8_0
./build/mini-llama bench models/tiny --tokenizer models/tiny/vocab.json -p hello -n 4 --quant q4_0
./build/mini-llama bench models/tiny --tokenizer models/tiny/vocab.json -p hello -n 4 --threads 4 --verbose
./build/mini-llama inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
./build/mini-llama bench models/chat -p hello -n 1
```

## 模型尺寸

MVP 使用极小的 toy 模型：

```text
vocab_size = 128
dim = 32
hidden_dim = 86
n_layers = 2
n_heads = 4
n_kv_heads = 4
head_dim = 8
max_seq_len = 128
```

总参数量约 33K，权重文件约 132KB。

> **注意**：权重为随机生成，输出为乱码。MVP 的价值是展示推理机制，而非生成有用文本。
