# 05 LLM 推理加速与边缘部署 —— 面试完全指南

---

## 一、项目一句话定位

这是一个**面向大语言模型的推理优化与边缘部署研究项目**，通过量化、KV Cache 优化和投机解码等技术，在保持模型精度的前提下显著提升推理速度并降低显存占用。

---

## 二、核心技术栈

- **Python 3.10** + **PyTorch 2.x**
- **Transformers + Accelerate**（HuggingFace）
- **AutoGPTQ / AWQ / BitsAndBytes**（INT4/INT8 权重量化）
- **vLLM**（PagedAttention 与批量推理）
- **ONNX Runtime / TensorRT-LLM**（部署后端）
- **llama.cpp**（CPU/边缘设备推理）

---

## 三、核心原理

大模型推理的瓶颈主要来自两个地方：**显存带宽**（Weight Loading）和**计算量**（Attention 计算）。本项目系统性地应用了业界主流优化技术：

1. **权重量化（Weight Quantization）**：
   将 FP16/FP32 的权重压缩到 INT8 甚至 INT4，减少显存占用和加载带宽。比较了 GPTQ、AWQ、BitsAndBytes 三种方案的精度和速度 trade-off。

2. **KV Cache 优化**：
   在自回归生成中，每一轮都需要重复计算前面 token 的 Key 和 Value。通过缓存并优化其内存布局（如 vLLM 的 PagedAttention），可以大幅提升长序列生成的吞吐。

3. **投机解码（Speculative Decoding）**：
   用一个小模型（Draft Model）快速生成多个候选 token，再由大模型（Target Model）一次性并行验证。如果验证通过，就能一次前进多步，显著降低总延迟。

4. **边缘部署适配**：
   将量化后的模型导出到 `llama.cpp` 格式，在纯 CPU 环境下运行，验证边缘设备（如 MacBook M2、树莓派）上的可行性。

---

## 四、实现细节

### 架构设计
```
experiments/
  ├── quant_gptq.py          # GPTQ INT4 量化实验
  ├── quant_awq.py           # AWQ 量化实验
  ├── quant_bnb.py           # BitsAndBytes 8bit 实验
  ├── vllm_benchmark.py      # vLLM 吞吐测试
  ├── speculative_decoding.py # 投机解码实现
  ├── edge_llamacpp.py       # llama.cpp 边缘部署
  └── benchmark_suite.py     # 统一评测脚本
```

### 关键实验
- **量化精度对比**：在 C-Eval / GSM8K 上测试 FP16、INT8、INT4 的精度差异。结果显示 AWQ 在 INT4 下精度损失最小（<1%），而 GPTQ 的速度最快。
- **吞吐对比**：用 vLLM 的 PagedAttention 对比 HuggingFace 原生 `generate()`，在 batch_size=16、seq_len=2048 时吞吐提升 **3.8x**。
- **投机解码**：用 1B 参数的小模型作为 Draft，配合 7B Target 模型，在代码生成任务上 latency 降低 **1.8x**。
- **边缘部署**：将 Qwen2-7B AWQ 模型转换为 GGUF 格式，在 MacBook M2 (16GB RAM) 上实现 8 tok/s 的可持续生成速度。

### 难点与解决
- **量化后模型精度骤降**：初期直接用 GPTQ 对某些数学推理任务精度掉得很厉害。通过对比发现 **AWQ 对激活值异常更鲁棒**，最终选择 AWQ 作为精度敏感任务的量化方案。
- **vLLM 在 Windows 上编译困难**：vLLM 的 CUDA kernel 在 Windows 上有很多兼容性问题。我们选择在 WSL2 中部署 vLLM，或者使用其预编译的 wheel。
- **投机解码的 Draft 模型选择**：小模型不能和 Target 模型差异太大，否则接受率过低。我们用 Target 模型的 1B 蒸馏版本作为 Draft，接受率稳定在 65% 左右。

---

## 五、对应岗位

- 大模型推理优化工程师
- AI Infra 工程师
- 端侧 AI 工程师
- 高性能计算（HPC）工程师

---

## 六、简历描述建议

> **系统研究并实践了大语言模型的推理优化与边缘部署技术栈**，覆盖权重量化（GPTQ / AWQ / BitsAndBytes）、KV Cache 优化（vLLM PagedAttention）、投机解码（Speculative Decoding）及边缘设备部署（llama.cpp GGUF）。在 7B 模型上完成了从 FP16 到 INT4 的完整精度-速度 trade-off 分析，筛选出 AWQ 作为精度敏感场景的最优量化方案；通过 vLLM 将批量推理吞吐提升了 3.8 倍；借助投机解码技术把单请求延迟降低了 1.8 倍；并成功将量化模型部署到 MacBook M2 上实现 8 tok/s 的可持续生成。

---

## 七、高频面试问题与回答

### Q1：GPTQ 和 AWQ 有什么区别？为什么 AWQ 精度更好？
**A**：
- **GPTQ** 是一种逐层/逐通道的后训练量化方法，它通过求解最小二乘问题来找到最接近原始权重的 INT4 表示。它的目标是**最小化权重本身的重建误差**。
- **AWQ（Activation-aware Weight Quantization）** 的核心洞察是：**并非所有权值对模型输出的贡献都相等**。那些与更大激活值相乘的权重更重要。AWQ 会对这些"关键权重"进行保护（比如保持更高的精度或做缩放），从而显著降低量化带来的精度损失。

在我们实验中的数学推理任务上，AWQ 的精度比 GPTQ 高出 2-3 个百分点。

### Q2：vLLM 的 PagedAttention 为什么快？
**A**：传统 LLM 推理中，KV Cache 是连续分配的，这会导致严重的内存碎片和过度分配（因为不知道最终序列有多长）。**PagedAttention 把 KV Cache 划分成固定大小的 Block（类似操作系统的虚拟内存分页）**，按需分配，不连续的 Block 通过查找表映射到物理内存。这样可以：
1. 消除内部碎片；
2. 支持动态扩展序列长度；
3. 大幅提高 batch size 和 GPU 利用率。

### Q3：投机解码的适用场景是什么？
**A**：投机解码最适合**生成内容具有较高局部确定性**的场景，比如：
- 代码生成（很多 token 是模板化的括号、关键字）
- 结构化输出（JSON、XML）
- 长文本续写

但如果 Draft 模型和 Target 模型差异太大，或者任务本身随机性很高（如创意写作），接受率会很低，反而增加开销。

### Q4：如果要部署到真正的 MCU（如 STM32）上，7B 模型可行吗？
**A**：**不可行**。即使是 INT4 量化的 7B 模型也需要约 4GB 内存，远超一般 MCU 的 SRAM（几十到几百 KB）。要跑在 MCU 上，必须使用** TinyML 级别**的模型（如 100M 以下的 Transformer 或 RNN）。不过我们可以把大模型放在云端或边缘网关（如树莓派），MCU 只做传感器数据采集和结果展示。这也是我后续项目（13-15）的一个延伸思考方向。
