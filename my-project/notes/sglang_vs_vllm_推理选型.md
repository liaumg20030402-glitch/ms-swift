# SGLang vs vLLM：推理框架选型笔记

本笔记结合 [../sft/infer_sglang.py](../sft/infer_sglang.py) 的实际用法，对比 SGLang 和 vLLM 两个主流 LLM 推理框架在**离线批量推理**与**在线服务**两种场景下的优劣，帮助决定后续什么场景该用哪个。

> 注：两个框架都在快速迭代（vLLM v1 大重写、SGLang 持续优化 RadixAttention），下面的对比基于 2025 年下半年到 2026 年初的稳定版本，差距随时可能收敛或反超。

---

## 一、一句话总结

| | 用一句话 |
|---|---|
| **SGLang** | 把"prompt 即程序"做到极致，**结构化生成 + 共享前缀**的场景特别强，离线批量 Engine API 设计干净 |
| **vLLM** | 推理框架界的"通用工业品"，**生态最广、硬件支持最全、模型覆盖最快**，OpenAI 兼容服务的事实标准 |

挑选规则的粗糙版：
- 你要**批量跑 SFT/Eval、prompt 高度结构化、共享前缀很长** → SGLang
- 你要**搭线上服务、跨硬件部署、模型/quant 选择多变** → vLLM
- 两者都行不知道选哪个 → vLLM（生态保险，踩坑少）

---

## 二、核心机制对比

### 2.1 注意力缓存

| | SGLang | vLLM |
|---|---|---|
| 机制 | **RadixAttention**（基数树前缀缓存） | **PagedAttention** + Prefix Caching（`enable_prefix_caching=True`） |
| 共享前缀 | **强**：树形结构天然支持分叉、复用、回退 | 好：哈希分片，共享长前缀 OK，分支结构不如 SGLang |
| KV cache 命中率 | 多轮对话/few-shot/agent 场景**显著更高** | 单轮独立 prompt 差距很小 |

**实际影响**：
- 如果你 batch 里 100 条样本共享同一段 8K 系统 prompt，SGLang 实测能省 30%~60% prefill 时间
- 如果每条样本 prompt 都不一样，两者基本打平

[infer_sglang.py:178-183](../sft/infer_sglang.py#L178-L183) 里同一个 task 的所有样本共享同一个 chat template 头部（`<|im_start|>system\n...<|im_end|>`），RadixAttention 在这里会自动命中。

### 2.2 结构化生成

| | SGLang | vLLM |
|---|---|---|
| JSON Schema | 原生支持，编译为正则 | `guided_json` 支持，xgrammar 后端 |
| Regex / EBNF | 原生支持，**和 KV cache 联动** | 支持，但 mask 计算和缓存分离 |
| 性能 | 约束生成几乎**零开销** | 有可见开销（10%~30% 慢） |

**实际影响**：要让模型输出严格 JSON、限定枚举值、走 ReAct 这种格式时，SGLang 的速度优势明显。

### 2.3 前端 DSL（独家特性）

SGLang 提供一套 Python DSL：

```python
@sgl.function
def few_shot_qa(s, question):
    s += "Q: ...\nA: ...\n"        # 共享前缀
    s += f"Q: {question}\nA: "
    s += sgl.gen("answer", max_tokens=64)
```

可以做 `fork`（并行分支）、`select`（多选评分）、嵌套 `gen`。**vLLM 没有对应的东西**——它本质是个推理引擎，不管控制流。

**实际影响**：写复杂 prompt 流水线（多步推理、self-consistency、tree-of-thought）时 SGLang 代码量少很多。但 [infer_sglang.py](../sft/infer_sglang.py) 用的是更朴素的 **Engine API**（`engine.generate(...)`），并没有用 DSL——这种情况下 SGLang 相对 vLLM 的优势主要剩 RadixAttention。

---

## 三、生态与覆盖面

### 3.1 模型支持

| | SGLang | vLLM |
|---|---|---|
| 新模型 day-0 | 通常滞后 1~2 周 | **基本当天就有 PR** |
| 多模态 | Qwen2.5/3-VL、LLaVA、InternVL 都支持 | 覆盖更全（含 Pixtral、Llama-3.2-Vision 等） |
| MoE | 支持，Qwen3-MoE / DeepSeek-V3 都有优化 | 支持，且通常优化更早进 |

### 3.2 硬件后端

| | SGLang | vLLM |
|---|---|---|
| NVIDIA CUDA | ✅ 一等公民 | ✅ 一等公民 |
| AMD ROCm | ✅ | ✅ 更稳 |
| Intel CPU/GPU | ⚠️ 部分 | ✅ |
| TPU / Inferentia / 昇腾 | ❌ | ✅ |

**实际影响**：你只用 NVIDIA → 没差别。混合硬件部署 → vLLM。

### 3.3 量化

| | SGLang | vLLM |
|---|---|---|
| FP8 (W8A8) | ✅ | ✅ |
| AWQ / GPTQ | ✅ | ✅ |
| INT4/Marlin/SmoothQuant | 部分支持 | **更全** |

**实际影响**：低比特量化场景 vLLM 选项更多。

### 3.4 服务化 / OpenAI 兼容

| | SGLang | vLLM |
|---|---|---|
| OpenAI 兼容 API | ✅ | ✅ **事实标准** |
| 指标 / 链路追踪 | 基本 | 更完整（Prometheus、OpenTelemetry） |
| LangChain / LiteLLM | 间接支持 | 直接集成 |
| 多副本 / Ray 集成 | 弱 | 强 |

**实际影响**：要部署成生产服务，vLLM 仍是更稳的选择。

---

## 四、性能维度

注意：**没有"普遍更快"这种说法**，下面是通常的趋势。

| 场景 | 谁更快 | 原因 |
|---|---|---|
| 共享长前缀的 batch 推理 | **SGLang** | RadixAttention |
| 结构化输出（JSON/Regex） | **SGLang** | 约束与 KV 一体化 |
| 纯解码吞吐（独立 prompt） | 接近 | 都用 CUDA Graph + FlashInfer/FlashAttn |
| 长上下文 prefill (>32K) | 接近 | 都做了 chunked prefill |
| 投机解码 (Speculative) | vLLM 略早 | vLLM v1 集成更深 |
| 多 LoRA 热切换 | **vLLM** | LoRA serving 更成熟 |
| 大规模 MoE | 接近 | 都做了 EP / 专家路由优化 |

---

## 五、回到本项目：为什么 [infer_sglang.py](../sft/infer_sglang.py) 选 SGLang？

结合脚本看实际收益：

1. **批量评测场景**，每个数据集所有样本共享同一段 system prompt（医疗指令、思考模式开关），**RadixAttention 命中率高**
2. **思考模式切换** 通过 `enable_thinking` 走 chat template，SGLang 的 Engine 对 `prompt + image_data` 的批接口足够干净（[infer_sglang.py:198-202](../sft/infer_sglang.py#L198-L202)）
3. **多模态**：Qwen3.5-VL（脚本注释里提的 27B/35B-A3B）SGLang 支持成熟，`image_data` 走 base64 list 这个接口很顺手
4. **离线场景对生态依赖低**：不需要 OpenAI 兼容、不需要 Ray、不需要监控——SGLang 的劣势这里都不构成问题

如果换成下面的场景，应该考虑切回 vLLM：

- 要起一个 OpenAI 兼容服务给前端调用
- 要支持多 LoRA 热切换（在线 A/B 不同 SFT 版本）
- 要部署到非 NVIDIA 硬件
- 模型刚发布几天，SGLang 还没接

---

## 六、可执行的"二选一"决策树

```
你要做什么？
├── 批量评测/数据合成（离线、固定 prompt）
│   └── prompt 高度共享前缀 / 需要结构化输出？
│       ├── 是 → SGLang ✅
│       └── 否 → 都行，看团队熟悉度
│
├── 上线生产服务
│   └── 需要 OpenAI 兼容 / 多 LoRA / 跨硬件 / 监控完整？
│       ├── 是 → vLLM ✅
│       └── 否 → 都行，vLLM 仍然更稳
│
├── 写 agent / 多步推理流水线
│   └── SGLang DSL ✅（fork、select、嵌套 gen 是独门武器）
│
└── 新模型刚发布，急着跑
    └── vLLM（day-0 支持概率更高）
```

---

## 七、配置等价对照（迁移参考）

如果哪天要把 [infer_sglang.py](../sft/infer_sglang.py) 改成 vLLM 版本，主要参数对照：

| SGLang | vLLM | 含义 |
|---|---|---|
| `sgl.Engine(model_path=...)` | `vllm.LLM(model=...)` | 引擎初始化 |
| `tp_size` | `tensor_parallel_size` | TP 大小 |
| `dp_size` | （无，需起多副本） | DP 大小 |
| `mem_fraction_static` | `gpu_memory_utilization` | 显存占用比例 |
| `engine.generate(prompt=..., image_data=..., sampling_params=...)` | `llm.generate(prompts=..., sampling_params=...)` + 多模态走 `multi_modal_data` | 推理调用 |
| `sampling_params={"max_new_tokens": ..., ...}` | `SamplingParams(max_tokens=..., ...)` | 采样参数（注意 `max_new_tokens` vs `max_tokens` 命名不同） |
| `out["text"]` | `out.outputs[0].text` | 取生成文本 |

---

## 八、相关链接

- SGLang 官方：https://github.com/sgl-project/sglang
- vLLM 官方：https://github.com/vllm-project/vllm
- 本项目推理脚本：[../sft/infer_sglang.py](../sft/infer_sglang.py)
- 训练参数笔记：[sft_train_参数详解.md](sft_train_参数详解.md)
