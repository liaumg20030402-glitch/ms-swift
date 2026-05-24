# `sft_train.sh` 参数详解（ms-swift / Megatron 后端）

本笔记对应脚本 [../sft/sft_train.sh](../sft/sft_train.sh)，逐项解释里面用到的训练参数。该脚本走的是 **`megatron sft`** 入口（不是 `swift sft`），所以参数大部分来自 ms-swift 的 Megatron 适配层 `swift/megatron/arguments/megatron_args.py`，少量从 `swift/arguments/base_args/` 继承。

---

## 一、入口与环境

```bash
megatron sft \
    --cached_dataset $CACHED_DATASET \
    --model $MODEL_PATH \
    ...
```

`megatron sft` 是 ms-swift 把 NVIDIA Megatron-Core 包装成 swift 风格 CLI 的入口（见 [../../swift/megatron/](../../swift/megatron/)）。和 `swift sft`（HuggingFace 后端）的区别：

| | `swift sft` | `megatron sft` |
|---|---|---|
| 计算后端 | HF `transformers` + `trainer` | Megatron-Core |
| 并行 | DDP / FSDP / DeepSpeed | TP / PP / SP / DP（更细粒度） |
| 适用规模 | 单机 / 小集群 | 多机大模型（27B/35B+） |
| 权重格式 | 直接读 HF safetensors | 通常先 HF→mcore 转换，再读 |

脚本里 `Qwen3.5-27B`、`Qwen3.5-35B-A3B`（MoE）这种体量，走 Megatron 是合理选择。

---

## 二、数据相关

### `--cached_dataset $CACHED_DATASET`
指向 **预先 tokenize 好的数据集目录**（见 [../../swift/arguments/export_args.py](../../swift/arguments/export_args.py) 中 `to_cached_dataset`）。流程是：
1. 用 `swift export --to_cached_dataset true ...` 把原始 JSONL → 已编码张量
2. 训练时直接 `--cached_dataset <dir>` 加载，**跳过模板渲染和 tokenize**

对大数据集（脚本里是 `train_sft_1043-20260430`）效果显著：避免每次 launch 都重复跑几十分钟预处理。

### `--split_dataset_ratio=0`
从训练集里切多少比例做 val。`=0` 表示**不切验证集**——通常配合 `--save_strategy epoch` 使用，整轮整轮地存 ckpt，靠下游评测脚本判断好坏。

### `--load_from_cache_file true`
HuggingFace `datasets` 的标准选项，复用磁盘上的 arrow cache。已经在用 `cached_dataset` 时其实不是关键，但开着没坏处。

### `--dataloader_num_workers 8` / `--dataset_num_proc 196`
- `dataloader_num_workers`：DataLoader 子进程数（运行时取 batch）
- `dataset_num_proc`：预处理时 `datasets.map` 的并发进程数（仅在没用 `cached_dataset` 时才大量起作用）

---

## 三、模型加载与冻结

### `--model $MODEL_PATH`
模型路径。Megatron 后端这里通常指向 **mcore 格式权重**（不是原始 HF），转换脚本见 [../../swift/megatron/convert.py](../../swift/megatron/convert.py)。

### `--save_safetensors true`
保存时同时输出一份 safetensors 格式，方便下游推理（SGLang / vLLM）直接加载。

### `--finetune true`
区分**继续训练 (continue training)** vs **微调 (finetune)**：
- `true`：把 `$MODEL_PATH` 视为预训练权重，**只加载权重、不加载 optimizer/lr scheduler/RNG**，从 step 0 开始
- `false`：恢复完整训练状态（用于 checkpoint 续训）

源码见 `megatron_args.py:455`。

### `--freeze_llm false / --freeze_vit true / --freeze_aligner true`
来自 `swift/megatron/arguments/megatron_args.py:337-340`。Qwen3-VL 这类多模态模型把参数分成三块：

| 组件 | 含义 | 本脚本 |
|---|---|---|
| `llm` | 语言模型主干 | **训** |
| `vit` | 视觉编码器 | 冻结 |
| `aligner` | 视觉→语言投影层 | 冻结 |

但脚本里 `MODEL_PATH=Qwen3.5-27B` 是纯文本模型，这两个 freeze 参数实际上**没东西可冻**，留着是模板写法，不影响。

---

## 四、并行策略（Megatron 核心）

### `--tensor_model_parallel_size 4`
**TP=4**：把单层权重矩阵切到 4 张卡上做矩阵乘。`NPROC_PER_NODE=8` 时，TP=4 → 每节点 2 个 TP 组。

### `--sequence_parallel true`
**SP**：和 TP 配套使用。把 LayerNorm/Dropout 这些非矩阵乘的算子按 seq 维切分，进一步省显存。源码做了校验（`megatron_args.py:726-727`）：
```python
if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
    self.sequence_parallel = False
```
即 **TP=1 时 SP 自动关掉**。

### `--micro_batch_size 1` / `--global_batch_size 144`
- `micro_batch_size`：单卡单次 forward 的样本数
- `global_batch_size`：一个 optimizer step 处理的总样本数
- 二者关系（隐含 DP 与梯度累积）：
  ```
  global = micro × DP × grad_accum_steps
  ```

本脚本：假设 2 节点 × 8 卡 = 16，TP=4 → DP=4，那么 `grad_accum = 144 / (1×4) = 36`。

### `--packing true`
**序列打包**：把多条短样本拼成一条 `max_length`，因果掩码确保互不串扰。和 `padding_free` 二选一即可（见 `template_args.py:60-64`）：

- `packing`：预处理时打包，**训练快、显存稳定**
- `padding_free`：运行时 unpad，**无预处理开销，但稍慢**

脚本里 `--packing true --padding_free true` **同时开**——megatron 后端允许这么做，packing 主导，padding_free 起兜底/对齐作用。

---

## 五、长上下文与思考模式

### `--max_length 131072`
单样本最长 token 数 = **128K**。来自 `template_args.py:28-31`：
> Samples exceeding this length are handled according to `truncation_strategy` to prevent OOM errors.

配合 `packing=true`，一条打包后的序列最长就是 128K。这是医疗长文档 SFT 常见配置。

### `--add_non_thinking_prefix true`
关键参数，来自 `template_args.py:103-106`：
> 训练时，对于 assistant 部分**不以 `<think>` 开头**的样本，自动加上一个 non-thinking 前缀（即 `<think>\n\n</think>\n\n`）。

作用：让一份数据集里既能包含 thinking 样本（保留 `<think>...</think>`）又能包含 non-thinking 样本（前缀被加成空 think 块），**统一格式训练混合思考模型**。这就是 Qwen3 在推理时能用 `enable_thinking=True/False` 切换两种模式的训练侧依据，参见姊妹笔记里讲 `infer_sglang.py` 的部分。

### `--loss_scale ignore_empty_think`
来自 `swift/loss_scale/other.py:5-6` 和 `template_args.py:80-81`：
> 对空的 `<think>\n\n</think>\n\n`（正则 `<think>\s*</think>\s*`）**不计算 loss**。

为什么要这么干：
- 上面那个 `add_non_thinking_prefix` 给 non-thinking 样本塞了空 think 块
- 如果把这个空块也计入 loss，模型会学到"在 non-thinking 模式下要先输出空 think"——这是套路记忆，没营养
- 所以这里**只对真正的内容计算 loss**，空 think 块当作格式而不是学习目标

可以和别的策略叠加，如 `'default+ignore_empty_think'`。

---

## 六、优化器与学习率

### `--lr 5e-6 / --min_lr 5e-7 / --lr_warmup_fraction 0.05`
- `lr`：峰值学习率
- `min_lr`：cosine decay 末端最小学习率
- `lr_warmup_fraction 0.05`：前 5% 训练步线性 warmup

5e-6 是大模型 full-finetune 的典型量级（比预训练 1e-4 低 20 倍）。

### `--num_train_epochs 3`
跑 3 个 epoch。Megatron 后端会换算成 `train_iters`（`megatron_args.py:833-836`）：
```
train_iters = (len(dataset) // step_batch_size × step_batch_size) × epochs // global_batch_size
```

### `--save_strategy epoch`
每个 epoch 末保存一次 ckpt。注意：流式数据集不支持这种策略（`megatron_args.py:832`）。

### `--no_save_optim false / --no_save_rng false`
双否定 = **保存** optimizer 状态和 RNG 状态。这样万一中断了能精确续训。代价是 ckpt 体积更大。

### `--optimizer_cpu_offload false`
不把 optimizer 状态 offload 到 CPU。27B 全参训练 + TP=4 显存够用就别 offload，offload 会拖慢一截。

---

## 七、性能与显存优化

### `--attention_backend flash`
来自 `megatron_args.py:387`，可选 `flash | fused | unfused | local | auto`。`flash` 即 FlashAttention，27B 长序列必开。

### `--cross_entropy_loss_fusion true`
把 final logits 的 cross-entropy 和前面的 matmul 融合成单 kernel，省显存（避免物化 vocab_size 大的 logits 张量）+ 提速。`megatron_args.py:384`。

### `--recompute_granularity full / --recompute_method uniform / --recompute_num_layers 1`
**激活重计算（gradient checkpointing）**，源码 `megatron_args.py:372-375`：

- `granularity`：
  - `selective`：只重算 attention（默认，开销小）
  - `full`：重算**整个 transformer block**（省显存最多）
  - `none`：关闭
- `method`（仅 `full` 时有效）：
  - `uniform`：每 N 层均匀切一刀
  - `block`：指定具体哪些块
- `num_layers 1`：配合 `uniform`，**每 1 层重算一次**（= 全部重算，最省显存）

校验逻辑：`selective` 配 `method` 会报错（`megatron_args.py:643-644`）。

27B + 128K seq 不开 full recompute 几乎必爆显存。

### `--logging_steps 1`
每 step 打一次 log，调试期开 1，稳定后建议 10/50。

---

## 八、参数关系速查图

```
                    数据
    ┌─────────────────────────────────────┐
    │ cached_dataset → 跳过 tokenize       │
    │ split_dataset_ratio=0 → 不切 val     │
    │ add_non_thinking_prefix=true ┐       │
    │ loss_scale=ignore_empty_think┘配套   │
    │ packing + padding_free → 长序列效率  │
    │ max_length=131072 → 128K 上限        │
    └─────────────────────────────────────┘
                    │
                    ▼
                  模型
    ┌─────────────────────────────────────┐
    │ model=Qwen3.5-27B (mcore 格式)       │
    │ finetune=true → 从权重起步、不复用 opt│
    │ freeze_llm=false → 全参训练          │
    └─────────────────────────────────────┘
                    │
                    ▼
                  并行
    ┌─────────────────────────────────────┐
    │ TP=4 + SP=true                       │
    │ 推断 DP = world_size / TP            │
    │ micro=1, global=144 → 自动算 accum   │
    └─────────────────────────────────────┘
                    │
                    ▼
                优化与显存
    ┌─────────────────────────────────────┐
    │ lr=5e-6, warmup=5%, min_lr=5e-7      │
    │ recompute=full/uniform/1层 → 省显存  │
    │ attention=flash, ce_fusion=true      │
    │ save_strategy=epoch × 3 epoch        │
    └─────────────────────────────────────┘
```

---

## 九、相关源码索引

- 通用模板参数：[../../swift/arguments/base_args/template_args.py](../../swift/arguments/base_args/template_args.py)
- 数据参数：[../../swift/arguments/base_args/data_args.py](../../swift/arguments/base_args/data_args.py)
- Megatron 专属参数：[../../swift/megatron/arguments/megatron_args.py](../../swift/megatron/arguments/megatron_args.py)
- `ignore_empty_think` 配置：[../../swift/loss_scale/other.py](../../swift/loss_scale/other.py)
- HF→mcore 权重转换：[../../swift/megatron/convert.py](../../swift/megatron/convert.py)
