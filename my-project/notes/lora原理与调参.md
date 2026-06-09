# LoRA 原理与调参笔记

> 配套实验：`my-project/homework`（Qwen3.5-4B 在 hendrycks_math 上做 LoRA 微调）

---

## 1. LoRA 原理

**核心假设**：大模型适配下游任务时，权重的更新量 ΔW 是**低秩**的——即 ΔW 可以用两个小矩阵的乘积近似。

对一个原始权重矩阵 `W ∈ R^{d×k}`（冻结不训），LoRA 在旁边加一条低秩支路：

```
W' = W + ΔW = W + (alpha / r) · B · A
     A ∈ R^{r×k}   (降维, 高斯初始化)
     B ∈ R^{d×r}   (升维, 零初始化 → 训练开始时 ΔW=0)
     r << min(d, k)
```

训练时**只更新 A、B**，原权重 W 保持冻结。

- **参数量**：从 `d×k` 降到 `r×(d+k)`，通常只占全参数的 0.1%~1%（本实验 16.2M / 4555M ≈ 0.36%）。
- **省显存**：不需要为 W 存优化器状态（Adam 的一阶/二阶动量）。
- **可插拔**：训练产物是小小的 adapter，推理时叠加到底模上；不改原权重，可随时卸载/切换。
- **B 初始化为 0**：保证微调起点等于原模型，训练平滑。

---

## 2. 训练关键参数（swift sft）

| 参数 | 含义 | 调参建议 |
|---|---|---|
| `--tuner_type lora` | 用 LoRA（相对全参数微调） | 固定 |
| `--lora_rank` (r) | 低秩支路的秩，**控制容量** | 8/16/32/64。小=省参省显存、容量低；大=容量高、更易过拟合。常用 8 起步 |
| `--lora_alpha` | 缩放系数，有效缩放 = `alpha/r` | 常设 `alpha=2r`（本实验 32/8=4 倍）或 `alpha=r`（=1 倍）。alpha 越大，LoRA 的影响越强 |
| `--lora_dropout` | LoRA 支路的 dropout，防过拟合 | 0.0~0.1，老师用 0.1 |
| `--target_modules` | 把 LoRA 注入哪些线性层 | `all-linear`=所有线性层（q/k/v/o/gate/up/down 等）；也可只注意力层 `q_proj k_proj v_proj o_proj`，容量更小、更不易破坏基座 |
| `--learning_rate` | 学习率 | LoRA 典型 `1e-4 ~ 2e-4`。**对已经很强的基座，过高的 lr 会破坏原能力**（见第 5 节） |
| `--num_train_epochs` | 训练轮数 | 1~3。窄数据上 epoch 越多越容易过拟合 |
| `--per_device_train_batch_size` | 每卡每步样本数 | 受显存限制，OOM 就调小 |
| `--gradient_accumulation_steps` | 梯度累积步数 | **有效 batch = per_device × grad_accum × 卡数**。本实验 2×8×4=64 |
| `--max_length` | 序列截断长度 | 覆盖大部分样本即可，越长越占显存 |
| `--warmup_ratio` | 学习率预热比例 | 0.03~0.1 |
| `--lr_scheduler_type` | 学习率调度 | `cosine`（缓降）常用 |
| `--group_by_length` | 按长度分组以减少 padding | Qwen3.5 transformers 后端不支持 packing 时用它提速（loss 曲线会轻微抖动） |
| `--use_liger_kernel` | 融合算子 | 大词表模型省显存的关键（避免物化 25 万词表的 logits） |

**有效 batch 的理解**：`per_device_train_batch_size` 不是全局 batch，一次真正的梯度更新用的是 `per_device × grad_accum × GPU数` 条样本。要省显存又保持训练动态，就「调小 per_device、调大 grad_accum」。

---

## 3. 推理/评测采样参数

| 参数 | 含义 | 评测准确率时的建议 |
|---|---|---|
| `temperature` | 采样温度，控制随机性 | **0 = greedy（贪心，确定性）**：每次结果一样，准确率可复现，**推荐用于评测**。`0.7`（老师设置）= 带随机性，同一题多次跑结果会变，准确率会有 ±波动 |
| `top_p` | nucleus 采样，只从累积概率 top-p 的词里采 | **仅在 temperature>0 时生效**。老师用 0.9 |
| `do_sample` | 是否采样 | temperature>0 时为 True；greedy 时为 False |
| `max_new_tokens` | 最大生成长度 | 非思考模式 512~1024 足够；思考模式要 ≥4096（推理链长） |

**为什么评测建议 greedy 而非老师的 0.7**：做「准确率对比」时，随机采样会让 baseline 和各 checkpoint 的数字都带噪声，不利于公平比较。greedy 去掉随机性，差异才反映真实能力。老师 notebook 用 0.7+0.9 是通用默认值，若要严格对齐老师，evaluate.py 加 `--temperature 0.7 --top-p 0.9` 即可（但建议主结果用 greedy，附录可补一组 0.7 的）。

---

## 4. 与老师 notebook 参数对照

老师示例（Llama-3.2-1B + algebra 子集）：

| 项 | 老师值 | 本实验值 | 说明 |
|---|---|---|---|
| lora_rank r | 8 | 8 | 一致 |
| lora_alpha | （peft 默认 8，缩放 1×） | 32（缩放 4×） | 本实验 LoRA 影响更强 |
| lora_dropout | 0.1 | 默认 0.05 | 可对齐为 0.1 |
| target_modules | q/k/v/o/gate/up/down | all-linear | 基本等价 |
| learning_rate | 2e-4 | 1e-4 | 见第 5 节，强基座宜更小 |
| 有效 batch | 8×4=32 | 2×8×4=64 | — |
| epoch | 1 | 3 | 多 epoch 在窄数据上风险更高 |
| max_length | 512 | 2048 | 数学解答可能较长 |
| 评测采样 | T=0.7, top_p=0.9 | 默认 greedy | 见第 3 节 |

> 注意：老师用的是 **1B 弱模型**，微调能明显提升；我们用的是 **4B 强基座**，情况不同（下节）。

---

## 5. ⚠️ 为什么 LoRA 后准确率反而下降了

本实验现象：baseline 58.78% → epoch1 46.96%，且 epoch 越多越差。这是一个**真实且常见**的现象，值得写进报告分析：

**原因**：Qwen3.5-4B 是已经很强的指令/推理模型，本身数学能力就高（58.78%）。在**窄数据集**（只有 MATH 解答）上用**偏高的 lr（1e-4）× 多个 epoch**做 SFT，会让模型**过拟合数据集的解答风格、丢失原有的通用推理能力**——即「灾难性遗忘」。epoch 越多越差，正是过拟合的确凿信号。

**这不是数据量不够的问题**，7500 条对 LoRA 足够；加更多数据（甚至污染 test）都救不了，反而掩盖问题。

**诊断方法（先做这一步）**：打开 `infer_output/pred_*.jsonl`，逐条看微调后模型的 `generation`，判断错因属于哪类：
1. **格式坏了**：不再输出 `\boxed{}`、或啰嗦跑题 → 训练把输出格式带偏了，extract 抓不到答案；
2. **boxed 但答错**：格式对、答案错 → 真实推理能力下降。

**缓解手段**（按优先级，一次改一项再对比）：
1. **降学习率**：`1e-4 → 2e-5 或 1e-5`，强基座最关键的一招；
2. **减 epoch**：先只训 1 个 epoch（甚至更少 step），别 3 个；
3. **降容量**：`lora_rank 8`、`target_modules` 只挂注意力层（`q_proj k_proj v_proj o_proj`），少动 MLP，减少对基座的破坏；
4. 评测确保非思考模式与训练一致（已是）。

目标不是「一定要超过 58.78%」，而是理解并展示「强基座上 LoRA 的得失」，这本身就是高质量的实验分析。

---

## 6. 数据划分：不要把 test 混入 train

针对「从 test 抽一部分当测试，还是把 test 混入 train」的纠结：

**标准且唯一推荐的做法**：数据集本身已经有干净的 `train`(7500) / `test`(5000) 划分——
- **训练**：用全部官方 train（不必单独切验证集，老师也没切；要监控 loss 可从 train 里切一小撮当 val）；
- **评测**：用官方 test。

**不要把 test 的一部分混进 train**：这是**数据泄漏**，会让 test 指标虚高、且与「从没见过任何数据的 baseline」不再可比，报告结论不可信。

**「训练数据更多结果会不会更好」**：在这个任务里不会——瓶颈是超参和强基座，不是数据量（见第 5 节）。想让评测更快，就固定抽 test 的一个**随机子集**（如 1000 条，baseline 和所有 checkpoint 用同一子集）来比较，这完全合规，也是老师做法（他只评 64 条）。

---

## 7. 本实验最终命令速查

```bash
# 训练（4×4090）
bash scripts/sft_lora.sh

# 评测对比（5 卡数据并行，greedy）
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node 5 scripts/evaluate.py \
    --adapters base <ckpt-epoch1> <ckpt-epoch2> <ckpt-epoch3>

# 若要对齐老师采样：追加 --temperature 0.7 --top-p 0.9
# 若要快速对比：追加 --num-samples 1000
```
