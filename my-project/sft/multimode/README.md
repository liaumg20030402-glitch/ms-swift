# Qwen3.5-27B 多模态 SFT 脚本说明

## 1. 这组脚本要验证什么

这四个脚本使用相同的模型、数据、训练超参数和分布式配置，只改变视觉编码器（ViT）与视觉对齐模块（aligner/projector）的可训练范围及学习率，构成一组多模态模块解冻策略的消融实验。

需要特别注意：四个脚本中的 `--freeze_llm false` 完全相同，因此 **LLM 在四组实验中始终参与全量训练**。文件名中的 `aligner` 不是“只训练 aligner”，而是“训练 LLM，并额外训练 aligner”。

| 脚本 | LLM | ViT | Aligner | 学习率 | 主要验证目标 |
| --- | --- | --- | --- | --- | --- |
| `train_20260526_aligner.sh` | 全量训练 | 冻结 | 训练 | LLM `5e-6`；aligner `1e-6` | 只更新 aligner 是否足以让视觉特征适配新的 SFT 目标 |
| `train_20260526_vit_top4.sh` | 全量训练 | 仅 blocks 23–26 | 冻结 | LLM `5e-6`；ViT `1e-7` | 不更新 aligner 时，仅微调 ViT 顶部四层是否有效 |
| `train_20260526_vit4_aligner.sh` | 全量训练 | 仅 blocks 23–26 | 训练 | LLM `5e-6`；ViT `1e-7`；aligner `1e-6` | 顶部四层 ViT 与 aligner 联合微调的效果 |
| `train_20260526_full_unfreeze.sh` | 全量训练 | 全量训练 | 训练 | LLM `5e-6`；ViT `1e-7`；aligner `1e-6` | 全多模态模型解冻的效果、显存成本和稳定性上限 |

这里将 blocks 23–26 称为“顶部四层”，前提是当前模型的 ViT block 编号确实为 0–26。正式实验前应从启动日志或模型参数名确认该假设。

## 2. SWIFT 中冻结和重新解冻的执行顺序

这些脚本运行的是 Megatron-SWIFT 的 full tuning，而不是 LoRA。SWIFT 的处理顺序是：

1. 根据 `freeze_llm`、`freeze_vit` 和 `freeze_aligner` 生成待冻结模块前缀。
2. 先冻结对应参数。
3. 最后执行 `trainable_parameters` 和 `trainable_parameters_regex`，把命中的参数重新设为可训练。

因此下面的组合不是互相冲突：

```bash
--freeze_vit true \
--trainable_parameters_regex "visual\.visual\.blocks\.2[3-6]\."
```

它表示先冻结整个 ViT，再重新解冻参数名中匹配 `visual.visual.blocks.23.`、`24.`、`25.`、`26.` 的参数。SWIFT 使用正则 `search` 匹配参数名，所以参数名前面即使带有 Megatron wrapper 前缀也可以匹配。

相关实现：

- [`_init_multimodal_full`](../../../swift/megatron/arguments/megatron_args.py)：把多模态冻结选项转换为模块前缀。
- [`prepare_mcore_model`](../../../swift/megatron/utils/utils.py)：先冻结参数，再调用 `activate_parameters` 重新解冻指定参数。
- [`activate_parameters`](../../../swift/utils/transformers_utils.py)：使用正则匹配并设置 `requires_grad=True`。
- [Megatron Trainer 参数分组](../../../swift/megatron/trainers/base.py)：按照 LLM、ViT、aligner 分配不同学习率。

## 3. 每个脚本相对其他脚本改了什么

### `train_20260526_aligner.sh`

关键参数：

```bash
--freeze_llm false
--freeze_vit true
--freeze_aligner false
--aligner_lr 1e-6
```

实际可训练范围为 LLM + aligner，整个 ViT 冻结。该实验用于测试在保持视觉编码器不变的情况下，仅让对齐层重新学习视觉特征到语言空间的映射是否足够。

### `train_20260526_vit_top4.sh`

相对 `aligner` 方案，它冻结 aligner，并通过正则重新解冻 ViT blocks 23–26：

```bash
--freeze_vit true
--freeze_aligner true
--vit_lr 1e-7
--trainable_parameters_regex "visual\.visual\.blocks\.2[3-6]\."
```

实际可训练范围为 LLM + ViT blocks 23–26。它用于观察仅调整视觉编码器高层表示、但不改变 aligner 时的收益。

### `train_20260526_vit4_aligner.sh`

相对 `vit_top4`，它把 aligner 解冻并增加独立学习率：

```bash
--freeze_vit true
--freeze_aligner false
--vit_lr 1e-7
--aligner_lr 1e-6
--trainable_parameters_regex "visual\.visual\.blocks\.2[3-6]\."
```

实际可训练范围为 LLM + ViT blocks 23–26 + aligner。这是介于 aligner-only 增量方案和全量解冻之间的折中方案。

### `train_20260526_full_unfreeze.sh`

关键参数：

```bash
--freeze_llm false
--freeze_vit false
--freeze_aligner false
--vit_lr 1e-7
--aligner_lr 1e-6
```

实际可训练范围为 LLM + 全部 ViT + aligner。它是可训练参数最多、显存和优化器状态开销最大的方案，用于验证全量视觉解冻是否比只解冻顶部四层产生额外收益。

## 4. 应如何形成有效对照

建议按以下成对结果判断具体模块的贡献：

| 对照 | 唯一主要变量 | 可以回答的问题 |
| --- | --- | --- |
| `aligner` vs `vit4_aligner` | 是否增加 ViT blocks 23–26 | 在 aligner 都训练时，顶部四层 ViT 是否带来收益 |
| `vit_top4` vs `vit4_aligner` | aligner 是否训练 | 在顶部四层 ViT 都训练时，aligner 是否仍有必要更新 |
| `vit4_aligner` vs `full_unfreeze` | 顶部四层 ViT vs 全部 ViT | 解冻更底层的视觉参数是否值得额外显存和训练成本 |

不能仅比较最终训练 loss 判断多模态能力。至少应在相同的图文理解、OCR、医学图像或目标业务验证集上比较，并同时记录语言任务退化、吞吐和峰值显存。

## 5. 四个脚本共有的训练配置

- 模型：`Qwen3.5-27B`。
- 最大长度：32768。
- TP：4；每节点使用 8 张 GPU。
- `micro_batch_size=2`，`global_batch_size=16`。实际梯度累积步数取决于总卡数和由 Megatron 推导出的 DP 大小。
- 训练 3 个 epoch，不切分验证集：`split_dataset_ratio=0`。
- 开启 `packing`、`padding_free`、FlashAttention 和 sequence parallel。
- 开启 full/uniform activation recompute，每个 recompute block 为 1 层。
- optimizer CPU offload 关闭，因此优化器状态保留在 GPU。
- `no_save_optim=false`、`no_save_rng=false`，会保存优化器和 RNG 状态，可用于继续训练。
- 每个 epoch 保存一次 checkpoint。
- 使用节点本地 `/tmp` 目录存放 ModelScope、Hugging Face 和 Triton 缓存，避免多节点共享缓存竞争。
- NCCL 使用 `eno1` 和指定的 8 个 IB HCA。

## 6. 启动后必须验证的内容

### 6.1 确认参数冻结范围

在日志中检查：

```bash
grep -E "freeze_parameters|additional trainable_parameters|model_parameter_info" train.log
```

预期现象：

- `aligner`：ViT 被列入冻结范围，aligner 被列入额外可训练范围。
- `vit_top4`：ViT 和 aligner 都先被冻结，但模型可训练参数数量应高于纯 LLM 方案。
- `vit4_aligner`：ViT 先被冻结、aligner可训练，且 blocks 23–26 被正则重新激活。
- `full_unfreeze`：ViT 和 aligner 均不应出现在冻结范围。

对于两个正则解冻脚本，还要确认日志里没有下面的警告：

```text
trainable_parameters_regex is provided but no parameters are activated
```

如果出现该警告，说明当前 MCore 模型的参数命名与正则不一致，所谓“顶部四层 ViT”实际上没有被解冻。

### 6.2 确认分组学习率

日志应显示类似：

```text
vit_lr: 1e-07, aligner_lr: 1e-06, llm_lr: 5e-06
```

被冻结模块即使配置了学习率，也不会进入 optimizer 参数组。

### 6.3 确认数据确实包含多模态输入

当前脚本的数据路径是：

```text
/train21/medcog/permanent/zefeng/Data/text/27B_SFT_20260424_17254.jsonl
```

目录名包含 `text`，但仅凭路径不能判断内容。必须检查 JSONL 中是否真的存在有效的 image/video 字段及对应文件。如果数据实际是纯文本，ViT 和 aligner 通常不会参与前向计算，也不会获得有效梯度；此时四组脚本无法验证不同视觉解冻策略，只能验证语言模型训练和不同配置的启动/显存行为。

### 6.4 验证集问题

四个脚本都设置了 `split_dataset_ratio=0`，因此训练过程没有内部验证集。若实验目的是比较哪种多模态解冻策略更好，必须准备完全相同的外部多模态评测集，否则训练 loss 的差异不足以支撑结论。

## 7. 当前脚本中值得修正或确认的地方

`PYTORCH_CUDA_ALLOC_CONF` 当前只是 shell 变量，没有导出：

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

如果希望 Python/PyTorch 子进程读取它，应改为：

```bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

另外，四个脚本中的中文注释出现了乱码，不影响执行，但建议统一保存为 UTF-8，避免后续维护时误解数据段落含义。

## 8. 建议最终记录的实验结果

每组至少记录：

- 实际可训练参数量及占总参数比例。
- ViT、aligner、LLM 的实际学习率。
- 峰值 GPU 显存、单 step 时间和吞吐。
- 是否出现无梯度参数、OOM、NaN 或 loss spike。
- 相同多模态验证集上的核心指标。
- 相同纯文本验证集上的指标，用于判断多模态微调是否损害语言能力。

只有在数据、随机种子、总卡数、batch size、训练步数和评测流程一致时，这四组结果才构成有效的消融实验。
