# Megatron-SWIFT 预训练参数说明

## 结论：能否把 `megatron sft` 直接换成 `megatron pt`

可以复用大部分训练和并行参数，但不能只替换命令后完全不检查数据与参数。

从 SWIFT 源码看，`MegatronPretrain` 直接继承 `MegatronSft`，因此模型加载、Megatron 并行、优化器、checkpoint 和训练循环基本相同。PT 专用参数类主要修改了两个默认值：

```python
use_chat_template = False
loss_scale = 'all'
```

也就是说：

- `megatron sft` 使用对话模板，通常只对 assistant 回复计算目标损失。
- `megatron pt` 使用 generation template，并对预训练文本的全部 token 计算损失。

相关源码：

- [`pretrain.py`](../../swift/megatron/pipelines/train/pretrain.py)
- [`pretrain_args.py`](../../swift/megatron/arguments/pretrain_args.py)
- [`sft.py`](../../swift/megatron/pipelines/train/sft.py)

如果原来的数据已经是标准预训练格式，并删除或修改 SFT 专用参数，那么把入口改为 `megatron pt` 是可行的。原始 SFT 对话数据不能在不确认训练目标的情况下直接当成预训练数据使用。

## 预训练数据格式

纯文本预训练推荐每条 JSONL 只有一条 assistant 消息：

```jsonl
{"messages": [{"role": "assistant", "content": "这里是一段用于预训练的连续文本。"}]}
{"messages": [{"role": "assistant", "content": "这是下一篇文档或下一段语料。"}]}
```

不要默认把下面这种 SFT 数据原样用于 PT：

```jsonl
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

`megatron pt` 默认 `loss_scale=all`，这意味着输入中的所有文本 token 都可能参与训练目标，而不再是典型 SFT 的“只监督回答部分”。

官方格式参考：[`Custom-dataset.md`](../../docs/source/Customization/Custom-dataset.md)。

## Streaming 数据是什么

Streaming 不是一种新的数据格式，而是一种数据加载方式。同一个 JSONL、本地分片目录或 Hugging Face/ModelScope 数据集，既可以普通加载，也可以通过下面的参数按流读取：

    --streaming true

普通模式通常先建立一个有确定长度、支持随机访问的数据集；streaming 模式返回可迭代数据集（IterableDataset），训练过程中边读取、边预处理、边消费样本，不需要先把完整语料物化到内存或本地 Arrow 文件。

| 项目 | 普通数据集 | Streaming 数据集 |
| --- | --- | --- |
| 数据格式 | JSONL等标准格式 | 格式相同，不是另一种JSONL |
| 数据长度 | 通常已知 | 通常未知 |
| 随机访问 | 支持 | 不支持，只能向前迭代 |
| Shuffle | 可以全局打乱 | 使用有限buffer近似打乱 |
| Epoch | 有明确语义 | 通常没有可靠的epoch语义 |
| 验证集比例切分 | 支持 | SWIFT会把比例强制改为0 |
| 适用场景 | 能完成索引的中小数据 | 数百GB/TB级、多分片或远程语料 |

### 预训练必须使用 Streaming 吗

不必须。megatron pt 决定使用预训练模板和全token loss；streaming只决定数据如何加载，两者相互独立。

数据量不大、能够快速完成索引时可以关闭：

    megatron pt \
        --dataset /path/pt.jsonl \
        --streaming false \
        --num_train_epochs 1

数百GB/TB级语料、大量shard或远程数据通常更适合开启：

    megatron pt \
        --dataset /path/pt_data \
        --streaming true \
        --train_iters 10000

当前项目的 pt_train_27b.sh 设置了 streaming=true，所以当前脚本会按流式方式继续预训练；但这不是Qwen3.5或预训练任务本身的硬性要求。

### SWIFT 开启 Streaming 后的具体行为

#### 1. 使用 step 控制训练和保存

Streaming数据通常没有长度，不能可靠地从num_train_epochs推导总步数，应显式设置：

    --train_iters 10000
    --save_steps 500

不能使用：

    --save_strategy epoch

否则源码会报错：

    streaming dataset is not supported with --save_strategy epoch

#### 2. 不能按比例自动切分验证集

同时设置streaming=true和split_dataset_ratio大于0时，SWIFT会自动把split_dataset_ratio改成0。需要验证时，应单独准备：

    --val_dataset /path/to/val.jsonl

如果验证集本身也是streaming数据，还应显式设置eval_iters，因为框架无法通过验证集长度推导评测步数。

#### 3. Shuffle 是有限buffer近似打乱

Streaming无法先读取全部样本再进行全局shuffle。SWIFT使用shuffle_buffer_size控制随机缓冲区，默认值为1000：

    --shuffle_buffer_size 1000

buffer越大，随机性通常越好，但会消耗更多CPU内存。如果预训练语料按领域、来源或时间连续排列，默认1000可能不足，应结合单条样本大小和主存容量调整。

#### 4. dataset_num_proc 不按普通模式工作

SWIFT的loader在streaming模式下会把预处理num_proc设置为None。因此脚本即使写了：

    --dataset_num_proc 8

也不会像普通数据集那样使用8个进程预处理完整数据集。

#### 5. Megatron dataloader worker 会被改为1

当前Megatron-SWIFT在streaming模式下会把大于1的dataloader_num_workers强制改为1。例如脚本设置：

    --dataloader_num_workers 4

实际运行会变成1，并打印：

    Using streaming dataset, setting args.dataloader_num_workers to 1.

这是为了避免多个worker重复消费IterableDataset或产生难以管理的迭代状态。因此当前pt_train_27b.sh中填写的4最终不会生效。

#### 6. Packing 仍然可以使用

Streaming与下面两个参数兼容：

    --packing true
    --padding_free true

SWIFT会在迭代读取数据时持续把短样本装入训练序列。纯文本PT也可以使用：

    --truncation_strategy split

将超过max_length的长文档拆成多个样本。该split策略不适用于多模态数据。

### 当前27B任务如何选择

- 单个或少量JSONL，能够快速完成索引：建议先关闭streaming，便于确认样本数、epoch、验证集切分和数据顺序。
- 数百GB/TB级语料、大量shard：建议开启streaming，使用train_iters和save_steps，并提供独立验证集。
- 48卡正式训练前，应先运行几十到几百step，确认各rank没有明显重复读取，且单worker数据处理没有限制GPU吞吐。

相关源码：

- [data_args.py](../../swift/arguments/base_args/data_args.py)：streaming参数、buffer shuffle和验证集切分。
- [loader.py](../../swift/dataset/loader.py)：IterableDataset以及streaming时禁用num_proc。
- [megatron_base_args.py](../../swift/megatron/arguments/megatron_base_args.py)：streaming时将dataloader worker调整为1。
- [megatron_args.py](../../swift/megatron/arguments/megatron_args.py)：train_iters、epoch保存和streaming验证集限制。

## 严格意义上的必需参数

### 1. 模型来源

基于已有 Qwen3.5-27B 权重继续预训练时需要：

```bash
--model /path/to/Qwen3.5-27B
```

SWIFT 源码要求提供 `--model` 或 `--mcore_model`。如果两者都不提供，则必须显式使用 `--perform_initialization true`，这属于随机初始化训练，不是当前脚本采用的继续预训练方式。

### 2. 训练数据

必须提供以下两者之一：

```bash
--dataset /path/to/pt.jsonl
```

或者：

```bash
--cached_dataset /path/to/cached_dataset
```

如果两者都没有，`MegatronSftArguments` 会直接报错 `Please input the training dataset`。

### 3. 训练时长

需要明确指定以下一种：

```bash
--train_iters 10000
```

或者在非 streaming 数据集上使用：

```bash
--num_train_epochs 1
```

如果设置了：

```bash
--streaming true
```

就应该显式提供 `--train_iters`。Streaming 数据集没有可靠的长度，SWIFT 无法通过 epoch 自动计算总训练步数。

### 4. 合法的 batch 和并行关系

参数虽然有默认值，但必须满足：

```text
world_size = TP × PP × CP × DP
global_batch_size % (micro_batch_size × DP) == 0
```

当前48卡文本预训练脚本使用：

```text
48 = TP4 × PP1 × CP2 × DP6
gradient_accumulation_steps = 48 / (1 × 6) = 8
```

对应参数：

```bash
--tensor_model_parallel_size 4
--pipeline_model_parallel_size 1
--context_parallel_size 2
--micro_batch_size 1
--global_batch_size 48
```

## 实际训练中应显式设置的参数

这些参数不一定是命令行解析器强制要求的，但真实预训练不应依赖隐式默认值。

### 输出目录

```bash
--output_dir /path/to/output
```

未设置时 SWIFT 会生成默认目录，但生产任务应明确指定，防止多个实验写入混乱位置。


### 学习率

```bash
--lr 1e-5
--lr_warmup_iters 300
--min_lr 1e-6
```

源码中的 `lr` 默认是 `None`，因此建议必须显式设置。默认衰减方式为 cosine，默认 Adam 参数为：

```text
weight_decay = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_eps = 1e-8
```

如果需要严格复现实验，也应把这些值写进脚本。

### Checkpoint

```bash
--save_steps 500
--save_safetensors true
--no_save_optim false
--no_save_rng false
```

长期预训练应该保存 optimizer 和 RNG，否则只能恢复模型权重，不能严格恢复学习率、优化器动量和数据随机状态。

## 从 `sft_train.sh` 迁移时如何处理参数

### 可以直接保留

以下参数同时适用于 SFT 和 PT：

```bash
--model
--dataset
--output_dir
--save_safetensors
--split_dataset_ratio
--load_from_cache_file
--tensor_model_parallel_size
--pipeline_model_parallel_size
--context_parallel_size
--micro_batch_size
--global_batch_size
--packing
--padding_free
--cross_entropy_loss_fusion
--lr
--lr_warmup_iters / --lr_warmup_fraction
--min_lr
--max_length
--sequence_parallel
--attention_backend
--recompute_granularity
--recompute_method
--recompute_num_layers
--optimizer_cpu_offload
--save_steps
```

### 应删除或修改

#### `--add_non_thinking_prefix true`

这是对话/SFT相关处理。纯文本 PT 不需要添加 non-thinking 前缀，应删除。

#### `--loss_scale ignore_empty_think`

这是针对带 `<think>` 对话输出的 SFT loss 规则。纯文本 PT 应删除，让 `megatron pt` 使用默认值：

```bash
--loss_scale all
```

通常不必显式写 `--loss_scale all`。

#### `--num_train_epochs 3`

非 streaming 数据可以保留，但大规模预训练通常使用：

```bash
--streaming true
--train_iters 10000
```

此时应删除 `--num_train_epochs`，由 `train_iters` 唯一控制训练时长。

#### `--save_strategy epoch`

Streaming 数据不支持按 epoch 保存，源码会直接报错。应改成：

```bash
--save_steps 500
```

### 纯文本 Qwen3.5 应保留的冻结参数

Qwen3.5-27B包含视觉模块。纯文本继续预训练时没有图像输入，建议：

```bash
--freeze_llm false
--freeze_vit true
--freeze_aligner true
```

这样只训练语言模型，避免为没有梯度的视觉模块创建不必要的优化器状态。

### PT 专用的超长文本切分

纯文本 PT 可以使用：

```bash
--truncation_strategy split
```

它会把超过 `max_length` 的长文本拆成多个训练样本，减少直接删除长文档造成的 token 浪费。该模式只适用于纯文本预训练，不适用于多模态数据。

## `--finetune true` 的含义

当前任务是从已有 Qwen3.5-27B 权重开始做 continued pre-training，因此保留：

```bash
--finetune true
```

这里的 `finetune` 不会把任务变回 SFT；任务究竟是 PT 还是 SFT，由命令入口 `megatron pt`、template 和 loss 规则决定。

## 推荐的纯文本 PT 命令主体

```bash
megatron pt \
    --dataset ${DATASETS[@]} \
    --split_dataset_ratio 0 \
    --model $MODEL_PATH \
    --output_dir $output_dir \
    --save_safetensors true \
    --load_from_cache_file true \
    --streaming true \
    --tuner_type full \
    --finetune true \
    --torch_dtype bfloat16 \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 48 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --padding_free true \
    --truncation_strategy split \
    --train_iters 10000 \
    --cross_entropy_loss_fusion true \
    --apply_wd_to_qk_layernorm true \
    --lr 1e-5 \
    --lr_warmup_iters 300 \
    --min_lr 1e-6 \
    --max_length 32768 \
    --sequence_parallel true \
    --attention_backend flash \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --save_steps 500 \
    --no_save_optim false \
    --no_save_rng false
```

官方基础示例：[`examples/megatron/pretrain.sh`](../../examples/megatron/pretrain.sh)。当前项目完整脚本：[`pt_train_27b.sh`](./pt_train_27b.sh)。

## Streaming 能否确定完整一遍需要多少 Step

Streaming模式不是完全无法计算step，而是SWIFT无法通过IterableDataset的len自动推导。可以通过离线统计token数量后自行计算：

    一遍数据的step数
      ≈ 总有效token数
        / (global_batch_size × max_length × packing利用率)

例如总量约780亿token，配置为：

    global_batch_size = 96
    max_length = 8192
    packing利用率 ≈ 0.95

则：

    steps ≈ 78,000,000,000 / (96 × 8192 × 0.95)
          ≈ 104,000

这里必须使用tokenizer统计后的token数。数据报告中的“平均长度”可能是字符数或字节数，不能直接当作精确token数。

Streaming训练达到train_iters后停止。如果数据流先耗尽，SWIFT的cyclic_iter会重新打开数据流继续训练，因此train_iters大于一遍数据所需step时会重复消费数据。之前日志中不断出现Epoch starts就是数据流快速耗尽并被反复重启。

如果需要框架自动根据数据长度计算一遍，应使用有len的普通或cached dataset，并设置：

    --streaming false
    --num_train_epochs 1

## 200GB数据是否只能使用 Streaming

不是。200GB数据有三种主要方案。

| 方案 | 是否提前完整预处理 | 能否获得确定长度 | 额外磁盘 | 适用场景 |
| --- | --- | --- | --- | --- |
| streaming=true | 否 | 不能由框架自动获得 | 较少 | 快速启动、磁盘不足、数据持续增加 |
| 普通dataset + load_from_cache_file | 首次运行时生成HF缓存 | 可以 | 较多 | 本地磁盘充足，希望重复实验 |
| 显式cached_dataset | 使用swift export离线导出 | 可以 | 较多 | 正式大规模训练、训练与预处理解耦 |

Hugging Face普通Dataset通常使用Arrow内存映射，并不要求把完整200GB同时放进CPU内存，但首次索引、预处理和缓存会消耗大量时间、磁盘以及文件系统I/O。对于48卡任务，如果每次启动都重新tokenize，容易浪费昂贵的GPU时间。

### load_from_cache_file 的作用

下面的参数：

    --load_from_cache_file true

表示复用Hugging Face datasets已有的预处理cache。它不是要求把所有数据加载进内存，也不等于SWIFT显式的cached_dataset。

- 在非streaming模式下，它可以避免相同数据和相同预处理配置被重复处理。
- 在streaming模式下，不会提前物化一个完整、可随机访问且有确定长度的数据集，因此不能解决自动计算epoch/step的问题。
- 多节点使用时，cache目录需要所有节点可访问，或者每个节点拥有一致的本地cache，否则可能重复构建。

### 显式导出 Cached Dataset

正式训练可以先在CPU任务中执行一次：

    swift export \
        --model /path/to/Qwen3.5-27B \
        --dataset /path/to/file1.jsonl /path/to/file2.jsonl \
        --to_cached_dataset true \
        --use_chat_template false \
        --loss_scale all \
        --truncation_strategy split \
        --max_length 8192 \
        --split_dataset_ratio 0 \
        --dataset_num_proc 64 \
        --output_dir /path/to/qwen3_5_27b_pt_cache

训练时不再使用原始dataset：

    megatron pt \
        --model /path/to/Qwen3.5-27B \
        --cached_dataset /path/to/qwen3_5_27b_pt_cache/train \
        --streaming false \
        --truncation_strategy split \
        --max_length 8192 \
        --num_train_epochs 1

使用truncation_strategy=split导出缓存时，导出和训练必须保持相同的max_length和truncation_strategy。该模式会保存input_ids，缓存可能显著增大；建议先对一小部分数据导出，测量缓存与原始数据的空间比例，再处理全部200GB。

对于当前任务的推荐顺序：

1. 为快速验证训练链路，先使用streaming跑200到500 step。
2. 确认数据质量和参数后，在独立CPU任务中导出cached dataset。
3. 正式训练使用cached_dataset和streaming=false，从而获得稳定的数据长度、shuffle和epoch语义。

官方示例：[cached_dataset/pretrained.sh](../../examples/train/cached_dataset/pretrained.sh)。

## lr_warmup_fraction 与 lr_warmup_iters

可以把：

    --lr_warmup_iters 300

替换为：

    --lr_warmup_fraction 0.05

二者的区别是：

- lr_warmup_iters：固定预热多少个optimizer iteration。
- lr_warmup_fraction：预热步数占整个lr decay阶段的比例。

当前源码中，如果lr_decay_iters未设置，它默认等于train_iters，因此近似关系是：

    warmup iteration = lr_warmup_fraction × train_iters

例如：

| train_iters | warmup_fraction | 实际warmup |
| ---: | ---: | ---: |
| 10000 | 0.05 | 500 step |
| 50000 | 0.05 | 2500 step |
| 100000 | 0.05 | 5000 step |

如果希望10000步训练仍然只warmup 300步，应使用：

    --lr_warmup_fraction 0.03

或者继续使用：

    --lr_warmup_iters 300

如果两个参数同时设置，源码优先使用lr_warmup_fraction，lr_warmup_iters不会生效。对于训练总步数可能调整的实验，fraction更方便；对于短期CPT或希望严格控制预热长度的实验，固定iters更明确。

## apply_wd_to_qk_layernorm 的作用

Qwen3.5的attention中包含Q/K LayerNorm参数，例如：

    q_layernorm.weight
    k_layernorm.weight

Megatron默认不对bias和一维Norm参数应用weight decay。设置：

    --apply_wd_to_qk_layernorm true

后，SWIFT会把Qwen3.5的q_layernorm和k_layernorm从“无权重衰减”参数组移到正常weight decay参数组，使其使用脚本中的weight_decay，默认值为0.1。

该参数：

- 只支持qwen3_next、qwen3_5和qwen3_5_moe。
- 主要用于Qwen3.5全参数训练或继续预训练，以对齐其预训练优化器分组方式。
- 不会开启新的模型结构，也不会明显改变显存。
- 不等于对所有LayerNorm应用weight decay，只针对Q/K LayerNorm。
- LoRA训练或冻结LLM时通常没有必要开启。

当前任务是Qwen3.5-27B全参数文本CPT，建议保留：

    --apply_wd_to_qk_layernorm true

相关源码：[base.py](../../swift/megatron/trainers/base.py) 和 [megatron_args.py](../../swift/megatron/arguments/megatron_args.py)。

## 启动前检查清单

1. 数据是否是一条 assistant message 的预训练格式，而不是未确认目标的 SFT 对话格式。
2. `megatron pt` 是否已替换 `megatron sft`。
3. 是否删除 `add_non_thinking_prefix` 和 `loss_scale ignore_empty_think`。
4. Streaming 时是否使用 `train_iters` 和 `save_steps`，而不是 epoch。
5. `global_batch_size` 是否能被 `micro_batch_size × DP` 整除。
6. 纯文本任务是否冻结 ViT 和 aligner。
7. 是否保存 optimizer/RNG，以满足真正的断点续训需求。
8. Qwen3.5 packing/CP 是否使用 MCore GDN，环境中不能保留 `USE_MCORE_GDN=0`。
