# `sft_train_397b.sh` 参数详解（Qwen3.5-397B-A17B MoE / Megatron 后端）

本笔记对应脚本 [../sft/sft_train_397b.sh](../sft/sft_train_397b.sh)，逐项解释**每一个**参数（含环境变量与 `megatron sft` 训练参数）。

和 27B 脚本的关系：基础参数与 [sft_train_参数详解.md](./sft_train_参数详解.md) 一致，**本文重点讲 397B 这种大型 MoE 多出来的部分**：MoE 专家并行、流水线并行（PP）、优化器 CPU offload。通用参数也一并收录，便于单文件查阅。

> 一句话背景：397B-A17B 是 **3970 亿总参 / 激活约 170 亿** 的 MoE 模型，必须用 Megatron 后端 + 多维并行才能训。

---

## 零、环境变量（脚本顶部）

### `source .bashrc` / `conda activate swift_sft_py312`
加载用户环境、激活训练用的 conda 环境（python3.12，FLA 要求 3.12）。

### 缓存目录
```bash
tmp_cache=/tmp/jycai6_swift_cache_$(hostname)_${RANK:-0}
export MODELSCOPE_CACHE=$tmp_cache/modelscope   # ModelScope 模型/数据缓存
export HF_HOME=$tmp_cache/huggingface           # HuggingFace 缓存(datasets/tokenizer)
export TRITON_CACHE_DIR=$tmp_cache/triton_cache # Triton 编译内核缓存
```
用 `hostname + RANK` 拼路径：**多机多进程各自独立缓存**，避免不同节点抢同一份 cache 造成冲突。

### `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`
PyTorch 显存分配器用「可扩展段」，**减少显存碎片**、缓解 OOM。大模型长序列必备。

### GPU 与分布式拓扑
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 本节点用 8 张卡
export NPROC_PER_NODE=8                        # 每节点起 8 个进程(1 卡 1 进程)
export NNODES=$WORLD_SIZE                       # 总节点数(由调度器注入)
export NODE_RANK=$RANK                          # 本节点编号
export MASTER_ADDR=$MASTER_ADDR                 # rank0 地址，集合通信汇合点
export MASTER_PORT=$MASTER_PORT                 # rank0 端口
```
总卡数 `world_size = NNODES × NPROC_PER_NODE`。**每个参与节点都必须是 8 卡 GPU 机**，混入无卡机会报 `ProcessGroupNCCL ... no GPUs found`。

### 网络后端（NCCL / GLOO over InfiniBand）
```bash
export GLOO_SOCKET_IFNAME=eno1   # GLOO(控制面)走的网卡
export NCCL_SOCKET_IFNAME=eno1   # NCCL 建链用的网卡
export NCCL_NET=IB               # 数据面走 InfiniBand
export NCCL_IB_HCA=mlx5_0..7     # 指定 8 张 IB 网卡(和 8 GPU 对应)
export NCCL_IB_DISABLE=0         # 启用 IB(0=不禁用)
export NCCL_P2P_DISABLE=0        # 启用 GPU 间 P2P(NVLink)
```
多机大模型的通信量极大，**IB + NVLink 必须开**，否则 TP/EP 的 all-reduce/all-to-all 会成为瓶颈。

### 其它
```bash
export CUDA_LAUNCH_BLOCKING=0            # 0=异步 kernel(正常训练)，调试时设 1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # NCCL 出错时异步抛错，避免整体卡死
```

---

## 一、路径变量

| 变量 | 含义 |
|---|---|
| `MODEL_PATH` | 模型权重目录（397B 的 safetensors，HF 格式；megatron 会在线转 mcore） |
| `DATASETS` | 训练数据 JSONL（bash 数组，可多个） |
| `output_dir` | 输出目录（ckpt + 日志） |

---

## 二、数据相关

### `--dataset $DATASETS`
训练数据。脚本里是预处理好的医疗 SFT JSONL。

### `--split_dataset_ratio=0`
从训练集切多少比例做验证集。`=0` = **不切 val**，配合 `--save_strategy epoch` 整轮存 ckpt、靠下游评测判优劣。

### `--load_from_cache_file true`
复用磁盘上 HuggingFace `datasets` 的 arrow 缓存，避免每次重复预处理。

### `--dataloader_num_workers 2`  ← 你问的
**DataLoader 取数据的子进程数（每个训练进程各起这么多）**。
- 训练时常驻：每张卡 1 个训练进程 × `num_workers` 个取数子进程。8 卡/节点 × 2 = 16 个/节点。
- 调大：取数更快、不易让 GPU 等数据；但每个 worker 都吃**主机内存**（预取的 batch 缓存）。
- **本脚本设 2 的原因**：397B 把优化器状态 offload 到 CPU 已占用大量主机内存，dataloader worker 太多会一起把 RAM 撑爆（`DataLoader worker killed by signal: Killed` 就是主机 OOM）。所以从 8 降到 2，给 offload 让出内存。

### `--dataset_num_proc 16`  ← 你问的
**预处理阶段 `datasets.map` 的并发进程数**（tokenize / 打包 / 算长度时用）。
- 只在**预处理**时起作用，训练跑起来后这些进程就退出了，**不占训练期显存/常驻内存**。
- 调大 = 预处理更快，但每个进程都加载 tokenizer + 数据分片，**预处理瞬间**会吃不少 RAM 和 CPU。
- 从 196 降到 16：196 个进程在 397B 这种场景下，预处理阶段也可能把 RAM 顶爆；16 更稳。数据已缓存时（`load_from_cache_file`）这步基本秒过。

> 一句话区分：`dataloader_num_workers` 是**训练时**取 batch 的常驻进程；`dataset_num_proc` 是**预处理时**一次性的并发进程。两者都吃内存，但生命周期完全不同。

---

## 三、模型加载与冻结

### `--model $MODEL_PATH` / `--save_safetensors true`
加载权重；保存时输出 safetensors，方便 vLLM/SGLang 下游推理直接读。

### `--finetune true`
把 `$MODEL_PATH` 当**预训练权重**：只加载权重，不加载 optimizer/lr/RNG，从 step0 起。（`false` 才是断点续训恢复完整状态。）

### `--freeze_llm false` / `--freeze_vit true` / `--freeze_aligner true`
多模态模型三段式冻结开关：`llm` 主干**训练**，`vit` 视觉编码器、`aligner` 投影层冻结。397B 这里若是纯文本则 vit/aligner 无实体，留着是模板写法，不影响。

---

## 四、并行策略（397B 的核心，相比 27B 新增 PP + 专家并行）

> 整除关系（任一不满足都会启动即报错）：
> 1. `world_size = TP × PP × CP x DP`（DP 自动推导）
> 2. ETP=1 时 `EP` 必须整除 `TP × DP`
> 3. `PP` 必须整除模型层数（397B=60 层，122B=48 层，27B=64层）
> 4. `EP` 必须整除专家总数（**397B=512，122B=256**）
> 5. `TP` 必须整除 `num_query_groups`（**397B/122B 实测=2 ⇒ TP 只能取 2**，新环境可以取4，解除了限制）

### `--tensor_model_parallel_size 2`（TP）
**张量并行**：把单层的权重矩阵切到 2 张卡上并行做矩阵乘。通信最重（每层 all-reduce），**只放节点内走 NVLink**。这里被 `num_query_groups=2` 限制，TP 最大只能 2。

### `--pipeline_model_parallel_size 4`（PP）★新增
**流水线并行**：把模型**按层切成 4 段**，不同段放不同卡，像流水线一样逐 micro-batch 传递。
- 作用：① 再砍一刀每卡显存（参数/梯度/激活都只剩 1/PP）；② 通信只在层边界传一点激活值（点对点、轻），**适合跨节点摊开大模型**。
- 代价：流水线气泡（填充/排空有空转），需 `micro-batch 数(=梯度累积) ≫ PP`；层数须被 PP 整除。

### `--expert_model_parallel_size 8`（EP）★新增
**专家并行**：把 MoE 的**专家分到 8 张卡**上（397B 的显存大头就是专家）。EP 越大、专家切得越细、每卡专家参数越少。
> 教训：EP 调小（如 8→4）会让每卡专家参数翻倍，**反而更容易 GPU OOM**。

### `--expert_tensor_parallel_size 1`（ETP）★新增
专家内部**不再做张量切分**。ETP=1 时整除关系简化为「EP 整除 TP×DP」。

### `--sequence_parallel true`（SP）
和 TP 配套，把 LayerNorm/Dropout 等非矩阵乘算子按序列维切分，**进一步省激活显存**。TP=1 时自动失效。

### `--micro_batch_size 1` / `--global_batch_size 64`
- `micro_batch_size`：单次 forward 的样本数。
- `global_batch_size`：一个 optimizer step 的总样本数。
- 关系：`global = micro × DP × 梯度累积`。本配置 64 卡：DP=8 ⇒ 梯度累积 = 64/(1×8) = 8（也=喂进 PP 的 micro-batch 数，需 ≫ PP=4，气泡可忽略）。
- **改卡数注意**：`global_batch_size` 必须能被 DP 整除（56 卡 DP=14 时得用 56）。

### `--packing true` / `--padding_free true`
长序列效率优化：`packing` 把多条短样本拼成一条满长度（因果掩码隔离）；`padding_free` 运行时去除 padding。megatron 后端允许同开，packing 主导。

---

## 五、MoE 专属优化（27B dense 模型没有这些）★全部新增

### `--moe_permute_fusion true`
融合 MoE 的 **token 重排(permute)** 算子——token 按路由分发到各专家前要重排，融合后省显存、提速。

### `--moe_grouped_gemm true`
**分组 GEMM**：把多个专家的矩阵乘合批成一次 grouped GEMM，避免逐专家小 kernel，显著提升吞吐。

### `--moe_shared_expert_overlap true`
**共享专家**的计算与（专家间 all-to-all）通信**重叠**，把通信藏到计算后面，提速。

### `--moe_aux_loss_coeff 1e-6`
**负载均衡辅助损失**的权重。MoE 路由器容易把 token 都送给少数专家（路由塌缩），这个 aux loss 惩罚不均衡。`1e-6` 是个很小的正则，既均衡又不干扰主损失。

### `--moe_expert_capacity_factor 2`
**每个专家的容量上限 = 2 × 平均负载**。超出容量的 token 被丢弃（不参与该专家计算）。作用：**给显存封顶**——否则某专家被路由到过多 token 会显存尖峰。设 2 是吞吐与质量的常见折中。

- **默认值 = `None`（dropless）**：源码 `megatron_args.py:534`。不设此参数即 dropless——不丢 token、不封顶，质量更好，均衡时显存反而更省（≈1×均值），但不均衡的 batch 可能显存尖峰。
- **何时可去掉走 dropless**：当 GPU 有明显余量时（实测 16k + PP=6 + offload=1.0 峰值仅 ~92GB/140GB，余 ~48GB），可去掉换「不丢 token」的质量收益；去掉后盯前 ~300-500 step 的 nvidia-smi 峰值，稳定 <125GB 即安全，往 140 爬就加回 `=2`（或给宽松上限如 `=4` 留安全阀）。
- **何时必须保留**：显存紧时（如 8k 时峰值 107GB，或上 32k 余量缩小）务必留 `=2`，否则尖峰会 OOM。**结论绑定具体余量，不是一刀切。**

---

## 六、优化器与学习率

### `--lr 5e-6` / `--min_lr 5e-7` / `--lr_warmup_fraction 0.05`
峰值学习率 / cosine 末端最小 lr / 前 5% 线性 warmup。5e-6 是大模型全参微调的典型量级。

### `--num_train_epochs 3`
跑 3 个 epoch，megatron 内部换算成 `train_iters`。

### `--cross_entropy_loss_fusion true`
把末端 logits 的 CE 与 matmul 融合成单 kernel，避免物化巨大的 vocab logits 张量，**省显存 + 提速**。

### `--loss_scale ignore_empty_think`
对空的 `<think>\n\n</think>` 块**不计 loss**（配合 `add_non_thinking_prefix`，避免模型把"输出空 think"当套路学）。详见 [27B 笔记第五节](./sft_train_参数详解.md)。

### `--add_non_thinking_prefix true`
对 assistant 不以 `<think>` 开头的样本自动补空 think 前缀，使同一数据集能混合训练 thinking / non-thinking 两种模式（Qwen3 混合思考的训练侧依据）。

---

## 七、显存优化：优化器 CPU offload（397B 关键）★新增/改动

> 全参 397B + Adam 的状态：每参数 16 字节 ⇒ 约 **6.35TB**（其中优化器状态 fp32 master+m+v ≈ 4.76TB）。单靠 GPU 装不下，必须把优化器状态卸到 CPU。

### `--optimizer_cpu_offload true`
把**优化器状态卸载到主机内存（CPU RAM）**，GPU 只留参数+梯度+激活。27B 这里是 `false`（够用就别 offload，offload 会拖慢）；397B 太大，32~48 卡时**必须开**。

### `--use_precision_aware_optimizer true`
**精度感知优化器**：offload + 低精度存储下仍保证数值精度（master 权重用 fp32 的逻辑不被破坏）。配合 cpu_offload 使用。

### `--optimizer_offload_fraction 0.5`
优化器状态**下放到 CPU 的比例**（0~1）。这是个 **CPU↔GPU 跷跷板**：
- 调高（→1.0）：CPU 吃得多、GPU 省 → 易 **主机 OOM**（`worker killed by signal: Killed`）。
- 调低（→0）：GPU 吃得多、CPU 省 → 易 **GPU OOM**（`CUDA out of memory`）。
- 卡多（如 64 卡）时每卡参数少、GPU 有余量，可调低甚至关 offload 提速；卡少时只能往中间凑。`0.5` 是 64 卡的折中起点。

### `--no_save_optim false` / `--no_save_rng false`
双否定 = **保存** optimizer 与 RNG 状态，便于精确续训（代价：ckpt 更大）。

---

## 八、其它性能/显存

### `--attention_backend flash`
FlashAttention 后端，长序列必开。

### `--recompute_granularity full` / `--recompute_method uniform` / `--recompute_num_layers 1`
**激活重计算（gradient checkpointing）**，用算力换显存：
- `full`：重算整个 transformer block（最省显存）；
- `uniform` + `num_layers 1`：每 1 层重算一次（=全部重算，最省）。

397B + 32K 序列不开 full recompute 必爆显存。

### `--max_length 32768`
单样本/单条打包序列最长 32K token。回到你的疑惑:max length从8k到16kstep 数为什么从 2308 变 1309
根因是 --packing true。 你的公式「数据总量 / global_batch」本身没错,但**「数据总量」不是你的原始样本条数,而是 packing 之后的序列条数**。

packing 会把多条短样本拼成一条 max_length 长的序列,所以拼出来的序列条数取决于 max_length:


packing 后序列数 ≈ 总 token 数 / max_length
每 epoch step 数 = packing 后序列数 / global_batch_size
总 step 数 = 每 epoch step 数 × epochs
max_length 翻倍(8k→16k)→ 每条序列装下 2× 的 token → 序列条数砍半 → step 数砍半。 global_batch 还是 64,没变,变的是「打包后的有效样本数」。

8k:2308 steps
16k:1309 steps(≈ 砍半,但不是精确 ÷2)
为什么不是精确的 1154:packing 不是完美装箱——8k 和 16k 的装箱效率、边角余料、超长样本的切分方式都不同,所以 2308/1309≈1.76,而非 2.0。正常现象。

关键:你并没有「少训」
虽然 step 数减半,但每个 step 处理的 token 翻倍了:

8k	16k
每序列 token	8192	16384
每 step token	64×8k=512k	64×16k=1M
step 数	2308	1309
总 token(=数据集)	一样	一样
所以:同一份数据、训的总 token 量不变,只是「装箱方式」不同。每 step 算得更多但步数更少,单 epoch 的总计算量/墙钟时间基本不变。不用担心 16k 训得比 8k 少。

对比:如果关掉 packing(只 padding_free 或都不开),step 数 = 原始样本条数 / global_batch,就不会随 max_length 变了。是 packing 把 step 数和 max_length 绑在了一起。

### `--logging_steps 1`
每 step 打一次 log（调试期 1，稳定后可调大）。

### `--save_strategy epoch`
每个 epoch 末存一次 ckpt。

---

## 九、相比 27B 脚本新增/改动一览（★）

| 类别 | 参数 | 说明 |
|---|---|---|
| 并行 | `--pipeline_model_parallel_size` | 按层切流水线，降每卡显存、跨节点摊开 |
| 并行 | `--expert_model_parallel_size` | 专家分卡，397B 显存大头 |
| 并行 | `--expert_tensor_parallel_size` | 专家内不再张量切分 |
| MoE | `--moe_permute_fusion` | 融合 token 重排 |
| MoE | `--moe_grouped_gemm` | 多专家 GEMM 合批 |
| MoE | `--moe_shared_expert_overlap` | 共享专家计算/通信重叠 |
| MoE | `--moe_aux_loss_coeff` | 路由负载均衡 |
| MoE | `--moe_expert_capacity_factor` | 专家容量上限、封顶显存 |
| 显存 | `--optimizer_cpu_offload`(false→true) | 优化器状态卸到 CPU |
| 显存 | `--use_precision_aware_optimizer` | offload 下保精度 |
| 显存 | `--optimizer_offload_fraction` | CPU↔GPU 下放比例 |
| 数据 | `dataloader_num_workers`(8→2) | 减少主机内存压力 |
| 数据 | `dataset_num_proc`(196→16) | 减少预处理内存压力 |
| 并行 | `tensor_model_parallel_size`(4→2) | 受 num_query_groups=2 限制 |

---

## 十、显存/OOM 排查速记

| 报错 | 性质 | 方向 |
|---|---|---|
| `num_query_groups (2) must be a multiple of TP (4)` | 整除校验 | TP 降到 2 |
| `DataLoader worker killed by signal: Killed` | **主机 RAM OOM** | offload_fraction 调低 / 减 dataloader_workers / 加节点 |
| `CUDA out of memory ... param_data = torch.zeros` | **GPU OOM** | EP/PP 调大(切更细) / offload_fraction 调高 / 加卡 / LoRA |
| `ProcessGroupNCCL ... no GPUs found` | 节点无 GPU | 把无卡机移出节点列表 |

**根本矛盾**：offload 高→CPU 爆，低→GPU 爆。脱困靠「**更多卡**」或「**LoRA**（消掉 4.76TB 优化器状态）」。

---

## 十一、训练日志字段解读

一条典型日志（训练中 + 结束汇总）：
```
{"loss": 0.371, "grad_norm": 0.178, "learning_rate": 5e-07, "load_balancing_loss": 1.622,
 "iteration": "85/85", "elapsed_time": "3h 48m", "remaining_time": "0s",
 "memory(GiB)": 112.9, "train_speed(s/it)": 161.1}
{"train_dataset": "32681.6±537.1, min=9964, max=32768, size=1821",
 "last_model_checkpoint": ".../checkpoint-85", "best_model_checkpoint": null, "best_metric": null}
```

### 每个 step 的字段
| 字段 | 含义 | 健康参考 |
|---|---|---|
| `loss` | **主训练损失**（交叉熵） | 平稳下降即可 |
| `grad_norm` | 梯度 L2 范数 | 小而稳（~0.1~1）；飙升=梯度爆炸 |
| `learning_rate` | 当前学习率 | 按 warmup→cosine 走，末尾到 `min_lr` |
| `load_balancing_loss` | **MoE 路由负载均衡损失**（见下） | ~1~2 健康；持续上飙=路由塌缩 |
| `iteration` | 当前步/总步 | 总步 = 打包后样本数×epoch / global_batch |
| `memory(GiB)` | torch 报告的 GPU 显存 | 比 nvidia-smi 低（不含 CUDA ctx/NCCL/碎片） |
| `train_speed(s/it)` | 每步秒数 | offload+重计算下 397B 偏慢正常 |

### `load_balancing_loss`（你问的）
MoE 的**辅助损失**，衡量 token 在各专家间分得均不均。**不是训练目标，是健康指标**：
- 公式上 **完全均衡 = 1.0**，越大越不均（最坏接近专家数 512）；
- 它乘 `moe_aux_loss_coeff(1e-6)` 才进总 loss（贡献极小），作用是轻推路由别塌缩；
- **看趋势**：稳定在 1~2 就好；一路上飙说明路由塌缩到少数专家，需警惕。

### `train_dataset` 统计（你问的）
格式 = **打包后训练序列的 token 长度分布**：`均值±标准差, min=, max=, size=`
- 例：`32681.6±537.1, min=9964, max=32768, size=1821`
- **均值≈max(32768)** ⇒ packing 几乎填满（装箱效率 ~99.7%，激活接近满载）；
- `min` = 最短的那条打包序列（边角余料箱）；
- **`size` = 打包后序列总条数**，决定 step 数：`size×epoch / global_batch`（1821×3/64 ≈ 85，与 `iteration 85/85` 对上）。
- 关联：均值越接近 max，激活越满、GPU 峰值越高（见第八节 packing 与显存峰值的讨论）。

### 结束汇总字段
- `last_model_checkpoint`：最后保存的 ckpt 路径；
- `best_model_checkpoint` / `best_metric`：**`split_dataset_ratio=0` 无验证集时为 `null`**（不追踪"最优"ckpt，靠下游评测判优劣）。

---

## 十二、相关源码索引

- Megatron 专属参数：[../../swift/megatron/arguments/megatron_args.py](../../swift/megatron/arguments/megatron_args.py)
- 通用模板参数：[../../swift/arguments/base_args/template_args.py](../../swift/arguments/base_args/template_args.py)
- 27B 基础版参数详解：[sft_train_参数详解.md](./sft_train_参数详解.md)
- Qwen3.5 官方最佳实践：[../../docs/source/BestPractices/Qwen3_5-Best-Practice.md](../../docs/source/BestPractices/Qwen3_5-Best-Practice.md)
