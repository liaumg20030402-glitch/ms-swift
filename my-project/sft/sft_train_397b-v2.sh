source /home3/medcog/jycai6/.bashrc
conda activate swift_sft_397b

tmp_cache=/tmp/jycai6_swift_cache_$(hostname)_${RANK:-0}
export MODELSCOPE_CACHE=$tmp_cache/modelscope
export HF_HOME=$tmp_cache/huggingface
export TRITON_CACHE_DIR=$tmp_cache/triton_cache


PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NPROC_PER_NODE=8
export NNODES=$WORLD_SIZE
export NODE_RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1



# single node
# export RANK=0
# export NODE_RANK=$RANK
# export WORLD_SIZE=1
# export NNODES=$WORLD_SIZE
# export MASTER_PORT=6223
# export MASTER_ADDR=localhost

# export GLOO_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_IFNAME=eth0

export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MODEL_PATH=/iflytek/jmli27/pretrain_models/Qwen3.5-397B-A17B

# ============ 训练数据 ============
DATASETS=("/train21/medcog/permanent/jycai6/sft_data_zlzl/data/20260522/2_merged/z1z1_20260522.jsonl")

output_dir=/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_397b_a17b_sft_med_20260501/model_output/qwen_397b_a17b_sft_med_32k_20260502_epoch_v2

# ====================================================================
# Qwen3.5-397B-A17B：48 卡全参数 SFT 并行配置
#
# 当前参数：TP=2、PP=6、CP=2、EP=8、ETP=1、micro_batch_size=1、global_batch_size=64。
#
# 1) 普通模型并行与数据并行
#      world_size = TP * PP * CP * DP
#      DP = world_size / (TP * PP * CP)
#      当前：48 = 2 * 6 * 2 * 2，因此 DP=2。
#
# 2) MoE 专家并行
#   world_size = ETP × EP × expert_DP × PP
#
# 3) Pipeline Parallel
#      PP 各阶段必须能合法分配 Transformer 层。397b有60层，当前 60 % 6 == 0，每个 PP stage 10 层。
#      若层数不能整除 PP，需使用 decoder_first/last_pipeline_num_layers 等参数显式调整。
#
# 4) Tensor Parallel
#      megatron-core>=0.16 已解除 TP 必须整除 num_query_groups 的旧限制，因此不再因
#      num_query_groups=2 将 TP 限制为 1 或 2；但 TP 仍需满足 hidden/head/expert tensor
#      等实际切分维度及所用 kernel 的约束。TP 与 EP 同时启用时保留 sequence_parallel=true。
#
# 5) EP 必须整除模型的专家总数（num_experts；397B=512，122B=256）
#
# 6) Batch 整除关系
#      global_batch_size 必须被 micro_batch_size * DP 整除。
#      当前梯度累积步数 = 64 / (1 * 2) = 32。
# ====================================================================

megatron sft \
    --dataset $DATASETS \
    --split_dataset_ratio=0 \
    --model $MODEL_PATH \
    --save_safetensors true \
    --load_from_cache_file true \
    --logging_steps 1 \
    --add_non_thinking_prefix true \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 6 \
    --context_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --moe_expert_capacity_factor 2 \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --num_train_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 5e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --loss_scale ignore_empty_think \
    --padding_free true \
    --output_dir $output_dir \
    --max_length 32768 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 16 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 1.0 \
    --attention_backend flash \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --save_strategy epoch
