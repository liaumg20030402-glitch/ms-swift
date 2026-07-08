source /home3/medcog/jycai6/.bashrc
conda activate swift_sft_py312

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
# Qwen3.5-397B-A17B 是大型 MoE 模型，必须使用 megatron 后端 + MoE 并行。
#
# 并行维度整除关系（务必满足，否则启动即报错）：
#   1) world_size(总卡数) = TP * PP * DP           
#        TP = tensor_model_parallel_size
#        PP = pipeline_model_parallel_size
#        DP = 数据并行度（由总卡数自动推导：DP = world_size / (TP*PP)）
#   2) MoE 专家并行（ETP=expert_tensor_parallel_size，此处=1）：
#      world_size = ETP × EP × expert_DP × PP
#   3) PP 必须整除模型层数（层数不整除时用 --decoder_last_pipeline_num_layers 调整）
#   4) EP 必须整除模型的专家总数（num_experts；397B=512，122B=256）
#   5) TP 必须整除注意力的 num_attention_heads / num_query_groups
#      注意：Qwen MoE 是 GQA，num_query_groups 很小。122B-A10B 和 397b-a17b 的
#        num_query_groups=2，所以 TP 只能取 1 或 2（TP=4 会直接报
#        "num_query_groups (2) must be a multiple of tensor_model_parallel_size (4)"）。
#        397B-A17B 实测 num_query_groups 同样=2 => TP 也只能取 2（与 122B 一致）。
#        （megatron-core>=0.16 可解除此限制，但需升级）
#   6) global batch 必须整除 DP"）
#
#
# 【397B-A17B（60 层, num_query_groups=2 => TP=2）配置】训 397B 时切到这套：
#     48 卡 32k work，64k OOM: TP=2, PP=6, DP=4,  EP=8  -> TP*DP=8,  EP=8 ✓；60%6✓；512%8 ✓ （当前，PP更大=每卡层数/激活更少，抗激活OOM）
#     56 卡 整除性不好，OOM: TP=2, PP=2, DP=14, EP=4  -> TP*DP=28, EP=4 ✓；60%2 ✓，512%4 ✓，global batch要整除 DP，设成56（56=7×8 带因子7，用 EP=8 需 PP=7,DP=4 且 60/48 不被7整除要 uneven 切，PP=2 OOM）
#             TP=2, PP=12, DP=2, EP=4 
#   层数不能被目标 PP 整除时，用 --decoder_last_pipeline_num_layers N 调尾段。
# 全参 397B 显存：params+grads≈1.6TB 上卡，优化器状态≈4.76TB 需 offload 到 CPU。
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
