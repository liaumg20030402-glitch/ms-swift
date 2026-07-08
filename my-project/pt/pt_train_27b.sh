#!/usr/bin/env bash

source /home3/medcog/jycai6/.bashrc
conda activate swift_sft_397b
export HOME=/home3/medcog/jycai6
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

MODEL_PATH=/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B

# 修改为实际的纯文本预训练数据路径
# 格式：{messages: [{role: "assistant", content: 预训练文本}]}
# {"messages": [{"role": "assistant", "content": "<image>是一只小狗，<image>是一只小猫"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
DATASET_ROOT=/V14/synthetic-pt

mapfile -d '' DATASETS < <(
    find "$DATASET_ROOT" -type f -name '*.jsonl' -print0 | sort -z
)

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No JSONL files found under $DATASET_ROOT"
    exit 1
fi

echo "Found ${#DATASETS[@]} JSONL files"

output_dir=/train21/medcog/permanent/jycai6/jmlli27/pt/exps/qwen3_5_27b_cpt_32k

# world size = TP4 * PP1 * CP2 * DP6
megatron pt \
    --dataset "${DATASETS[@]}" \
    --split_dataset_ratio 0 \
    --model $MODEL_PATH \
    --output_dir $output_dir \
    --save_safetensors true \
    --load_from_cache_file true \
    --streaming true \
    --dataset_shuffle true \
    --shuffle_buffer_size 10000 \
    --logging_steps 1 \
    --tuner_type full \
    --finetune true \
    --torch_dtype bfloat16 \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 96 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --padding_free true \
    --truncation_strategy split \
    --num_train_epochs 1 \
    --train_iters 120000 \
    --cross_entropy_loss_fusion true \
    --apply_wd_to_qk_layernorm true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 196 \
    --no_save_optim false \
    --no_save_rng false \
    --sequence_parallel true \
    --optimizer_cpu_offload false \
    --attention_backend flash \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --save_steps 20000 
