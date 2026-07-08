source /home3/medcog/jycai6/.bashrc
conda activate swift_sft_py312

tmp_cache=/tmp/zefeng_swift_cache_$(hostname)_${RANK:-0}
export MODELSCOPE_CACHE=$tmp_cache/modelscope
export HF_HOME=$tmp_cache/huggingface
export TRITON_CACHE_DIR=$tmp_cache/triton_cache

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# multi-node
export NPROC_PER_NODE=8
export NNODES=$WORLD_SIZE
export NODE_RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1

export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


MODEL_PATH=/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B

# ============ 训练数据 ============
DATASET=/train21/medcog/permanent/zefeng/Data/text/27B_SFT_20260424_17254.jsonl
output_dir=/train21/medcog/permanent/zefeng/Model/med_sft_train_swift/exps/qwen_27b_sft_v4.2_full_20260420/model_output/qwen_27b_sft_full_32k_20260526_full_unfreeze


megatron sft \
    --dataset $DATASET \
    --split_dataset_ratio=0 \
    --model $MODEL_PATH \
    --save_safetensors true \
    --load_from_cache_file true \
    --logging_steps 1 \
    --add_non_thinking_prefix true \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --vit_lr 1e-7 \
    --aligner_lr 1e-6 \
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
    --save_strategy epoch
