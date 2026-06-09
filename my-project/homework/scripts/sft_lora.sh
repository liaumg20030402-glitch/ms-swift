#!/bin/bash
# Qwen3.5 LoRA 微调 - hendrycks_math 数学数据集
# 参考: docs/source/BestPractices/Qwen3_5-Best-Practice.md (Dense 模型示例)
set -e

# ============ conda 环境 ============
# 环境安装见同目录 install.sh
# conda activate swift_sft_py312

# ============ 可配置参数 ============
# --- 硬件 ---
# 4 张 4090: GPUS="0,1,2,3"  NPROC=4
# 3 张 A100: GPUS="0,1,2"    NPROC=3
GPUS="2,3,4"
NPROC=3

# --- 模型 ---
MODEL="/home/lijinmei/pretrain_models/Qwen3.5-4B"

# --- 数据（先运行 prepare_data.py 生成）---
# 验证集由 swift 从 train 自动切出 1%（--split_dataset_ratio 0.01），无需单独的 val.jsonl
DATA_DIR="/home/lijinmei/swift_lora/data/swift_format"
TRAIN_FILE="${DATA_DIR}/train.jsonl"

# --- 输出 ---
OUTPUT_DIR="/home/lijinmei/swift_lora/output"
mkdir -p "${OUTPUT_DIR}"
# 训练日志（stdout/stderr 全量落盘，方便事后排查）
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"


# ============ 训练 ============
# 说明:
# - Qwen3.5 的 GatedDeltaNet (transformers 后端) 不支持 packing/padding_free，
#   因此用 --group_by_length true 做负载均衡加速（会让 loss 曲线轻微跳动，属正常）。
# - 数学解答不含 <think>，按非思考模式训练: --add_non_thinking_prefix + --loss_scale ignore_empty_think。
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=${NPROC} \
CUDA_VISIBLE_DEVICES=${GPUS} \
swift sft \
    --model ${MODEL} \
    --tuner_type lora \
    --dataset ${TRAIN_FILE} \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --use_liger_kernel true \
    --group_by_length true \
    --output_dir ${OUTPUT_DIR} \
    --save_strategy epoch \
    --eval_strategy steps \
    --eval_steps 20 \
    --logging_steps 1 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 5e-6}' \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --attn_impl sdpa \
    --report_to tensorboard 2>&1 | tee "${LOG_FILE}"

echo "训练日志已保存: ${LOG_FILE}"
echo "训练完成，adapter 保存在: ${OUTPUT_DIR}"

