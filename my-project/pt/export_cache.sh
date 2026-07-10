#!/usr/bin/env bash

source /home3/medcog/jycai6/.bashrc
conda activate swift_sft_397b

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B
DATASET_ROOT=/V14/synthetic-pt
CACHE_ROOT=/train21/medcog/permanent/jycai6/jmlli27/cpt/cache
OUTPUT_DIR=$CACHE_ROOT/qwen3_5_27b_pt_cached_8k

tmp_cache=/tmp/jycai6_swift_cache_$(hostname)_${RANK:-0}
export MODELSCOPE_CACHE=$tmp_cache/modelscope
export HF_HOME=$tmp_cache/huggingface
export TRITON_CACHE_DIR=$tmp_cache/triton_cache
export TMPDIR=$tmp_cache/tmp

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TMPDIR"

mapfile -d '' DATASETS < <(
    find "$DATASET_ROOT" -type f -name '*.jsonl' -print0 | sort -z
)

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No JSONL files found under $DATASET_ROOT"
    exit 1
fi

echo "Found ${#DATASETS[@]} JSONL files"

swift export \
    --model "$MODEL_PATH" \
    --dataset "${DATASETS[@]}" \
    --to_cached_dataset true \
    --use_chat_template false \
    --loss_scale all \
    --truncation_strategy split \
    --max_length 8192 \
    --split_dataset_ratio 0 \
    --dataset_num_proc 64 \
    --output_dir "$OUTPUT_DIR"