#!/bin/bash
# 397B-A17B SFT 推理脚本 - 使用 sglang 原生后端，离线批量推理
# 与 27B 版(infer_sglang.sh)的核心区别：
#   27B 能塞进单卡 -> TP=1, DP=8（8 个副本拼吞吐）
#   397B 权重 794GB(=397B×2字节) 单卡放不下 -> 必须 TP=8 跨 8 卡切一个副本, DP=1
source /home3/medcog/jycai6/.bashrc

module use /opt/tool/modulefiles/
module load cuda/12.9


tmp_cache="/tmp/jmli27_verl_cache_$(hostname -s)"
export MODELSCOPE_CACHE=$tmp_cache/modelscope
export HF_HOME=$tmp_cache/huggingface
export TRITON_CACHE_DIR="${tmp_cache}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${tmp_cache}/inductor_cache"
export VLLM_CONFIG_ROOT="${tmp_cache}/vllm_config"
export FLASHINFER_WORKSPACE_BASE="${tmp_cache}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${tmp_cache}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${tmp_cache}/xdg_cache"
export SGLANG_DG_CACHE_DIR="${tmp_cache}/deep_gemm"
export DG_JIT_CACHE_DIR="${SGLANG_DG_CACHE_DIR}/cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CONFIG_ROOT}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}" "${SGLANG_DG_CACHE_DIR}" "${DG_JIT_CACHE_DIR}"

# Conda 环境配置
CONDA_ENV_NAME="sglang_infer"
conda activate ${CONDA_ENV_NAME}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# single node
export RANK=0
export NODE_RANK=$RANK
export WORLD_SIZE=1
export NNODES=$WORLD_SIZE
export MASTER_PORT=6223
export MASTER_ADDR=localhost

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ===== 模型路径：换成你的 397B 权重 / SFT 后的 checkpoint =====
MODEL_PATH="/iflytek/jmli27/pretrain_models/Qwen3.5-397B-A17B"
MODEL_ID="qwen_397b_a17b_baseline"
# SFT 后推理示例（换成你训练输出的 -resave 目录）：
# MODEL_PATH="/train21/.../qwen_397b_a17b_sft_med_.../v0-xxx/checkpoint-xxx-resave"
# MODEL_ID="qwen_397b_a17b_sft_med_epoch1"

# 输出配置
OUTPUT_DIR="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_397b_a17b_sft_med_20260501/test_output"

# 解析命令行参数
THINKING_MODE="all"   # "all", "fast" 或 "slow"
while [[ $# -gt 0 ]]; do
    case $1 in
        --thinking-mode)
            THINKING_MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ===== 并行：397B 单节点 8 卡，TP=8 切一个副本，DP=1 =====
# 794GB 权重 / 8 ≈ 99GB/卡，剩 ~41GB/卡 给 KV cache + 激活。
# 注：GQA num_query_groups=2，但 SGLang 会在 TP>KV头数 时自动复制 KV 头，
#     所以 TP=8 推理没问题（不像 Megatron 训练那样强制整除）。
TP_SIZE=8
DP_SIZE=1
DTYPE="bfloat16"  # 此行没用

# 推理参数
MAX_NEW_TOKENS=32768
TEMPERATURE=0.7
TOP_P=1.0
TOP_K=-1
BATCH_SIZE=128

# 显存利用 mem_fraction_static = (模型权重 + KV cache 池) / GPU 总显存
# 397B 权重已占 ~99GB/140GB(≈71%)，留给 KV+激活的空间不多：
#   - 0.9 → ~126GB 用于权重+KV，剩 ~14GB 给激活/CUDA graph，可能偏紧；
#   - 遇到 OOM 先调到 0.85，再不行 0.8；
#   - 长上下文(32k)/大并发会吃更多 KV，相应再调小。
MEM_FRACTION=0.9

# 数据集路径（支持多个输入文件）
INPUT_FILES=(
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/医考等级集610题/医考等级集610题.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/病历质控-无锡集/病历质控-无锡集.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/IFEval/ifeval.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/KIE-科研分析/省立线上数据开发集.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/病历生成/病历生成精简测试_顺序2.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/病历生成/门诊新验收集130.jsonl"
)

# 执行推理
SCRIPT_DIR=$(dirname "$0")

python ${SCRIPT_DIR}/infer_sglang.py \
    --model ${MODEL_PATH} \
    --input "${INPUT_FILES[@]}" \
    --output-dir ${OUTPUT_DIR} \
    --model-id ${MODEL_ID} \
    --thinking-mode ${THINKING_MODE} \
    --tp ${TP_SIZE} \
    --dp ${DP_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    --mem-fraction ${MEM_FRACTION}
