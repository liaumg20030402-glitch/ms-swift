
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

BF16_MODEL=/path/to/checkpoint-xxx-resave
FP8_MODEL=/path/to/checkpoint-xxx-resave-FP8

megatron export \
    --model $BF16_MODEL \
    --output_dir $FP8_MODEL \
    --to_hf true \
    --fp8_recipe blockwise \
    --fp8_format e4m3 \
    --fp8_param_gather true \
    --linear_decoupled_in_proj true \
    --mtp_num_layers 0 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 6 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1