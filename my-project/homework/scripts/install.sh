conda create -n swift_sft_py312 python=3.12 -y
conda activate swift_sft_py312

# 先升级 pip 工具链，避免拿不到预编译 wheel 而去源码编译（libcst/pyarrow 等）
pip install --upgrade pip setuptools wheel

cd ms-swift-main
# 安装 swift 及其依赖项
pip install -e .
# "transformers==5.2.*" encounters compatibility issues with vllm. See this issue: https://github.com/modelscope/ms-swift/issues/8254
# "transformers==5.3.*" encounters video training issues. See this issue: https://github.com/modelscope/ms-swift/issues/8362
pip install -U "transformers==5.3.0" "qwen_vl_utils>=0.0.14" peft liger-kernel

# flash-linear-attention（纯 Triton，Qwen3.5 线性注意力必需，不需要 nvcc）
# If you encounter slow training issues, please refer to: https://github.com/fla-org/flash-linear-attention/issues/758
# Please use Python 3.12: https://github.com/fla-org/flash-linear-attention/issues/121
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation

# deepspeed training
pip install deepspeed

# 说明：本机无匹配 torch(cu13) 的 nvcc，且系统 CUDA 仅 11.8，故不安装需要现场编译 CUDA 扩展的
# causal_conv1d / flash-attn（它们只是加速算子，非必需）。训练时使用 --attn_impl sdpa 即可。
# 若日后拿到匹配的 CUDA 工具链，可再补装：
#   pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
#   pip install "flash-attn==2.8.3" --no-build-isolation

pip install modelscope
modelscope download --model Qwen/Qwen3.5-4B \
    --local_dir /share1/home/jinmei/pretrain_models/Qwen3.5-4B
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128