#!/bin/bash
# nohup ./install.sh > build_log2.txt 2>&1 &
# conda env remove -n swift_rl -y
conda create -n swift_rl_py312 python=3.12 -y
source /home3/medcog/jycai6/.bashrc
conda activate swift_rl_py312
export HOME=/home3/medcog/jycai6
export PIP_CONFIG_FILE=/home3/medcog/jycai6/.pip/pip.conf
export CONDARC=/home3/medcog/jycai6/.condarc

# 2. 配置 CUDA 和编译环境变量
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
cd ~/pkgs
unzip ms-swift-main.zip
cd ms-swift-main-20260629
# 安装 swift 及其依赖项
pip install -e .
pip install -U "transformers>=5.9" "qwen_vl_utils>=0.0.14" peft liger-kernel


# 开启多线程编译，加速过程
export MAX_JOBS=64
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

echo "================================================="
echo "开始编译安装 CUDA 扩展库"
echo "开始时间: $(date)"
echo "================================================="

echo -e "\n>>> [1/4] 正在编译 Flash-Attention "
pip install "flash-attn==2.8.3" --no-build-isolation -v

echo -e "\n>>> [2/4]正在编译 NVIDIA Apex "
cd ~/pkgs/apex-master
python setup.py install --cpp_ext --cuda_ext

echo -e "\n>>> [3/4] 正在编译 Causal-Conv1d..."
cd ~/pkgs/causal-conv1d-main
pip install -v . --no-build-isolation

echo -e "\n>>> [4/4] 正在编译 Flash-Linear-Attention..."
cd ~/pkgs/flash-linear-attention-main
pip install -v . --no-build-isolation

pip install deepspeed
# vllm (torch2.10) for inference/deployment/RL
pip install -U "vllm>=0.17.0"
# 对于强化学习 (RL) 训练，需要覆盖 vLLM 的默认安装版本
pip install -U "transformers>=5.3.0"

pip install math_verify

# 单独装一个较新的 C++ 标准库
conda install libgcc-ng libstdcxx-ng -y

pip install pybind11
pip install --no-build-isolation "transformer_engine[pytorch]"
# pip install "megatron-core==0.15.*" -U
pip install \
    "mcore-bridge==1.5.0" \（mtp要1.5.2）
    "megatron-core==0.17.1"

pip install tilelang

python -m pip install \
  --force-reinstall \
  --no-cache-dir \
  "apache-tvm-ffi==0.1.10"

pip uninstall -y megatron-core

cd ~/pkgs/Megatron-LM-main
MAX_JOBS=4 python -m pip install --no-build-isolation -e .