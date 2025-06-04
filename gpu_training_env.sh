#!/bin/bash
# Add cuDNN to library path
export LD_LIBRARY_PATH=/venv/main/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Make sure PyTorch can find cuDNN
export CUDNN_LIBRARY=/venv/main/lib/python3.10/site-packages/nvidia/cudnn/lib
export CUDNN_INCLUDE_DIR=/venv/main/lib/python3.10/site-packages/nvidia/cudnn/include

# Force PyTorch to use CUDA
export CUDA_VISIBLE_DEVICES=0
export TORCH_DEVICE=cuda

# Improve memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "✅ GPU training environment set up"
echo "🎮 Using cuDNN libraries from: /venv/main/lib/python3.10/site-packages/nvidia/cudnn/lib"
