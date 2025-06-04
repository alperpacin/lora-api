#!/bin/bash
# Add our cuDNN directory to library path
export LD_LIBRARY_PATH=$HOME/.cudnn/lib:$LD_LIBRARY_PATH

# Tell PyTorch where to find cuDNN libraries
export CUDNN_LIBRARY=$HOME/.cudnn/lib
export CUDNN_INCLUDE_DIR=$HOME/.cudnn/include

# Prevent PyTorch from searching for other cuDNN libraries
export CUDNN_LIBRARY_PATH=$HOME/.cudnn/lib

# Set library loading verbose mode for debugging
export LD_DEBUG=libs

echo "✅ CUDA environment variables set"
