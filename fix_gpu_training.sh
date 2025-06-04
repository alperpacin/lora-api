#!/usr/bin/env bash
set -e

echo "🔧 Setting up GPU-focused training with correct cuDNN..."

# Detect CUDA version used by PyTorch
PYTORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "📋 PyTorch CUDA version: $PYTORCH_CUDA_VERSION"

# Find PyTorch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "📋 PyTorch version: $PYTORCH_VERSION"

# Clean up any previous cuDNN environment settings
echo "🧹 Cleaning up previous cuDNN settings..."
rm -rf ~/.cudnn
unset CUDNN_LIBRARY
unset CUDNN_INCLUDE_DIR
unset CUDNN_LIBRARY_PATH
unset PYTORCH_CUDNN_ENABLED
unset TORCH_USE_CUDNN

# Install matching cuDNN version
echo "📦 Installing matching cuDNN version for PyTorch CUDA $PYTORCH_CUDA_VERSION..."
if [[ "$PYTORCH_CUDA_VERSION" == 11.* ]]; then
  pip uninstall -y nvidia-cudnn-cu11
  pip install nvidia-cudnn-cu11==8.5.0.96
elif [[ "$PYTORCH_CUDA_VERSION" == 12.* ]]; then
  pip uninstall -y nvidia-cudnn-cu12
  pip install nvidia-cudnn-cu12==8.9.2.26
else
  echo "❌ Unsupported CUDA version: $PYTORCH_CUDA_VERSION"
  exit 1
fi

# Find the installed cuDNN library
echo "🔍 Finding installed cuDNN libraries..."
CUDNN_LIB_DIR=""
for dir in $(find /venv -path "*/nvidia/cudnn/lib" -type d 2>/dev/null); do
  if [ -d "$dir" ]; then
    CUDNN_LIB_DIR="$dir"
    echo "✅ Found cuDNN libraries at: $CUDNN_LIB_DIR"
    break
  fi
done

if [ -z "$CUDNN_LIB_DIR" ]; then
  echo "❌ Could not find cuDNN libraries. Trying alternative locations..."
  for dir in $(find /venv -name "libcudnn*.so*" -type f -exec dirname {} \; | sort -u 2>/dev/null); do
    if [ -d "$dir" ]; then
      CUDNN_LIB_DIR="$dir"
      echo "✅ Found cuDNN libraries at: $CUDNN_LIB_DIR"
      break
    fi
  done
fi

# Create environment setup script
cat > gpu_training_env.sh << EOF
#!/bin/bash
# Add cuDNN to library path
export LD_LIBRARY_PATH=$CUDNN_LIB_DIR:\$LD_LIBRARY_PATH

# Make sure PyTorch can find cuDNN
export CUDNN_LIBRARY=$CUDNN_LIB_DIR
export CUDNN_INCLUDE_DIR=$(dirname $CUDNN_LIB_DIR)/include

# Force PyTorch to use CUDA
export CUDA_VISIBLE_DEVICES=0
export TORCH_DEVICE=cuda

# Improve memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "✅ GPU training environment set up"
echo "🎮 Using cuDNN libraries from: $CUDNN_LIB_DIR"
EOF

chmod +x gpu_training_env.sh

echo "🏁 GPU training setup complete!"
echo ""
echo "To apply the changes and enable GPU training, run:"
echo "  source gpu_training_env.sh"
echo ""
echo "Then make an API call with 'use_safe_training' set to false to disable CPU fallback:"
echo "{\"use_safe_training\": false, ...other parameters...}"
echo ""
echo "This will force the training to only run on GPU." 