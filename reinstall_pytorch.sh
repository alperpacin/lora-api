#!/usr/bin/env bash
set -e

echo "🔄 Reinstalling PyTorch with correct CUDA version..."

# Get system CUDA version
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d\.]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

echo "📋 Detected CUDA version: $CUDA_VERSION (Major: $CUDA_MAJOR, Minor: $CUDA_MINOR)"

# Uninstall existing PyTorch packages
echo "🗑️ Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio

# Install PyTorch based on detected CUDA version
if [ "$CUDA_MAJOR" = "12" ]; then
  if [ "$CUDA_MINOR" -ge "1" ]; then
    echo "📦 Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    echo "📦 Installing PyTorch with CUDA 12.0 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
  fi
elif [ "$CUDA_MAJOR" = "11" ]; then
  if [ "$CUDA_MINOR" -ge "8" ]; then
    echo "📦 Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  elif [ "$CUDA_MINOR" -ge "7" ]; then
    echo "📦 Installing PyTorch with CUDA 11.7 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  else
    echo "📦 Installing PyTorch with CUDA 11.6 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
  fi
else
  echo "❌ Unsupported CUDA version: $CUDA_VERSION"
  echo "📦 Installing latest PyTorch version (might not be compatible)..."
  pip install torch torchvision torchaudio
fi

# Install xformers for acceleration
echo "📦 Installing xformers..."
pip install --no-deps xformers

# Install cuDNN
echo "📦 Installing cuDNN compatible with PyTorch..."
if [ "$CUDA_MAJOR" = "12" ]; then
  pip install nvidia-cudnn-cu12
elif [ "$CUDA_MAJOR" = "11" ]; then
  pip install nvidia-cudnn-cu11
fi

# Verify installation
echo "🧪 Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    try:
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print('✅ PyTorch installation successful!')
    except Exception as e:
        print(f'⚠️ cuDNN not properly configured: {e}')
else:
    print('⚠️ CUDA not available in PyTorch')
"

echo "🏁 PyTorch reinstallation complete!"
echo ""
echo "Next steps:"
echo "1. Run the cuDNN fix script to ensure proper library paths:"
echo "   bash fix_cudnn_paths.sh"
echo ""
echo "2. Then apply the environment settings:"
echo "   source setup_cuda_env.sh"
echo ""
echo "3. Try running your training job again" 