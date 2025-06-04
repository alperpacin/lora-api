#!/bin/bash
# fix_cudnn.sh - Script to fix cuDNN library issues

echo "🔧 Fixing cuDNN library issues..."

# Check current CUDA version
echo "📋 Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    nvcc --version 2>/dev/null || echo "nvcc not found"
else
    echo "❌ NVIDIA driver not found"
fi

echo ""
echo "📋 Checking current PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available in PyTorch')
"

echo ""
echo "🔄 Fixing cuDNN issues..."

# Method 1: Reinstall PyTorch with specific CUDA version
echo "Method 1: Reinstalling PyTorch with CUDA 11.8..."
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Method 2: Install cuDNN via conda if available
if command -v conda &> /dev/null; then
    echo "Method 2: Installing cuDNN via conda..."
    conda install -y cudnn -c conda-forge
fi

# Method 3: Set library path
echo "Method 3: Setting up library paths..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Method 4: Install specific cuDNN version
echo "Method 4: Installing cuDNN 8.x..."
pip install nvidia-cudnn-cu11

echo ""
echo "🧪 Testing PyTorch with cuDNN..."
python -c "
import torch
import torch.nn as nn

print('Testing PyTorch installation...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    try:
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'cuDNN enabled: {torch.backends.cudnn.enabled}')
        
        # Test cuDNN functionality
        x = torch.randn(1, 3, 224, 224).cuda()
        conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
        y = conv(x)
        print('✅ cuDNN test passed')
        
    except Exception as e:
        print(f'❌ cuDNN test failed: {e}')
        print('Trying to disable cuDNN...')
        torch.backends.cudnn.enabled = False
        print('cuDNN disabled, will use slower fallback')
else:
    print('CUDA not available, using CPU mode')
"

# Create environment setup script
echo ""
echo "📝 Creating environment setup script..."
cat > setup_cuda_env.sh << 'EOF'
#!/bin/bash
# Set up CUDA environment variables
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Disable cuDNN if needed (fallback)
export PYTORCH_CUDNN_ENABLED=1

echo "CUDA environment variables set"
echo "If you still have cuDNN issues, you can disable it with:"
echo "export PYTORCH_CUDNN_ENABLED=0"
EOF

chmod +x setup_cuda_env.sh

echo ""
echo "🏁 cuDNN fix complete!"
echo ""
echo "Next steps:"
echo "1. Run: source setup_cuda_env.sh"
echo "2. If training still fails, try CPU-only mode"
echo "3. Or disable cuDNN with: export PYTORCH_CUDNN_ENABLED=0"