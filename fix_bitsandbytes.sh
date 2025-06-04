#!/bin/bash
# fix_bitsandbytes.sh - Script to fix bitsandbytes CUDA issues

echo "Checking CUDA setup and fixing bitsandbytes issues..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver found:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "No NVIDIA driver found. Installing CPU-only versions..."
    export CUDA_VISIBLE_DEVICES=""
fi

# Check Python and virtual environment
echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"

# Set environment variables to disable problematic bitsandbytes features
export BITSANDBYTES_NOWELCOME=1
export DISABLE_BITSANDBYTES=1

# Check bitsandbytes installation
echo "Checking bitsandbytes..."
python -c "
import sys
try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__}')
    print('Testing bitsandbytes CUDA setup...')
    try:
        # Test basic functionality
        import torch
        if torch.cuda.is_available():
            print(f'CUDA available: {torch.cuda.get_device_name(0)}')
            # Try to create a simple 8bit optimizer
            model = torch.nn.Linear(10, 1).cuda()
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
            print('bitsandbytes 8bit optimizer works!')
        else:
            print('CUDA not available, 8bit optimizers will not work')
    except Exception as e:
        print(f'bitsandbytes CUDA test failed: {e}')
        print('Will fall back to regular optimizers')
except ImportError as e:
    print(f'bitsandbytes not installed: {e}')
    print('Installing CPU-only version...')
"

# Try to fix common bitsandbytes issues
echo "Attempting to fix bitsandbytes installation..."

# Uninstall and reinstall bitsandbytes
pip uninstall -y bitsandbytes

# Install CPU-only version if CUDA is not available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing CPU-only bitsandbytes..."
    pip install bitsandbytes-cpu
else
    echo "Installing CUDA-compatible bitsandbytes..."
    pip install bitsandbytes
fi

# Test the installation
echo "Testing fixed installation..."
python -c "
import os
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
try:
    import bitsandbytes as bnb
    print('bitsandbytes imported successfully')
except Exception as e:
    print(f'Still having issues: {e}')
    print('Setting environment to disable 8bit optimizations')
    os.environ['DISABLE_BITSANDBYTES'] = '1'
"

echo "Fix complete. You can now run training with regular AdamW optimizer."
echo "To use 8bit optimizers, ensure CUDA is properly set up and LD_LIBRARY_PATH includes CUDA libraries."