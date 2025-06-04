#!/usr/bin/env bash
set -e

echo "🔧 Fixing cuDNN library paths for PyTorch..."

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d\.]+')
echo "📋 Detected CUDA version: $CUDA_VERSION"

# Get PyTorch CUDA version
PYTORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "📋 PyTorch CUDA version: $PYTORCH_CUDA_VERSION"

# Find installed cuDNN packages
echo "📋 Finding installed cuDNN packages..."
pip list | grep cudnn

# Find actual cuDNN libraries
echo "📋 Finding cuDNN libraries in the system..."
find /venv -name "libcudnn*.so*" 2>/dev/null || echo "No cuDNN libraries found in /venv"
find /usr/local -name "libcudnn*.so*" 2>/dev/null || echo "No cuDNN libraries found in /usr/local"

# Create a directory for our cuDNN symlinks
mkdir -p ~/.cudnn/lib
echo "📁 Created directory for cuDNN symlinks: ~/.cudnn/lib"

# Find the source files for symlinks
CUDNN_SOURCE_DIR=""
for dir in /venv/main/lib/python*/site-packages/nvidia/cudnn/lib /usr/local/cuda*/lib64 /venv/lib /usr/lib/x86_64-linux-gnu; do
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        CUDNN_FILES=$(find "$dir" -name "libcudnn*.so*" 2>/dev/null)
        if [ -n "$CUDNN_FILES" ]; then
            CUDNN_SOURCE_DIR="$dir"
            echo "🔍 Found cuDNN libraries in: $CUDNN_SOURCE_DIR"
            break
        fi
    fi
done

# Create symlinks from source directory to our cuDNN directory
if [ -n "$CUDNN_SOURCE_DIR" ]; then
    echo "🔗 Creating symlinks from $CUDNN_SOURCE_DIR to ~/.cudnn/lib"
    for file in $(find "$CUDNN_SOURCE_DIR" -name "libcudnn*.so*" 2>/dev/null); do
        base_name=$(basename "$file")
        ln -sf "$file" ~/.cudnn/lib/"$base_name"
        echo "   Created symlink for $base_name"
    done
    
    # Create specific symlinks needed by PyTorch
    for version in 9 9.0 9.1 9.5; do
        if [ ! -f ~/.cudnn/lib/libcudnn_graph.so."$version" ] && [ -f ~/.cudnn/lib/libcudnn_graph.so ]; then
            ln -sf ~/.cudnn/lib/libcudnn_graph.so ~/.cudnn/lib/libcudnn_graph.so."$version"
            echo "   Created specific symlink for libcudnn_graph.so.$version"
        fi
    done
else
    echo "❌ Could not find cuDNN libraries to create symlinks"
    # Attempt to install a compatible version of cuDNN
    echo "📦 Installing cuDNN..."
    if [[ "$PYTORCH_CUDA_VERSION" == 11.* ]]; then
        pip install nvidia-cudnn-cu11
    elif [[ "$PYTORCH_CUDA_VERSION" == 12.* ]]; then
        pip install nvidia-cudnn-cu12
    else
        echo "❌ Could not determine correct cuDNN package for PyTorch CUDA version"
    fi
    
    # Try again to find cuDNN libraries
    echo "🔍 Finding cuDNN libraries after installation..."
    CUDNN_SOURCE_DIR=""
    for dir in /venv/main/lib/python*/site-packages/nvidia/cudnn/lib /usr/local/cuda*/lib64; do
        if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
            CUDNN_FILES=$(find "$dir" -name "libcudnn*.so*" 2>/dev/null)
            if [ -n "$CUDNN_FILES" ]; then
                CUDNN_SOURCE_DIR="$dir"
                echo "🔍 Found cuDNN libraries in: $CUDNN_SOURCE_DIR"
                
                # Create symlinks from new installation
                echo "🔗 Creating symlinks from $CUDNN_SOURCE_DIR to ~/.cudnn/lib"
                for file in $(find "$CUDNN_SOURCE_DIR" -name "libcudnn*.so*" 2>/dev/null); do
                    base_name=$(basename "$file")
                    ln -sf "$file" ~/.cudnn/lib/"$base_name"
                    echo "   Created symlink for $base_name"
                done
                
                # Create specific symlinks needed by PyTorch
                for version in 9 9.0 9.1 9.5; do
                    if [ ! -f ~/.cudnn/lib/libcudnn_graph.so."$version" ] && [ -f ~/.cudnn/lib/libcudnn_graph.so ]; then
                        ln -sf ~/.cudnn/lib/libcudnn_graph.so ~/.cudnn/lib/libcudnn_graph.so."$version"
                        echo "   Created specific symlink for libcudnn_graph.so.$version"
                    fi
                done
                break
            fi
        fi
    done
fi

# Create environment setup script
cat > setup_cuda_env.sh << 'EOF'
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
EOF

chmod +x setup_cuda_env.sh

echo "🏁 cuDNN library paths fixed!"
echo ""
echo "To apply the fix, run:"
echo "  source setup_cuda_env.sh"
echo ""
echo "Then try running your training job again."
echo ""
echo "If issues persist, try running:"
echo "  export LD_DEBUG=libs"
echo "before your command to see detailed library loading information." 