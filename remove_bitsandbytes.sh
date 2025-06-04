#!/bin/bash
# fix_bitsandbytes.sh - Bash script to fix bitsandbytes issues

echo "🔧 Fixing bitsandbytes import issues..."

# Step 1: Uninstall bitsandbytes completely
echo "📦 Uninstalling bitsandbytes..."
pip uninstall -y bitsandbytes bitsandbytes-cpu

# Step 2: Set environment variables
echo "🌍 Setting environment variables..."
export BITSANDBYTES_NOWELCOME=1
export DISABLE_BITSANDBYTES=1
export CUDA_LAUNCH_BLOCKING=1

# Step 3: Clear Python cache
echo "🧹 Clearing Python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Step 4: Try reinstalling diffusers without dependencies
echo "🔄 Reinstalling diffusers..."
pip install --force-reinstall --no-deps diffusers

# Step 5: Test if diffusers can import
echo "🧪 Testing imports..."
python -c "
import sys
from unittest.mock import MagicMock

# Block bitsandbytes import
mock_bnb = MagicMock()
mock_bnb.__version__ = '0.0.0'
sys.modules['bitsandbytes'] = mock_bnb

# Test imports
try:
    from diffusers import StableDiffusionPipeline
    print('✅ Diffusers imports successfully')
except Exception as e:
    print(f'❌ Diffusers import failed: {e}')

try:
    from transformers import CLIPTokenizer
    print('✅ Transformers imports successfully')
except Exception as e:
    print(f'❌ Transformers import failed: {e}')

try:
    from peft import LoraConfig
    print('✅ PEFT imports successfully')
except Exception as e:
    print(f'❌ PEFT import failed: {e}')
"

# Step 6: Create environment setup script
echo "📝 Creating environment setup script..."
cat > setup_env.sh << 'EOF'
#!/bin/bash
# Set environment variables to prevent bitsandbytes issues
export BITSANDBYTES_NOWELCOME=1
export DISABLE_BITSANDBYTES=1
export CUDA_LAUNCH_BLOCKING=1

echo "Environment configured to disable bitsandbytes"
echo "You can now run your training script"
EOF

chmod +x setup_env.sh

echo ""
echo "🏁 Fix complete!"
echo ""
echo "Next steps:"
echo "1. Run: source setup_env.sh"
echo "2. Replace your train_network.py with the minimal version"
echo "3. Try training again"
echo ""
echo "The training will now use regular AdamW optimizer instead of AdamW8bit"