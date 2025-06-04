#!/usr/bin/env bash
set -e

echo "🚀 Starting LoRA training with cuDNN safety measures..."

# Check if job_id is provided
if [ $# -lt 1 ]; then
  echo "❌ Error: Job ID is required"
  echo "Usage: $0 <job_id> [model_path]"
  exit 1
fi

JOB_ID="$1"
MODEL_PATH="${2:-sd-models/realVisXLv50.safetensors}"

# Set environment variables to help with cuDNN issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Add our custom cuDNN directory to the library path if it exists
if [ -d "$HOME/.cudnn/lib" ]; then
  export LD_LIBRARY_PATH="$HOME/.cudnn/lib:$LD_LIBRARY_PATH"
  echo "✅ Added custom cuDNN path to LD_LIBRARY_PATH"
fi

# Determine job directory
JOB_DIR="/home/user/lora_data/lora_jobs/$JOB_ID"
TRAIN_DIR=$(find /home/user/lora_data/train_images -type d -name "*" | grep -v "^/home/user/lora_data/train_images$" | head -1)

if [ ! -d "$JOB_DIR" ]; then
  echo "❌ Error: Job directory not found: $JOB_DIR"
  exit 1
fi

if [ ! -d "$TRAIN_DIR" ]; then
  echo "❌ Error: No training data directories found in /home/user/lora_data/train_images"
  exit 1
fi

echo "📂 Job directory: $JOB_DIR"
echo "📂 Training data directory: $TRAIN_DIR"

# Create output directory if it doesn't exist
OUTPUT_DIR="$JOB_DIR/output"
mkdir -p "$OUTPUT_DIR"

# First try: Run with normal settings
echo "🔄 Attempt 1: Running with normal settings..."
python -m accelerate.commands.launch \
  --mixed_precision=fp16 \
  train_network.py \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --train_data_dir="$TRAIN_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --network_module=networks.lora \
  --learning_rate=0.0001 \
  --max_train_steps=1000 \
  --resolution=1024,1024 \
  --train_batch_size=1 \
  --network_alpha=128 \
  --mixed_precision=fp16 \
  --save_model_as=safetensors \
  --cache_latents \
  --optimizer_type=AdamW \
  --xformers \
  --bucket_no_upscale || {
    
    echo "⚠️ First attempt failed, trying with cuDNN disabled..."
    # Second try: Disable cuDNN
    export PYTORCH_CUDNN_ENABLED=0
    echo "🔄 Attempt 2: Running with cuDNN disabled..."
    python -m accelerate.commands.launch \
      --mixed_precision=fp16 \
      train_network.py \
      --pretrained_model_name_or_path="$MODEL_PATH" \
      --train_data_dir="$TRAIN_DIR" \
      --output_dir="$OUTPUT_DIR" \
      --network_module=networks.lora \
      --learning_rate=0.0001 \
      --max_train_steps=1000 \
      --resolution=1024,1024 \
      --train_batch_size=1 \
      --network_alpha=128 \
      --mixed_precision=fp16 \
      --save_model_as=safetensors \
      --cache_latents \
      --optimizer_type=AdamW \
      --xformers \
      --bucket_no_upscale || {
        
        echo "⚠️ Second attempt failed, trying with CPU..."
        # Third try: Use CPU
        echo "🔄 Attempt 3: Running on CPU (slower but more compatible)..."
        python -m accelerate.commands.launch \
          --use_cpu \
          train_network.py \
          --pretrained_model_name_or_path="$MODEL_PATH" \
          --train_data_dir="$TRAIN_DIR" \
          --output_dir="$OUTPUT_DIR" \
          --network_module=networks.lora \
          --learning_rate=0.0001 \
          --max_train_steps=500 \
          --resolution=512,512 \
          --train_batch_size=1 \
          --network_alpha=128 \
          --mixed_precision=no \
          --save_model_as=safetensors \
          --cache_latents \
          --optimizer_type=AdamW \
          --bucket_no_upscale || {
            echo "❌ All training attempts failed."
            exit 1
          }
      }
  }

echo "✅ Training completed successfully!" 