#!/bin/bash
# Set environment variables to prevent bitsandbytes issues
export BITSANDBYTES_NOWELCOME=1
export DISABLE_BITSANDBYTES=1
export CUDA_LAUNCH_BLOCKING=1

echo "Environment configured to disable bitsandbytes"
echo "You can now run your training script"
