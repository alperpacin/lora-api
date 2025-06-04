#!/usr/bin/env bash
# setup.sh - Script to set up LoRA training environment

# Detect OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
else
    IS_WINDOWS=false
fi

# Set base directory to script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create Python virtual environment
echo "Creating Python virtual environment..."
if [ "$IS_WINDOWS" = true ]; then
    python -m venv venv
    ACTIVATE_PATH="venv/Scripts/activate"
else
    python3 -m venv venv
    ACTIVATE_PATH="venv/bin/activate"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ "$IS_WINDOWS" = true ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
if [ "$IS_WINDOWS" = true ]; then
    python -m pip install --upgrade pip
else
    python3 -m pip install --upgrade pip
fi

# Install dependencies
echo "Installing dependencies..."
if [ "$IS_WINDOWS" = true ]; then
    python -m pip install -r requirements.txt
else
    python3 -m pip install -r requirements.txt
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p "$(dirname "$0")/scripts/stable"
mkdir -p "$(dirname "$0")/huggingface"
mkdir -p "$HOME/lora_data/train_images"
mkdir -p "$HOME/lora_data/lora_jobs"

# Copy training script to scripts directory
echo "Setting up training scripts..."
cp "$(dirname "$0")/train_network.py" "$(dirname "$0")/scripts/stable/"

# Configure accelerate
echo "Configuring accelerate..."
if command -v accelerate &> /dev/null; then
    accelerate config default
else
    echo "Warning: accelerate command not found. Please run 'accelerate config' manually."
fi

echo "========================================================================"
echo "Setup completed successfully!"
echo ""
echo "To run the API server:"
echo "  1. Activate the virtual environment: source $ACTIVATE_PATH"
echo "  2. Start the server: uvicorn api_server:app --host 0.0.0.0 --port 8000"
echo "========================================================================" 