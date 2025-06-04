# Remove diffusers completely and reinstall with minimal dependencies
pip uninstall -y diffusers transformers accelerate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2 accelerate==0.25.0
pip install diffusers==0.25.0 --no-deps
pip install peft safetensors pillow numpy tqdm

# Set environment variables
export BITSANDBYTES_NOWELCOME=1
export DISABLE_BITSANDBYTES=1

# Test
python3 -c "from diffusers import StableDiffusionPipeline; print('Success!')"