# File: api_server.py

import os
import uuid
import shutil
import zipfile
import tempfile
import requests
import subprocess
import platform

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional

app = FastAPI()

# Detect operating system
IS_WINDOWS = platform.system() == "Windows"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Request schema including:
#    - image_zip_url: URL to a ZIP of images
#    - folder_name: subfolder under /mnt/data/train_images/
#    - lora_name: desired LoRA filename (no extension)
#    - pretrained_model, hyperparameters…
# ─────────────────────────────────────────────────────────────────────────────
class LoRATrainRequest(BaseModel):
    image_zip_url: HttpUrl
    folder_name: str
    lora_name: str
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    resolution: str = "512,512"
    train_batch_size: int = 1
    network_alpha: int = 128
    mixed_precision: Optional[str] = "fp16"
    # Add new parameter for CUDA/cuDNN safety
    use_safe_training: bool = True
    # Add parameter to force GPU-only mode (no CPU fallback)
    force_gpu: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Directory constants
USER_HOME = os.path.expanduser("~")
TRAIN_BASE = os.path.join(USER_HOME, "lora_data/train_images")
JOBS_ROOT = os.path.join(USER_HOME, "lora_data/lora_jobs")
# Path to the Python virtual environment
VENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
# Path to the training scripts
TRAINING_SCRIPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

os.makedirs(TRAIN_BASE, exist_ok=True)
os.makedirs(JOBS_ROOT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Setup cuDNN paths and environment
# ─────────────────────────────────────────────────────────────────────────────
def setup_cudnn_environment():
    """Set up cuDNN environment variables for better compatibility"""
    # Create directory for cuDNN symlinks if it doesn't exist
    cudnn_lib_dir = os.path.join(USER_HOME, ".cudnn", "lib")
    os.makedirs(cudnn_lib_dir, exist_ok=True)
    
    # Set environment variables to help with cuDNN issues
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    
    # Add our custom cuDNN directory to the library path if it exists
    if os.path.exists(cudnn_lib_dir):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if cudnn_lib_dir not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_dir}:{current_ld_path}"
        
        # Tell PyTorch where to find cuDNN libraries
        os.environ["CUDNN_LIBRARY"] = cudnn_lib_dir
        os.environ["CUDNN_INCLUDE_DIR"] = os.path.join(USER_HOME, ".cudnn", "include")
        os.environ["CUDNN_LIBRARY_PATH"] = cudnn_lib_dir
    
    return cudnn_lib_dir

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper: download & extract ZIP
def download_and_extract_zip(zip_url: str, target_folder: str):
    """
    1. Downloads the ZIP from `zip_url` into a temp file.
    2. Extracts its contents into `target_folder`.
    3. Removes the temp file.
    Raises HTTPException on any failure.
    """
    try:
        fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot allocate temp file: {e}")

    try:
        with requests.get(zip_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp_zip_path, "wb") as out_f:
                for chunk in resp.iter_content(chunk_size=8192):
                    out_f.write(chunk)
    except requests.RequestException as e:
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)
        raise HTTPException(status_code=400, detail=f"Failed to download ZIP: {e}")

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)

    try:
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(target_folder)
    except zipfile.BadZipFile as e:
        os.remove(tmp_zip_path)
        shutil.rmtree(target_folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"ZIP is invalid or corrupted: {e}")
    except Exception as e:
        os.remove(tmp_zip_path)
        shutil.rmtree(target_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error extracting ZIP: {e}")
    finally:
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper: Copy required training files to job directory
def setup_training_environment(job_dir):
    """
    Copy necessary training scripts and files to the job directory
    """
    required_files = [
        "train_network.py",
        "train.sh"
    ]
    
    # Copy training scripts from the scripts directory
    scripts_source = TRAINING_SCRIPTS_PATH
    if not os.path.exists(scripts_source):
        # Fallback to current directory
        scripts_source = os.path.dirname(os.path.abspath(__file__))
    
    for filename in required_files:
        src_path = os.path.join(scripts_source, filename)
        dst_path = os.path.join(job_dir, filename)
        
        if os.path.exists(src_path):
            try:
                shutil.copyfile(src_path, dst_path)
                if not IS_WINDOWS and filename.endswith('.sh'):
                    os.chmod(dst_path, 0o755)
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to copy {filename}: {e}"
                )
        else:
            # If train_network.py doesn't exist, we need to handle this
            if filename == "train_network.py":
                raise HTTPException(
                    status_code=500,
                    detail=f"Required training script {filename} not found in {scripts_source}. "
                           f"Please ensure the training scripts are properly installed."
                )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Helper: Create Windows-compatible training script
def create_windows_training_script(job_dir, extracted_folder, req):
    """
    For Windows, create a direct Python script to handle training
    instead of trying to run the bash script
    """
    train_py_path = os.path.join(job_dir, "train_lora_wrapper.py")
    
    # Create a Python training script for Windows
    train_py_content = f"""
import os
import sys
import subprocess

# Set up training parameters
os.environ["HF_HOME"] = "huggingface"
os.environ["PYTHONPATH"] = os.getcwd()

# Parameters from the request
pretrained_model = "{req.pretrained_model}"
train_data_dir = r"{extracted_folder}"
output_dir = r"{job_dir}\\output"
learning_rate = {req.learning_rate}
max_train_steps = {req.max_train_steps}
resolution = "{req.resolution}"
train_batch_size = {req.train_batch_size}
network_alpha = {req.network_alpha}
mixed_precision = "{req.mixed_precision}"

# Build command-line arguments
args = [
    "--pretrained_model_name_or_path", pretrained_model,
    "--train_data_dir", train_data_dir,
    "--output_dir", output_dir,
    "--network_module", "networks.lora",
    "--learning_rate", str(learning_rate),
    "--max_train_steps", str(max_train_steps),
    "--resolution", resolution,
    "--train_batch_size", str(train_batch_size),
    "--network_alpha", str(network_alpha),
    "--mixed_precision", mixed_precision,
    "--save_model_as", "safetensors",
    "--cache_latents",
    "--optimizer_type", "AdamW8bit",
    "--xformers",
    "--bucket_no_upscale"
]

# This function will be called to run the actual training
def main():
    try:
        # Create accelerate config directory if it doesn't exist
        os.makedirs(os.path.expanduser("~/.cache/huggingface/accelerate"), exist_ok=True)
        
        # Create default config if needed
        config_path = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write('''compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false''')
        
        # Verify train_network.py exists
        train_script = os.path.join(os.getcwd(), "train_network.py")
        if not os.path.exists(train_script):
            print(f"ERROR: train_network.py not found at {{train_script}}")
            return 1
        
        # Run the training directly
        print("Starting training with accelerate...")
        command = ["accelerate", "launch", "--mixed_precision=fp16", "train_network.py"] + args
        print(f"Running command: {{' '.join(command)}}")
        result = subprocess.run(command, check=True)
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {{e.returncode}}: {{e}}")
        return e.returncode
    except Exception as e:
        print(f"Training failed: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    try:
        with open(train_py_path, "w", encoding="utf-8") as f:
            f.write(train_py_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Windows training script: {e}")
    
    return train_py_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. Generate platform-specific scripts with safety measures
def generate_training_script(job_dir, extracted_folder, req, use_safe_training=True):
    """Generate appropriate training script based on platform with additional safety measures"""
    # Use force_gpu parameter to determine if we should skip CPU fallback
    skip_cpu_fallback = req.force_gpu
    if IS_WINDOWS:
        # Windows batch script
        script_path = os.path.join(job_dir, "run_lora.bat")
        venv_activate = os.path.abspath(os.path.join(VENV_PATH, "Scripts", "activate.bat"))
        
        # For Windows, create a Python script to handle the training
        train_py_path = create_windows_training_script(job_dir, extracted_folder, req)
        
        # Add cuDNN safety measures
        safety_content = """
REM -----------------------------------------------------------------------------
REM Set environment variables to help with cuDNN issues
SET PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SET CUDA_LAUNCH_BLOCKING=1
SET PYTORCH_NO_CUDA_MEMORY_CACHING=1
"""
        
        script_content = f"""@echo off
REM -----------------------------------------------------------------------------
REM 1. Activate the Python venv
call "{venv_activate}"
cd /d "{job_dir}"

{safety_content if use_safe_training else ""}

REM -----------------------------------------------------------------------------
REM 2. Set environment variables
SET PYTHONPATH=.
SET HF_HOME=huggingface
SET S3_BUCKET_NAME=angelo-ai
SET FOLDER_NAME={req.folder_name}
SET LORA_NAME={req.lora_name}
SET MODEL_PATH={job_dir}/output/{req.lora_name}.safetensors

echo Starting LoRA training...
echo Working directory: {job_dir}
echo Train data directory: {extracted_folder}
echo Output directory: {job_dir}/output

REM -----------------------------------------------------------------------------
REM 3. Run the training script
"""
        
        # Add fallback mechanisms if safe training is enabled
        if use_safe_training:
            script_content += f"""
echo Attempt 1: Running with normal settings...
python "{train_py_path}"
if %errorlevel% neq 0 (
  echo First attempt failed, trying with cuDNN disabled...
  SET PYTORCH_CUDNN_ENABLED=0
  SET TORCH_USE_CUDNN=0
  echo Attempt 2: Running with cuDNN disabled...
  
  python -c "
import sys
import os
import subprocess

# Create GPU-only accelerate config (no cuDNN)
os.makedirs(os.path.expanduser('~/.cache/huggingface/accelerate'), exist_ok=True)
no_cudnn_config = os.path.expanduser('~/.cache/huggingface/accelerate/no_cudnn_config.yaml')
with open(no_cudnn_config, 'w') as f:
    f.write('''compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false''')

args = [
    'accelerate', 'launch',
    '--config_file', no_cudnn_config,
    'train_network.py',
    '--pretrained_model_name_or_path', '{req.pretrained_model}',
    '--train_data_dir', r'{extracted_folder}',
    '--output_dir', r'{job_dir}\\output',
    '--network_module', 'networks.lora',
    '--learning_rate', '{req.learning_rate}',
    '--max_train_steps', '{req.max_train_steps}',
    '--resolution', '{req.resolution}',
    '--train_batch_size', '{req.train_batch_size}',
    '--network_alpha', '{req.network_alpha}',
    '--mixed_precision', '{req.mixed_precision}',
    '--save_model_as', 'safetensors',
    '--cache_latents',
    '--optimizer_type', 'AdamW',
    '--xformers',
    '--bucket_no_upscale'
]

sys.exit(subprocess.call(args))
"
  if %errorlevel% neq 0 (
    echo Second attempt failed, trying with CPU...
    {f'''
    echo GPU training failed and CPU fallback is disabled (force_gpu=true)
    echo Training failed - please check GPU/CUDA compatibility or enable CPU fallback
    exit /b 1
    ''' if skip_cpu_fallback else '''
    SET PYTORCH_CUDA_ALLOC_CONF=
    SET CUDA_LAUNCH_BLOCKING=
    SET PYTORCH_NO_CUDA_MEMORY_CACHING=
    SET PYTORCH_CUDNN_ENABLED=
    SET TORCH_USE_CUDNN=
    echo Attempt 3: Running on CPU (slower but more compatible)...
    
    python -c "
import sys
import os
import subprocess

# Create CPU-specific accelerate config
os.makedirs(os.path.expanduser('~/.cache/huggingface/accelerate'), exist_ok=True)
cpu_config = os.path.expanduser('~/.cache/huggingface/accelerate/cpu_config.yaml')
with open(cpu_config, 'w') as f:
    f.write('''compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: no
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true''')

args = [
    'accelerate', 'launch',
    '--config_file', cpu_config,
    'train_network.py',
    '--pretrained_model_name_or_path', '{req.pretrained_model}',
    '--train_data_dir', r'{extracted_folder}',
    '--output_dir', r'{job_dir}\\output',
    '--network_module', 'networks.lora',
    '--learning_rate', '{req.learning_rate}',
    '--max_train_steps', '500',
    '--resolution', '512,512',
    '--train_batch_size', '{req.train_batch_size}',
    '--network_alpha', '{req.network_alpha}',
    '--mixed_precision', 'no',
    '--save_model_as', 'safetensors',
    '--cache_latents',
    '--optimizer_type', 'AdamW',
    '--bucket_no_upscale'
]

sys.exit(subprocess.call(args))
"
    if %errorlevel% neq 0 (
      echo All training attempts failed.
      exit /b 1
    )
    ''')}
  )
)
"""
        else:
            script_content += f"""
python "{train_py_path}"
if %errorlevel% neq 0 (
  echo ERROR: Training failed
  exit /b 1
)
"""
        
        # Continue with S3 upload code
        script_content += """
echo Training successful!

REM -----------------------------------------------------------------------------
REM 4. Upload to S3
echo Uploading LoRA model to S3...
"""
        
        # Use utf-8 encoding specifically for Windows
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
        except Exception as e:
            # Fallback to ASCII if utf-8 fails
            try:
                # Replace any potential non-ASCII characters
                ascii_content = script_content.encode('ascii', 'replace').decode('ascii')
                with open(script_path, "w", encoding="ascii") as f:
                    f.write(ascii_content)
            except Exception as inner_e:
                raise HTTPException(status_code=500, detail=f"Failed to write script: {e}. Fallback also failed: {inner_e}")
    else:
        # Unix/Linux bash script
        script_path = os.path.join(job_dir, "run_lora.sh")
        # Use absolute path to virtual environment
        venv_activate = os.path.join(VENV_PATH, "bin/activate")
        
        # Add cuDNN safety measures
        safety_content = """
# ─────────────────────────────────────────────────────────────────────────────
# Set environment variables to help with cuDNN issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Add custom cuDNN directory to library path if it exists
if [ -d "$HOME/.cudnn/lib" ]; then
  export LD_LIBRARY_PATH="$HOME/.cudnn/lib:$LD_LIBRARY_PATH"
  echo "Added custom cuDNN path to LD_LIBRARY_PATH"
fi
"""
        
        script_content = f"""#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 1. Activate the Python venv
if [ -f "{venv_activate}" ]; then
    source "{venv_activate}"
else
    echo "WARNING: Venv not found at {venv_activate}, using system Python"
fi

cd "{job_dir}"

{safety_content if use_safe_training else ""}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Set environment variables
export PYTHONPATH=.
export HF_HOME=huggingface
export S3_BUCKET_NAME=angelo-ai
export FOLDER_NAME="{req.folder_name}"
export LORA_NAME="{req.lora_name}"
export MODEL_PATH="{job_dir}/output/{req.lora_name}.safetensors"
export AWS_DEFAULT_REGION="us-west-1"

echo "Starting LoRA training..."
echo "Working directory: {job_dir}"
echo "Train data directory: {extracted_folder}"
echo "Output directory: {job_dir}/output"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run the training script
"""
        
        # Add fallback mechanisms if safe training is enabled
        if use_safe_training:
            script_content += f"""
# First try: Run with normal settings
echo "Attempt 1: Running with normal settings..."
accelerate launch --mixed_precision=fp16 train_network.py \
  --pretrained_model_name_or_path="{req.pretrained_model}" \
  --train_data_dir="{extracted_folder}" \
  --output_dir="{job_dir}/output" \
  --network_module=networks.lora \
  --learning_rate={req.learning_rate} \
  --max_train_steps={req.max_train_steps} \
  --resolution={req.resolution} \
  --train_batch_size={req.train_batch_size} \
  --network_alpha={req.network_alpha} \
  --mixed_precision={req.mixed_precision} \
  --save_model_as=safetensors \
  --cache_latents \
  --optimizer_type=AdamW \
  --xformers \
  --bucket_no_upscale

if [ $? -ne 0 ]; then
  echo "First attempt failed, trying with cuDNN disabled..."
  # Second try: Disable cuDNN more forcefully
  export PYTORCH_CUDNN_ENABLED=0
  export TORCH_USE_CUDNN=0
  # Remove cuDNN from LD_LIBRARY_PATH if it exists
  if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "cudnn" | tr '\n' ':')
  fi
  # Create GPU-only accelerate config (no cuDNN)
  mkdir -p ~/.cache/huggingface/accelerate
  cat > ~/.cache/huggingface/accelerate/no_cudnn_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
  echo "Attempt 2: Running with cuDNN disabled..."
  accelerate launch --config_file ~/.cache/huggingface/accelerate/no_cudnn_config.yaml \
    train_network.py \
    --pretrained_model_name_or_path="{req.pretrained_model}" \
    --train_data_dir="{extracted_folder}" \
    --output_dir="{job_dir}/output" \
    --network_module=networks.lora \
    --learning_rate={req.learning_rate} \
    --max_train_steps={req.max_train_steps} \
    --resolution={req.resolution} \
    --train_batch_size={req.train_batch_size} \
    --network_alpha={req.network_alpha} \
    --mixed_precision={req.mixed_precision} \
    --save_model_as=safetensors \
    --cache_latents \
    --optimizer_type=AdamW \
    --xformers \
    --bucket_no_upscale
  
  if [ $? -ne 0 ]; then
    echo "Second attempt failed, trying with CPU..."
    # Third try: Use CPU
    {f'''
    echo "❌ GPU training failed and CPU fallback is disabled (force_gpu=true)"
    echo "Training failed - please check GPU/CUDA compatibility or enable CPU fallback"
    exit 1
    ''' if skip_cpu_fallback else '''
    unset PYTORCH_CUDA_ALLOC_CONF
    unset CUDA_LAUNCH_BLOCKING
    unset PYTORCH_NO_CUDA_MEMORY_CACHING
    unset PYTORCH_CUDNN_ENABLED
    unset TORCH_USE_CUDNN
    echo "Attempt 3: Running on CPU (slower but more compatible)..."
    # Create CPU-specific accelerate config
    mkdir -p ~/.cache/huggingface/accelerate
    cat > ~/.cache/huggingface/accelerate/cpu_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: no
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
EOF
    
    accelerate launch --config_file ~/.cache/huggingface/accelerate/cpu_config.yaml \\
      train_network.py \\
      --pretrained_model_name_or_path="{req.pretrained_model}" \\
      --train_data_dir="{extracted_folder}" \\
      --output_dir="{job_dir}/output" \\
      --network_module=networks.lora \\
      --learning_rate={req.learning_rate} \\
      --max_train_steps=500 \\
      --resolution=512,512 \\
      --train_batch_size={req.train_batch_size} \\
      --network_alpha={req.network_alpha} \\
      --mixed_precision=no \\
      --save_model_as=safetensors \\
      --cache_latents \\
      --optimizer_type=AdamW \\
      --bucket_no_upscale
    
    if [ $? -ne 0 ]; then
      echo "All training attempts failed."
      exit 1
    fi
    '''}
  fi
fi
"""
        else:
            script_content += f"""
accelerate launch --mixed_precision={req.mixed_precision} train_network.py \\
  --pretrained_model_name_or_path="{req.pretrained_model}" \\
  --train_data_dir="{extracted_folder}" \\
  --output_dir="{job_dir}/output" \\
  --network_module=networks.lora \\
  --learning_rate={req.learning_rate} \\
  --max_train_steps={req.max_train_steps} \\
  --resolution={req.resolution} \\
  --train_batch_size={req.train_batch_size} \\
  --network_alpha={req.network_alpha} \\
  --mixed_precision={req.mixed_precision} \\
  --save_model_as=safetensors \\
  --cache_latents \\
  --optimizer_type=AdamW \\
  --xformers \\
  --bucket_no_upscale

if [ $? -ne 0 ]; then
  echo "ERROR: Training failed"
  exit 1
fi
"""
        
        # Continue with the S3 upload code (existing code)
        script_content += """
echo "Training successful!"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Find the safetensors file
OUTPUT_DIR="${job_dir}/output"
MODEL_FILE=$(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -n 1)

if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: No .safetensors found in $OUTPUT_DIR" >&2
  echo "Contents of output directory:"
  ls -la "$OUTPUT_DIR"
  exit 1
fi

echo "Found model at: $MODEL_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Upload using Boto3 (Python snippet)
export MODEL_PATH="$MODEL_FILE"
"""
    
    # Write the script
    try:
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Set execute permissions on Unix systems
        if not IS_WINDOWS:
            os.chmod(script_path, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write script: {e}")
    
    return script_path

# ─────────────────────────────────────────────────────────────────────────────
# 7. POST /train_lora
@app.post("/train_lora")
async def train_lora(req: LoRATrainRequest):
    # Setup cuDNN environment if using safe training
    if req.use_safe_training:
        setup_cudnn_environment()
    
    # 7.a. Validate folder_name
    if "/" in req.folder_name or "\\" in req.folder_name:
        raise HTTPException(status_code=400, detail="folder_name must not contain path separators")

    # 7.b. Download & extract ZIP
    extracted_folder = os.path.join(TRAIN_BASE, req.folder_name)
    try:
        download_and_extract_zip(req.image_zip_url, extracted_folder)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected ZIP error: {e}")

    # 7.c. Ensure folder is not empty
    if not os.path.isdir(extracted_folder) or len(os.listdir(extracted_folder)) == 0:
        raise HTTPException(status_code=400, detail="ZIP extracted but found no files")

    # 7.d. Create job directory
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Create output directory
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 7.e. Setup training environment (copy required files)
    try:
        setup_training_environment(job_dir)
    except HTTPException as he:
        raise he

    # 7.f. Generate platform-specific script with safety measures if requested
    run_script_path = generate_training_script(
        job_dir, 
        extracted_folder, 
        req, 
        use_safe_training=req.use_safe_training
    )

    # 7.g. Launch training in background, log to training.log
    log_file_path = os.path.join(job_dir, "training.log")
    try:
        log_file = open(log_file_path, "w")
        
        if IS_WINDOWS:
            # On Windows, use different approach to run batch file
            subprocess.Popen(
                ["cmd", "/c", run_script_path],
                cwd=job_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=True
            )
        else:
            # On Unix/Linux
            subprocess.Popen(
                ["/usr/bin/env", "bash", run_script_path],
                cwd=job_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return {
        "status": "training_started",
        "job_id": job_id,
        "folder": req.folder_name,
        "lora_name": req.lora_name,
        "log_file": log_file_path,
        "safe_training": req.use_safe_training
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. GET /job_status/{job_id}
@app.get("/job_status/{job_id}")
async def job_status(job_id: str):
    # Basic validation
    if not job_id or not all(c.isalnum() or c == "-" for c in job_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID format (must be alphanumeric + hyphens only)",
        )

    # Check job exists
    job_dir = os.path.join(JOBS_ROOT, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")

    # Get log file
    log_file_path = os.path.join(job_dir, "training.log")
    log_content = ""
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, "r") as f:
                log_content = f.read()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read log file: {e}",
            }

    # Check for output directory and safetensors file
    output_dir = os.path.join(job_dir, "output")
    has_output_dir = os.path.isdir(output_dir)

    # Look for any .safetensors files
    lora_files = []
    if has_output_dir:
        lora_files = [
            f for f in os.listdir(output_dir) if f.endswith(".safetensors")
        ]

    # Error detection patterns
    error_patterns = [
        "Error while finding module specification",
        "ModuleNotFoundError",
        "No module named",
        "ImportError",
        "ERROR:",
        "Exception in",
        "Traceback (most recent call last)",
        "Failed to",
        "Could not",
        "Cannot",
        "No such file or directory",
        "CalledProcessError"
    ]
    
    # Check if any error pattern exists in the log
    has_error = any(pattern in log_content for pattern in error_patterns)

    # Determine status based on the artifacts we found and logs
    if lora_files:
        status = "completed"
    elif "Upload successful" in log_content:
        status = "uploaded"
    elif has_error:
        status = "failed"
    elif os.path.exists(os.path.join(job_dir, "run_lora.sh")) or os.path.exists(os.path.join(job_dir, "run_lora.bat")):
        status = "training"
    else:
        status = "unknown"

    return {
        "job_id": job_id,
        "status": status,
        "has_output_dir": has_output_dir,
        "lora_files": lora_files,
        "log": log_content[-4000:] if log_content else "",  # Last 4K of log
    }