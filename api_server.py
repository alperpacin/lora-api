import os
import sys
import uuid
import shutil
import tempfile
import zipfile
import subprocess
import platform
import requests
import json
import psutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict

app = FastAPI()

# Detect operating system
IS_WINDOWS = platform.system() == "Windows"

# ─────────────────────────────────────────────────────────────────────────────
# Directory constants
USER_HOME = os.path.expanduser("~")
TRAIN_BASE = os.path.join(USER_HOME, "lora_data/train_images")
JOBS_ROOT = os.path.join(USER_HOME, "lora_data/lora_jobs")

# Model directories - check multiple possible locations
POSSIBLE_MODEL_PATHS = [
    "/workspace/lora-api/sd-models",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-models"),
    os.path.join(USER_HOME, "sd-models"),
    "/models/sd-models"
]

# Path to the Python virtual environment
VENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
# Path to the training scripts
TRAINING_SCRIPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

os.makedirs(TRAIN_BASE, exist_ok=True)
os.makedirs(JOBS_ROOT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────
class LoRATrainRequest(BaseModel):
    image_zip_url: HttpUrl
    folder_name: str
    lora_name: str
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    network_dim: int = 32  # LoRA rank (typically 4-128)
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    resolution: str = "512,512"
    train_batch_size: int = 1
    network_alpha: int = 128
    mixed_precision: Optional[str] = "fp16"
    use_safe_training: bool = True
    force_gpu: bool = False
    # NEW: Additional training parameters
    gradient_accumulation_steps: int = 1
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    save_every_n_steps: Optional[int] = None
    caption_extension: str = ".txt"
    seed: Optional[int] = None
    clip_skip: int = 1
    use_wandb: bool = False
    wandb_project: Optional[str] = None

class JobMetadata(BaseModel):
    job_id: str
    created_at: datetime
    status: str
    folder_name: str
    lora_name: str
    model_path: str
    is_sdxl: bool
    network_dim: int
    total_steps: int
    current_step: int = 0
    loss: Optional[float] = None

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Job tracking system
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_JOBS: Dict[str, JobMetadata] = {}

def save_job_metadata(job_id: str, metadata: JobMetadata):
    """Save job metadata to disk and memory"""
    ACTIVE_JOBS[job_id] = metadata
    metadata_path = os.path.join(JOBS_ROOT, job_id, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata.dict(), f, default=str)

def load_job_metadata(job_id: str) -> Optional[JobMetadata]:
    """Load job metadata from disk or memory"""
    if job_id in ACTIVE_JOBS:
        return ACTIVE_JOBS[job_id]
    
    metadata_path = os.path.join(JOBS_ROOT, job_id, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            data = json.load(f)
            return JobMetadata(**data)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Training progress parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_training_progress(log_content: str) -> Dict:
    """Parse training log to extract progress information"""
    lines = log_content.split('\n')
    progress = {
        "current_step": 0,
        "total_steps": 0,
        "current_loss": None,
        "learning_rate": None,
        "eta": None
    }
    
    for line in reversed(lines):
        # Look for training progress lines
        if "Training:" in line and "loss=" in line:
            # Extract step information
            if "[" in line and "]" in line:
                try:
                    step_info = line.split('[')[1].split(']')[0]
                    if "/" in step_info:
                        current, total = step_info.split('/')
                        progress["current_step"] = int(current.strip())
                        progress["total_steps"] = int(total.strip())
                except:
                    pass
            
            # Extract loss
            if "loss=" in line:
                try:
                    loss_str = line.split("loss=")[1].split()[0].rstrip(']')
                    progress["current_loss"] = float(loss_str)
                except:
                    pass
            
            break
    
    return progress

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Find and verify model path
# ─────────────────────────────────────────────────────────────────────────────
def find_model_path(model_ref: str) -> tuple[str, bool]:
    """
    Find and verify model path. Returns (resolved_path, is_sdxl)
    """
    # Check if it's a local model reference
    if model_ref.startswith("sd-models/"):
        model_filename = model_ref.replace("sd-models/", "")
        
        # Try to find the model in possible locations
        for base_path in POSSIBLE_MODEL_PATHS:
            if os.path.exists(base_path):
                model_full_path = os.path.join(base_path, model_filename)
                if os.path.exists(model_full_path):
                    # Detect if it's an SDXL model
                    is_sdxl = "xl" in model_filename.lower() or "sdxl" in model_filename.lower()
                    return model_full_path, is_sdxl
        
        # If not found, list available models for better error message
        available_models = []
        for base_path in POSSIBLE_MODEL_PATHS:
            if os.path.exists(base_path):
                try:
                    models = [f for f in os.listdir(base_path) if f.endswith('.safetensors') or f.endswith('.ckpt')]
                    available_models.extend(models)
                except:
                    pass
        
        error_msg = f"Model '{model_filename}' not found in any of these locations: {', '.join(POSSIBLE_MODEL_PATHS)}"
        if available_models:
            error_msg += f"\n\nAvailable models: {', '.join(set(available_models))}"
        
        raise HTTPException(status_code=400, detail=error_msg)
    
    else:
        # It's a HuggingFace model ID
        is_sdxl = "sdxl" in model_ref.lower() or "xl" in model_ref.lower()
        return model_ref, is_sdxl

# ─────────────────────────────────────────────────────────────────────────────
# Setup cuDNN environment
# ─────────────────────────────────────────────────────────────────────────────
def setup_cudnn_environment():
    """Set up cuDNN environment variables for better compatibility"""
    cudnn_lib_dir = os.path.join(USER_HOME, ".cudnn", "lib")
    os.makedirs(cudnn_lib_dir, exist_ok=True)
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    
    if os.path.exists(cudnn_lib_dir):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if cudnn_lib_dir not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_dir}:{current_ld_path}"
        
        os.environ["CUDNN_LIBRARY"] = cudnn_lib_dir
        os.environ["CUDNN_INCLUDE_DIR"] = os.path.join(USER_HOME, ".cudnn", "include")
        os.environ["CUDNN_LIBRARY_PATH"] = cudnn_lib_dir
    
    return cudnn_lib_dir

# ─────────────────────────────────────────────────────────────────────────────
# Helper: download & extract ZIP
# ─────────────────────────────────────────────────────────────────────────────
def download_and_extract_zip(zip_url: str, target_folder: str):
    """Downloads and extracts ZIP file"""
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
# NEW: Validate training images
# ─────────────────────────────────────────────────────────────────────────────
def validate_training_images(folder_path: str) -> Dict:
    """Validate training images and captions"""
    validation_result = {
        "total_images": 0,
        "valid_images": 0,
        "missing_captions": [],
        "invalid_images": [],
        "image_sizes": {},
        "warnings": []
    }
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's an image
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_formats:
            validation_result["total_images"] += 1
            
            # Check if image is valid
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    validation_result["valid_images"] += 1
                    size_key = f"{img.width}x{img.height}"
                    validation_result["image_sizes"][size_key] = validation_result["image_sizes"].get(size_key, 0) + 1
                    
                    # Check for caption file
                    caption_path = os.path.splitext(file_path)[0] + ".txt"
                    if not os.path.exists(caption_path):
                        validation_result["missing_captions"].append(filename)
            except Exception as e:
                validation_result["invalid_images"].append({"filename": filename, "error": str(e)})
    
    # Add warnings
    if validation_result["total_images"] < 5:
        validation_result["warnings"].append("Less than 5 images found. Consider adding more for better results.")
    
    if len(validation_result["image_sizes"]) > 1:
        validation_result["warnings"].append("Multiple image sizes detected. Consider using uniform sizes.")
    
    return validation_result

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Copy required training files to job directory
# ─────────────────────────────────────────────────────────────────────────────
def setup_training_environment(job_dir: str):
    """
    Copy the correct training script(s) into the freshly-created job folder.
    We always rename the main script to train_network.py inside the job dir
    so the launch command stays the same.
    """
    # Everything we want to end up inside <job_dir>
    required = ["train_network.py", "train.sh"]

    base_scripts = TRAINING_SCRIPTS_PATH          # e.g.  .../scripts
    stable_dir   = os.path.join(base_scripts, "stable")

    for fname in required:
        # pick the right source location
        if fname == "train_network.py":
            src = os.path.join(stable_dir, fname)     # <scripts>/stable/train_network.py
        else:  # train.sh or anything else
            src = os.path.join(base_scripts, fname)   # <scripts>/train.sh

        dst = os.path.join(job_dir, fname)

        # sanity-check & copy
        if not os.path.exists(src):
            if fname == "train_network.py":           # mandatory
                raise HTTPException(
                    status_code=500,
                    detail=f"Required script {src} not found."
                )
            # optional helpers can just be skipped
            continue

        try:
            shutil.copyfile(src, dst)
            if not IS_WINDOWS and fname.endswith(".sh"):
                os.chmod(dst, 0o755)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to copy {fname}: {e}"
            )
# ─────────────────────────────────────────────────────────────────────────────
# Generate platform-specific scripts with safety measures
# ─────────────────────────────────────────────────────────────────────────────
def generate_training_script(job_dir, extracted_folder, req, resolved_model_path, is_sdxl, use_safe_training=True):
    """Generate appropriate training script based on platform"""
    skip_cpu_fallback = req.force_gpu
    
    # Common training arguments
    common_args = f"""
  --pretrained_model_name_or_path="{resolved_model_path}" \\
  --train_data_dir="{extracted_folder}" \\
  --output_dir="{job_dir}/output" \\
  --output_name="{req.lora_name}" \\
  --network_module=networks.lora \\
  --network_dim={req.network_dim} \\
  --network_alpha={req.network_alpha} \\
  --learning_rate={req.learning_rate} \\
  --max_train_steps={req.max_train_steps} \\
  --resolution={req.resolution} \\
  --train_batch_size={req.train_batch_size} \\
  --mixed_precision={req.mixed_precision} \\
  --save_model_as=safetensors \\
  --save_every_n_epochs=1 \\
  --cache_latents \\
  --optimizer_type=AdamW8bit \\
  --xformers \\
  --bucket_no_upscale \\
  --gradient_accumulation_steps={req.gradient_accumulation_steps} \\
  --lr_scheduler={req.lr_scheduler} \\
  --lr_warmup_steps={req.lr_warmup_steps} \\
  --caption_extension={req.caption_extension} \\
  --clip_skip={req.clip_skip}"""
    
    # Add optional parameters
    if req.save_every_n_steps:
        common_args += f" \\\n  --save_every_n_steps={req.save_every_n_steps}"
    
    if req.seed is not None:
        common_args += f" \\\n  --seed={req.seed}"
    
    if req.use_wandb and req.wandb_project:
        common_args += f" \\\n  --log_with=wandb \\\n  --wandb_project={req.wandb_project}"
    
    if is_sdxl:
        common_args += " \\\n  --no_half_vae \\\n  --sdxl"
    
    if IS_WINDOWS:
        # Windows batch script
        script_path = os.path.join(job_dir, "run_lora.bat")
        venv_activate = os.path.abspath(os.path.join(VENV_PATH, "Scripts", "activate.bat"))
        
        safety_content = """
REM Set environment variables to help with cuDNN issues
SET PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SET CUDA_LAUNCH_BLOCKING=1
SET PYTORCH_NO_CUDA_MEMORY_CACHING=1
"""
        
        script_content = f"""@echo off
REM Activate the Python venv
call "{venv_activate}"
cd /d "{job_dir}"

{safety_content if use_safe_training else ""}

REM Set environment variables
SET PYTHONPATH=.
SET HF_HOME=huggingface
SET MODEL_PATH={job_dir}/output/{req.lora_name}.safetensors

echo Starting LoRA training...
echo Model: {resolved_model_path}
echo Working directory: {job_dir}
echo Train data directory: {extracted_folder}
echo Output directory: {job_dir}/output
echo Network dimension: {req.network_dim}
echo Learning rate: {req.learning_rate}

REM Run the training script
accelerate launch --mixed_precision={req.mixed_precision} train_network.py ^
{common_args.replace(chr(92), '^')}

if %errorlevel% neq 0 (
  echo ERROR: Training failed
  exit /b 1
)

echo Training successful!
"""
        
    else:
        # Unix/Linux bash script
        script_path = os.path.join(job_dir, "run_lora.sh")
        venv_activate = os.path.join(VENV_PATH, "bin/activate")
        
        safety_content = """
# Set environment variables to help with cuDNN issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

if [ -d "$HOME/.cudnn/lib" ]; then
  export LD_LIBRARY_PATH="$HOME/.cudnn/lib:$LD_LIBRARY_PATH"
  echo "Added custom cuDNN path to LD_LIBRARY_PATH"
fi
"""
        
        script_content = f"""#!/usr/bin/env bash
# Activate the Python venv
if [ -f "{venv_activate}" ]; then
    source "{venv_activate}"
else
    echo "WARNING: Venv not found at {venv_activate}, using system Python"
fi

cd "{job_dir}"

{safety_content if use_safe_training else ""}

# Set environment variables
export PYTHONPATH=.
export HF_HOME=huggingface
export MODEL_PATH="{job_dir}/output/{req.lora_name}.safetensors"

echo "Starting LoRA training..."
echo "Model: {resolved_model_path}"
echo "Working directory: {job_dir}"
echo "Train data directory: {extracted_folder}"
echo "Output directory: {job_dir}/output"
echo "Network dimension: {req.network_dim}"
echo "Learning rate: {req.learning_rate}"

# Run the training script
export PYTHONPATH="/workspace/lora-api:$PYTHONPATH"
accelerate launch --mixed_precision={req.mixed_precision} train_network.py \
{common_args.lstrip()}

if [ $? -ne 0 ]; then
  echo "ERROR: Training failed"
  exit 1
fi

echo "Training successful!"
"""
    
    # Write the script
    try:
        with open(script_path, "w") as f:
            f.write(script_content)
        
        if not IS_WINDOWS:
            os.chmod(script_path, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write script: {e}")
    
    return script_path

# ─────────────────────────────────────────────────────────────────────────────
# POST /train_lora
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/train_lora")
async def train_lora(req: LoRATrainRequest):
    # Setup cuDNN environment if using safe training
    if req.use_safe_training:
        setup_cudnn_environment()
    
    # Validate folder_name
    if "/" in req.folder_name or "\\" in req.folder_name:
        raise HTTPException(status_code=400, detail="folder_name must not contain path separators")
    
    # Find and verify model path
    try:
        resolved_model_path, is_sdxl = find_model_path(req.pretrained_model)
    except HTTPException as he:
        raise he
    
    # Download & extract ZIP
    extracted_folder = os.path.join(TRAIN_BASE, req.folder_name)
    try:
        download_and_extract_zip(req.image_zip_url, extracted_folder)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected ZIP error: {e}")
    
    # Ensure folder is not empty
    if not os.path.isdir(extracted_folder) or len(os.listdir(extracted_folder)) == 0:
        raise HTTPException(status_code=400, detail="ZIP extracted but found no files")
    
    # NEW: Validate training images
    validation_result = validate_training_images(extracted_folder)
    if validation_result["total_images"] == 0:
        raise HTTPException(status_code=400, detail="No valid images found in the extracted folder")
    
    # Create job directory
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Create output directory
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # NEW: Save job metadata
    metadata = JobMetadata(
        job_id=job_id,
        created_at=datetime.now(),
        status="preparing",
        folder_name=req.folder_name,
        lora_name=req.lora_name,
        model_path=resolved_model_path,
        is_sdxl=is_sdxl,
        network_dim=req.network_dim,
        total_steps=req.max_train_steps
    )
    save_job_metadata(job_id, metadata)
    
    # Setup training environment (copy required files)
    try:
        setup_training_environment(job_dir)
    except HTTPException as he:
        raise he
    
    # Generate platform-specific script
    run_script_path = generate_training_script(
        job_dir, 
        extracted_folder, 
        req,
        resolved_model_path,
        is_sdxl,
        use_safe_training=req.use_safe_training
    )
    
    # Launch training in background
    log_file_path = os.path.join(job_dir, "training.log")
    try:
        log_file = open(log_file_path, "w")
        
        if IS_WINDOWS:
            subprocess.Popen(
                ["cmd", "/c", run_script_path],
                cwd=job_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=True
            )
        else:
            subprocess.Popen(
                ["/usr/bin/env", "bash", run_script_path],
                cwd=job_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        
        # Update status
        metadata.status = "training"
        save_job_metadata(job_id, metadata)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")
    
    return {
        "status": "training_started",
        "job_id": job_id,
        "folder": req.folder_name,
        "lora_name": req.lora_name,
        "log_file": log_file_path,
        "model_path": resolved_model_path,
        "is_sdxl": is_sdxl,
        "network_dim": req.network_dim,
        "safe_training": req.use_safe_training,
        "validation": validation_result
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /job_status/{job_id}
# ─────────────────────────────────────────────────────────────────────────────
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
    
    # Load metadata
    metadata = load_job_metadata(job_id)
    
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
    
    # Determine status
    if lora_files:
        status = "completed"
    elif "Training successful!" in log_content:
        status = "completed"
    elif has_error:
        status = "failed"
    elif "Starting LoRA training..." in log_content:
        status = "training"
    else:
        status = "preparing"
    
    # Get file sizes for LoRA files
    lora_file_info = []
    if lora_files:
        for lora_file in lora_files:
            file_path = os.path.join(output_dir, lora_file)
            file_size = os.path.getsize(file_path)
            lora_file_info.append({
                "filename": lora_file,
                "size_mb": round(file_size / (1024 * 1024), 2)
            })
    
    # NEW: Parse training progress
    progress = parse_training_progress(log_content) if status == "training" else None
    
    # Update metadata if available
    if metadata and status != metadata.status:
        metadata.status = status
        if progress:
            metadata.current_step = progress["current_step"]
            metadata.loss = progress["current_loss"]
        save_job_metadata(job_id, metadata)
    
    response = {
        "job_id": job_id,
        "status": status,
        "has_output_dir": has_output_dir,
        "lora_files": lora_file_info,
        "log": log_content[-4000:] if log_content else "",  # Last 4K of log
    }
    
    # Add metadata and progress if available
    if metadata:
        response["metadata"] = metadata.dict()
    if progress:
        response["progress"] = progress
    
    return response

# ─────────────────────────────────────────────────────────────────────────────
# GET /list_models
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/list_models")
async def list_models():
    """List all available models"""
    models = []
    
    for base_path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(base_path):
            try:
                for filename in os.listdir(base_path):
                    if filename.endswith('.safetensors') or filename.endswith('.ckpt'):
                        file_path = os.path.join(base_path, filename)
                        file_size = os.path.getsize(file_path)
                        
                        models.append({
                            "filename": filename,
                            "reference": f"sd-models/{filename}",
                            "path": file_path,
                            "size_mb": round(file_size / (1024 * 1024), 2),
                            "is_sdxl": "xl" in filename.lower() or "sdxl" in filename.lower()
                        })
            except Exception as e:
                pass
    
    # Remove duplicates by filename
    seen = set()
    unique_models = []
    for model in models:
        if model["filename"] not in seen:
            seen.add(model["filename"])
            unique_models.append(model)
    
    return {
        "models": unique_models,
        "count": len(unique_models),
        "search_paths": POSSIBLE_MODEL_PATHS
    }

# ─────────────────────────────────────────────────────────────────────────────
# NEW: GET /list_jobs
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/list_jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List all training jobs with optional status filter"""
    jobs = []
    
    try:
        # Get all job directories
        job_dirs = [d for d in os.listdir(JOBS_ROOT) if os.path.isdir(os.path.join(JOBS_ROOT, d))]
        
        for job_id in job_dirs[:limit]:
            # Try to load metadata
            metadata = load_job_metadata(job_id)
            if metadata:
                if status is None or metadata.status == status:
                    jobs.append(metadata.dict())
            else:
                # Fallback to basic info if no metadata
                job_info = {
                    "job_id": job_id,
                    "status": "unknown",
                    "created_at": datetime.fromtimestamp(os.path.getctime(os.path.join(JOBS_ROOT, job_id)))
                }
                if status is None or job_info["status"] == status:
                    jobs.append(job_info)
        
        # Sort by creation date (newest first)
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}")
    
    return {
        "jobs": jobs,
        "count": len(jobs),
        "filter": {"status": status} if status else None
    }

# ─────────────────────────────────────────────────────────────────────────────
# NEW: GET /download_lora/{job_id}/{filename}
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/download_lora/{job_id}/{filename}")
async def download_lora(job_id: str, filename: str):
    """Download a trained LoRA file"""
    # Validate inputs
    if not job_id or not all(c.isalnum() or c == "-" for c in job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    
    if not filename.endswith('.safetensors'):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Construct file path
    file_path = os.path.join(JOBS_ROOT, job_id, "output", filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return file
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

# ─────────────────────────────────────────────────────────────────────────────
# NEW: DELETE /cancel_job/{job_id}
# ─────────────────────────────────────────────────────────────────────────────
@app.delete("/cancel_job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running training job"""
    # Validate job ID
    if not job_id or not all(c.isalnum() or c == "-" for c in job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    
    # Check if job exists
    job_dir = os.path.join(JOBS_ROOT, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Find and kill the process
    try:
        # Look for the training process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and job_id in ' '.join(cmdline):
                    proc.terminate()
                    proc.wait(timeout=5)
                    
                    # Update job status
                    metadata = load_job_metadata(job_id)
                    if metadata:
                        metadata.status = "cancelled"
                        save_job_metadata(job_id, metadata)
                    
                    return {"message": f"Job {job_id} cancelled successfully"}
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {"message": f"No active process found for job {job_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# NEW: POST /cleanup_old_jobs
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/cleanup_old_jobs")
async def cleanup_old_jobs(days_old: int = 7, dry_run: bool = True):
    """Clean up old job directories"""
    if days_old < 1:
        raise HTTPException(status_code=400, detail="days_old must be at least 1")
    
    cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    jobs_to_clean = []
    space_to_free = 0
    
    try:
        for job_id in os.listdir(JOBS_ROOT):
            job_path = os.path.join(JOBS_ROOT, job_id)
            if os.path.isdir(job_path):
                # Check creation time
                if os.path.getctime(job_path) < cutoff_time:
                    # Calculate size
                    size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(job_path)
                        for filename in filenames
                    )
                    
                    jobs_to_clean.append({
                        "job_id": job_id,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(os.path.getctime(job_path)).isoformat()
                    })
                    space_to_free += size
                    
                    # Actually delete if not dry run
                    if not dry_run:
                        shutil.rmtree(job_path)
                        # Remove from active jobs
                        if job_id in ACTIVE_JOBS:
                            del ACTIVE_JOBS[job_id]
        
        return {
            "dry_run": dry_run,
            "jobs_cleaned": len(jobs_to_clean),
            "space_freed_mb": round(space_to_free / (1024 * 1024), 2),
            "jobs": jobs_to_clean[:20]  # Limit to first 20 for response size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# NEW: GET /system_status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/system_status")
async def system_status():
    """Get system resource status"""
    try:
        # Get disk usage
        disk_usage = psutil.disk_usage(JOBS_ROOT)
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        # Get GPU info (if nvidia-ml-py is available)
        gpu_info = []
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_info.append({
                    "id": i,
                    "name": pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                    "memory_used": pynvml.nvmlDeviceGetMemoryInfo(handle).used // 1024 // 1024,
                    "memory_total": pynvml.nvmlDeviceGetMemoryInfo(handle).total // 1024 // 1024,
                    "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                })
            pynvml.nvmlShutdown()
        except:
            pass
        
        # Count active jobs
        active_jobs = sum(1 for m in ACTIVE_JOBS.values() if m.status == "training")
        
        return {
            "disk": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "percent": disk_usage.percent
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent": memory.percent
            },
            "gpus": gpu_info,
            "jobs": {
                "active": active_jobs,
                "total_tracked": len(ACTIVE_JOBS)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "platform": platform.system(),
        "venv_exists": os.path.exists(VENV_PATH),
        "scripts_path": TRAINING_SCRIPTS_PATH,
        "jobs_directory": JOBS_ROOT,
        "train_images_directory": TRAIN_BASE
    }

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Startup event to load existing jobs
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load existing job metadata on startup"""
    try:
        if os.path.exists(JOBS_ROOT):
            for job_id in os.listdir(JOBS_ROOT):
                if os.path.isdir(os.path.join(JOBS_ROOT, job_id)):
                    metadata = load_job_metadata(job_id)
                    if metadata:
                        ACTIVE_JOBS[job_id] = metadata
        print(f"Loaded {len(ACTIVE_JOBS)} existing jobs")
    except Exception as e:
        print(f"Failed to load existing jobs: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)