# File: api_server.py

import os
import uuid
import shutil
import zipfile
import tempfile
import requests
import subprocess

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional

app = FastAPI()

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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Directory constants
USER_HOME = os.path.expanduser("~")
TRAIN_BASE = os.path.join(USER_HOME, "lora_data/train_images")
JOBS_ROOT = os.path.join(USER_HOME, "lora_data/lora_jobs")

os.makedirs(TRAIN_BASE, exist_ok=True)
os.makedirs(JOBS_ROOT, exist_ok=True)


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
# 4. POST /train_lora
@app.post("/train_lora")
async def train_lora(req: LoRATrainRequest):
    # 4.a. Validate folder_name
    if "/" in req.folder_name or "\\" in req.folder_name:
        raise HTTPException(status_code=400, detail="folder_name must not contain path separators")

    # 4.b. Download & extract ZIP
    extracted_folder = os.path.join(TRAIN_BASE, req.folder_name)
    try:
        download_and_extract_zip(req.image_zip_url, extracted_folder)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected ZIP error: {e}")

    # 4.c. Ensure folder is not empty
    if not os.path.isdir(extracted_folder) or len(os.listdir(extracted_folder)) == 0:
        raise HTTPException(status_code=400, detail="ZIP extracted but found no files")

    # 4.d. Create job directory
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 4.e. Copy train.sh
    src_train_sh = os.path.abspath("train.sh")
    dst_train_sh = os.path.join(job_dir, "train.sh")
    try:
        shutil.copyfile(src_train_sh, dst_train_sh)
        os.chmod(dst_train_sh, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stage train.sh: {e}")

    # 4.f. Build run_lora.sh
    run_script = f"""#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 1. Activate the Python venv
source "{os.path.abspath('venv/bin/activate')}"
cd "{job_dir}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Set hyperparameters & environment variables
PRETRAINED_MODEL="{req.pretrained_model}"
TRAIN_DATA_DIR="{extracted_folder}"
OUTPUT_DIR="{job_dir}/output"
LR="{req.learning_rate}"
MAX_TRAIN_STEPS="{req.max_train_steps}"
RESOLUTION="{req.resolution}"
TRAIN_BATCH_SIZE="{req.train_batch_size}"
NETWORK_ALPHA="{req.network_alpha}"
MIXED_PRECISION="{req.mixed_precision}"

# Info for S3 upload
FOLDER_NAME="{req.folder_name}"
LORA_NAME="{req.lora_name}"
export S3_BUCKET_NAME
export AWS_DEFAULT_REGION
export FOLDER_NAME
export LORA_NAME

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run LoRA training
bash "{dst_train_sh}" \\
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \\
  --train_data_dir="$TRAIN_DATA_DIR" \\
  --output_dir="$OUTPUT_DIR" \\
  --network_module="lora" \\
  --learning_rate="$LR" \\
  --max_train_steps="$MAX_TRAIN_STEPS" \\
  --resolution="$RESOLUTION" \\
  --train_batch_size="$TRAIN_BATCH_SIZE" \\
  --network_alpha="$NETWORK_ALPHA" \\
  --mixed_precision="$MIXED_PRECISION"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Locate the .safetensors file
MODEL_FILE=$(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -n 1)

if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: No .safetensors found in $OUTPUT_DIR" >&2
  exit 1
fi

echo "Found model at: $MODEL_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Upload using Boto3 (Python snippet)
export MODEL_PATH="$MODEL_FILE"

python3 - << EOF
import os
import sys
import boto3
from botocore.exceptions import BotoCoreError, ClientError

model_path = os.getenv("MODEL_PATH")
bucket_name = os.getenv("S3_BUCKET_NAME")
folder = os.getenv("FOLDER_NAME")
lora_name = os.getenv("LORA_NAME")
region = os.getenv("AWS_DEFAULT_REGION")

if not model_path or not bucket_name or not folder or not lora_name or not region:
    print("ERROR: Missing required environment variables for S3 upload", file=sys.stderr)
    sys.exit(1)

# Construct the S3 key: loras/{{folder}}/{{lora_name}}.safetensors
key = f"loras/{{folder}}/{{lora_name}}.safetensors"

# Boto3 will auto‐pick up Cognito‐issued temp credentials in the environment
s3 = boto3.client("s3", region_name=region)

try:
    print(f"Uploading {{model_path}} to s3://{{bucket_name}}/{{key}} …")
    s3.upload_file(model_path, bucket_name, key)
    print("Upload successful")
except (BotoCoreError, ClientError) as e:
    print(f"ERROR: Boto3 upload failed: {{e}}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
  echo "ERROR: Python‐based S3 upload failed" >&2
  exit 1
fi

echo "S3 upload done: s3://$S3_BUCKET_NAME/loras/$FOLDER_NAME/$LORA_NAME.safetensors"

# Optionally, remove local copy:
# rm -f "$MODEL_FILE"

exit 0
"""

    # 4.g. Write run_lora.sh and chmod
    run_path = os.path.join(job_dir, "run_lora.sh")
    try:
        with open(run_path, "w") as f:
            f.write(run_script)
        os.chmod(run_path, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write run_lora.sh: {e}")

    # 4.h. Launch training in background, log to training.log
    log_file_path = os.path.join(job_dir, "training.log")
    try:
        log_file = open(log_file_path, "w")
        subprocess.Popen(
            ["/usr/bin/env", "bash", run_path],
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
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /job_status/{job_id}
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
    elif os.path.exists(os.path.join(job_dir, "run_lora.sh")):
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
