# File: api_server.py

import os
import uuid
import shutil
import zipfile
import tempfile
import requests
import subprocess

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional

app = FastAPI()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Request schema including:
#    - image_zip_url: URL to a ZIP archive of images
#    - folder_name: target folder under /mnt/data/train_images/
#    - lora_name: desired name for the resulting LoRA
#    - hyperparameters for training
# ─────────────────────────────────────────────────────────────────────────────
class LoRATrainRequest(BaseModel):
    image_zip_url: HttpUrl    # e.g. "https://example.com/user123_images.zip"
    folder_name: str          # e.g. "user123"
    lora_name: str            # e.g. "user123_lora"
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    resolution: str = "512,512"
    train_batch_size: int = 1
    network_alpha: int = 128
    mixed_precision: Optional[str] = "fp16"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Directory constants
#    TRAIN_BASE: root where all image subfolders are created/extracted
#    JOBS_ROOT: root where per-job data (scripts, logs, outputs) live
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_BASE = "/mnt/data/train_images"
JOBS_ROOT = "/mnt/data/lora_jobs"

# Ensure these directories exist
os.makedirs(TRAIN_BASE, exist_ok=True)
os.makedirs(JOBS_ROOT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper function: download a ZIP from a URL, extract into target_folder, then delete it
# ─────────────────────────────────────────────────────────────────────────────
def download_and_extract_zip(zip_url: str, target_folder: str):
    """
    Downloads the ZIP from `zip_url` into a temporary file,
    extracts its contents into `target_folder`, then removes the ZIP.
    Raises HTTPException on failure.
    """
    # 3.a. Create a temporary file path for the ZIP
    try:
        tmp_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(tmp_fd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create temp file: {e}")

    # 3.b. Download the ZIP via streaming
    try:
        with requests.get(zip_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.RequestException as e:
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)
        raise HTTPException(status_code=400, detail=f"Failed to download ZIP: {e}")

    # 3.c. Ensure the target folder is fresh
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)

    # 3.d. Extract ZIP into target_folder
    try:
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(target_folder)
    except zipfile.BadZipFile as e:
        os.remove(tmp_zip_path)
        shutil.rmtree(target_folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Downloaded file is not a valid ZIP: {e}")
    except Exception as e:
        os.remove(tmp_zip_path)
        shutil.rmtree(target_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract ZIP: {e}")
    finally:
        # 3.e. Delete the temporary ZIP file
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. POST /train_lora endpoint
#
#    Steps:
#      1. Validate folder_name
#      2. Download & extract images.zip → /mnt/data/train_images/folder_name
#      3. Create job_dir under JOBS_ROOT
#      4. Copy train.sh into job_dir
#      5. Write run_lora.sh to:
#           - activate venv
#           - set hyperparameters & environment variables
#           - run train.sh
#           - locate the resulting .safetensors
#           - upload it to S3 using AWS CLI
#      6. Spawn subprocess to run run_lora.sh in background, log to training.log
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/train_lora")
async def train_lora(req: LoRATrainRequest):
    # 4.a. Validate folder_name (no path separators)
    if "/" in req.folder_name or "\\" in req.folder_name:
        raise HTTPException(status_code=400, detail="folder_name must not contain path separators")

    # 4.b. Define extraction folder and download/extract ZIP there
    extracted_folder = os.path.join(TRAIN_BASE, req.folder_name)
    try:
        download_and_extract_zip(req.image_zip_url, extracted_folder)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during ZIP handling: {e}")

    # 4.c. Confirm the extracted folder has files
    if not os.path.isdir(extracted_folder) or len(os.listdir(extracted_folder)) == 0:
        raise HTTPException(status_code=400, detail="ZIP extracted but no files found in folder")

    # 4.d. Prepare a new job directory under JOBS_ROOT
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 4.e. Copy train.sh into job_dir and make it executable
    src_train_sh = os.path.abspath("train.sh")
    dst_train_sh = os.path.join(job_dir, "train.sh")
    try:
        shutil.copyfile(src_train_sh, dst_train_sh)
        os.chmod(dst_train_sh, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy train.sh: {e}")

    # 4.f. Build run_lora.sh content
    run_script = f"""#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 1. Activate the Python venv
source "{os.path.abspath('venv/bin/activate')}"
cd "{job_dir}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Set hyperparameters and environment variables
PRETRAINED_MODEL="{req.pretrained_model}"
TRAIN_DATA_DIR="{extracted_folder}"
OUTPUT_DIR="{job_dir}/output"
LR="{req.learning_rate}"
MAX_TRAIN_STEPS="{req.max_train_steps}"
RESOLUTION="{req.resolution}"
TRAIN_BATCH_SIZE="{req.train_batch_size}"
NETWORK_ALPHA="{req.network_alpha}"
MIXED_PRECISION="{req.mixed_precision}"

# For S3 upload:
FOLDER_NAME="{req.folder_name}"        # e.g. "user123"
LORA_NAME="{req.lora_name}"            # e.g. "user123_lora"
S3_BUCKET="${{S3_BUCKET_NAME}}"        # from env var
S3_KEY="loras/${{FOLDER_NAME}}/${{LORA_NAME}}.safetensors"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run the standard LoRA training (via train.sh)
bash "{dst_train_sh}" \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --network_module="lora" \
  --learning_rate="$LR" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --resolution="$RESOLUTION" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --network_alpha="$NETWORK_ALPHA" \
  --mixed_precision="$MIXED_PRECISION"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Locate the resulting .safetensors file
MODEL_FILE=$(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -n 1)

if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: No .safetensors file found in $OUTPUT_DIR" >&2
  exit 1
fi

echo "Uploading $MODEL_FILE to s3://$S3_BUCKET/$S3_KEY …"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Upload to S3 using AWS CLI
aws s3 cp "$MODEL_FILE" "s3://$S3_BUCKET/$S3_KEY" \
  --acl bucket-owner-full-control

if [ $? -ne 0 ]; then
  echo "ERROR: S3 upload failed" >&2
  exit 1
fi

echo "Upload complete: s3://$S3_BUCKET/$S3_KEY"

# (Optional) Remove the local model file to conserve space
# rm -f "$MODEL_FILE"

exit 0
"""

    # 4.g. Write run_lora.sh into job_dir and make it executable
    run_path = os.path.join(job_dir, "run_lora.sh")
    try:
        with open(run_path, "w") as f:
            f.write(run_script)
        os.chmod(run_path, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write run_lora.sh: {e}")

    # 4.h. Launch the training in a background subprocess, redirect stdout/stderr to training.log
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
        raise HTTPException(status_code=500, detail=f"Failed to start training subprocess: {e}")

    # 4.i. Return job_id for client to poll status
    return {"job_id": job_id, "status": "started"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /job_status/{job_id} endpoint
#
#    - Checks if /mnt/data/lora_jobs/{job_id}/output contains any .safetensors
#    - Reads training.log and returns it along with `completed` boolean
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/job_status/{job_id}")
async def job_status(job_id: str):
    job_dir = os.path.join(JOBS_ROOT, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(job_dir, "output")
    completed = (os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0)

    log_file_path = os.path.join(job_dir, "training.log")
    try:
        log_contents = open(log_file_path, "r").read()
    except FileNotFoundError:
        log_contents = ""

    return {
        "job_id": job_id,
        "completed": completed,
        "log": log_contents,
    }
