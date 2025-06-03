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
# 1. Request schema including the new fields:
#    - image_zip_url: a URL pointing to a .zip of images
#    - folder_name: the subfolder name under /mnt/data/train_images/ where images go
# ─────────────────────────────────────────────────────────────────────────────
class LoRATrainRequest(BaseModel):
    image_zip_url: HttpUrl       # e.g. "https://example.com/user123_images.zip"
    folder_name: str             # e.g. "user123"
    lora_name: str               # e.g. "user123_lora"
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    resolution: str = "1024,1024"
    train_batch_size: int = 1
    network_alpha: int = 128
    mixed_precision: Optional[str] = "fp16"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Constants for where we store things on disk during training:
#    - TRAIN_BASE: root where all image‐folders live (/mnt/data/train_images)
#    - JOBS_ROOT: where job subfolders are created (/mnt/data/lora_jobs)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_BASE = "/mnt/data/train_images"
JOBS_ROOT = "/mnt/data/lora_jobs"

# Ensure directories exist
os.makedirs(TRAIN_BASE, exist_ok=True)
os.makedirs(JOBS_ROOT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper: download a ZIP, extract to a target folder, then delete the ZIP
# ─────────────────────────────────────────────────────────────────────────────
def download_and_extract_zip(zip_url: str, target_folder: str):
    """
    Downloads the ZIP from `zip_url` into a temporary file,
    extracts all contents into `target_folder`, then removes the ZIP.
    Raises an HTTPException on failure.
    """
    # Create a temporary file path for the ZIP
    try:
        tmp_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(tmp_fd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create temp file: {e}")

    # 3.a. Download the ZIP with streaming
    try:
        with requests.get(zip_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.RequestException as e:
        # Clean up the temp file if it was created
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)
        raise HTTPException(status_code=400, detail=f"Failed to download ZIP: {e}")

    # 3.b. Ensure the target folder exists (and is empty)
    if os.path.exists(target_folder):
        # If the folder already exists, remove it entirely to avoid mixing old files
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)

    # 3.c. Extract ZIP into target_folder
    try:
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(target_folder)
    except zipfile.BadZipFile as e:
        # Clean up both the tmp ZIP and the (likely corrupted) target folder
        os.remove(tmp_zip_path)
        shutil.rmtree(target_folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Downloaded file is not a valid ZIP: {e}")
    except Exception as e:
        os.remove(tmp_zip_path)
        shutil.rmtree(target_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract ZIP: {e}")
    finally:
        # 3.d. Delete the temporary ZIP (successful or not)
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. POST /train_lora
#
#    Workflow:
#      1. Generate job_id & job_dir
#      2. Download ZIP → extract to /mnt/data/train_images/<folder_name>
#      3. Delete ZIP (handled in helper)
#      4. Copy train.sh into job_dir
#      5. Write run_lora.sh wrapper setting train_data_dir to the extracted folder
#      6. Launch subprocess.Popen([...]) to run training in background
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/train_lora")
async def train_lora(req: LoRATrainRequest):
    # 4.a. Validate folder_name: must be a single folder‐safe name (no "../" etc.)
    if "/" in req.folder_name or "\\" in req.folder_name:
        raise HTTPException(status_code=400, detail="folder_name must not contain path separators")

    # 4.b. Define where to extract images and ensure parent dir exists
    extracted_folder = os.path.join(TRAIN_BASE, req.folder_name)

    # 4.c. Download and extract images.zip → extracted_folder
    try:
        download_and_extract_zip(req.image_zip_url, extracted_folder)
    except HTTPException as he:
        # Propagate the HTTPException with the same status/detail
        raise he
    except Exception as e:
        # Catch‐all: wrap in 500
        raise HTTPException(status_code=500, detail=f"Unexpected error during ZIP handling: {e}")

    # 4.d. Now extracted_folder exists with all images; confirm it’s not empty
    if not os.path.isdir(extracted_folder) or len(os.listdir(extracted_folder)) == 0:
        raise HTTPException(status_code=400, detail="ZIP extracted but no files found in folder")

    # 4.e. Prepare a new job subdirectory under JOBS_ROOT
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 4.f. Copy train.sh into job_dir and make it executable
    src_train_sh = os.path.abspath("train.sh")
    dst_train_sh = os.path.join(job_dir, "train.sh")
    try:
        shutil.copyfile(src_train_sh, dst_train_sh)
        os.chmod(dst_train_sh, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy train.sh: {e}")

    # 4.g. Create the run_lora.sh wrapper script inside job_dir
    #      This script sets environment variables (including TRAIN_DATA_DIR) and calls train.sh
    run_script = f"""#!/usr/bin/env bash
# Activate the Python venv
source "{os.path.abspath('venv/bin/activate')}"
cd "{job_dir}"

# Override defaults for this job
PRETRAINED_MODEL="{req.pretrained_model}"
TRAIN_DATA_DIR="{extracted_folder}"
OUTPUT_DIR="{job_dir}/output"
LR="{req.learning_rate}"
MAX_TRAIN_STEPS="{req.max_train_steps}"
RESOLUTION="{req.resolution}"
TRAIN_BATCH_SIZE="{req.train_batch_size}"
NETWORK_ALPHA="{req.network_alpha}"
MIXED_PRECISION="{req.mixed_precision}"

# Call the original train.sh with flags
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
"""
    run_path = os.path.join(job_dir, "run_lora.sh")
    try:
        with open(run_path, "w") as f:
            f.write(run_script)
        os.chmod(run_path, 0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write run_lora.sh: {e}")

    # 4.h. Finally, launch the training in a background subprocess (stdout+stderr → training.log)
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

    # 4.i. Return the job_id so client can poll /job_status/{job_id}
    return {"job_id": job_id, "status": "started"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /job_status/{job_id}
#    Same as before: check if /mnt/data/lora_jobs/{job_id}/output exists & nonempty,
#    then return "completed": True/False and the contents of training.log
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
