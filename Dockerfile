# ────────────────────────────────────────────────────────────────────────────
# STEP 1: Base image—PyTorch with Python 3.10 + CUDA 11.7
# NVIDIA’s NGC image: Ubuntu 20.04 + CUDA 11.7 + PyTorch + cuDNN
FROM nvcr.io/nvidia/pytorch:23.03-py3 AS base

# ────────────────────────────────────────────────────────────────────────────
# STEP 2: Install system packages needed by lora-scripts and FastAPI
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    wget \
    unzip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

  # ── (somewhere near the top of your Dockerfile, after "apt-get update && apt-get install ...") ──



# ────────────────────────────────────────────────────────────────────────────
# STEP 3: Create non-root user 'appuser' (UID 1000) for security
ARG USER=appuser
ARG UID=1000
RUN useradd --create-home --shell /bin/bash --uid ${UID} ${USER}

USER ${USER}
WORKDIR /home/${USER}

# Install AWS CLI v2
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      unzip \
      curl \
  && rm -rf /var/lib/apt/lists/* \
  && \
  # Download and install AWS CLI v2 silently
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
  unzip /tmp/awscliv2.zip -d /tmp && \
  /tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update && \
  rm -rf /tmp/awscliv2.zip /tmp/aws

# Switch back to non-root user (appuser)
USER appuser

# ────────────────────────────────────────────────────────────────────────────
# STEP 4: Clone your lora-api repo (with all code: lora-scripts + FastAPI)
RUN git clone --recurse-submodules https://github.com/alperpacin/lora-api.git app

WORKDIR /home/${USER}/app

# ────────────────────────────────────────────────────────────────────────────
# STEP 5: Create a Python 3.10 venv under /home/appuser/app/venv
RUN python3 -m venv ./venv

# Upgrade pip in the venv
RUN /home/${USER}/app/venv/bin/pip install --upgrade pip

# ────────────────────────────────────────────────────────────────────────────
# STEP 6: Install FastAPI dependencies (including requests) from requirements.txt
COPY --chown=${USER}:${USER} requirements.txt ./
RUN /home/${USER}/app/venv/bin/pip install -r requirements.txt

# ────────────────────────────────────────────────────────────────────────────
# STEP 7: Run lora-scripts’ install.bash inside the venv to install ML deps
# This will install torch, diffusers, safetensors, xformers, etc.
RUN chmod +x install.bash \
 && /bin/bash -lc "source venv/bin/activate && ./install.bash"

# ────────────────────────────────────────────────────────────────────────────
# STEP 8: Ensure train.sh is executable
RUN chmod +x train.sh

# ────────────────────────────────────────────────────────────────────────────
# STEP 9: Create folders under /mnt/data (will be a Docker volume at runtime)
RUN mkdir -p /mnt/data/train_images /mnt/data/lora_jobs \
 && chmod 777 /mnt/data/train_images /mnt/data/lora_jobs

# ────────────────────────────────────────────────────────────────────────────
# STEP 10: Expose port 8000 for FastAPI
EXPOSE 8000

# ────────────────────────────────────────────────────────────────────────────
# STEP 11: Entry point—start Uvicorn with the app’s FastAPI instance
ENTRYPOINT [ "/home/appuser/app/venv/bin/uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000" ]
