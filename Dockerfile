# ─────────────────────────────────────────────────────────────────────────────
# 1. Base image: NVIDIA PyTorch (CUDA 11.7, Python 3.10, PyTorch preinstalled)
FROM nvcr.io/nvidia/pytorch:23.03-py3 AS base

# ─────────────────────────────────────────────────────────────────────────────
# 2. Install system packages (git, unzip, curl, ffmpeg, libs)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      unzip \
      ffmpeg \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      build-essential \
      cmake \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# 3. Create non-root user "appuser" (UID=1000)
ARG USER=appuser
ARG UID=1000
RUN useradd --create-home --shell /bin/bash --uid ${UID} ${USER}
USER ${USER}
WORKDIR /home/${USER}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Clone (or COPY) the lora-api project into /home/appuser/app
#    If you prefer to COPY from local build context instead of git clone,
#    replace the RUN git clone line with:
#      COPY --chown=appuser:appuser . .
RUN git clone --recurse-submodules https://github.com/alperpacin/lora-scripts.git app
WORKDIR /home/${USER}/app

# ─────────────────────────────────────────────────────────────────────────────
# 5. Create Python venv & install Python dependencies
RUN python3 -m venv venv \
 && ./venv/bin/pip install --upgrade pip

# Copy requirements.txt (should be at repo root)
COPY --chown=${USER}:${USER} requirements.txt ./
RUN /home/${USER}/app/venv/bin/pip install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# 6. Run install.bash to pull in LoRA-scripts dependencies (diffusers, xformers, etc.)
RUN chmod +x install.bash \
 && /bin/bash -lc "source venv/bin/activate && ./install.bash"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Ensure train.sh is executable
RUN chmod +x train.sh

# ─────────────────────────────────────────────────────────────────────────────
# 8. Pre-create folders under /mnt/data (will be volume-mounted at runtime)
RUN mkdir -p /mnt/data/train_images \
 && mkdir -p /mnt/data/lora_jobs \
 && chmod 777 /mnt/data/train_images \
 && chmod 777 /mnt/data/lora_jobs

# ─────────────────────────────────────────────────────────────────────────────
# 9. Expose FastAPI port
EXPOSE 8000

# ─────────────────────────────────────────────────────────────────────────────
# 10. Entrypoint: run Uvicorn via the venv
ENTRYPOINT ["/home/appuser/app/venv/bin/uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
