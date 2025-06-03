# ─────────────────────────────────────────────────────────────────────────────
# 1. Base image: NVIDIA PyTorch (CUDA 11.7, Python 3.8, PyTorch preinstalled)
FROM nvcr.io/nvidia/pytorch:23.03-py3 AS base

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prevent tzdata prompts and set default timezone to UTC
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
 && echo "Etc/UTC" > /etc/timezone

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install system packages (git, python3-pip, tzdata, etc.)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      tzdata \
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
      python3-pip \
      python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-create /mnt/data folders so the non-root user can write there later
RUN mkdir -p /mnt/data/train_images \
 && mkdir -p /mnt/data/lora_jobs \
 && chmod 777 /mnt/data/train_images \
 && chmod 777 /mnt/data/lora_jobs

# ─────────────────────────────────────────────────────────────────────────────
# 5. Create a non-root user "appuser"
ARG USER=appuser
ARG UID=1000
RUN useradd --create-home --shell /bin/bash --uid ${UID} ${USER}
USER ${USER}
WORKDIR /home/${USER}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Copy all local files from lora-api/ into /home/appuser/app
#    This includes: install.bash, train.sh, scripts/, api_server.py, requirements.txt, etc.
COPY --chown=${USER}:${USER} . app
WORKDIR /home/${USER}/app

# ─────────────────────────────────────────────────────────────────────────────
# 7. Install Python dependencies using pip3 (system-wide)
RUN pip3 install --upgrade pip \
 && pip3 install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# 8. Run install.bash to install LoRA-scripts dependencies (diffusers, xformers, etc.)
RUN chmod +x install.bash \
 && /bin/bash -lc "bash install.bash"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Ensure train.sh is executable
RUN chmod +x train.sh

# ─────────────────────────────────────────────────────────────────────────────
# 10. Expose FastAPI port and set entrypoint to start Uvicorn
EXPOSE 8000
ENTRYPOINT ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
