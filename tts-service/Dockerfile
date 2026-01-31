# TTS Service Dockerfile
# Uses standard vLLM to serve Qwen3-TTS model
# Qwen3-TTS is a standard causal LM model that works with vLLM

ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/mnt/aetherpro-extra1/hf
ENV TRANSFORMERS_CACHE=/mnt/aetherpro-extra1/hf
ENV VLLM_LOGGING_LEVEL=INFO

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       git \
       curl \
       wget \
       libsndfile1 \
       ffmpeg \
       libgomp1 \
       python3-pip \
       python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (standard version - TTS model can use regular vLLM)
RUN pip install --no-cache-dir vllm==0.6.3 transformers accelerate

# Install audio dependencies
RUN pip install --no-cache-dir soundfile librosa

WORKDIR /app

# Expose API port
EXPOSE 8001