# TTS Service Dockerfile
# Uses Qwen3-TTS model via the qwen-tts package (NOT standard vLLM)

ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/mnt/aetherpro-extra1/hf
ENV TRANSFORMERS_CACHE=/mnt/aetherpro-extra1/hf

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       git \
       curl \
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

# Install qwen-tts package and dependencies
RUN pip install --no-cache-dir qwen-tts transformers accelerate soundfile

# Install FastAPI for serving
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy the TTS server script
COPY tts_server.py /app/tts_server.py

WORKDIR /app

# Expose API port
EXPOSE 8001

# Default command
CMD ["python3", "-u", "tts_server.py"]