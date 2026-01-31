# TTS Service Dockerfile
# Uses qwen-tts package to serve Qwen3-TTS model
# Qwen3-TTS requires the qwen-tts Python package, NOT standard vLLM

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

# Install qwen-tts package (required for Qwen3-TTS model)
# This provides Qwen3TTSModel and Qwen3TTSTokenizer classes
RUN pip install --no-cache-dir -U qwen-tts

# Install FastAPI and server dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic httpx

# Install audio dependencies
RUN pip install --no-cache-dir soundfile librosa

# Set working directory
WORKDIR /app

# Copy the TTS server script
COPY tts_server.py /app/tts_server.py

# Expose API port (matches the port in tts_server.py)
EXPOSE 8001

# Run the custom TTS server (NOT vLLM - Qwen3-TTS is not vLLM-compatible)
CMD ["python", "tts_server.py"]
