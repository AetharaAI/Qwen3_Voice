# TTS Service Dockerfile
# Uses vLLM for Qwen3-TTS model with proper GPU support

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG PYTHON_VERSION

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VLLM_LOGGING_LEVEL=INFO
ENV VLLM_NO_USAGE_STATS=1

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       software-properties-common \
       git \
       curl \
       libsndfile1 \
       ffmpeg \
       libgomp1 \
       libnccl2 \
       python3-pip \
       python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (CPU-only during build to avoid pynvml issues)
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM without importing it (avoid pynvml at build time)
RUN pip install --no-cache-dir vllm==0.6.3 --no-deps || true
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    transformers \
    accelerate \
    soundfile \
    librosa \
    pydantic \
    fastapi \
    uvicorn \
    httpx \
    orjson \
    einops \
    sentencepiece \
    protobuf

# Install remaining vLLM dependencies
RUN pip install --no-cache-dir vllm==0.6.3

# Expose API port
EXPOSE 8001

# Default command
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "--port", "8001", "--host", "0.0.0.0", "--dtype", "float16", "--gpu-memory-utilization", "0.10", "--max-model-len", "4096"]