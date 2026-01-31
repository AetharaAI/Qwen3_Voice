# vLLM-Omni Dockerfile - Build from vllm-project/vllm-omni
# Supports Qwen3-Omni models with Thinker-Talker architecture
# Single-stage build for simplicity

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG PYTHON_VERSION
ARG CUDA_VERSION

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VLLM_LOGGING_LEVEL=INFO
ENV VLLM_NO_USAGE_STATS=1

# Set CUDA environment variables for flash-attn build
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       software-properties-common \
       git \
       curl \
       wget \
       vim \
       gcc \
       g++ \
       make \
       libsndfile1 \
       ffmpeg \
       libgomp1 \
       libnccl2 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
       python${PYTHON_VERSION} \
       python${PYTHON_VERSION}-dev \
       python${PYTHON_VERSION}-venv \
       python3-pip \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel packaging

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install psutil (required by flash-attn before its own setup)
RUN pip install --no-cache-dir psutil

# Install flash-attention (required for vLLM-Omni)
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# Clone and install vLLM-Omni from source
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm-omni.git vllm-omni \
    && cd vllm-omni \
    && git submodule update --init --recursive

# Install vLLM-Omni with all audio dependencies
WORKDIR /workspace/vllm-omni
RUN pip install --no-cache-dir -e .[audio] \
    && pip install --no-cache-dir transformers accelerate

# Install additional audio processing dependencies
RUN pip install --no-cache-dir \
    librosa \
    soundfile \
    webrtcvad \
    numpy \
    scipy

# Set working directory
WORKDIR /workspace/vllm-omni

# Expose API port
EXPOSE 8000

# Default entrypoint
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]