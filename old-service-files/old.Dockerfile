# BlackBoxAudio Voice API - GPU-enabled build aligned with AetherVoice

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# 1) System dependencies (mirror AetherVoice)
# ADDED: espeak-ng (Required for Kokoro, StyleTTS2, Phonemizer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*


# 2) Use python3.10 as default "python"
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# 3) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy application code
COPY . .

# 5) Runtime env
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/mnt/aetherpro/hf \
    TRANSFORMERS_CACHE=/mnt/aetherpro/hf
# ADDED: NLTK data for Phonemizer
RUN python -m nltk.downloader -d /app/nltk_data punkt
# 6) Expose API port
EXPOSE 8000

# 7) Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/v1/health').raise_for_status()"

# 8) Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
