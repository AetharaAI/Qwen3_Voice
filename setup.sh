#!/bin/bash
set -e

echo "========================================="
echo "  Omni Voice Service - Setup Script"
echo "========================================="

# Create volume directories
echo ""
echo "Creating volume directories..."
mkdir -p ${MODEL_CACHE:-/mnt/models}
mkdir -p ${AUDIO_CACHE:-/mnt/audio}
mkdir -p ${HUGGINGFACE_HOME:-/mnt/huggingface}

# Set permissions
echo "Setting permissions..."
chmod 755 ${MODEL_CACHE:-/mnt/models}
chmod 755 ${AUDIO_CACHE:-/mnt/audio}
chmod 755 ${HUGGINGFACE_HOME:-/mnt/huggingface}

# Check NVIDIA driver
echo ""
echo "Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "✓ NVIDIA driver detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi
echo "✓ Docker detected: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    exit 1
fi
echo "✓ Docker Compose detected: $(docker-compose --version)"

# Check .env file
if [ ! -f .env ]; then
    echo "WARNING: .env file not found. Creating from template..."
    cat > .env << 'EOF'
# HuggingFace Token (required for model downloads)
HF_TOKEN=your_huggingface_token_here

# NVIDIA API Key (optional)
NVIDIA_API_KEY=

# Gateway Configuration
GATEWAY_LOG_LEVEL=info

# TTS Service Configuration
TTS_MAX_AUDIO_LENGTH_SECS=30
TTS_DEFAULT_SAMPLE_RATE=24000

# Volume Mounts (adjust to your system)
MODEL_CACHE=/mnt/models
AUDIO_CACHE=/mnt/audio
HUGGINGFACE_HOME=/mnt/huggingface
EOF
    echo "✓ Created .env file. Please edit it to add your HF_TOKEN."
else
    echo "✓ .env file found"
    if grep -q "your_huggingface_token_here" .env 2>/dev/null || ! grep -q "HF_TOKEN=" .env 2>/dev/null; then
        echo "WARNING: HF_TOKEN not set in .env. Models may fail to download."
    fi
fi

# Create network if it doesn't exist
if ! docker network ls | grep -q omni-voice-network; then
    echo ""
    echo "Creating Docker network: omni-voice-network"
    docker network create omni-voice-network
fi

# Build images
echo ""
echo "========================================="
echo "Building Docker images..."
echo "========================================="

export COMPOSE_BAKE=true
docker-compose build --parallel

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To start the services:"
echo "  docker-compose up -d"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop:"
echo "  docker-compose down"
