# Implementation Plan: Omni Voice Service Gateway

## Overview

This document describes the production-ready voice service implementation using vLLM-Omni architecture.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HOST (Ubuntu 22.04)                         │
│                           NVIDIA L40S (GPU 1)                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐   │
│  │   Port 8004  │────│      FastAPI Gateway (main.py)           │   │
│  │  (External)  │    │      Port 8000 (Internal)                │   │
│  └──────────────┘    └──────────────────────────────────────────┘   │
│                                │                                    │
│            ┌───────────────────┴───────────────────┐                │
│            │                                       │                │
│    ┌───────▼────────┐                    ┌────────▼──────┐         │
│    │  Omni Service   │                    │  TTS Service  │         │
│    │  vLLM-Omni      │                    │  vLLM         │         │
│    │  Port 8000      │                    │  Port 8001    │         │
│    │  GPU: 65% VRAM  │                    │  GPU: 10% VRAM│         │
│    └─────────────────┘                    └───────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
.
├── docker-compose.yml          # Docker Compose configuration
├── vllm-omni/
│   └── Dockerfile             # vLLM-Omni build from source
└── gateway/
    ├── Dockerfile             # FastAPI gateway image
    ├── requirements.txt       # Python dependencies
    ├── main.py               # FastAPI application entry
    ├── config.py             # Environment configuration
    ├── routers/
    │   ├── __init__.py
    │   ├── health.py         # /health, /models, /config, /languages
    │   ├── voice.py          # WebSocket /voice, /voice/stream, POST /voice/batch, /voice/transcribe
    │   ├── tts.py            # /read and WebSocket /read/stream
    │   └── metrics.py        # Prometheus /metrics endpoint
    ├── services/
    │   ├── __init__.py
    │   ├── omni_client.py    # WebSocket client for vLLM-Omni
    │   └── tts_client.py     # HTTP client for TTS service
    └── utils/
        ├── __init__.py
        └── gpu_monitor.py    # GPU stats monitoring for /health
```

---

## Services

### 1. vLLM-Omni Service (Port 8004 external)

**Image**: Built from `vllm-project/vllm-omni` repository

**Model**: `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`

**Configuration**:
- GPU Memory: 65% utilization
- Quantization: AWQ-4bit
- Max Context: 32768 tokens
- Audio Output: 24kHz PCM
- Port: 8004 (host) → 8000 (container)
- GPU: device=1

**Features**:
- Thinker-Talker architecture
- Real-time audio streaming via WebSocket
- `--omni` flag enabled

### 2. vLLM-TTS Service (Port 8005 external)

**Image**: `vllm/vllm-openai:latest`

**Model**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

**Configuration**:
- GPU Memory: 10% utilization
- Dtype: float16
- Max Context: 4096 tokens
- Port: 8005 (host) → 8001 (container)
- GPU: device=1

**Interface**: OpenAI-compatible `/v1/audio/speech`

### 3. Gateway Service (Port 8000 internal)

**Base**: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`

**Python**: 3.11

**Framework**: FastAPI with WebSocket support

---

## API Endpoints

### Core Operational

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service liveness + readiness with GPU stats |
| `/models` | GET | Active model inventory with capabilities |
| `/config` | GET | Runtime configuration (read-only) |
| `/languages` | GET | Supported languages discovery |

### Voice & Audio

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/voice` | WebSocket | Full-duplex real-time voice interaction |
| `/voice/stream` | WebSocket | Semantic alias for `/voice` |
| `/voice/batch` | POST | Non-streaming TTS generation (WAV output) |
| `/voice/transcribe` | POST | Audio → text transcription |

### Read-Aloud

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/read` | POST | Long-form read-aloud TTS |
| `/read/stream` | WebSocket | Streamed read-aloud with pause/resume/cancel |

### Observability

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Prometheus metrics (latency, GPU, streams) |

---

## Configuration

### Environment Variables

```bash
# Service URLs
OMNI_SERVICE_URL=http://omni-service:8000
OMNI_WS_URL=ws://omni-service:8000
TTS_SERVICE_URL=http://tts-service:8001

# Ports
GATEWAY_PORT=8000

# GPU
GPU_DEVICE=1

# Audio
AUDIO_SAMPLE_RATE=24000
AUDIO_FORMAT=pcm
AUDIO_CHANNELS=1
AUDIO_BIT_DEPTH=16

# Models
OMNI_MODEL_NAME=cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-Base

# Cache
HF_HOME=/mnt/aetherpro-extra1/hf

# Streaming
STREAMING_ENABLED=true
WS_MAX_SIZE=16777216
WS_PING_INTERVAL=20.0
WS_PING_TIMEOUT=20.0
```

---

## Model Cache

Models are cached at: `/mnt/aetherpro-extra1/hf`

Both services share this host-mounted volume for HuggingFace cache.

---

## Audio Format

- **Format**: Raw PCM
- **Sample Rate**: 24kHz
- **Bit Depth**: 16-bit signed
- **Channels**: Mono (1)
- **Endianness**: Little-endian

---

## Docker Network

All services communicate via the `voice-gateway` bridge network:
- `omni-service:8000` - vLLM-Omni internal
- `tts-service:8001` - vLLM-TTS internal
- `gateway:8000` - FastAPI gateway internal

---

## Usage

### Start Services

```bash
docker-compose up -d
```

### Health Check

```bash
curl http://localhost:8004/health
```

### Real-time Voice (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:8004/voice');
ws.binaryType = 'arraybuffer';

// Send audio (PCM 24kHz, 16-bit, mono)
ws.send(audioBuffer);

// Receive audio
ws.onmessage = (event) => {
    const audioData = new Int16Array(event.data);
    // Play audio
};
```

### Read-Aloud

```bash
curl -X POST http://localhost:8004/read \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}' \
  --output speech.wav
```

### Transcription

```bash
curl -X POST http://localhost:8004/voice/transcribe \
  -F "audio=@recording.wav" \
  -F "language=en"
```

---

## Monitoring

Prometheus metrics available at: `http://localhost:8004/metrics`

Metrics include:
- Request latency and count by endpoint
- Audio generation latency
- Active WebSocket streams
- GPU utilization and memory
- Tokens per second

---

## Implementation Details

### WebSocket Protocol (/voice)

**Client → Gateway**:
- Binary: Raw PCM audio chunks
- Text (JSON): Control messages (`ping`, `interrupt`, `audio.commit`)

**Gateway → Client**:
- Binary: Synthesized PCM audio
- Text (JSON): Control responses (`pong`, `error`)

### TTS Streaming Protocol (/read/stream)

**Client Messages**:
```json
{"type": "start", "text": "...", "voice": "...", "language": "..."}
{"type": "pause"}
{"type": "resume"}
{"type": "cancel"}
{"type": "ping"}
{"type": "close"}
```

**Server Messages**:
```json
{"type": "started"}
{"type": "paused"}
{"type": "resumed"}
{"type": "cancelled"}
{"type": "complete"}
{"type": "error", "message": "..."}
{"type": "pong"}
```

---

## Requirements

- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Docker Runtime
- NVIDIA Driver 535+
- CUDA 12.1+
- Ubuntu 22.04 host

---

## Notes

1. **vLLM-Omni Build**: The Omni service builds from the official vllm-omni repository for deterministic audio kernel support.

2. **GPU Pinning**: Both model services are pinned to GPU device=1 as specified.

3. **VRAM Budgets**: Omni uses 65%, TTS uses 10%, leaving 25% headroom on L40S (48GB).

4. **Health Dependencies**: Gateway waits for both model services to be healthy before starting.

5. **OpenAI Compatibility**: TTS service uses standard `/v1/audio/speech` endpoint for compatibility.

6. **Container Safety**: All endpoints are stateless; no persistent user state inside containers.

---

## Validation

Run the following to validate the deployment:

```bash
# Check all services are running
docker-compose ps

# Health check
curl http://localhost:8004/health

# List models
curl http://localhost:8004/models

# View config
curl http://localhost:8004/config

# Prometheus metrics
curl http://localhost:8004/metrics
