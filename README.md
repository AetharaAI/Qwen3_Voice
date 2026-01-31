# AetherPro Omni Voice Gateway

A production-grade, multi-modal voice gateway providing real-time audio-to-audio interaction, read-aloud TTS, transcription, and model introspection. Built on vLLM-Omni architecture with Qwen3 models.

## Overview

This service replaces legacy audio stacks with a modern architecture featuring:

- **Real-time Voice AI**: Full-duplex WebSocket streaming with the Qwen3-Omni 30B model (AWQ-4bit)
- **Read-Aloud TTS**: HTTP and streaming WebSocket endpoints using Qwen3-TTS 0.6B
- **Audio Transcription**: Speech-to-text via the Omni model
- **Production Monitoring**: Prometheus metrics, health checks, GPU monitoring
- **Fleet-Ready**: Stateless design suitable for orchestration environments

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Gateway (Port 8004)                                         │
│  ├── /health, /models, /config, /languages                   │
│  ├── WebSocket /voice (streaming audio-to-audio)             │
│  ├── POST /read (read-aloud TTS)                             │
│  ├── WebSocket /read/stream (streaming TTS)                  │
│  └── GET /metrics (Prometheus)                               │
└───────────────────┬──────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐     ┌────────▼──────┐
│  Omni Service   │     │  TTS Service  │
│  vLLM-Omni      │     │  vLLM         │
│  30B AWQ-4bit   │     │  0.6B Base    │
│  GPU 1 (65%)    │     │  GPU 1 (10%)  │
└─────────────────┘     └───────────────┘
```

## Quick Start

### Prerequisites

- Ubuntu 22.04 host
- NVIDIA L40S GPU (device=1)
- Docker 24.0+ with NVIDIA runtime
- Docker Compose 2.20+
- CUDA 12.1+ drivers

### Deploy

```bash
# Clone and enter directory
cd /path/to/voice-gateway

# Start services
docker-compose up -d

# Wait for model downloads (~30-60 minutes first run)
docker-compose logs -f omni-service

# Verify health
curl http://localhost:8004/health
```

## API Endpoints

### Core Operational

```bash
# Health check with GPU stats
curl http://localhost:8004/health

# List active models
curl http://localhost:8004/models

# View runtime configuration
curl http://localhost:8004/config

# Get supported languages
curl http://localhost:8004/languages
```

### Voice Conversation

```bash
# WebSocket streaming (use WebSocket client)
ws://localhost:8004/voice

# Batch TTS generation
curl -X POST http://localhost:8004/voice/batch \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}' \
  --output speech.wav

# Audio transcription
curl -X POST http://localhost:8004/voice/transcribe \
  -F "audio=@recording.wav" \
  -F "language=en"
```

### Read-Aloud

```bash
# Single read-aloud request
curl -X POST http://localhost:8004/read \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "voice": "default",
    "language": "en",
    "speed": 1.0
  }' \
  --output read_aloud.wav

# WebSocket streaming with pause/resume
ws://localhost:8004/read/stream
```

### Monitoring

```bash
# Prometheus metrics
curl http://localhost:8004/metrics
```

## WebSocket Protocols

### /voice (Real-time Conversation)

**Send** (Binary): PCM audio chunks (24kHz, 16-bit, mono, little-endian)

**Send** (JSON control):
```json
{"type": "audio.commit"}
{"type": "interrupt"}
{"type": "ping"}
```

**Receive** (Binary): Synthesized PCM audio

### /read/stream (Read-Aloud Streaming)

**Client → Server**:
```json
{"type": "start", "text": "...", "voice": "...", "language": "..."}
{"type": "pause"}
{"type": "resume"}
{"type": "cancel"}
{"type": "ping"}
```

**Server → Client**:
```json
{"type": "started"}
{"type": "paused"}
{"type": "resumed"}
{"type": "cancelled"}
{"type": "complete"}
{"type": "error", "message": "..."}
```

## Configuration

Environment variables (set in `docker-compose.yml` or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNI_SERVICE_URL` | http://omni-service:8000 | Omni HTTP endpoint |
| `OMNI_WS_URL` | ws://omni-service:8000 | Omni WebSocket endpoint |
| `TTS_SERVICE_URL` | http://tts-service:8001 | TTS HTTP endpoint |
| `GPU_DEVICE` | 1 | GPU device index |
| `AUDIO_SAMPLE_RATE` | 24000 | Audio sample rate (Hz) |
| `HF_HOME` | /mnt/aetherpro-extra1/hf | Model cache directory |

## Models

| Model | Role | Size | VRAM | Quantization |
|-------|------|------|------|--------------|
| `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` | Omni | 30B (3B active) | 65% | AWQ-4bit |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | TTS | 0.6B | 10% | float16 |

## Monitoring

Prometheus metrics exposed at `/metrics`:

- `voice_gateway_requests_total` - Request count by method/endpoint/status
- `voice_gateway_request_latency_seconds` - Request latency histogram
- `voice_gateway_audio_generation_latency_seconds` - TTS generation latency
- `voice_gateway_active_streams` - Active WebSocket streams
- `voice_gateway_gpu_utilization_percent` - GPU utilization
- `voice_gateway_gpu_memory_used_mb` - GPU memory usage
- `voice_gateway_tokens_per_second` - Generation throughput

## File Structure

```
.
├── docker-compose.yml          # Service orchestration
├── README.md                   # This file
├── DEPLOYMENT.md              # Deployment guide
├── LICENSE                    # AetherPro Technologies License
├── vllm-omni/
│   └── Dockerfile             # vLLM-Omni build
└── gateway/
    ├── Dockerfile             # FastAPI gateway
    ├── requirements.txt       # Python deps
    ├── main.py               # Application entry
    ├── config.py             # Configuration
    ├── routers/              # API endpoints
    ├── services/             # Client libraries
    └── utils/                # Utilities
```

## Documentation

- [Implementation Plan](Implementation_Plan_Omni_Voice_Gateway.md) - Technical architecture details
- [Deployment Guide](DEPLOYMENT.md) - Production deployment procedures

## Support

For issues, questions, or contributions, contact AetherPro Technologies engineering.

---

**License**: AetherPro Technologies Proprietary License 2026

**Copyright © 2026 AetherPro Technologies. All rights reserved.**