# Deployment Guide - AetherPro Omni Voice Gateway

Complete deployment procedures for production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Preparation](#system-preparation)
3. [Deployment](#deployment)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Production Checklist](#production-checklist)
7. [Monitoring Setup](#monitoring-setup)
8. [Backup & Recovery](#backup--recovery)

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA L40S with 48GB VRAM (or equivalent)
- **GPU Device Index**: 1 (configurable)
- **System RAM**: 64GB minimum
- **Storage**: 200GB free space for models
- **Network**: 1Gbps recommended

### Software Requirements

- **OS**: Ubuntu 22.04 LTS
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **NVIDIA Driver**: 535.54.03+
- **CUDA**: 12.1+
- **NVIDIA Container Toolkit**: Latest

---

## System Preparation

### 1. Install NVIDIA Drivers

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (example for 535)
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot
```

### 2. Install Docker & NVIDIA Runtime

```bash
# Install Docker
sudo apt install -y docker.io
sudo systemctl enable --now docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker for NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verify GPU Access

```bash
# Test nvidia-smi
nvidia-smi

# Expected output should show L40S on device 1
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.1     |
# |-----------------------------------------+----------------------+----------------------+
# |   1  NVIDIA L40S                    Off| 00000000:00:00.0 Off |                    0 |
# | 46%   55C    P8              35W / 350W|      0MiB / 49152MiB |      0%      Default |
# +-----------------------------------------+----------------------+----------------------+
```

### 4. Create Model Cache Directory

```bash
# Create cache directory
sudo mkdir -p /mnt/aetherpro-extra1/hf

# Set ownership (use docker user or current user)
sudo chown -R $USER:$USER /mnt/aetherpro-extra1/hf

# Verify permissions
ls -la /mnt/aetherpro-extra1/
```

---

## Deployment

### 1. Clone Repository

```bash
cd /opt
sudo mkdir -p aetherpro
cd aetherpro

# Clone or copy deployment files
sudo cp -r /path/to/voice-gateway ./
cd voice-gateway
```

### 2. Configure Environment

Edit `docker-compose.yml` to match your environment:

```yaml
# Update these values as needed:
# - GPU device index (default: 1)
# - Model cache path (default: /mnt/aetherpro-extra1/hf)
# - Port mappings (default: 8004, 8005)
```

Optional: Create `.env` file:

```bash
cat > .env << 'EOF'
GPU_DEVICE=1
HF_HOME=/mnt/aetherpro-extra1/hf
LOG_LEVEL=INFO
EOF
```

### 3. Build Services

```bash
# Build all services (this will take 30-60 minutes for vLLM-Omni)
docker-compose build

# Monitor build progress
docker-compose build --no-cache --progress=plain 2>&1 | tee build.log
```

### 4. Start Services

```bash
# Start in detached mode
docker-compose up -d

# View startup logs
docker-compose logs -f
```

### 5. Model Download

First startup will download models (~40GB):

```bash
# Monitor Omni model download
docker-compose logs -f omni-service | grep -E "(Downloading|Loading|Ready)"

# Monitor TTS model download
docker-compose logs -f tts-service | grep -E "(Downloading|Loading|Ready)"

# Check download progress
watch -n 5 'du -sh /mnt/aetherpro-extra1/hf/*'
```

---

## Verification

### 1. Service Health

```bash
# Check all services are running
docker-compose ps

# Expected output:
# NAME                COMMAND                  SERVICE             STATUS              PORTS
# omni_voice_service  "python -m vllm.entr…"   omni-service        running (healthy)   0.0.0.0:8004->8000/tcp
# tts_read_service    "python -m vllm.entr…"   tts-service         running (healthy)   0.0.0.0:8005->8001/tcp
# voice_gateway       "uvicorn main:app --…"   gateway             running (healthy)   0.0.0.0:8000->8000/tcp
```

### 2. API Health Check

```bash
# Gateway health
curl -s http://localhost:8004/health | jq

# Expected response:
# {
#   "status": "ok",
#   "model_loaded": true,
#   "active_models": [...],
#   "gpu_id": 1,
#   "vram_used_mb": 24576,
#   "vram_total_mb": 49152,
#   "uptime_seconds": 120
# }
```

### 3. Model Verification

```bash
# List models
curl -s http://localhost:8004/models | jq

# Check configuration
curl -s http://localhost:8004/config | jq
```

### 4. End-to-End Test

```bash
# Test read-aloud
curl -X POST http://localhost:8004/read \
  -H "Content-Type: application/json" \
  -d '{"text": "Deployment successful"}' \
  --output test.wav

# Verify audio file
file test.wav
# Expected: test.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 24000 Hz
```

### 5. GPU Verification

```bash
# Check GPU utilization
nvidia-smi

# Check Prometheus GPU metrics
curl -s http://localhost:8004/metrics | grep gpu
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Check resource limits
docker stats

# Restart service
docker-compose restart <service-name>
```

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check device permissions
ls -la /dev/nvidia*

# Restart Docker
sudo systemctl restart docker
```

### Model Download Issues

```bash
# Check disk space
df -h /mnt/aetherpro-extra1

# Check network connectivity
curl -I https://huggingface.co

# Clear cache and retry
rm -rf /mnt/aetherpro-extra1/hf/*
docker-compose restart
```

### Out of Memory

```bash
# Check VRAM usage
nvidia-smi

# Reduce batch size (edit docker-compose.yml)
# --max-num-seqs 256

# Adjust GPU memory utilization
# --gpu-memory-utilization 0.60
```

### Port Conflicts

```bash
# Check port usage
sudo lsof -i :8004
sudo lsof -i :8005

# Change ports in docker-compose.yml
```

---

## Production Checklist

- [ ] GPU driver installed and verified
- [ ] NVIDIA Container Toolkit configured
- [ ] Model cache directory created with correct permissions
- [ ] Docker Compose configured with correct GPU device
- [ ] Services built successfully
- [ ] Models downloaded (check disk space)
- [ ] All health checks passing
- [ ] End-to-end test successful
- [ ] Prometheus metrics accessible
- [ ] Log rotation configured
- [ ] Firewall rules configured
- [ ] SSL/TLS certificates (if exposing externally)
- [ ] Backup strategy implemented

---

## Monitoring Setup

### Prometheus

Add to your Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'voice-gateway'
    static_configs:
      - targets: ['localhost:8004']
    scrape_interval: 15s
    metrics_path: /metrics
```

### Grafana Dashboard

Key metrics to visualize:

- Request rate and latency
- Active WebSocket streams
- GPU utilization and memory
- Audio generation duration
- Error rates by endpoint

### Alerting Rules

```yaml
groups:
  - name: voice-gateway
    rules:
      - alert: VoiceGatewayDown
        expr: up{job="voice-gateway"} == 0
        for: 1m
        annotations:
          summary: "Voice Gateway is down"
      
      - alert: HighGPUMemory
        expr: voice_gateway_gpu_memory_used_mb / voice_gateway_gpu_memory_total_mb > 0.9
        for: 5m
        annotations:
          summary: "GPU memory usage above 90%"
      
      - alert: HighErrorRate
        expr: rate(voice_gateway_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
```

---

## Backup & Recovery

### Backup Strategy

```bash
# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz docker-compose.yml .env gateway/

# Backup model cache (optional - can re-download)
rsync -av /mnt/aetherpro-extra1/hf/ backup/hf/
```

### Recovery Procedure

```bash
# Stop services
docker-compose down

# Restore configuration
tar -xzf config-backup-YYYYMMDD.tar.gz

# Start services
docker-compose up -d

# Verify health
curl http://localhost:8004/health
```

---

## Updates

### Updating Services

```bash
# Pull latest images/code
git pull origin main

# Rebuild if needed
docker-compose build --no-cache

# Restart with zero downtime (if using multiple instances)
docker-compose up -d

# Verify
curl http://localhost:8004/health
```

### Model Updates

```bash
# Stop services
docker-compose down

# Clear model cache
rm -rf /mnt/aetherpro-extra1/hf/models--*

# Update model name in docker-compose.yml if needed

# Start services (will re-download)
docker-compose up -d
```

---

## Security Considerations

- Never expose ports 8004/8005 directly to the internet without authentication
- Use reverse proxy (nginx/traefik) with SSL/TLS
- Implement API key authentication for production
- Regular security updates for host system
- Network isolation with Docker networks
- Log sanitization to prevent data leakage

---

## Support

For deployment issues:

1. Check logs: `docker-compose logs -f`
2. Verify GPU: `nvidia-smi`
3. Test connectivity: `curl http://localhost:8004/health`
4. Review this guide's Troubleshooting section

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-01-31  
**Maintainer**: AetherPro Technologies Engineering