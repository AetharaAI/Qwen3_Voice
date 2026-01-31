"""
Prometheus metrics endpoint for monitoring and observability.
"""

import time
import logging
from typing import Callable
from functools import wraps

from fastapi import APIRouter, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from config import get_settings
from utils.gpu_monitor import gpu_monitor

logger = logging.getLogger(__name__)
router = APIRouter()

# Prometheus metrics
# Request metrics
REQUEST_COUNT = Counter(
    'voice_gateway_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'voice_gateway_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Audio generation metrics
AUDIO_GENERATION_LATENCY = Histogram(
    'voice_gateway_audio_generation_latency_seconds',
    'Audio generation latency',
    ['type'],  # 'tts' or 'omni'
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

TOKENS_PER_SECOND = Gauge(
    'voice_gateway_tokens_per_second',
    'Tokens per second generation rate',
    ['model']
)

# Active connections
ACTIVE_STREAMS = Gauge(
    'voice_gateway_active_streams',
    'Number of active WebSocket streams',
    ['type']  # 'voice' or 'read'
)

# GPU metrics
GPU_UTILIZATION = Gauge(
    'voice_gateway_gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id']
)

GPU_MEMORY_USED = Gauge(
    'voice_gateway_gpu_memory_used_mb',
    'GPU memory used in MB',
    ['device_id']
)

GPU_MEMORY_TOTAL = Gauge(
    'voice_gateway_gpu_memory_total_mb',
    'GPU memory total in MB',
    ['device_id']
)

GPU_TEMPERATURE = Gauge(
    'voice_gateway_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['device_id']
)

# Service info
SERVICE_INFO = Info(
    'voice_gateway',
    'Voice Gateway service information'
)

# Track startup time
_startup_time = time.time()


def init_metrics():
    """Initialize service info metrics."""
    settings = get_settings()
    SERVICE_INFO.info({
        'version': '1.0.0',
        'omni_model': settings.omni_model_name,
        'tts_model': settings.tts_model_name,
        'gpu_device': str(settings.gpu_device)
    })


def track_request(method: str, endpoint: str, status_code: int, duration: float):
    """Track HTTP request metrics."""
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def track_audio_generation(type_: str, duration: float):
    """Track audio generation latency."""
    AUDIO_GENERATION_LATENCY.labels(type=type_).observe(duration)


def set_active_streams(type_: str, count: int):
    """Set active stream count."""
    ACTIVE_STREAMS.labels(type=type_).set(count)


def update_gpu_metrics():
    """Update GPU metrics from monitor."""
    if not gpu_monitor.available:
        return
    
    settings = get_settings()
    stats = gpu_monitor.get_gpu_stats(settings.gpu_device)
    
    if stats:
        device_id = str(stats.device_id)
        GPU_UTILIZATION.labels(device_id=device_id).set(stats.utilization_percent)
        GPU_MEMORY_USED.labels(device_id=device_id).set(stats.used_memory_mb)
        GPU_MEMORY_TOTAL.labels(device_id=device_id).set(stats.total_memory_mb)
        if stats.temperature_celsius is not None:
            GPU_TEMPERATURE.labels(device_id=device_id).set(stats.temperature_celsius)


class MetricsMiddleware:
    """FastAPI middleware for tracking request metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        # Capture response status
        status_code = 200
        
        async def wrapped_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)
        
        await self.app(scope, receive, wrapped_send)
        
        # Track metrics
        duration = time.time() - start_time
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        
        # Normalize endpoint label
        endpoint = path
        if "/voice" in path:
            endpoint = "/voice"
        elif "/read" in path:
            endpoint = "/read"
        elif path == "/health":
            endpoint = "/health"
        elif path == "/metrics":
            endpoint = "/metrics"
        
        track_request(method, endpoint, status_code, duration)


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes:
    - Request latency and count
    - Audio generation latency
    - Active stream counts
    - GPU utilization and memory
    - Tokens per second
    
    Must not block inference.
    """
    # Update GPU metrics before serving
    update_gpu_metrics()
    
    # Generate Prometheus format output
    output = generate_latest()
    
    return Response(
        content=output,
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/metrics/health")
async def metrics_health():
    """
    Lightweight health check for metrics endpoint itself.
    """
    return {
        "status": "ok",
        "metrics_enabled": True,
        "gpu_monitor_available": gpu_monitor.available
    }