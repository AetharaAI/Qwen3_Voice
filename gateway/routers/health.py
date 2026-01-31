"""
Health, models, config, and languages endpoints.
"""

import time
import logging
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from config import get_settings
from utils.gpu_monitor import gpu_monitor

logger = logging.getLogger(__name__)
router = APIRouter()

# Track service start time
_start_time = time.time()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    active_models: List[str]
    gpu_id: int
    vram_used_mb: int
    vram_total_mb: int
    uptime_seconds: int


class ModelInfo(BaseModel):
    """Model information model."""
    name: str
    role: str
    modalities: List[str]
    quantization: str
    max_context: int
    audio_capabilities: List[str]


class ConfigResponse(BaseModel):
    """Configuration response model."""
    audio_sample_rate: int
    streaming_mode_enabled: bool
    default_language: str
    container_port_mappings: Dict[str, Any]
    environment_flags: Dict[str, Any]


class LanguagesResponse(BaseModel):
    """Languages response model."""
    input_languages: List[str]
    output_languages: List[str]
    default_language: str
    auto_detect: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """
    Service liveness and readiness check.
    Returns 200 OK only when service is fully ready.
    """
    settings = get_settings()
    
    # Get GPU stats
    gpu_stats = gpu_monitor.to_dict(settings.gpu_device)
    
    # Calculate uptime
    uptime_seconds = int(time.time() - _start_time)
    
    # Check if we have GPU access
    model_loaded = gpu_stats.get("available", False)
    
    # Determine status
    if model_loaded:
        status = "ok"
    elif uptime_seconds < 60:
        status = "loading"
    else:
        status = "error"
    
    response = {
        "status": status,
        "model_loaded": model_loaded,
        "active_models": [settings.omni_model_name, settings.tts_model_name],
        "gpu_id": settings.gpu_device,
        "vram_used_mb": gpu_stats.get("vram_used_mb", 0),
        "vram_total_mb": gpu_stats.get("vram_total_mb", 49152),  # L40S default
        "uptime_seconds": uptime_seconds
    }
    
    if status == "error":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response
        )
    
    return response


@router.get("/models", response_model=List[ModelInfo])
async def list_models() -> List[Dict[str, Any]]:
    """
    Runtime introspection of loaded models.
    Returns information about all active models and their capabilities.
    """
    settings = get_settings()
    
    models = [
        {
            "name": settings.omni_model_name,
            "role": "omni",
            "modalities": settings.omni_modalities,
            "quantization": settings.omni_quantization,
            "max_context": settings.omni_max_context,
            "audio_capabilities": ["tts", "stt", "full_duplex"]
        },
        {
            "name": settings.tts_model_name,
            "role": "tts",
            "modalities": settings.tts_modalities,
            "quantization": settings.tts_quantization,
            "max_context": settings.tts_max_context,
            "audio_capabilities": ["tts"]
        }
    ]
    
    return models


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> Dict[str, Any]:
    """
    Runtime configuration for debug and audit visibility.
    Read-only endpoint showing current service configuration.
    """
    settings = get_settings()
    
    return {
        "audio_sample_rate": settings.audio_sample_rate,
        "streaming_mode_enabled": settings.streaming_enabled,
        "default_language": settings.default_language,
        "container_port_mappings": {
            "gateway": {
                "internal": 8000,
                "external": 8004
            },
            "omni_service": {
                "internal": 8000,
                "external": 8004
            },
            "tts_service": {
                "internal": 8001,
                "external": 8005
            }
        },
        "environment_flags": {
            "gpu_device": settings.gpu_device,
            "audio_format": settings.audio_format,
            "audio_channels": settings.audio_channels,
            "audio_bit_depth": settings.audio_bit_depth,
            "omni_model": settings.omni_model_name,
            "tts_model": settings.tts_model_name,
            "hf_home": settings.hf_home,
            "auto_detect_language": settings.auto_detect_language,
            "max_audio_duration_seconds": settings.max_audio_duration_seconds,
            "max_text_length": settings.max_text_length
        }
    }


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages() -> Dict[str, Any]:
    """
    Capability discovery for supported languages.
    Returns supported input/output languages and auto-detection status.
    """
    settings = get_settings()
    
    return {
        "input_languages": settings.supported_input_languages,
        "output_languages": settings.supported_output_languages,
        "default_language": settings.default_language,
        "auto_detect": settings.auto_detect_language
    }