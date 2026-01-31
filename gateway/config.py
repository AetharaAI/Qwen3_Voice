"""
Configuration module for Voice Gateway.
Loads settings from environment variables.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Gateway settings
    gateway_port: int = Field(default=8000, description="Gateway service port")
    gateway_host: str = Field(default="0.0.0.0", description="Gateway host")
    
    # Service URLs
    omni_service_url: str = Field(
        default="http://omni-service:8000",
        description="Omni service HTTP URL"
    )
    omni_ws_url: str = Field(
        default="ws://omni-service:8000",
        description="Omni service WebSocket URL"
    )
    tts_service_url: str = Field(
        default="http://tts-service:8001",
        description="TTS service URL"
    )
    
    # GPU settings
    gpu_device: int = Field(default=1, description="GPU device index")
    
    # Audio settings
    audio_sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    audio_format: str = Field(default="pcm", description="Audio format (pcm/wav)")
    audio_channels: int = Field(default=1, description="Number of audio channels")
    audio_bit_depth: int = Field(default=16, description="Audio bit depth")
    
    # Model settings
    omni_model_name: str = Field(
        default="cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
        description="Omni model identifier"
    )
    tts_model_name: str = Field(
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        description="TTS model identifier"
    )
    
    # Model capabilities
    omni_modalities: List[str] = Field(
        default=["text", "audio", "vision"],
        description="Omni model supported modalities"
    )
    tts_modalities: List[str] = Field(
        default=["text", "audio"],
        description="TTS model supported modalities"
    )
    omni_quantization: str = Field(default="awq-4bit", description="Omni model quantization")
    tts_quantization: str = Field(default="float16", description="TTS model quantization")
    omni_max_context: int = Field(default=32768, description="Omni model max context length")
    tts_max_context: int = Field(default=4096, description="TTS model max context length")
    
    # Language settings
    default_language: str = Field(default="en", description="Default language code")
    supported_input_languages: List[str] = Field(
        default=["en", "zh", "es", "fr", "de", "ja", "ko", "auto"],
        description="Supported input languages"
    )
    supported_output_languages: List[str] = Field(
        default=["en", "zh", "es", "fr", "de", "ja", "ko"],
        description="Supported output languages"
    )
    auto_detect_language: bool = Field(default=True, description="Enable language auto-detection")
    
    # Cache settings
    hf_home: str = Field(default="/mnt/aetherpro-extra1/hf", description="HuggingFace cache directory")
    
    # Streaming settings
    streaming_enabled: bool = Field(default=True, description="Enable streaming mode")
    ws_max_size: int = Field(default=16 * 1024 * 1024, description="WebSocket max message size")
    ws_ping_interval: float = Field(default=20.0, description="WebSocket ping interval")
    ws_ping_timeout: float = Field(default=20.0, description="WebSocket ping timeout")
    
    # Request settings
    max_audio_duration_seconds: int = Field(default=300, description="Max audio duration")
    max_text_length: int = Field(default=10000, description="Max text length for TTS")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings