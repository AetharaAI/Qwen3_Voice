"""
AetherPro Omni Voice Gateway - Main FastAPI Application

Provides a production-grade, multi-modal voice gateway with:
- Real-time audio-to-audio conversation via WebSocket
- Read-aloud TTS endpoints (HTTP and WebSocket streaming)
- Audio transcription
- Health monitoring and Prometheus metrics
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from routers import health, voice, tts, metrics
from routers.metrics import MetricsMiddleware, init_metrics
from utils.gpu_monitor import gpu_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Voice Gateway Service")
    logger.info("=" * 60)
    
    settings = get_settings()
    
    # Log configuration
    logger.info(f"Gateway Port: {settings.gateway_port}")
    logger.info(f"GPU Device: {settings.gpu_device}")
    logger.info(f"Omni Service: {settings.omni_service_url}")
    logger.info(f"TTS Service: {settings.tts_service_url}")
    logger.info(f"Audio Sample Rate: {settings.audio_sample_rate} Hz")
    logger.info(f"Omni Model: {settings.omni_model_name}")
    logger.info(f"TTS Model: {settings.tts_model_name}")
    
    # Initialize GPU monitor
    if gpu_monitor.available:
        logger.info(f"GPU Monitor: Available ({gpu_monitor.get_device_count()} device(s))")
        stats = gpu_monitor.get_gpu_stats(settings.gpu_device)
        if stats:
            logger.info(f"GPU {settings.gpu_device}: {stats.name}")
            logger.info(f"VRAM: {stats.used_memory_mb}/{stats.total_memory_mb} MB")
    else:
        logger.warning("GPU Monitor: Not available")
    
    # Initialize Prometheus metrics
    init_metrics()
    logger.info("Prometheus metrics initialized")
    
    logger.info("-" * 60)
    logger.info("Voice Gateway ready")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Voice Gateway...")
    gpu_monitor.shutdown()
    logger.info("Voice Gateway stopped")


# Create FastAPI application
app = FastAPI(
    title="AetherPro Omni Voice Gateway",
    description="""
    Multimodal voice gateway providing real-time audio-to-audio interaction,
    read-aloud TTS, transcription, and model introspection.
    
    ## Endpoints
    
    ### Core Operational
    - `GET /health` - Service health and readiness
    - `GET /models` - Active model inventory
    - `GET /config` - Runtime configuration
    - `GET /languages` - Supported languages
    
    ### Voice & Audio
    - `WS /voice` - Real-time voice conversation
    - `WS /voice/stream` - Alias for /voice
    - `POST /voice/batch` - Batch TTS generation
    - `POST /voice/transcribe` - Audio transcription
    
    ### Read-Aloud
    - `POST /read` - Read-aloud TTS
    - `WS /read/stream` - Streamed read-aloud
    
    ### Observability
    - `GET /metrics` - Prometheus metrics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(health.router, tags=["Health & Info"])
app.include_router(voice.router, prefix="/voice", tags=["Voice"])
app.include_router(tts.router, tags=["Read-Aloud"])
app.include_router(metrics.router, tags=["Metrics"])


@app.get("/")
async def root():
    """Root endpoint - redirects to docs."""
    return {
        "service": "AetherPro Omni Voice Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if get_settings().log_level == "DEBUG" else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        log_level=settings.log_level.lower(),
        access_log=True
    )