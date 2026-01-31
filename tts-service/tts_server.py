#!/usr/bin/env python3
"""
TTS Service using Qwen3-TTS model via qwen-tts package.
Provides OpenAI-compatible /v1/audio/speech endpoint.
"""

import os
import io
import base64
import logging
from typing import Optional
from contextlib import asynccontextmanager

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None

class TTSRequest(BaseModel):
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    input: str
    voice: Optional[str] = "default"
    response_format: str = "wav"
    speed: float = 1.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    
    logger.info("Loading Qwen3-TTS model...")
    
    try:
        from qwen_tts import Qwen3TTSModel
        
        # Load model
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
        )
        
        logger.info("Qwen3-TTS model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down TTS service...")

app = FastAPI(
    title="Qwen3-TTS Service",
    description="Text-to-Speech service using Qwen3-TTS model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible speech endpoint.
    
    Note: Qwen3-TTS requires reference audio for voice cloning.
    For now, we use a default reference or the user can provide voice ID.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating speech for text: {request.input[:50]}...")
        
        # Use default reference audio for voice cloning
        # In production, you'd map voice IDs to specific reference audios
        ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
        ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
        
        # Generate speech
        wavs, sr = model.generate_voice_clone(
            text=request.input,
            language="English",  # Could detect from text
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        
        # Convert to bytes
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        
        logger.info(f"Generated audio: {len(audio_data)} bytes")
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/audio/voices")
async def list_voices():
    """List available voices."""
    return {
        "voices": [
            {"voice_id": "default", "name": "Default Voice"},
            {"voice_id": "clone", "name": "Voice Clone"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)