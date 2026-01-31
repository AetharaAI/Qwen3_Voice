"""
Read-Aloud TTS endpoints: /read and /read/stream.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel

from config import get_settings
from services.tts_client import TTSClient, TTSRequest, TTSStreamingSession

logger = logging.getLogger(__name__)
router = APIRouter()

# Track active TTS sessions
_active_tts_sessions: Dict[str, TTSStreamingSession] = {}


class ReadAloudRequest(BaseModel):
    """Read-aloud request model."""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    speed: float = 1.0


class ReadAloudResponse(BaseModel):
    """Read-aloud response model."""
    status: str
    audio_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    message: Optional[str] = None


@router.post("/read")
async def read_aloud(request: ReadAloudRequest):
    """
    Convert long text into spoken audio (single response).
    
    Optimized for long-form reading with:
    - Stable pacing
    - Minimal latency spikes
    - Complete audio file response
    """
    settings = get_settings()
    
    # Validate input
    if not request.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is required"
        )
    
    if len(request.text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text exceeds maximum length of {settings.max_text_length} characters"
        )
    
    try:
        # Initialize TTS client
        tts_client = TTSClient(
            base_url=settings.tts_service_url,
            model=settings.tts_model_name
        )
        
        # Build TTS request
        tts_request = TTSRequest(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
            response_format="wav",
            sample_rate=settings.audio_sample_rate
        )
        
        logger.info(f"Read-aloud request: {len(request.text)} chars")
        
        # Synthesize audio
        tts_response = await tts_client.synthesize(tts_request)
        await tts_client.close()
        
        logger.info(f"Read-aloud complete: {len(tts_response.audio_data)} bytes, "
                   f"{tts_response.duration_seconds:.2f}s")
        
        # Return audio file
        return Response(
            content=tts_response.audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=read_aloud.wav",
                "X-Audio-Duration": str(tts_response.duration_seconds or 0),
                "X-Audio-Sample-Rate": str(settings.audio_sample_rate)
            }
        )
        
    except Exception as e:
        logger.error(f"Read-aloud error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


@router.websocket("/read/stream")
async def read_stream_websocket(websocket: WebSocket):
    """
    Streamed read-aloud WebSocket endpoint.
    
    Emits audio chunks incrementally for UI "Read Aloud" buttons.
    Supports pause, resume, and cancel operations.
    
    Protocol:
    - Client sends: {"type": "start", "text": "...", "voice": "...", "language": "..."}
    - Server sends: Binary audio chunks (PCM 24kHz)
    - Client sends: {"type": "pause"} to pause
    - Client sends: {"type": "resume"} to resume
    - Client sends: {"type": "cancel"} to stop
    - Client sends: {"type": "ping"} for keepalive
    """
    settings = get_settings()
    
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    
    logger.info(f"Read stream WebSocket connected: {client_id}")
    
    session: Optional[TTSStreamingSession] = None
    tts_client: Optional[TTSClient] = None
    streaming_task: Optional[asyncio.Task] = None
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            try:
                import json
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "start":
                    # Start new streaming session
                    text = data.get("text", "")
                    voice = data.get("voice")
                    language = data.get("language")
                    speed = data.get("speed", 1.0)
                    
                    if not text:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Text is required"
                        })
                        continue
                    
                    # Cancel existing session if any
                    if streaming_task and not streaming_task.done():
                        if session:
                            session.cancel()
                        streaming_task.cancel()
                        try:
                            await streaming_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Initialize new session
                    tts_client = TTSClient(
                        base_url=settings.tts_service_url,
                        model=settings.tts_model_name
                    )
                    session = TTSStreamingSession(tts_client)
                    _active_tts_sessions[client_id] = session
                    
                    # Start streaming
                    async def stream_audio():
                        try:
                            async for chunk in session.start(text, voice, language):
                                await websocket.send_bytes(chunk)
                            
                            # Send completion signal
                            await websocket.send_json({
                                "type": "complete",
                                "message": "Stream finished"
                            })
                            
                        except asyncio.CancelledError:
                            logger.info(f"Streaming cancelled for {client_id}")
                            raise
                        except Exception as e:
                            logger.error(f"Streaming error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })
                        finally:
                            if tts_client:
                                await tts_client.close()
                            if client_id in _active_tts_sessions:
                                del _active_tts_sessions[client_id]
                    
                    streaming_task = asyncio.create_task(stream_audio())
                    
                    await websocket.send_json({
                        "type": "started",
                        "message": "Streaming started"
                    })
                
                elif msg_type == "pause":
                    if session:
                        session.pause()
                        await websocket.send_json({
                            "type": "paused",
                            "message": "Stream paused"
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No active stream"
                        })
                
                elif msg_type == "resume":
                    if session:
                        session.resume()
                        await websocket.send_json({
                            "type": "resumed",
                            "message": "Stream resumed"
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No active stream"
                        })
                
                elif msg_type == "cancel":
                    if session:
                        session.cancel()
                    if streaming_task and not streaming_task.done():
                        streaming_task.cancel()
                        try:
                            await streaming_task
                        except asyncio.CancelledError:
                            pass
                    
                    await websocket.send_json({
                        "type": "cancelled",
                        "message": "Stream cancelled"
                    })
                
                elif msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                
                elif msg_type == "close":
                    break
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        logger.info(f"Read stream WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Read stream error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        # Cleanup
        if session:
            session.cancel()
        if streaming_task and not streaming_task.done():
            streaming_task.cancel()
        if client_id in _active_tts_sessions:
            del _active_tts_sessions[client_id]
        if tts_client:
            await tts_client.close()
        
        logger.info(f"Read stream session cleaned up: {client_id}")