"""
Voice endpoints: WebSocket streaming, batch TTS, and transcription.
"""

import asyncio
import logging
import base64
from typing import Optional, Dict, Any
from io import BytesIO

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from config import get_settings
from services.omni_client import OmniClient, OmniStreamingSession

logger = logging.getLogger(__name__)
router = APIRouter()

# Track active WebSocket connections
_active_connections: Dict[str, WebSocket] = {}


class BatchTTSRequest(BaseModel):
    """Batch TTS request model."""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Transcription response model."""
    text: str
    confidence: float
    language: Optional[str] = None


async def handle_omni_stream(
    websocket: WebSocket,
    omni_client: OmniClient
) -> None:
    """
    Handle bidirectional audio streaming with Omni service.
    Receives audio from client, forwards to Omni, returns synthesized audio.
    """
    settings = get_settings()
    
    # Queue for audio chunks to send to client
    output_queue: asyncio.Queue[bytes] = asyncio.Queue()
    
    async def output_handler(audio_data: bytes):
        """Callback for audio output from Omni."""
        await output_queue.put(audio_data)
    
    # Create streaming session
    session = OmniStreamingSession(omni_client, output_callback=output_handler)
    
    try:
        await session.start()
        logger.info("Omni streaming session started")
        
        # Start tasks for receiving from client and sending to client
        client_recv_task = asyncio.create_task(_receive_from_client(websocket, session))
        client_send_task = asyncio.create_task(_send_to_client(websocket, output_queue))
        
        # Wait for either task to complete (usually due to disconnect)
        done, pending = await asyncio.wait(
            [client_recv_task, client_send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error(f"Error in Omni stream handler: {e}")
        raise
    finally:
        await session.stop()
        logger.info("Omni streaming session ended")


async def _receive_from_client(
    websocket: WebSocket,
    session: OmniStreamingSession
) -> None:
    """Receive audio chunks from client WebSocket."""
    try:
        while True:
            # Receive message (binary or text)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio data
                audio_chunk = message["bytes"]
                await session.send_audio_chunk(audio_chunk)
                
            elif "text" in message:
                # Control message
                try:
                    import json
                    control = json.loads(message["text"])
                    msg_type = control.get("type")
                    
                    if msg_type == "audio.commit":
                        await session.client.commit_audio()
                    elif msg_type == "interrupt":
                        await session.client.interrupt()
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg_type == "close":
                        break
                        
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON control message")
                    
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error receiving from client: {e}")


async def _send_to_client(
    websocket: WebSocket,
    output_queue: asyncio.Queue[bytes]
) -> None:
    """Send audio chunks to client WebSocket."""
    try:
        while True:
            # Wait for audio data from Omni
            audio_data = await asyncio.wait_for(
                output_queue.get(),
                timeout=30.0
            )
            
            # Send binary audio data
            await websocket.send_bytes(audio_data)
            
    except asyncio.TimeoutError:
        logger.info("No audio output for 30 seconds, closing stream")
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected while sending")
    except Exception as e:
        logger.error(f"Error sending to client: {e}")


@router.websocket("/voice")
async def voice_websocket(websocket: WebSocket):
    """
    Full-duplex real-time voice interaction (audio in → audio out).
    WebSocket endpoint for conversational voice AI.
    Accepts raw PCM audio (16-bit, 24kHz, mono, little-endian).
    """
    settings = get_settings()
    
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    _active_connections[client_id] = websocket
    
    logger.info(f"WebSocket connection accepted from {client_id}")
    
    # Initialize Omni client
    omni_client = OmniClient(
        base_url=settings.omni_service_url,
        ws_url=settings.omni_ws_url.replace("ws://", "wss://").replace("http://", "ws://"),
        model=settings.omni_model_name,
        sample_rate=settings.audio_sample_rate
    )
    
    try:
        await handle_omni_stream(websocket, omni_client)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        del _active_connections[client_id]
        await omni_client.disconnect()


@router.websocket("/voice/stream")
async def voice_stream_websocket(websocket: WebSocket):
    """
    Semantic alias of /voice.
    Behaves identically to /voice but clearly expresses streaming intent.
    """
    # Delegate to the main voice handler
    await voice_websocket(websocket)


@router.post("/voice/batch")
async def voice_batch(request: BatchTTSRequest):
    """
    Non-streaming text-to-speech generation.
    Returns complete audio file (WAV format).
    
    Used for offline generation and file-based workflows.
    """
    settings = get_settings()
    
    # Validate input
    if not request.text or len(request.text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text must be between 1 and {settings.max_text_length} characters"
        )
    
    try:
        # Use TTS service for batch generation
        from services.tts_client import TTSClient, TTSRequest as TTSReq
        
        tts_client = TTSClient(
            base_url=settings.tts_service_url,
            model=settings.tts_model_name
        )
        
        tts_request = TTSReq(
            text=request.text,
            voice=request.voice,
            language=request.language,
            response_format="wav",
            sample_rate=settings.audio_sample_rate
        )
        
        tts_response = await tts_client.synthesize(tts_request)
        await tts_client.close()
        
        # Return audio file
        return Response(
            content=tts_response.audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Audio-Duration": str(tts_response.duration_seconds or 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Batch TTS error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}"
        )


@router.post("/voice/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV or PCM)"),
    language: Optional[str] = Form(None, description="Expected language (optional)")
):
    """
    Audio → text transcription.
    Accepts audio file and returns transcription with confidence score.
    Used for logging, RAG ingestion, and debugging.
    """
    settings = get_settings()
    
    # Validate file
    if not audio.content_type or not (
        audio.content_type.startswith("audio/") or 
        audio.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Must be audio/wav, audio/pcm, or application/octet-stream"
        )
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file"
            )
        
        # Use Omni model for transcription via HTTP API
        import httpx
        
        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Build request for Omni ASR
        payload = {
            "model": settings.omni_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Transcribe this audio accurately."
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0
        }
        
        if language:
            payload["language"] = language
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.omni_service_url}/v1/chat/completions",
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            
            result = response.json()
            transcription = result["choices"][0]["message"]["content"]
            
            # Calculate pseudo-confidence based on response presence
            confidence = 0.95 if transcription else 0.0
            
            return {
                "text": transcription,
                "confidence": confidence,
                "language": language or "auto"
            }
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Transcription HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Transcription service error: {e.response.status_code}"
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )