"""
HTTP client for vLLM-TTS service.
Handles text-to-speech requests via OpenAI-compatible API.
"""

import logging
from typing import Optional, AsyncIterator, Dict, Any, Literal
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TTSRequest:
    """Text-to-speech request parameters."""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    speed: float = 1.0
    response_format: Literal["pcm", "wav", "mp3"] = "pcm"
    sample_rate: int = 24000


@dataclass
class TTSResponse:
    """Text-to-speech response."""
    audio_data: bytes
    content_type: str
    sample_rate: int
    duration_seconds: Optional[float] = None


class TTSClient:
    """HTTP client for TTS service using OpenAI-compatible API."""
    
    def __init__(
        self,
        base_url: str,
        model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True
        )
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if TTS service is healthy."""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"TTS health check failed: {e}")
            return False
    
    async def synthesize(
        self,
        request: TTSRequest
    ) -> TTSResponse:
        """
        Synthesize text to speech (non-streaming).
        Uses OpenAI-compatible /v1/audio/speech endpoint.
        """
        # Build OpenAI-compatible request
        payload = {
            "model": self.model,
            "input": request.text,
            "voice": request.voice or "default",
            "response_format": request.response_format,
            "speed": request.speed
        }
        
        # Add optional parameters
        if request.language:
            payload["language"] = request.language
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "audio/pcm, audio/wav, audio/mpeg, application/octet-stream"
        }
        
        try:
            logger.debug(f"Sending TTS request: {len(request.text)} chars")
            
            response = await self.client.post(
                "/v1/audio/speech",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            audio_data = response.content
            content_type = response.headers.get("content-type", "audio/pcm")
            
            # Calculate duration from audio data
            duration = self._calculate_duration(
                audio_data,
                request.sample_rate,
                request.response_format
            )
            
            logger.info(f"TTS synthesis complete: {len(audio_data)} bytes, {duration:.2f}s")
            
            return TTSResponse(
                audio_data=audio_data,
                content_type=content_type,
                sample_rate=request.sample_rate,
                duration_seconds=duration
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"TTS HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise
    
    async def synthesize_streaming(
        self,
        request: TTSRequest
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to speech with streaming response.
        Yields audio chunks as they become available.
        """
        payload = {
            "model": self.model,
            "input": request.text,
            "voice": request.voice or "default",
            "response_format": request.response_format,
            "speed": request.speed,
            "stream": True
        }
        
        if request.language:
            payload["language"] = request.language
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, audio/pcm, audio/wav"
        }
        
        try:
            logger.debug(f"Sending streaming TTS request: {len(request.text)} chars")
            
            async with self.client.stream(
                "POST",
                "/v1/audio/speech",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                content_type = response.headers.get("content-type", "")
                
                if "text/event-stream" in content_type:
                    # Server-sent events format
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data == "[DONE]":
                                break
                            # Decode base64 audio chunk
                            import base64
                            try:
                                chunk = base64.b64decode(data)
                                yield chunk
                            except Exception:
                                logger.warning(f"Failed to decode audio chunk: {data[:50]}")
                else:
                    # Raw binary stream
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        yield chunk
                        
        except httpx.HTTPStatusError as e:
            logger.error(f"TTS streaming HTTP error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise
    
    def _calculate_duration(
        self,
        audio_data: bytes,
        sample_rate: int,
        format_type: str
    ) -> Optional[float]:
        """Calculate audio duration from raw data."""
        try:
            if format_type == "pcm":
                # PCM: 16-bit = 2 bytes per sample
                num_samples = len(audio_data) // 2
                return num_samples / sample_rate
            elif format_type == "wav":
                # WAV header is 44 bytes, data is after
                # Parse header to get actual data size
                if len(audio_data) < 44:
                    return None
                # Subchunk2Size at offset 40-43
                import struct
                data_size = struct.unpack('<I', audio_data[40:44])[0]
                # Bits per sample at offset 34-35
                bits_per_sample = struct.unpack('<H', audio_data[34:36])[0]
                bytes_per_sample = bits_per_sample // 8
                num_samples = data_size // bytes_per_sample
                return num_samples / sample_rate
            else:
                return None
        except Exception as e:
            logger.warning(f"Failed to calculate duration: {e}")
            return None
    
    async def get_voices(self) -> list:
        """Get available voices from TTS service."""
        try:
            response = await self.client.get("/v1/audio/voices")
            if response.status_code == 200:
                data = response.json()
                return data.get("voices", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []


class TTSStreamingSession:
    """Manages a streaming TTS session with pause/resume support."""
    
    def __init__(self, client: TTSClient):
        self.client = client
        self._paused = False
        self._cancelled = False
        self._stream_iterator: Optional[AsyncIterator[bytes]] = None
    
    async def start(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Start streaming TTS."""
        self._cancelled = False
        self._paused = False
        
        request = TTSRequest(
            text=text,
            voice=voice,
            language=language,
            response_format="pcm"
        )
        
        self._stream_iterator = self.client.synthesize_streaming(request)
        
        async for chunk in self._stream_iterator:
            if self._cancelled:
                break
            
            while self._paused and not self._cancelled:
                await asyncio.sleep(0.1)
            
            if not self._cancelled:
                yield chunk
    
    def pause(self):
        """Pause the stream."""
        self._paused = True
        logger.info("TTS stream paused")
    
    def resume(self):
        """Resume the stream."""
        self._paused = False
        logger.info("TTS stream resumed")
    
    def cancel(self):
        """Cancel the stream."""
        self._cancelled = True
        logger.info("TTS stream cancelled")
    
    @property
    def is_paused(self) -> bool:
        """Check if stream is paused."""
        return self._paused