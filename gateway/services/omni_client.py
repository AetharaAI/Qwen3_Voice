"""
WebSocket client for vLLM-Omni service.
Handles audio streaming to/from the Omni model.
"""

import asyncio
import json
import logging
import base64
from typing import Optional, Callable, AsyncIterator, Dict, Any
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class OmniClient:
    """WebSocket client for vLLM-Omni real-time voice interaction."""
    
    def __init__(
        self,
        base_url: str,
        ws_url: str,
        model: str = "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
        sample_rate: int = 24000
    ):
        self.base_url = base_url.rstrip('/')
        self.ws_url = ws_url.rstrip('/')
        self.model = model
        self.sample_rate = sample_rate
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._session_id: Optional[str] = None
    
    async def connect(self) -> bool:
        """Connect to Omni WebSocket endpoint."""
        try:
            # Build WebSocket URL with model parameter
            ws_endpoint = f"{self.ws_url}/v1/audio/chat/completions"
            
            logger.info(f"Connecting to Omni WebSocket: {ws_endpoint}")
            
            self.ws = await websockets.connect(
                ws_endpoint,
                ping_interval=20,
                ping_timeout=20,
                max_size=16 * 1024 * 1024
            )
            
            # Send initial configuration
            init_msg = {
                "model": self.model,
                " modalities": ["audio", "text"],
                "audio": {
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "channels": 1,
                    "bit_depth": 16
                },
                "stream": True
            }
            
            await self.ws.send(json.dumps(init_msg))
            
            # Wait for acknowledgment
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            if response_data.get("type") == "session.created":
                self._session_id = response_data.get("session", {}).get("id")
                self._connected = True
                logger.info(f"Omni WebSocket connected, session: {self._session_id}")
                return True
            else:
                logger.error(f"Unexpected response from Omni: {response_data}")
                await self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Omni WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Omni WebSocket."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None
                self._connected = False
                self._session_id = None
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio chunk to Omni service."""
        if not self._connected or not self.ws:
            logger.error("Cannot send audio: not connected")
            return False
        
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            
            await self.ws.send(json.dumps(message))
            return True
            
        except ConnectionClosed:
            logger.error("WebSocket connection closed while sending audio")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            return False
    
    async def commit_audio(self) -> bool:
        """Signal end of audio input to trigger generation."""
        if not self._connected or not self.ws:
            return False
        
        try:
            message = {
                "type": "input_audio_buffer.commit"
            }
            await self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error committing audio: {e}")
            return False
    
    async def interrupt(self) -> bool:
        """Send interruption signal to stop current generation."""
        if not self._connected or not self.ws:
            return False
        
        try:
            message = {
                "type": "response.cancel"
            }
            await self.ws.send(json.dumps(message))
            logger.info("Sent interruption signal")
            return True
        except Exception as e:
            logger.error(f"Error sending interrupt: {e}")
            return False
    
    async def receive_stream(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Receive streaming responses from Omni.
        Yields parsed message dictionaries.
        """
        if not self._connected or not self.ws:
            logger.error("Cannot receive: not connected")
            return
        
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    yield data
                    
                    # Check for session termination
                    if data.get("type") == "session.terminated":
                        logger.info("Session terminated by server")
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message[:100]}")
                    
        except ConnectionClosed as e:
            logger.info(f"WebSocket connection closed: {e}")
            self._connected = False
        except Exception as e:
            logger.error(f"Error receiving from WebSocket: {e}")
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self.ws is not None
    
    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._session_id


class OmniStreamingSession:
    """Manages a single streaming session with Omni."""
    
    def __init__(
        self,
        client: OmniClient,
        output_callback: Optional[Callable[[bytes], None]] = None
    ):
        self.client = client
        self.output_callback = output_callback
        self._active = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the streaming session."""
        if not self.client.is_connected():
            if not await self.client.connect():
                raise ConnectionError("Failed to connect to Omni service")
        
        self._active = True
        self._task = asyncio.create_task(self._stream_handler())
    
    async def stop(self):
        """Stop the streaming session."""
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.client.disconnect()
    
    async def send_audio_chunk(self, chunk: bytes):
        """Send an audio chunk."""
        await self.client.send_audio(chunk)
    
    async def _stream_handler(self):
        """Handle incoming audio stream from Omni."""
        try:
            async for message in self.client.receive_stream():
                if not self._active:
                    break
                
                msg_type = message.get("type")
                
                if msg_type == "response.audio.delta":
                    # Received audio chunk
                    audio_b64 = message.get("delta", "")
                    if audio_b64 and self.output_callback:
                        audio_data = base64.b64decode(audio_b64)
                        await self._call_output_callback(audio_data)
                
                elif msg_type == "response.audio.done":
                    logger.info("Audio generation complete")
                
                elif msg_type == "response.text.delta":
                    # Text transcription/response
                    text = message.get("delta", "")
                    logger.debug(f"Text delta: {text}")
                
                elif msg_type == "error":
                    logger.error(f"Omni error: {message.get('error', 'Unknown')}")
                    
        except asyncio.CancelledError:
            logger.info("Stream handler cancelled")
            raise
        except Exception as e:
            logger.error(f"Stream handler error: {e}")
    
    async def _call_output_callback(self, audio_data: bytes):
        """Call output callback with audio data."""
        if self.output_callback:
            if asyncio.iscoroutinefunction(self.output_callback):
                await self.output_callback(audio_data)
            else:
                self.output_callback(audio_data)