# built-in imports
import json
import logging
import asyncio

# fastapi imports
from fastapi import WebSocket

# handlers imports
from app.handlers.stream_state import StreamState
import time
from app.handlers.tools.tool_response import send_tool_result


# client import
from app.client.client import OpenAIWebSocketClient

logger = logging.getLogger(__name__)

class OpenAIMessageHandler:
    """Handles responses from OpenAI.
    
    This class processes responses from the OpenAI API, managing audio deltas,
    speech events, and coordinating the response flow to the client.

    Attributes:
        websocket (WebSocket): The WebSocket connection to the client
        state (StreamState): The current state of the media stream
        openai_client (OpenAIWebSocketClient): Client for communicating with OpenAI's API
    """
    def __init__(self, websocket: WebSocket, state: StreamState, openai_client: OpenAIWebSocketClient):
        self.websocket = websocket
        self.state = state
        self.openai_client = openai_client
        # Track actual audio duration for interruption handling
        self.audio_duration_ms = 0
        self.audio_start_time = None

    async def process_response(self, response: dict):
        """Routes OpenAI responses to appropriate handlers."""
        try:
            if not isinstance(response, dict):
                logger.warning(f"Received non-dict response: {type(response)}")
                return

            response_type = response.get('type')
            if not response_type:
                logger.warning("Response missing type field")
                return

            if response_type == 'session.created' or response_type == 'session.updated':
                logger.info(f"Session {response_type.split('.')[1]} successfully")
            elif response_type == 'response.audio.delta':
                await self._handle_audio_delta(response)
            elif response_type == 'input_audio_buffer.speech_started':
                await self._handle_speech_started()
            elif response_type == 'input_audio_buffer.speech_stopped':
                await self._handle_speech_stopped()
            elif response_type == 'response.text.delta':
                await self._handle_text_delta(response)
            elif response_type == 'response.function_call.start':
                logger.info(f"Function call started: {response.get('function_call', {}).get('name', 'unknown')}")
            elif response_type == 'response.function_call.arguments.delta':
                logger.debug(f"Function call arguments delta: {response.get('delta', {})}")
            elif response_type == 'response.function_call_arguments.done':
                await self._handle_function_call(response)
            elif response_type == 'response.done':
                await self._handle_response_done(response)
            elif response_type == 'error':
                await self._handle_error(response)
            else:
                logger.debug(f"Unhandled response type: {response_type}")

        except Exception as e:
            logger.error(f"Error processing response: {e}", exc_info=True)

    async def _handle_transcript(self, response: dict):
        """Handle transcript responses from OpenAI."""
        try:
            text = response.get('text', '')
            if text:
                logger.info(f"Transcript: {text}")
                await self.websocket.send_json({
                    "type": "transcript",
                    "text": text
                })
        except Exception as e:
            logger.error(f"Error handling transcript: {e}", exc_info=True)

    async def _handle_audio_delta(self, response: dict):
        """Processes audio delta responses from OpenAI."""
        try:
            # Check if WebSocket is still connected
            if self.websocket.client_state.name != 'CONNECTED':
                logger.debug("WebSocket not connected, skipping audio delta")
                return
                
            # OpenAI sends audio data in the 'delta' field for audio deltas
            audio_data = response.get('delta')
                
            if not audio_data:
                logger.warning("Audio delta missing audio data in 'delta' field")
                logger.debug(f"Response structure: {list(response.keys())}")
                return
                
            # Track audio chunks with minimal logging
            if not hasattr(self, '_audio_chunk_count'):
                self._audio_chunk_count = 0
                logger.info("Starting to receive audio chunks from OpenAI")
                # Reset audio tracking when new audio starts
                self.audio_duration_ms = 0
                self.audio_start_time = time.time() * 1000  # Convert to milliseconds
            
            self._audio_chunk_count += 1
            
            # Estimate audio duration based on chunk count and timing
            # This is a rough estimate - each chunk is typically ~20ms of audio
            current_time = time.time() * 1000
            if self.audio_start_time:
                self.audio_duration_ms = current_time - self.audio_start_time
                
            # Only log occasionally to reduce overhead
            if self._audio_chunk_count % 50 == 0:
                logger.info(f"Processed {self._audio_chunk_count} audio chunks, estimated duration: {self.audio_duration_ms:.0f}ms")
                
            # Send the audio data to client with error handling
            try:
                await self.websocket.send_json({
                    "type": "audio",
                    "audio": audio_data  # Already base64 encoded from OpenAI
                })
            except RuntimeError as e:
                if "websocket.close" in str(e):
                    logger.info("WebSocket closed, stopping audio transmission")
                    return
                else:
                    raise
            
            # Update state tracking
            if response.get('item_id'):
                self.state.last_assistant_item = response['item_id']

            if self.state.response_start_time is None:
                self.state.response_start_time = self.state.latest_timestamp
                
            # Mark assistant as speaking
            self.state.is_assistant_speaking = True
            
        except Exception as e:
            logger.error(f"Error handling audio delta: {e}", exc_info=True)

    async def _handle_speech_started(self):
        """Handles detection of user starting to speak."""
        try:
            logger.info("Speech started detected")
            
            # If there's an ongoing response, handle interruption
            if self.state.is_assistant_speaking:
                logger.info("Interruption detected, handling...")
                await self._handle_interruption()
            
            self.state.is_user_speaking = True
            self.state.latest_timestamp = self.state.get_current_timestamp()

        except Exception as e:
            logger.error(f"Error handling speech started: {e}", exc_info=True)

    async def _handle_speech_stopped(self):
        """Handles detection of user stopping speech."""
        try:
            logger.info("Speech stopped detected")
            self.state.is_user_speaking = False
            
        except Exception as e:
            logger.error(f"Error handling speech stopped: {e}", exc_info=True)

    async def _handle_interruption(self):
        """Handles interruption of assistant's speech with improved logic."""
        try:
            if not self.state.last_assistant_item:
                logger.debug("No active assistant speech to interrupt")
                return
                
            # Use the actual audio duration we've been tracking
            actual_audio_duration = self.audio_duration_ms
            
            # Only interrupt if there's been sufficient audio
            if actual_audio_duration < 400:  # Less than 500ms, too short to truncate
                logger.debug(f"Skipping interruption, audio too short: {actual_audio_duration:.0f}ms")
                return
                
            # Use 90% of actual duration to ensure we don't exceed the audio length
            safe_truncate_time = max(100, actual_audio_duration * 0.9)
            
            logger.info(f"Interrupting assistant speech at {safe_truncate_time:.0f}ms (actual duration: {actual_audio_duration:.0f}ms)")
            
            # Send truncate event to OpenAI
            truncate_event = {
                "type": "conversation.item.truncate",
                "item_id": self.state.last_assistant_item,
                "content_index": 0,
                "audio_end_ms": int(safe_truncate_time)
            }
            await self.openai_client.send(truncate_event)
            
            # Reset state
            self._reset_audio_tracking()
            
        except Exception as e:
            logger.error(f"Error handling interruption: {e}", exc_info=True)

    async def _handle_response_done(self, response: dict):
        """Handle response completion from OpenAI."""
        try:
            logger.info("Response completed")
            self._reset_audio_tracking()
        except Exception as e:
            logger.error(f"Error handling response done: {e}", exc_info=True)

    def _reset_audio_tracking(self):
        """Reset audio tracking state."""
        self.state.last_assistant_item = None
        self.state.response_start_time = None
        self.state.is_assistant_speaking = False
        self.audio_duration_ms = 0
        self.audio_start_time = None
        if hasattr(self, '_audio_chunk_count'):
            delattr(self, '_audio_chunk_count')

    async def _handle_text_delta(self, response: dict):
        """Handle text delta responses from OpenAI."""
        try:
            text = response.get('text', '')
            if text:
                logger.info(f"Assistant: {text}")
                await self.websocket.send_json({
                    "type": "text",
                    "text": text
                })
        except Exception as e:
            logger.error(f"Error handling text delta: {e}", exc_info=True)

    async def _handle_function_call(self, response: dict):
        """Handle function call responses from OpenAI."""
        try:
            logger.info(f"Function call received: {response.get('type')}")
            
            function_name = response.get('name')
            call_id = response.get('call_id')
            
            logger.info(f"Function call from realtime API - Name: {function_name}, ID: {call_id}")
            
            if call_id:
                result = {
                    "status": "success",
                    "message": f"Function {function_name} called, but tools are not implemented yet"
                }
                
                await send_tool_result(self.openai_client, call_id, result)
                logger.info(f"Sent placeholder result for function {function_name}")
            else:
                logger.warning(f"No call_id found in function call response")
                
        except Exception as e:
            logger.error(f"Error in _handle_function_call: {e}", exc_info=True)

    async def _handle_error(self, response: dict):
        """Handle error responses from OpenAI."""
        try:
            error_msg = response.get('error', {}).get('message', 'Unknown error')
            logger.error(f"OpenAI error: {error_msg}")
            
            # If it's an interruption error, reset state and continue
            if "already shorter than" in error_msg:
                logger.info("Interruption timing error, resetting state")
                self._reset_audio_tracking()
                return
            
            await self.websocket.send_json({
                "type": "error",
                "message": error_msg
            })
        except Exception as e:
            logger.error(f"Error handling error response: {e}", exc_info=True)