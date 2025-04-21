# built-in imports
import json
import base64
import logging
import asyncio
import traceback
import time

# fastapi imports
from fastapi import WebSocket

# handlers imports
from app.handlers.stream_state import StreamState
from app.handlers.interruption_handler import InterruptionHandler
from app.handlers.tools.tool_response import send_tool_result


# client import
from app.client.client import OpenAIWebSocketClient

# config import

logger = logging.getLogger(__name__)

class OpenAIMessageHandler:
    """Handles responses from OpenAI.
    
    This class processes responses from the OpenAI API, managing audio deltas,
    speech events, and coordinating the response flow back to Twilio.

    Attributes:
        websocket (WebSocket): The WebSocket connection to Twilio
        state (StreamState): The current state of the media stream
        openai_client (OpenAIWebSocketClient): Client for communicating with OpenAI's API
    """
    def __init__(self, websocket: WebSocket, state: StreamState, openai_client: OpenAIWebSocketClient):
        self.websocket = websocket
        self.state = state
        self.openai_client = openai_client

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

            # Only log non-audio response types to reduce overhead
            # if response_type != 'response.audio.delta':
            #     logger.info(f"Processing response type: {response_type}")

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
            # Add explicit handling for function call start
            elif response_type == 'response.function_call.start':
                logger.info(f"Function call started: {response.get('function_call', {}).get('name', 'unknown')}")
            # Add explicit handling for function call arguments delta
            elif response_type == 'response.function_call.arguments.delta':
                logger.debug(f"Function call arguments delta: {response.get('delta', {})}")
            # More visible logging for function call completion
            elif response_type == 'response.function_call_arguments.done':
                logger.info(f"Function call arguments done, processing now...")
                # Log the full message for debugging
                logger.info(f"Function call details: {json.dumps(response)}")
                # Also print directly to console for immediate debugging
                await self._handle_function_call(response)
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
                # You can add additional handling here if needed
        except Exception as e:
            logger.error(f"Error handling transcript: {e}", exc_info=True)

    async def _handle_audio_delta(self, response: dict):
        """Processes audio delta responses from OpenAI."""
        try:
            # Check for both possible audio data formats with minimal overhead
            audio_data = response.get('delta') or response.get('audio')
                
            if not audio_data:
                logger.warning("Audio delta missing audio data in both 'delta' and 'audio' fields")
                return
                
            # Track audio chunks with minimal logging
            if not hasattr(self, '_audio_chunk_count'):
                self._audio_chunk_count = 0
            self._audio_chunk_count += 1
                
            # Only log occasionally to reduce overhead
            if self._audio_chunk_count % 100 == 0:
                logger.info(f"Processed {self._audio_chunk_count} audio chunks")
                
            # Send the audio data to Twilio
            await self.websocket.send_json({
                "event": "media",
                "streamSid": self.state.stream_sid,
                "media": {
                    "payload": audio_data  # Already base64 encoded
                }
            })
            
            if response.get('item_id'):
                self.state.last_assistant_item = response['item_id']

            if self.state.response_start_timestamp_twilio is None:
                self._set_response_start_timestamp()

            await self._send_mark()
            
        except Exception as e:
            logger.error(f"Error handling audio delta: {e}", exc_info=True)

    async def _handle_speech_started(self):
        """Handles detection of user starting to speak.
        
        Logs the speech start event and initiates interruption handling if there's 
        an ongoing assistant response. Sends clear event to Twilio and resets state.

        Returns:
            None
        """
        try:
            logger.info("Speech started detected")
            
            # If there's an ongoing response, handle interruption
            if self.state.mark_queue and self.state.response_start_timestamp_twilio is not None:
                logger.info("Interruption detected, handling...")
                await InterruptionHandler(self.websocket, self.state, self.openai_client).handle()
                
                # Send clear event to Twilio
                await self.websocket.send_json({
                    "event": "clear",
                    "streamSid": self.state.stream_sid
                })
                
                # Reset state
                self.state.mark_queue.clear()
                self.state.last_assistant_item = None
                self.state.response_start_timestamp_twilio = None
            
            self.state.response_start_timestamp_twilio = self.state.latest_media_timestamp

        except Exception as e:
            logger.error(f"Error handling speech started: {e}", exc_info=True)

    async def _handle_speech_stopped(self):
        """Handles detection of user stopping speech."""
        try:
            logger.info("Speech stopped detected")
                
            # Following the twilio-realtime-main approach:
            # Do NOT request a response after speech ends
            # Let OpenAI decide when to respond naturally
            logger.info("Not requesting a response after speech stopped - letting OpenAI respond naturally")
            
        except Exception as e:
            logger.error(f"Error handling speech stopped: {e}", exc_info=True)

    async def _request_response_with_retry(self, max_retries=3, delay=1.0):
        """Request a response from OpenAI with retries."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Requesting response from OpenAI after speech ended (attempt {attempt}/{max_retries})")
                await self.openai_client.send({
                    "type": "response.create"
                })
                logger.info("Response request sent to OpenAI")
                return True
            except Exception as e:
                logger.error(f"Error requesting response (attempt {attempt}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    
        logger.error(f"Failed to request response after {max_retries} attempts")
        return False

    async def _handle_text_delta(self, response: dict):
        """Handle text delta responses from OpenAI."""
        try:
            text = response.get('text', '')
            if text:
                logger.info(f"Assistant: {text}")
        except Exception as e:
            logger.error(f"Error handling text delta: {e}", exc_info=True)

    async def _handle_function_call(self, response: dict):
        """Handle function call responses from OpenAI (simplified placeholder)."""
        try:
            logger.info(f"Function call received: {response.get('type')}")
            
            # Extract basic function info - simplified for now
            function_name = response.get('name')
            call_id = response.get('call_id')
            
            logger.info(f"Function call from realtime API - Name: {function_name}, ID: {call_id}")
            
            # Send a simple placeholder result
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
            
            # Notify Twilio of the error
            await self.websocket.send_json({
                "event": "error",
                "streamSid": self.state.stream_sid,
                "message": error_msg
            })
        except Exception as e:
            logger.error(f"Error handling error response: {e}", exc_info=True)

    def _set_response_start_timestamp(self):
        """Records the timestamp when assistant starts responding.
        
        Updates state with current media timestamp and optionally logs timing
        information if debug flag is enabled.

        Returns:
            None
        """
        self.state.response_start_timestamp_twilio = self.state.latest_media_timestamp

    async def _send_mark(self):
        """Sends a mark event to Twilio for response synchronization.
        
        Creates and sends a mark event if stream ID exists, and adds
        the mark to the state's queue for tracking.

        Returns:
            None
        """
        if self.state.stream_sid:
            try:
                mark_event = {
                    "event": "mark",
                    "streamSid": self.state.stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await self.websocket.send_json(mark_event)
                self.state.mark_queue.append('responsePart')
                logger.debug("Sent mark event")
            except Exception as e:
                logger.error(f"Error sending mark: {e}", exc_info=True)