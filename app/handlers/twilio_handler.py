# built-in imports
import json
import logging
import base64
import asyncio

# fastapi imports
from fastapi import WebSocket

# handlers imports
from app.handlers.stream_state import StreamState

# Remove the circular import
# from app.client.client import OpenAIWebSocketClient

logger = logging.getLogger(__name__)

class TwilioMessageHandler:
    """Handles incoming messages from Twilio.
    
    This class processes and routes different types of events received from Twilio's media stream,
    including media chunks, stream start events, and mark events. It maintains the stream state
    and forwards audio data to OpenAI.

    Attributes:
        websocket (WebSocket): Connection to send events to Twilio
        state (StreamState): Current state of the media stream
        openai_client (OpenAIWebSocketClient): Client for sending events to OpenAI
    """
    def __init__(self, websocket: WebSocket, state: StreamState, openai_client):
        # Using type annotation as a comment to avoid import
        # openai_client: OpenAIWebSocketClient
        self.websocket = websocket
        self.state = state
        self.openai_client = openai_client

    async def process_message(self, data: dict):
        """Routes incoming Twilio messages to appropriate handlers.
        
        Examines the event type of incoming messages and delegates processing to
        the appropriate handler method. Handles media chunks, stream start events,
        and mark events.

        Args:
            data (dict): The message payload from Twilio containing event information
        """
        event_type = data.get('event')
        logger.debug(f"Processing Twilio message of type: {event_type}")
        
        try:
            if event_type == 'media' and self.openai_client.connected:
                await self._handle_media_event(data)
            elif event_type == 'start':
                await self._handle_start_event(data)
            elif event_type == 'mark':
                await self._handle_mark_event()
            elif event_type == 'stop':
                await self._handle_stop_event(data)
            else:
                logger.debug(f"Unhandled event type: {event_type}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    async def _handle_media_event(self, data: dict):
        """Processes incoming media chunks from Twilio."""
        try:
            if 'media' not in data:
                logger.warning("Received media event without media data")
                return

            media_data = data['media']
            if 'payload' not in media_data:
                logger.warning("Media data missing payload")
                return

            # Update timestamp but minimize logging for regular media chunks
            self.state.latest_media_timestamp = int(media_data.get('timestamp', 0))
            
            # Increment media chunk counter
            self.state.media_count += 1
            
            # Reuse the existing payload directly without additional processing
            # Pre-construct the message format once to avoid repetitive work
            if not hasattr(self, '_audio_append_template'):
                self._audio_append_template = {
                    "type": "input_audio_buffer.append",
                    "audio": ""  # This will be replaced with the actual payload
                }
            
            # Directly modify the template instead of creating a new dictionary each time
            self._audio_append_template["audio"] = media_data['payload']
            
            # Only log detailed info very occasionally to reduce overhead
            if self.state.media_count % 500 == 0:
                chunk_size = len(media_data['payload'])
                logger.debug(f"Sending audio chunk #{self.state.media_count}: {chunk_size} bytes")
            
            # Send audio data with minimal overhead
            await self.openai_client.send(self._audio_append_template)
            
        except Exception as e:
            logger.error(f"Error handling media event: {e}", exc_info=True)

    async def _handle_start_event(self, data: dict):
        """Handles the start of a new media stream.
        
        Resets the stream state and initializes a new stream session with the
        provided stream ID from Twilio.

        Args:
            data (dict): Start event data containing the new stream ID
        """
        try:
            if 'start' not in data or 'streamSid' not in data['start']:
                logger.warning("Invalid start event data")
                return
                
            self.state.reset()
            self.state.stream_sid = data['start']['streamSid']
            logger.info(f"Incoming stream has started: {self.state.stream_sid}")
            
            # Send acknowledgment
            await self.websocket.send_json({
                "event": "connected",
                "streamSid": self.state.stream_sid
            })
            
        except Exception as e:
            logger.error(f"Error handling start event: {e}", exc_info=True)

    async def _handle_mark_event(self):
        """Processes mark events for stream synchronization.
        
        Removes the oldest mark from the queue when a mark event is received,
        helping maintain synchronization between Twilio and OpenAI streams.
        """
        try:
            if self.state.mark_queue:
                mark = self.state.mark_queue.pop(0)
                logger.debug(f"Processed mark event: {mark}")
        except Exception as e:
            logger.error(f"Error handling mark event: {e}", exc_info=True)

    async def _handle_stop_event(self, data: dict):
        """Handles the end of a media stream."""
        try:
            logger.info(f"Stream ending: {self.state.stream_sid}")
            
            # Only send commit if we've actually received media data
            if self.state.media_count > 0:
                logger.info(f"Sending input_audio_buffer.commit after {self.state.media_count} media chunks")
                # Send end of stream marker to OpenAI
                await self.openai_client.send({
                    "type": "input_audio_buffer.commit"
                })
            else:
                logger.warning("No audio data received, skipping input_audio_buffer.commit")
            
            # Following the twilio-realtime-main approach:
            # Do NOT force a response at the end of the call
            # Let OpenAI decide when and if to respond
            logger.info("Call ending - not forcing a final response")
            
        except Exception as e:
            logger.error(f"Error handling stop event: {e}", exc_info=True)