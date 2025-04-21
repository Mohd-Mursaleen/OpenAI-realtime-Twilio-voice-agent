# built-in imports
import json
import logging

# fastapi imports
from fastapi import WebSocket

# handlers imports
from app.handlers.stream_state import StreamState

# client import
from app.client.client import OpenAIWebSocketClient

logger = logging.getLogger(__name__)

class InterruptionHandler:
    """Handles user interruptions of assistant speech.

    This class manages the process of gracefully handling user interruptions during assistant responses.
    It coordinates truncating the current response, clearing the stream state, and resetting for the
    next interaction.

    Attributes:
        websocket (WebSocket): Connection to send events to Twilio
        state (StreamState): Current state of the media stream
        openai_client (OpenAIWebSocketClient): Client for sending events to OpenAI
    """

    def __init__(
        self, websocket: WebSocket, state: StreamState, openai_client: OpenAIWebSocketClient
    ):
        self.websocket = websocket
        self.state = state
        self.openai_client = openai_client

    async def handle(self):
        """Manages the interruption flow.

        Calculates elapsed time since response started, sends truncation event to OpenAI
        to stop the current response, clears the stream state, and resets for next interaction.
        Early returns if no response is currently in progress.
        """

        # If there's no assistant response start timestamp, return
        if self.state.response_start_timestamp_twilio is None:
            logger.debug("No active response to interrupt")
            return

        # Add debounce to prevent repeated interrupts
        now_ts = self.state.latest_media_timestamp
        if self.state.last_interrupt_ts and (now_ts - self.state.last_interrupt_ts < 1000):
            logger.debug("Ignoring interruption: debounce active")
            return
        self.state.last_interrupt_ts = now_ts

        # If there's a last assistant item, send a truncation event
        if self.state.last_assistant_item:
            logger.info(f"Handling interruption for assistant item: {self.state.last_assistant_item}")

            # Send mark before interruption for analytics
            await self.websocket.send_json({
                "event": "mark",
                "streamSid": self.state.stream_sid,
                "mark": {"name": "interrupted"}
            })

            # Calculate elapsed time since response started
            elapsed_time = (
                self.state.latest_media_timestamp
                - self.state.response_start_timestamp_twilio
            )
            
            await self._send_truncate_event(elapsed_time)

            # Always send clear event to immediately stop playback
            await self.websocket.send_json({
                "event": "clear",
                "streamSid": self.state.stream_sid
            })

        self.state.response_start_timestamp_twilio = None
        self.state.last_assistant_item = None
        

        
        # Reset audio chunk count for clean logs
        if hasattr(self, '_audio_chunk_count'):
            self._audio_chunk_count = 0
            
        logger.info("Interruption handled, state reset")

    async def _send_truncate_event(self, elapsed_time: int):
        """Sends truncation event to OpenAI to stop the current response.

        Args:
            elapsed_time (int): Milliseconds elapsed since response started, used to
                              indicate where to truncate the audio
        """
        # Clamp elapsed_time to the actual audio duration to prevent issues

        # Send a truncate event
        truncate_event = {
            "type": "conversation.item.truncate",
            "item_id": self.state.last_assistant_item,
            "content_index": 0,
            "audio_end_ms": elapsed_time,
        }
        await self.openai_client.send(truncate_event)  # client.send now handles dict conversion
        logger.info(f"Sent truncate event for item {self.state.last_assistant_item}")
