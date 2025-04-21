import logging
import json
import base64
import asyncio
from typing import Dict, Any, Callable, Optional
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from app.handlers.stream_state import StreamState
from app.client.client import OpenAIWebSocketClient
from app.handlers.twilio_handler import TwilioMessageHandler
from app.handlers.openai_handler import OpenAIMessageHandler

logger = logging.getLogger(__name__)

class MediaStreamHandler:
    """Coordinates bidirectional media streaming between Twilio and OpenAI.
    
    This class serves as the main coordinator for the WebSocket connection,
    managing the bidirectional flow of audio data between Twilio and OpenAI.
    It initializes and maintains the necessary handlers and state management
    for the streaming session.

    Attributes:
        websocket (WebSocket): The WebSocket connection to Twilio
        openai_client (OpenAIWebSocketClient): Client for communicating with OpenAI's API
        state (StreamState): The current state of the media stream
        twilio_handler (TwilioMessageHandler): Handler for Twilio messages
        openai_handler (OpenAIMessageHandler): Handler for OpenAI responses
    """
    def __init__(self, websocket: WebSocket, openai_client: OpenAIWebSocketClient):
        self.websocket = websocket
        self.openai_client = openai_client
        self.state = StreamState()
        self.twilio_handler = TwilioMessageHandler(websocket, self.state, openai_client)
        self.openai_handler = OpenAIMessageHandler(websocket, self.state, openai_client)

    async def handle_connection(self):
        """Main entry point for handling WebSocket connections.
        
        Accepts the initial WebSocket connection, establishes connection with OpenAI,
        and starts concurrent tasks for bidirectional communication between Twilio
        and OpenAI. Handles any connection errors and ensures proper cleanup.

        Raises:
            Exception: If any error occurs during the connection handling process
        """
        logger.info("Client connected")
        
        # Send initial status
        await self.websocket.send_json({
            "event": "status_update",
            "message": "Connection established"
        })

        try:
            await asyncio.gather(
                self._receive_from_twilio(),
                self._process_openai_messages()
            )
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Performs cleanup operations when the connection terminates.
        
        Checks if the OpenAI WebSocket connection is still open and closes it
        properly to prevent resource leaks and ensure clean termination.
        """
        logger.info("Cleaning up connection")
        if self.openai_client.connected:
            await self.openai_client.close()

    async def _receive_from_twilio(self):
        """Handles incoming messages from Twilio's WebSocket connection.
        
        Continuously listens for incoming messages from Twilio, deserializes the
        JSON data, and processes them through the Twilio message handler. Handles
        disconnection events gracefully.

        Raises:
            WebSocketDisconnect: When the client disconnects from the WebSocket
        """
        try:
            async for message in self.websocket.iter_text():
                data = json.loads(message)
                event_type = data.get('event', 'unknown')
                logger.debug(f"Received Twilio message type: {event_type}")
                
                # Send acknowledgment for media events
                if event_type == 'media':
                    await self.websocket.send_json({
                        "event": "media_ack",
                        "streamSid": data.get("streamSid", "")
                    })
                
                await self.twilio_handler.process_message(data)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
            await self._cleanup()
        except Exception as e:
            logger.error(f"Error in receive_from_twilio: {e}")
            raise

    async def _process_openai_messages(self):
        """Processes messages from OpenAI and sends them to Twilio."""
        try:
            while True:
                if not self.openai_client.connected:
                    logger.warning("OpenAI client disconnected")
                    break
                
                try:
                    response = await self.openai_client.receive_message()
                    response_type = response.get('type', 'unknown')
                    logger.debug(f"Processing OpenAI message type: {response_type}")
                    await self.openai_handler.process_response(response)
                except Exception as e:
                    logger.error(f"Error processing OpenAI message: {e}")
                    continue
                
                await asyncio.sleep(0.05)  # Prevent CPU spinning
        except Exception as e:
            logger.error(f"Error in process_openai_messages: {e}")
            raise


async def handle_media_stream(websocket: WebSocket, openai_client: OpenAIWebSocketClient):
    """Entry point for handling WebSocket connections.
    
    Creates a new MediaStreamHandler instance and initiates the connection
    handling process for a new WebSocket connection from Twilio.

    Args:
        websocket (WebSocket): The incoming WebSocket connection from Twilio
        openai_client (OpenAIWebSocketClient): Client for communicating with OpenAI's API

    Returns:
        None
    """
    handler = MediaStreamHandler(websocket, openai_client)
    await handler.handle_connection()
