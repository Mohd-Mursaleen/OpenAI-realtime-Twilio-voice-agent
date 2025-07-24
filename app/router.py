from fastapi import APIRouter, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
import os
import logging
import traceback
import asyncio
from websockets.exceptions import ConnectionClosed
from starlette.websockets import WebSocketDisconnect
from app.client.pool_instance import client_pool
from app.handlers.stream_state import StreamState
from app.handlers.openai_handler import OpenAIMessageHandler
import json
from app.client.client import OpenAIWebSocketClient
# Configure logging
logger = logging.getLogger(__name__)

# Initialize API router
router = APIRouter()

@router.websocket("/audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection for audio streaming directly to OpenAI."""
    logger.info("New WebSocket connection request")
    
    # Accept the WebSocket connection first
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    logger.info("Initializing OpenAI client...")
    openai_client = OpenAIWebSocketClient()
    await openai_client.connect()  
    
    # Create stream state
    state = StreamState()
    
    # Initialize handler
    openai_handler = OpenAIMessageHandler(websocket, state, openai_client)
    
    try:
        logger.info("Starting audio stream handling")
        
        # Send confirmation to client
        await websocket.send_json({
            "event": "connection_ready",
            "message": "OpenAI client connected and ready to receive audio"
        })
        
        logger.info("WebSocket connection established and OpenAI client ready")
        
        # Main message handling loop
        async def receive_audio():
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    logger.debug(f"Received message type: {message_type}")
                    
                    # Handle different message types
                    if message_type == 'audio':
                        # Forward audio data to OpenAI in the correct format
                        await openai_client.send({
                            "type": "input_audio_buffer.append",
                            "audio": data['audio']  # base64 encoded audio from client
                        })
                        logger.debug("Forwarded audio data to OpenAI")
                        
                    elif message_type == 'start_speaking':
                        # Handle when user starts speaking
                        await openai_client.send({
                            "type": "input_audio_buffer.start"
                        })
                        logger.info("Sent start speaking to OpenAI")
                        
                    elif message_type == 'end_speaking':
                        # Handle when user stops speaking
                        await openai_client.send({
                            "type": "input_audio_buffer.commit"
                        })
                        # Request response from OpenAI
                        await openai_client.send({
                            "type": "response.create"
                        })
                        logger.info("Committed audio buffer and requested response")
                        
                    elif message_type == 'session.create':
                        # Client is requesting session creation - this is handled automatically
                        logger.info("Client requested session creation (handled automatically)")
                        
                    elif message_type == 'input_audio_buffer.commit':
                        # Forward commit request to OpenAI
                        await openai_client.send({
                            "type": "input_audio_buffer.commit"
                        })
                        logger.info("Committed audio buffer")
                        
                    elif message_type == 'response.create':
                        # Forward response creation request to OpenAI
                        await openai_client.send({
                            "type": "response.create"
                        })
                        logger.info("Requested response from OpenAI")
                        
                    elif message_type == 'input_audio_buffer.clear':
                        # Forward clear request to OpenAI
                        await openai_client.send({
                            "type": "input_audio_buffer.clear"
                        })
                        logger.info("Cleared audio buffer")
                        
                    else:
                        logger.warning(f"Unhandled message type from client: {message_type}")
                        
            except WebSocketDisconnect:
                logger.info("Client disconnected")
            except Exception as e:
                logger.error(f"Error in receive_audio: {e}")
                raise
        
        async def process_openai_messages():
            try:
                while True:
                    if not openai_client.connected:
                        logger.warning("OpenAI client disconnected")
                        break
                    
                    try:
                        # More aggressive polling
                        for _ in range(5):  # Process multiple messages if available
                            try:
                                response = await asyncio.wait_for(
                                    openai_client.receive_message(),
                                    timeout=0.01  # Short timeout for more responsive polling
                                )
                                await openai_handler.process_response(response)
                            except asyncio.TimeoutError:
                                # No more messages available
                                break
                            except ConnectionClosed:
                                logger.error("OpenAI connection closed")
                                return
                    except Exception as e:
                        logger.error(f"Error processing OpenAI message: {e}")
                    
                    await asyncio.sleep(0.01)  # Prevent CPU spinning but keep responsive
            except Exception as e:
                logger.error(f"Error in process_openai_messages: {e}")
                raise
        
        # Run both tasks concurrently
        await asyncio.gather(
            receive_audio(),
            process_openai_messages()
        )
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({
                "event": "error",
                "message": f"Internal server error: {str(e)}"
            })
        except:
            pass
    finally:
        # Return the client to the pool
        await client_pool.release_client(openai_client)
        logger.info("Websocket endpoint completed, client released to pool")