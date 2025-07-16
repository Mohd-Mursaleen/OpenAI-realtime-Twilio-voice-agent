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

# Configure logging
logger = logging.getLogger(__name__)

# Initialize API router
router = APIRouter()

@router.get("/voice-agent", response_class=HTMLResponse)
async def agent_status():
    """Returns the status of the voice agent service."""
    return HTMLResponse(
        content="""
        <html>
            <head>
                <title>Voice Agent</title>
            </head>
            <body>
                <h1>Voice Agent</h1>
                <p>The voice agent service is running.</p>
            </body>
        </html>
        """
    )

@router.websocket("/audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection for audio streaming directly to OpenAI."""
    logger.info("New WebSocket connection request")
    
    # Accept the WebSocket connection first
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # Get a client from the pool
    openai_client = await client_pool.get_client()
    
    if not openai_client:
        logger.error("No available OpenAI clients")
        await websocket.send_json({
            "event": "error",
            "message": "No available OpenAI clients"
        })
        await websocket.close(code=1013, reason="No available OpenAI clients")
        return
    
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
        
        # Main message handling loop
        async def receive_audio():
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get('type') == 'audio':
                        # Forward audio data to OpenAI
                        await openai_client.send({
                            "type": "input_audio_buffer.append",
                            "audio": data['audio']  # Expecting base64 encoded audio
                        })
                    elif data.get('type') == 'start_speaking':
                        # Handle when user starts speaking
                        await openai_client.send({
                            "type": "input_audio_buffer.start"
                        })
                    elif data.get('type') == 'end_speaking':
                        # Handle when user stops speaking
                        await openai_client.send({
                            "type": "input_audio_buffer.commit"
                        })
                        # Request response from OpenAI
                        await openai_client.send({
                            "type": "response.create"
                        })
                        
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