from fastapi import APIRouter, WebSocket, Request, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
import os
import logging
import traceback
from typing import Dict
from starlette.websockets import WebSocketDisconnect
import time
from app.client.pool_instance import client_pool
from app.handlers.media_stream_handler import handle_media_stream
from twilio.rest import Client

# Configure logging
logger = logging.getLogger(__name__)

# Initialize API router
router = APIRouter()

# Get Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Initialize Twilio client
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {e}")
    twilio_client = None

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

@router.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request, from_phone_number: str = Query(None)):
    """
    Handle incoming call and return TwiML response.
    
    For inbound calls: Uses the Twilio request data
    For outbound calls: Uses the FROM phone number passed as a query parameter
    """
    try:
        form_data = await request.form()
        
        response = VoiceResponse()
        webhook_url = os.environ.get("WEBHOOK_URL")        
        
        if webhook_url:
            webhook_url = webhook_url.replace("https://", "")
            hostname = webhook_url.split("/")[0]
        else:
            hostname = request.headers.get("host") or request.url.hostname

        # Set up media stream (for simplicity, we're using a fixed assistant ID here)
        connect = Connect()
        assistant_id = "default-assistant"
        ws_url = f"wss://{hostname}/media-stream/{assistant_id}"
        logger.info(f"WebSocket URL: {ws_url}")
        
        connect.stream(url=ws_url)
        response.append(connect)

        logger.info("TwiML response generated successfully")
        return HTMLResponse(
            content=str(response),
            media_type="application/xml"
        )
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/media-stream/{assistant_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    assistant_id: str
):
    """Handle WebSocket connection for media streaming."""
    logger.info(f"New WebSocket connection request for assistant: {assistant_id}")
    
    # Accept the WebSocket connection first - do this immediately
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
    
    try:
        logger.info("Starting media stream handling")
        # Handle the media stream
        await handle_media_stream(websocket, openai_client)
        
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

@router.post("/make-call")
async def initiate_call(request: Request, x_api_key: str = Header(..., description="API key for authentication")) -> Dict[str, str]:
    """
    Initiate an outbound call using Twilio.
    
    Request body should contain:
    - to_number: The phone number to call
    - from_number: Optional phone number to call from (uses default if not provided)
    """
    # Simple API key check
    api_key = os.environ.get("API_KEY")
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not twilio_client:
        raise HTTPException(status_code=500, detail="Twilio client not initialized")
    
    try:
        data = await request.json()
        to_number = data.get("to_phone_number")
        from_number = data.get("from_phone_number", TWILIO_PHONE_NUMBER)
        
        if not to_number:
            raise HTTPException(status_code=400, detail="to_number is required")
            
        if not from_number:
            raise HTTPException(status_code=400, detail="from_number not provided and no default configured")
        
        # Get webhook URL for Twilio to connect back to
        webhook_url = os.environ.get("WEBHOOK_URL")
        if not webhook_url:
            webhook_url = f"https://{request.headers.get('host')}/incoming-call"
        
        # Add parameters for the outbound call
        webhook_url += f"?from_phone_number={from_number}"
        
        logger.info(f"Initiating outbound call to {to_number} from {from_number}")
        logger.info(f"Webhook URL: {webhook_url}")
        
        # Create the call
        call = twilio_client.calls.create(
            to=to_number,
            from_=from_number,
            url=webhook_url,
        )
        
        logger.info(f"Call initiated with SID: {call.sid}")
        
        return {
            "status": "success",
            "call_sid": call.sid,
            "message": f"Call initiated to {to_number}"
        }
    except Exception as e:
        logger.error(f"Error making call: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 