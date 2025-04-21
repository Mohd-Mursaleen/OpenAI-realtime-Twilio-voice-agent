import json
import logging
import asyncio
from app.client.client import OpenAIWebSocketClient

logger = logging.getLogger(__name__)

async def send_tool_result(openai_client: OpenAIWebSocketClient, function_id: str, result: str):
    """Send a tool call result back to OpenAI.
    
    Args:
        openai_client: The OpenAI client to use for sending the message
        function_id: The ID of the function call to respond to
        result: The JSON string result to send back
    """
    try:
        logger.info(f"Sending tool result for function {function_id}")
        
        # Create the tool result message
        message = {
            "type": "function.response",
            "function_id": function_id,
            "response": result
        }
        
        # Send the result
        await openai_client.send(message)
        logger.info(f"Tool result sent successfully")
        
    except Exception as e:
        logger.error(f"Error sending tool result: {e}") 