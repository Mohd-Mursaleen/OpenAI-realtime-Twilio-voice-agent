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
from app.agent.handlers.stream_state import StreamState
from app.agent.handlers.interruption_handler import InterruptionHandler
from app.agent.tools.tool_response import send_tool_result
from app.services.google_sheets_access import SpreadsheetAccess

# client import
from app.agent.client import OpenAIWebSocketClient

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
        """Handle function call responses from OpenAI."""
        try:
            # Debug: Print the raw response to stdout for direct debugging
            print("\n\n==== DEBUGGING FUNCTION CALL RESPONSE ====")
            print(f"Response type: {response.get('type')}")
            print(f"Response data: {json.dumps(response, indent=2)}")
            print("==========================================\n\n")
            
            logger.info(f"Handling function call - Response type: {response.get('type')}")
            
            # For realtime API, we need to handle response.function_call_arguments.done directly
            if response.get('type') == 'response.function_call_arguments.done':
                # Twilio RAG example style - directly use the response fields
                function_name = response.get('name')
                arguments = response.get('arguments', '{}')
                call_id = response.get('call_id')  # Get call_id from the top level of the response
                
                print(f"Function call from realtime API - Name: {function_name}, ID: {call_id}")
                print(f"Function arguments: {arguments}")
                
                if not function_name:
                    print("ERROR: No function name found in response")
                    return
                
                try:
                    # Parse arguments if they're a string
                    if isinstance(arguments, str):
                        try:
                            arguments_dict = json.loads(arguments)
                        except json.JSONDecodeError:
                            print(f"WARNING: Could not parse arguments as JSON: {arguments}")
                            arguments_dict = {"raw_arguments": arguments}
                    else:
                        arguments_dict = arguments                    
                    
                    print(f"Executing tool: {function_name}")
                    print(f"Arguments: {arguments_dict}")

                    tools = self.openai_client.assistant_handler.tools
                    tools_metadata = self.openai_client.assistant_handler.tools_metadata
                    spreadsheet_metadata = self.openai_client.assistant_handler.spreadsheet_metadata
                    schemas = self.openai_client.assistant_handler.schemas
                    
                    print("Inside of Handle function call")
                    print(f"Tools: {tools}")
                    print(f"Tools Metadata: {tools_metadata}")
                    print(f"Spreadsheet metadata: {spreadsheet_metadata}")

                    # Get the tool metadata for the function
                    tool_metadata = tools_metadata.get(function_name)
                    result = None
                    
                    if not tool_metadata:
                        print(f"ERROR: No metadata found for tool {function_name}")
                        result = "Error: Tool metadata not found"
                    else:
                        # Initialize the SheetHandler with the assistant ID
                        print(spreadsheet_metadata,"metadata,++++++")
                        spreadsheet_id = spreadsheet_metadata.get('id')
                        spreadsheet = SpreadsheetAccess(spreadsheet_id)
                        
                        operation_type = tool_metadata.get('operation_type')
                        table_reference = tool_metadata.get('table_reference', {})
                        table_name = table_reference.get('table_name')
                        
                        def match_columns(schema_columns, args_dict):
                            """
                            Match column names from schema with argument keys by converting to snake_case.
                            Also ensures the output keys match the spreadsheet headers.
                            """
                            matched_data = {}
                            for column in schema_columns:
                                col_name = column.get('name')
                                # Use snake_case for the output key to match spreadsheet headers
                                spreadsheet_col_name = col_name.lower().replace(' ', '_')
                                
                                # Convert schema column name to snake_case for matching
                                col_name_snake = col_name.lower().replace(' ', '_')
                                
                                # Find matching argument
                                for arg_key, arg_value in args_dict.items():
                                    # Convert argument key to snake_case
                                    arg_key_snake = arg_key.lower().replace(' ', '_')
                                    if arg_key_snake == col_name_snake:
                                        # Store using the spreadsheet column name format
                                        matched_data[spreadsheet_col_name] = arg_value
                                        break
                            
                            return matched_data
                        
                        if not table_name:
                            print(f"ERROR: No table name found for tool {function_name}")
                            result = "Error: Table name not specified in metadata"
                        elif operation_type == 'read':
                            try:
                                print("Reading spreadsheet data...")
                                
                                # Check if we can use vector search first
                                use_vector_search = True  # Default to trying vector search
                                vector_results = None
                                
                                try:
                                    # Create a natural language query from arguments
                                    if arguments_dict:
                                        # Convert arguments to a natural language query
                                        query_parts = []
                                        for key, value in arguments_dict.items():
                                            query_parts.append(f"{key} is {value}")
                                        
                                        query = " and ".join(query_parts)
                                        
                                        # Try semantic search with vector database
                                        start_time = time.time()
                                        vector_results = await self.openai_client.assistant_handler.semantic_search_spreadsheet(
                                            table_name, 
                                            query
                                        )
                                        search_time = time.time() - start_time
                                        
                                        if vector_results:
                                            logger.info(f"VECTOR SEARCH: Found {len(vector_results)} results for '{query}' in {table_name} [{search_time*1000:.2f}ms]")
                                        else:
                                            logger.info(f"VECTOR SEARCH: No results for '{query}' in {table_name} [{search_time*1000:.2f}ms]")
                                            use_vector_search = False
                                    else:
                                        # No filter criteria, don't use vector search for full table scan
                                        use_vector_search = False
                                        
                                except Exception as ve:
                                    logger.error(f"Vector search failed: {ve}")
                                    use_vector_search = False
                                
                                # Use vector results if available, otherwise fall back to direct access
                                if use_vector_search and vector_results:
                                    # Vector search was successful, use these results
                                    logger.info(f"Using vector search results for {table_name}")
                                    
                                    # Prepare data in expected format
                                    # Extract columns from first result
                                    columns = list(vector_results[0].keys()) if vector_results else []
                                    sheet_data = {
                                        "columns": columns,
                                        "rows": vector_results
                                    }
                                else:
                                    # Fall back to direct spreadsheet access only if vector search failed or returned no results
                                    logger.info(f"DIRECT SPREADSHEET: Falling back to direct search for {table_name}")
                                    start_time = time.time()
                                    
                                    # Read data from the specified sheet
                                    sheet_data = spreadsheet.read_sheet_data(table_name)
                                    
                                    # Extract search criteria from arguments
                                    if arguments_dict:
                                        filtered_rows = []
                                        for row in sheet_data.get('rows', []):
                                            match = True
                                            for key, value in arguments_dict.items():
                                                if key in row and str(row[key]) != str(value):
                                                    match = False
                                                    break
                                            if match:
                                                filtered_rows.append(row)
                                        
                                        sheet_data['rows'] = filtered_rows
                                    
                                    direct_time = time.time() - start_time
                                    logger.info(f"DIRECT SPREADSHEET: Retrieved {len(sheet_data.get('rows', []))} rows from {table_name} [{direct_time*1000:.2f}ms]")
                                    
                                    # Trigger background vectorization of this sheet data ONLY when fallback was used
                                    # This ensures we vectorize data that was missing from the vector database
                                    try:
                                        from app.services.vector_background_task import vectorize_spreadsheet_background
                                        vectorize_spreadsheet_background(
                                            self.openai_client.assistant_handler.assistant_id,
                                            spreadsheet_id,
                                            [table_name]
                                        )
                                        logger.info(f"Triggered background vectorization for {table_name} after fallback search")
                                    except Exception as ve:
                                        # Don't fail the main flow if vectorization fails
                                        logger.error(f"Background vectorization failed: {ve}")
                                result = {
                                    "status": "success",
                                    "data": sheet_data
                                }
                            except Exception as e:
                                print(f"ERROR reading from sheet: {e}")
                                logger.error(f"Error reading from sheet: {e}", exc_info=True)
                                result = {
                                    "status": "error",
                                    "message": f"Failed to read from {table_name}: {str(e)}"
                                }
                        elif operation_type == 'write':
                            # For write operations, add a record to the sheet
                            try:
                                logger.info(f"Writing data to sheet: {table_name}")
                                print(f"Writing data to sheet: {table_name}")

                                # Find the schema for this table
                                table_schema = None
                                
                                # Check the type of schemas to handle both formats
                                if isinstance(schemas, dict):
                                    # If schemas is a dictionary (from get_assistant_headers)
                                    # The key is the table name and the value is the list of headers
                                    if table_name in schemas:
                                        headers = schemas.get(table_name, [])
                                        # Create a schema dictionary with the headers
                                        table_schema = {
                                            "name": table_name,
                                            "columns": [{"name": header} for header in headers]
                                        }
                                        logger.debug(f"Found schema for table {table_name} using headers dictionary")
                                else:
                                    # Original behavior for list of schemas
                                    for schema in schemas:
                                        if isinstance(schema, dict) and schema.get('name', '').lower() == table_name.lower():
                                            table_schema = schema
                                            logger.debug(f"Found schema for table {table_name}")
                                            break

                                if not table_schema:
                                    raise Exception(f"Schema not found for table {table_name}")

                                # Process the arguments according to schema
                                schema_columns = table_schema.get('columns', [])
                                processed_data = match_columns(schema_columns, arguments_dict)

                                logger.info(f"Processed data for writing: {processed_data}")
                                print(f"Processed data for writing: {processed_data}")

                                # Append the data to the sheet
                                append_result = spreadsheet.append_sheet_data(table_name, [processed_data])
                                logger.info(f"Append result: {append_result}")
                                print(f"Append result: {append_result}")

                                # Make sure we set the result correctly before exiting the try block
                                result = {
                                    "status": "success",
                                    "message": f"Successfully added new record to {table_name}",
                                    "details": append_result
                                }
                                logger.info(f"Successfully wrote data to {table_name}: {append_result}")
                                print(f"Successfully wrote data to {table_name}: {append_result}")
                                
                            except Exception as e:
                                error_msg = f"Failed to write to {table_name}: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                print(f"ERROR writing to sheet: {e}")
                                result = {
                                    "status": "error", 
                                    "message": error_msg
                                }
                        
                        else:
                            print(f"ERROR: Unknown operation type {operation_type}")
                            result = {
                                "status": "error",
                                "message": f"Unknown operation type: {operation_type}"
                            }

                    if result:
                        print(f"Tool execution result: {result}")
                        # Send the result back to OpenAI using the call_id
                        await send_tool_result(self.openai_client, call_id, result)
                    else:
                        print("ERROR: Tool execution returned no result")
                        await send_tool_result(self.openai_client, call_id, {
                            "status": "error",
                            "message": "Tool execution failed with no result"
                        })
                        
                except ImportError as e:
                    print(f"ERROR importing tool modules: {e}")
                    await send_tool_result(self.openai_client, call_id, {
                        "status": "error",
                        "message": f"Import error: {str(e)}"
                    })
                except Exception as e:
                    print(f"ERROR executing tool: {e}")
                    print(traceback.format_exc())
                    await send_tool_result(self.openai_client, call_id, {
                        "status": "error",
                        "message": f"Execution error: {str(e)}"
                    })
            
            else:
                print(f"WARNING: Unhandled function call response type: {response.get('type')}")
                
        except Exception as e:
            print(f"ERROR in _handle_function_call: {e}")
            print(traceback.format_exc())
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