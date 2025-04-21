# Voice Agent Package

This package contains the core functionality for a voice agent system that integrates Twilio with OpenAI's realtime API.

## Features
- Voice agent functionality with Twilio integration
- OpenAI client pool management for efficient API usage
- Database integration with Supabase
- Websocket handling for real-time communication
- Audio streaming support
- Client management with connection pooling
- Hardcoded tools for the agent
- Call handling (both inbound and outbound)

## Setup with Poetry

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
cd voice_agent_package
poetry install
```

3. Configure environment variables by copying and editing .env.example:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. Start the server:
```bash
poetry run start
```

Alternatively, you can use:
```bash
poetry run uvicorn voice_agent_package.main:app --host 0.0.0.0 --port 8000
```

## Testing the Setup

### 1. Make sure Ngrok is running (for Twilio to reach your local server)
```bash
ngrok http 8000
```
Update the WEBHOOK_URL in your .env file with the ngrok URL.

### 2. Test the health endpoint
```bash
curl http://localhost:8000/health
```
You should see: `{"status":"healthy","version":"1.0.0"}`

### 3. Test the voice agent status page
```bash
curl http://localhost:8000/voice-agent
```
This should return an HTML page confirming the service is running.

### 4. Make an outbound call via API
```bash
curl -X POST http://localhost:8000/make-call \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key_for_outbound_calls" \
  -d '{"to_number": "+1234567890"}'
```
Replace with the actual phone number to call and your API key.

### 5. Receiving Calls
Ensure your Twilio phone number's voice webhook is configured to point to your ngrok URL + "/incoming-call", e.g.:
```
https://xxxx-xxx-xxx-xxx.ngrok-free.app/incoming-call
```

When someone calls your Twilio number, it will connect them to the AI assistant.

## Usage

- The main endpoint for incoming Twilio calls is at `/incoming-call`
- To make outbound calls, use the `/make-call` endpoint
- WebSocket communication for audio streaming happens at `/media-stream/{assistant_id}` 