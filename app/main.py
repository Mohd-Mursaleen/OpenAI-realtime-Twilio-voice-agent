from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import sys
from contextlib import asynccontextmanager
from app.router import router as agent_router
from app.client.client import OpenAIWebSocketClient
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get logger
logger = logging.getLogger(__name__)

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup events
#     try:
#         logger.info("Initializing OpenAI client...")
#         client = OpenAIWebSocketClient()
#         await client.connect()        
#         logger.info("Startup complete - server ready to accept requests")

#     except Exception as e:
#         logger.error(f"Error during startup: {e}")
#         # Re-raise to prevent server from starting if critical initialization fails
#         raise
    
#     yield
    
#     # Shutdown events
#     try:
#         logger.info("Shutting down Voice Agent API...")
#         # Close all connections
#         await client.close()
#         logger.info("OpenAI client connections closed")
#     except Exception as e:
#         logger.error(f"Error during shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Voice Agent API",
    description="Voice agent services for handling voice interactions using OpenAI",
    version="1.0.0",
    # lifespan=lifespan,
)
# Configure CORS - Simple configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include the Agent API routes
app.include_router(agent_router, tags=["agent"])

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

def main():
    """Main entry point for the application."""
    logger.info("Starting up Voice Agent server...")
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()