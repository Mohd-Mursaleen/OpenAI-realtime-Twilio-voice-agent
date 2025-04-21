from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import sys
from contextlib import asynccontextmanager
from app.client.pool_instance import client_pool, schedule_health_check
from app.db.database import initialize_supabase
from app.router import router as agent_router

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    try:
        logger.info("Starting up Voice Agent API...")
        initialize_supabase()
        
        # Pre-initialize the client pool - await it to ensure completion
        logger.info("Initializing OpenAI client pool...")
        await client_pool.initialize()
        logger.info("OpenAI client pool initialized successfully")
        
        # Start the health check task
        logger.info("Starting health check background task...")
        schedule_health_check()
        logger.info("Health check task scheduled")
        
        logger.info("Startup complete - server ready to accept requests")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Re-raise to prevent server from starting if critical initialization fails
        raise
    
    yield
    
    # Shutdown events
    try:
        logger.info("Shutting down Voice Agent API...")
        # Close all connections
        await client_pool.close_all()
        logger.info("All OpenAI client connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Voice Agent API",
    description="Voice agent services for handling voice interactions using OpenAI and Twilio",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

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