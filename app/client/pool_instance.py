import asyncio
import logging
import os
from app.client.pool import OpenAIClientPool

# Configure logging
logger = logging.getLogger(__name__)

# Create a singleton instance of the client pool
client_pool = OpenAIClientPool(
    min_connections=int(os.getenv("MIN_CONNECTIONS", "2")),
    max_connections=int(os.getenv("MAX_CONNECTIONS", "5"))
)

# Set up a recurring health check task
health_check_task = None
health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # Default: 5 minutes

async def periodic_health_check():
    """Perform periodic health checks on the client pool."""
    try:
        while True:
            # Wait before first check to allow system to initialize
            await asyncio.sleep(health_check_interval)
            try:
                await client_pool.health_check()
            except Exception as e:
                logger.error(f"Error during health check: {e}")
    except asyncio.CancelledError:
        logger.info("Health check task cancelled")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in health check task: {e}")
        # Restart the task if unexpected error occurs
        schedule_health_check()

def schedule_health_check():
    """Schedule the periodic health check task."""
    global health_check_task
    
    # Cancel existing task if needed
    if health_check_task and not health_check_task.done():
        health_check_task.cancel()
    
    # Create a new health check task    
    health_check_task = asyncio.create_task(periodic_health_check())
    logger.info(f"Scheduled health check task (every {health_check_interval} seconds)")
    
    # Don't wait for task completion
    return health_check_task 