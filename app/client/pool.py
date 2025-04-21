import asyncio
from typing import List, Optional
import logging
from app.client.client import OpenAIWebSocketClient

logger = logging.getLogger(__name__)

class OpenAIClientPool:
    def __init__(self, min_connections=2, max_connections=5):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.available_clients: List[OpenAIWebSocketClient] = []
        self.in_use_clients = set()
        self.initialization_lock = asyncio.Lock()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the minimum number of connections at startup"""
        async with self.initialization_lock:
            if self.initialized:
                return
                
            logger.info(f"Pre-initializing {self.min_connections} OpenAI clients")
            successful_connections = 0
            
            # Try to create the minimum number of connections
            for _ in range(self.min_connections):
                try:
                    client = OpenAIWebSocketClient()
                    await client.connect()
                    
                    # Verify the connection is actually established
                    if client.connected:
                        self.available_clients.append(client)
                        successful_connections += 1
                    else:
                        logger.warning("Failed to establish connection for a client during initialization")
                except Exception as e:
                    logger.error(f"Error initializing client: {e}")
            
            # Only mark as initialized if we have at least one working connection
            if successful_connections > 0:
                self.initialized = True
                logger.info(f"Successfully pre-initialized {successful_connections} OpenAI clients")
            else:
                logger.error("Failed to initialize any OpenAI clients")
                # We'll still mark as initialized to avoid repeated failed attempts
                self.initialized = True
    
    async def get_client(self) -> Optional[OpenAIWebSocketClient]:
        """Get an available client from the pool or create a new one if needed"""
        if not self.initialized:
            await self.initialize()
            
        client = None
        
        # Try to get a client from the available pool
        while self.available_clients:
            client = self.available_clients.pop()
            
            # Verify the connection is still valid
            if client.connected:
                logger.info("Using pre-initialized OpenAI client")
                break
            else:
                # Connection is no longer valid, try to reconnect
                logger.warning("Found disconnected client in pool, reconnecting...")
                try:
                    await client.connect()
                    # If reconnect successful, use this client
                    if client.connected:
                        logger.info("Successfully reconnected client")
                        break
                except Exception as e:
                    logger.error(f"Failed to reconnect client: {e}")
                    # Discard this client
                    client = None
                    continue
        
        # If no valid client found in pool, create a new one if under capacity
        if client is None and len(self.in_use_clients) < self.max_connections:
            logger.info("Creating new client as no valid clients in pool")
            try:
                client = OpenAIWebSocketClient()
                await client.connect()
                if not client.connected:
                    logger.error("Failed to establish connection for new client")
                    return None
            except Exception as e:
                logger.error(f"Error creating new client: {e}")
                return None
        elif client is None:
            logger.warning("No available clients and at max capacity")
            return None
            
        if client:
            self.in_use_clients.add(client)
            
        return client
    
    async def release_client(self, client: OpenAIWebSocketClient):
        """Return a client to the available pool"""
        if client in self.in_use_clients:
            self.in_use_clients.remove(client)
            # Reset client state but keep connection open
            client.reset_state()
            self.available_clients.append(client)
            
    async def close_all(self):
        """Close all connections when shutting down"""
        for client in self.available_clients + list(self.in_use_clients):
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing client: {e}")
    
    async def health_check(self):
        """Perform a health check on all clients and reconnect as needed"""
        logger.info("Starting client pool health check")
        
        # Check available clients
        healthy_clients = []
        for client in self.available_clients:
            if client.connected:
                healthy_clients.append(client)
            else:
                # Try to reconnect
                try:
                    logger.warning("Reconnecting disconnected client during health check")
                    await client.connect()
                    if client.connected:
                        healthy_clients.append(client)
                except Exception as e:
                    logger.error(f"Failed to reconnect client during health check: {e}")
        
        # Replace available_clients with only healthy ones
        self.available_clients = healthy_clients
        
        # Create additional clients if needed to maintain minimum connections
        needed_clients = max(0, self.min_connections - len(self.available_clients) - len(self.in_use_clients))
        if needed_clients > 0:
            logger.info(f"Creating {needed_clients} additional clients to maintain minimum pool size")
            for _ in range(needed_clients):
                try:
                    client = OpenAIWebSocketClient()
                    await client.connect()
                    if client.connected:
                        self.available_clients.append(client)
                except Exception as e:
                    logger.error(f"Failed to create additional client during health check: {e}")
        
        logger.info(f"Health check complete. Pool status: {len(self.available_clients)} available, {len(self.in_use_clients)} in use") 