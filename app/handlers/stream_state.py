# built-in imports
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class StreamState:
    """Manages the state of an audio stream connection.
    
    This class maintains the current state of an audio stream session, tracking important
    timestamps and conversation state for managing the interaction between user and assistant.

    Attributes:
        latest_timestamp (int): Timestamp of the most recent event
        last_assistant_item (Optional[str]): ID of the last response from the assistant
        response_start_time (Optional[int]): Timestamp when the assistant started responding
        is_user_speaking (bool): Flag indicating if the user is currently speaking
        is_assistant_speaking (bool): Flag indicating if the assistant is currently speaking
        media_count (int): Counter for tracking the number of media chunks processed
    """
    latest_timestamp: int = 0
    last_assistant_item: Optional[str] = None
    response_start_time: Optional[int] = None
    is_user_speaking: bool = False
    is_assistant_speaking: bool = False
    media_count: int = 0

    def get_current_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)

    def reset(self):
        """Reset stream state for new connection.
        
        Clears all state variables to their initial values, preparing the state
        for a new stream session.
        """
        self.latest_timestamp = 0
        self.last_assistant_item = None
        self.response_start_time = None
        self.is_user_speaking = False
        self.is_assistant_speaking = False
        self.media_count = 0

    def start_user_speaking(self):
        """Mark that the user has started speaking."""
        self.is_user_speaking = True
        self.latest_timestamp = self.get_current_timestamp()

    def stop_user_speaking(self):
        """Mark that the user has stopped speaking."""
        self.is_user_speaking = False
        self.latest_timestamp = self.get_current_timestamp()

    def start_assistant_speaking(self):
        """Mark that the assistant has started speaking."""
        self.is_assistant_speaking = True
        self.response_start_time = self.get_current_timestamp()

    def stop_assistant_speaking(self):
        """Mark that the assistant has stopped speaking."""
        self.is_assistant_speaking = False
        self.response_start_time = None