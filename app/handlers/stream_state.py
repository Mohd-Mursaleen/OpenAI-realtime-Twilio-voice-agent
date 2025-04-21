# built-in imports
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class StreamState:
    """Manages the state of a media stream connection.
    
    This class maintains the current state of an audio stream session, tracking important
    timestamps, stream identifiers, and message queues. It provides functionality to
    manage and reset stream state information during a conversation.

    Attributes:
        stream_sid (Optional[str]): Unique identifier for the Twilio stream session
        latest_media_timestamp (int): Timestamp of the most recent media chunk received
        last_assistant_item (Optional[str]): ID of the last response from the assistant
        response_start_timestamp_twilio (Optional[int]): Timestamp when the assistant started responding
        mark_queue (List[str]): Queue to track response markers for synchronization
        media_count (int): Counter for tracking the number of media chunks processed
    """
    stream_sid: Optional[str] = None
    latest_media_timestamp: int = 0
    last_assistant_item: Optional[str] = None
    response_start_timestamp_twilio: Optional[int] = None
    mark_queue: List[str] = None
    media_count: int = 0
    last_interrupt_ts: Optional[int] = None


    def __post_init__(self):
        self.mark_queue = []

    def reset(self):
        """Reset stream state for new connection.
        
        Clears all state variables to their initial values, preparing the state
        for a new stream session.
        """
        self.stream_sid = None
        self.latest_media_timestamp = 0
        self.last_assistant_item = None
        self.response_start_timestamp_twilio = None
        self.mark_queue.clear()
        self.media_count = 0 