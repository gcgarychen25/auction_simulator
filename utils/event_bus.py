"""
Event bus for auction event handling and logging.
"""

import time
from typing import List, Dict, Any, Callable
from schemas import Event


class EventBus:
    """Simple event bus for auction events."""
    
    def __init__(self):
        self.events: List[Event] = []
        self.listeners: Dict[str, List[Callable]] = {}
    
    def emit(self, event_type: str, actor: str, payload: Dict[str, Any]):
        """Emit an event."""
        event = Event(
            ts=time.time(),
            type=event_type,
            actor=actor,
            payload=payload
        )
        self.events.append(event)
        
        # Notify listeners
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                try:
                    listener(event)
                except Exception as e:
                    print(f"Error in event listener: {e}")
    
    def subscribe(self, event_type: str, listener: Callable):
        """Subscribe to an event type."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)
    
    def get_events(self, event_type: str = None) -> List[Event]:
        """Get events, optionally filtered by type."""
        if event_type is None:
            return self.events
        return [e for e in self.events if e.type == event_type]
    
    def clear(self):
        """Clear all events."""
        self.events.clear() 