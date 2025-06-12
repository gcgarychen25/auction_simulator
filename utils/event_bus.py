import asyncio
import json
from schemas import Event

LIVE_LOG_FILE = "live_auction.log"

class EventBus:
    def __init__(self, live: bool = False):
        self.live = live

    async def log(self, event: Event, state=None):
        if state is not None and hasattr(state, 'event_log'):
            state.event_log.append(event)
        
        if self.live:
            with open(LIVE_LOG_FILE, 'a') as f:
                f.write(json.dumps(event.model_dump()) + '\n') 