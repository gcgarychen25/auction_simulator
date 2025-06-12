import asyncio
from typing import Optional
from schemas import Event, AuctionState

GLOBAL_QUEUE: asyncio.Queue = asyncio.Queue()


class EventBus:
    def __init__(self, state: AuctionState, live: bool = False):
        self.state = state
        self.live = live

    async def log(self, event: Event) -> None:
        self.state.event_log.append(event)
        if self.live:
            await GLOBAL_QUEUE.put(event.model_dump())
