import asyncio
import json
import websockets
from utils.event_bus import GLOBAL_QUEUE

CLIENTS = set()


async def producer():
    while True:
        event = await GLOBAL_QUEUE.get()
        for ws in CLIENTS.copy():
            try:
                await ws.send(json.dumps(event))
            except Exception:
                CLIENTS.discard(ws)


async def handler(ws):
    CLIENTS.add(ws)
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await producer()


if __name__ == "__main__":
    asyncio.run(main())
