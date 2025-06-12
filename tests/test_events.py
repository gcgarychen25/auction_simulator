import asyncio


import pytest

from utils import load_config
import sys
import types

from schemas import Action, SellerResponse

agents_stub = types.ModuleType("agents")


class DummyParser:
    def get_format_instructions(self):
        return ""


agents_stub.action_parser = DummyParser()


def fake_buyer_runnable(persona):
    class R:
        async def ainvoke(self, _):
            return Action(action="bid", amount=10000, commentary="test")

    return R()


def fake_seller_runnable(config):
    class R:
        async def ainvoke(self, _):
            return SellerResponse(answer="ok", commentary="")

    return R()


agents_stub.create_buyer_agent_runnable = fake_buyer_runnable
agents_stub.create_seller_runnable = fake_seller_runnable
sys.modules["agents"] = agents_stub

langgraph = types.ModuleType("langgraph")
graph_mod = types.ModuleType("langgraph.graph")
message_mod = types.ModuleType("langgraph.graph.message")


def add_messages(x):
    return x


class DummyStateGraph:
    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, *a, **k):
        pass

    def add_conditional_edges(self, *a, **k):
        pass

    def add_edge(self, *a, **k):
        pass

    def set_entry_point(self, *a, **k):
        pass

    def compile(self):
        class C:
            async def ainvoke(self, inputs):
                return inputs

        return C()


graph_mod.StateGraph = DummyStateGraph
graph_mod.END = "END"
message_mod.add_messages = add_messages
graph_mod.message = message_mod
langgraph.graph = graph_mod

sys.modules["langgraph"] = langgraph
sys.modules["langgraph.graph"] = graph_mod
sys.modules["langgraph.graph.message"] = message_mod

import graph
from graph import AuctionState
from utils.event_bus import EventBus
from schemas import Event


def test_auction_end_event(monkeypatch):
    config = load_config("config.yaml")
    config["environment"]["auction"]["max_rounds"] = 1

    async def fake_run(cfg, live=False):
        state = AuctionState(config=cfg, current_price=0.0, active_buyers=[])
        bus = EventBus(state, live=False)
        await bus.log(Event(type="auction_end", actor="system", payload={}))
        return state

    monkeypatch.setattr(graph, "run_auction_episode", fake_run)

    state = asyncio.run(graph.run_auction_episode(config, live=False))
    assert any(e.type == "auction_end" for e in state.event_log)
