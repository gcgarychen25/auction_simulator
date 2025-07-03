
# ✨ Hybrid LangGraph ✕ MCP Refactor

> **Goal:** Replace the legacy phase-graph with a minimal two-node loop while future-proofing tool integration via the **Model Context Protocol** (MCP).

---

## 1 High-Level Architecture

```
┌───────────────────────────────┐
│        LangGraph Runtime      │
│                               │
│   ┌────────┐  Action  ┌──────┐│
│   │ Agent  │──────────│ Tool ││
│   │  LLM   │◄─────────│Node  ││
│   └────────┘Observation└──────┘│
└────────────────┬───────────────┘
                 ▼
          MCP Servers + Local Tools
```

* **Agent Node:** Single LLM prompt → returns `{tool, args, commentary}` or `FINISH`.
* **Tool Node:** Dispatches calls

  * **Local Tools** (Python functions) for latency-critical actions (`BID`, `CALL`, etc.).
  * **MCP client** for remote / shareable tools (`ASK_SELLER`, future APIs).

---

## 2 Directory / File Plan

| Path                            | Purpose                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `loop_graph.py`                 | Builds the two-node LangGraph (`agent`, `tool_dispatcher`).                                      |
| `agents.py`                     | **Unchanged** (buyer/seller LLM runnables).                                                      |
| `tools/__init__.py`             | Imports registry of all tools.                                                                   |
| `tools/builtin/`                | Pure-Python implementations:<br> `bid.py` `call.py` `fold.py` `status.py` `check_termination.py` |
| `tools/mcp_client.py`           | Generic MCP JSON-RPC wrapper: `invoke(tool_name, args)`.                                         |
| `schemas.py`                    | Add `Action` (`tool:str`, `args:dict`, `commentary:str`) and `LoopState`.                        |
| `config.yaml`                   | Per-tool backend flag:<br>`tools:`→`ASK_SELLER: mcp`, `BID: local`, …                            |
| `run.py`                        | Imports `loop_graph` instead of old `graph.py`.                                                  |
| `design/hybrid_mcp_refactor.md` | **← this doc** (commit for reference).                                                           |

---

## 3 Core Types

```python
# schemas.py
class Action(BaseModel):
    tool: Literal["BID", "CALL", "FOLD", "ASK_SELLER", "STATUS", "CHECK_TERMINATION", "FINISH"]
    args: Dict[str, Any] = {}
    commentary: str

class LoopState(TypedDict):
    context: List[str]           # transcript
    observation: Optional[str]
    auction: AuctionState        # ← reuse existing model
    tool_usage: Dict[str, int]
    event_bus: EventBus
```

---

## 4 Agent Prompt Template

```jinja
SYSTEM:
You are an autonomous auction participant. Use tools to achieve the best outcome.

Context:
{{context}}

Last observation:
{{observation}}

Available tools:
{% for t in tools %}• {{t.name}} – {{t.description}}
{% endfor %}

Respond ONLY with JSON:
{
 "tool": "<TOOL_NAME | FINISH>",
 "args": { ... },
 "commentary": "<short rationale>"
}
```

Parser: `PydanticOutputParser(Action)`.

---

## 5 Tool Dispatcher (Tool Node)

```python
async def tool_dispatcher(state: LoopState) -> LoopState:
    action: Action = parse_last_action(state.context)

    if action.tool == "FINISH":
        return state

    if is_local(action.tool):
        observation = await LOCAL_TOOLS[action.tool](action.args, state)
    else:
        observation = await invoke_mcp(action.tool, action.args)

    state.observation = observation
    state.context.append(f"TOOL:{action.tool} -> {observation}")
    return state
```

Edge router:

```python
def route(state: LoopState) -> str:
    return "end" if '"tool": "FINISH"' in state.context[-1] else "tool"
```

---

## 6 Local Tool Specs

| Tool                   | Args              | Behaviour                                                           |
| ---------------------- | ----------------- | ------------------------------------------------------------------- |
| **BID**                | `{amount: float}` | Validate > `auction.current_price`; update price; emit `bid` event. |
| **CALL**               | `{}`              | Keep bidder in; emit `call` event.                                  |
| **FOLD**               | `{}`              | Remove bidder; emit `fold` event.                                   |
| **STATUS**             | `{}`              | Return JSON snapshot of `auction`.                                  |
| **CHECK\_TERMINATION** | `{}`              | Return `"continue"` / `"end"` (phase limits, no bidders, etc.).     |

*All emit events via `state.event_bus`.*

---

## 7 MCP Client Helper

```python
# tools/mcp_client.py
import mcp  # assumes python-mcp SDK
client = mcp.Client()           # connect over unix socket / tcp

async def invoke(tool: str, args: dict) -> str:
    return await client.call(tool, args)    # returns JSON string/obj
```

---

## 8 Implementation Steps (Cursor Tasks)

1. **Create branch** `hybrid_mcp_refactor`.
2. Scaffold folders & empty files per §2.
3. Copy unchanged code: `agents.py`, `config_utils.py`, `event_bus.py`, `viz/`.
4. Implement `schemas.Action` + `LoopState`.
5. Build **local tool modules** in `tools/builtin/`.
6. Write `tools/mcp_client.py` (use `python-mcp`).
7. Implement `loop_graph.build_graph()` – two nodes, router, entry.
8. Update `run.py` to `from loop_graph import run_episode`.
9. Unit tests:

   * `pytest tests/test_bid_tool.py` etc.
   * integration: one toy auction, deterministic seed.
10. Add seller Q\&A MCP server (`servers/ask_seller_server.py`) – reuse seller prompt.
11. Update README with run instructions.

---

## 9 Future Extensions

* Swap in other MCP servers (e.g., `GET_TAX_RECORDS`, `MARKET_NEWS`).
* Summarisation or memory compression tool to keep `context` short.
* PPO trainer hooks via the dispatcher.

---

### ✅ Deliverables

* New branch `hybrid_mcp_refactor` containing all modules above.
* Passing unit + integration tests.
* Updated documentation & this design doc.
