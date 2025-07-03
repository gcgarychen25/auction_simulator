"""
LangGraph two-node loop implementation for hybrid MCP architecture.
"""

import json
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain.output_parsers import PydanticOutputParser

from schemas import LoopState, Action, AuctionState
from agents import BuyerAgent
from tools import LOCAL_TOOLS, is_local_tool, is_mcp_tool
from tools.mcp_client import invoke_mcp
from utils import EventBus


def parse_last_action(context: list) -> Action:
    """Parse the last agent response as an Action."""
    if not context:
        return Action(tool="STATUS", args={}, commentary="Starting auction")
    
    last_message = context[-1]
    if not last_message.startswith("AGENT:"):
        return Action(tool="STATUS", args={}, commentary="No agent response found")
    
    # Extract JSON from agent response
    json_str = last_message.replace("AGENT:", "").strip()
    
    try:
        data = json.loads(json_str)
        return Action(**data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse action: {e}")
        return Action(tool="STATUS", args={}, commentary="Failed to parse previous action")


async def agent_node(state: LoopState) -> LoopState:
    """
    Agent Node: Single LLM prompt → returns {tool, args, commentary} or FINISH.
    """
    # Create agent (in real implementation, this would be cached)
    agent = BuyerAgent()
    
    # Get agent response
    response = await agent.invoke(
        context=state["context"],
        observation=state.get("observation")
    )
    
    # Add agent response to context
    state["context"].append(f"AGENT: {response}")
    
    return state


async def tool_dispatcher(state: LoopState) -> LoopState:
    """
    Tool Node: Dispatches calls to local tools or MCP client.
    """
    # Parse the last action from context
    action = parse_last_action(state["context"])
    
    if action.tool == "FINISH":
        state["observation"] = "Agent finished participation"
        return state
    
    try:
        # Dispatch to appropriate tool
        if is_local_tool(action.tool):
            observation = await LOCAL_TOOLS[action.tool](action.args, state)
        elif is_mcp_tool(action.tool):
            observation = await invoke_mcp(action.tool, action.args)
        else:
            observation = f"ERROR: Unknown tool '{action.tool}'"
        
        # Update state
        state["observation"] = observation
        state["context"].append(f"TOOL:{action.tool} -> {observation}")
        
    except Exception as e:
        error_msg = f"ERROR: Tool execution failed: {str(e)}"
        state["observation"] = error_msg
        state["context"].append(f"TOOL:{action.tool} -> {error_msg}")
    
    return state


def route_after_agent(state: LoopState) -> str:
    """Route after agent node."""
    # Parse last action to check if it's FINISH
    if state["context"]:
        last_message = state["context"][-1]
        if "FINISH" in last_message:
            return END
    return "tool"


def route_after_tool(state: LoopState) -> str:
    """Route after tool node."""
    # Check termination conditions
    if state.get("observation", "").startswith("end:"):
        return END
    return "agent"


def build_graph() -> StateGraph:
    """
    Build the two-node LangGraph.
    
    Returns:
        Configured StateGraph
    """
    # Create the graph
    workflow = StateGraph(LoopState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_dispatcher)
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool": "tool",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "tool", 
        route_after_tool,
        {
            "agent": "agent",
            END: END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    return workflow.compile()


async def run_episode(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single auction episode.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Episode results
    """
    # Initialize auction state
    property_info = config["property"]
    auction = AuctionState(
        property_id=property_info["property_id"],
        current_price=property_info["starting_price"],
        active_buyers=["buyer"],  # Single buyer for now
        config=config.get("auction", {})
    )
    
    # Initialize loop state
    event_bus = EventBus()
    initial_state: LoopState = {
        "context": [f"Starting auction for {property_info['property_id']} at ${property_info['starting_price']:,.2f}"],
        "observation": None,
        "auction": auction,
        "tool_usage": {},
        "event_bus": event_bus
    }
    
    # Build and run graph
    graph = build_graph()
    
    try:
        # Run the graph
        final_state = await graph.ainvoke(initial_state)
        
        # Prepare results
        results = {
            "auction_state": final_state["auction"],
            "tool_usage": final_state["tool_usage"], 
            "events": event_bus.get_events(),
            "context": final_state["context"],
            "success": True
        }
        
        return results
        
    except Exception as e:
        return {
            "auction_state": auction,
            "tool_usage": initial_state["tool_usage"],
            "events": event_bus.get_events(),
            "context": initial_state["context"],
            "success": False,
            "error": str(e)
        } 