"""STATUS tool implementation."""

from typing import Dict, Any
from schemas import LoopState


async def status_tool(args: Dict[str, Any], state: LoopState) -> str:
    """
    STATUS tool: Return JSON snapshot of auction state.
    
    Args:
        args: {} (no parameters)
        state: Current loop state
    
    Returns:
        JSON string with auction status
    """
    auction = state["auction"]
    
    # Get auction status
    status = auction.get_state_summary()
    
    # Update tool usage
    state["tool_usage"]["STATUS"] = state["tool_usage"].get("STATUS", 0) + 1
    
    return f"AUCTION STATUS:\n{status}" 