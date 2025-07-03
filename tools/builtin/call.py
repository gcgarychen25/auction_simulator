"""CALL tool implementation."""

from typing import Dict, Any
from schemas import LoopState


async def call_tool(args: Dict[str, Any], state: LoopState) -> str:
    """
    CALL tool: Keep bidder in auction; emit call event.
    
    Args:
        args: {} (no parameters)
        state: Current loop state
    
    Returns:
        Observation string
    """
    auction = state["auction"]
    
    # Check if buyer is still active
    if "buyer" not in auction.active_buyers:
        return "ERROR: Cannot call - you are not an active participant in this auction"
    
    # Add to history
    history_entry = f"Round {auction.round}: CALL (staying in auction at ${auction.current_price:,.2f})"
    auction.history.append(history_entry)
    
    # Emit call event
    state["event_bus"].emit(
        "call",
        "buyer",
        {
            "current_price": auction.current_price,
            "round": auction.round,
            "leading_bidder": auction.leading_bidder
        }
    )
    
    # Update tool usage
    state["tool_usage"]["CALL"] = state["tool_usage"].get("CALL", 0) + 1
    
    return f"SUCCESS: Called - staying in auction at current price ${auction.current_price:,.2f}" 