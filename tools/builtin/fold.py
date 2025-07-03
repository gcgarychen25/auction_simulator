"""FOLD tool implementation."""

from typing import Dict, Any
from schemas import LoopState


async def fold_tool(args: Dict[str, Any], state: LoopState) -> str:
    """
    FOLD tool: Remove bidder from auction; emit fold event.
    
    Args:
        args: {} (no parameters)
        state: Current loop state
    
    Returns:
        Observation string
    """
    auction = state["auction"]
    
    # Check if buyer is still active
    if "buyer" not in auction.active_buyers:
        return "ERROR: Cannot fold - you are not an active participant in this auction"
    
    # Remove buyer from active participants
    auction.active_buyers = [b for b in auction.active_buyers if b != "buyer"]
    
    # Add to history
    history_entry = f"Round {auction.round}: FOLD (withdrew from auction at ${auction.current_price:,.2f})"
    auction.history.append(history_entry)
    
    # Emit fold event
    state["event_bus"].emit(
        "fold",
        "buyer",
        {
            "current_price": auction.current_price,
            "round": auction.round,
            "remaining_buyers": auction.active_buyers
        }
    )
    
    # Update tool usage
    state["tool_usage"]["FOLD"] = state["tool_usage"].get("FOLD", 0) + 1
    
    return f"SUCCESS: Folded - withdrew from auction at ${auction.current_price:,.2f}. Remaining buyers: {len(auction.active_buyers)}" 