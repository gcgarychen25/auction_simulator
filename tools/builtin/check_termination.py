"""CHECK_TERMINATION tool implementation."""

from typing import Dict, Any
from schemas import LoopState


async def check_termination_tool(args: Dict[str, Any], state: LoopState) -> str:
    """
    CHECK_TERMINATION tool: Return "continue" / "end" based on auction conditions.
    
    Args:
        args: {} (no parameters)
        state: Current loop state
    
    Returns:
        "continue" or "end" with reason
    """
    auction = state["auction"]
    config = auction.config
    
    # Check max rounds limit
    max_rounds = config.get("max_rounds", 50)
    if auction.round >= max_rounds:
        auction.failure_reason = f"Maximum rounds ({max_rounds}) reached"
        return "end: Maximum rounds reached"
    
    # Check if no active buyers
    if len(auction.active_buyers) == 0:
        auction.failure_reason = "No active buyers remaining"
        return "end: No active buyers remaining"
    
    # Check if only one buyer remains
    if len(auction.active_buyers) == 1:
        auction.winner = auction.active_buyers[0]
        auction.final_price = auction.current_price
        return "end: Only one buyer remaining - auction won"
    
    # Check if no bid in current round (could be configurable)
    if not auction.round_had_bid and auction.round > 0:
        # For now, continue even without bids
        pass
    
    # Update tool usage
    state["tool_usage"]["CHECK_TERMINATION"] = state["tool_usage"].get("CHECK_TERMINATION", 0) + 1
    
    return "continue: Auction conditions allow continuation" 