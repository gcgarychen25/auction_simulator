"""BID tool implementation."""

from typing import Dict, Any
from schemas import LoopState


async def bid_tool(args: Dict[str, Any], state: LoopState) -> str:
    """
    BID tool: Validate bid amount > current_price; update price; emit bid event.
    
    Args:
        args: {"amount": float}
        state: Current loop state
    
    Returns:
        Observation string
    """
    amount = args.get("amount")
    if not amount:
        return "ERROR: BID requires 'amount' parameter"
    
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return f"ERROR: Invalid bid amount '{amount}'. Must be a number."
    
    auction = state["auction"]
    
    # Validate bid amount
    if amount <= auction.current_price:
        return f"ERROR: Bid amount ${amount:,.2f} must be greater than current price ${auction.current_price:,.2f}"
    
    # Update auction state
    previous_price = auction.current_price
    previous_leader = auction.leading_bidder
    
    auction.current_price = amount
    auction.leading_bidder = "buyer"  # Assuming single buyer for now
    auction.round_had_bid = True
    
    # Add to history
    history_entry = f"Round {auction.round}: BID ${amount:,.2f} (was ${previous_price:,.2f})"
    auction.history.append(history_entry)
    
    # Emit bid event
    state["event_bus"].emit(
        "bid",
        "buyer",
        {
            "amount": amount,
            "previous_price": previous_price,
            "previous_leader": previous_leader,
            "round": auction.round
        }
    )
    
    # Update tool usage
    state["tool_usage"]["BID"] = state["tool_usage"].get("BID", 0) + 1
    
    return f"SUCCESS: Bid placed for ${amount:,.2f}. You are now the leading bidder." 