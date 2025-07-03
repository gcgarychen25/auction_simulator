"""
Pydantic schemas for the hybrid LangGraph + MCP auction simulator.
"""

from typing import List, Dict, Any, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
import json


class Action(BaseModel):
    """Action returned by the agent LLM."""
    tool: Literal["BID", "CALL", "FOLD", "ASK_SELLER", "STATUS", "CHECK_TERMINATION", "FINISH"]
    args: Dict[str, Any] = Field(default_factory=dict)
    commentary: str = Field(..., description="Short rationale for the action")


class AuctionState(BaseModel):
    """Manages the state of a single auction."""
    property_id: str
    round: int = 0
    current_price: float
    leading_bidder: Optional[str] = None
    active_buyers: List[str]
    round_had_bid: bool = False
    history: List[str] = Field(default_factory=list)
    winner: Optional[str] = None
    final_price: Optional[float] = None
    failure_reason: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def get_state_summary(self) -> str:
        """Creates a concise JSON summary of the auction state for prompts."""
        state_dict = {
            "property_id": self.property_id,
            "round": self.round,
            "current_price": self.current_price,
            "leading_bidder": self.leading_bidder,
            "active_buyers": self.active_buyers,
            "recent_history": self.history[-5:] if self.history else [],
        }
        return json.dumps(state_dict, indent=2)


class LoopState(TypedDict):
    """State for the LangGraph two-node loop."""
    context: List[str]           # transcript
    observation: Optional[str]
    auction: AuctionState        # auction state
    tool_usage: Dict[str, int]
    event_bus: Any               # EventBus instance


class Event(BaseModel):
    """Event for logging and live streaming."""
    ts: float
    type: str
    actor: str
    payload: Dict[str, Any]


class PropertyInfo(BaseModel):
    """Property information for auction setup."""
    property_id: str
    starting_price: float
    description: str
    location: str
    bedrooms: int
    bathrooms: int
    sqft: int
    lot_size: float
    year_built: int
    seller_notes: str = ""


class BuyerProfile(BaseModel):
    """Buyer profile for agent configuration."""
    name: str
    budget: float
    preferences: Dict[str, Any]
    strategy: str = "balanced" 