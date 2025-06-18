"""
Pydantic schemas for the auction simulator.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field

# --- Action and Response Schemas ---

class Action(BaseModel):
    """A buyer's action, which can be asking a question, bidding, or folding."""
    action: Literal["ask", "bid", "call", "fold"]
    property_id: Optional[str] = Field(None, description="The ID of the property this action applies to.")
    amount: Optional[float] = Field(None, description="The amount to bid. Required for 'bid' action.")
    question: Optional[str] = Field(None, description="The question to ask. Required for 'ask' action.")
    commentary: str = Field(..., description="The reasoning behind the action.")

class SellerResponse(BaseModel):
    """A structured response from the seller to a buyer's question."""
    answer: str = Field(description="The seller's direct answer to the question.")
    commentary: str = Field(description="Brief commentary on the seller's thinking.")

class Participation(BaseModel):
    """A buyer's decision on which auctions to join, with commentary."""
    auctions_to_join: List[str] = Field(..., description="A list of property IDs the buyer will participate in.")
    commentary: str = Field(..., description="The reasoning behind the participation decision.")

# --- State Management Schema ---

class AuctionState(BaseModel):
    """Manages the full state of a single auction for one property."""
    config: Dict[str, Any]
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
    event_log: List['Event'] = Field(default_factory=list)  # Stores all events for live streaming and analytics

    # Pydantic models are immutable by default, so we need to allow mutation
    class Config:
        arbitrary_types_allowed = True

    def get_state_summary(self) -> str:
        """Creates a concise summary of the auction state for prompts."""
        # Find the property details from the config
        property_details = "Unknown Property"
        for prop in self.config['environment']['properties']:
            if prop['id'] == self.property_id:
                property_details = prop['details']['address']
                break

        summary = [
            f"This is Round {self.round} of a real estate auction for the property at {property_details}.",
            f"Current Price: ${self.current_price:,.2f}",
            f"Leading Bidder: {self.leading_bidder or 'None'}",
            f"Active Buyers ({len(self.active_buyers)}): {', '.join(self.active_buyers)}",
            "---",
            "Recent History:",
        ]
        summary.extend(f"- {h}" for h in self.history[-5:])
        return "\n".join(summary) 

class Event(BaseModel):
    """Represents a single event in the auction for logging and live streaming."""
    ts: float
    type: str
    actor: str
    payload: dict 