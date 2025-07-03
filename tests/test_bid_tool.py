"""Tests for the BID tool."""

import pytest
from schemas import AuctionState, LoopState
from tools.builtin.bid import bid_tool
from utils.event_bus import EventBus


@pytest.mark.asyncio
async def test_bid_tool_success():
    """Test successful bid placement."""
    # Setup
    auction = AuctionState(
        property_id="TEST_001",
        current_price=100000.0,
        active_buyers=["buyer"],
    )
    event_bus = EventBus()
    
    state: LoopState = {
        "context": [],
        "observation": None,
        "auction": auction,
        "tool_usage": {},
        "event_bus": event_bus
    }
    
    # Execute
    result = await bid_tool({"amount": 110000.0}, state)
    
    # Assert
    assert "SUCCESS" in result
    assert auction.current_price == 110000.0
    assert auction.leading_bidder == "buyer"
    assert auction.round_had_bid == True
    assert len(auction.history) == 1
    assert state["tool_usage"]["BID"] == 1
    assert len(event_bus.get_events("bid")) == 1


@pytest.mark.asyncio
async def test_bid_tool_too_low():
    """Test bid that's too low."""
    # Setup
    auction = AuctionState(
        property_id="TEST_001", 
        current_price=100000.0,
        active_buyers=["buyer"],
    )
    event_bus = EventBus()
    
    state: LoopState = {
        "context": [],
        "observation": None,
        "auction": auction,
        "tool_usage": {},
        "event_bus": event_bus
    }
    
    # Execute
    result = await bid_tool({"amount": 90000.0}, state)
    
    # Assert
    assert "ERROR" in result
    assert auction.current_price == 100000.0  # Unchanged
    assert auction.leading_bidder is None
    assert auction.round_had_bid == False


@pytest.mark.asyncio
async def test_bid_tool_missing_amount():
    """Test bid without amount parameter."""
    # Setup
    auction = AuctionState(
        property_id="TEST_001",
        current_price=100000.0,
        active_buyers=["buyer"],
    )
    event_bus = EventBus()
    
    state: LoopState = {
        "context": [],
        "observation": None,
        "auction": auction,
        "tool_usage": {},
        "event_bus": event_bus
    }
    
    # Execute
    result = await bid_tool({}, state)
    
    # Assert
    assert "ERROR" in result
    assert "requires 'amount' parameter" in result


@pytest.mark.asyncio
async def test_bid_tool_invalid_amount():
    """Test bid with invalid amount."""
    # Setup
    auction = AuctionState(
        property_id="TEST_001",
        current_price=100000.0,
        active_buyers=["buyer"],
    )
    event_bus = EventBus()
    
    state: LoopState = {
        "context": [],
        "observation": None,
        "auction": auction,
        "tool_usage": {},
        "event_bus": event_bus
    }
    
    # Execute
    result = await bid_tool({"amount": "invalid"}, state)
    
    # Assert
    assert "ERROR" in result
    assert "Invalid bid amount" in result 