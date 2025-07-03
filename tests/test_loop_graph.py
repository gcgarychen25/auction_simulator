"""Integration tests for the loop graph."""

import pytest
from unittest.mock import AsyncMock, patch
from loop_graph import parse_last_action, build_graph, run_episode
from schemas import Action


def test_parse_last_action_valid():
    """Test parsing valid action from context."""
    context = [
        "Starting auction...",
        'AGENT: {"tool": "BID", "args": {"amount": 100000}, "commentary": "Initial bid"}'
    ]
    
    action = parse_last_action(context)
    
    assert action.tool == "BID"
    assert action.args == {"amount": 100000}
    assert action.commentary == "Initial bid"


def test_parse_last_action_invalid_json():
    """Test parsing invalid JSON from context."""
    context = [
        "Starting auction...",
        'AGENT: {"tool": "BID", "args": invalid json}'
    ]
    
    action = parse_last_action(context)
    
    assert action.tool == "STATUS"  # Default fallback
    assert "Failed to parse" in action.commentary


def test_parse_last_action_empty_context():
    """Test parsing from empty context."""
    action = parse_last_action([])
    
    assert action.tool == "STATUS"
    assert action.commentary == "Starting auction"


def test_build_graph():
    """Test that graph builds successfully."""
    graph = build_graph()
    
    # Check that graph has the expected structure
    assert graph is not None
    # Further graph structure tests would go here


@pytest.mark.asyncio
async def test_run_episode_basic():
    """Test basic episode execution with mocked LLM."""
    config = {
        "property": {
            "property_id": "TEST_001",
            "starting_price": 100000
        },
        "auction": {
            "max_rounds": 5
        },
        "buyer_profile": {
            "budget": 200000
        }
    }
    
    # Mock the LLM response to immediately finish
    with patch('agents.BuyerAgent.invoke', new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = '{"tool": "FINISH", "args": {}, "commentary": "Test finish"}'
        
        results = await run_episode(config)
        
        assert results["success"] == True
        assert "auction_state" in results
        assert "tool_usage" in results
        assert "events" in results
        assert results["auction_state"].property_id == "TEST_001"


@pytest.mark.asyncio  
async def test_run_episode_with_bid():
    """Test episode with bid action."""
    config = {
        "property": {
            "property_id": "TEST_001", 
            "starting_price": 100000
        },
        "auction": {
            "max_rounds": 5
        },
        "buyer_profile": {
            "budget": 200000
        }
    }
    
    # Mock sequence: STATUS, BID, FINISH
    responses = [
        '{"tool": "STATUS", "args": {}, "commentary": "Check status"}',
        '{"tool": "BID", "args": {"amount": 110000}, "commentary": "Place bid"}',
        '{"tool": "FINISH", "args": {}, "commentary": "Done"}'
    ]
    
    with patch('agents.BuyerAgent.invoke', new_callable=AsyncMock) as mock_invoke:
        mock_invoke.side_effect = responses
        
        results = await run_episode(config)
        
        assert results["success"] == True
        assert results["tool_usage"].get("STATUS", 0) >= 1
        assert results["tool_usage"].get("BID", 0) >= 1
        assert results["auction_state"].current_price >= 100000 