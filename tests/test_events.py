import pytest
import asyncio
from graph import run_auction_episode
from schemas import Event

def minimal_config():
    # Placeholder: Provide a minimal config dict for a 1-round auction
    return {
        'environment': {
            'auction': {'start_price': 100, 'max_rounds': 1, 'bid_limit_per_buyer': 1},
            'seller': {'reserve_price': 50},
            'property': {'address': 'Test', 'bedrooms': 1, 'bathrooms': 1, 'size_sqft': 1, 'description': 'Test'},
            'buyers': [
                {'id': 'B1', 'max_wtp': 200, 'risk_aversion': 0.5, 'ask_prob': 0.0, 'requirements': 'Test'}
            ]
        }
    }

@pytest.mark.asyncio
async def test_auction_end_event():
    config = minimal_config()
    state = await run_auction_episode(config, live=False)
    # event_log is attached to state
    event_types = [e.type for e in getattr(state, 'event_log', [])]
    assert 'auction_end' in event_types, 'auction_end event not found in event_log' 