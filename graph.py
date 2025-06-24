"""
Auction orchestration using LangGraph.
"""

import asyncio
import yaml
from typing import Dict, Any, List, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from schemas import AuctionState, Action, SellerResponse, Event, Participation
from agents import (
    create_buyer_agent_runnable,
    create_seller_runnable,
    action_parser,
    create_buyer_preference_runnable
)
from utils.event_bus import EventBus
import time

# --- Graph State ---

class GraphState(TypedDict):
    """
    Represents the state of our graph for a single auction.
    
    Attributes:
        state: The full auction state for one property.
        agent_runnables: A dictionary of all buyer agent runnables.
        seller_runnable: The seller agent runnable for the specific property.
        messages: A list of messages for communication between nodes.
        event_bus: The event bus for logging events.
    """
    state: AuctionState
    agent_runnables: Dict[str, Any]
    seller_runnable: Any
    messages: Annotated[list, add_messages]
    event_bus: EventBus


# --- Graph Nodes ---

async def _handle_buyer_question(
    buyer_id: str,
    state: AuctionState,
    agent_runnable: Any,
    seller_runnable: Any,
    event_bus: EventBus,
    action_parser: Any,
) -> Action:
    """Handles a single buyer's question, gets an answer, logs it, and returns the action."""
    persona = next(p for p in state.config['environment']['buyers'] if p['id'] == buyer_id)
    property_details = next(p for p in state.config['environment']['properties'] if p['id'] == state.property_id)['details']

    # Let the buyer agent decide whether to ask a question
    action = await agent_runnable.ainvoke({
        "persona_summary": f"ID: {persona['id']}, Max WTP: ${persona['max_wtp']:,}, Tendency: {persona['ask_tendency']}",
        "property_details": yaml.dump(property_details),
        "state_summary": state.get_state_summary(),
        "phase_instructions": "It is the Q&A phase. You can 'ask' a question or 'fold'.",
        "format_instructions": action_parser.get_format_instructions(),
    })

    if action.action == 'ask' and action.question:
        # Get the seller's answer before logging anything to ensure bundling
        response = await seller_runnable.ainvoke({"question": action.question})
        answer_text = response.answer if response else "Seller failed to provide an answer."
        
        # Log the question and answer together as a single atomic event
        await event_bus.log(Event(
            ts=time.time(), 
            type="qa_pair", 
            actor=buyer_id, 
            payload={"property_id": state.property_id, "question": action.question, "answer": answer_text}
        ), state=state)
        
        print(f"  - {buyer_id}: ASK ({action.commentary})\n     L> Q: {action.question}\n     L> A: {answer_text.strip()}")
        state.history.append(f"Q&A Round {state.round} on {state.property_id}: {buyer_id} asked '{action.question}' -> Answered.")

    return action


async def qa_phase_node(graph_state: GraphState) -> Dict[str, Any]:
    """Node for the Q&A phase where buyers can ask questions and have them answered in real-time."""
    state = graph_state['state']
    event_bus = graph_state['event_bus']
    seller_runnable = graph_state['seller_runnable']
    
    state.round += 1
    print(f"\n\n--- Round {state.round} ({state.property_id}) ---")
    print("--- Phase: Q&A (Parallel) ---")

    buyer_tasks = [
        _handle_buyer_question(
            buyer_id,
            state,
            graph_state['agent_runnables'][buyer_id],
            seller_runnable,
            event_bus,
            action_parser,
        )
        for buyer_id in state.active_buyers
    ]

    try:
        buyer_actions_list = await asyncio.gather(*buyer_tasks)
        buyer_actions = dict(zip(state.active_buyers, buyer_actions_list))
    except Exception as e:
        print(f"🚨 Error during Q&A phase on {state.property_id}: {e}")
        buyer_actions = {b_id: Action(action="fold", commentary=f"Error: {e}") for b_id in state.active_buyers}

    folded_buyers = {buyer_id for buyer_id, action in buyer_actions.items() if action.action == 'fold'}
    if folded_buyers:
        print(f"  Folding in Q&A on {state.property_id}: {', '.join(folded_buyers)}")
        state.active_buyers = [b for b in state.active_buyers if b not in folded_buyers]
    
    return {"state": state}


async def bidding_phase_node(graph_state: GraphState) -> Dict[str, Any]:
    """Node for the bidding phase."""
    state = graph_state['state']
    event_bus = graph_state['event_bus']
    print(f"\n--- Phase: Bidding ({state.property_id}) ---")
    bidding_instructions = (
        "It is the Bidding phase. You can 'bid' to raise the price, "
        "'call' to match the current price and stay in, or 'fold' to exit the auction. "
        "A 'bid' amount must be higher than the current price."
    )
    
    tasks = []
    property_details = next(p for p in state.config['environment']['properties'] if p['id'] == state.property_id)['details']
    for buyer_id in state.active_buyers:
        persona = next(b for b in state.config['environment']['buyers'] if b['id'] == buyer_id)
        runnable = graph_state['agent_runnables'][buyer_id]
        task = runnable.ainvoke({
             "persona_summary": f"ID: {persona['id']}, Max WTP: ${persona['max_wtp']:,}, ...",
             "property_details": yaml.dump(property_details),
             "state_summary": state.get_state_summary(),
             "phase_instructions": bidding_instructions,
             "format_instructions": action_parser.get_format_instructions(),
        })
        tasks.append(task)
    
    try:
        bidding_actions_list = await asyncio.gather(*tasks)
        bidding_actions = dict(zip(state.active_buyers, bidding_actions_list))
    except Exception as e:
        print(f"🚨 Error during bidding on {state.property_id}: {e}")
        bidding_actions = {b_id: Action(action="fold", commentary=f"Error: {e}") for b_id in state.active_buyers}

    # Log and print all actions for visualization and terminal output
    print(f"  --- Bidding Actions ({state.property_id}) ---")
    for buyer_id, action in bidding_actions.items():
        event_type = action.action
        log_payload = {"property_id": state.property_id, "commentary": action.commentary}
        
        commentary_str = f"({action.commentary})"

        if event_type == 'bid':
            amount_str = f"${action.amount:,.2f}" if action.amount is not None else "an invalid amount"
            print(f"  - {buyer_id}: BIDS {amount_str} {commentary_str}")
            log_payload["amount"] = action.amount
        elif event_type == 'fold':
            print(f"  - {buyer_id}: FOLDS {commentary_str}")
        elif event_type == 'call':
            print(f"  - {buyer_id}: CALLS {commentary_str}")
        
        if event_type in ["bid", "fold", "call"]:
             await event_bus.log(Event(ts=time.time(), type=event_type, actor=buyer_id, payload=log_payload), state=state)

    # Reset bid flag for the round
    state.round_had_bid = False
    
    # Process bids and update state
    valid_bids = {b: a for b, a in bidding_actions.items() if a.action == 'bid' and a.amount is not None and a.amount > state.current_price}
    
    if valid_bids:
        state.round_had_bid = True
        highest_bidder = max(valid_bids, key=lambda k: valid_bids[k].amount)
        state.leading_bidder = highest_bidder
        state.current_price = valid_bids[highest_bidder].amount
        log_msg = f"Round {state.round} Bidding on {state.property_id}: New high bid of ${state.current_price:,.2f} from {state.leading_bidder}."
        print(f"\n{log_msg}")
        state.history.append(log_msg)

    # Process folds - determine who is still active
    folded_buyers = {b_id for b_id, action in bidding_actions.items() if action.action == 'fold'}
    state.active_buyers = [b_id for b_id in state.active_buyers if b_id not in folded_buyers]
    
    return {"state": state}

# --- Conditional Edges ---

def should_continue(graph_state: GraphState) -> str:
    """Determine whether to continue the auction or end."""
    state = graph_state['state']
    if len(state.active_buyers) <= 1:
        print(f"\n--- Condition Met ({state.property_id}): Auction ending due to lack of bidders. ---")
        return "end"
    if not state.round_had_bid and state.round > 0:
        print(f"\n--- Condition Met ({state.property_id}): Auction ending because bidding has stabilized. ---")
        return "end"
    if state.round >= state.config['environment']['auction']['max_rounds']:
        print(f"\n--- Condition Met ({state.property_id}): Auction ending due to reaching max rounds. ---")
        return "end"
        
    return "continue"

def finalize_auction_node(graph_state: GraphState) -> Dict[str, Any]:
    """Finalizes the auction, determining winner or failure reason."""
    state = graph_state['state']
    
    # Find the property to get the seller's reserve price factor
    prop_config = next(p for p in state.config['environment']['properties'] if p['id'] == state.property_id)
    # Use the explicit estimated market value from the config
    estimated_market_value = prop_config['estimated_market_value']
    reserve_price = estimated_market_value * prop_config['seller']['reserve_price_factor']

    if state.leading_bidder and state.current_price >= reserve_price:
        state.winner = state.leading_bidder
        state.final_price = state.current_price
        state.history.append(f"Conclusion for {state.property_id}: SOLD to {state.winner} for ${state.final_price:,.2f}.")
    else:
        reason = "Reserve price not met." if state.leading_bidder else "No valid bids were placed."
        state.failure_reason = f"Failed to sell. {reason}"
        state.history.append(f"Conclusion for {state.property_id}: FAILED. {state.failure_reason}")
    
    print(f"--- AUCTION ENDED ({state.property_id}): {state.history[-1]} ---")
    return {"state": state}


# --- Graph Definition ---

def build_graph():
    """Builds the LangGraph for a single property auction."""
    workflow = StateGraph(GraphState)

    workflow.add_node("qa_phase", qa_phase_node)
    workflow.add_node("bidding_phase", bidding_phase_node)
    workflow.add_node("finalize_auction", finalize_auction_node)

    workflow.add_conditional_edges(
        "bidding_phase",
        should_continue,
        {
            "continue": "qa_phase",
            "end": "finalize_auction",
        },
    )
    workflow.add_edge("qa_phase", "bidding_phase")
    workflow.add_edge("finalize_auction", END)
    
    workflow.set_entry_point("qa_phase")
    
    return workflow.compile()


# --- Main Orchestration Function ---

async def run_auction_episode(config: Dict[str, Any], live: bool = False) -> List[AuctionState]:
    """
    Runs a full market episode with multiple auctions, one for each property.
    1. Buyers reflect and choose which auctions to join.
    2. Auctions are run sequentially for each property.
    """
    event_bus = EventBus(live=live)
    env_config = config['environment']
    
    # --- 1. Buyer Reflection and Participation Choice ---
    print("--- 📢 Market Announcement & Buyer Reflection 📢 ---")
    properties_summary = yaml.dump([
        {"id": p['id'], **p['details']} for p in env_config['properties']
    ])
    
    preference_tasks = []
    for buyer_persona in env_config['buyers']:
        runnable = create_buyer_preference_runnable(buyer_persona)
        task = runnable.ainvoke({"properties_summary": properties_summary})
        preference_tasks.append(task)
        
    participation_results: List[Participation] = await asyncio.gather(*preference_tasks)
    
    participation_map = {
        buyer['id']: result.auctions_to_join
        for buyer, result in zip(env_config['buyers'], participation_results)
    }

    print("\n--- 📝 Buyer Participation Intentions ---")
    for buyer_id, property_ids in participation_map.items():
        print(f"  - {buyer_id} will join auctions for: {', '.join(property_ids) or 'None'}")
    
    # --- 2. Run Auctions Sequentially for Each Property ---
    final_states = []
    graph = build_graph()

    for prop in env_config['properties']:
        property_id = prop['id']
        print(f"\n\n--- 🚀 Starting Auction for Property: {property_id} 🚀 ---")

        # Determine active buyers for this specific auction
        active_buyers_for_this_auction = [
            b_id for b_id, p_ids in participation_map.items() if property_id in p_ids
        ]
        
        if not active_buyers_for_this_auction:
            print(f"--- ⏩ Skipping auction for {property_id}: No buyers interested. ---")
            # Create a dummy final state for reporting
            final_states.append(AuctionState(
                config=config,
                property_id=property_id,
                current_price=env_config['auction']['start_price'],
                active_buyers=[],
                failure_reason="No buyers participated."
            ))
            continue

        # Initial state setup for this specific auction
        initial_state = AuctionState(
            config=config,
            property_id=property_id,
            current_price=env_config['auction']['start_price'],
            active_buyers=active_buyers_for_this_auction,
        )
        
        # Agent setup for this auction
        buyer_runnables = {
            buyer['id']: create_buyer_agent_runnable(buyer)
            for buyer in env_config['buyers'] if buyer['id'] in active_buyers_for_this_auction
        }
        seller_runnable = create_seller_runnable(prop)
        
        # Initial inputs for the graph
        inputs = {
            "state": initial_state,
            "agent_runnables": buyer_runnables,
            "seller_runnable": seller_runnable,
            "messages": [],
            "event_bus": event_bus,
        }
        
        await event_bus.log(Event(ts=time.time(), type="auction_start", actor="system", payload={"config": config, "property_id": property_id}), state=initial_state)
        
        final_graph_state = await graph.ainvoke(inputs)
        final_state = final_graph_state['state']
        final_states.append(final_state)
        
        await event_bus.log(Event(ts=time.time(), type="auction_end", actor="system", payload={"property_id": property_id, "winner": final_state.winner, "final_price": final_state.final_price, "failure_reason": final_state.failure_reason}), state=final_state)
        
    return final_states

# TODO: Instrument bid, ask, fold, and chat events throughout the graph nodes for full live streaming support. 