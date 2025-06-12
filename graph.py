"""
Auction orchestration using LangGraph.
"""

import asyncio
from typing import Dict, Any, List, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from schemas import AuctionState, Action, SellerResponse, Event
from agents import create_buyer_agent_runnable, create_seller_runnable, action_parser
from utils.event_bus import EventBus
import time

# --- Graph State ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        state: The full auction state.
        agent_runnables: A dictionary of all buyer agent runnables.
        seller_runnable: The seller agent runnable.
        messages: A list of messages for communication between nodes (not used here).
        event_bus: The event bus for logging events.
    """
    state: AuctionState
    agent_runnables: Dict[str, Any]
    seller_runnable: Any
    messages: Annotated[list, add_messages]
    event_bus: EventBus


# --- Graph Nodes ---

async def qa_phase_node(graph_state: GraphState) -> Dict[str, Any]:
    """Node for the Q&A phase where buyers ask questions."""
    state = graph_state['state']
    event_bus = graph_state['event_bus']
    state.round += 1
    print(f"\n\n--- Round {state.round} ---")
    print("--- Phase: Q&A ---")

    qa_instructions = "It is the Q&A phase... Your goal is to gather information... Do NOT 'bid' yet."
    
    tasks = []
    for buyer_id in state.active_buyers:
        persona = next(b for b in state.config['environment']['buyers'] if b['id'] == buyer_id)
        runnable = graph_state['agent_runnables'][buyer_id]
        task = runnable.ainvoke({
            "persona_summary": f"ID: {persona['id']}, Max WTP: ${persona['max_wtp']:,}, ...",
            "state_summary": state.get_state_summary(),
            "phase_instructions": qa_instructions,
            "format_instructions": action_parser.get_format_instructions(),
        })
        tasks.append(task)
        
    try:
        qa_actions_list = await asyncio.gather(*tasks)
        qa_actions = dict(zip(state.active_buyers, qa_actions_list))
    except Exception as e:
        print(f"🚨 Error during Q&A: {e}")
        qa_actions = {b_id: Action(action="fold", commentary=f"Error: {e}") for b_id in state.active_buyers}

    # Process questions and get answers
    question_tasks, askers = [], []
    for buyer_id, action in qa_actions.items():
        if action.action == 'ask' and action.question:
            await event_bus.log(Event(ts=time.time(), type="ask", actor=buyer_id, payload={"question": action.question}), state=state)
            askers.append(buyer_id)
            task = graph_state['seller_runnable'].ainvoke({"question": action.question})
            question_tasks.append(task)
    
    seller_answers = {}
    if question_tasks:
        print(f"  Answering {len(question_tasks)} question(s) from {', '.join(askers)}...")
        try:
            seller_responses_list = await asyncio.gather(*question_tasks)
            seller_answers = dict(zip(askers, seller_responses_list))
        except Exception as e:
            print(f"🚨 Error during seller response: {e}")
            for asker_id in askers:
                seller_answers[asker_id] = SellerResponse(answer=f"Error: {e}", commentary="Error")

    # Log Q&A results
    for buyer_id, action in qa_actions.items():
        if action.action == "ask" and action.question:
            response = seller_answers.get(buyer_id)
            answer_text = response.answer if response else "Seller failed to provide an answer."
            # Log the answer event
            if response:
                await event_bus.log(Event(ts=time.time(), type="answer", actor="Seller", payload={"answer": answer_text}), state=state)
            print(f"  - {buyer_id}: ASK ({action.commentary})\n     L> Q: {action.question}\n     L> A: {answer_text.strip()}")
            state.history.append(f"Q&A Round {state.round}: {buyer_id} asked '{action.question}' -> Answered.")
    
    return {"state": state}


async def bidding_phase_node(graph_state: GraphState) -> Dict[str, Any]:
    """Node for the bidding phase."""
    state = graph_state['state']
    event_bus = graph_state['event_bus']
    print("\n--- Phase: Bidding ---")
    bidding_instructions = "It is the Bidding phase... you must now 'bid' or 'fold'."
    
    tasks = []
    for buyer_id in state.active_buyers:
        persona = next(b for b in state.config['environment']['buyers'] if b['id'] == buyer_id)
        runnable = graph_state['agent_runnables'][buyer_id]
        task = runnable.ainvoke({
             "persona_summary": f"ID: {persona['id']}, Max WTP: ${persona['max_wtp']:,}, ...",
             "state_summary": state.get_state_summary(),
             "phase_instructions": bidding_instructions,
             "format_instructions": action_parser.get_format_instructions(),
        })
        tasks.append(task)
    
    try:
        bidding_actions_list = await asyncio.gather(*tasks)
        bidding_actions = dict(zip(state.active_buyers, bidding_actions_list))
    except Exception as e:
        print(f"🚨 Error during bidding: {e}")
        bidding_actions = {b_id: Action(action="fold", commentary=f"Error: {e}") for b_id in state.active_buyers}

    # Log bid and fold events
    for buyer_id, action in bidding_actions.items():
        if action.action == 'bid':
            await event_bus.log(Event(ts=time.time(), type="bid", actor=buyer_id, payload={"amount": action.amount}), state=state)
        elif action.action == 'fold':
            await event_bus.log(Event(ts=time.time(), type="fold", actor=buyer_id, payload={}), state=state)

    # Process bids and update state
    valid_bids = {b: a for b, a in bidding_actions.items() if a.action == 'bid' and a.amount > state.current_price}
    
    if valid_bids:
        highest_bidder = max(valid_bids, key=lambda k: valid_bids[k].amount)
        state.leading_bidder = highest_bidder
        state.current_price = valid_bids[highest_bidder].amount
        log_msg = f"Round {state.round} Bidding: New high bid of ${state.current_price:,.2f} from {state.leading_bidder}."
        print(f"\n{log_msg}")
        state.history.append(log_msg)

    folded_buyers = {b for b, a in bidding_actions.items() if a.action == 'fold'}
    state.active_buyers = [b for b in state.active_buyers if b not in folded_buyers]

    return {"state": state}

# --- Conditional Edges ---

def should_continue(graph_state: GraphState) -> str:
    """Determine whether to continue the auction or end."""
    state = graph_state['state']
    if len(state.active_buyers) <= 1 or state.round >= state.config['environment']['auction']['max_rounds']:
        print("\n--- Condition Met: Ending Auction ---")
        return "end"
    return "continue"

def finalize_auction_node(graph_state: GraphState) -> Dict[str, Any]:
    """Finalizes the auction, determining winner or failure reason."""
    state = graph_state['state']
    reserve_price = state.config['environment']['seller']['reserve_price']
    
    if state.leading_bidder and state.current_price >= reserve_price:
        state.winner = state.leading_bidder
        state.final_price = state.current_price
        state.history.append(f"Conclusion: SOLD to {state.winner} for ${state.final_price:,.2f}.")
    else:
        state.failure_reason = "No valid bids met the reserve price."
        state.history.append(f"Conclusion: FAILED. {state.failure_reason}")
    
    print(f"--- AUCTION ENDED: {state.history[-1]} ---")
    return {"state": state}


# --- Graph Definition ---

def build_graph():
    """Builds the LangGraph for the auction."""
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

async def run_auction_episode(config: Dict[str, Any], live: bool = False):
    """Runs a full auction episode using the LangGraph orchestrator."""
    
    # Initial state setup
    event_bus = EventBus(live=live)
    env_config = config['environment']
    initial_state = AuctionState(
        config=config,
        current_price=env_config['auction']['start_price'],
        active_buyers=[b['id'] for b in env_config['buyers']],
    )
    # Add event_log to state if not present
    if not hasattr(initial_state, 'event_log'):
        initial_state.event_log = []
    
    # Agent setup
    buyer_runnables = {
        buyer['id']: create_buyer_agent_runnable(buyer)
        for buyer in config['environment']['buyers']
    }
    seller_runnable = create_seller_runnable(config)
    
    graph = build_graph()

    # Initial inputs for the graph
    inputs = {
        "state": initial_state,
        "agent_runnables": buyer_runnables,
        "seller_runnable": seller_runnable,
        "messages": [],
        "event_bus": event_bus,
    }
    
    print("--- 🚀 Starting New Auction Episode 🚀 ---")
    # Instrumentation: emit auction_start event
    await event_bus.log(Event(ts=time.time(), type="auction_start", actor="system", payload={"config": config}), state=initial_state)
    final_graph_state = await graph.ainvoke(inputs)
    
    # Instrumentation: emit auction_end event
    await event_bus.log(Event(ts=time.time(), type="auction_end", actor="system", payload={"winner": final_graph_state['state'].winner, "final_price": final_graph_state['state'].final_price, "failure_reason": final_graph_state['state'].failure_reason}), state=final_graph_state['state'])
    return final_graph_state['state']

# TODO: Instrument bid, ask, fold, and chat events throughout the graph nodes for full live streaming support. 