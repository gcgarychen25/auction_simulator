"""
Multi-Agent Auction Orchestrator using Agent-as-Tool Pattern

This module implements the auction using a robust, multi-agent architecture
inspired by the "agent-as-tool" pattern. A central Auctioneer agent
orchestrates multiple specialist Buyer agents.
"""

import asyncio
import os
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
import yaml

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- Core Data Structures & LLM Setup ---

class Action(BaseModel):
    """A structured action to be taken in the auction."""
    action: Literal["bid", "fold", "ask"] = Field(description="Action to take: bid, fold, or ask.")
    amount: float = Field(default=0.0, description="The bid amount, if applicable. Must be higher than the current price.")
    commentary: str = Field(description="Brief reasoning for the action.")

# Initialize the Gemini model via LangChain, now with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # If the key is not found, we should stop execution.
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.8)
action_parser = PydanticOutputParser(pydantic_object=Action)

# --- State Management ---

class AuctionState:
    """Manages the state of the auction."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_config = config['environment']
        self.round = 0
        self.current_price = env_config['auction']['start_price']
        self.leading_bidder = None
        self.active_buyers = [b['id'] for b in env_config['buyers']]
        self.history = []
        self.winner = None
        self.final_price = None
        self.failure_reason = ""

    def get_state_summary(self) -> str:
        """Creates a concise summary of the auction state for prompts."""
        summary = [
            f"This is Round {self.round} of a real estate auction.",
            f"Current Price: ${self.current_price:,.2f}",
            f"Leading Bidder: {self.leading_bidder or 'None'}",
            f"Active Buyers ({len(self.active_buyers)}): {', '.join(self.active_buyers)}",
            "---",
            "Recent History:",
        ]
        summary.extend(f"- {h}" for h in self.history[-3:])
        return "\n".join(summary)

    def update_with_bids(self, bids: Dict[str, Action]):
        """Updates the state based on the bids received in a round."""
        self.round += 1
        
        valid_bids = {
            bidder: action for bidder, action in bids.items()
            if action.action == 'bid' and action.amount > self.current_price
        }

        if valid_bids:
            highest_bidder = max(valid_bids, key=lambda k: valid_bids[k].amount)
            self.leading_bidder = highest_bidder
            self.current_price = valid_bids[highest_bidder].amount
            log_msg = f"Round {self.round}: New high bid of ${self.current_price:,.2f} from {self.leading_bidder}."
        else:
            log_msg = f"Round {self.round}: No new valid bids."
        
        self.history.append(log_msg)
        print(f"\n{log_msg}")

        # Update active buyers (remove those who folded)
        folded_buyers = {bidder for bidder, action in bids.items() if action.action == 'fold'}
        self.active_buyers = [b for b in self.active_buyers if b not in folded_buyers]
        
    def end_auction(self):
        """Finalizes the auction, determining winner or failure reason."""
        reserve_price = self.config['environment']['seller']['reserve_price']
        if self.leading_bidder and self.current_price >= reserve_price:
            self.winner = self.leading_bidder
            self.final_price = self.current_price
            self.history.append(f"Conclusion: SOLD to {self.winner} for ${self.final_price:,.2f}.")
        else:
            if not self.leading_bidder:
                self.failure_reason = "No bids were placed that met the criteria."
            else:
                self.failure_reason = f"Highest bid of ${self.current_price:,.2f} was below the ${reserve_price:,.2f} reserve."
            self.history.append(f"Conclusion: FAILED. {self.failure_reason}")
        print(f"--- AUCTION ENDED: {self.history[-1]} ---")


# --- Agent & Tool Definitions ---

def create_buyer_agent_prompt():
    """Creates the prompt template for a buyer agent."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a participant in a real estate auction. Your goal is to win the auction if the price is right for your persona.

Your persona is defined as:
{persona_summary}

The current state of the auction is:
{state_summary}

You must decide whether to 'bid', 'fold', or 'ask'.
- If you bid, the amount MUST be higher than the current price.
- If you fold, you are out of the auction for good.
- Provide a brief commentary explaining your reasoning.

Respond ONLY with the required JSON object.
{format_instructions}
"""),
        ("human", "Based on your persona and the auction state, what is your next action?"),
    ])

def create_buyer_agent_runnable(persona: Dict[str, Any]):
    """Creates a complete LangChain runnable for a single buyer agent."""
    prompt = create_buyer_agent_prompt()
    chain = prompt | llm | action_parser
    return chain.with_config({"run_name": f"Buyer_{persona['id']}"})

# --- Main Orchestration Loop ---

async def run_auction_episode(config: Dict[str, Any]):
    """
    Runs a full auction episode using the agent-as-tool pattern.
    The core logic resides here, acting as the orchestrator.
    """
    state = AuctionState(config)
    
    # Create a runnable for each buyer persona
    buyer_runnables = {
        buyer['id']: create_buyer_agent_runnable(buyer)
        for buyer in config['environment']['buyers']
    }
    
    print("--- 🚀 Starting New Auction Episode 🚀 ---")
    print(state.get_state_summary())

    while True:
        # 1. Check termination conditions
        if not state.active_buyers or state.round >= config['environment']['auction']['max_rounds']:
            print("\n--- Condition Met: Ending Auction ---")
            break

        # 2. Gather actions from all active buyers in parallel
        print(f"\n--- Round {state.round + 1}: Gathering bids from {len(state.active_buyers)} active buyers ---")
        
        tasks = []
        for buyer_id in state.active_buyers:
            persona = next(b for b in config['environment']['buyers'] if b['id'] == buyer_id)
            runnable = buyer_runnables[buyer_id]
            task = runnable.ainvoke({
                "persona_summary": f"ID: {persona['id']}, Max WTP: ${persona['max_wtp']:,}, Risk Aversion: {persona['risk_aversion']}",
                "state_summary": state.get_state_summary(),
                "format_instructions": action_parser.get_format_instructions(),
            })
            tasks.append(task)
            
        try:
            buyer_actions_list = await asyncio.gather(*tasks)
            buyer_actions = dict(zip(state.active_buyers, buyer_actions_list))
        except Exception as e:
            print(f"🚨 An error occurred during buyer action gathering: {e}")
            # On error, make all active buyers fold
            buyer_actions = {b_id: Action(action="fold", commentary=f"Error: {e}") for b_id in state.active_buyers}

        # Print actions for this round
        for buyer_id, action in buyer_actions.items():
            print(f"  - {buyer_id}: {action.action.upper()} ${action.amount:,.2f} ({action.commentary})")
            
        # 3. Update state with the new bids
        state.update_with_bids(buyer_actions)

    # 4. Finalize auction
    state.end_auction()
    
    return state 