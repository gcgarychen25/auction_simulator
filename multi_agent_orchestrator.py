"""
Multi-Agent Auction Orchestrator using Agent-as-Tool Pattern

This module implements the auction using a robust, multi-agent architecture
inspired by the "agent-as-tool" pattern. A central Auctioneer agent
orchestrates multiple specialist Buyer agents.
"""

import asyncio
import os
from typing import List, Dict, Any, Literal, Optional
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
    question: Optional[str] = Field(default=None, description="The question for the seller, if action is 'ask'.")
    commentary: str = Field(description="Brief reasoning for the action.")

class SellerResponse(BaseModel):
    """A structured response from the seller to a buyer's question."""
    answer: str = Field(description="The seller's direct answer to the question.")
    commentary: str = Field(description="Brief commentary on the seller's thinking.")

# Initialize the Gemini model via LangChain, now with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # If the key is not found, we should stop execution.
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.8)
action_parser = PydanticOutputParser(pydantic_object=Action)
seller_parser = PydanticOutputParser(pydantic_object=SellerResponse)

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
        summary.extend(f"- {h}" for h in self.history[-5:])
        return "\n".join(summary)

    def update_with_bids(self, bids: Dict[str, Action]):
        """Updates the state based on the bids received in a round."""
        # self.round is now managed by the main loop
        
        valid_bids = {
            bidder: action for bidder, action in bids.items()
            if action.action == 'bid' and action.amount > self.current_price
        }

        if valid_bids:
            highest_bidder = max(valid_bids, key=lambda k: valid_bids[k].amount)
            self.leading_bidder = highest_bidder
            self.current_price = valid_bids[highest_bidder].amount
            log_msg = f"Round {self.round} Bidding: New high bid of ${self.current_price:,.2f} from {self.leading_bidder}."
        else:
            log_msg = f"Round {self.round} Bidding: No new valid bids."
        
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

def create_seller_prompt():
    """Creates the prompt for the seller to answer a question."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are the seller in a real estate auction. You must answer a buyer's question based *only* on the provided property information. Be honest but frame your answers to maintain interest in the property.

Property Information:
{property_details}

Do not invent information. If the answer is not in the property details, state that the information is not available.

Respond ONLY with the required JSON object.
{format_instructions}"""),
        ("human", "A buyer has asked the following question: '{question}'"),
    ])

def create_seller_runnable(config: Dict[str, Any]):
    """Creates a runnable for the seller agent."""
    prompt = create_seller_prompt()
    # Use .partial() to pre-fill the static parts of the prompt.
    # This ensures the chain correctly expects only the 'question' variable at invocation time.
    prompt = prompt.partial(
        property_details=yaml.dump(config['environment']['property']),
        format_instructions=seller_parser.get_format_instructions(),
    )
    chain = prompt | llm | seller_parser
    return chain.with_config({"run_name": "SellerAgent"})

def create_buyer_agent_prompt():
    """Creates the prompt template for a buyer agent."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a participant in a real estate auction. Your goal is to win the auction if the price is right for your persona.

Your persona is defined as:
{persona_summary}

The current state of the auction is:
{state_summary}

{phase_instructions}

When asking a question, use the 'ask' action and populate the 'question' field.
When bidding or folding, the 'question' field must be null.

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
    
    # Create a runnable for each buyer persona, which will be invoked with different prompts
    buyer_runnables = {
        buyer['id']: create_buyer_agent_runnable(buyer)
        for buyer in config['environment']['buyers']
    }
    seller_runnable = create_seller_runnable(config)
    
    print("--- 🚀 Starting New Auction Episode 🚀 ---")
    print(state.get_state_summary())

    while True:
        # 1. Check termination conditions
        if len(state.active_buyers) <= 1 or state.round >= config['environment']['auction']['max_rounds']:
            print("\n--- Condition Met: Ending Auction ---")
            break

        state.round += 1
        print(f"\n\n--- Round {state.round} ---")

        # =================================================================
        # 2. Q&A Phase
        # =================================================================
        print("--- Phase: Q&A ---")
        qa_instructions = "It is the Q&A phase. Your goal is to gather information. You can either 'ask' a question based on your persona's requirements or use the 'fold' action to PASS on asking a question for this round. Do NOT 'bid' yet."
        
        tasks = []
        for buyer_id in state.active_buyers:
            persona = next(b for b in config['environment']['buyers'] if b['id'] == buyer_id)
            runnable = buyer_runnables[buyer_id]
            task = runnable.ainvoke({
                "persona_summary": f"ID: {persona['id']}, Max WTP: ${persona['max_wtp']:,}, Risk Aversion: {persona['risk_aversion']}, Requirements: {persona.get('requirements', 'None')}",
                "state_summary": state.get_state_summary(),
                "phase_instructions": qa_instructions,
                "format_instructions": action_parser.get_format_instructions(),
            })
            tasks.append(task)
            
        try:
            qa_actions_list = await asyncio.gather(*tasks)
            qa_actions = dict(zip(state.active_buyers, qa_actions_list))
        except Exception as e:
            print(f"🚨 An error occurred during Q&A action gathering: {e}")
            qa_actions = {b_id: Action(action="fold", commentary=f"Error during Q&A phase: {e}") for b_id in state.active_buyers}

        # Handle questions, get answers, and log everything in a compact format.
        question_tasks, askers = [], []
        for buyer_id, action in qa_actions.items():
            if action.action == 'ask' and action.question:
                askers.append(buyer_id)
                task = seller_runnable.ainvoke({"question": action.question})
                question_tasks.append(task)

        seller_answers: Dict[str, SellerResponse] = {}
        if question_tasks:
            print(f"  Answering {len(question_tasks)} question(s) from {', '.join(askers)}...")
            try:
                seller_responses_list = await asyncio.gather(*question_tasks)
                seller_answers = dict(zip(askers, seller_responses_list))
            except Exception as e:
                print(f"🚨 An error occurred during seller response generation: {e}")
                for asker_id in askers:
                    seller_answers[asker_id] = SellerResponse(answer=f"Error: Could not generate a response. {e}", commentary="Error")

        # Log Q&A results
        for buyer_id, action in qa_actions.items():
            if action.action == "ask" and action.question:
                response = seller_answers.get(buyer_id)
                answer_text = response.answer if response else "Seller failed to provide an answer."

                print(f"  - {buyer_id}: ASK ({action.commentary})")
                print(f"     L> Question: {action.question}")
                print(f"     L> Seller's Answer: {answer_text.strip()}")

                qa_log = f"Q&A in Round {state.round}: {buyer_id} asked: '{action.question}' -> Seller answered: '{answer_text}'"
                state.history.append(qa_log)
            else:
                 print(f"  - {buyer_id}: PASS ({action.commentary})")

        if not question_tasks:
            print("  No questions were asked in this round.")

        # =================================================================
        # 3. Bidding Phase
        # =================================================================
        print("\n--- Phase: Bidding ---")
        if question_tasks:
             print("Bidders should now consider the answers from the Q&A phase.")

        bidding_instructions = "It is the Bidding phase. Based on all available information, including any new answers from the seller, you must now 'bid' or 'fold'. A 'fold' action is final and removes you from the auction. Do NOT 'ask' questions in this phase."
        
        tasks = []
        for buyer_id in state.active_buyers:
            persona = next(b for b in config['environment']['buyers'] if b['id'] == buyer_id)
            runnable = buyer_runnables[buyer_id]
            task = runnable.ainvoke({
                "persona_summary": f"ID: {persona['id']}, Max WTP: ${persona['max_wtp']:,}, Risk Aversion: {persona['risk_aversion']}, Requirements: {persona.get('requirements', 'None')}",
                "state_summary": state.get_state_summary(),
                "phase_instructions": bidding_instructions,
                "format_instructions": action_parser.get_format_instructions(),
            })
            tasks.append(task)
            
        try:
            bidding_actions_list = await asyncio.gather(*tasks)
            bidding_actions = dict(zip(state.active_buyers, bidding_actions_list))
        except Exception as e:
            print(f"🚨 An error occurred during bidding action gathering: {e}")
            bidding_actions = {b_id: Action(action="fold", commentary=f"Error: {e}") for b_id in state.active_buyers}

        # Print actions for this round
        for buyer_id, action in bidding_actions.items():
            # Filter out any stray 'ask' actions from the bidding phase
            if action.action != 'ask':
                 print(f"  - {buyer_id}: {action.action.upper()} ${action.amount:,.2f} ({action.commentary})")
            else:
                 print(f"  - {buyer_id}: Invalid action 'ASK' in bidding phase. Action ignored.")
            
        # 4. Update state with the new bids
        state.update_with_bids(bidding_actions)

    # 5. Finalize auction
    state.end_auction()
    
    return state 