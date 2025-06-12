"""
Agent definitions for the auction simulator.
"""

import os
import yaml
from typing import Dict, Any

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import Action, SellerResponse
from prompts import create_seller_prompt, create_buyer_agent_prompt

# --- LLM and Parser Setup ---

# Initialize the Gemini model via LangChain, now with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # If the key is not found, we should stop execution.
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.8)
action_parser = PydanticOutputParser(pydantic_object=Action)
seller_parser = PydanticOutputParser(pydantic_object=SellerResponse)


# --- Agent Runnable Definitions ---

def create_seller_runnable(config: Dict[str, Any]):
    """Creates a runnable for the seller agent."""
    prompt = create_seller_prompt()
    # Use .partial() to pre-fill the static parts of the prompt.
    prompt = prompt.partial(
        property_details=yaml.dump(config['environment']['property']),
        format_instructions=seller_parser.get_format_instructions(),
    )
    chain = prompt | llm | seller_parser
    return chain.with_config({"run_name": "SellerAgent"})

def create_buyer_agent_runnable(persona: Dict[str, Any]):
    """Creates a complete LangChain runnable for a single buyer agent."""
    prompt = create_buyer_agent_prompt()
    chain = prompt | llm | action_parser
    return chain.with_config({"run_name": f"Buyer_{persona['id']}"}) 