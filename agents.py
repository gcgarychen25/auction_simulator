"""
Agent definitions for the auction simulator.
"""

import os
import yaml
from typing import Dict, Any

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from schemas import Action, SellerResponse, Participation
from prompts import create_seller_prompt, create_buyer_agent_prompt, create_buyer_preference_prompt

# --- LLM and Parser Setup ---

def get_llm():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.8"))
    if openai_api_key:
        return ChatOpenAI(model=openai_model, openai_api_key=openai_api_key, temperature=temperature)
    elif gemini_api_key:
        return ChatGoogleGenerativeAI(model=gemini_model, google_api_key=gemini_api_key, temperature=temperature)
    else:
        raise ValueError("No LLM API key found. Set either OPENAI_API_KEY or GEMINI_API_KEY in your environment.")

llm = get_llm()
action_parser = PydanticOutputParser(pydantic_object=Action)
seller_parser = PydanticOutputParser(pydantic_object=SellerResponse)
participation_parser = PydanticOutputParser(pydantic_object=Participation)


# --- Agent Runnable Definitions ---

def create_buyer_preference_runnable(persona: Dict[str, Any]):
    """Creates a runnable for a buyer to decide which auctions to join."""
    prompt = create_buyer_preference_prompt().partial(
        persona_summary=yaml.dump(persona),
        format_instructions=participation_parser.get_format_instructions()
    )
    return (prompt | llm | participation_parser).with_config({"run_name": f"Preference_{persona['id']}"})

def create_seller_runnable(property_config: Dict[str, Any]):
    """Creates a runnable for the seller agent for a specific property."""
    prompt = create_seller_prompt()
    # Use .partial() to pre-fill the static parts of the prompt.
    prompt = prompt.partial(
        property_details=yaml.dump(property_config['details']),
        format_instructions=seller_parser.get_format_instructions(),
    )
    chain = prompt | llm | seller_parser
    return chain.with_config({"run_name": f"SellerAgent_{property_config['id']}"})

def create_buyer_agent_runnable(persona: Dict[str, Any]):
    """Creates a complete LangChain runnable for a single buyer agent."""
    buyer_prompt_template = create_buyer_agent_prompt()
    
    chain = buyer_prompt_template | llm | action_parser
    return chain.with_config({"run_name": f"Buyer_{persona['id']}"}) 