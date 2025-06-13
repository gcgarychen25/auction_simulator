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

from schemas import Action, SellerResponse
from prompts import create_seller_prompt, create_buyer_agent_prompt

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
    buyer_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a participant in a multi-agent auction for a house. You must act according to your persona.\n"
             "Your persona summary is: {persona_summary}\n\n"
             "The current state of the auction is: {state_summary}\n\n"
             "You have two phases: Q&A and Bidding. Your current phase is: {phase_instructions}\n\n"
             "You must respond with a JSON object matching the following schema: {format_instructions}"
             ),
            ("human", "It is your turn to act. Decide your next move (ask, bid, call, or fold).")
        ]
    )
    chain = buyer_prompt_template | llm | action_parser
    return chain.with_config({"run_name": f"Buyer_{persona['id']}"}) 