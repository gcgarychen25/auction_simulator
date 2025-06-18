"""
Prompt templates for the auction agents.
"""

from langchain_core.prompts import ChatPromptTemplate

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

def create_buyer_preference_prompt():
    """Creates the prompt for a buyer to decide which auctions to join."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a potential buyer in a real estate market with multiple properties up for auction. Your task is to decide which auctions to participate in based on your persona and the property details.

        Your persona is defined as:
        {persona_summary}

        Here are the properties available:
        {properties_summary}

        Based on your persona, decide which properties you are interested in. You can join one, multiple, or no auctions.

        Respond ONLY with the required JSON object, including your reasoning in the commentary.
        {format_instructions}
        """),
        ("human", "Review the properties and decide which auctions you will join."),
    ])

def create_buyer_agent_prompt():
    """Creates the prompt template for a buyer agent for a specific auction."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a participant in a real estate auction for a specific property. Your goal is to win the auction if the price is right for your persona.

        Your persona is defined as:
        {persona_summary}

        The property being auctioned is:
        {property_details}

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