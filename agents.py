"""
Agent LLM runnables for the auction simulator.
"""

from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from schemas import Action


# Agent prompt template as specified in design doc
AGENT_PROMPT_TEMPLATE = """SYSTEM:
You are an autonomous auction participant. Use tools to achieve the best outcome.

Context:
{context}

Last observation:
{observation}

Available tools:
• BID – Place a bid with specified amount (must be higher than current price)
• CALL – Stay in auction at current price
• FOLD – Withdraw from auction
• ASK_SELLER – Ask the seller a question about the property
• STATUS – Get current auction status
• CHECK_TERMINATION – Check if auction should end
• FINISH – End your participation

Respond ONLY with JSON:
{{
 "tool": "<TOOL_NAME | FINISH>",
 "args": {{ ... }},
 "commentary": "<short rationale>"
}}

Remember:
- You want to win valuable properties at good prices
- Don't overbid beyond reasonable value
- Use ASK_SELLER to gather information before making decisions
- CHECK_TERMINATION to understand auction status
- When ready to stop, use FINISH"""


class BuyerAgent:
    """Buyer agent using LLM for decision making."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", buyer_profile: Dict[str, Any] = None):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        self.profile = buyer_profile or {}
        
        # Set up prompt and parser
        self.prompt = PromptTemplate(
            template=AGENT_PROMPT_TEMPLATE,
            input_variables=["context", "observation"]
        )
        self.parser = PydanticOutputParser(pydantic_object=Action)
        
        # Create the runnable chain
        self.runnable = self.prompt | self.llm | StrOutputParser()
    
    async def invoke(self, context: List[str], observation: str = None) -> str:
        """
        Invoke the buyer agent.
        
        Args:
            context: List of context strings (conversation history)
            observation: Last observation from tool execution
            
        Returns:
            Raw LLM response (should be JSON)
        """
        context_str = "\n".join(context) if context else "Auction starting..."
        observation_str = observation or "Auction beginning"
        
        response = await self.runnable.ainvoke({
            "context": context_str,
            "observation": observation_str
        })
        
        return response
    
    def get_profile_context(self) -> str:
        """Get buyer profile as context string."""
        if not self.profile:
            return "Buyer profile: Standard buyer"
        
        profile_parts = []
        if "budget" in self.profile:
            profile_parts.append(f"Budget: ${self.profile['budget']:,.2f}")
        if "strategy" in self.profile:
            profile_parts.append(f"Strategy: {self.profile['strategy']}")
        if "preferences" in self.profile:
            prefs = self.profile["preferences"]
            if isinstance(prefs, dict):
                pref_strs = [f"{k}: {v}" for k, v in prefs.items()]
                profile_parts.append(f"Preferences: {', '.join(pref_strs)}")
        
        return f"Buyer profile: {'; '.join(profile_parts)}"


def create_buyer_agent(config: Dict[str, Any]) -> BuyerAgent:
    """
    Factory function to create a buyer agent from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured BuyerAgent instance
    """
    model_name = config.get("model_name", "gpt-3.5-turbo")
    buyer_profile = config.get("buyer_profile", {})
    
    return BuyerAgent(model_name=model_name, buyer_profile=buyer_profile) 