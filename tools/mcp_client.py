"""
MCP client for remote tool invocation.
"""

import json
import asyncio
from typing import Dict, Any


class MCPClient:
    """Generic MCP JSON-RPC client wrapper."""
    
    def __init__(self):
        self.connected = False
        # In a real implementation, this would connect to MCP servers
        # For now, we'll simulate with a mock
    
    async def connect(self):
        """Connect to MCP servers."""
        # Mock connection
        self.connected = True
        print("MCP Client: Connected to servers")
    
    async def invoke(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Invoke a tool via MCP.
        
        Args:
            tool_name: Name of the tool to invoke
            args: Arguments for the tool
            
        Returns:
            Result string from the tool
        """
        if not self.connected:
            await self.connect()
        
        # For ASK_SELLER, simulate a seller response
        if tool_name == "ASK_SELLER":
            question = args.get("question", "")
            if not question:
                return "ERROR: ASK_SELLER requires 'question' parameter"
            
            # Mock seller response
            mock_responses = {
                "condition": "The property is in excellent condition with recent renovations.",
                "history": "This house was built in 2010 and has been well-maintained.",
                "neighborhood": "This is a quiet family neighborhood with great schools nearby.",
                "price": "The price reflects recent market conditions and comparable sales.",
                "default": f"Thank you for asking '{question}'. I'm happy to share that this property offers excellent value."
            }
            
            # Simple keyword matching for mock response
            question_lower = question.lower()
            for keyword, response in mock_responses.items():
                if keyword in question_lower:
                    return f"SELLER RESPONSE: {response}"
            
            return f"SELLER RESPONSE: {mock_responses['default']}"
        
        # Default response for unknown tools
        return f"ERROR: Unknown MCP tool '{tool_name}'"
    
    async def disconnect(self):
        """Disconnect from MCP servers."""
        self.connected = False
        print("MCP Client: Disconnected")


# Global MCP client instance
mcp_client = MCPClient()


async def invoke_mcp(tool_name: str, args: Dict[str, Any]) -> str:
    """Convenience function to invoke MCP tools."""
    return await mcp_client.invoke(tool_name, args) 