"""
MCP server for ASK_SELLER tool.
Reuses seller prompt logic for answering buyer questions.
"""

import asyncio
import json
from typing import Dict, Any


class SellerAgent:
    """Mock seller agent for responding to buyer questions."""
    
    def __init__(self):
        self.property_info = {
            "condition": "The property is in excellent condition with recent renovations to the kitchen and bathrooms.",
            "history": "This house was built in 2010 and has been well-maintained by a single family owner.",
            "neighborhood": "This is a quiet family neighborhood with great schools nearby and low crime rates.",
            "price": "The price reflects recent market conditions and comparable sales in the area.",
            "features": "The property features hardwood floors, granite countertops, and a large backyard.",
            "utilities": "All utilities are included except for electricity. Water and gas are very efficient.",
            "parking": "There's a 2-car garage plus additional driveway parking for guests.",
            "schools": "The elementary school is rated 9/10 and is just a 5-minute walk away.",
            "maintenance": "All major systems (HVAC, plumbing, electrical) were updated within the last 5 years."
        }
    
    async def answer_question(self, question: str) -> str:
        """
        Answer a buyer's question about the property.
        
        Args:
            question: The buyer's question
            
        Returns:
            Seller's response
        """
        question_lower = question.lower()
        
        # Simple keyword matching
        for keyword, response in self.property_info.items():
            if keyword in question_lower:
                return f"SELLER: {response}"
        
        # General response if no specific match
        return f"SELLER: Thank you for asking '{question}'. I'm happy to share that this property offers excellent value and would be perfect for the right buyer. Feel free to ask any specific questions you have!"


class AskSellerMCPServer:
    """MCP server for the ASK_SELLER tool."""
    
    def __init__(self):
        self.seller = SellerAgent()
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP request.
        
        Args:
            request: MCP JSON-RPC request
            
        Returns:
            MCP JSON-RPC response
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "ask_seller":
                question = params.get("question", "")
                if not question:
                    raise ValueError("Missing 'question' parameter")
                
                response = await self.seller.answer_question(question)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"response": response}
                }
            else:
                return {
                    "jsonrpc": "2.0", 
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
                
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id, 
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    
    async def start_server(self, socket_path: str = "/tmp/ask_seller.sock"):
        """
        Start the MCP server.
        
        Args:
            socket_path: Unix socket path for communication
        """
        print(f"Starting ASK_SELLER MCP server on {socket_path}")
        
        # In a real implementation, this would set up a proper MCP server
        # For now, we'll just log that the server would be running
        print("ASK_SELLER MCP server running (mock implementation)")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    server = AskSellerMCPServer()
    asyncio.run(server.start_server()) 