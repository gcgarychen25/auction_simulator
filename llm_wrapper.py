"""
LLM Wrapper for Auction Simulator

Handles all communication with external LLM APIs, specifically Gemini.
Uses asyncio and httpx for efficient, non-blocking I/O with retry logic.
"""

import asyncio
import httpx
import json
import os
from typing import Tuple, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Async wrapper for making robust calls to LLM APIs.
    
    Features:
    - Exponential backoff retry mechanism
    - Response parsing and error handling
    - Rate limiting and timeout management
    """
    
    def __init__(self, model_name: str = None, max_retries: int = 3):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_name: Name of the model to use (defaults to env var or gemini-2.5-flash-preview-05-20)
            max_retries: Maximum number of retry attempts
        """
        # Allow model configuration via environment variable
        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = 1.0  # Base delay for exponential backoff
        
        # TODO: Set your Gemini API key as environment variable
        # export GEMINI_API_KEY="your_api_key_here"
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables!")
            logger.warning("Please set your API key: export GEMINI_API_KEY='your_key_here'")
        
        # Gemini API endpoint
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
    async def call(self, prompt: str, temperature: float = 0.7) -> Tuple[str, str]:
        """
        Make an async call to the LLM API with retry logic.
        
        Args:
            prompt: The input prompt for the LLM
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Tuple of (action, commentary) parsed from response
        """
        if not self.api_key:
            # Fallback to heuristic behavior for testing
            logger.warning("No API key found, using fallback response")
            return self._fallback_response(prompt)
        
        # Prepare request payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1000,  # Generous limit for Phase 0
                "topP": 0.8,
                "topK": 10
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = self._extract_response_text(result)
                        return self.parse_response(response_text)
                    
                    elif response.status_code >= 500:
                        # Server error - retry with backoff
                        if attempt < self.max_retries:
                            delay = self.base_delay * (2 ** attempt)
                            logger.warning(f"Server error {response.status_code}, retrying in {delay}s...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise Exception(f"Server error after {self.max_retries} retries")
                    
                    elif response.status_code == 429:
                        # Rate limit - wait longer
                        if attempt < self.max_retries:
                            delay = self.base_delay * (3 ** attempt)
                            logger.warning(f"Rate limited, retrying in {delay}s...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise Exception("Rate limited after maximum retries")
                    
                    else:
                        # Client error - don't retry
                        error_msg = f"API error {response.status_code}: {response.text}"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                        
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Request timeout, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise Exception("Request timeout after maximum retries")
                    
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error: {e}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed after {self.max_retries} retries: {e}")
                    # Fallback to heuristic response
                    return self._fallback_response(prompt)
    
    async def call_direct(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Make a direct call without action parsing - returns raw response.
        Useful for question answering where we don't need action numbers.
        """
        if not self.api_key:
            logger.warning("No API key found, using fallback response")
            return "I don't have access to specific property details, but encourage you to review the inspection reports."
        
        # Prepare request payload (same as call method)
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1000,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        # Single attempt for direct calls (simpler logic)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._extract_response_text(result)
                else:
                    logger.error(f"Direct API call failed: {response.status_code}")
                    return "I apologize, I'm unable to provide specific details at this time."
                    
        except Exception as e:
            logger.error(f"Direct API call error: {e}")
            return "I apologize, I'm unable to provide specific details at this time."
    
    def _extract_response_text(self, response_data: dict) -> str:
        """Extract text from Gemini API response."""
        try:
            # Debug: Uncomment to see raw responses
            # print(f"\n🔍 DEBUG: Raw API Response:")
            # print(json.dumps(response_data, indent=2))
            # print("=" * 50)
            
            candidates = response_data.get("candidates", [])
            
            if candidates:
                candidate = candidates[0]
                
                # Check for safety filter blocks
                finish_reason = candidate.get("finishReason")
                if finish_reason == "SAFETY":
                    logger.warning("Response blocked by safety filters")
                    return ""
                elif finish_reason == "MAX_TOKENS":
                    logger.warning(f"Hit max tokens limit. Extracting partial response.")
                elif finish_reason and finish_reason != "STOP":
                    logger.warning(f"Response blocked due to: {finish_reason}")
                    return ""
                
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                if parts:
                    text = parts[0].get("text", "")
                    logger.debug(f"LLM Response: '{text}'")  # Debug level instead of info
                    return text
                    
            logger.warning("No text found in response structure")
            return ""
        except Exception as e:
            logger.error(f"Error extracting response text: {e}")
            logger.error(f"Response data: {response_data}")
            return ""
    
    def parse_response(self, response_text: str) -> Tuple[str, str]:
        """
        Parse LLM response into action and commentary.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            Tuple of (action, commentary)
        """
        if not response_text:
            return "0", "No response received"
        
        # Split response by whitespace and get first word as action
        parts = response_text.strip().split()
        if not parts:
            return "0", "Empty response"
        
        action = parts[0].lower()
        commentary = " ".join(parts[1:]) if len(parts) > 1 else ""
        
        # Map natural language to action numbers
        action_mapping = {
            # Buyer actions
            'fold': '0',
            'pass': '0',
            'quit': '0',
            'bid_500': '1',
            'bid500': '1',
            'small_bid': '1',
            'bid_1000': '2',
            'bid1000': '2',
            'large_bid': '2',
            'bid': '2',  # Default to large bid
            'question': '3',
            'ask': '3',
            'ask_question': '3',
            
            # Seller actions
            'announce': '0',
            'next': '0',
            'answer': '1',
            'respond': '1',
            'close': '2',
            'sold': '2',
            'end': '2',
        }
        
        # Try to map action to number
        mapped_action = action_mapping.get(action, action)
        
        # Validate it's a valid action number
        if mapped_action not in ['0', '1', '2', '3']:
            # If we can't parse it, default to safe action
            mapped_action = '0'
            commentary = f"Could not parse action '{action}', defaulting to fold/announce"
        
        return mapped_action, commentary
    
    def _fallback_response(self, prompt: str) -> Tuple[str, str]:
        """
        Provide a fallback response when API is unavailable.
        
        This is used for testing without API keys.
        """
        if "seller" in prompt.lower():
            return "0", "Announcing next round (fallback)"
        else:
            # For buyers, return a conservative action
            return "0", "Folding (fallback - no API key)"


# Convenience function for single calls
async def call_llm(prompt: str, model_name: str = None) -> Tuple[str, str]:
    """
    Convenience function for making a single LLM call.
    
    Args:
        prompt: The input prompt
        model_name: Model to use (defaults to env var or gemini-2.5-flash-preview-05-20)
        
    Returns:
        Tuple of (action, commentary)
    """
    wrapper = LLMWrapper(model_name=model_name)
    return await wrapper.call(prompt) 