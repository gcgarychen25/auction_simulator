"""
Heuristic Policies for Auction Agents

Implements rule-based policies for each agent persona, providing deterministic
behavior based on persona characteristics like risk aversion and ask probability.
"""

from typing import Dict, List, Any
import numpy as np

class HeuristicSellerPolicy:
    """A simple rule-based policy for the seller."""
    def get_seller_action(self, observation: Dict, info: Dict) -> int:
        """
        Decide whether to continue, answer, or close the auction.
        This is a simplified version of the logic in run.py for heuristic mode.
        """
        current_price = observation['price'][0]
        active_buyers = sum(observation['active_mask'])
        round_no = observation['round_no'][0]
        # In a real scenario, this would need the reserve price from config,
        # but we'll keep it simple as the main logic is in run.py's simulator class.
        
        # If there are questions, the environment logic will likely force an answer.
        if info.get('questions'):
            return 1 # Answer

        # Close auction if only one bidder is left after a few rounds
        if round_no > 3 and active_buyers <= 1:
            return 2 # Close

        # Otherwise, continue
        return 0

def create_heuristic_policies(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create heuristic policies for all agents from config.
    
    Args:
        config: The environment configuration dictionary.
        
    Returns:
        A dictionary containing the seller policy and a list of buyer policies.
    """
    # Seller policy
    seller_policy = HeuristicSellerPolicy()
    
    # Buyer policies
    buyers = config['environment']['buyers']
    buyer_policies = []
    for buyer_config in buyers:
        # Each buyer gets their own policy instance, configured by their persona
        policy = HeuristicPolicy(buyer_config)
        buyer_policies.append(policy)
        
    return {'seller': seller_policy, 'buyers': buyer_policies}

class HeuristicPolicy:
    """
    A rule-based heuristic policy for a single buyer persona.
    The policy's behavior is determined by the persona's configuration.
    """
    def __init__(self, persona_config: Dict[str, Any]):
        """
        Initialize the policy with a specific buyer persona.
        
        Args:
            persona_config: Dictionary defining the buyer's persona.
        """
        self.persona = persona_config
        self.id = persona_config['id']
        self.max_wtp = persona_config['max_wtp']
        self.risk_aversion = persona_config.get('risk_aversion', 0.5)
        self.ask_prob = persona_config.get('ask_prob', 0.1)

    def get_buyer_action(self, observation: Dict, persona: Dict[str, Any], buyer_idx: int) -> int:
        """
        Determine the buyer's action based on observation and persona, with enhanced randomness.
        
        Args:
            observation: The current state of the auction environment.
            persona: The persona configuration for the specific buyer.
            buyer_idx: The index of the buyer making the decision.
            
        Returns:
            An integer representing the chosen action.
        """
        
        # Unpack observation
        current_price = observation['price'][0]
        active_mask = observation['active_mask']
        bids_left = observation['bids_left'][buyer_idx]
        
        # --- Pre-computation & State Analysis ---
        
        # Immediately fold if not active, has no bids left, or price exceeds max willingness-to-pay
        if not active_mask[buyer_idx] or bids_left == 0 or current_price > self.max_wtp:
            return 0  # Fold

        # Define action constants for clarity
        FOLD = 0
        BID_500 = 1
        BID_1000 = 2
        ASK_QUESTION = 3
        
        # Calculate how close the current price is to the buyer's maximum limit
        price_to_wtp_ratio = current_price / self.max_wtp
        
        # Initialize action weights
        weights = {FOLD: 0.0, BID_500: 0.0, BID_1000: 0.0, ASK_QUESTION: 0.0}
        
        # --- Persona-Specific Weighting Logic ---
        
        if "CONSERVATIVE" in self.id:
            # High risk aversion: fold probability increases sharply with price
            weights[FOLD] = self.risk_aversion * (price_to_wtp_ratio ** 2)
            # Prefers smaller bids, probability decreases as it gets riskier
            weights[BID_500] = (1 - price_to_wtp_ratio) * 5
            # Very low chance of a large bid
            weights[BID_1000] = (1 - price_to_wtp_ratio) * 0.1
            weights[ASK_QUESTION] = self.ask_prob
            
        elif "AGGRESSIVE" in self.id:
            # Low risk aversion: fold probability is low until the very end
            weights[FOLD] = self.risk_aversion * (price_to_wtp_ratio ** 5)
            # Prefers larger bids to intimidate, less sensitive to price
            weights[BID_1000] = (1 - self.risk_aversion) * 5
            # Still has a chance to make a smaller bid
            weights[BID_500] = (1 - self.risk_aversion) * 2
            weights[ASK_QUESTION] = self.ask_prob

        elif "ANALYTICAL" in self.id:
            # High probability of asking questions first
            weights[ASK_QUESTION] = self.ask_prob * 10
            # If not asking, acts cautiously
            weights[FOLD] = self.risk_aversion * price_to_wtp_ratio
            weights[BID_500] = (1 - price_to_wtp_ratio) * 2
            weights[BID_1000] = 0.0 # An analytical buyer would likely not make large jumps

        elif "BUDGET" in self.id:
            # Very high probability of folding near budget
            if price_to_wtp_ratio > 0.9:
                weights[FOLD] = (price_to_wtp_ratio ** 10) * 100
            else:
                 weights[FOLD] = 0.1 # Always a small chance to fold
            # Only makes small bids
            weights[BID_500] = (1 - price_to_wtp_ratio)
            weights[BID_1000] = 0.0 # Never makes large bids
            weights[ASK_QUESTION] = self.ask_prob

        elif "FOMO" in self.id:
            num_active_buyers = np.sum(active_mask)
            # Fear of missing out increases with more bidders
            activity_bonus = max(0, num_active_buyers - 1)
            weights[BID_500] = 1.0 + activity_bonus * 0.5
            weights[BID_1000] = 0.5 + activity_bonus * 1.0
            # Becomes more likely to fold as price increases or people leave
            weights[FOLD] = self.risk_aversion + (price_to_wtp_ratio**2)
            weights[ASK_QUESTION] = self.ask_prob

        else:
            # A generic cautious strategy if persona is not recognized
            weights[FOLD] = 0.5 * price_to_wtp_ratio
            weights[BID_500] = 1 - price_to_wtp_ratio
            weights[BID_1000] = 0.0
            weights[ASK_QUESTION] = 0.1
            
        # --- Final Action Selection ---
        
        action_options = list(weights.keys())
        action_weights = np.array(list(weights.values()))
        
        # Ensure no negative weights and handle edge cases
        action_weights[action_weights < 0] = 0
        if action_weights.sum() == 0:
            return 0  # Default to folding if no valid actions
        
        # Normalize weights to get probabilities
        probabilities = action_weights / action_weights.sum()
        
        # Choose action based on probabilities
        return np.random.choice(action_options, p=probabilities)