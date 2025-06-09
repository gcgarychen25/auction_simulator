"""
Heuristic Policies for Auction Agents

Implements rule-based policies for each agent persona, providing deterministic
behavior based on persona characteristics like risk aversion and ask probability.
"""

import numpy as np
from typing import Dict, List, Any
import random


class HeuristicPolicy:
    """
    Implements deterministic, rule-based logic for auction agents.
    
    Each buyer persona has different behavior patterns based on:
    - Risk aversion level (0.0 to 1.0)
    - Max willingness to pay
    - Probability of asking questions
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the heuristic policy with a random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def get_buyer_action(self, state: Dict[str, np.ndarray], persona: Dict[str, Any], 
                        buyer_idx: int) -> int:
        """
        Generate an action for a buyer based on current state and persona.
        
        Args:
            state: Current environment observation
            persona: Buyer persona configuration
            buyer_idx: Index of the buyer (0-4)
            
        Returns:
            Action integer: 0=fold, 1=bid_500, 2=bid_1000, 3=ask_question
        """
        current_price = float(state['price'][0])
        bids_left = int(state['bids_left'][buyer_idx])
        active = bool(state['active_mask'][buyer_idx])
        round_no = int(state['round_no'][0])
        
        # If not active or no bids left, fold
        if not active or bids_left <= 0:
            return 0  # fold
        
        # Extract persona characteristics
        max_wtp = persona['max_wtp']
        risk_aversion = persona['risk_aversion']
        ask_prob = persona['ask_prob']
        persona_id = persona['id']
        
        # Check if we should ask a question first
        if round_no <= 3 and self.rng.random() < ask_prob:
            return 3  # ask_question
        
        # Persona-specific logic
        if persona_id == "B1_CONSERVATIVE_INVESTOR":
            return self._conservative_strategy(current_price, max_wtp, risk_aversion, bids_left)
            
        elif persona_id == "B2_AGGRESSIVE_TRADER":
            return self._aggressive_strategy(current_price, max_wtp, risk_aversion, bids_left)
            
        elif persona_id == "B3_ANALYTICAL_BUYER":
            return self._analytical_strategy(current_price, max_wtp, risk_aversion, bids_left, round_no)
            
        elif persona_id == "B4_BUDGET_CONSCIOUS":
            return self._budget_conscious_strategy(current_price, max_wtp, risk_aversion, bids_left)
            
        elif persona_id == "B5_FOMO_BIDDER":
            return self._fomo_strategy(current_price, max_wtp, risk_aversion, bids_left, state)
            
        else:
            # Default conservative behavior
            return self._conservative_strategy(current_price, max_wtp, risk_aversion, bids_left)
    
    def _conservative_strategy(self, current_price: float, max_wtp: float, 
                             risk_aversion: float, bids_left: int) -> int:
        """Strategy for conservative investor - prefers small, safe steps."""
        # Fold if price is too close to max WTP
        safety_margin = max_wtp * (0.1 + risk_aversion * 0.2)
        if current_price + 500 > max_wtp - safety_margin:
            return 0  # fold
        
        # Prefer smaller bids
        if current_price + 500 <= max_wtp * 0.8:
            return 1  # bid_500 (small bid)
        elif current_price + 1000 <= max_wtp - safety_margin:
            # Only do large bid if very confident
            if self.rng.random() < (1 - risk_aversion):
                return 2  # bid_1000
            else:
                return 1  # bid_500
        else:
            return 0  # fold
    
    def _aggressive_strategy(self, current_price: float, max_wtp: float, 
                           risk_aversion: float, bids_left: int) -> int:
        """Strategy for aggressive trader - makes large bids to intimidate."""
        # Will bid up to very close to max WTP
        if current_price + 1000 <= max_wtp * 0.95:
            return 2  # bid_1000 (large bid)
        elif current_price + 500 <= max_wtp:
            return 1  # bid_500
        else:
            return 0  # fold
    
    def _analytical_strategy(self, current_price: float, max_wtp: float, 
                           risk_aversion: float, bids_left: int, round_no: int) -> int:
        """Strategy for analytical buyer - gathers info before acting."""
        # Ask questions early in the auction
        if round_no <= 2 and self.rng.random() < 0.7:
            return 3  # ask_question
        
        # Be conservative but strategic
        if current_price < max_wtp * 0.7:
            # Early rounds - small bids
            return 1  # bid_500
        elif current_price < max_wtp * 0.9:
            # Mid rounds - moderate approach
            if bids_left > 1:
                return 1  # bid_500
            else:
                return 2  # bid_1000 (last chance)
        else:
            return 0  # fold
    
    def _budget_conscious_strategy(self, current_price: float, max_wtp: float, 
                                 risk_aversion: float, bids_left: int) -> int:
        """Strategy for budget conscious buyer - strict budget limits."""
        # Very strict - will not exceed max_wtp under any circumstances
        if current_price + 500 > max_wtp:
            return 0  # fold
        elif current_price + 1000 > max_wtp:
            return 1  # bid_500 (only small bid possible)
        else:
            # Choose based on how much budget is left
            remaining_budget = max_wtp - current_price
            if remaining_budget > 2000:
                return 2  # bid_1000
            else:
                return 1  # bid_500
    
    def _fomo_strategy(self, current_price: float, max_wtp: float, 
                      risk_aversion: float, bids_left: int, state: Dict) -> int:
        """Strategy for FOMO bidder - influenced by others' actions."""
        # Count how many other buyers are still active
        active_buyers = int(np.sum(state['active_mask'])) - 1  # Exclude self
        
        # More likely to bid if others are active (FOMO effect)
        fomo_multiplier = 1.0 + (active_buyers * 0.2)
        effective_wtp = min(max_wtp, max_wtp * fomo_multiplier)
        
        if current_price + 1000 <= effective_wtp * 0.9:
            # High activity makes them more aggressive
            if active_buyers >= 3:
                return 2  # bid_1000
            else:
                return 1  # bid_500
        elif current_price + 500 <= effective_wtp:
            return 1  # bid_500
        else:
            return 0  # fold
    
    def get_seller_action(self, state: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Generate an action for the seller based on current state.
        
        Args:
            state: Current environment observation
            info: Additional information from the environment
            
        Returns:
            Action integer: 0=announce_next, 1=answer_question, 2=close_auction
        """
        round_no = int(state['round_no'][0])
        active_buyers = int(np.sum(state['active_mask']))
        
        # Check if there are any questions to answer
        questions = info.get('questions', [])
        if questions:
            return 1  # answer_question
        
        # Close auction if no active buyers or max rounds reached
        if active_buyers == 0 or round_no >= 15:
            return 2  # close_auction
        
        # Check if bidding has stalled (no bids in this round)
        new_bids = info.get('new_bids', [])
        if not new_bids and round_no > 5:
            # Give it one more round, then close
            if self.rng.random() < 0.3:
                return 2  # close_auction
        
        # Default: announce next round
        return 0  # announce_next


def create_heuristic_policies(config: Dict[str, Any], seed: int = 42) -> Dict[str, HeuristicPolicy]:
    """
    Create heuristic policies for all agents.
    
    Args:
        config: Configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of policies for seller and buyers
    """
    policies = {}
    
    # Create seller policy
    policies['seller'] = HeuristicPolicy(seed=seed)
    
    # Create buyer policies (same policy class, different personas)
    policies['buyers'] = []
    for i in range(len(config['buyers'])):
        policy = HeuristicPolicy(seed=seed + i + 1)  # Different seed for each buyer
        policies['buyers'].append(policy)
    
    return policies 