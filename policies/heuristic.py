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
    
    def __init__(self, seed: int = 42, config: Dict[str, Any] = None):
        """Initialize the heuristic policy with a random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        self.config = config
        
        # Persona variation cache (reset per episode)
        self.persona_variations = {}
        self.current_episode = -1
    
    def _get_varied_persona(self, persona: Dict[str, Any], buyer_idx: int, round_no: int) -> Dict[str, Any]:
        """
        Apply configurable variations to persona parameters.
        Variations are consistent within an episode but change between episodes.
        """
        # Detect new episode (when round resets to 1)
        if round_no == 1 or self.current_episode == -1:
            self.current_episode += 1
            self.persona_variations = {}  # Reset variations for new episode
        
        # Check if persona variation is disabled
        if not self.config or not self.config.get('persona_variation', {}).get('enabled', False):
            return persona  # Return original persona unchanged
        
        # Cache key for this persona in this episode
        cache_key = f"{persona['id']}_{self.current_episode}"
        
        if cache_key not in self.persona_variations:
            # Generate new variations for this episode
            variation_config = self.config['persona_variation']
            
            # Vary max_wtp (±8% by default)
            wtp_variance = variation_config.get('max_wtp_variance', 0.08)
            wtp_multiplier = 1.0 + self.rng.uniform(-wtp_variance, wtp_variance)
            varied_max_wtp = int(persona['max_wtp'] * wtp_multiplier)
            
            # Vary risk_aversion (±0.15 by default, clamped to 0.0-1.0)
            risk_variance = variation_config.get('risk_aversion_variance', 0.15)
            risk_delta = self.rng.uniform(-risk_variance, risk_variance)
            varied_risk_aversion = max(0.0, min(1.0, persona['risk_aversion'] + risk_delta))
            
            # Vary ask_prob (±10% relative by default, clamped to 0.0-1.0)
            ask_variance = variation_config.get('ask_prob_variance', 0.10)
            ask_multiplier = 1.0 + self.rng.uniform(-ask_variance, ask_variance)
            varied_ask_prob = max(0.0, min(1.0, persona['ask_prob'] * ask_multiplier))
            
            # Create varied persona
            varied_persona = persona.copy()
            varied_persona.update({
                'max_wtp': varied_max_wtp,
                'risk_aversion': varied_risk_aversion,
                'ask_prob': varied_ask_prob,
                # Store original values for reference
                '_original_max_wtp': persona['max_wtp'],
                '_original_risk_aversion': persona['risk_aversion'],
                '_original_ask_prob': persona['ask_prob']
            })
            
            self.persona_variations[cache_key] = varied_persona
        
        return self.persona_variations[cache_key]
    
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
        
        # Apply persona variations (if enabled)
        varied_persona = self._get_varied_persona(persona, buyer_idx, round_no)
        
        # Extract (potentially varied) persona characteristics
        max_wtp = varied_persona['max_wtp']
        risk_aversion = varied_persona['risk_aversion']
        ask_prob = varied_persona['ask_prob']
        persona_id = varied_persona['id']
        
        # Dynamic question asking influenced by ask_prob, price pressure, and persona
        price_pressure = current_price / max_wtp
        question_motivation = ask_prob * (1 + price_pressure * 0.5)  # More questions as price rises
        
        # Adjust question probability based on risk aversion and round
        if round_no <= 3:
            risk_adjusted_ask_prob = question_motivation * (1 + risk_aversion * 0.3)
            if self.rng.random() < risk_adjusted_ask_prob:
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
        """Strategy for conservative investor - prefers small, safe steps with cautious randomness."""
        # Economic rationality check: ensure minimum surplus
        min_surplus_required = max_wtp * (0.05 + risk_aversion * 0.1)  # Conservative requires good surplus
        
        # Dynamic safety margin based on risk aversion and some randomness
        base_safety = 0.1 + risk_aversion * 0.2
        safety_noise = self.rng.uniform(-0.05, 0.05)  # ±5% randomness
        safety_margin = max_wtp * max(0.05, base_safety + safety_noise)
        
        # Add hesitation probability based on risk aversion
        hesitation_prob = risk_aversion * 0.3
        if self.rng.random() < hesitation_prob:
            return 0  # Conservative hesitation - sometimes fold even when could bid
        
        # Economic rationality: fold if surplus would be insufficient
        if current_price + 500 >= max_wtp - min_surplus_required:
            return 0  # fold - surplus too small
            
        # Also fold if price is too close to max WTP (safety margin)
        if current_price + 500 > max_wtp - safety_margin:
            return 0  # fold
        
        # Probabilistic bidding based on comfort level
        price_comfort = (max_wtp - current_price) / max_wtp
        
        if current_price + 500 <= max_wtp * 0.8:
            # Early stage - mostly small bids but some randomness
            if self.rng.random() < 0.8 + (1 - risk_aversion) * 0.15:
                return 1  # bid_500 (small bid)
            else:
                return 0  # Conservative fold
        elif current_price + 1000 <= max_wtp - safety_margin:
            # Mid stage - risk-aversion influences decision
            large_bid_prob = (1 - risk_aversion) * price_comfort * 0.6
            if self.rng.random() < large_bid_prob:
                return 2  # bid_1000
            elif self.rng.random() < 0.7:
                return 1  # bid_500
            else:
                return 0  # fold
        else:
            return 0  # fold
    
    def _aggressive_strategy(self, current_price: float, max_wtp: float, 
                           risk_aversion: float, bids_left: int) -> int:
        """Strategy for aggressive trader - makes large bids to intimidate with strategic variation."""
        # Economic rationality: even aggressive traders need some surplus
        min_surplus_required = max_wtp * (0.02 + risk_aversion * 0.05)  # Aggressive accepts smaller surplus
        
        # Aggressive threshold varies based on confidence and remaining bids
        confidence = (1 - risk_aversion) * (bids_left / 3.0)  # More confident with more bids
        aggressive_threshold = 0.92 + self.rng.uniform(-0.05, 0.08)  # 87-100% of max_wtp
        
        # Economic check: don't bid if surplus would be too small
        if current_price + 1000 >= max_wtp - min_surplus_required:
            if current_price + 500 >= max_wtp - min_surplus_required:
                return 0  # fold - even small bid leaves insufficient surplus
            # Only allow small bid if it preserves minimum surplus
        
        # Sometimes go all-in early to intimidate (influenced by low risk aversion)
        early_aggression_prob = (1 - risk_aversion) * 0.4
        if current_price <= max_wtp * 0.7 and self.rng.random() < early_aggression_prob:
            # Strategic large bid to establish dominance
            return 2  # bid_1000
        
        # Core aggressive logic with randomness
        if current_price + 1000 <= max_wtp * aggressive_threshold:
            # Vary between large and medium bids based on confidence
            large_bid_prob = 0.7 + confidence * 0.2
            if self.rng.random() < large_bid_prob:
                return 2  # bid_1000 (large bid)
            else:
                return 1  # bid_500 (still aggressive but smaller)
        elif current_price + 500 <= max_wtp * (0.98 + self.rng.uniform(-0.02, 0.02)):
            # Sometimes push to very edge of max_wtp
            final_push_prob = (1 - risk_aversion) * 0.8
            if self.rng.random() < final_push_prob:
                return 1  # bid_500
            else:
                return 0  # Rare aggressive fold (strategic retreat)
        else:
            return 0  # fold
    
    def _analytical_strategy(self, current_price: float, max_wtp: float, 
                           risk_aversion: float, bids_left: int, round_no: int) -> int:
        """Strategy for analytical buyer - gathers info before acting with calculated risks."""
        # Dynamic question asking based on ask_prob and auction progress
        price_ratio = current_price / max_wtp
        question_urgency = min(1.0, price_ratio * 2)  # More urgent as price approaches max_wtp
        
        # Ask questions with dynamic probability
        if round_no <= 3:
            base_ask_prob = 0.85  # High analytical tendency
            adjusted_ask_prob = base_ask_prob * (1 + question_urgency * 0.3)
            if self.rng.random() < adjusted_ask_prob:
                return 3  # ask_question
        
        # Analytical decision making with calculated risks
        market_position = current_price / max_wtp
        confidence_factor = (1 - risk_aversion) * (1 - market_position)
        
        if current_price < max_wtp * 0.65:
            # Early stage - careful entry with some randomness
            entry_prob = 0.6 + confidence_factor * 0.3
            if self.rng.random() < entry_prob:
                return 1  # bid_500
            else:
                return 0  # Wait and analyze more
                
        elif current_price < max_wtp * (0.85 + self.rng.uniform(-0.05, 0.05)):
            # Mid stage - strategic bidding based on remaining bids and confidence
            if bids_left > 1:
                strategic_bid_prob = 0.7 + confidence_factor * 0.25
                if self.rng.random() < strategic_bid_prob:
                    # Choose bid size based on analysis
                    if self.rng.random() < 0.3 + confidence_factor * 0.4:
                        return 2  # bid_1000 (confident analytical play)
                    else:
                        return 1  # bid_500 (conservative analytical approach)
                else:
                    return 0  # Analytical patience
            else:
                # Last bid - make it count
                final_commitment_prob = confidence_factor * 0.8 + 0.3
                if self.rng.random() < final_commitment_prob:
                    return 2  # bid_1000 (last chance)
                else:
                    return 1  # bid_500 (conservative final bid)
        else:
            # Late stage - analytical risk assessment
            late_stage_prob = confidence_factor * 0.5
            if self.rng.random() < late_stage_prob:
                return 1  # bid_500 (calculated late entry)
            else:
                return 0  # fold (analytical discipline)
    
    def _budget_conscious_strategy(self, current_price: float, max_wtp: float, 
                                 risk_aversion: float, bids_left: int) -> int:
        """Strategy for budget conscious buyer - strict budget limits with smart allocation."""
        # Strict budget adherence but with some strategic randomness in allocation
        remaining_budget = max_wtp - current_price
        budget_utilization = current_price / max_wtp
        
        # Anxiety increases as budget gets tight (influenced by risk aversion)
        budget_anxiety = risk_aversion * budget_utilization
        
        # Very strict - will not exceed max_wtp under any circumstances
        if current_price + 500 > max_wtp:
            return 0  # fold
            
        elif current_price + 1000 > max_wtp:
            # Only small bid possible - but add hesitation based on anxiety
            final_bid_hesitation = budget_anxiety * 0.4
            if self.rng.random() < final_bid_hesitation:
                return 0  # Budget anxiety causes fold
            else:
                return 1  # bid_500 (only small bid possible)
        else:
            # Strategic budget allocation with randomness
            conservative_threshold = 1500 + self.rng.uniform(-200, 300)  # Some variation in threshold
            
            if remaining_budget > conservative_threshold:
                # Early stage - more willing to make larger bids
                large_bid_comfort = (1 - budget_anxiety) * 0.7
                if self.rng.random() < large_bid_comfort:
                    return 2  # bid_1000
                else:
                    # Sometimes conservative even with budget available
                    small_bid_prob = 0.6 + risk_aversion * 0.3
                    if self.rng.random() < small_bid_prob:
                        return 1  # bid_500 (budget conscious choice)
                    else:
                        return 0  # Save budget for later
            else:
                # Tight budget - very conservative
                frugal_bid_prob = (1 - budget_anxiety) * 0.5 + 0.2
                if self.rng.random() < frugal_bid_prob:
                    return 1  # bid_500 (stretch the budget)
                else:
                    return 0  # fold (budget discipline)
    
    def _fomo_strategy(self, current_price: float, max_wtp: float, 
                      risk_aversion: float, bids_left: int, state: Dict) -> int:
        """Strategy for FOMO bidder - highly influenced by market activity and emotions."""
        # Count how many other buyers are still active
        active_buyers = int(np.sum(state['active_mask'])) - 1  # Exclude self
        round_no = int(state['round_no'][0])
        
        # FOMO intensity increases with market activity and round progression
        fomo_intensity = (active_buyers / 4.0) * (1 - risk_aversion)
        time_pressure = min(1.0, round_no / 10.0)  # Increases over time
        
        # Emotional decision making - FOMO can override rational limits
        emotional_boost = fomo_intensity * (0.1 + self.rng.uniform(0, 0.15))
        effective_wtp = max_wtp * (1 + emotional_boost)
        
        # Panic bidding in high activity situations
        panic_threshold = 0.7 + fomo_intensity * 0.3
        if active_buyers >= 3 and self.rng.random() < panic_threshold:
            # High FOMO - might make irrational bids
            if current_price + 1000 <= effective_wtp and bids_left > 0:
                return 2  # bid_1000 (FOMO panic bid)
        
        # Fear of missing out on early opportunities
        early_fomo_prob = fomo_intensity * 0.6 + time_pressure * 0.2
        if round_no <= 3 and active_buyers >= 2:
            if self.rng.random() < early_fomo_prob:
                if current_price + 1000 <= effective_wtp * 0.95:
                    return 2  # bid_1000 (early FOMO)
                elif current_price + 500 <= effective_wtp:
                    return 1  # bid_500 (moderate FOMO)
        
        # Standard FOMO logic with emotional volatility
        if current_price + 1000 <= effective_wtp * (0.85 + self.rng.uniform(0, 0.1)):
            # Activity-based bidding decision
            activity_aggression = min(0.9, 0.4 + active_buyers * 0.15 + fomo_intensity * 0.3)
            if self.rng.random() < activity_aggression:
                # Choose bid size based on FOMO level
                if active_buyers >= 3 or self.rng.random() < fomo_intensity:
                    return 2  # bid_1000 (high FOMO)
                else:
                    return 1  # bid_500 (moderate FOMO)
            else:
                # Sometimes resist FOMO
                return 0  # fold (rare FOMO resistance)
                
        elif current_price + 500 <= effective_wtp:
            # Last chance FOMO
            last_chance_prob = fomo_intensity * 0.8 + time_pressure * 0.4
            if self.rng.random() < last_chance_prob:
                return 1  # bid_500 (last chance FOMO)
            else:
                return 0  # fold (overcome FOMO)
        else:
            # Even FOMO has limits (sometimes)
            desperation_prob = fomo_intensity * time_pressure * 0.3
            if self.rng.random() < desperation_prob and current_price + 500 <= max_wtp * 1.05:
                return 1  # bid_500 (desperate FOMO - slight overpay)
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
    policies['seller'] = HeuristicPolicy(seed=seed, config=config)
    
    # Create buyer policies (same policy class, different personas)
    policies['buyers'] = []
    for i in range(len(config['buyers'])):
        policy = HeuristicPolicy(seed=seed + i + 1, config=config)  # Different seed for each buyer
        policies['buyers'].append(policy)
    
    return policies 