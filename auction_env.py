"""
Multi-Agent Real-Estate Auction Environment

Implements the auction dynamics within a AuctionEnv class that inherits from 
gymnasium.Env, making it compatible with standard RL libraries.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import yaml
from collections import defaultdict
import copy


class AuctionEnv(gym.Env):
    """
    A Gymnasium-compliant environment for real estate auctions.
    
    Features:
    - One seller and 5 buyers with distinct personas
    - English auction format with bidding rounds
    - Natural language interaction via LLM integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the auction environment."""
        super().__init__()
        
        self.config = config
        
        # Core environment parameters from the 'environment' section
        env_config = self.config['environment']
        self.start_price = env_config['auction']['start_price']
        self.max_rounds = env_config['auction']['max_rounds']
        self.bid_limit = env_config['auction']['bid_limit_per_buyer']
        self.reserve_price = env_config['seller']['reserve_price']
        
        # Property Information
        self.property_info = env_config.get('property', {})
        
        # Buyer configurations
        self.num_buyers = len(env_config['buyers'])
        self.base_buyers_config = env_config['buyers']
        
        # Persona variation config from 'phase1_heuristic_settings'
        self.persona_variation_config = self.config.get('phase1_heuristic_settings', {}).get('persona_variation', {})
        
        # Initialize state attributes
        self.reset()
        
        # State variables (reset in reset())
        self.current_price = None
        self.round_no = None
        self.bids_left = None  # Array of remaining bids for each buyer
        self.active_mask = None  # Binary mask of active buyers
        self.last_increment = None
        self.leading_bidder = None  # Index of current high bidder
        self.auction_ended = None
        self.winner = None
        self.final_price = None
        
        # Action history for context
        self.action_history = []
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'round_no': spaces.Box(low=0, high=self.max_rounds, shape=(1,), dtype=np.int32),
            'bids_left': spaces.Box(low=0, high=self.bid_limit, 
                                  shape=(self.num_buyers,), dtype=np.int32),
            'active_mask': spaces.Box(low=0, high=1, shape=(self.num_buyers,), dtype=np.int32),
            'last_increment': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        })
        
        # Define action spaces
        # Buyer actions: 0=fold, 1=bid_500, 2=bid_1000, 3=ask_question
        # Seller actions: 0=announce_next, 1=answer_question, 2=close_auction
        self.buyer_action_space = spaces.Discrete(4)
        self.seller_action_space = spaces.Discrete(3)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        
        # Vary personas at the start of each episode if enabled
        if self.persona_variation_config.get('enabled', False):
            self.buyers_config = self._vary_personas()
            self.varied_personas_info = self.buyers_config
        else:
            self.buyers_config = self.base_buyers_config
            self.varied_personas_info = None
        
        # State variables
        self.current_price = float(self.start_price)
        self.round_no = 0
        self.bids_left = np.array([self.bid_limit] * self.num_buyers, dtype=np.int32)
        self.active_mask = np.ones(self.num_buyers, dtype=np.int32)
        self.last_increment = 0.0
        self.leading_bidder = -1 # Initialize to -1 (no bidder)
        self.winner = None
        self.final_price = None
        self.auction_ended = False
        self.questions_this_round = []
        
        # Clear history
        self.action_history = []
        
        # Return initial observation and info
        obs = self._get_obs()
        info = {
            'auction_just_started': True,
            'buyers_config': self.buyers_config,
            'seller_config': self.config['environment']['seller'],
            'varied_personas': self.varied_personas_info
        }
        return obs, info
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Process one round of actions from seller and buyers.
        
        Args:
            actions: Dict with keys 'seller' and 'buyers' (list of 5 actions)
        
        Returns:
            observation, rewards, terminated, truncated, info
        """
        self.round_no += 1
        info = {}
        
        # Process seller action first
        seller_action = actions.get('seller', 0)
        seller_response = self._process_seller_action(seller_action)
        
        # Process buyer actions
        buyer_actions = actions.get('buyers', [0] * self.num_buyers)
        new_bids, questions = self._process_buyer_actions(buyer_actions)
        
        # Update price and leading bidder if there were bids
        if new_bids:
            highest_bid = max(new_bids, key=lambda x: x[1])  # (buyer_idx, bid_amount)
            self.last_increment = highest_bid[1] - self.current_price
            self.current_price = highest_bid[1]
            self.leading_bidder = highest_bid[0]
        
        # Store action history for context
        self.action_history.append({
            'round': self.round_no,
            'seller_action': seller_action,
            'buyer_actions': buyer_actions,
            'new_price': self.current_price,
            'leading_bidder': self.leading_bidder,
            'questions': questions
        })
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.round_no >= self.max_rounds
        
        if terminated or truncated:
            self.auction_ended = True
            # Only declare winner if reserve price is met AND there's a leading bidder
            if self.leading_bidder >= 0 and self.current_price >= self.reserve_price:
                self.winner = self.leading_bidder
                self.final_price = self.current_price
            else:
                # Auction failed - either no bids or price below reserve
                self.winner = None
                self.final_price = None
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Update info
        info.update({
            'seller_response': seller_response,
            'new_bids': new_bids,
            'questions': questions,
            'auction_ended': self.auction_ended,
            'winner': self.winner,
            'final_price': self.final_price,
            'round_no': self.round_no
        })
        
        observation = self._get_obs()
        
        return observation, rewards, terminated, truncated, info
    
    def _process_seller_action(self, action: int) -> str:
        """Process seller action and return response text."""
        if action == 0:  # announce_next
            return f"Round {self.round_no}: Current price is ${self.current_price:,.0f}. What are your bids?"
        elif action == 1:  # answer_question
            return "Thank you for your question. The property is in excellent condition."
        elif action == 2:  # close_auction
            return "Going once, going twice, sold!"
        else:
            return "Invalid seller action."
    
    def _process_buyer_actions(self, actions: List[int]) -> Tuple[List[Tuple[int, float]], List[Tuple[int, str]]]:
        """
        Process buyer actions and return new bids and questions.
        
        Returns:
            (new_bids, questions) where:
            - new_bids: List of (buyer_idx, bid_amount) tuples
            - questions: List of (buyer_idx, question_text) tuples
        """
        new_bids = []
        questions = []
        
        for buyer_idx, action in enumerate(actions):
            # Skip if buyer is inactive or out of bids
            if not self.active_mask[buyer_idx]:
                continue
                
            if action == 0:  # fold
                self.active_mask[buyer_idx] = 0
                
            elif action == 1:  # bid_500
                if self.bids_left[buyer_idx] > 0:
                    bid_amount = self.current_price + 500
                    new_bids.append((buyer_idx, bid_amount))
                    self.bids_left[buyer_idx] -= 1
                    
            elif action == 2:  # bid_1000
                if self.bids_left[buyer_idx] > 0:
                    bid_amount = self.current_price + 1000
                    new_bids.append((buyer_idx, bid_amount))
                    self.bids_left[buyer_idx] -= 1
                    
            elif action == 3:  # ask_question
                persona_id = self.base_buyers_config[buyer_idx]['id']
                questions.append((buyer_idx, f"Question from {persona_id}: What is the condition of the property?"))
        
        return new_bids, questions
    
    def _check_termination(self) -> bool:
        """Check if auction should terminate."""
        # Auction ends if no buyers are active
        if np.sum(self.active_mask) == 0:
            return True
            
        # Auction ends if all active buyers are out of bids
        active_buyers_with_bids = np.sum(self.active_mask * (self.bids_left > 0))
        if active_buyers_with_bids == 0:
            return True
            
        return False
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {
            'seller': 0.0,
            'buyers': [0.0] * self.num_buyers
        }
        
        if self.auction_ended and self.winner is not None:
            # Winning buyer reward: max_wtp - final_price
            winning_buyer_wtp = self.base_buyers_config[self.winner]['max_wtp']
            rewards['buyers'][self.winner] = winning_buyer_wtp - self.final_price
            
            # Seller reward: final_price - reserve_price
            rewards['seller'] = self.final_price - self.reserve_price
        
        return rewards
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Compile current state into observation dictionary."""
        return {
            'price': np.array([self.current_price], dtype=np.float32),
            'round_no': np.array([self.round_no], dtype=np.int32),
            'bids_left': self.bids_left.copy(),
            'active_mask': self.active_mask.copy(),
            'last_increment': np.array([self.last_increment], dtype=np.float32),
        }
    
    def get_action_history(self) -> List[Dict]:
        """Get the action history for context in prompts."""
        return self.action_history.copy()
    
    def get_buyer_personas(self) -> List[Dict]:
        """Get buyer persona configurations."""
        return self.base_buyers_config.copy()

    def _vary_personas(self) -> List[Dict[str, Any]]:
        """
        Vary the personas of the buyers based on the persona_variation_config.
        
        Returns:
            List of varied buyer configurations
        """
        varied_buyers = []
        for buyer in self.base_buyers_config:
            varied_buyer = copy.deepcopy(buyer)
            for variation, config in self.persona_variation_config.items():
                if variation in buyer:
                    varied_buyer[variation] = config.get(buyer[variation], buyer[variation])
            varied_buyers.append(varied_buyer)
        return varied_buyers

    def get_property_info(self) -> Dict[str, Any]:
        """Returns the property information dictionary."""
        return self.property_info


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) 