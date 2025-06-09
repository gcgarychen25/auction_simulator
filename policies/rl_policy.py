"""
RL Policy Wrapper for Auction Simulator

Wraps a trained Stable Baselines 3 model for RL-based action selection.
This is used in Phase 2 of the project to compare against heuristic baselines.
"""

import numpy as np
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
    SB3_AVAILABLE = True
except ImportError:
    logger.warning("Stable Baselines 3 not installed. RL policies will use fallback behavior.")
    SB3_AVAILABLE = False
    BaseAlgorithm = object  # Fallback for type hints


class RLPolicy:
    """
    A wrapper for a trained RL model.
    
    This class loads a pre-trained model and provides a consistent interface
    for action selection, making it easy to swap between heuristic and RL policies.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "PPO"):
        """
        Initialize the RL policy wrapper.
        
        Args:
            model_path: Path to the saved model file (.zip format)
            model_type: Type of RL algorithm ("PPO", "A2C", etc.)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, model_type)
        else:
            logger.warning(f"Model path {model_path} not found. Using fallback behavior.")
    
    def load_model(self, model_path: str, model_type: str = "PPO"):
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the model file
            model_type: Type of RL algorithm
        """
        if not SB3_AVAILABLE:
            logger.error("Cannot load RL model: Stable Baselines 3 not installed")
            return
        
        try:
            if model_type.upper() == "PPO":
                self.model = PPO.load(model_path)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return
            
            self.is_loaded = True
            logger.info(f"Successfully loaded {model_type} model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model = None
            self.is_loaded = False
    
    def get_action(self, observation: Dict[str, np.ndarray], deterministic: bool = True) -> int:
        """
        Get an action from the RL model.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action integer
        """
        if not self.is_loaded or self.model is None:
            # Fallback to conservative action
            logger.warning("RL model not available, using fallback action")
            return self._fallback_action(observation)
        
        try:
            # Convert observation to format expected by model
            obs_array = self._process_observation(observation)
            
            # Get action from model
            action, _ = self.model.predict(obs_array, deterministic=deterministic)
            
            # Ensure action is valid integer
            if isinstance(action, np.ndarray):
                action = int(action[0])
            else:
                action = int(action)
            
            return action
            
        except Exception as e:
            logger.error(f"Error getting action from RL model: {e}")
            return self._fallback_action(observation)
    
    def _process_observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert dictionary observation to flat array format expected by RL model.
        
        Args:
            observation: Dictionary observation from environment
            
        Returns:
            Flattened numpy array
        """
        # Concatenate all observation components
        obs_parts = []
        
        # Add each component in consistent order
        obs_parts.append(observation['price'].flatten())
        obs_parts.append(observation['round_no'].flatten().astype(np.float32))
        obs_parts.append(observation['bids_left'].flatten().astype(np.float32))
        obs_parts.append(observation['active_mask'].flatten().astype(np.float32))
        obs_parts.append(observation['last_increment'].flatten())
        
        # Concatenate into single array
        flat_obs = np.concatenate(obs_parts)
        
        return flat_obs
    
    def _fallback_action(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Provide a fallback action when RL model is not available.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Conservative fallback action
        """
        # Simple fallback logic - be conservative
        current_price = float(observation['price'][0])
        round_no = int(observation['round_no'][0])
        
        # Early rounds: small bid
        if round_no <= 3 and current_price < 12000:
            return 1  # bid_500
        # Later rounds: be more conservative
        elif current_price < 11000:
            return 1  # bid_500
        else:
            return 0  # fold
    
    def save_model(self, save_path: str):
        """
        Save the current model to file.
        
        Args:
            save_path: Path where to save the model
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        try:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


class RLPolicyManager:
    """
    Manages multiple RL policies for different agents.
    
    This class handles loading and managing separate models for
    the seller and each buyer agent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RL policy manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.seller_policy = None
        self.buyer_policies = []
        
    def load_policies(self, model_dir: str):
        """
        Load RL policies for all agents from a directory.
        
        Args:
            model_dir: Directory containing saved model files
        """
        # Load seller policy
        seller_model_path = os.path.join(model_dir, "seller_model.zip")
        self.seller_policy = RLPolicy(seller_model_path)
        
        # Load buyer policies
        self.buyer_policies = []
        num_buyers = len(self.config['buyers'])
        
        for i in range(num_buyers):
            buyer_model_path = os.path.join(model_dir, f"buyer_{i}_model.zip")
            buyer_policy = RLPolicy(buyer_model_path)
            self.buyer_policies.append(buyer_policy)
    
    def get_seller_action(self, observation: Dict[str, np.ndarray]) -> int:
        """Get action for seller."""
        if self.seller_policy:
            return self.seller_policy.get_action(observation)
        else:
            return 0  # Default: announce next round
    
    def get_buyer_action(self, observation: Dict[str, np.ndarray], buyer_idx: int) -> int:
        """Get action for a specific buyer."""
        if buyer_idx < len(self.buyer_policies) and self.buyer_policies[buyer_idx]:
            return self.buyer_policies[buyer_idx].get_action(observation)
        else:
            return 0  # Default: fold


# TODO: For Phase 2 - Training utilities
def train_rl_policies(config: Dict[str, Any], save_dir: str, total_timesteps: int = 100000):
    """
    Train RL policies for all agents.
    
    This function will be implemented in Phase 2 to train PPO agents
    on the auction environment.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save trained models
        total_timesteps: Number of training timesteps
    """
    logger.info("RL policy training not yet implemented - this is for Phase 2")
    logger.info("Current Phase 0 focuses on heuristic policies only")
    
    # TODO: Implement multi-agent training loop
    # 1. Create auction environment
    # 2. Initialize PPO agents
    # 3. Train agents in parallel
    # 4. Save trained models to save_dir
    
    pass 