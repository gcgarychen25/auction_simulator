"""
Reinforcement Learning Policy Manager with PPO and GAE

Implements a manager for creating, training, and coordinating multiple independent
RL agents using Proximal Policy Optimization (PPO) with Generalized Advantage
Estimation (GAE).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Any
import logging
from pathlib import Path

from .networks import ActorCritic

# Configure logging
logger = logging.getLogger(__name__)

class RLPolicyManager:
    """
    Manages multiple independent PPO agents for each buyer.
    """
    def __init__(self, config: Dict[str, Any], training_mode: bool = True):
        self.config = config
        self.training_mode = training_mode
        self.buyers = config['environment']['buyers']
        self.num_buyers = len(self.buyers)
        self.observation_dim = 8  # Expanded features
        self.action_dim = 4      # Fold, Bid 500, Bid 1000, Ask
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PPO Hyperparameters from config
        ppo_config = config.get('phase2_rl_settings', {})
        self.gamma = ppo_config.get('gamma', 0.99)
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.lr = ppo_config.get('lr', 0.0003)
        self.clip_epsilon = ppo_config.get('clip_epsilon', 0.2)
        self.ppo_epochs = ppo_config.get('ppo_epochs', 10)
        self.batch_size = ppo_config.get('batch_size', 32)
        self.vf_coef = ppo_config.get('vf_coef', 0.5)
        self.ent_coef = ppo_config.get('ent_coef', 0.01)
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        self.seed = ppo_config.get('seed', 42)

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.policies = [ActorCritic(self.observation_dim, self.action_dim).to(self.device) for _ in range(self.num_buyers)]
        self.optimizers = [optim.Adam(policy.parameters(), lr=self.lr) for policy in self.policies]
        
        self.memory = [self._create_memory_buffer() for _ in range(self.num_buyers)]
        self.training_history = {'episode_rewards': [[] for _ in range(self.num_buyers)], 'performance_metrics': []}
        
        logger.info(f"🤖 PPO PolicyManager initialized with {self.num_buyers} agents on {self.device}")

    def _create_memory_buffer(self):
        return {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}

    def _get_state_tensor(self, observation: Dict, buyer_idx: int) -> torch.Tensor:
        """Constructs a feature-rich state tensor."""
        # Use varied personas if they exist for the current episode
        if 'varied_personas' in observation and observation['varied_personas']:
            persona = observation['varied_personas'][buyer_idx]
        else:
            persona = self.buyers[buyer_idx]

        price = observation['price'][0]
        max_wtp = persona.get('max_wtp', 1)
        
        # Calculate headroom and price ratio
        price_ratio = min(price / max_wtp, 2.0) if max_wtp > 0 else 2.0  # Cap at 2.0
        headroom = max(max_wtp - price, 0) / max_wtp if max_wtp > 0 else 0.0
        
        state = [
            price / 1_000_000,  # Normalize price
            observation['round_no'][0] / 20.0, # Normalize round
            observation['bids_left'][buyer_idx] / 5.0, # Normalize bids left
            observation['active_mask'][buyer_idx],
            observation['last_increment'][0] / 5000.0, # Normalize increment
            price_ratio,
            headroom,
            (sum(observation['active_mask']) - observation['active_mask'][buyer_idx]) / (self.num_buyers -1) if self.num_buyers > 1 else 0
        ]
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_buyer_action(self, observation: Dict, buyer_idx: int) -> int:
        """Select an action for a buyer and store experience if training."""
        if not observation['active_mask'][buyer_idx]:
             return 0  # Inactive agents must fold

        state_tensor = self._get_state_tensor(observation, buyer_idx)
        policy = self.policies[buyer_idx]
        
        # Action masking
        action_mask = torch.ones(self.action_dim, device=self.device)
        if observation['bids_left'][buyer_idx] == 0:
            action_mask[1] = 0 # Cannot bid 500
            action_mask[2] = 0 # Cannot bid 1000
            
        with torch.no_grad():
            action, log_prob, entropy = policy.get_action(state_tensor, action_mask)
        
        # We need the value for GAE calculation
        _, state_value = policy(state_tensor)

        if self.training_mode:
            self.memory[buyer_idx]['states'].append(state_tensor)
            self.memory[buyer_idx]['actions'].append(torch.tensor([action], device=self.device))
            self.memory[buyer_idx]['log_probs'].append(log_prob)
            self.memory[buyer_idx]['values'].append(state_value)

        return action

    def record_round_results(self, terminated: bool, truncated: bool):
        """Record rewards and done flags for each agent after a round."""
        if not self.training_mode:
            return
        
        done = terminated or truncated
        for i in range(self.num_buyers):
            # Only record results for agents that took an action this round
            if len(self.memory[i]['values']) > len(self.memory[i]['rewards']):
                # Placeholder reward, to be filled at episode end. Dones are for THIS step.
                self.memory[i]['rewards'].append(0.0) 
                self.memory[i]['dones'].append(done)

    def finalize_episode_and_update(self, final_info: Dict, final_rewards: Dict):
        """At the end of an episode, compute GAE and update PPO policies."""
        if not self.training_mode:
            return
            
        for buyer_idx in range(self.num_buyers):
            # Skip update if agent took no actions
            if not self.memory[buyer_idx]['states']:
                continue
            
            # --- Calculate a detailed, persona-specific reward ---
            final_reward = self._calculate_persona_reward(buyer_idx, final_info, final_rewards)
            self.training_history['episode_rewards'][buyer_idx].append(final_reward)

            # --- Assign Final Reward to Memory ---
            # The final reward is assigned to the last step of the episode
            if self.memory[buyer_idx]['rewards']:
                self.memory[buyer_idx]['rewards'][-1] = final_reward
            
            # --- Compute GAE & Update ---
            advantages, returns = self._compute_gae(buyer_idx)
            self._update_ppo_policy(buyer_idx, advantages, returns)
            
            # --- Clear Memory for Next Episode ---
            self.memory[buyer_idx] = self._create_memory_buffer()

    def _calculate_persona_reward(self, buyer_idx: int, final_info: Dict, final_rewards: Dict) -> float:
        """Calculates the final episode reward for an agent based on its persona."""
        persona = self.buyers[buyer_idx]
        max_wtp = persona['max_wtp']
        
        winner_idx = final_info.get('winner')
        is_winner = (buyer_idx == winner_idx)
        
        # 1. Primary Reward: Individual Surplus
        # This is the main driver for agent behavior.
        if is_winner:
            # Winner's reward is their direct surplus
            primary_reward = final_rewards['buyers'][winner_idx]
        else:
            # Non-winners get a small penalty for losing to incentivize winning
            primary_reward = -100  

        # 2. Shared Economic Welfare Reward (for all participants)
        # Encourages agents to cooperate to make a successful auction happen
        if winner_idx is not None:
            total_market_surplus = final_rewards.get('seller', 0) + final_rewards['buyers'][winner_idx]
            shared_reward = total_market_surplus * 0.1 # All agents get 10% of total surplus created
        else:
            shared_reward = -200 # Larger penalty if the auction fails entirely

        # 3. Persona-Specific "Style" Bonus (small nudge, only for the winner)
        style_bonus = 0
        if is_winner:
            actions_taken = [a.item() for a in self.memory[buyer_idx]['actions']]
            persona_type = persona['id'].split('_')[1]

            if persona_type == "AGGRESSIVE" and 2 in actions_taken:
                style_bonus = 50
            elif persona_type == "ANALYTICAL" and 3 in actions_taken:
                style_bonus = 75
            
        total_reward = primary_reward + shared_reward + style_bonus
        return total_reward
    
    def _compute_gae(self, buyer_idx: int) -> (torch.Tensor, torch.Tensor):
        """Compute Generalized Advantage Estimation."""
        memory = self.memory[buyer_idx]
        rewards = memory['rewards']
        values = torch.cat(memory['values']).squeeze(-1).detach()
        dones = memory['dones']
        
        advantages = []
        last_advantage = 0
        
        # Get the value of the state that comes *after* the last action
        with torch.no_grad():
            if dones[-1]:
                next_value = 0
            else:
                 # Bootstrap from the value of the final state
                _, next_value_tensor = self.policies[buyer_idx](memory['states'][-1])
                next_value = next_value_tensor.item()

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t].item()
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            advantages.insert(0, last_advantage)
            next_value = values[t].item()
            
        returns = [adv + val.item() for adv, val in zip(advantages, values)]
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages only if there's more than one sample to avoid NaN from std()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update_ppo_policy(self, buyer_idx: int, advantages: torch.Tensor, returns: torch.Tensor):
        """Perform PPO update for a single agent."""
        memory = self.memory[buyer_idx]
        states = torch.cat(memory['states']).squeeze(1)
        actions = torch.cat(memory['actions'])
        old_log_probs = torch.stack(memory['log_probs']).detach()
        
        policy = self.policies[buyer_idx]
        optimizer = self.optimizers[buyer_idx]
        
        for _ in range(self.ppo_epochs):
            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch from data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Evaluate current policy on batch
                log_probs, state_values, entropy = policy.evaluate_action(batch_states, batch_actions)
                
                # Calculate PPO ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Calculate surrogate objectives
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # Calculate losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (state_values.squeeze() - batch_returns).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
                
                # Update policy
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                optimizer.step()

    def save_models(self, directory: str = "rl_models"):
        """Save trained models and optimizers for all agents."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for i, policy in enumerate(self.policies):
            torch.save(policy.state_dict(), path / f"buyer_{i}_policy_{self.buyers[i]['id']}.pth")
        logger.info(f"💾 Models saved to {directory}")

    def load_models(self, directory: str = "rl_models"):
        """Load pre-trained models for all agents."""
        path = Path(directory)
        if not path.exists():
            logger.warning(f"⚠️  Could not load models: Directory '{directory}' not found.")
            return

        for i, policy in enumerate(self.policies):
            # Use the same naming pattern as save_models to locate checkpoints
            model_path = path / f"buyer_{i}_policy_{self.buyers[i]['id']}.pth"
            if model_path.exists():
                policy.load_state_dict(torch.load(model_path, map_location=self.device))
                policy.eval()  # Set to evaluation mode
            else:
                logger.warning(f"⚠️  Model for buyer {i} not found at {model_path}")
        logger.info(f"✅ Models loaded from {directory}")

    def save_training_history(self, directory: str = "rl_models"):
        """Save training history (rewards, etc.) to a file."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        history_file = path / "training_history.json"
        
        # Convert tensors in performance metrics to serializable format
        history_to_save = {
            'episode_rewards': self.training_history['episode_rewards'],
            'performance_metrics': []
        }
        for perf_record in self.training_history.get('performance_metrics', []):
            serializable_record = {'episode': perf_record['episode'], 'metrics': {}}
            for group, metrics in perf_record['metrics'].items():
                serializable_record['metrics'][group] = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
            history_to_save['performance_metrics'].append(serializable_record)

        with open(history_file, 'w') as f:
            json.dump(history_to_save, f, indent=4)
        logger.info(f"📊 Training history saved to {history_file}")
        
    def get_seller_action(self, observation: Dict, info: Dict) -> int:
        """
        Provides a basic heuristic action for the seller.
        This RL manager is focused on buyer agents, so seller logic is simple.
        """
        current_price = observation['price'][0]
        reserve_price = self.config['environment']['seller']['reserve_price']
        active_buyers = sum(observation['active_mask'])
        
        # If reserve is met and few buyers are left, consider closing.
        if current_price >= reserve_price and active_buyers <= 1:
            return 2 # Close auction
        
        # Otherwise, just continue.
        return 0 # Announce next round 

    def update_config(self, new_config: Dict[str, Any]):
        """
        Updates the manager's configuration.
        This is crucial when the environment changes between episodes during training.
        """
        self.config = new_config

    def _initialize_policies(self) -> Dict[str, nn.Module]:
        """Initializes the neural network policies for all agents."""
        policies = {}
        # Assuming obs_size and action_size can be determined from config
        # ... existing code ... 