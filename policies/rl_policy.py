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

# Configure logging
logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Contains a shared feature backbone, a policy head (actor), and a value head (critic).
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.shared_layer(x)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_probs, state_value

class RLPolicyManager:
    """
    Manages multiple independent PPO agents for each buyer.
    """
    def __init__(self, config: Dict[str, Any], training_mode: bool = True):
        self.config = config
        self.training_mode = training_mode
        self.buyers = config['buyers']
        self.num_buyers = len(self.buyers)
        self.observation_dim = 8  # Expanded features
        self.action_dim = 4      # Fold, Bid 500, Bid 1000, Ask
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PPO Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.batch_size = 32

        self.policies = [ActorCritic(self.observation_dim, self.action_dim).to(self.device) for _ in range(self.num_buyers)]
        self.optimizers = [optim.Adam(policy.parameters(), lr=0.0003) for policy in self.policies]
        
        self.memory = [self._create_memory_buffer() for _ in range(self.num_buyers)]
        self.training_history = {'episode_rewards': [[] for _ in range(self.num_buyers)], 'performance_metrics': []}
        
        logger.info(f"🤖 PPO PolicyManager initialized with {self.num_buyers} agents on {self.device}")

    def _create_memory_buffer(self):
        return {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}

    def _get_state_tensor(self, observation: Dict, buyer_idx: int) -> torch.Tensor:
        """Constructs a feature-rich state tensor."""
        persona = self.buyers[buyer_idx]
        price = observation['price'][0]
        wtp = persona.get('max_wtp', 1)
        
        state = [
            price,
            observation['round_no'][0],
            observation['bids_left'][buyer_idx],
            observation['active_mask'][buyer_idx],
            observation['last_increment'][0],
            min(price / wtp, 1.0) if wtp > 0 else 1.0,
            max(wtp - price, 0) / wtp if wtp > 0 else 0,
            sum(observation['active_mask']) - observation['active_mask'][buyer_idx]
        ]
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_buyer_action(self, observation: Dict, buyer_idx: int) -> int:
        """Select an action for a buyer and store experience if training."""
        if not observation['active_mask'][buyer_idx]:
             return 0  # Inactive agents must fold

        state_tensor = self._get_state_tensor(observation, buyer_idx)
        policy = self.policies[buyer_idx]
        
        with torch.no_grad():
            action_probs, state_value = policy(state_tensor)
        
        m = Categorical(action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        if self.training_mode:
            self.memory[buyer_idx]['states'].append(state_tensor)
            self.memory[buyer_idx]['actions'].append(action)
            self.memory[buyer_idx]['log_probs'].append(log_prob)
            self.memory[buyer_idx]['values'].append(state_value)

        return action.item()

    def record_round_results(self, terminated: bool, truncated: bool):
        """Record rewards and done flags for each agent after a round."""
        if not self.training_mode:
            return
        
        done = terminated or truncated
        for i in range(self.num_buyers):
            # Only record results for agents that took an action this round
            if len(self.memory[i]['values']) > len(self.memory[i]['rewards']):
                self.memory[i]['rewards'].append(0.0) # Placeholder reward, to be filled at episode end
                self.memory[i]['dones'].append(done)

    def finalize_episode_and_update(self, final_info: Dict, final_rewards: Dict):
        """At the end of an episode, compute GAE and update PPO policies."""
        if not self.training_mode:
                return
            
        winner_idx = final_info.get('winner')
        
        # --- Calculate the Shared Market Welfare Reward ---
        if winner_idx is not None and final_rewards:
            total_market_surplus = final_rewards.get('seller', 0) + final_rewards['buyers'][winner_idx]
        else:
            total_market_surplus = 0 # No surplus if the auction fails

        for buyer_idx in range(self.num_buyers):
            if not self.memory[buyer_idx]['states']:
                continue
            
            is_winner = (buyer_idx == winner_idx)

            # Assign the final, shaped rewards to the agent's memory
            self._shape_and_assign_rewards(buyer_idx, is_winner, total_market_surplus)
            
            # --- Compute GAE & Update ---
            advantages, returns = self._compute_gae(buyer_idx)
            self._update_ppo_policy(buyer_idx, advantages, returns)
            
            # --- Clear Memory ---
            self.memory[buyer_idx] = self._create_memory_buffer()
    
    def _shape_and_assign_rewards(self, buyer_idx: int, is_winner: bool, market_surplus: float):
        """Calculates and assigns the final rewards for an episode's trajectory."""
        num_steps = len(self.memory[buyer_idx]['rewards'])
        if num_steps == 0:
            self.training_history['episode_rewards'][buyer_idx].append(0)
            return

        # 1. Start with exploration bonuses/penalties for each step
        step_rewards = []
        for i in range(num_steps):
            action = self.memory[buyer_idx]['actions'][i].item()
            if action in [1, 2]: # Any bid action
                step_rewards.append(1.0)
            elif action == 0: # Fold action
                step_rewards.append(-1.0)
            else: # Ask question
                step_rewards.append(0.0)

        # 2. Add the main terminal reward (shared market surplus) to the last step
        step_rewards[-1] += market_surplus

        # 3. Add a persona-specific bonus for the winner
        if is_winner:
            persona = self.buyers[buyer_idx]
            winner_bonus = 0
            
            if persona['id'] == 'B2_AGGRESSIVE_TRADER':
                num_aggressive_bids = sum(1 for action in self.memory[buyer_idx]['actions'] if action.item() == 2)
                winner_bonus += num_aggressive_bids * 50
            elif persona['id'] == 'B3_ANALYTICAL_BUYER':
                num_questions = sum(1 for action in self.memory[buyer_idx]['actions'] if action.item() == 3)
                winner_bonus += num_questions * 75
            
            step_rewards[-1] += winner_bonus
        
        # 4. Replace the placeholder rewards in memory with the final calculated rewards
        self.memory[buyer_idx]['rewards'] = step_rewards
        
        # 5. Log the final total reward for analytics
        total_reward = sum(step_rewards)
        self.training_history['episode_rewards'][buyer_idx].append(total_reward)
    
    def _compute_gae(self, buyer_idx: int) -> (torch.Tensor, torch.Tensor):
        """Compute Generalized Advantage Estimation."""
        # This function now assumes self.memory[buyer_idx]['rewards'] has been finalized
        rewards = self.memory[buyer_idx]['rewards']
        values = self.memory[buyer_idx]['values']
        dones = self.memory[buyer_idx]['dones']
        
        advantages = []
        last_advantage = 0
        # The last value is not from memory, it's the value of the state AFTER the last action
        # If the last state is terminal, its value is 0
        if dones[-1]:
            last_value = 0
        else:
            # Bootstrap from the value of the next state (which we don't have, so we use the last known value)
             last_value = values[-1].item()

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            # Use .item() to get scalar values for calculation
            delta = rewards[t] + self.gamma * last_value * mask - values[t].item()
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            advantages.insert(0, last_advantage)
            last_value = values[t].item()
            
        returns = [adv + val.item() for adv, val in zip(advantages, values)]
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update_ppo_policy(self, buyer_idx: int, advantages: torch.Tensor, returns: torch.Tensor):
        """Perform PPO update for a single agent."""
        memory = self.memory[buyer_idx]
        states = torch.cat(memory['states'])
        actions = torch.cat(memory['actions'])
        old_log_probs = torch.cat(memory['log_probs']).detach()
        
        policy = self.policies[buyer_idx]
        optimizer = self.optimizers[buyer_idx]
        
        for _ in range(self.ppo_epochs):
            # Create batches
            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get new policy probabilities and values
                new_probs, new_values = policy(batch_states)
                new_values = new_values.squeeze(-1) # Ensure correct shape
                
                m = Categorical(new_probs)
                new_log_probs = m.log_prob(batch_actions)
                
                # --- PPO Policy (Actor) Loss ---
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # --- Value (Critic) Loss ---
                critic_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                # --- Entropy Bonus ---
                entropy = m.entropy().mean()
                
                # --- Total Loss ---
                # The critic loss coefficient is often 0.5, entropy bonus coefficient 0.01
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                # --- Update ---
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5) # Gradient clipping
                optimizer.step()

    # --- Model Loading/Saving ---
    def save_models(self, directory: str = "rl_models"):
        """Save all policy models to a directory."""
        path = Path(directory)
        path.mkdir(exist_ok=True)
        for i, policy in enumerate(self.policies):
            model_path = path / f"buyer_{i}_{self.buyers[i]['id']}.pth"
            torch.save(policy.state_dict(), model_path)
        logger.info(f"💾 PPO models saved to {directory}")

    def load_models(self, directory: str = "rl_models"):
        """Load all policy models from a directory."""
        path = Path(directory)
        if not path.exists():
            logger.warning(f"RL model directory not found: {directory}. Using fresh models.")
            return

        for i, policy in enumerate(self.policies):
            model_path = path / f"buyer_{i}_{self.buyers[i]['id']}.pth"
            if model_path.exists():
                policy.load_state_dict(torch.load(model_path, map_location=self.device))
                policy.eval()
                logger.info(f"✅ Loaded PPO model for {self.buyers[i]['id']}")
            else:
                logger.warning(f"⚠️ Model file not found for {self.buyers[i]['id']}. Using fresh model.")
        self.training_mode = False

    def save_training_history(self, directory: str = "rl_models"):
        path = Path(directory)
        path.mkdir(exist_ok=True)
        history_file = path / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        logger.info(f"📊 Training history saved to {history_file}")

    def get_seller_action(self, observation: Dict, info: Dict) -> int:
        """Provides a simple heuristic for the seller in an RL context."""
        active_buyers = sum(observation['active_mask'])
        round_no = observation['round_no'][0]
        reserve_met = observation['price'][0] >= self.config['seller']['reserve_price']

        if round_no > 5 and active_buyers <= 1 and reserve_met:
            return 2
        
        return 0


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