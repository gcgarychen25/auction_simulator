"""
Phase 2 RL Agent - PPO with GAE Implementation

Implements a single RL agent with:
- Actor-Critic neural network architecture  
- PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation)
- Personality-specific objective functions
- Online learning during simulation
- State-action mapping for auction environment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, List, Any, Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic neural network for PPO.
    
    Features:
    - Shared feature extraction layers
    - Separate actor (policy) and critic (value) heads
    - 3-layer feedforward architecture as specified
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers (first 2 layers)
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy) - outputs action probabilities
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function) - outputs state value
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with Xavier initialization."""
        for layer in [self.shared_fc1, self.shared_fc2, self.actor_head, self.critic_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Returns:
            action_probs: Probability distribution over actions
            state_value: Estimated value of the state
        """
        # Shared feature extraction
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor head - action probabilities
        action_logits = self.actor_head(x)
        # Clamp logits to prevent inf/nan in softmax
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Ensure probabilities are valid
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        # Critic head - state value
        state_value = self.critic_head(x)
        
        return action_probs, state_value
    
    def get_action_and_value(self, state):
        """Get action probabilities and state value."""
        return self.forward(state)
    
    def get_value(self, state):
        """Get only the state value (for critic evaluation)."""
        _, value = self.forward(state)
        return value


class PPOExperience:
    """Container for storing PPO experience data."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
    
    def store(self, state, action, reward, value, log_prob, done, next_state=None):
        """Store a single step of experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(next_state)
    
    def clear(self):
        """Clear all stored experience."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.next_states.clear()
    
    def get_tensors(self):
        """Convert stored experience to tensors."""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.log_probs),
            torch.BoolTensor(self.dones)
        )


class PPOAgent:
    """
    PPO agent with GAE for auction bidding.
    
    Features:
    - PPO with clipped objective
    - GAE for advantage estimation
    - Personality-specific reward functions
    - Experience replay with proper advantage calculation
    """
    
    def __init__(self, buyer_config: Dict[str, Any], state_dim: int = 8, 
                 action_dim: int = 4, learning_rate: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5):
        """
        Initialize PPO agent.
        
        Args:
            buyer_config: Configuration dict with persona info
            state_dim: Dimension of state representation  
            action_dim: Number of possible actions (4: fold, bid_500, bid_1000, ask)
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
        """
        self.config = buyer_config
        self.persona_id = buyer_config['id']
        self.max_wtp = buyer_config['max_wtp']
        self.risk_aversion = buyer_config['risk_aversion']
        self.ask_prob = buyer_config['ask_prob']
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        
        # Neural network
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience storage
        self.experience = PPOExperience()
        self.episode_experience = PPOExperience()
        
        # Training parameters
        self.batch_size = 64
        self.ppo_epochs = 4  # Number of PPO training epochs per batch
        self.max_grad_norm = 0.5
        
        # Tracking
        self.episode_rewards = []
        self.training_step = 0
        self.episode_count = 0
        
        # Current episode tracking
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards_step = []
        self.episode_values = []
        self.episode_log_probs = []
        
        # Personality-specific parameters
        self._setup_personality_objectives()
    
    def _setup_personality_objectives(self):
        """Setup personality-specific learning objectives and reward weights."""
        persona_type = self.persona_id.split('_')[1]  # Extract type from ID
        
        if persona_type == "CONSERVATIVE":
            self.surplus_weight = 1.0
            self.risk_penalty = 0.3
            self.patience_bonus = 0.1
            self.exploration_decay = 0.9995
            
        elif persona_type == "AGGRESSIVE": 
            self.surplus_weight = 0.7
            self.speed_bonus = 0.4
            self.intimidation_bonus = 0.2
            self.exploration_decay = 0.999
            
        elif persona_type == "ANALYTICAL":
            self.surplus_weight = 0.9
            self.information_bonus = 0.3
            self.timing_bonus = 0.2
            self.exploration_decay = 0.9998
            
        elif persona_type == "BUDGET":
            self.surplus_weight = 1.0
            self.budget_penalty = 0.5
            self.efficiency_bonus = 0.2
            self.exploration_decay = 0.9999
            
        elif persona_type == "FOMO":
            self.surplus_weight = 0.8
            self.activity_bonus = 0.3
            self.competition_response = 0.2
            self.exploration_decay = 0.999
            
        else:
            self.surplus_weight = 1.0
            self.exploration_decay = 0.9995
    
    def state_to_features(self, state: Dict[str, np.ndarray], buyer_idx: int) -> np.ndarray:
        """
        Convert auction state to neural network input features.
        
        Args:
            state: Environment observation
            buyer_idx: Index of this buyer (0-4)
            
        Returns:
            Feature vector for neural network
        """
        current_price = float(state['price'][0])
        round_no = float(state['round_no'][0])
        bids_left = float(state['bids_left'][buyer_idx])
        active = float(state['active_mask'][buyer_idx])
        last_increment = float(state['last_increment'][0])
        
        # Additional computed features
        price_ratio = current_price / self.max_wtp  # How close to max WTP
        remaining_surplus = max(0, self.max_wtp - current_price) / self.max_wtp
        active_competitors = float(np.sum(state['active_mask']) - 1)  # Other active buyers
        
        features = np.array([
            current_price / 20000.0,  # Normalized price
            round_no / 20.0,          # Normalized round
            bids_left / 3.0,          # Normalized remaining bids
            active,                   # Whether still active
            last_increment / 2000.0,  # Normalized last increment
            price_ratio,              # Price as fraction of max WTP
            remaining_surplus,        # Potential surplus remaining
            active_competitors / 4.0  # Normalized competitor count
        ])
        
        return features
    
    def get_action(self, state: Dict[str, np.ndarray], buyer_idx: int, 
                   training: bool = True) -> Tuple[int, float, float]:
        """
        Get action from PPO policy.
        
        Args:
            state: Environment observation
            buyer_idx: Index of this buyer
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        features = self.state_to_features(state, buyer_idx)
        
        try:
            state_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, state_value = self.network(state_tensor)
                action_probs = action_probs.squeeze(0)
                state_value = state_value.squeeze(0)
                
                # Sample action
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                action_int = action.item()
                log_prob_float = log_prob.item()
                value_float = state_value.item()
                
                # Store for episode tracking
                if training:
                    self.episode_states.append(features)
                    self.episode_actions.append(action_int)
                    self.episode_values.append(value_float)
                    self.episode_log_probs.append(log_prob_float)
                
                return action_int, log_prob_float, value_float
                
        except Exception as e:
            print(f"Warning: Error in get_action for {self.persona_id}: {e}")
            # Fallback
            action = random.randint(0, 3)
            if training:
                self.episode_states.append(features)
                self.episode_actions.append(action)
                self.episode_values.append(0.0)
                self.episode_log_probs.append(0.0)
            return action, 0.0, 0.0
    
    def calculate_reward(self, final_price: float, winner_idx: Optional[int], buyer_idx: int,
                        episode_length: int, actions_taken: List[int]) -> float:
        """
        Calculate personality-specific reward for the episode.
        """
        base_reward = 0.0
        
        # Basic economic reward: surplus if won, small penalty if lost
        if winner_idx == buyer_idx:
            surplus = self.max_wtp - final_price
            base_reward = surplus * self.surplus_weight
        else:
            # Small penalty for not winning to encourage competitive behavior
            base_reward = -10.0
        
        # Personality-specific bonuses/penalties
        persona_type = self.persona_id.split('_')[1]
        
        if persona_type == "CONSERVATIVE":
            if winner_idx == buyer_idx:
                surplus = self.max_wtp - final_price
                if surplus > self.max_wtp * 0.1:  # Good margin
                    base_reward += surplus * self.patience_bonus
            # Penalty for risky behavior
            if final_price > self.max_wtp * 0.9:
                base_reward -= self.risk_penalty * 100
                
        elif persona_type == "AGGRESSIVE":
            if winner_idx == buyer_idx:
                # Bonus for quick wins
                speed_bonus = max(0, (10 - episode_length) * self.speed_bonus * 10)
                base_reward += speed_bonus
            # Bonus for intimidation tactics
            large_bids = sum(1 for action in actions_taken if action == 2)
            base_reward += large_bids * self.intimidation_bonus * 20
            
        elif persona_type == "ANALYTICAL":
            # Bonus for information gathering
            questions_asked = sum(1 for action in actions_taken if action == 3)
            base_reward += questions_asked * self.information_bonus * 30
            # Bonus for strategic timing
            if winner_idx == buyer_idx and 4 <= episode_length <= 8:
                base_reward += self.timing_bonus * 50
                
        elif persona_type == "BUDGET":
            # Strong penalty for exceeding budget
            if final_price > self.max_wtp:
                base_reward -= self.budget_penalty * 200
            # Bonus for efficient wins
            if winner_idx == buyer_idx:
                efficiency = (self.max_wtp - final_price) / self.max_wtp
                base_reward += efficiency * self.efficiency_bonus * 100
                
        elif persona_type == "FOMO":
            # Bonus for active participation
            if len(actions_taken) > 3:
                base_reward += self.activity_bonus * 30
            # Bonus for competitive responses
            competitive_actions = sum(1 for action in actions_taken if action in [1, 2])
            base_reward += competitive_actions * self.competition_response * 15
        
        return base_reward
    
    def store_step_reward(self, reward: float):
        """Store step-wise reward during episode."""
        self.episode_rewards_step.append(reward)
    
    def finish_episode(self, final_reward: float, next_state: Optional[np.ndarray] = None):
        """
        Finish episode and store experience with GAE calculation.
        
        Args:
            final_reward: Final episode reward
            next_state: Final state (for bootstrap value)
        """
        if not self.episode_states:
            return
        
        self.episode_count += 1
        self.episode_rewards.append(final_reward)
        
        # Calculate step rewards (distribute final reward across episode)
        episode_length = len(self.episode_states)
        if not self.episode_rewards_step:
            # If no step rewards, distribute final reward
            step_rewards = [final_reward / episode_length] * episode_length
        else:
            # Add final reward to last step
            step_rewards = self.episode_rewards_step[:]
            if step_rewards:
                step_rewards[-1] += final_reward
        
        # Get next state value for bootstrapping
        if next_state is not None:
            try:
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    _, next_value = self.network(next_state_tensor)
                    next_value = next_value.item()
            except:
                next_value = 0.0
        else:
            next_value = 0.0
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae(step_rewards, self.episode_values, next_value)
        returns = advantages + np.array(self.episode_values)
        
        # Store experience
        for i in range(episode_length):
            self.experience.store(
                state=self.episode_states[i],
                action=self.episode_actions[i],
                reward=returns[i],  # Use return as reward for training
                value=self.episode_values[i],
                log_prob=self.episode_log_probs[i],
                done=(i == episode_length - 1)
            )
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards_step.clear()
        self.episode_values.clear()
        self.episode_log_probs.clear()
    
    def _calculate_gae(self, rewards: List[float], values: List[float], 
                      next_value: float) -> np.ndarray:
        """
        Calculate Generalized Advantage Estimation.
        
        Args:
            rewards: List of step rewards
            values: List of value estimates
            next_value: Bootstrap value for last state
            
        Returns:
            Advantage estimates
        """
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        
        # Work backwards through episode
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val - values[t]
            
            # GAE calculation
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
        
        return advantages
    
    def train(self) -> Dict[str, float]:
        """
        Train the PPO agent using stored experience.
        
        Returns:
            Training statistics
        """
        if len(self.experience.states) < self.batch_size:
            return {}
        
        # Get all experience as tensors
        states, actions, returns, old_values, old_log_probs, dones = self.experience.get_tensors()
        
        # Calculate advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # PPO training epochs
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            batch_indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_idx = batch_indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                
                # Forward pass
                action_probs, new_values = self.network(batch_states)
                
                # Calculate new log probabilities
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -self.entropy_coef * entropy
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Clear experience buffer
        self.experience.clear()
        self.training_step += 1
        
        # Return training stats
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs, 
            'entropy_loss': total_entropy_loss / self.ppo_epochs,
            'training_step': self.training_step
        }
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics for monitoring."""
        if not self.episode_rewards:
            return {}
        
        return {
            'avg_reward': np.mean(self.episode_rewards[-100:]),
            'reward_std': np.std(self.episode_rewards[-100:]),
            'training_steps': self.training_step,
            'episodes': self.episode_count,
            'experience_buffer_size': len(self.experience.states)
        }
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'persona_id': self.persona_id,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_loss_coef': self.value_loss_coef
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint.get('episode_count', 0)


# Backward compatibility alias
RLBuyerAgent = PPOAgent 