import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    An Actor-Critic network for the RL agent.
    
    This network shares a common backbone and has two heads:
    - Actor head: Outputs a probability distribution over actions.
    - Critic head: Outputs a single value estimating the state's quality.
    """
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        
        # Shared backbone
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Actor head
        self.actor = nn.Linear(128, n_actions)
        
        # Critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor: returns action probabilities
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: returns state value
        state_value = self.critic(x)
        
        return action_probs, state_value

    def get_action(self, state, action_mask=None):
        """
        Selects an action based on the policy and optionally masks invalid actions.
        
        Args:
            state: The current environment state.
            action_mask: A binary mask where 1s are valid actions and 0s are invalid.
            
        Returns:
            action, log_prob, entropy
        """
        action_probs, _ = self.forward(state)
        
        if action_mask is not None:
            # Apply the mask: set probs of invalid actions to near-zero
            # Add a small epsilon to avoid all-zero probabilities if mask is all zeros
            action_probs = action_probs * action_mask + 1e-8
            
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy
    
    def evaluate_action(self, state, action):
        """
        Evaluates a given action in a given state. Used in PPO update.
        
        Args:
            state: The state where the action was taken.
            action: The action that was taken.
            
        Returns:
            log_prob, state_value, entropy
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, state_value, entropy 