"""
Multi-Agent Trainer for PPO Auction Agents

Coordinates training of multiple PPO agents with personality-specific objectives.
Handles simultaneous learning with proper experience collection and batch training.
"""

import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from .rl_agent import PPOAgent


class MultiAgentTrainer:
    """
    Trains multiple PPO agents simultaneously in the auction environment.
    
    Features:
    - Independent PPO agents for each buyer persona
    - Coordinated experience collection and training
    - Performance tracking and model persistence
    - Training progress monitoring
    """
    
    def __init__(self, buyers_config: List[Dict[str, Any]], 
                 learning_rate: float = 3e-4,
                 train_frequency: int = 5,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5):
        """
        Initialize multi-agent trainer.
        
        Args:
            buyers_config: List of buyer configuration dictionaries
            learning_rate: Learning rate for all agents
            train_frequency: Train every N episodes
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
        """
        self.buyers_config = buyers_config
        self.num_agents = len(buyers_config)
        self.train_frequency = train_frequency
        
        # Create PPO agents for each buyer persona
        self.agents = []
        for buyer_config in buyers_config:
            agent = PPOAgent(
                buyer_config=buyer_config,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                value_loss_coef=value_loss_coef
            )
            self.agents.append(agent)
        
        # Training tracking
        self.episode_count = 0
        self.training_history = {
            'episode_rewards': [[] for _ in range(self.num_agents)],
            'win_rates': [[] for _ in range(self.num_agents)],
            'training_losses': [[] for _ in range(self.num_agents)],
            'performance_metrics': []
        }
        
        # Episode tracking
        self.episode_winners = []
        self.episode_prices = []
        self.episode_lengths = []
        
        # Current episode tracking for each agent
        self.current_episode_actions = [[] for _ in range(self.num_agents)]
        
        self.logger = logging.getLogger(__name__)
    
    def get_all_actions(self, state: Dict[str, np.ndarray], training: bool = True) -> Tuple[List[int], List[float], List[float]]:
        """
        Get actions from all PPO agents.
        
        Args:
            state: Environment observation
            training: Whether in training mode
            
        Returns:
            Tuple of (actions, log_probs, values) for all 5 buyers
        """
        actions = []
        log_probs = []
        values = []
        
        for i, agent in enumerate(self.agents):
            # Check if buyer is still active
            if state['active_mask'][i] == 0:
                action = 0  # Fold if not active
                log_prob = 0.0
                value = 0.0
            else:
                action, log_prob, value = agent.get_action(state, i, training=training)
            
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            # Track actions for episode summary
            if training:
                self.current_episode_actions[i].append(action)
        
        return actions, log_probs, values
    
    def store_step_rewards(self, step_rewards: List[float]):
        """
        Store step-wise rewards for all agents.
        
        Args:
            step_rewards: List of rewards for each agent this step
        """
        for i, reward in enumerate(step_rewards):
            if i < len(self.agents):
                self.agents[i].store_step_reward(reward)
    
    def finish_episode(self, final_price: Optional[float], winner_idx: Optional[int],
                      episode_length: int, final_state: Optional[Dict[str, np.ndarray]] = None):
        """
        Finish episode for all agents and handle training.
        
        Args:
            final_price: Final auction price (None if no sale)
            winner_idx: Index of winning buyer (None if no winner)
            episode_length: Number of rounds in episode
            final_state: Final state for bootstrapping values
        """
        self.episode_count += 1
        
        # Store episode metrics
        self.episode_winners.append(winner_idx)
        self.episode_prices.append(final_price if final_price else 0)
        self.episode_lengths.append(episode_length)
        
        # Calculate final rewards and finish episodes for each agent
        training_stats = {}
        for i, agent in enumerate(self.agents):
            # Calculate personality-specific reward
            final_reward = agent.calculate_reward(
                final_price=final_price if final_price else 0,
                winner_idx=winner_idx,
                buyer_idx=i,
                episode_length=episode_length,
                actions_taken=self.current_episode_actions[i]
            )
            
            # Get next state features for bootstrapping (if available)
            next_state_features = None
            if final_state is not None:
                try:
                    next_state_features = agent.state_to_features(final_state, i)
                except:
                    next_state_features = None
            
            # Finish episode (this calculates GAE and stores experience)
            agent.finish_episode(final_reward, next_state_features)
            
            # Track rewards in training history
            self.training_history['episode_rewards'][i].append(final_reward)
        
        # Clear current episode action tracking
        for i in range(self.num_agents):
            self.current_episode_actions[i].clear()
        
        # Train agents if it's time
        if self.episode_count % self.train_frequency == 0:
            training_stats = self.train_all_agents()
        
        # Log progress periodically
        if self.episode_count % 50 == 0:
            self._log_training_progress()
        
        return training_stats
    
    def train_all_agents(self) -> Dict[str, Any]:
        """
        Train all PPO agents simultaneously.
        
        Returns:
            Training statistics for all agents
        """
        training_stats = {
            'episode': self.episode_count,
            'agents': {}
        }
        
        for i, agent in enumerate(self.agents):
            try:
                # Train the agent
                agent_stats = agent.train()
                
                if agent_stats:  # Only log if training occurred
                    training_stats['agents'][agent.persona_id] = agent_stats
                    
                    # Store in training history
                    if 'policy_loss' in agent_stats:
                        self.training_history['training_losses'][i].append(agent_stats['policy_loss'])
                    
                    self.logger.debug(f"Trained {agent.persona_id}: {agent_stats}")
                    
            except Exception as e:
                self.logger.warning(f"Training failed for {agent.persona_id}: {e}")
        
        return training_stats
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for all agents.
        
        Returns:
            Dictionary with performance statistics
        """
        if self.episode_count == 0:
            return {}
        
        metrics = {
            'total_episodes': self.episode_count,
            'agents': {},
            'overall': {}
        }
        
        # Calculate metrics for each agent
        all_win_rates = []
        all_avg_rewards = []
        
        for i, agent in enumerate(self.agents):
            persona_id = agent.persona_id
            
            # Win rate calculation
            agent_wins = sum(1 for winner in self.episode_winners if winner == i)
            win_rate = agent_wins / self.episode_count
            all_win_rates.append(win_rate)
            
            # Average reward (last 100 episodes)
            recent_rewards = self.training_history['episode_rewards'][i][-100:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            reward_std = np.std(recent_rewards) if recent_rewards else 0
            all_avg_rewards.append(avg_reward)
            
            # Training statistics
            agent_training_stats = agent.get_training_stats()
            
            # Recent performance trend (last 20 episodes)
            recent_20_rewards = self.training_history['episode_rewards'][i][-20:]
            if len(recent_20_rewards) >= 10:
                first_half = np.mean(recent_20_rewards[:len(recent_20_rewards)//2])
                second_half = np.mean(recent_20_rewards[len(recent_20_rewards)//2:])
                trend = "↗" if second_half > first_half * 1.1 else ("↘" if second_half < first_half * 0.9 else "→")
            else:
                trend = "→"
            
            metrics['agents'][persona_id] = {
                'win_rate': win_rate,
                'total_wins': agent_wins,
                'avg_reward': avg_reward,
                'reward_std': reward_std,
                'trend': trend,
                **agent_training_stats
            }
        
        # Overall statistics
        metrics['overall'] = {
            'avg_win_rate': np.mean(all_win_rates),
            'win_rate_std': np.std(all_win_rates),
            'avg_reward': np.mean(all_avg_rewards),
            'avg_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'avg_final_price': np.mean([p for p in self.episode_prices[-100:] if p > 0]) if self.episode_prices else 0,
            'success_rate': sum(1 for p in self.episode_prices[-100:] if p > 0) / min(100, len(self.episode_prices)) if self.episode_prices else 0
        }
        
        return metrics
    
    def _log_training_progress(self):
        """Log training progress summary."""
        metrics = self.get_performance_metrics()
        
        if not metrics:
            return
        
        self.logger.info(f"\n📊 TRAINING PROGRESS - Episode {self.episode_count}")
        self.logger.info("=" * 60)
        
        # Overall stats
        overall = metrics['overall']
        self.logger.info(f"🎯 Success Rate: {overall['success_rate']:.1%}")
        self.logger.info(f"💰 Avg Final Price: ${overall['avg_final_price']:,.0f}")
        self.logger.info(f"⏱️  Avg Episode Length: {overall['avg_episode_length']:.1f} rounds")
        
        # Agent performance
        self.logger.info(f"\n👥 AGENT PERFORMANCE:")
        for persona_id, stats in metrics['agents'].items():
            persona_short = persona_id.split('_')[1]
            self.logger.info(
                f"  {persona_short:12} | Win: {stats['win_rate']:5.1%} | "
                f"Reward: {stats['avg_reward']:6.0f} | "
                f"Episodes: {stats.get('episodes', 0):4d} | "
                f"Trend: {stats['trend']}"
            )
        
        self.logger.info("=" * 60)
    
    def save_models(self, save_dir: str, episode_suffix: str = ""):
        """
        Save all trained models.
        
        Args:
            save_dir: Directory to save models
            episode_suffix: Optional suffix for filenames
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            persona_id = agent.persona_id
            filename = f"{persona_id}_episode_{self.episode_count}{episode_suffix}.pth"
            filepath = os.path.join(save_dir, filename)
            
            try:
                agent.save_model(filepath)
                self.logger.info(f"Saved model for {persona_id} to {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save model for {persona_id}: {e}")
        
        # Save training history
        import pickle
        history_file = os.path.join(save_dir, f"training_history_episode_{self.episode_count}{episode_suffix}.pkl")
        try:
            with open(history_file, 'wb') as f:
                pickle.dump({
                    'training_history': self.training_history,
                    'episode_winners': self.episode_winners,
                    'episode_prices': self.episode_prices,
                    'episode_lengths': self.episode_lengths,
                    'episode_count': self.episode_count
                }, f)
            self.logger.info(f"Saved training history to {history_file}")
        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")
    
    def load_models(self, save_dir: str, episode_number: Optional[int] = None):
        """
        Load trained models.
        
        Args:
            save_dir: Directory containing saved models
            episode_number: Specific episode number to load (latest if None)
        """
        if not os.path.exists(save_dir):
            self.logger.error(f"Save directory {save_dir} does not exist")
            return False
        
        loaded_count = 0
        for i, agent in enumerate(self.agents):
            persona_id = agent.persona_id
            
            if episode_number:
                filename = f"{persona_id}_episode_{episode_number}.pth"
            else:
                # Find latest model for this persona
                import glob
                pattern = os.path.join(save_dir, f"{persona_id}_episode_*.pth")
                files = glob.glob(pattern)
                if not files:
                    self.logger.warning(f"No saved models found for {persona_id}")
                    continue
                filename = os.path.basename(max(files, key=os.path.getctime))
            
            filepath = os.path.join(save_dir, filename)
            
            try:
                agent.load_model(filepath)
                loaded_count += 1
                self.logger.info(f"Loaded model for {persona_id} from {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to load model for {persona_id}: {e}")
        
        # Load training history if available
        try:
            import pickle
            import glob
            history_pattern = os.path.join(save_dir, "training_history_episode_*.pkl")
            history_files = glob.glob(history_pattern)
            if history_files:
                latest_history = max(history_files, key=os.path.getctime)
                with open(latest_history, 'rb') as f:
                    data = pickle.load(f)
                    self.training_history = data['training_history']
                    self.episode_winners = data['episode_winners']
                    self.episode_prices = data['episode_prices']
                    self.episode_lengths = data['episode_lengths']
                    self.episode_count = data['episode_count']
                self.logger.info(f"Loaded training history from {latest_history}")
        except Exception as e:
            self.logger.warning(f"Failed to load training history: {e}")
        
        return loaded_count == len(self.agents)
    
    def reset_training_stats(self):
        """Reset training statistics (useful for evaluation phases)."""
        self.training_history = {
            'episode_rewards': [[] for _ in range(self.num_agents)],
            'win_rates': [[] for _ in range(self.num_agents)],
            'training_losses': [[] for _ in range(self.num_agents)],
            'performance_metrics': []
        }
        self.episode_winners = []
        self.episode_prices = []
        self.episode_lengths = []
        self.episode_count = 0
        
        # Reset agents' episode counts
        for agent in self.agents:
            agent.episode_count = 0
            agent.episode_rewards = []
        
        self.logger.info("Reset training statistics")
    
    def set_training_mode(self, training: bool):
        """Enable/disable training mode for all agents."""
        # Note: PPO agents don't have a direct training mode toggle
        # Training is controlled by whether we call train() and finish_episode()
        self.logger.info(f"Training mode set to: {training}")
    
    def get_win_rates(self) -> Dict[str, float]:
        """Get current win rates for all agents."""
        if self.episode_count == 0:
            return {agent.persona_id: 0.0 for agent in self.agents}
        
        win_rates = {}
        for i, agent in enumerate(self.agents):
            agent_wins = sum(1 for winner in self.episode_winners if winner == i)
            win_rates[agent.persona_id] = agent_wins / self.episode_count
        
        return win_rates 