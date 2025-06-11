"""
Core Auction Simulator

Defines the main AuctionSimulator class that orchestrates auction episodes,
managing agent interactions, and the environment loop for heuristic and RL policies.
"""

import asyncio
import numpy as np
from collections import deque
from typing import Dict, List, Any, Tuple
import logging

from auction_env import AuctionEnv
from policies.heuristic import create_heuristic_policies
from policies.rl_policy import RLPolicyManager

logger = logging.getLogger(__name__)


class AuctionSimulator:
    """
    Main simulator class that orchestrates auction episodes.
    
    Supports "heuristic" and "rl" policy types for baseline and training purposes.
    """
    
    def __init__(self, config: Dict[str, Any], policy_type: str = "heuristic", training_mode: bool = False, rl_manager: RLPolicyManager = None):
        """
        Initialize the auction simulator.
        
        Args:
            config: Configuration dictionary
            policy_type: "heuristic" or "rl"
            training_mode: Whether the simulation is for RL training
            rl_manager: (Optional) An existing RLPolicyManager to use. If not provided, a new one is created for RL mode.
        """
        self.config = config
        self.policy_type = policy_type
        self.training_mode = training_mode
        
        # Initialize environment
        self.env = AuctionEnv(config)
        
        # Initialize policies
        if policy_type == "heuristic":
            self.policies = create_heuristic_policies(config)
            self.rl_manager = None
        elif policy_type == "rl":
            self.policies = None
            if rl_manager:
                self.rl_manager = rl_manager
                # Critical: Update the manager's config to the one for this specific simulator instance
                self.rl_manager.update_config(config)
            else:
                self.rl_manager = RLPolicyManager(config, training_mode=self.training_mode)
            
            if not training_mode and not rl_manager: # Only load models if evaluating from scratch
                self.rl_manager.load_models()
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # History for context (last 10 rounds)
        self.history = deque(maxlen=10)
        
    async def run_episode(self, episode_id: int = 0, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single auction episode.
        
        Args:
            episode_id: Unique identifier for this episode
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with episode results
        """
        if verbose:
            self._log_episode_start(episode_id)
        
        # Reset environment and history
        observation, info = self.env.reset()
        self.history.clear()
        
        episode_log = []
        round_count = 0
        
        # Episode loop
        while True:
            round_count += 1
            
            # Get actions from all agents
            try:
                actions = await self._get_all_actions(observation, info)
                seller_commentary = actions.pop('seller_commentary', '')
            except Exception as e:
                logger.error(f"Error getting actions: {e}")
                break
            
            # Store observation for RL context before stepping
            pre_step_observation = observation
            
            # Step environment
            observation, rewards, terminated, truncated, info = self.env.step(actions)
            
            # If training RL, record the results of the round for each agent
            if self.policy_type == "rl" and self.training_mode:
                self.rl_manager.record_round_results(terminated, truncated)

            # Enhanced round logging
            if verbose:
                self._log_round_structured(round_count, observation, actions, seller_commentary, info)
            
            # Log this round for analytics and potential RL training
            round_log = {
                'round': round_count,
                'observation': pre_step_observation, # State when action was taken
                'actions': actions,
                'rewards': rewards,
                'info': info,
                # Redundant flattened data for easier CSV analysis
                'price': float(observation['price'][0]),
                'active_buyers': int(sum(observation['active_mask'])),
                'terminated': terminated,
                'truncated': truncated,
                'seller_response': seller_commentary,
                'new_bids': info.get('new_bids', []),
                'questions': info.get('questions', []),
                'q_and_a': [] # To be populated with conversation
            }
            # Add Q&A to the log
            if 'questions' in info and info['questions']:
                # This assumes seller response is a direct answer
                round_log['q_and_a'].append({
                    "speaker": "Seller",
                    "message": seller_commentary
                })

            episode_log.append(round_log)
            
            # Add to history for LLM context
            self.history.append({
                'round': round_count,
                'price': float(observation['price'][0]),
                'actions': actions,
                'result': seller_commentary
            })
            
            # Check if episode ended
            if terminated or truncated:
                break
                
            # Safety check
            if round_count >= 50:
                logger.warning("Episode exceeded 50 rounds, terminating")
                break
        
        # For RL training, update policies at the end of the episode
        if self.policy_type == "rl" and self.training_mode:
            self.rl_manager.finalize_episode_and_update(info, rewards)

        # Calculate final results
        results = self._calculate_episode_results(episode_log, rewards, info)
        
        if verbose:
            self._print_episode_summary(results)
        
        return results
    
    def _log_episode_start(self, episode_id: int):
        """Log a clean episode start banner."""
        policy_type = "RL" if self.policy_type == "rl" else "Heuristic"
        logger.info("")
        logger.info("🚀" + "=" * 58 + "🚀")
        logger.info(f"🏠              AUCTION EPISODE {episode_id} STARTING              🏠")
        logger.info("🚀" + "=" * 58 + "🚀")
        logger.info(f"🤖 POLICY TYPE: {policy_type}")
        logger.info(f"💰 STARTING PRICE: ${self.config['environment']['auction']['start_price']:,.0f}")
        logger.info(f"📊 RESERVE PRICE: ${self.config['environment']['seller']['reserve_price']:,.0f}")
        logger.info(f"👥 BUYERS: 5 personas with different strategies")
        logger.info("🚀" + "=" * 58 + "🚀")
    
    def _log_round_structured(self, round_count: int, observation: Dict, actions: Dict, 
                            seller_commentary: str, info: Dict):
        """
        Enhanced structured logging for each round.
        Shows state, then actions in a clear format.
        """
        current_price = observation['price'][0]
        active_buyers = sum(observation['active_mask'])
        leading_bidder_idx = getattr(self.env, 'leading_bidder', -1)
        leading_bidder = self.config['environment']['buyers'][leading_bidder_idx]['id'] if leading_bidder_idx >= 0 else "None"
        
        # Round header with state
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"🏠 ROUND {round_count}")
        logger.info(f"{'='*60}")
        logger.info(f"💰 Current Price: ${current_price:,.0f}")
        logger.info(f"👥 Active Buyers: {active_buyers}/5")
        logger.info(f"🎯 Leading Bidder: {leading_bidder}")
        logger.info(f"📊 Reserve Price: ${self.config['environment']['seller']['reserve_price']:,.0f} {'✅ MET' if current_price >= self.config['environment']['seller']['reserve_price'] else '❌ NOT MET'}")
        
        # Seller Action
        action_descriptions = {
            0: "ANNOUNCE (continue auction)",
            1: "ANSWER (respond to questions)", 
            2: "CLOSE (end auction)"
        }
        action_desc = action_descriptions.get(actions['seller'], "UNKNOWN")
        logger.info(f"\n🤖 SELLER ACTION: [{action_desc}]")
        if seller_commentary:
            logger.info(f"💬 \"{seller_commentary}\"")
        
        # Buyer Actions - organize by action type
        buyer_actions = actions.get('buyers', [])
        new_bids = info.get('new_bids', [])
        questions = info.get('questions', [])
        
        logger.info(f"\n👥 BUYER ACTIONS:")
        
        # Show bids first (most important)
        if new_bids:
            logger.info(f"💵 NEW BIDS:")
            for buyer_idx, bid_amount in new_bids:
                persona_id = self.config['environment']['buyers'][buyer_idx]['id']
                logger.info(f"   • {persona_id}: ${bid_amount:,.0f}")
        
        # Show questions
        if questions:
            logger.info(f"❓ QUESTIONS:")
            for buyer_idx, question in questions:
                persona_id = self.config['environment']['buyers'][buyer_idx]['id'] 
                logger.info(f"   • {persona_id}: {question}")
        
        # Show all buyer actions summary
        logger.info(f"\n📋 ALL BUYER ACTIONS:")
        action_names = {0: "FOLD", 1: "BID_$500", 2: "BID_$1000", 3: "ASK_QUESTION"}
        for buyer_idx, action in enumerate(buyer_actions):
            persona_id = self.config['environment']['buyers'][buyer_idx]['id']
            active = observation['active_mask'][buyer_idx]
            bids_left = observation['bids_left'][buyer_idx]
            status = "🟢" if active else "🔴"
            action_name = action_names.get(action, f"UNKNOWN_{action}")
            logger.info(f"   {status} {persona_id}: {action_name} (Bids left: {bids_left})")
    
    async def _get_all_actions(self, observation: Dict, info: Dict) -> Dict[str, Any]:
        """
        Get actions from all agents (seller and buyers).
        
        Args:
            observation: Current environment observation
            info: Additional environment information
            
        Returns:
            Dictionary with 'seller', 'buyers' actions and 'seller_commentary'
        """
        # Get seller action and commentary
        seller_action, seller_commentary = await self._get_seller_action(observation, info)
        
        # Get buyer actions based on policy type
        buyer_actions = await self._get_buyer_actions(observation, info)
        
        return {
            'seller': seller_action,
            'buyers': buyer_actions,
            'seller_commentary': seller_commentary
        }
    
    async def _get_seller_action(self, observation: Dict, info: Dict) -> Tuple[int, str]:
        """Get action from the seller (heuristic or RL)."""
        # Use RL manager for seller action if in RL mode
        if self.policy_type == "rl":
            action = self.rl_manager.get_seller_action(observation, info)
            return action, "RL seller decision"
        else:
            # Use heuristic seller policy
            action = self.policies['seller'].get_seller_action(observation, info)
            next_bid = observation['price'][0] + 500
            commentary = f"Going once at ${observation['price'][0]:,.0f}! Do I hear ${next_bid:,.0f}?"
            return action, commentary
    
    async def _get_buyer_actions(self, observation: Dict, info: Dict) -> List[int]:
        """Get actions from all buyers."""
        num_buyers = len(self.config['environment']['buyers'])
        actions = []
        
        for buyer_idx in range(num_buyers):
            if self.policy_type == "heuristic":
                persona = self.config['environment']['buyers'][buyer_idx]
                action = self.policies['buyers'][buyer_idx].get_buyer_action(
                    observation, persona, buyer_idx
                )
            else:
                # RL policy
                action = self.rl_manager.get_buyer_action(observation, buyer_idx)
            
            actions.append(action)
        
        return actions
    
    def _calculate_episode_results(self, episode_log: List[Dict], final_rewards: Dict, info: Dict) -> Dict[str, Any]:
        """Calculate and return episode statistics with enhanced economic metrics."""
        winner = info.get('winner')
        final_price = info.get('final_price')
        reserve_price = self.config['environment']['seller']['reserve_price']
        start_price = self.config['environment']['auction']['start_price']
        
        # Basic auction outcome
        auction_successful = winner is not None and final_price is not None
        reserve_met = final_price >= reserve_price if final_price else False
        
        # Enhanced failure categorization
        failure_reason = None
        if not auction_successful:
            if final_price is None:
                failure_reason = "no_bids"
            elif final_price < reserve_price:
                failure_reason = "below_reserve"
        
        # Economic surplus calculation (only if auction succeeded)
        if auction_successful:
            seller_surplus = final_rewards['seller']
            winner_surplus = final_rewards['buyers'][winner]
            total_surplus = seller_surplus + winner_surplus
            
            # Winner value metrics (use episode-specific WTP if available)
            if 'varied_personas' in info:
                winner_max_wtp = info['varied_personas'][winner]['max_wtp']
            else:
                winner_max_wtp = self.config['environment']['buyers'][winner]['max_wtp']

            winner_surplus_ratio = winner_surplus / winner_max_wtp if winner_max_wtp > 0 else 0
            
            # Market efficiency metrics
            theoretical_max_surplus = winner_max_wtp - reserve_price
            surplus_efficiency = total_surplus / theoretical_max_surplus if theoretical_max_surplus > 0 else 0
            
        else:
            # Failed auction - no surplus generated
            seller_surplus = 0
            winner_surplus = 0
            total_surplus = 0
            winner_max_wtp = 0
            winner_surplus_ratio = 0
            surplus_efficiency = 0
        
        # Price dynamics
        if final_price:
            price_premium_over_reserve = (final_price - reserve_price) / reserve_price
            price_premium_over_start = (final_price - start_price) / start_price
        else:
            price_premium_over_reserve = None
            price_premium_over_start = None
        
        results = {
            # Basic metrics
            'episode_length': len(episode_log),
            'final_price': final_price,
            'winner': winner,
            'winner_persona': self.config['environment']['buyers'][winner]['id'] if winner is not None else None,
            
            # Auction outcome
            'auction_successful': auction_successful,
            'reserve_met': reserve_met,
            'failure_reason': failure_reason,
            
            # Economic metrics
            'seller_reward': seller_surplus,
            'winner_surplus': winner_surplus,
            'total_surplus': total_surplus,
            'winner_max_wtp': winner_max_wtp,
            'winner_surplus_ratio': winner_surplus_ratio,
            'surplus_efficiency': surplus_efficiency,
            
            # Price metrics
            'price_premium_over_reserve': price_premium_over_reserve,
            'price_premium_over_start': price_premium_over_start,
            'reserve_price': reserve_price,
            'start_price': start_price,
            
            # Raw data for further analysis
            'buyer_rewards': final_rewards['buyers'],
            'episode_log': episode_log
        }
        
        return results    
    def _print_episode_summary(self, results: Dict[str, Any]):
        """Print a summary of the episode results with enhanced economic analysis."""
        logger.info("")
        logger.info("🏁" + "=" * 58 + "🏁")
        logger.info("🏆                    AUCTION COMPLETE                    🏆")
        logger.info("🏁" + "=" * 58 + "🏁")
        
        if results['auction_successful']:
            logger.info(f"✅ RESULT: SOLD for ${results['final_price']:,.0f}")
            logger.info(f"🏆 WINNER: {results['winner_persona']}")
            logger.info(f"💰 SELLER SURPLUS: ${results['seller_reward']:,.0f}")
            logger.info(f"🛒 WINNER SURPLUS: ${results['winner_surplus']:,.0f}")
            logger.info(f"📊 SURPLUS EFFICIENCY: {results['surplus_efficiency']:.1%}")
            
            # Price analysis
            if results['price_premium_over_reserve']:
                premium = results['price_premium_over_reserve'] * 100
                logger.info(f"📈 PRICE PREMIUM OVER RESERVE: {premium:+.1f}%")
            
            if results['price_premium_over_start']:
                start_premium = results['price_premium_over_start'] * 100
                logger.info(f"🚀 PRICE INCREASE FROM START: {start_premium:+.1f}%")
                
        else:
            failure_reason = results.get('failure_reason', 'unknown')
            if failure_reason == 'no_bids':
                logger.info("❌ RESULT: NO SALE - No bids received")
            elif failure_reason == 'below_reserve':
                logger.info(f"❌ RESULT: NO SALE - Final price below reserve (${results['reserve_price']:,.0f})")
                if results['final_price']:
                    logger.info(f"💰 HIGHEST BID: ${results['final_price']:,.0f}")
            else:
                logger.info("❌ RESULT: NO SALE - Auction failed")
            
            logger.info("🔍 ECONOMIC INSIGHT: No deal was the economically rational outcome")
        
        logger.info(f"⏱️  DURATION: {results['episode_length']} rounds")
        logger.info(f"💎 TOTAL ECONOMIC SURPLUS: ${results['total_surplus']:,.0f}")
        logger.info("🏁" + "=" * 58 + "🏁")
