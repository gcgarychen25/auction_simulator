"""
Main Orchestration Script for Auction Simulator

Drives the simulation with support for different policy types and batch execution.
Uses asyncio for concurrent LLM calls and provides CLI interface.
"""

# Load environment variables from .env file
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Please install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

import asyncio
import argparse
import yaml
import json
import csv
import time
import numpy as np
from collections import deque
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path
import traceback

from auction_env import AuctionEnv, load_config
from llm_wrapper import LLMWrapper
from policies.heuristic import create_heuristic_policies
from policies.rl_policy import RLPolicyManager

# Configure logging with improved formatting
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP and LLM logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llm_wrapper").setLevel(logging.WARNING)
logging.getLogger("policies.rl_policy").setLevel(logging.CRITICAL)


class AuctionSimulator:
    """
    Main simulator class that orchestrates auction episodes.
    
    Supports different policy types and can run single episodes or batch simulations.
    """
    
    def __init__(self, config: Dict[str, Any], policy_type: str = "heuristic", 
                 use_llm_seller: bool = False, training_mode: bool = False):
        """
        Initialize the auction simulator.
        
        Args:
            config: Configuration dictionary
            policy_type: "heuristic" or "rl"
            use_llm_seller: Whether to use LLM for seller decisions
            training_mode: Whether the simulation is for RL training
        """
        self.config = config
        self.policy_type = policy_type
        self.use_llm_seller = use_llm_seller
        self.training_mode = training_mode
        
        # Initialize environment
        self.env = AuctionEnv(config)
        
        # Initialize policies
        if policy_type == "heuristic":
            self.policies = create_heuristic_policies(config)
            self.rl_manager = None
        elif policy_type == "rl":
            self.policies = None
            self.rl_manager = RLPolicyManager(config, training_mode=self.training_mode)
            if not training_mode:
                self.rl_manager.load_models() # Load pre-trained models for evaluation
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Initialize LLM wrapper if needed
        self.llm_wrapper = None
        if use_llm_seller:
            self.llm_wrapper = LLMWrapper()
        
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
                'questions': info.get('questions', [])
            }
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
        policy_type = "LLM-Enhanced" if self.use_llm_seller else "Heuristic"
        logger.info("")
        logger.info("🚀" + "=" * 58 + "🚀")
        logger.info(f"🏠              AUCTION EPISODE {episode_id} STARTING              🏠")
        logger.info("🚀" + "=" * 58 + "🚀")
        logger.info(f"🤖 POLICY TYPE: {policy_type}")
        logger.info(f"💰 STARTING PRICE: ${self.config['auction']['start_price']:,.0f}")
        logger.info(f"📊 RESERVE PRICE: ${self.config['seller']['reserve_price']:,.0f}")
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
        leading_bidder = self.config['buyers'][leading_bidder_idx]['id'] if leading_bidder_idx >= 0 else "None"
        
        # Round header with state
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"🏠 ROUND {round_count}")
        logger.info(f"{'='*60}")
        logger.info(f"💰 Current Price: ${current_price:,.0f}")
        logger.info(f"👥 Active Buyers: {active_buyers}/5")
        logger.info(f"🎯 Leading Bidder: {leading_bidder}")
        logger.info(f"📊 Reserve Price: ${self.config['seller']['reserve_price']:,.0f} {'✅ MET' if current_price >= self.config['seller']['reserve_price'] else '❌ NOT MET'}")
        
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
                persona_id = self.config['buyers'][buyer_idx]['id']
                logger.info(f"   • {persona_id}: ${bid_amount:,.0f}")
        
        # Show questions
        if questions:
            logger.info(f"❓ QUESTIONS:")
            for buyer_idx, question in questions:
                persona_id = self.config['buyers'][buyer_idx]['id'] 
                logger.info(f"   • {persona_id}: {question}")
        
        # Show all buyer actions summary
        logger.info(f"\n📋 ALL BUYER ACTIONS:")
        action_names = {0: "FOLD", 1: "BID_$500", 2: "BID_$1000", 3: "ASK_QUESTION"}
        for buyer_idx, action in enumerate(buyer_actions):
            persona_id = self.config['buyers'][buyer_idx]['id']
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
        
        # Get buyer actions (can be done in parallel)
        buyer_actions = await self._get_buyer_actions(observation, info)
        
        return {
            'seller': seller_action,
            'buyers': buyer_actions,
            'seller_commentary': seller_commentary
        }
    
    async def _get_seller_action(self, observation: Dict, info: Dict) -> Tuple[int, str]:
        """Get action from seller with smart LLM usage."""
        # Use RL manager for seller action if in RL mode (and not using LLM)
        if self.policy_type == "rl" and not self.use_llm_seller:
            action = self.rl_manager.get_seller_action(observation, info)
            return action, "RL (heuristic) seller decision"

        questions = info.get('questions', [])
        current_price = observation['price'][0]
        active_buyers = sum(observation['active_mask'])
        
        # Decision logic: When to use LLM vs deterministic behavior
        if questions and self.use_llm_seller and self.llm_wrapper:
            # Use LLM only for answering questions
            action = 1  # Answer questions
            prompt = self._build_question_answering_prompt(questions, observation, info)
            
            # For question answering, we want the full response, not parsed action
            try:
                response = await self.llm_wrapper.call_direct(prompt)
                commentary = response.strip()
            except:
                # Fallback if call_direct doesn't exist
                _, commentary = await self.llm_wrapper.call(prompt)
            
            return action, f"LLM: {commentary}"
            
        elif self.use_llm_seller and self.llm_wrapper:
            # Use logic for announce/close decisions, LLM for commentary only if needed
            action = self._decide_announce_or_close(observation, info)
            
            if action == 0:  # Announce - deterministic
                next_bid = current_price + 500
                commentary = f"Going once at ${current_price:,.0f}! Do I hear ${next_bid:,.0f}?"
                return action, commentary
            elif action == 2:  # Close - deterministic  
                commentary = "Going once, going twice, SOLD!"
                return action, commentary
            else:
                # Fallback to continue
                commentary = f"We're at ${current_price:,.0f}!"
                return 0, commentary
        else:
            # Use heuristic seller policy
            if self.policy_type == "heuristic":
                action = self.policies['seller'].get_seller_action(observation, info)
                return action, "Heuristic seller decision"
            else:
                # This case is now handled at the top of the function
                action = self.rl_manager.get_seller_action(observation, info)
                return action, "RL seller decision"
    
    async def _get_buyer_actions(self, observation: Dict, info: Dict) -> List[int]:
        """Get actions from all buyers."""
        num_buyers = len(self.config['buyers'])
        actions = []
        
        for buyer_idx in range(num_buyers):
            if self.policy_type == "heuristic":
                persona = self.config['buyers'][buyer_idx]
                action = self.policies['buyers'][buyer_idx].get_buyer_action(
                    observation, persona, buyer_idx
                )
            else:
                # RL policy
                action = self.rl_manager.get_buyer_action(observation, buyer_idx)
            
            actions.append(action)
        
        return actions
    
    def _build_seller_prompt(self, observation: Dict, info: Dict) -> str:
        """Build a prompt for the seller LLM."""
        current_price = observation['price'][0]
        round_no = observation['round_no'][0]
        active_buyers = sum(observation['active_mask'])
        
        # Build context from recent history
        history_text = ""
        if self.history:
            history_text = "\nRecent auction activity:\n"
            for h in list(self.history)[-3:]:  # Last 3 rounds
                history_text += f"Round {h['round']}: ${h['price']:,.0f} - {h['result']}\n"
        
        # Check for questions to answer
        questions = info.get('questions', [])
        if questions:
            questions_text = "\nBuyer questions to address:\n"
            for buyer_idx, question in questions:
                buyer_persona = self.config['buyers'][buyer_idx]['id']
                questions_text += f"- {buyer_persona}: {question}\n"
        else:
            questions_text = ""
        
        # Calculate auction progress
        reserve_price = self.config['seller']['reserve_price']
        above_reserve = current_price >= reserve_price
        
        prompt = f"""Auctioneer: Round {round_no}, ${current_price:,.0f}, {active_buyers} bidders.{questions_text}
Actions: 0=Continue, 1=Answer, 2=Close
Format: [NUMBER] [5 words max]
Example: "0 Going once!"
"""
        
        return prompt
    
    def _build_question_answering_prompt(self, questions: List, observation: Dict, info: Dict) -> str:
        """Build a focused prompt for answering buyer questions."""
        current_price = observation['price'][0]
        
        questions_text = "Questions to answer:\n"
        for buyer_idx, question in questions:
            buyer_persona = self.config['buyers'][buyer_idx]['id']
            questions_text += f"- {buyer_persona}: {question}\n"
        
        prompt = f"""Real estate auctioneer answering buyer questions. Current bid: ${current_price:,.0f}
{questions_text}
Provide helpful, professional responses about the property. Keep each answer to 1-2 sentences.
Answer the questions directly and concisely:"""
        
        return prompt
    
    def _decide_announce_or_close(self, observation: Dict, info: Dict) -> int:
        """Deterministic logic for when to announce vs close the auction."""
        current_price = observation['price'][0]
        active_buyers = sum(observation['active_mask'])
        reserve_price = self.config['seller']['reserve_price']
        new_bids = info.get('new_bids', [])
        round_no = observation['round_no'][0]
        
        # Never close in the first few rounds - give auction time to develop
        if round_no <= 3:
            return 0  # Always continue early in auction
        
        # Close auction only if reserve is met AND one of these conditions:
        if current_price >= reserve_price:
            if active_buyers <= 1:
                return 2  # Close - only one bidder left
            elif round_no >= 5 and len(new_bids) == 0:
                return 2  # Close - no activity after round 5
            elif current_price >= reserve_price * 1.5:  # 50% above reserve
                return 2  # Close - excellent price achieved
        
        # Otherwise continue the auction
        return 0  # Announce/Continue
    
    def _calculate_episode_results(self, episode_log: List[Dict], final_rewards: Dict, info: Dict) -> Dict[str, Any]:
        """Calculate and return episode statistics with enhanced economic metrics."""
        winner = info.get('winner')
        final_price = info.get('final_price')
        reserve_price = self.config['seller']['reserve_price']
        start_price = self.config['auction']['start_price']
        
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
                winner_max_wtp = self.config['buyers'][winner]['max_wtp']

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
            'winner_persona': self.config['buyers'][winner]['id'] if winner is not None else None,
            
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


async def run_single_episode(config_path: str = "config.yaml", policy_type: str = "heuristic", 
                           use_llm_seller: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Run a single episode with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        policy_type: "heuristic" or "rl" 
        use_llm_seller: Whether to use LLM for seller
        verbose: Whether to print detailed output
        
    Returns:
        Episode results dictionary
    """
    config = load_config(config_path)
    simulator = AuctionSimulator(config, policy_type, use_llm_seller)
    return await simulator.run_episode(verbose=verbose)


async def run_batch_episodes(config_path: str = "config.yaml", policy_type: str = "heuristic",
                           num_episodes: int = 10, output_file: str = "results.csv") -> List[Dict[str, Any]]:
    """
    Run multiple episodes and save results to CSV.
    
    Args:
        config_path: Path to configuration file
        policy_type: "heuristic" or "rl"
        num_episodes: Number of episodes to run
        output_file: CSV file to save results
        
    Returns:
        List of episode results
    """
    config = load_config(config_path)
    simulator = AuctionSimulator(config, policy_type, use_llm_seller=False)  # No LLM for batch
    
    results = []
    start_time = time.time()
    
    logger.info(f"Running {num_episodes} episodes with {policy_type} policies...")
    
    for episode_id in range(num_episodes):
        if episode_id % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Episode {episode_id}/{num_episodes} (elapsed: {elapsed:.1f}s)")
        
        episode_result = await simulator.run_episode(episode_id, verbose=False)
        results.append(episode_result)
    
    # Save to CSV
    save_results_to_csv(results, output_file)
    
    total_time = time.time() - start_time
    logger.info(f"Completed {num_episodes} episodes in {total_time:.1f}s")
    
    return results


def save_results_to_csv(results: List[Dict[str, Any]], filename: str):
    """Save episode results to CSV file with enhanced economic metrics."""
    if not results:
        return
    
    # Extract key metrics for CSV
    csv_data = []
    for i, result in enumerate(results):
        row = {
            # Basic episode info
            'episode': i,
            'episode_length': result['episode_length'],
            
            # Auction outcome
            'auction_successful': result['auction_successful'],
            'reserve_met': result['reserve_met'],
            'failure_reason': result.get('failure_reason', ''),
            
            # Price information
            'final_price': result['final_price'],
            'start_price': result['start_price'],
            'reserve_price': result['reserve_price'],
            'price_premium_over_reserve': result.get('price_premium_over_reserve'),
            'price_premium_over_start': result.get('price_premium_over_start'),
            
            # Winner information
            'winner': result['winner'],
            'winner_persona': result['winner_persona'],
            'winner_max_wtp': result.get('winner_max_wtp', 0),
            
            # Economic metrics
            'seller_reward': result['seller_reward'],
            'winner_surplus': result.get('winner_surplus', 0),
            'total_surplus': result['total_surplus'],
            'winner_surplus_ratio': result.get('winner_surplus_ratio', 0),
            'surplus_efficiency': result.get('surplus_efficiency', 0)
        }
        csv_data.append(row)
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = csv_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    logger.info(f"Results saved to {filename}")


async def train_and_evaluate_rl_agents(config_path: str, num_training_episodes: int, num_eval_episodes: int):
    """Orchestrates the full RL training and evaluation pipeline."""
    logger.info("="*60)
    logger.info("🤖 STARTING PHASE 2: RL AGENT TRAINING & EVALUATION 🤖")
    logger.info("="*60)
    
    config = load_config(config_path)
    
    # --- 1. Training Phase ---
    logger.info(f"\n--- 🏋️ TRAINING FOR {num_training_episodes} EPISODES ---")
    training_simulator = AuctionSimulator(config, policy_type="rl", training_mode=True)
    start_time = time.time()
    
    for episode_id in range(num_training_episodes):
        if (episode_id + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Training Episode {episode_id + 1}/{num_training_episodes} (elapsed: {elapsed:.1f}s)")
        
        await training_simulator.run_episode(episode_id, verbose=False)

    total_training_time = time.time() - start_time
    logger.info(f"✅ Training complete in {total_training_time:.1f}s")

    # Save trained models and history
    training_simulator.rl_manager.save_models()
    training_simulator.rl_manager.save_training_history()

    # --- 2. Evaluation Phase ---
    logger.info(f"\n--- 📊 EVALUATING FOR {num_eval_episodes} EPISODES ---")
    # New simulator instance with trained models loaded (training_mode=False)
    eval_simulator = AuctionSimulator(config, policy_type="rl", training_mode=False)
    
    eval_results = []
    for episode_id in range(num_eval_episodes):
        if (episode_id + 1) % 50 == 0:
            logger.info(f"Evaluation Episode {episode_id + 1}/{num_eval_episodes}")
        
        result = await eval_simulator.run_episode(episode_id, verbose=False)
        eval_results.append(result)
        
    # Save evaluation results to CSV for analysis
    output_file = "phase2_rl_results.csv"
    save_results_to_csv(eval_results, output_file)
    logger.info(f"📈 Evaluation results saved to {output_file}")
    logger.info("="*60)

    return output_file


async def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Real-Estate Auction Simulator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Core Phase-based argument
    parser.add_argument(
        "--phase", 
        type=int, 
        choices=[0, 1, 2], 
        required=True,
        help="""Specify the project phase to run:
  - 0: Smoke Test (1 heuristic episode)
  - 1: Monte Carlo (batch run with heuristic policies)
  - 2: Reinforcement Learning (train and evaluate RL policies)"""
    )
    
    # Phase-specific arguments
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes for Phase 1 Monte Carlo runs.")
    parser.add_argument("--train-episodes", type=int, default=1000, help="Number of training episodes for Phase 2.")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of evaluation episodes for Phase 2.")

    # General configuration
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--output", default="phase1_results.csv", help="Output CSV file for Phase 1 results.")
    parser.add_argument("--llm-seller", action="store_true", help="Use LLM for seller decisions (primarily for Phase 0).")
    parser.add_argument("--no-analysis", action="store_true", help="Skip the automatic analysis report generation for Phase 1 or 2.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for detailed round-by-round output.")

    args = parser.parse_args()
    
    # --- Phase 0: Smoke Test ---
    if args.phase == 0:
        logger.info("🚀 Running Phase 0: Smoke Test (1 heuristic episode)")
        await run_single_episode(
            config_path=args.config,
            policy_type="heuristic",
            use_llm_seller=args.llm_seller,
            verbose=True  # Always verbose for smoke test
        )
        logger.info("\n✅ Phase 0 smoke test completed successfully.")
    
    # --- Phase 1: Monte Carlo Analysis ---
    elif args.phase == 1:
        logger.info(f"🚀 Running Phase 1: Monte Carlo Analysis ({args.episodes} episodes)")
        results = await run_batch_episodes(
            config_path=args.config,
            policy_type="heuristic",
            num_episodes=args.episodes,
            output_file=args.output
        )
        
        if not args.no_analysis:
            logger.info(f"\n🔍 Running Phase 1 Analysis...")
            try:
                from phase1_analytics import run_phase1_analysis
                run_phase1_analysis(args.output, args.config, save_plots=True)
                logger.info(f"📊 Phase 1 analysis complete! Report and plots generated.")
            except ImportError:
                logger.warning("⚠️ phase1_analytics.py not found. Skipping analysis.")
            except Exception as e:
                logger.error(f"❌ Error running Phase 1 analysis: {e}", exc_info=True)
        else:
            logger.info("✅ Phase 1 data generation complete. Analysis skipped as requested.")

    # --- Phase 2: Reinforcement Learning ---
    elif args.phase == 2:
        logger.info("🚀 Running Phase 2: Reinforcement Learning Pipeline")
        try:
            eval_results_file = await train_and_evaluate_rl_agents(
                config_path=args.config,
                num_training_episodes=args.train_episodes,
                num_eval_episodes=args.eval_episodes
            )
            logger.info("\n✅ RL training and evaluation pipeline complete.")
            
            if not args.no_analysis:
                logger.info(f"\n🔍 Running Phase 2 RL Analysis...")
                from phase2_analytics import run_phase2_analysis
                
                baseline_file = "phase1_results.csv"
                if not Path(baseline_file).exists():
                    logger.warning(f"⚠️ Baseline file '{baseline_file}' not found. Analytics will run without comparison.")
                    baseline_file = None

                run_phase2_analysis(
                    rl_results_file=eval_results_file, 
                    baseline_results_file=baseline_file, 
                    save_plots=True,
                    num_training_episodes=args.train_episodes
                )
                logger.info(f"📊 Phase 2 analysis complete! Report and plots generated.")
            else:
                logger.info("✅ Phase 2 training complete. Analysis skipped as requested.")

        except Exception as e:
            logger.error(f"❌ An error occurred during the RL pipeline: {e}")
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    import numpy as np  # Import here to avoid dependency issues
    # Ensure policies directory is in path
    import sys
    sys.path.append(os.path.dirname(__file__))
    asyncio.run(main()) 