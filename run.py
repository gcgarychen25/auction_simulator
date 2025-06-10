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
import csv
import time
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import traceback

from auction_env import load_config
from simulator import AuctionSimulator

# Configure logging with improved formatting
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP and LLM logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llm_wrapper").setLevel(logging.WARNING)
logging.getLogger("policies.rl_policy").setLevel(logging.CRITICAL)


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
    
    # Define evaluation interval
    eval_interval = 100 # Evaluate every 100 training episodes
    
    for episode_id in range(num_training_episodes):
        # Run a training episode
        await training_simulator.run_episode(episode_id, verbose=False)

        # --- Periodic Evaluation for Learning Curve Analysis ---
        if (episode_id + 1) % eval_interval == 0:
            logger.info(f"--- 📈 Performing mid-training evaluation at episode {episode_id + 1} ---")
            
            # Use a separate, non-training simulator for evaluation
            temp_eval_simulator = AuctionSimulator(config, policy_type="rl", training_mode=False)
            temp_eval_simulator.rl_manager.policies = training_simulator.rl_manager.policies
            
            eval_results = []
            for _ in range(20): # Run 20 episodes for a quick evaluation
                result = await temp_eval_simulator.run_episode(verbose=False)
                eval_results.append(result)
            
            # Calculate and store performance metrics
            avg_price = np.mean([r['final_price'] for r in eval_results if r['final_price'] is not None])
            success_rate = np.mean([r['auction_successful'] for r in eval_results])
            avg_len = np.mean([r['episode_length'] for r in eval_results])
            
            training_simulator.rl_manager.training_history['performance_metrics'].append({
                'episode': episode_id + 1,
                'metrics': {
                    'market': {
                        'avg_price': avg_price,
                        'success_rate': success_rate,
                        'avg_episode_length': avg_len
                    }
                }
            })
            logger.info(f"--- Evaluation complete: Avg Price ${avg_price:,.0f}, Success {success_rate:.1%} ---")

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
    output_file = "phase2_results.csv"
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
                    baseline_file=baseline_file, 
                    rl_file=eval_results_file, 
                    config_file=args.config
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