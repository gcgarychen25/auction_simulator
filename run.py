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
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from auction_env import load_config
# The old simulator is only needed for Phase 1 & 2, so we won't import it here.

# Configure logging with improved formatting
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP and LLM logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("policies.rl_policy").setLevel(logging.CRITICAL)

# New Multi-Agent Imports
try:
    from multi_agent_orchestrator import run_auction_episode
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️  Could not import multi_agent_orchestrator: {e}. Phase 3 will fail.")

# Local (legacy simulator) imports - guarded
try:
    from simulator import AuctionSimulator
    from analysis.main_analyzer import MainAnalyzer
    from policies.rl_policy import train_and_evaluate_rl_agents
    LEGACY_SIM_AVAILABLE = True
except ImportError as e:
    LEGACY_SIM_AVAILABLE = False
    SIMULATOR_IMPORT_ERROR = e

# Phase 3: New Multi-Agent Orchestrator
try:
    from multi_agent.orchestrator import run_auction_episode as run_phase3_episode
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    MULTI_AGENT_AVAILABLE = False
    MULTI_AGENT_IMPORT_ERROR = e

async def run_single_episode(config_path: str = "config.yaml", policy_type: str = "heuristic", verbose: bool = True) -> Dict[str, Any]:
    """
    Run a single episode with the legacy simulator.
    """
    from simulator import AuctionSimulator
    config = load_config(config_path)
    simulator = AuctionSimulator(config, policy_type)
    return await simulator.run_episode(verbose=verbose)


async def run_batch_episodes(config_path: str = "config.yaml", policy_type: str = "heuristic",
                           num_episodes: int = 10, output_file: str = "results.csv") -> List[Dict[str, Any]]:
    """
    Run multiple episodes and save results to CSV using the old simulator.
    
    Args:
        config_path: Path to configuration file
        policy_type: "heuristic" or "rl"
        num_episodes: Number of episodes to run
        output_file: CSV file to save results
        
    Returns:
        List of episode results
    """
    from simulator import AuctionSimulator # Import only when needed
    config = load_config(config_path)
    simulator = AuctionSimulator(config, policy_type) 
    
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


def randomize_buyer_personas(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a deep copy of the config and adds random variations to heuristic buyer personas.
    This is used during RL training to expose the agent to a variety of opponent behaviors.
    """
    import copy
    import random

    new_config = copy.deepcopy(config)
    for buyer_config in new_config['environment']['buyers']:
        # Only vary heuristic buyers, as RL buyers' behavior is learned and we want them to face varied opponents
        if buyer_config.get('type') == 'heuristic':
            # Add variation to WTP factor (e.g., +/- 10% from its base)
            wtp_variation = random.uniform(-0.1, 0.1)
            base_wtp = buyer_config.get('willingness_to_pay_factor', 1.1)
            buyer_config['willingness_to_pay_factor'] = max(1.0, base_wtp * (1 + wtp_variation))
            
            # Add variation to increment factor (e.g., +/- 20% from its base)
            inc_variation = random.uniform(-0.2, 0.2)
            base_inc = buyer_config.get('increment_factor', 0.1)
            buyer_config['increment_factor'] = max(0.05, base_inc * (1 + inc_variation))
            
            # Add variation to patience (e.g., +/- 15% from its base)
            patience_variation = random.uniform(-0.15, 0.15)
            base_patience = buyer_config.get('patience', 0.5)
            buyer_config['patience'] = max(0.1, min(1.0, base_patience * (1 + patience_variation)))
            
    return new_config


async def train_and_evaluate_rl_agents(config_path: str, num_training_episodes: int, num_eval_episodes: int):
    """Orchestrates the full RL training and evaluation pipeline with opponent randomization."""
    logger.info("="*60)
    logger.info("🤖 STARTING PHASE 2: RL AGENT TRAINING & EVALUATION 🤖")
    logger.info("="*60)
    
    from simulator import AuctionSimulator
    from policies.rl_policy import RLPolicyManager

    base_config = load_config(config_path)
    
    # --- 1. Initialize a single, persistent RL Manager ---
    # This agent will learn across all varied episodes.
    rl_manager = RLPolicyManager(base_config, training_mode=True)
    start_time = time.time()

    # --- 2. Training Phase with Randomized Opponents ---
    logger.info(f"\n--- 🏋️ TRAINING FOR {num_training_episodes} EPISODES (with opponent randomization) ---")
    
    for episode_id in range(num_training_episodes):
        # Create a new environment with slightly different opponents for this episode
        random_config = randomize_buyer_personas(base_config)
        
        # Create a temporary simulator for this single episode, passing in the persistent rl_manager
        training_simulator = AuctionSimulator(
            config=random_config, 
            policy_type="rl", 
            training_mode=True,
            rl_manager=rl_manager
        )
        
        # Run the training episode
        await training_simulator.run_episode(episode_id, verbose=False)
        
        if (episode_id + 1) % 100 == 0:
            logger.info(f"--- Training episode {episode_id + 1}/{num_training_episodes} complete ---")

    total_training_time = time.time() - start_time
    logger.info(f"✅ Training complete in {total_training_time:.1f}s")

    # Save trained models and history
    rl_manager.save_models()
    rl_manager.save_training_history()

    # --- 3. Evaluation Phase ---
    logger.info(f"\n--- 📊 EVALUATING FOR {num_eval_episodes} EPISODES (with fixed opponents) ---")
    # For evaluation, we use a simulator with the original, non-randomized config for consistency.
    # We create a new manager and load the just-saved models into it.
    eval_rl_manager = RLPolicyManager(base_config, training_mode=False)
    eval_rl_manager.load_models()
    
    eval_simulator = AuctionSimulator(
        config=base_config, 
        policy_type="rl", 
        training_mode=False,
        rl_manager=eval_rl_manager
    )
    
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


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Auction Simulator")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--output", default="phase1_results.csv", help="Output CSV file for Phase 1 results.")
    parser.add_argument("--no-analysis", action="store_true", help="Skip the automatic analysis report generation for Phase 1 or 2.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for detailed round-by-round output.")
    
    # Phase selection
    parser.add_argument('--phase', type=int, default=0, help='Which phase to run: 0 (smoke test), 1 (monte carlo), 2 (rl), 3 (multi-agent)')
    
    # Phase 1/2 specific arguments
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run for Phase 1/2')
    parser.add_argument('--policy-type', type=str, default='heuristic', choices=['heuristic', 'rl'], help='Policy type for Phase 1/2')
    
    # Phase 2 specific arguments
    parser.add_argument('--training-steps', type=int, default=10000, help='Number of training steps for RL')
    
    args = parser.parse_args()
    return args


async def main():
    """Main entry point."""
    args = parse_args()
    
    console = Console()

    # --- Set up Logging ---
    log_file_path = "auction.log"
    
    # Configure root logger for console output
    root_logger = logging.getLogger()
    if args.verbose:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING) # Suppress info logs from console
        
    # Configure file handler to capture ALL info-level logs
    file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    console.print(Panel(Text("Auction Simulator", justify="center", style="bold magenta"), title="Welcome"))
    console.print(f"📝 All detailed logs are being saved to [bold cyan]{log_file_path}[/bold cyan]")
    logging.info("\n" + "="*80 + f"\n🚀 Starting New Simulation Run: Phase {args.phase}\n" + "="*80)

    # --- Phase Dispatcher ---
    try:
        if args.phase == 0:
            # --- PHASE 0: SMOKE TEST ---
            console.print("[bold yellow]🚀 Running Phase 0: Legacy Simulator Smoke Test...[/bold yellow]")
            
            # For smoke test, we want to see console output regardless of --verbose
            original_console_level = root_logger.level
            if original_console_level > logging.INFO:
                root_logger.setLevel(logging.INFO)

            await run_single_episode(
                config_path=args.config,
                policy_type="heuristic",
                verbose=True
            )
            
            # Restore original logging level
            if original_console_level > logging.INFO:
                root_logger.setLevel(original_console_level)

        elif args.phase == 1:
            # --- PHASE 1: MONTE CARLO (HEURISTIC) ---
            console.print(f"[bold yellow]🚀 Running Phase 1: Monte Carlo Simulation ({args.episodes} episodes)...[/bold yellow]")
            
            results = await run_batch_episodes(
                config_path=args.config,
                policy_type=args.policy_type,
                num_episodes=args.episodes,
                output_file=args.output,
            )
            
            # Optional Analysis
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

        elif args.phase == 2:
            # --- PHASE 2: RL TRAINING AND EVALUATION ---
            console.print("[bold yellow]🚀 Running Phase 2: RL Training and Evaluation...[/bold yellow]")
            
            try:
                eval_results_file = await train_and_evaluate_rl_agents(
                    config_path=args.config,
                    num_training_episodes=args.training_steps,
                    num_eval_episodes=args.training_steps
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

        elif args.phase == 3:
            # --- PHASE 3: LLM MULTI-AGENT SIMULATION ---
            console.print("[bold yellow]🚀 Running Phase 3: LLM Multi-Agent Simulation...[/bold yellow]")
            if LANGGRAPH_AVAILABLE:
                config = load_config(args.config)
                final_state = await run_auction_episode(config)
                logger.info("\n--- ✅ Multi-Agent Episode Complete ---")
                logger.info(f"Winner: {final_state.winner}")
                logger.info(f"Final Price: ${final_state.final_price:,.2f}" if final_state.final_price else "N/A")
                logger.info(f"Reason: {final_state.failure_reason}" if final_state.failure_reason else "Auction successful.")
            else:
                logger.error("❌ Could not run Phase 3 simulation because the orchestrator failed to import.")
            logger.info("\n✅ Phase 3 simulation completed.")

    except Exception as e:
        logger.error(f"❌ An error occurred during the main execution: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import numpy as np  # Import here to avoid dependency issues
    # Ensure policies directory is in path
    import sys
    sys.path.append(os.path.dirname(__file__))
    asyncio.run(main()) 