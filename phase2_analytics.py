"""
Phase 2 RL Analytics - Multi-Agent Training Analysis

Analyzes RL training results and compares with Phase 1 baseline:
- Individual agent learning curves
- Personality-specific performance evolution
- Market dynamics changes during training
- Comparison with heuristic baseline
- RL vs Heuristic performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import yaml
import json
from pathlib import Path


def run_phase2_analysis(rl_results_file: str,
                       baseline_results_file: Optional[str],
                       config_file: str = "config.yaml",
                       training_history_file: str = "rl_models/training_history.json",
                       save_plots: bool = True,
                       num_training_episodes: int = 1000) -> Dict[str, Any]:
    """
    Run comprehensive Phase 2 RL analysis comparing with Phase 1 baseline.
    
    Args:
        rl_results_file: CSV with RL simulation results
        baseline_results_file: CSV with heuristic baseline results
        config_file: Configuration YAML file
        training_history_file: JSON with RL training history
        save_plots: Whether to save visualization plots
        num_training_episodes: The number of episodes used for RL training
        
    Returns:
        Complete analysis report dictionary
    """
    
    print(f"🤖 Running Phase 2 RL Analysis")
    print(f"RL Results: {rl_results_file}")
    print(f"Baseline: {baseline_results_file}")
    
    # Load data
    rl_df = pd.read_csv(rl_results_file)
    baseline_df = pd.read_csv(baseline_results_file) if baseline_results_file and Path(baseline_results_file).exists() else None
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load training history if available
    training_history = {}
    if Path(training_history_file).exists():
        with open(training_history_file, 'r') as f:
            training_history = json.load(f)
    
    buyers = {buyer['id']: buyer for buyer in config['buyers']}
    
    num_baseline_eps = len(baseline_df) if baseline_df is not None else 0
    print(f"📊 Analyzing {len(rl_df)} RL episodes vs {num_baseline_eps} baseline episodes")
    
    # Performance Comparison Analysis
    print("\n" + "="*60)
    print("🏆 PERFORMANCE COMPARISON: RL vs HEURISTIC")
    print("="*60)
    
    comparison_metrics = _compare_performance(rl_df, baseline_df, buyers)
    
    # Learning Curve Analysis
    print("\n" + "="*60)
    print("📈 LEARNING CURVE ANALYSIS")
    print("="*60)
    
    learning_analysis = _analyze_learning_curves(rl_df, training_history, buyers)
    
    # Market Dynamics Evolution
    print("\n" + "="*60)
    print("🏛️ MARKET DYNAMICS EVOLUTION")
    print("="*60)
    
    market_evolution = _analyze_market_evolution(rl_df, baseline_df)
    
    # Personality-Specific Analysis
    print("\n" + "="*60)
    print("🎭 PERSONALITY-SPECIFIC PERFORMANCE")
    print("="*60)
    
    personality_analysis = _analyze_personality_performance(rl_df, baseline_df, buyers, training_history)
    
    # Generate Visualizations
    if save_plots:
        print("\n📊 GENERATING RL ANALYSIS VISUALIZATIONS")
        print("-" * 40)
        
        _generate_rl_visualizations(rl_df, baseline_df, buyers, training_history, 
                                  comparison_metrics, learning_analysis)
    
    # Generate comprehensive report
    _generate_phase2_report(rl_df, baseline_df, buyers, training_history,
                          comparison_metrics, learning_analysis, market_evolution,
                          personality_analysis, rl_results_file, baseline_results_file,
                          num_training_episodes)
    
    # Final Summary
    print("\n" + "="*60)
    print("🎯 PHASE 2 RL ANALYSIS SUMMARY")
    print("="*60)
    
    _print_phase2_summary(comparison_metrics, learning_analysis, market_evolution)
    
    return {
        'comparison_metrics': comparison_metrics,
        'learning_analysis': learning_analysis,
        'market_evolution': market_evolution,
        'personality_analysis': personality_analysis
    }


def _compare_performance(rl_df: pd.DataFrame, baseline_df: Optional[pd.DataFrame], 
                        buyers: Dict[str, Dict]) -> Dict[str, Any]:
    """Compare overall performance between RL and heuristic approaches."""
    
    # Basic performance metrics
    rl_metrics = {
        'success_rate': rl_df['auction_successful'].mean(),
        'avg_price': rl_df[rl_df['auction_successful']]['final_price'].mean(),
        'avg_surplus': rl_df[rl_df['auction_successful']]['total_surplus'].mean(),
        'price_std': rl_df[rl_df['auction_successful']]['final_price'].std(),
        'avg_episode_length': rl_df['episode_length'].mean()
    }
    
    baseline_metrics = {}
    if baseline_df is not None:
        baseline_metrics = {
            'success_rate': baseline_df['auction_successful'].mean(),
            'avg_price': baseline_df[baseline_df['auction_successful']]['final_price'].mean(),
            'avg_surplus': baseline_df[baseline_df['auction_successful']]['total_surplus'].mean(),
            'price_std': baseline_df[baseline_df['auction_successful']]['final_price'].std(),
            'avg_episode_length': baseline_df['episode_length'].mean()
        }
    
    # Calculate improvements
    improvements = {}
    if baseline_metrics:
        for metric in rl_metrics:
            if baseline_metrics.get(metric, 0) > 0:
                improvement = (rl_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric] * 100
                improvements[metric] = improvement
    
    # Winner distribution comparison
    rl_winners = rl_df['winner_persona'].value_counts()
    baseline_winners = baseline_df['winner_persona'].value_counts() if baseline_df is not None else pd.Series()
    
    # Efficiency comparison
    rl_allocative_eff = _calculate_allocative_efficiency(rl_df, buyers)
    baseline_allocative_eff = _calculate_allocative_efficiency(baseline_df, buyers) if baseline_df is not None else 0.0
    
    print(f"🎯 Success Rate: RL {rl_metrics['success_rate']:.1%} vs Heuristic {baseline_metrics.get('success_rate', 0):.1%}")
    print(f"💰 Average Price: RL ${rl_metrics.get('avg_price', 0):,.0f} vs Heuristic ${baseline_metrics.get('avg_price', 0):,.0f}")
    print(f"💎 Average Surplus: RL ${rl_metrics.get('avg_surplus', 0):,.0f} vs Heuristic ${baseline_metrics.get('avg_surplus', 0):,.0f}")
    print(f"⚡ Allocative Efficiency: RL {rl_allocative_eff:.1%} vs Heuristic {baseline_allocative_eff:.1%}")
    
    return {
        'rl_metrics': rl_metrics,
        'baseline_metrics': baseline_metrics,
        'improvements': improvements,
        'rl_winners': rl_winners.to_dict(),
        'baseline_winners': baseline_winners.to_dict(),
        'rl_allocative_efficiency': rl_allocative_eff,
        'baseline_allocative_efficiency': baseline_allocative_eff
    }


def _analyze_learning_curves(rl_df: pd.DataFrame, training_history: Dict, 
                           buyers: Dict[str, Dict]) -> Dict[str, Any]:
    """Analyze learning curves and training progression."""
    
    if not training_history:
        print("⚠️ No training history available for learning curve analysis")
        return {}
    
    # Extract learning metrics
    episode_rewards = training_history.get('episode_rewards', [])
    performance_metrics = training_history.get('performance_metrics', [])
    
    analysis = {}
    
    if episode_rewards:
        # Analyze reward progression for each agent
        agent_learning = {}
        for i, (buyer_id, buyer_config) in enumerate(buyers.items()):
            if i < len(episode_rewards):
                rewards = episode_rewards[i]
                if rewards:
                    # Calculate moving averages
                    window = min(50, len(rewards) // 4)
                    if window > 0:
                        moving_avg = pd.Series(rewards).rolling(window=window).mean()
                        
                        agent_learning[buyer_id] = {
                            'total_episodes': len(rewards),
                            'final_avg_reward': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
                            'initial_avg_reward': np.mean(rewards[:50]) if len(rewards) >= 50 else np.mean(rewards[:10]),
                            'reward_improvement': np.mean(rewards[-50:]) - np.mean(rewards[:50]) if len(rewards) >= 100 else 0,
                            'reward_volatility': np.std(rewards),
                            'learning_trend': 'improving' if len(rewards) > 50 and np.mean(rewards[-25:]) > np.mean(rewards[:25]) else 'stable'
                        }
        
        analysis['agent_learning'] = agent_learning
        
        # Overall learning statistics
        if agent_learning:
            analysis['overall_learning'] = {
                'agents_improving': sum(1 for data in agent_learning.values() if data['learning_trend'] == 'improving'),
                'total_agents': len(agent_learning),
                'avg_reward_improvement': np.mean([data['reward_improvement'] for data in agent_learning.values()])
            }
    
    if performance_metrics:
        # Analyze performance evolution over time
        episodes = [entry['episode'] for entry in performance_metrics]
        market_metrics = [entry['metrics']['market'] for entry in performance_metrics]
        
        if market_metrics:
            analysis['market_evolution'] = {
                'episodes': episodes,
                'success_rates': [m.get('success_rate', 0) for m in market_metrics],
                'avg_prices': [m.get('avg_price', 0) for m in market_metrics],
                'episode_lengths': [m.get('avg_episode_length', 0) for m in market_metrics]
            }
    
    # Print learning summary
    if 'agent_learning' in analysis:
        print("🧠 Learning Progress Summary:")
        for buyer_id, data in analysis['agent_learning'].items():
            trend = "📈" if data['learning_trend'] == 'improving' else "📊"
            print(f"   {trend} {buyer_id}: Reward {data['final_avg_reward']:.1f} (trend: {data['learning_trend']})")
    
    return analysis


def _analyze_market_evolution(rl_df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how market dynamics evolved during RL training."""
    
    # Split RL data into early and late training phases
    split_point = len(rl_df) // 2
    early_rl = rl_df.iloc[:split_point]
    late_rl = rl_df.iloc[split_point:]
    
    phases = {
        'baseline': baseline_df,
        'early_rl': early_rl,
        'late_rl': late_rl
    }
    
    evolution = {}
    
    for phase_name, phase_df in phases.items():
        if phase_df is None:
            continue

        if len(phase_df) > 0 and 'winner_persona' in phase_df.columns:
            successful = phase_df[phase_df['auction_successful']]
            
            # Ensure winner_persona column is not all NaN before calling mode()
            winner_series = phase_df['winner_persona'].dropna()
            if not winner_series.empty:
                dominant_winner = winner_series.mode().iloc[0]
                dominant_win_rate = winner_series.value_counts(normalize=True).iloc[0]
            else:
                dominant_winner = "None"
                dominant_win_rate = 0
                
            evolution[phase_name] = {
                'success_rate': phase_df['auction_successful'].mean(),
                'avg_price': successful['final_price'].mean() if len(successful) > 0 else 0,
                'price_std': successful['final_price'].std() if len(successful) > 0 else 0,
                'avg_surplus': successful['total_surplus'].mean() if len(successful) > 0 else 0,
                'winner_diversity': phase_df['winner_persona'].nunique(),
                'dominant_winner': dominant_winner,
                'dominant_win_rate': dominant_win_rate
            }
    
    # Calculate evolution trends
    if 'baseline' in evolution and 'late_rl' in evolution:
        price_change = evolution['late_rl']['avg_price'] - evolution['baseline']['avg_price']
        surplus_change = evolution['late_rl']['avg_surplus'] - evolution['baseline']['avg_surplus']
        diversity_change = evolution['late_rl']['winner_diversity'] - evolution['baseline']['winner_diversity']
        
        print(f"🏛️ Market Evolution:")
        print(f"   Price Change: ${price_change:+,.0f}")
        print(f"   Surplus Change: ${surplus_change:+,.0f}")
        print(f"   Winner Diversity Change: {diversity_change:+d}")
    
    return evolution


def _analyze_personality_performance(rl_df: pd.DataFrame, baseline_df: pd.DataFrame,
                                   buyers: Dict[str, Dict], training_history: Dict) -> Dict[str, Any]:
    """Analyze performance by personality type."""
    
    personality_analysis = {}
    
    # Extract personality types
    personality_mapping = {}
    for buyer_id, buyer_config in buyers.items():
        persona_type = buyer_id.split('_')[1]  # Extract personality type
        personality_mapping[buyer_id] = persona_type
    
    for buyer_id, persona_type in personality_mapping.items():
        # Baseline performance
        baseline_wins = (baseline_df['winner_persona'] == buyer_id).sum()
        baseline_win_rate = baseline_wins / len(baseline_df)
        
        # RL performance
        rl_wins = (rl_df['winner_persona'] == buyer_id).sum()
        rl_win_rate = rl_wins / len(rl_df)
        
        # Performance change
        win_rate_change = rl_win_rate - baseline_win_rate
        
        personality_analysis[buyer_id] = {
            'persona_type': persona_type,
            'baseline_win_rate': baseline_win_rate,
            'rl_win_rate': rl_win_rate,
            'win_rate_change': win_rate_change,
            'performance_trend': 'improved' if win_rate_change > 0.05 else 'declined' if win_rate_change < -0.05 else 'stable'
        }
    
    # Print personality summary
    print("🎭 Personality Performance Changes:")
    for buyer_id, data in personality_analysis.items():
        trend_emoji = "📈" if data['performance_trend'] == 'improved' else "📉" if data['performance_trend'] == 'declined' else "📊"
        print(f"   {trend_emoji} {data['persona_type']}: {data['baseline_win_rate']:.1%} → {data['rl_win_rate']:.1%} ({data['win_rate_change']:+.1%})")
    
    return personality_analysis


def _calculate_allocative_efficiency(df: pd.DataFrame, buyers: Dict[str, Dict]) -> float:
    """Calculate allocative efficiency (highest WTP buyer wins)."""
    if len(df) == 0:
        return 0.0
    
    # Find highest WTP buyer
    highest_wtp_buyer = max(buyers.items(), key=lambda x: x[1]['max_wtp'])[0]
    
    # Calculate efficiency
    highest_wtp_wins = (df['winner_persona'] == highest_wtp_buyer).sum()
    total_successful = df['auction_successful'].sum()
    
    return highest_wtp_wins / total_successful if total_successful > 0 else 0.0


def _generate_rl_visualizations(rl_df: pd.DataFrame, baseline_df: pd.DataFrame,
                              buyers: Dict[str, Dict], training_history: Dict,
                              comparison_metrics: Dict, learning_analysis: Dict):
    """Generate individual RL analysis visualizations (modular approach)."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    plot_files = []
    
    # 1. Win Rate Comparison
    plt.figure(figsize=(12, 8))
    rl_winners = pd.Series(comparison_metrics['rl_winners'])
    baseline_winners = pd.Series(comparison_metrics['baseline_winners'])
    
    # Ensure all buyer personas are represented
    all_personas = list(buyers.keys())
    rl_win_rates = [rl_winners.get(persona, 0) / len(rl_df) for persona in all_personas]
    baseline_win_rates = [baseline_winners.get(persona, 0) / len(baseline_df) for persona in all_personas]
    
    x = np.arange(len(all_personas))
    width = 0.35
    
    plt.bar(x - width/2, baseline_win_rates, width, label='Heuristic Baseline', alpha=0.8, color='lightcoral')
    plt.bar(x + width/2, rl_win_rates, width, label='RL Training', alpha=0.8, color='skyblue')
    
    plt.xlabel('Buyer Persona')
    plt.ylabel('Win Rate')
    plt.title('Win Rate Comparison: RL vs Heuristic Baseline')
    plt.xticks(x, [persona.split('_')[1] for persona in all_personas], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (base_rate, rl_rate) in enumerate(zip(baseline_win_rates, rl_win_rates)):
        plt.text(i - width/2, base_rate + 0.01, f'{base_rate:.1%}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, rl_rate + 0.01, f'{rl_rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = 'phase2_plot1_win_rates.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plot_files.append(filename)
    print(f"📁 Saved: {filename}")
    plt.close()
    
    # 2. Price Distribution Comparison
    plt.figure(figsize=(12, 8))
    rl_prices = rl_df[rl_df['auction_successful']]['final_price'].dropna()
    baseline_prices = baseline_df[baseline_df['auction_successful']]['final_price'].dropna()
    
    if len(rl_prices) > 0 and len(baseline_prices) > 0:
        # Create overlapping histograms with proper bins
        min_price = min(rl_prices.min(), baseline_prices.min())
        max_price = max(rl_prices.max(), baseline_prices.max())
        bins = np.linspace(min_price, max_price, 25)
        
        plt.hist(baseline_prices, bins=bins, alpha=0.6, label=f'Heuristic (n={len(baseline_prices)})', 
                density=True, color='lightcoral', edgecolor='black')
        plt.hist(rl_prices, bins=bins, alpha=0.6, label=f'RL (n={len(rl_prices)})', 
                density=True, color='skyblue', edgecolor='black')
        
        # Add mean lines
        plt.axvline(baseline_prices.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Heuristic Mean: ${baseline_prices.mean():,.0f}')
        plt.axvline(rl_prices.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'RL Mean: ${rl_prices.mean():,.0f}')
    
    plt.xlabel('Final Price ($)')
    plt.ylabel('Density')
    plt.title('Price Distribution Comparison: RL vs Heuristic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = 'phase2_plot2_price_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plot_files.append(filename)
    print(f"📁 Saved: {filename}")
    plt.close()
    
    # 3. Surplus Distribution Comparison
    plt.figure(figsize=(12, 8))
    rl_surplus = rl_df[rl_df['auction_successful']]['total_surplus'].dropna()
    baseline_surplus = baseline_df[baseline_df['auction_successful']]['total_surplus'].dropna()
    
    if len(rl_surplus) > 0 and len(baseline_surplus) > 0:
        # Create side-by-side box plots
        data_to_plot = [baseline_surplus, rl_surplus]
        labels = [f'Heuristic\n(n={len(baseline_surplus)})', f'RL\n(n={len(rl_surplus)})']
        
        box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightcoral', 'skyblue']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean markers
        means = [baseline_surplus.mean(), rl_surplus.mean()]
        plt.scatter([1, 2], means, color='red', s=100, zorder=10, label='Mean Values')
        
        # Add mean value labels
        for i, mean_val in enumerate(means, 1):
            plt.text(i, mean_val + (max(rl_surplus.max(), baseline_surplus.max()) * 0.02), 
                    f'${mean_val:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Total Surplus ($)')
    plt.title('Economic Surplus Distribution Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    filename = 'phase2_plot3_surplus_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plot_files.append(filename)
    print(f"📁 Saved: {filename}")
    plt.close()
    
    # 4. Individual Learning Curves (Fixed)
    if training_history.get('episode_rewards'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        episode_rewards = training_history['episode_rewards']
        
        for i, (buyer_id, buyer_config) in enumerate(buyers.items()):
            if i < len(episode_rewards) and i < len(axes) and len(episode_rewards[i]) > 0:
                rewards = episode_rewards[i]
                episodes = range(len(rewards))
                
                # Plot raw rewards with transparency
                axes[i].plot(episodes, rewards, alpha=0.2, color='blue', linewidth=0.5)
                
                # Calculate and plot moving average
                window_size = max(50, len(rewards) // 100)  # Adaptive window size
                if len(rewards) > window_size:
                    moving_avg = pd.Series(rewards).rolling(window=window_size, center=True).mean()
                    axes[i].plot(episodes, moving_avg, color='red', linewidth=2, 
                               label=f'MA({window_size})')
                
                # Calculate and show trend
                if len(rewards) > 100:
                    early_mean = np.mean(rewards[:len(rewards)//4])
                    late_mean = np.mean(rewards[-len(rewards)//4:])
                    trend = "↗" if late_mean > early_mean else "↘" if late_mean < early_mean else "→"
                    axes[i].set_title(f'{buyer_id.split("_")[1]} Learning Curve {trend}')
                else:
                    axes[i].set_title(f'{buyer_id.split("_")[1]} Learning Curve')
                
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel('Reward')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Add final performance annotation
                if len(rewards) > 50:
                    final_avg = np.mean(rewards[-50:])
                    axes[i].text(0.02, 0.98, f'Final Avg: {final_avg:.1f}', 
                               transform=axes[i].transAxes, va='top', 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[i].text(0.5, 0.5, 'No Learning Data', transform=axes[i].transAxes, 
                           ha='center', va='center', fontsize=14)
                axes[i].set_title(f'{list(buyers.keys())[i].split("_")[1]} - No Data')
        
        # Hide unused subplots
        for i in range(len(buyers), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Individual Agent Learning Curves', fontsize=16)
        plt.tight_layout()
        filename = 'phase2_plot4_learning_curves.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
        plt.close()
    
    # 5. Market Evolution Over Time
    plt.figure(figsize=(12, 8))
    if 'market_evolution' in learning_analysis and learning_analysis['market_evolution'].get('episodes'):
        evolution = learning_analysis['market_evolution']
        episodes = evolution['episodes']
        success_rates = evolution['success_rates']
        avg_prices = evolution['avg_prices']
        
        # Create subplot for multiple metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Success rate evolution
        ax1.plot(episodes, success_rates, marker='o', linewidth=2, color='green')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Market Success Rate Evolution During Training')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Price evolution
        ax2.plot(episodes, avg_prices, marker='s', linewidth=2, color='blue')
        ax2.set_ylabel('Average Price ($)')
        ax2.set_xlabel('Episode')
        ax2.set_title('Average Price Evolution During Training')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'phase2_plot5_market_evolution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
    else:
        # Create a placeholder plot
        plt.text(0.5, 0.5, 'Market Evolution Data Not Available', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title('Market Evolution Over Time')
        filename = 'phase2_plot5_market_evolution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
    plt.close()
    
    # 6. Comprehensive Efficiency Comparison
    plt.figure(figsize=(12, 8))
    efficiency_metrics = ['Welfare\nEfficiency', 'Allocative\nEfficiency', 'Revenue\nEfficiency']
    
    # Calculate all efficiency metrics
    rl_welfare_eff = comparison_metrics['rl_metrics']['avg_surplus'] / comparison_metrics['baseline_metrics']['avg_surplus'] if comparison_metrics['baseline_metrics']['avg_surplus'] > 0 else 0
    baseline_welfare_eff = 1.0  # Baseline reference
    
    rl_values = [
        rl_welfare_eff,
        comparison_metrics['rl_allocative_efficiency'],
        comparison_metrics['rl_metrics']['avg_price'] / comparison_metrics['baseline_metrics']['avg_price'] if comparison_metrics['baseline_metrics']['avg_price'] > 0 else 0
    ]
    
    baseline_values = [
        baseline_welfare_eff,
        comparison_metrics['baseline_allocative_efficiency'],
        1.0  # Baseline reference
    ]
    
    x = np.arange(len(efficiency_metrics))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Heuristic Baseline', alpha=0.8, color='lightcoral')
    plt.bar(x + width/2, rl_values, width, label='RL Training', alpha=0.8, color='skyblue')
    
    plt.ylabel('Efficiency Ratio')
    plt.title('Market Efficiency Comparison: RL vs Heuristic')
    plt.xticks(x, efficiency_metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (base_val, rl_val) in enumerate(zip(baseline_values, rl_values)):
        plt.text(i - width/2, base_val + 0.02, f'{base_val:.2f}', ha='center', va='bottom', fontweight='bold')
        plt.text(i + width/2, rl_val + 0.02, f'{rl_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    filename = 'phase2_plot6_efficiency_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plot_files.append(filename)
    print(f"📁 Saved: {filename}")
    plt.close()
    
    print(f"✅ Generated {len(plot_files)} individual visualization files")
    return plot_files


def _generate_phase2_report(rl_df: pd.DataFrame, baseline_df: Optional[pd.DataFrame],
                          buyers: Dict[str, Dict], training_history: Dict,
                          comparison_metrics: Dict, learning_analysis: Dict,
                          market_evolution: Dict, personality_analysis: Dict,
                          rl_results_file: str, baseline_results_file: Optional[str],
                          num_training_episodes: int):
    """Generate comprehensive Phase 2 markdown report."""
    
    from datetime import datetime
    
    baseline_episodes = len(baseline_df) if baseline_df is not None else 0
    
    report_content = f"""# Phase 2 RL Training Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**RL Training Episodes:** {num_training_episodes}  
**RL Evaluation Episodes:** {len(rl_df)} (from `{rl_results_file}`)  
**Baseline Dataset:** {baseline_results_file or 'N/A'} ({baseline_episodes} episodes)  

---

## Executive Summary

This report analyzes the performance of multi-agent reinforcement learning buyers compared to the Phase 1 heuristic baseline. Each buyer agent trained their own neural network policy with personality-specific reward functions while competing in auction environments.

### Key Results
- **RL Success Rate:** {comparison_metrics['rl_metrics']['success_rate']:.1%} vs Heuristic {comparison_metrics['baseline_metrics'].get('success_rate', 0):.1%}
- **Average Price:** RL ${comparison_metrics['rl_metrics'].get('avg_price', 0):,.0f} vs Heuristic ${comparison_metrics['baseline_metrics'].get('avg_price', 0):,.0f}
- **Allocative Efficiency:** RL {comparison_metrics['rl_allocative_efficiency']:.1%} vs Heuristic {comparison_metrics.get('baseline_allocative_efficiency', 0):.1%}
- **Learning Agents:** {learning_analysis.get('overall_learning', {}).get('agents_improving', 0)}/{learning_analysis.get('overall_learning', {}).get('total_agents', 5)} showed improving trends

---

## Performance Comparison Analysis

### Overall Market Metrics

#### Win Rate Analysis
![Win Rate Comparison](phase2_plot1_win_rates.png)

#### Price Distribution Analysis  
![Price Distribution Comparison](phase2_plot2_price_distribution.png)

#### Economic Surplus Analysis
![Surplus Distribution Comparison](phase2_plot3_surplus_comparison.png)

**Market Efficiency:**
- **Allocative Efficiency Change:** {(comparison_metrics['rl_allocative_efficiency'] - comparison_metrics.get('baseline_allocative_efficiency', 0)) * 100:+.1f} percentage points
- **Price Volatility:** RL ${comparison_metrics['rl_metrics']['price_std']:.0f} vs Heuristic ${comparison_metrics['baseline_metrics'].get('price_std', 0):.0f}
- **Economic Surplus:** RL ${comparison_metrics['rl_metrics'].get('avg_surplus', 0):,.0f} vs Heuristic ${comparison_metrics['baseline_metrics'].get('avg_surplus', 0):,.0f}

**Winner Distribution Changes:**
"""
    
    # Add winner distribution analysis
    baseline_total_wins = len(baseline_df) if baseline_df is not None else 1
    for buyer_id in buyers.keys():
        rl_wins = comparison_metrics['rl_winners'].get(buyer_id, 0)
        baseline_wins = comparison_metrics['baseline_winners'].get(buyer_id, 0)
        rl_rate = rl_wins / len(rl_df) * 100 if len(rl_df) > 0 else 0
        baseline_rate = baseline_wins / baseline_total_wins * 100
        change = rl_rate - baseline_rate
        
        persona_type = buyer_id.split('_')[1]
        report_content += f"- **{persona_type}:** {baseline_rate:.1f}% → {rl_rate:.1f}% ({change:+.1f}pp)\n"
    
    report_content += f"""

---

## Learning Curve Analysis

### Individual Agent Learning Progression
![Individual Learning Curves](phase2_plot4_learning_curves.png)

### Market Evolution During Training
![Market Evolution](phase2_plot5_market_evolution.png)

### Efficiency Comparison
![Efficiency Comparison](phase2_plot6_efficiency_comparison.png)

### Individual Agent Performance
"""
    
    # Add learning analysis for each agent
    if 'agent_learning' in learning_analysis:
        for buyer_id, data in learning_analysis['agent_learning'].items():
            persona_type = buyer_id.split('_')[1]
            trend_emoji = "📈" if data['learning_trend'] == 'improving' else "📊"
            
            report_content += f"""
**{trend_emoji} {persona_type} Agent:**
- **Learning Trend:** {data['learning_trend'].capitalize()}
- **Final Avg Reward:** {data['final_avg_reward']:.1f}
- **Reward Improvement:** {data['reward_improvement']:+.1f}
- **Training Episodes:** {data['total_episodes']}
"""
    
    report_content += f"""

### Training Convergence
"""
    
    if 'overall_learning' in learning_analysis:
        overall = learning_analysis['overall_learning']
        report_content += f"""
- **Agents Showing Improvement:** {overall['agents_improving']}/{overall['total_agents']}
- **Average Reward Improvement:** {overall['avg_reward_improvement']:+.1f}
"""
    
    report_content += f"""

---

## Market Evolution Analysis

### Training Phase Comparison
"""
    
    for phase_name, phase_data in market_evolution.items():
        phase_display = phase_name.replace('_', ' ').title()
        report_content += f"""
**{phase_display}:**
- Success Rate: {phase_data['success_rate']:.1%}
- Average Price: ${phase_data['avg_price']:,.0f}
- Winner Diversity: {phase_data['winner_diversity']} different personas
- Dominant Winner: {phase_data['dominant_winner']} ({phase_data['dominant_win_rate']:.1%})
"""
    
    report_content += f"""

---

## Personality-Specific Analysis

### Learning by Personality Type
"""
    
    for buyer_id, data in personality_analysis.items():
        trend_emoji = "📈" if data['performance_trend'] == 'improved' else "📉" if data['performance_trend'] == 'declined' else "📊"
        persona_config = buyers[buyer_id]
        
        report_content += f"""
**{trend_emoji} {data['persona_type']} ({buyer_id}):**
- **Max WTP:** ${persona_config['max_wtp']:,}
- **Risk Aversion:** {persona_config['risk_aversion']}
- **Performance Change:** {data['baseline_win_rate']:.1%} → {data['rl_win_rate']:.1%} ({data['win_rate_change']:+.1%})
- **Assessment:** {data['performance_trend'].capitalize()}

"""
    
    report_content += f"""

---

## Technical Implementation

### Neural Network Architecture
- **Network Type:** Actor-Critic with a shared feature layer
- **State Features:** 8-dimensional: price, round, bids left, active status, last increment, price ratio, surplus potential, competitor count
- **Action Space:** 4 actions: fold, bid $500, bid $1000, ask question
- **Training Algorithm:** PPO with Generalized Advantage Estimation (GAE)

### Personality-Specific Objectives
Each agent is driven by economic surplus, with two additional reward shaping mechanisms:
1.  **Exploration Bonus:** A small reward for bidding and penalty for folding is applied in every round to encourage participation.
2.  **Winner's Bonus:** The winning agent receives an additional reward bonus based on persona-specific actions taken *during the winning episode* (e.g., the Aggressive agent is rewarded for using larger bids to win).

### Training Configuration
- **Training Episodes:** {num_training_episodes}
- **Evaluation Episodes:** {len(rl_df)}
- **PPO Epochs:** 10 per update
- **Learning Rate:** 0.0003
- **GAE Lambda:** 0.95

---

## Conclusions and Insights

### RL vs Heuristic Performance
"""
    
    # Performance assessment
    price_improvement = comparison_metrics['improvements']['avg_price']
    surplus_improvement = comparison_metrics['improvements']['avg_surplus']
    
    if abs(price_improvement) < 5 and abs(surplus_improvement) < 5:
        assessment = "The RL agents achieved comparable performance to heuristic baselines"
    elif price_improvement > 10 or surplus_improvement > 10:
        assessment = "The RL agents significantly outperformed heuristic baselines"
    elif price_improvement < -10 or surplus_improvement < -10:
        assessment = "The heuristic baselines outperformed RL agents"
    else:
        assessment = "Mixed results with some improvements in specific areas"
    
    report_content += f"""
{assessment}, with:

- **Price Performance:** {price_improvement:+.1f}% change
- **Surplus Performance:** {surplus_improvement:+.1f}% change
- **Allocative Efficiency:** {(comparison_metrics['rl_allocative_efficiency'] - comparison_metrics.get('baseline_allocative_efficiency', 0)) * 100:+.1f}pp change

### Key Findings
1. **Learning Capability:** {learning_analysis.get('overall_learning', {}).get('agents_improving', 0)} out of 5 agents showed improvement during training
2. **Personality Adaptation:** Different personas developed distinct bidding strategies aligned with their reward functions
3. **Market Dynamics:** {"Stable" if abs(price_improvement) < 10 else "Significant changes in"} market outcomes compared to heuristic baseline

### Recommendations for Future Development
1. **Extended Training:** Consider longer training periods for better convergence
2. **Curriculum Learning:** Implement progressive difficulty in auction scenarios
3. **Opponent Modeling:** Add opponent awareness to state representation
4. **Hyperparameter Tuning:** Optimize learning rates and exploration strategies per personality

---

*Report generated by Phase 2 RL Analytics System*
"""
    
    # Save the report
    with open('phase2_rl_report.md', 'w') as f:
        f.write(report_content)
    
    print("📄 Generated comprehensive RL report: phase2_rl_report.md")


def _print_phase2_summary(comparison_metrics: Dict, learning_analysis: Dict, 
                         market_evolution: Dict):
    """Print final Phase 2 summary."""
    
    print(f"📊 DATASET COMPARISON:")
    print(f"   • RL Training Episodes: {comparison_metrics['rl_metrics'].get('avg_episode_length', 0):.1f} avg rounds")
    print(f"   • Success Rate Change: {(comparison_metrics['rl_metrics']['success_rate'] - comparison_metrics['baseline_metrics'].get('success_rate', 0)) * 100:+.1f}pp")
    
    print(f"\n🤖 RL TRAINING EFFECTIVENESS:")
    if 'overall_learning' in learning_analysis:
        overall = learning_analysis['overall_learning']
        print(f"   • Agents Improving: {overall['agents_improving']}/{overall['total_agents']}")
        print(f"   • Avg Reward Change: {overall['avg_reward_improvement']:+.1f}")
    
    print(f"\n⚡ EFFICIENCY COMPARISON:")
    eff_change = comparison_metrics['rl_allocative_efficiency'] - comparison_metrics.get('baseline_allocative_efficiency', 0)
    print(f"   • Allocative Efficiency: {eff_change * 100:+.1f}pp change")
    print(f"   • Final RL Efficiency: {comparison_metrics['rl_allocative_efficiency']:.1%}")
    
    print(f"\n🎯 PHASE 2 ASSESSMENT:")
    if eff_change > 0.1:
        print("   • ✅ RL significantly improved market efficiency")
    elif eff_change > 0.05:
        print("   • ✅ RL moderately improved market efficiency") 
    elif eff_change > -0.05:
        print("   • 📊 RL maintained baseline efficiency levels")
    else:
        print("   • ⚠️ RL efficiency below baseline - consider extended training")


if __name__ == "__main__":
    import sys
    
    rl_file = sys.argv[1] if len(sys.argv) > 1 else "phase2_rl_results.csv"
    baseline_file = sys.argv[2] if len(sys.argv) > 2 else "phase1_final.csv"
    num_training_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    if Path(rl_file).exists():
        print(f"🚀 Running Phase 2 RL Analysis")
        report = run_phase2_analysis(
            rl_results_file=rl_file, 
            baseline_results_file=baseline_file,
            num_training_episodes=num_training_episodes
        )
        print(f"\n✅ Phase 2 analysis complete!")
    else:
        print(f"❌ Missing RL results file: {rl_file}")
        print("💡 Run RL training first: python run.py --train-rl") 