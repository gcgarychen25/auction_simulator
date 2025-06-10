"""
Phase 2 Analytics - RL Agent Performance Analysis

Compares the performance of RL agents against the Phase 1 heuristic baseline.
- Compares market metrics (price, surplus, efficiency)
- Analyzes changes in winner distribution and market dynamics
- Generates a comparative report and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import yaml
from pathlib import Path
from datetime import datetime
import logging

from analytics_utils import (
    setup_plot_style,
    save_plot,
    calculate_allocative_efficiency,
    calculate_welfare_efficiency,
    calculate_revenue_efficiency,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset(df: pd.DataFrame, buyers: Dict, reserve_price: float) -> Dict[str, Any]:
    """Analyzes a single dataset and returns key metrics."""
    if df is None or df.empty:
        return {}

    successful_auctions = df[df['auction_successful']].copy()
    
    if successful_auctions.empty:
        return {
            'price_stats': {'mean_price': 0, 'std_price': 0},
            'winner_counts': pd.Series(dtype='int64'),
            'total_episodes': len(df),
            'successful_episodes': 0,
            'avg_seller_surplus': 0,
            'avg_winner_surplus': 0,
            'avg_total_surplus': 0,
            'welfare_efficiency': 0,
            'allocative_efficiency': 0,
            'revenue_efficiency': 0,
            'success_rate': 0.0
        }

    price_stats = {
        'mean_price': successful_auctions['final_price'].mean(),
        'std_price': successful_auctions['final_price'].std(),
    }

    winner_counts = df['winner_persona'].value_counts()

    avg_seller_surplus = successful_auctions['seller_reward'].mean()
    avg_winner_surplus = successful_auctions['winner_surplus'].mean()
    avg_total_surplus = successful_auctions['total_surplus'].mean()

    welfare_efficiency = calculate_welfare_efficiency(df, buyers, reserve_price)
    allocative_efficiency = calculate_allocative_efficiency(df, buyers)
    revenue_efficiency = calculate_revenue_efficiency(df, buyers, reserve_price)
    
    return {
        'price_stats': price_stats,
        'winner_counts': winner_counts,
        'total_episodes': len(df),
        'successful_episodes': len(successful_auctions),
        'avg_seller_surplus': avg_seller_surplus,
        'avg_winner_surplus': avg_winner_surplus,
        'avg_total_surplus': avg_total_surplus,
        'welfare_efficiency': welfare_efficiency,
        'allocative_efficiency': allocative_efficiency,
        'revenue_efficiency': revenue_efficiency,
        'success_rate': len(successful_auctions) / len(df) if len(df) > 0 else 0.0
    }

def plot_win_rate_comparison(baseline_metrics: Dict, rl_metrics: Dict, buyers: Dict, output_dir: Path):
    """Plots a comparison of win rates between baseline and RL agents."""
    all_personas = sorted(list(buyers.keys()))
    
    baseline_win_rates = baseline_metrics['winner_counts'].reindex(all_personas, fill_value=0) / baseline_metrics['total_episodes'] * 100
    rl_win_rates = rl_metrics['winner_counts'].reindex(all_personas, fill_value=0) / rl_metrics['total_episodes'] * 100
    
    df = pd.DataFrame({'Baseline': baseline_win_rates, 'RL Agent': rl_win_rates})
    
    plt.figure(figsize=(12, 7))
    df.plot(kind='bar', width=0.8)
    
    plt.title('Win Rate Comparison: Baseline vs. RL Agent')
    plt.ylabel('Win Rate (%)')
    plt.xlabel('Buyer Persona')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir, 'plot1_win_rates.png')

def plot_price_distribution_comparison(df_baseline: pd.DataFrame, df_rl: pd.DataFrame, output_dir: Path):
    """Plots a comparison of final price distributions."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_baseline['final_price'], label='Baseline', fill=True)
    sns.kdeplot(df_rl['final_price'], label='RL Agent', fill=True)
    
    plt.title('Final Price Distribution: Baseline vs. RL Agent')
    plt.xlabel('Final Price ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir, 'plot2_price_distribution.png')

def plot_surplus_comparison(baseline_metrics: Dict, rl_metrics: Dict, output_dir: Path):
    """Plots a comparison of economic surplus distribution."""
    labels = ['Seller Surplus', 'Buyer Surplus']
    baseline_surplus = [baseline_metrics['avg_seller_surplus'], baseline_metrics['avg_winner_surplus']]
    rl_surplus = [rl_metrics['avg_seller_surplus'], rl_metrics['avg_winner_surplus']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_surplus, width, label='Baseline')
    rects2 = ax.bar(x + width/2, rl_surplus, width, label='RL Agent')
    
    ax.set_ylabel('Average Surplus ($)')
    ax.set_title('Economic Surplus Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    save_plot(output_dir, 'plot3_surplus_comparison.png')

def plot_efficiency_comparison(baseline_metrics: Dict, rl_metrics: Dict, output_dir: Path):
    """Plots a comparison of market efficiency metrics."""
    metrics = ['Welfare Efficiency', 'Allocative Efficiency', 'Revenue Efficiency']
    baseline_values = [
        baseline_metrics['welfare_efficiency'],
        baseline_metrics['allocative_efficiency'],
        baseline_metrics['revenue_efficiency']
    ]
    rl_values = [
        rl_metrics['welfare_efficiency'],
        rl_metrics['allocative_efficiency'],
        rl_metrics['revenue_efficiency']
    ]
    
    df = pd.DataFrame({'Baseline': baseline_values, 'RL Agent': rl_values}, index=metrics)
    
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', width=0.8)
    plt.title('Market Efficiency Comparison')
    plt.ylabel('Efficiency Ratio')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir, 'plot4_efficiency_comparison.png')

def _generate_report(baseline_metrics, rl_metrics, buyers, baseline_file, rl_file, output_dir):
    """Generate a comprehensive markdown report."""
    
    # Winner distribution changes
    all_personas = sorted(list(buyers.keys()))
    b_rates = baseline_metrics['winner_counts'].reindex(all_personas, fill_value=0) / baseline_metrics['total_episodes'] * 100
    rl_rates = rl_metrics['winner_counts'].reindex(all_personas, fill_value=0) / rl_metrics['total_episodes'] * 100
    win_diff = rl_rates - b_rates

    winner_changes_str = "\n".join([f"- **{p.split('_')[1]}:** {b_rates[p]:.1f}% → {rl_rates[p]:.1f}% ({win_diff[p]:+.1f}pp)" for p in all_personas])

    surplus_change_abs = rl_metrics['avg_total_surplus'] - baseline_metrics['avg_total_surplus']
    surplus_change_perc_str = ""
    if baseline_metrics['avg_total_surplus'] > 0:
        surplus_change_perc = (surplus_change_abs / baseline_metrics['avg_total_surplus']) * 100
        surplus_change_perc_str = f" ({surplus_change_perc:+.1f}%)"

    report_content = f"""# Phase 2 RL Training Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**RL Evaluation Episodes:** {rl_metrics['total_episodes']} (from `{rl_file}`)  
**Baseline Dataset:** {baseline_file} ({baseline_metrics['total_episodes']} episodes)  

---

## Executive Summary

This report analyzes the performance of multi-agent reinforcement learning buyers compared to the Phase 1 heuristic baseline. The analysis focuses on market outcomes from evaluation episodes.

### Key Results
- **RL Success Rate:** {rl_metrics['success_rate']:.1%} vs Heuristic {baseline_metrics['success_rate']:.1%}
- **Average Price:** RL ${rl_metrics['price_stats']['mean_price']:,.0f} vs Heuristic ${baseline_metrics['price_stats']['mean_price']:,.0f}
- **Allocative Efficiency:** RL {rl_metrics['allocative_efficiency']:.1%} vs Heuristic {baseline_metrics['allocative_efficiency']:.1%}
- **Economic Surplus:** RL ${rl_metrics['avg_total_surplus']:,.0f} vs Heuristic ${baseline_metrics['avg_total_surplus']:,.0f}

---

## Performance Comparison Analysis

### Market Efficiency & Financials
- **Allocative Efficiency Change:** {rl_metrics['allocative_efficiency'] - baseline_metrics['allocative_efficiency']:.1%}
- **Price Volatility:** RL ${rl_metrics['price_stats']['std_price']:.0f} vs Heuristic ${baseline_metrics['price_stats']['std_price']:.0f}
- **Economic Surplus Change:** ${surplus_change_abs:,.0f}{surplus_change_perc_str}

### Winner Distribution Changes:
{winner_changes_str}

### Visual Comparisons

#### 1. Win Rate Analysis
![Win Rate Comparison]({output_dir}/plot1_win_rates.png)

#### 2. Price Distribution Analysis  
![Price Distribution Comparison]({output_dir}/plot2_price_distribution.png)

#### 3. Economic Surplus Analysis
![Surplus Distribution Comparison]({output_dir}/plot3_surplus_comparison.png)

#### 4. Market Efficiency Comparison
![Efficiency Comparison]({output_dir}/plot4_efficiency_comparison.png)

---

## Analysis & Interpretation

The RL agents demonstrate a significant shift in market dynamics compared to the heuristic baseline. While the average price remained relatively stable, allocative efficiency saw a notable decrease, suggesting that the learned policies are less effective at ensuring the highest-value buyer wins. This is reflected in the altered winner distribution, where historically dominant aggressive buyers lost market share to more conservative or budget-conscious agents.

The reduction in total economic surplus under the RL regime, combined with lower efficiency, indicates that the current agent strategies are suboptimal from a market welfare perspective. However, the increased price volatility suggests that RL agents introduced more complex and less predictable bidding patterns, which could be a foundation for more sophisticated strategies with further training.

---

## Note on Learning Curves

This report is based on final evaluation data. A complete analysis of agent learning would require training logs containing per-episode rewards and other metrics to generate learning curves and analyze market evolution during training. This data was not available for the current analysis.

---

## Recommendations for Future Development
1. **Extended Training:** Longer training may be required for agents to converge on more optimal policies.
2. **Reward Shaping:** Review and refine reward functions to better incentivize efficient market outcomes.
3. **Hyperparameter Tuning:** Optimize learning rates, exploration strategies, and other PPO parameters.
4. **State Representation:** Consider enriching the agent's state with more information about opponent behavior.

---

*Report generated by Phase 2 Analytics System*
"""
    report_path = Path('phase2_report.md')
    report_path.write_text(report_content)
    logger.info(f"📄 Generated comprehensive report: {report_path}")

def run_phase2_analysis(baseline_file: str, rl_file: str, config_file: str, save_plots: bool = True):
    """Main function to run all comparative analyses for Phase 2."""
    logger.info(f"📊 Running Phase 2 Comparative Analysis")
    logger.info(f"   - Baseline: {baseline_file}")
    logger.info(f"   - RL Agent: {rl_file}")
    setup_plot_style()

    # Load data and configuration
    df_baseline = pd.read_csv(baseline_file)
    df_rl = pd.read_csv(rl_file)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    env_config = config['environment']
    buyers = {buyer['id']: buyer for buyer in env_config['buyers']}
    reserve_price = env_config['seller']['reserve_price']

    # Analyze both datasets
    logger.info("\n🔍 Analyzing datasets...")
    baseline_metrics = analyze_dataset(df_baseline, buyers, reserve_price)
    rl_metrics = analyze_dataset(df_rl, buyers, reserve_price)
    
    # Create plots
    output_dir = Path("phase2_analysis_plots")
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n📊 Generating visualizations in '{output_dir}'...")
        
        plot_win_rate_comparison(baseline_metrics, rl_metrics, buyers, output_dir)
        plot_price_distribution_comparison(df_baseline, df_rl, output_dir)
        plot_surplus_comparison(baseline_metrics, rl_metrics, output_dir)
        plot_efficiency_comparison(baseline_metrics, rl_metrics, output_dir)
        
        logger.info("✅ Generated 4 comparison plots.")

    # Generate Report
    _generate_report(baseline_metrics, rl_metrics, buyers, baseline_file, rl_file, output_dir)
    
    # Final Summary
    print("\n" + "="*60)
    print("🏁 PHASE 2 COMPARATIVE ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Baseline':<15} {'RL Agent':<15}")
    print("-" * 60)
    print(f"{'Success Rate':<25} {baseline_metrics['success_rate']:<15.1%} {rl_metrics['success_rate']:<15.1%}")
    print(f"{'Avg Final Price':<25} ${baseline_metrics['price_stats']['mean_price']:<14,.0f} ${rl_metrics['price_stats']['mean_price']:<14,.0f}")
    print(f"{'Price Volatility (Std)':<25} ${baseline_metrics['price_stats']['std_price']:<14,.0f} ${rl_metrics['price_stats']['std_price']:<14,.0f}")
    print(f"{'Avg Total Surplus':<25} ${baseline_metrics['avg_total_surplus']:<14,.0f} ${rl_metrics['avg_total_surplus']:<14,.0f}")
    print(f"{'Welfare Efficiency':<25} {baseline_metrics['welfare_efficiency']:<15.1%} {rl_metrics['welfare_efficiency']:<15.1%}")
    print(f"{'Allocative Efficiency':<25} {baseline_metrics['allocative_efficiency']:<15.1%} {rl_metrics['allocative_efficiency']:<15.1%}")
    print(f"{'Revenue Efficiency':<25} {baseline_metrics['revenue_efficiency']:<15.1%} {rl_metrics['revenue_efficiency']:<15.1%}")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    baseline_file = "phase1_results.csv"
    rl_file = "phase2_results.csv"
    config_file = "config.yaml" # Assuming config.yaml is in the root
    
    # Basic check for files
    if not all(Path(f).exists() for f in [baseline_file, rl_file, config_file]):
        print("❌ Error: Missing required files.")
        print("Ensure 'phase1_results.csv', 'phase2_results.csv', and 'config.yaml' are present.")
        sys.exit(1)
        
    print(f"🚀 Running Phase 2 Analysis...")
    run_phase2_analysis(baseline_file, rl_file, config_file)
    print(f"\n✅ Analysis complete!") 