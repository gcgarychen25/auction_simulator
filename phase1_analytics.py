"""
Phase 1 Analytics - Monte Carlo Baseline Analysis

Provides comprehensive analysis of heuristic policy performance including:
- Price and surplus analysis
- Economic welfare breakdown  
- Winner patterns and market efficiency
- Comparative visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
from datetime import datetime


def _generate_detailed_report(df, buyers, price_stats, avg_total_welfare, avg_seller_surplus, 
                            avg_buyer_surplus, seller_share, buyer_share, winner_counts, 
                            unique_prices, price_concentration, welfare_efficiency, 
                            allocative_efficiency, revenue_efficiency, highest_wtp_persona,
                            reserve_price, results_file):
    """Generate a comprehensive markdown report analyzing each visualization and metric."""
    
    report_content = f"""# Phase 1 Monte Carlo Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** {results_file}  
**Episodes Analyzed:** {len(df)}  

---

## Executive Summary

This report provides a comprehensive analysis of {len(df)} auction episodes using heuristic buyer policies. The analysis examines market efficiency, price dynamics, winner patterns, and economic welfare distribution through six key visualizations.

### Key Findings
- **Average Final Price:** ${price_stats['mean_price']:,.0f} ± ${price_stats['std_price']:.0f}
- **Market Efficiency:** {welfare_efficiency:.1%} welfare efficiency, {allocative_efficiency:.1%} allocative efficiency
- **Winner Diversity:** {len(winner_counts)} different buyer personas achieved victories
- **Economic Welfare:** ${avg_total_welfare:,.0f} total surplus with {seller_share:.1f}% going to seller

---

## Visualization Analysis

### 1. 📊 Price Distribution Analysis

![Price Distribution](plot1_price_distribution.png)

**What This Shows:** Histogram of final auction prices across all episodes, with mean price and reserve price marked.

**Key Metrics:**
- **Price Range:** ${price_stats['min_price']:,.0f} - ${price_stats['max_price']:,.0f}
- **Standard Deviation:** ${price_stats['std_price']:.0f}
- **Price Variance:** {(price_stats['std_price']/price_stats['mean_price']*100):.1f}% coefficient of variation

**Interpretation:**
{_analyze_price_distribution(price_stats, reserve_price, unique_prices)}

**Market Implications:**
{_price_distribution_implications(price_stats, unique_prices)}

---

### 2. 🏆 Winner Distribution Analysis

![Winner Distribution](plot2_winner_distribution.png)

**What This Shows:** Pie chart showing the percentage of auctions won by each buyer persona.

**Key Metrics:**
{_format_winner_metrics(winner_counts, len(df), buyers)}

**Interpretation:**
{_analyze_winner_distribution(winner_counts, len(df), buyers, highest_wtp_persona, allocative_efficiency)}

**Strategic Insights:**
{_winner_distribution_insights(winner_counts, buyers)}

---

### 3. 💰 Economic Surplus Distribution

![Economic Surplus Distribution](plot3_surplus_distribution.png)

**What This Shows:** Breakdown of total economic welfare between seller and winning buyers.

**Key Metrics:**
- **Total Economic Welfare:** ${avg_total_welfare:,.0f} per auction
- **Seller Share:** ${avg_seller_surplus:,.0f} ({seller_share:.1f}%)
- **Buyer Share:** ${avg_buyer_surplus:,.0f} ({buyer_share:.1f}%)
- **Surplus Ratio:** {seller_share/buyer_share:.2f}:1 (seller:buyer)

**Interpretation:**
{_analyze_surplus_distribution(seller_share, buyer_share, avg_total_welfare, reserve_price)}

**Economic Implications:**
{_surplus_distribution_implications(seller_share, buyer_share)}

---

### 4. ⏱️ Episode Length Distribution

![Episode Length Distribution](plot4_episode_length.png)

**What This Shows:** Distribution of auction durations in rounds.

**Key Metrics:**
- **Average Duration:** {df['episode_length'].mean():.1f} rounds
- **Range:** {df['episode_length'].min()}-{df['episode_length'].max()} rounds
- **Standard Deviation:** {df['episode_length'].std():.1f} rounds

**Interpretation:**
{_analyze_episode_length(df)}

**Efficiency Insights:**
{_episode_length_insights(df)}

---

### 5. 📈 Price Consistency Analysis

![Price Consistency](plot5_price_consistency.png)

**What This Shows:** Final prices plotted against episode number to detect trends or patterns.

**Key Metrics:**
- **Price Concentration:** {price_concentration:.1%} at most common price
- **Unique Price Points:** {unique_prices}
- **Temporal Stability:** {_calculate_price_stability(df)}

**Interpretation:**
{_analyze_price_consistency(df, price_concentration, unique_prices)}

**Pattern Recognition:**
{_price_consistency_insights(df, unique_prices)}

---

### 6. ⚡ Market Efficiency Metrics

![Market Efficiency Metrics](plot6_efficiency_metrics.png)

**What This Shows:** Bar chart comparing three key efficiency measures.

**Key Metrics:**
- **Welfare Efficiency:** {welfare_efficiency:.1%} (actual vs theoretical maximum welfare)
- **Allocative Efficiency:** {allocative_efficiency:.1%} (highest-value buyer wins)
- **Revenue Efficiency:** {revenue_efficiency:.1%} (seller revenue optimization)

**Interpretation:**
{_analyze_efficiency_metrics(welfare_efficiency, allocative_efficiency, revenue_efficiency)}

**Benchmarking:**
{_efficiency_benchmarking(welfare_efficiency, allocative_efficiency, revenue_efficiency)}

---

## Competitive Dynamics Analysis

### Market Power Distribution
{_analyze_market_power(winner_counts, buyers)}

### Buyer Persona Performance
{_analyze_persona_performance(winner_counts, buyers, df)}

### Auction Mechanism Effectiveness
{_analyze_mechanism_effectiveness(welfare_efficiency, allocative_efficiency, avg_total_welfare)}

---

## Implications for Phase 2 (RL Development)

### Baseline Targets
- **Performance Benchmark:** {allocative_efficiency:.1%} allocative efficiency to match or exceed
- **Welfare Target:** ${avg_total_welfare:,.0f} total surplus per auction
- **Competition Level:** Must compete against {winner_counts.iloc[0]:.0f}% win rate by strongest persona

### Strategic Opportunities
{_phase2_opportunities(winner_counts, allocative_efficiency, welfare_efficiency)}

### Training Considerations
{_phase2_training_considerations(df, winner_counts, price_stats)}

---

## Methodology Notes

### Data Collection
- **Episodes:** {len(df)} independent auction simulations
- **Policy Type:** Heuristic rule-based agents with persona variation
- **Environment:** Gymnasium-compliant auction environment
- **Randomization:** {_describe_randomization_level(price_stats)}

### Metrics Definitions
- **Welfare Efficiency:** Ratio of actual to theoretical maximum economic welfare
- **Allocative Efficiency:** Frequency of highest-WTP buyer winning
- **Revenue Efficiency:** Seller revenue as fraction of maximum possible
- **Economic Surplus:** Consumer + producer surplus (total welfare)

### Limitations
{_describe_limitations(df, len(winner_counts))}

---

*Report generated by Phase 1 Analytics System*
"""

    # Save the report
    with open('phase1_report.md', 'w') as f:
        f.write(report_content)
    
    print("📄 Generated comprehensive report: phase1_report.md")


def _analyze_price_distribution(price_stats, reserve_price, unique_prices):
    """Generate price distribution analysis."""
    cv = price_stats['std_price'] / price_stats['mean_price'] * 100
    
    if cv < 5:
        volatility = "very low volatility, indicating highly predictable market outcomes"
    elif cv < 10:
        volatility = "low volatility with consistent pricing patterns"
    elif cv < 20:
        volatility = "moderate volatility showing healthy market dynamics"
    else:
        volatility = "high volatility indicating diverse competitive scenarios"
    
    price_premium = (price_stats['mean_price'] - reserve_price) / reserve_price * 100
    
    return f"""The price distribution shows {volatility}. With {unique_prices} distinct price points and a {price_premium:.1f}% premium above reserve price, the market demonstrates {'efficient' if unique_prices <= 3 else 'diverse'} price discovery. The standard deviation of ${price_stats['std_price']:.0f} represents {cv:.1f}% of the mean price, indicating {'consistent' if cv < 10 else 'variable'} auction outcomes."""


def _price_distribution_implications(price_stats, unique_prices):
    """Generate price distribution implications."""
    if unique_prices <= 2:
        return "The limited price variation suggests strong convergence in buyer strategies, which may indicate either highly effective heuristics or insufficient market complexity for diverse outcomes."
    elif unique_prices <= 4:
        return "The moderate price variety indicates healthy competition with multiple equilibrium points, suggesting the auction mechanism effectively captures different market conditions."
    else:
        return "The high price diversity demonstrates complex market dynamics with multiple competitive scenarios, indicating robust buyer strategy variation."


def _format_winner_metrics(winner_counts, total_episodes, buyers):
    """Format winner distribution metrics."""
    metrics = []
    for persona, count in winner_counts.items():
        percentage = count / total_episodes * 100
        max_wtp = buyers[persona]['max_wtp']
        win_efficiency = percentage / 100  # Win rate as efficiency measure
        metrics.append(f"- **{persona}:** {count}/{total_episodes} wins ({percentage:.1f}%) - Max WTP: ${max_wtp:,}")
    
    return "\n".join(metrics)


def _analyze_winner_distribution(winner_counts, total_episodes, buyers, highest_wtp_persona, allocative_efficiency):
    """Generate winner distribution analysis."""
    dominant_persona = winner_counts.index[0]
    dominant_percentage = winner_counts.iloc[0] / total_episodes * 100
    winner_diversity = len(winner_counts)
    
    dominance_level = "monopolistic" if dominant_percentage > 90 else "dominant" if dominant_percentage > 75 else "competitive" if dominant_percentage > 50 else "balanced"
    
    efficiency_note = f"The allocative efficiency of {allocative_efficiency:.1%} indicates that the highest-WTP buyer ({highest_wtp_persona}) wins {allocative_efficiency:.1%} of auctions, which is {'optimal' if allocative_efficiency > 0.9 else 'good' if allocative_efficiency > 0.7 else 'suboptimal'}."
    
    return f"""The market shows a {dominance_level} structure with {dominant_persona} winning {dominant_percentage:.1f}% of auctions. With {winner_diversity} different winners observed, the market demonstrates {'limited' if winner_diversity <= 2 else 'moderate' if winner_diversity <= 3 else 'high'} competitive diversity. {efficiency_note}"""


def _winner_distribution_insights(winner_counts, buyers):
    """Generate strategic insights from winner distribution."""
    winner_wtps = [(persona, buyers[persona]['max_wtp'], count) for persona, count in winner_counts.items()]
    winner_wtps.sort(key=lambda x: x[1], reverse=True)  # Sort by WTP
    
    if winner_wtps[0][0] == winner_counts.index[0]:  # Highest WTP wins most
        return "The results align with economic theory: higher willingness-to-pay generally translates to market success. This suggests the auction mechanism effectively allocates resources to highest-value buyers."
    else:
        return "Interestingly, the most frequent winner is not the highest-WTP buyer, indicating that strategic bidding behavior and risk preferences significantly impact outcomes beyond raw purchasing power."


def _analyze_surplus_distribution(seller_share, buyer_share, avg_total_welfare, reserve_price):
    """Analyze economic surplus distribution."""
    if seller_share > 70:
        market_power = "strong seller advantage"
    elif seller_share > 60:
        market_power = "moderate seller advantage"
    elif seller_share > 50:
        market_power = "balanced market power"
    else:
        market_power = "buyer-favorable conditions"
    
    welfare_level = "high" if avg_total_welfare > reserve_price * 0.5 else "moderate" if avg_total_welfare > reserve_price * 0.3 else "limited"
    
    return f"""The surplus distribution reveals {market_power}, with sellers capturing {seller_share:.1f}% of total economic value. This {welfare_level} welfare generation of ${avg_total_welfare:,.0f} per auction indicates {'efficient' if seller_share < 70 else 'seller-dominated'} market dynamics. The {seller_share/buyer_share:.1f}:1 seller-to-buyer surplus ratio suggests {'competitive' if seller_share < 65 else 'monopolistic'} pricing power."""


def _surplus_distribution_implications(seller_share, buyer_share):
    """Generate surplus distribution implications."""
    if seller_share > 70:
        return "The high seller surplus share may indicate limited buyer competition or highly effective seller pricing strategies. This could signal opportunities for more aggressive buyer policies in RL training."
    elif seller_share < 50:
        return "The buyer-favorable surplus distribution suggests intense competition among buyers or conservative seller pricing. This indicates a competitive market environment."
    else:
        return "The balanced surplus distribution indicates healthy market dynamics with effective competition benefiting both sides of the transaction."


def _analyze_episode_length(df):
    """Analyze episode length patterns."""
    avg_length = df['episode_length'].mean()
    length_variety = df['episode_length'].nunique()
    
    if avg_length < 4:
        speed = "very fast"
    elif avg_length < 6:
        speed = "efficient"
    elif avg_length < 8:
        speed = "moderate"
    else:
        speed = "extended"
    
    return f"""Auctions complete in an average of {avg_length:.1f} rounds, indicating {speed} market clearing. With {length_variety} different duration patterns observed, the market shows {'consistent' if length_variety <= 2 else 'variable'} convergence timing. This suggests {'aggressive' if avg_length < 5 else 'strategic'} bidding behavior leading to {'rapid' if avg_length < 5 else 'deliberate'} price discovery."""


def _episode_length_insights(df):
    """Generate episode length insights."""
    if df['episode_length'].std() < 1:
        return "The consistent episode duration indicates predictable auction dynamics, which may suggest opportunities for strategic timing manipulation in RL agents."
    else:
        return "The variable episode duration reflects diverse competitive scenarios, indicating that auction length is influenced by the specific mix of buyer behaviors and market conditions."


def _calculate_price_stability(df):
    """Calculate price stability measure."""
    price_changes = abs(df['final_price'].diff()).sum()
    max_possible_change = (df['final_price'].max() - df['final_price'].min()) * len(df)
    if max_possible_change == 0:
        return "Perfect stability (no price variation)"
    stability = 1 - (price_changes / max_possible_change)
    return f"{stability:.1%} stability index"


def _analyze_price_consistency(df, price_concentration, unique_prices):
    """Analyze price consistency patterns."""
    if price_concentration > 0.9:
        consistency = "extremely high price consistency"
        implication = "indicating convergent market behavior"
    elif price_concentration > 0.8:
        consistency = "high price consistency"
        implication = "suggesting stable competitive dynamics"
    elif price_concentration > 0.6:
        consistency = "moderate price consistency"
        implication = "showing some market variability"
    else:
        consistency = "low price consistency"
        implication = "indicating diverse market outcomes"
    
    return f"""The analysis reveals {consistency} with {price_concentration:.1%} of auctions settling at the most common price. {implication.capitalize()}. The presence of {unique_prices} distinct price points {'confirms limited' if unique_prices <= 3 else 'demonstrates significant'} market variation across episodes."""


def _price_consistency_insights(df, unique_prices):
    """Generate price consistency insights."""
    if unique_prices <= 2:
        return "The limited price variation suggests either highly optimized strategies or insufficient market complexity. RL agents should focus on finding strategies that can break these patterns."
    else:
        return "The price variation indicates multiple market equilibria, suggesting that different buyer combinations and strategies lead to different outcomes - a rich environment for RL exploration."


def _analyze_efficiency_metrics(welfare_efficiency, allocative_efficiency, revenue_efficiency):
    """Analyze market efficiency metrics."""
    overall_efficiency = (welfare_efficiency + allocative_efficiency + revenue_efficiency) / 3
    
    if overall_efficiency > 0.85:
        market_performance = "highly efficient market"
    elif overall_efficiency > 0.7:
        market_performance = "efficient market"
    elif overall_efficiency > 0.5:
        market_performance = "moderately efficient market"
    else:
        market_performance = "inefficient market"
    
    strongest_metric = max([
        ("welfare", welfare_efficiency),
        ("allocative", allocative_efficiency), 
        ("revenue", revenue_efficiency)
    ], key=lambda x: x[1])
    
    return f"""The market demonstrates {market_performance} with an average efficiency of {overall_efficiency:.1%} across all metrics. {strongest_metric[0].capitalize()} efficiency is the strongest at {strongest_metric[1]:.1%}, while the overall balance suggests {'well-functioning' if overall_efficiency > 0.7 else 'room for improvement in'} auction mechanisms."""


def _efficiency_benchmarking(welfare_efficiency, allocative_efficiency, revenue_efficiency):
    """Provide efficiency benchmarking context."""
    benchmarks = []
    
    if welfare_efficiency > 0.9:
        benchmarks.append("Welfare efficiency exceeds typical market standards (>90%)")
    elif welfare_efficiency > 0.8:
        benchmarks.append("Welfare efficiency meets good market standards (80-90%)")
    else:
        benchmarks.append("Welfare efficiency below optimal standards (<80%)")
    
    if allocative_efficiency > 0.8:
        benchmarks.append("Allocative efficiency demonstrates effective price discovery")
    else:
        benchmarks.append("Allocative efficiency suggests room for strategic improvement")
    
    return " • ".join(benchmarks) + "."


def _analyze_market_power(winner_counts, buyers):
    """Analyze market power distribution."""
    total_wins = winner_counts.sum()
    market_concentration = (winner_counts.iloc[0] / total_wins) ** 2 * 10000  # Simplified HHI
    
    power_analysis = f"""Market concentration analysis reveals a {'highly concentrated' if market_concentration > 2500 else 'moderately concentrated' if market_concentration > 1500 else 'competitive'} structure. """
    
    # Analyze relationship between WTP and success
    winner_wtps = [(persona, buyers[persona]['max_wtp']) for persona in winner_counts.index]
    wtp_correlation = "Strong" if winner_wtps[0][1] == max(buyer['max_wtp'] for buyer in buyers.values()) else "Weak"
    
    power_analysis += f"""{wtp_correlation} correlation between maximum WTP and market success suggests that {'economic fundamentals drive' if wtp_correlation == "Strong" else 'strategic behavior influences'} outcomes."""
    
    return power_analysis


def _analyze_persona_performance(winner_counts, buyers, df):
    """Analyze individual persona performance."""
    performance_analysis = []
    
    for persona, wins in winner_counts.items():
        win_rate = wins / len(df) * 100
        max_wtp = buyers[persona]['max_wtp']
        risk_aversion = buyers[persona]['risk_aversion']
        
        # Performance relative to WTP ranking
        wtp_rank = sorted(buyers.items(), key=lambda x: x[1]['max_wtp'], reverse=True)
        wtp_position = [i for i, (p, _) in enumerate(wtp_rank) if p == persona][0] + 1
        
        performance = "overperforming" if win_rate > (6-wtp_position)*20 else "underperforming" if win_rate < (6-wtp_position)*15 else "expected"
        
        performance_analysis.append(f"**{persona}:** {performance} with {win_rate:.1f}% wins (WTP rank #{wtp_position}, risk aversion: {risk_aversion})")
    
    return "\n".join(performance_analysis)


def _analyze_mechanism_effectiveness(welfare_efficiency, allocative_efficiency, avg_total_welfare):
    """Analyze auction mechanism effectiveness."""
    if welfare_efficiency > 0.9 and allocative_efficiency > 0.8:
        return f"""The auction mechanism demonstrates high effectiveness with {welfare_efficiency:.1%} welfare efficiency and {allocative_efficiency:.1%} allocative efficiency. The average welfare of ${avg_total_welfare:,.0f} per auction indicates strong value creation. This suggests the English auction format with current rules effectively facilitates price discovery and resource allocation."""
    
    elif welfare_efficiency > 0.7 or allocative_efficiency > 0.7:
        return f"""The auction mechanism shows good effectiveness with mixed efficiency results. Areas for potential improvement include {'allocation mechanisms' if allocative_efficiency < 0.7 else 'welfare optimization'}. The mechanism successfully generates ${avg_total_welfare:,.0f} average welfare per auction."""
    
    else:
        return f"""The auction mechanism exhibits suboptimal effectiveness with room for significant improvement in both welfare and allocative efficiency. Consider mechanism design changes to improve the ${avg_total_welfare:,.0f} average welfare generation."""


def _phase2_opportunities(winner_counts, allocative_efficiency, welfare_efficiency):
    """Identify Phase 2 RL opportunities."""
    opportunities = []
    
    if allocative_efficiency < 0.9:
        opportunities.append(f"**Allocative Improvement:** Current {allocative_efficiency:.1%} efficiency leaves room for RL agents to improve winner selection")
    
    if len(winner_counts) <= 2:
        opportunities.append("**Market Diversification:** Limited winner variety suggests opportunities for new strategic approaches")
    
    if welfare_efficiency < 0.95:
        opportunities.append(f"**Welfare Optimization:** {welfare_efficiency:.1%} efficiency indicates potential for better surplus extraction")
    
    dominant_share = winner_counts.iloc[0] / winner_counts.sum()
    if dominant_share > 0.8:
        opportunities.append("**Competitive Disruption:** High market concentration creates opportunities for disruptive RL strategies")
    
    return "\n".join(opportunities) if opportunities else "Market shows high efficiency across all metrics, requiring sophisticated RL strategies to achieve improvements."


def _phase2_training_considerations(df, winner_counts, price_stats):
    """Provide Phase 2 training considerations."""
    considerations = []
    
    if price_stats['std_price'] < 100:
        considerations.append("**Exploration Challenge:** Low price variance requires RL agents capable of breaking established patterns")
    
    if len(winner_counts) <= 2:
        considerations.append("**Strategy Diversity:** Limited winner types suggest need for diverse training scenarios")
    
    avg_length = df['episode_length'].mean()
    if avg_length < 5:
        considerations.append("**Rapid Decision Making:** Short episodes require RL agents optimized for quick convergence")
    elif avg_length > 7:
        considerations.append("**Strategic Patience:** Longer episodes allow for complex multi-round strategies")
    
    considerations.append("**Baseline Competition:** RL agents must outperform established heuristic strategies with known win rates")
    
    return "\n".join(considerations)


def _describe_randomization_level(price_stats):
    """Describe the level of randomization in the system."""
    cv = price_stats['std_price'] / price_stats['mean_price'] * 100
    if cv < 5:
        return "Low randomization with highly consistent outcomes"
    elif cv < 15:
        return "Moderate randomization with controlled variability"
    else:
        return "High randomization with significant outcome diversity"


def _describe_limitations(df, winner_variety):
    """Describe analysis limitations."""
    limitations = []
    
    if len(df) < 50:
        limitations.append("Small sample size may limit statistical significance")
    
    if winner_variety <= 2:
        limitations.append("Limited winner diversity may not fully represent market potential")
    
    limitations.append("Heuristic policies may not capture full range of possible strategic behavior")
    limitations.append("Analysis assumes current auction rules and environment constraints")
    
    return " • ".join(limitations)


def run_phase1_analysis(results_file: str = "phase1_results.csv", 
                       config_file: str = "config.yaml",
                       save_plots: bool = True) -> Dict[str, Any]:
    """
    Run complete Phase 1 analysis on batch simulation results.
    
    Args:
        results_file: Path to CSV with simulation results
        config_file: Path to configuration YAML
        save_plots: Whether to save visualization plots
        
    Returns:
        Complete analysis report dictionary
    """
    
    print(f"📊 Running Phase 1 Analysis on {results_file}")
    
    # Load data
    df = pd.read_csv(results_file)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract buyer configurations for analysis
    buyers = {buyer['id']: buyer for buyer in config['buyers']}
    reserve_price = config['seller']['reserve_price']
    start_price = config['auction']['start_price']
    
    print(f"⚙️  Configuration: {len(buyers)} buyers, reserve ${reserve_price:,}")
    print(f"📈 Analyzing {len(df)} episodes")
    
    # Basic Statistics
    print("\n" + "="*60)
    print("📈 BASIC STATISTICS")
    print("="*60)
    
    price_stats = {
        'mean_price': df['final_price'].mean(),
        'std_price': df['final_price'].std(),
        'min_price': df['final_price'].min(),
        'max_price': df['final_price'].max(),
        'median_price': df['final_price'].median()
    }
    
    print(f"💰 Average Final Price: ${price_stats['mean_price']:,.0f} ± ${price_stats['std_price']:.0f}")
    print(f"📊 Price Range: ${price_stats['min_price']:,.0f} - ${price_stats['max_price']:,.0f}")
    print(f"📈 Median Price: ${price_stats['median_price']:,.0f}")
    print(f"💎 Average Total Surplus: ${df['total_surplus'].mean():,.0f}")
    print(f"⏱️  Average Episode Length: {df['episode_length'].mean():.1f} rounds")
    
    # Economic Welfare Analysis
    print("\n" + "="*60)
    print("💎 ECONOMIC WELFARE ANALYSIS")
    print("="*60)
    
    # Calculate buyer surplus for each episode
    buyer_surplus_data = []
    for _, row in df.iterrows():
        winner_persona = row['winner_persona']
        winner_surplus = row['total_surplus'] - row['seller_reward']
        winner_max_wtp = buyers[winner_persona]['max_wtp']
        
        buyer_surplus_data.append({
            'episode': row['episode'],
            'winner': winner_persona,
            'winner_surplus': winner_surplus,
            'seller_surplus': row['seller_reward'],
            'total_surplus': row['total_surplus']
        })
    
    surplus_df = pd.DataFrame(buyer_surplus_data)
    
    avg_seller_surplus = surplus_df['seller_surplus'].mean()
    avg_buyer_surplus = surplus_df['winner_surplus'].mean()
    avg_total_welfare = surplus_df['total_surplus'].mean()
    
    seller_share = (avg_seller_surplus / avg_total_welfare) * 100
    buyer_share = (avg_buyer_surplus / avg_total_welfare) * 100
    
    print(f"💰 Total Economic Welfare: ${avg_total_welfare:,.0f}")
    print(f"🏛️  Seller Share: ${avg_seller_surplus:,.0f} ({seller_share:.1f}%)")
    print(f"🛒 Buyer Share: ${avg_buyer_surplus:,.0f} ({buyer_share:.1f}%)")
    
    # Market Patterns
    print("\n" + "="*60)
    print("🏆 MARKET PATTERNS")
    print("="*60)
    
    winner_counts = df['winner_persona'].value_counts()
    print(f"🎯 Winner Distribution:")
    for persona, count in winner_counts.items():
        percentage = (count / len(df)) * 100
        max_wtp = buyers[persona]['max_wtp']
        print(f"   • {persona}: {count}/{len(df)} ({percentage:.1f}%) - Max WTP: ${max_wtp:,}")
    
    unique_prices = df['final_price'].nunique()
    most_common_price = df['final_price'].mode().iloc[0]
    price_concentration = (df['final_price'] == most_common_price).mean()
    
    print(f"\n💰 Price Patterns:")
    print(f"   • Unique Prices: {unique_prices}")
    print(f"   • Most Common: ${most_common_price:,.0f}")
    print(f"   • Concentration: {price_concentration:.1%}")
    
    # Market Efficiency
    print("\n" + "="*60)
    print("⚡ MARKET EFFICIENCY")
    print("="*60)
    
    # Calculate efficiency metrics
    max_possible_wtp = max(buyer['max_wtp'] for buyer in buyers.values())
    theoretical_max_welfare = max_possible_wtp - reserve_price
    
    welfare_efficiency = avg_total_welfare / theoretical_max_welfare
    
    highest_wtp_persona = max(buyers.items(), key=lambda x: x[1]['max_wtp'])[0]
    allocative_efficiency = (df['winner_persona'] == highest_wtp_persona).mean()
    
    revenue_efficiency = avg_seller_surplus / (max_possible_wtp - reserve_price)
    
    print(f"💎 Welfare Efficiency: {welfare_efficiency:.3f} ({welfare_efficiency*100:.1f}%)")
    print(f"🎯 Allocative Efficiency: {allocative_efficiency:.3f} ({allocative_efficiency*100:.1f}%)")
    print(f"💰 Revenue Efficiency: {revenue_efficiency:.3f} ({revenue_efficiency*100:.1f}%)")
    print(f"🏆 Highest WTP Buyer ({highest_wtp_persona}) Win Rate: {allocative_efficiency:.1%}")
    
    # Create Individual Visualizations
    if save_plots:
        print("\n📊 GENERATING INDIVIDUAL VISUALIZATIONS")
        print("-" * 40)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        plot_files = []
        
        # 1. Price Distribution
        plt.figure(figsize=(10, 6))
        df['final_price'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(df['final_price'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${df["final_price"].mean():,.0f}')
        plt.axvline(reserve_price, color='green', linestyle='--', 
                   label=f'Reserve: ${reserve_price:,}')
        plt.xlabel('Final Price ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Final Prices')
        plt.legend()
        plt.tight_layout()
        filename = 'plot1_price_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
        plt.close()
        
        # 2. Winner Distribution
        plt.figure(figsize=(8, 8))
        winner_counts = df['winner_persona'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(winner_counts)))
        plt.pie(winner_counts.values, labels=winner_counts.index, 
               autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Winner Distribution')
        plt.tight_layout()
        filename = 'plot2_winner_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
        plt.close()
        
        # 3. Surplus Breakdown
        plt.figure(figsize=(8, 8))
        surplus_data = [avg_seller_surplus, avg_buyer_surplus]
        surplus_labels = ['Seller Surplus', 'Buyer Surplus']
        colors = ['lightcoral', 'lightblue']
        plt.pie(surplus_data, labels=surplus_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Economic Surplus Distribution')
        plt.tight_layout()
        filename = 'plot3_surplus_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
        plt.close()
        
        # 4. Episode Length Distribution  
        plt.figure(figsize=(10, 6))
        df['episode_length'].hist(bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(df['episode_length'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["episode_length"].mean():.1f} rounds')
        plt.xlabel('Episode Length (rounds)')
        plt.ylabel('Frequency')
        plt.title('Auction Duration Distribution')
        plt.legend()
        plt.tight_layout()
        filename = 'plot4_episode_length.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
        plt.close()
        
        # 5. Price vs Episode (sample for large datasets)
        plt.figure(figsize=(12, 6))
        if len(df) > 100:
            # Sample episodes for readability
            sample_df = df.sample(n=min(100, len(df)), random_state=42).sort_values('episode')
            plt.plot(sample_df['episode'], sample_df['final_price'], 'o-', alpha=0.7, markersize=4)
            plt.title(f'Price Consistency Across Episodes (Sample of {len(sample_df)} episodes)')
        else:
            plt.plot(df['episode'], df['final_price'], 'o-', alpha=0.7)
            plt.title('Price Consistency Across Episodes')
        
        plt.axhline(df['final_price'].mean(), color='red', linestyle='--', alpha=0.7, 
                   label=f'Overall Mean: ${df["final_price"].mean():,.0f}')
        plt.xlabel('Episode Number')
        plt.ylabel('Final Price ($)')
        plt.legend()
        plt.tight_layout()
        filename = 'plot5_price_consistency.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
        plt.close()
        
        # 6. Efficiency Metrics
        plt.figure(figsize=(10, 6))
        metrics = ['Welfare\nEfficiency', 'Allocative\nEfficiency', 'Revenue\nEfficiency']
        values = [welfare_efficiency, allocative_efficiency, revenue_efficiency]
        colors = ['gold', 'lightblue', 'lightcoral']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        plt.ylabel('Efficiency Ratio')
        plt.title('Market Efficiency Metrics')
        plt.ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = 'plot6_efficiency_metrics.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plot_files.append(filename)
        print(f"📁 Saved: {filename}")
        plt.close()
        
        print(f"✅ Generated {len(plot_files)} individual visualization files")
        
        # Generate detailed markdown report
        _generate_detailed_report(df, buyers, price_stats, avg_total_welfare, avg_seller_surplus, 
                                avg_buyer_surplus, seller_share, buyer_share, winner_counts, 
                                unique_prices, price_concentration, welfare_efficiency, 
                                allocative_efficiency, revenue_efficiency, highest_wtp_persona,
                                reserve_price, results_file)
    
    # Final Summary
    print("\n" + "="*60)
    print("🏁 PHASE 1 MONTE CARLO ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total Episodes: {len(df)}")
    print(f"   • Success Rate: {df['auction_successful'].mean():.1%}")
    print(f"   • Reserve Met Rate: {df['reserve_met'].mean():.1%}")
    
    print(f"\n💰 FINANCIAL SUMMARY:")
    print(f"   • Average Price: ${price_stats['mean_price']:,.0f}")
    print(f"   • Price Volatility: ${price_stats['std_price']:.0f}")
    print(f"   • Total Economic Welfare: ${avg_total_welfare:,.0f}")
    
    print(f"\n⚡ EFFICIENCY METRICS:")
    print(f"   • Welfare Efficiency: {welfare_efficiency:.1%}")
    print(f"   • Allocative Efficiency: {allocative_efficiency:.1%}")
    print(f"   • Revenue Efficiency: {revenue_efficiency:.1%}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    if price_stats['std_price'] < 100:
        print("   • Highly consistent pricing (low volatility)")
    if allocative_efficiency > 0.8:
        print("   • Efficient allocation (high-value buyers winning)")
    if seller_share > 60:
        print("   • Seller-favorable surplus distribution")
    
    print("="*60)
    
    return {
        'price_statistics': price_stats,
        'welfare_metrics': {
            'total_welfare': avg_total_welfare,
            'seller_share': avg_seller_surplus,
            'buyer_share': avg_buyer_surplus,
            'seller_percentage': seller_share,
            'buyer_percentage': buyer_share
        },
        'efficiency_metrics': {
            'welfare_efficiency': welfare_efficiency,
            'allocative_efficiency': allocative_efficiency,
            'revenue_efficiency': revenue_efficiency
        },
        'market_patterns': {
            'winner_distribution': winner_counts.to_dict(),
            'price_consistency': {
                'unique_prices': unique_prices,
                'concentration': price_concentration
            }
        }
    }


if __name__ == "__main__":
    import sys
    
    results_file = sys.argv[1] if len(sys.argv) > 1 else "phase1_results.csv"
    
    if Path(results_file).exists():
        print(f"🚀 Running Phase 1 Analysis on {results_file}")
        report = run_phase1_analysis(results_file)
        print(f"\n✅ Analysis complete!")
    else:
        print(f"❌ Results file {results_file} not found!")
        print("💡 Run batch simulation first: python run.py --episodes 50 --output phase1_results.csv")
