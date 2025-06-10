"""
Shared Analytics Utilities

Provides common functions for metric calculation and plotting used across
Phase 1 and Phase 2 analytics scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def setup_plot_style():
    """Sets a consistent and professional style for all plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.dpi': 100, 
        'savefig.dpi': 300,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18
    })

def save_plot(output_dir: Path, filename: str):
    """Saves the current matplotlib plot to a specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    plt.savefig(filepath, bbox_inches='tight')
    plt.close() # close the current figure
    logger.info(f"📁 Saved plot: {filepath}")

def calculate_allocative_efficiency(df: pd.DataFrame, buyers: Dict[str, Dict]) -> float:
    """
    Calculate allocative efficiency: the frequency of the highest-WTP buyer winning.
    
    This is calculated as the proportion of *successful* auctions won by the 
    buyer with the highest willingness-to-pay.
    """
    if df is None or df.empty or 'auction_successful' not in df.columns:
        return 0.0
    
    successful_auctions = df[df['auction_successful']].copy()
    if successful_auctions.empty:
        return 0.0
        
    try:
        highest_wtp_buyer = max(buyers.items(), key=lambda x: x[1]['max_wtp'])[0]
    except (ValueError, KeyError):
        return 0.0 # No buyers or malformed buyer dict

    highest_wtp_wins = (successful_auctions['winner_persona'] == highest_wtp_buyer).sum()
    
    return highest_wtp_wins / len(successful_auctions)


def calculate_welfare_efficiency(df: pd.DataFrame, buyers: Dict[str, Dict], reserve_price: float) -> float:
    """
    Calculates welfare efficiency: the ratio of actual surplus to max possible surplus.
    """
    if df is None or df.empty or 'auction_successful' not in df.columns:
        return 0.0

    successful_auctions = df[df['auction_successful']]
    if successful_auctions.empty:
        return 0.0

    avg_total_surplus = successful_auctions['total_surplus'].mean()
    
    try:
        max_possible_wtp = max(buyer['max_wtp'] for buyer in buyers.values())
    except ValueError:
        return 0.0 # No buyers
        
    theoretical_max_welfare = max_possible_wtp - reserve_price

    if theoretical_max_welfare <= 0:
        return 0.0

    return avg_total_surplus / theoretical_max_welfare

def calculate_revenue_efficiency(df: pd.DataFrame, buyers: Dict[str, Dict], reserve_price: float) -> float:
    """
    Calculates revenue efficiency: seller revenue as a fraction of maximum possible welfare.
    """
    if df is None or df.empty or 'auction_successful' not in df.columns:
        return 0.0
        
    successful_auctions = df[df['auction_successful']]
    if successful_auctions.empty:
        return 0.0
        
    avg_seller_surplus = successful_auctions['seller_reward'].mean()
    
    try:
        max_possible_wtp = max(buyer['max_wtp'] for buyer in buyers.values())
    except ValueError:
        return 0.0 # No buyers

    theoretical_max_welfare = max_possible_wtp - reserve_price

    if theoretical_max_welfare <= 0:
        return 0.0

    return avg_seller_surplus / theoretical_max_welfare 