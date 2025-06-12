"""
Main Orchestration Script for the Multi-Agent Auction Simulator.

This script drives a single auction episode using a multi-agent framework,
leveraging LangGraph for orchestration. It loads the auction configuration,
runs the simulation, and prints the final results.
"""

# Load environment variables from .env file
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print(
        "⚠️  python-dotenv not installed. Please install with: pip install python-dotenv"
    )
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

import asyncio
import argparse
import logging
import traceback
import subprocess
import sys
import webbrowser
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from utils import load_config
from graph import run_auction_episode

# Configure logging with improved formatting
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose HTTP and other noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.CRITICAL)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Agent Auction Simulator")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for detailed round-by-round output.",
    )
    parser.add_argument(
        "--live", action="store_true", help="Enable live visualization dashboard."
    )

    args = parser.parse_args()
    return args


async def main():
    """Main entry point for the multi-agent simulation."""
    args = parse_args()

    console = Console()

    # --- Set up Logging ---
    # Configure root logger for console output
    root_logger = logging.getLogger()
    if args.verbose:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

    console.print(
        Panel(
            Text(
                "Multi-Agent Auction Simulator", justify="center", style="bold magenta"
            ),
            title="Welcome",
        )
    )
    console.print("ℹ️  Run with --verbose for detailed, round-by-round logs.")
    print("\n" + "=" * 80 + "\n🚀 Starting New Multi-Agent Simulation Run\n" + "=" * 80)

    # --- Run the Multi-Agent Simulation ---
    try:
        console.print("[bold yellow]🚀 Running Multi-Agent Simulation...[/bold yellow]")

        config = load_config(args.config)

        if args.live:
            subprocess.Popen([sys.executable, "live_server.py"])
            subprocess.Popen(
                [
                    "streamlit",
                    "run",
                    "viz/live_dashboard.py",
                    "--server.headless",
                    "true",
                ]
            )
            webbrowser.open("http://localhost:8501")

        final_state = await run_auction_episode(config, live=args.live)

        print("\n" + "-" * 35 + " ✅ Simulation Complete " + "-" * 35)
        console.print(
            Panel(
                f"[bold green]Winner:[/bold green] {final_state.winner}\n"
                f"[bold green]Final Price:[/bold green] ${final_state.final_price:,.2f}\n"
                f"[bold green]Outcome:[/bold green] {'Auction successful.' if not final_state.failure_reason else final_state.failure_reason}",
                title="[bold]Auction Results[/bold]",
                expand=False,
            )
        )

    except Exception as e:
        logger.error(f"❌ An error occurred during the main execution: {e}")
        logger.error(traceback.format_exc())
        console.print(
            f"[bold red]❌ Simulation failed. Run with --verbose for detailed logs.[/bold red]"
        )


if __name__ == "__main__":
    # Ensure the current directory is in the path to allow for module imports
    sys.path.append(os.path.dirname(__file__))

    asyncio.run(main())
