"""
Main entry point for the hybrid LangGraph + MCP auction simulator.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config_utils import load_config
from loop_graph import run_episode


async def main():
    """Main entry point for the auction simulator."""
    print("🏠 Hybrid LangGraph + MCP Auction Simulator")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        print(f"✅ Configuration loaded from config.yaml")
        
        # Set up results directory
        results_dir = Path(config.get("simulation", {}).get("results_dir", "results"))
        results_dir.mkdir(exist_ok=True)
        
        # Run episode
        print(f"🚀 Starting auction for property {config['property']['property_id']}")
        print(f"💰 Starting price: ${config['property']['starting_price']:,.2f}")
        print(f"👤 Buyer budget: ${config['buyer_profile']['budget']:,.2f}")
        print()
        
        results = await run_episode(config)
        
        # Print results
        print("📊 AUCTION RESULTS")
        print("=" * 30)
        
        if results["success"]:
            auction_state = results["auction_state"]
            print(f"✅ Auction completed successfully")
            print(f"🏆 Winner: {auction_state.winner or 'No winner'}")
            print(f"💵 Final price: ${auction_state.final_price or auction_state.current_price:,.2f}")
            print(f"🔄 Total rounds: {auction_state.round}")
            print(f"📝 Failure reason: {auction_state.failure_reason or 'None'}")
        else:
            print(f"❌ Auction failed: {results.get('error', 'Unknown error')}")
        
        print(f"\n🔧 Tool usage:")
        for tool, count in results["tool_usage"].items():
            print(f"  {tool}: {count} times")
        
        print(f"\n📈 Events: {len(results['events'])} total")
        
        # Save results
        if config.get("simulation", {}).get("save_results", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"auction_results_{timestamp}.json"
            
            # Convert results to JSON-serializable format
            json_results = {
                "success": results["success"],
                "auction_state": auction_state.dict() if results["success"] else None,
                "tool_usage": results["tool_usage"],
                "events": [e.dict() for e in results["events"]],
                "context": results["context"],
                "timestamp": timestamp,
                "config": config
            }
            
            if not results["success"]:
                json_results["error"] = results["error"]
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {results_file}")
        
        # Display conversation log
        print(f"\n📝 Conversation Log:")
        print("-" * 50)
        for i, message in enumerate(results["context"], 1):
            print(f"{i:2d}. {message}")
        
    except FileNotFoundError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Set up environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Set your OpenAI API key to use LLM agents")
        print()
    
    exit_code = asyncio.run(main())
    exit(exit_code) 