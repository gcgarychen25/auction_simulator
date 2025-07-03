# 🏠 Hybrid LangGraph ✕ MCP Auction Simulator

A next-generation auction simulator built with **LangGraph** and **Model Context Protocol (MCP)** for future-proof tool integration.

## 🎯 Features

- **Minimal Two-Node Architecture**: Clean agent-tool loop using LangGraph
- **Hybrid Tool System**: Local tools for performance + MCP for extensibility  
- **Real-time Event System**: Built-in event bus for logging and visualization
- **Configurable Agents**: YAML-based configuration for easy customization
- **Future-Ready**: MCP integration allows easy addition of new tools and APIs

## 🏗️ Architecture

```
┌───────────────────────────────┐
│        LangGraph Runtime      │
│                               │
│   ┌────────┐  Action  ┌──────┐│
│   │ Agent  │──────────│ Tool ││
│   │  LLM   │◄─────────│Node  ││
│   └────────┘Observation└──────┘│
└────────────────┬───────────────┘
                 ▼
          MCP Servers + Local Tools
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
# Clone and enter directory
git clone <repository-url>
cd auction_simulator

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY="your-api-key-here"
```

### Running a Simulation

```bash
# Run with default configuration
python run.py

# The simulator will:
# 1. Load config.yaml
# 2. Initialize auction with property details
# 3. Run LangGraph agent-tool loop
# 4. Save results to results/ directory
```

## 📁 Project Structure

```
auction_simulator/
├── agents.py              # LLM agents (buyer logic)
├── loop_graph.py          # Two-node LangGraph implementation  
├── schemas.py             # Pydantic models (Action, LoopState, etc.)
├── config.yaml            # Configuration file
├── run.py                 # Main entry point
├── tools/                 # Tool implementations
│   ├── __init__.py        # Tool registry
│   ├── builtin/           # Local Python tools
│   │   ├── bid.py         # BID tool
│   │   ├── call.py        # CALL tool  
│   │   ├── fold.py        # FOLD tool
│   │   ├── status.py      # STATUS tool
│   │   └── check_termination.py
│   └── mcp_client.py      # MCP integration
├── utils/
│   └── event_bus.py       # Event system
├── tests/                 # Unit tests
└── design_docs.md         # Architecture documentation
```

## 🔧 Available Tools

### Local Tools (Python)
- **BID** - Place a bid with specified amount
- **CALL** - Stay in auction at current price  
- **FOLD** - Withdraw from auction
- **STATUS** - Get current auction status
- **CHECK_TERMINATION** - Check if auction should end

### MCP Tools (Remote)
- **ASK_SELLER** - Ask the seller questions about the property

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# LLM Configuration
llm:
  model_name: "gpt-3.5-turbo"
  temperature: 0.7

# Tool Backend Selection
tools:
  BID: local          # Fast local implementation
  ASK_SELLER: mcp     # Remote MCP server

# Buyer Profile
buyer_profile:
  budget: 500000
  strategy: "balanced"
  preferences:
    bedrooms: 3
    location: "suburban"

# Property Details
property:
  property_id: "PROP_001"
  starting_price: 250000
  description: "Beautiful 3-bedroom house"
  # ... more property details
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bid_tool.py

# Run with coverage
pytest --cov=.
```

## 📊 Sample Output

```
🏠 Hybrid LangGraph + MCP Auction Simulator
==================================================
✅ Configuration loaded from config.yaml
🚀 Starting auction for property PROP_001
💰 Starting price: $250,000.00
👤 Buyer budget: $500,000.00

📊 AUCTION RESULTS
==============================
✅ Auction completed successfully
🏆 Winner: buyer
💵 Final price: $275,000.00
🔄 Total rounds: 3

🔧 Tool usage:
  STATUS: 2 times
  BID: 1 times
  CHECK_TERMINATION: 3 times

📈 Events: 4 total
💾 Results saved to results/auction_results_20241201_143022.json
```

## 🔮 Future Extensions

The MCP architecture enables easy addition of:

- **External APIs**: Market data, tax records, neighborhood info
- **Advanced Tools**: Property valuation, mortgage calculators  
- **Multi-Agent**: Seller agents, competing buyers
- **ML Integration**: PPO training, strategy optimization

## 🏛️ Design Philosophy

1. **Simplicity**: Two-node loop vs complex state machines
2. **Performance**: Local tools for latency-critical operations
3. **Extensibility**: MCP for future tool integration
4. **Observability**: Rich event system for debugging and analysis

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ using LangGraph and MCP** 