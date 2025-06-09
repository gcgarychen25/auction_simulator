# 🏠 Multi-Agent Real-Estate Auction Simulator

A sophisticated multi-agent system featuring one LLM-driven seller and five distinct buyer personas transacting in natural language within a simulated English auction environment.

## 📋 Project Overview

This project implements a phased approach to auction simulation:

- **Phase 0** (✅ Complete): Smoke test with heuristic agents
- **Phase 1** (🔄 Next): Monte Carlo analysis with 500+ episodes  
- **Phase 2** (🚀 Future): RL agent training and comparison

## 🏗️ Architecture

```
auction_simulator/
├── auction_env.py       # Gymnasium-compliant auction environment
├── llm_wrapper.py       # Async Gemini API wrapper with retry logic
├── policies/
│   ├── heuristic.py     # Rule-based policies for agent personas
│   └── rl_policy.py     # RL policy wrapper (Phase 2)
├── run.py              # Main orchestration script with CLI
├── config.yaml         # Centralized configuration
├── notebook.ipynb      # Interactive demo and analysis
└── requirements.txt    # Dependencies

```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
cd auction_simulator

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage (No API Key Required)

```bash
# Run single episode with heuristic agents
python run.py --policy heuristic --episodes 1

# Run batch simulation
python run.py --policy heuristic --episodes 50 --output results.csv
```

### 3. Interactive Demo

```bash
# Option 1: Open in Cursor IDE (Recommended)
# Simply open notebook.ipynb in Cursor and run cells

# Option 2: Traditional Jupyter browser
jupyter notebook notebook.ipynb
```

## 🔑 LLM Integration Setup

### **TODO: Setup Your Gemini API Key**

To use LLM-powered seller decisions:

1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Configure Environment**: 
   ```bash
   # Recommended: Use .env file
   mv env_template .env
   # Then edit .env file and replace 'your_api_key_here' with your actual key
   
   # Alternative: Set environment variable directly
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. **Run with LLM**:
   ```bash
   python run.py --policy heuristic --llm-seller --episodes 1
   ```

## 👥 Buyer Personas

| Persona | Max WTP | Risk Aversion | Behavior |
|---------|---------|---------------|----------|
| **Conservative Investor** | $12,000 | 0.9 | Small, safe bids |
| **Aggressive Trader** | $15,000 | 0.1 | Large, intimidating bids |
| **Analytical Buyer** | $14,000 | 0.6 | Information gathering focused |
| **Budget Conscious** | $11,500 | 0.8 | Strict budget limits |
| **FOMO Bidder** | $13,000 | 0.2 | Influenced by others' actions |

## 🎮 Command Line Interface

```bash
python run.py [OPTIONS]

Options:
  --config PATH           Configuration file (default: config.yaml)
  --policy {heuristic,rl} Policy type to use (default: heuristic)
  --episodes INT          Number of episodes (default: 1)
  --output PATH           CSV output file (default: results.csv)
  --llm-seller           Use LLM for seller decisions
  --verbose              Verbose output (default: True)
```

## 📊 Phase 0 Success Criteria

- ✅ **Core Environment Loop**: Gymnasium-compliant environment working
- ✅ **Action Parsing**: All agent actions processed correctly
- ✅ **Heuristic Policies**: 5 distinct buyer personas implemented
- ✅ **LLM Integration**: Async Gemini API wrapper with fallback
- ✅ **Flexible Architecture**: Ready for Phase 1 & 2 extensions

## 🔄 Next Steps

### **TODOs for You:**

1. **🔑 Set up Gemini API key** (see LLM Integration section above)

2. **📊 Test batch simulation** for Phase 1 preparation:
   ```bash
   python run.py --episodes 100 --output phase1_test.csv
   ```

3. **🧠 Install RL dependencies** when ready for Phase 2:
   ```bash
   pip install stable-baselines3 torch
   ```

### **Phase 1 Development (Monte Carlo):**
- Implement analytics visualization
- Generate baseline performance statistics
- Compare persona performance metrics

### **Phase 2 Development (RL):**
- Complete multi-agent training loop
- Implement model persistence
- Add RL vs heuristic comparison tools

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **LLM Fallback**: Without API key, system uses heuristic fallback
   - This is expected behavior and allows testing without LLM

3. **Notebook Issues**: Ensure Jupyter is properly installed
   ```bash
   pip install jupyter ipywidgets
   jupyter notebook
   ```

## 📁 Key Files

- **`config.yaml`**: Auction rules and buyer personas
- **`notebook.ipynb`**: Interactive demo and validation
- **`run.py`**: Main simulation runner
- **`auction_env.py`**: Core environment implementation

## 🎯 Technical Highlights

- **Gymnasium Compliance**: Standard RL environment interface
- **Async LLM Calls**: Non-blocking API integration with retry logic
- **Modular Design**: Easy to extend and modify
- **Comprehensive Logging**: Detailed simulation tracking
- **Error Resilience**: Graceful fallbacks and error handling

---

**🚀 Phase 0 Complete - Ready for Scale!**
