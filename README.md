# 🏠 Multi-Agent Real-Estate Auction Simulator

A sophisticated multi-agent system featuring one LLM-driven seller and five distinct buyer personas transacting in natural language within a simulated English auction environment.

## 📋 Project Overview

This project implements a phased approach to auction simulation:

- **Phase 0** (✅ Complete): Smoke test with heuristic agents
- **Phase 1** (✅ Complete): Monte Carlo analysis with comprehensive baseline analytics
- **Phase 2** (🔄 Next): RL agent training and comparison against established baseline

## 🏗️ Architecture

```
auction_simulator/
├── auction_env.py          # Gymnasium-compliant auction environment
├── llm_wrapper.py          # Async Gemini API wrapper with retry logic
├── policies/
│   ├── heuristic.py        # Rule-based policies for agent personas
│   └── rl_policy.py        # RL policy wrapper (Phase 2)
├── run.py                  # Main orchestration script with CLI
├── phase1_analytics.py     # Monte Carlo analysis and visualization
├── config.yaml            # Centralized configuration
├── PHASE1_REPORT.md        # Comprehensive Phase 1 documentation
└── requirements.txt       # Dependencies

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

### 3. Phase 1 Monte Carlo Analysis

```bash
# Run comprehensive Phase 1 analysis (automatic with 10+ episodes)
python run.py --episodes 100 --output phase1_baseline.csv

# Force Phase 1 analysis for smaller batches
python run.py --episodes 5 --phase1 --output small_analysis.csv

# Run standalone analysis on existing results
python phase1_analytics.py results.csv
```

### 4. Interactive Demo

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
  --phase1               Force Phase 1 Monte Carlo analysis
  --no-analysis          Skip automatic analysis for batch runs
```

## 📊 Implementation Status

### Phase 0 ✅ Complete
- ✅ **Core Environment Loop**: Gymnasium-compliant environment working
- ✅ **Action Parsing**: All agent actions processed correctly
- ✅ **Heuristic Policies**: 5 distinct buyer personas implemented
- ✅ **LLM Integration**: Async Gemini API wrapper with fallback
- ✅ **Flexible Architecture**: Ready for Phase 1 & 2 extensions

### Phase 1 ✅ Complete
- ✅ **Monte Carlo Analysis**: 100+ episode batch processing
- ✅ **Comprehensive Analytics**: Price, surplus, and efficiency metrics
- ✅ **Economic Welfare**: Detailed surplus distribution analysis  
- ✅ **Market Efficiency**: Welfare, allocative, and revenue efficiency
- ✅ **Visualization Dashboard**: 6-panel analytical dashboard
- ✅ **Automated Integration**: Seamless analysis with batch simulations
- ✅ **Baseline Establishment**: $13K price, $5.2K welfare, 100% efficiency

## 🔄 Next Steps

### **TODOs for You:**

1. **🔑 Set up Gemini API key** (optional, see LLM Integration section above)

2. **📊 Explore Phase 1 baseline** (ready to use):
   ```bash
   # View comprehensive Phase 1 report
   cat PHASE1_REPORT.md
   
   # Run your own Phase 1 analysis
   python run.py --episodes 100 --output your_baseline.csv
   ```

3. **🧠 Install RL dependencies** for Phase 2 development:
   ```bash
   pip install stable-baselines3 torch
   ```

### **Phase 2 Development (RL) - Ready to Start:**
- **Baseline Available**: $13K price, $5.2K welfare, 100% efficiency to beat
- **Data Pipeline**: Proven batch simulation and analysis framework
- **Architecture**: Modular design ready for RL policy integration
- **Metrics**: Comprehensive efficiency tracking for performance comparison

### **Development Priorities:**
- Complete multi-agent RL training loop
- Implement model persistence and loading
- Add RL vs heuristic performance comparisons
- Integrate RL policies with existing analytics framework

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
- **`run.py`**: Main simulation runner with Phase 1 integration
- **`phase1_analytics.py`**: Comprehensive Monte Carlo analysis
- **`PHASE1_REPORT.md`**: Detailed Phase 1 implementation report
- **`auction_env.py`**: Core environment implementation
- **`phase1_analysis_dashboard.png`**: Latest visualization dashboard

## 🎯 Technical Highlights

- **Gymnasium Compliance**: Standard RL environment interface
- **Async LLM Calls**: Non-blocking API integration with retry logic
- **Modular Design**: Easy to extend and modify
- **Comprehensive Logging**: Detailed simulation tracking
- **Error Resilience**: Graceful fallbacks and error handling

---

**🚀 Phase 1 Complete - Comprehensive Baseline Established!**

✅ **Validated Performance**: $13K price, $5.2K welfare, 100% efficiency  
📊 **Rich Analytics**: 6-panel dashboard with full economic analysis  
🎯 **Ready for Phase 2**: Solid baseline for RL comparison  

📖 **See [PHASE1_REPORT.md](PHASE1_REPORT.md) for comprehensive analysis and insights**
