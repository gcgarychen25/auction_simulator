# 🏠 Multi-Agent Real-Estate Auction Simulator

A sophisticated multi-agent auction system with 5 buyer personas, LLM-driven seller, and reinforcement learning capabilities.

## 🚀 Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Optional: LLM Integration
```bash
# Get API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your_api_key_here"
```

## 📊 Three-Phase Execution

### Phase 0: Foundation Test
```bash
python run.py --phase 0
```
Single episode smoke test to validate core functionality.

### Phase 1: Monte Carlo Baseline  
```bash
python run.py --phase 1 --episodes 100 --output phase1_results.csv
```
Statistical baseline with comprehensive analytics. Generates `reports/phase1_analysis.md`.

### Phase 2: Reinforcement Learning
```bash
python run.py --phase 2 --episodes 100 --training-steps 1000 --output phase2_results.csv
```
PPO training and RL vs heuristic comparison. Generates `reports/phase2_analysis.md`.

### Phase 3: LLM Integration
```bash
python run.py --phase 3
```

## 📱 Interactive Notebook

```bash
# Open notebook.ipynb in Cursor/Jupyter
jupyter notebook notebook.ipynb
```
Phase-focused notebook with executable cells and configuration guides.

## 👥 Buyer Personas

| Persona | Max WTP | Risk | Behavior |
|---------|---------|------|----------|
| Conservative | $12K | 0.9 | Safe bids |
| Aggressive | $15K | 0.1 | Large bids |
| Analytical | $14K | 0.6 | Ask questions |
| Budget | $11.5K | 0.8 | Strict limits |
| FOMO | $13K | 0.2 | Follow others |

## 🏗️ Architecture

```
auction_simulator/
├── run.py                  # CLI orchestration
├── notebook.ipynb          # Interactive interface
├── auction_env.py          # Gymnasium environment
├── llm_wrapper.py          # Gemini API integration
├── policies/               # Heuristic & RL policies
├── phase1_analytics.py     # Monte Carlo analysis
├── phase2_analytics.py     # RL analysis
└── reports/               # Generated analysis reports
```


---

**🎯 Complete multi-agent RL auction system with comprehensive analytics and interactive notebook interface.**
