# 🏠 Multi-Agent Real-Estate Auction Simulator

A sophisticated multi-agent auction system with 5 buyer personas, LLM-driven seller, and reinforcement learning capabilities.

## 🚀 Quick Start

### Install dependencies in your conda env
```bash
pip install -r requirements.txt
```

### Optional: LLM Integration
```bash
# Get API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your_api_key_here"
export GEMINI_MODEL="gemini-2.0-flash-lite"
```

## 📊 Three-Phase Execution

### Phase 0: Foundation Test
```bash
python run.py --phase 0
```
Single episode smoke test to validate core functionality.

### Phase 1: Monte Carlo Baseline  (1-2 min)
```bash
python run.py --phase 1 --episodes 10000 --output phase1_results.csv
```
Statistical baseline with comprehensive analytics. Generates `reports/phase1_analysis.md`.

### Phase 2: Reinforcement Learning (1-2 min)
```bash
python run.py --phase 2 --episodes 200 --training-steps 1000 --output phase2_results.csv
```
PPO training and RL vs heuristic comparison. Generates `reports/phase2_analysis.md`.

### Phase 3: Smarter LLM Integration with property info and buyer requirement
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

The project is organized into a modular, phase-driven structure:

```
auction_simulator/
├── run.py                       # Main CLI to orchestrate all simulation phases
├── config.yaml                  # Defines environment, buyers, seller, and persona requirements
├── README.md                    # You are here!
│
├── multi_agent_orchestrator.py  # CORE: Phase 3 LLM-based multi-agent simulation
│
├── simulator.py                 # LEGACY: Core engine for Phase 1 (Monte Carlo) & 2 (RL)
├── auction_env.py               # LEGACY: Gymnasium environment for RL training
├── policies/                    # LEGACY: Heuristic and RL agent policies
│
├── phase1_analytics.py          # Analysis scripts for Monte Carlo results
├── phase2_analytics.py          # Analysis scripts for RL agent performance
├── phase1_report.md     
├── phase2_report.md              
│
└── notebook.ipynb               # Interactive notebook for experimentation and visualization
```


---

**🎯 Complete multi-agent RL auction system with comprehensive analytics and interactive notebook interface, evolving from phase 0 smoke test to phase 1 heuristics based agents to phase 2 RL trained agents to phase 3 LLM agents.**
