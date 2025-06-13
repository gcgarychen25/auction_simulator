# 🏠 Multi-Agent Real-Estate Auction Simulator

A sophisticated multi-agent auction system built with LangGraph. This simulator orchestrates a real-estate auction featuring a central auctioneer and multiple LLM-powered buyer agents, each with a unique persona.

## 🚀 Quick Start

### 1. Set Up Environment

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the root directory and add your Gemini API key:
```
GEMINI_API_KEY="your_google_api_key_here"
```
You can obtain a key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3. Run the Simulation

Execute the main script to start the auction:
```bash
python run.py
```

For more detailed, round-by-round output, use the `--verbose` flag:
```bash
python run.py --verbose
```

## 🏗️ Architecture

The project is organized into a modular, agent-focused structure, making it easy to extend and maintain.

```
auction_simulator/
├── run.py                 # Main CLI entry point to start the simulation
├── config.yaml            # Defines the auction, property, buyers, and seller details
├── graph.py               # Core orchestration logic using a LangGraph StateGraph
├── agents.py              # Creates the LLM-powered buyer and seller agent runnables
├── prompts.py             # Contains all prompt templates for the agents
├── schemas.py             # Defines all Pydantic data structures (e.g., AuctionState, Action)
├── config_utils.py        # Utility functions, like loading the configuration
├── requirements.txt       # Project dependencies
└── README.md              # You are here!
```

## 🛠️ Modifying the Workflow

### Key Scripts for Future Modifications:

*   **To change an agent's personality, reasoning, or decision-making process:**
    *   **File to Modify:** `prompts.py`
    *   **Why:** This file contains the "soul" of your agents. By changing the system messages in `create_buyer_agent_prompt()` or `create_seller_prompt()`, you can alter their behavior, risk tolerance, and how they interpret their persona. This is the most common place you will make changes.

*   **To change the auction's rules or the flow of events:**
    *   **File to Modify:** `graph.py`
    *   **Why:** This file defines the state machine of the auction.
        *   To add a new step (e.g., a "final negotiation" phase), you would create a new node function and add it to the graph.
        *   To change when the auction ends, you would modify the logic in the `should_continue()` conditional edge function.
        *   To alter what happens in the Q&A or Bidding phases, you would edit the `qa_phase_node()` or `bidding_phase_node()` functions.

*   **To change the data the agents work with:**
    *   **File to Modify:** `schemas.py`
    *   **Why:** If you want to add new information to the state (e.g., an "interest rate" variable) or give agents new abilities (e.g., a "request\_financing" action), you would first define these new data structures in the Pydantic models here. This change would then ripple to `prompts.py` (to teach the agents about the new data) and `graph.py` (to handle the new logic).

*   **To change the underlying LLM or how agents are built:**
    *   **File to Modify:** `agents.py`
    *   **Why:** This file handles the technical setup of the agents. If you wanted to switch from Gemini to another model, or change the `temperature` to make the agents more or less creative, you would make those changes to the `llm` object here.