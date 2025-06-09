
# **Multi-Agent Real-Estate Auction Simulator Design Docs**

This document outlines the technical specifications for a multi-agent real-estate auction simulator. The system features one LLM-driven seller and five distinct buyer personas transacting in natural language within a simulated English auction.

### Phased Rollout

The project is developed in stages to ensure robustness and modularity.

| Phase | Description | Goal |
|:--- |:--- |:--- |
| **0 — Smoke Test** | Runs a single episode with hard-coded heuristic agents. | Validate the core environment loop and action parsing. |
| **1 — Monte-Carlo**| Runs 500+ episodes to gather statistics on heuristic performance. | Generate baseline analytics for price and surplus. |
| **2 — RL Upgrade** | Swaps heuristic policies with a trained PPO agent. | Demonstrate policy modularity and evaluate RL performance against the baseline. |

---
## **System Architecture & Components**

The system is designed with a decoupled architecture where a core simulation engine is controlled and visualized via a Jupyter notebook.

```
auction_simulator/
├── auction_env.py    # Defines the simulation environment as a Gymnasium-compliant class.
├── llm_wrapper.py    # Provides an async wrapper for making robust calls to LLM APIs.
├── policies/
│   ├── heuristic.py  # Implements rule-based policies for each agent persona.
│   └── rl_policy.py    # Wraps a trained Stable Baselines 3 model for RL-based action.
├── run.py            # Main execution script with CLI for running batch simulations.
├── config.yaml       # Centralized configuration for all parameters and agent traits.
└── notebook.ipynb    # Interactive front-end for demos, analysis, and visualization.
```

---
## **1. Configuration (`config.yaml`)**

This file centralizes all tunable parameters, allowing for easy experimentation without code changes. It's loaded at the start of `run.py`.

```yaml
# Top-level auction rules
auction:
  start_price: 10000
  max_rounds: 20          # Safety limit for rounds in a single auction
  bid_limit_per_buyer: 3  # Each buyer can only place 3 raising bids

# Seller-specific parameters
seller:
  reserve_price: 9800 # Private price below which the seller will not sell

# Buyer persona definitions
buyers:
  # B1: Cautious and prefers small, safe steps. Avoids risk.
  - id: B1_CONSERVATIVE_INVESTOR
    max_wtp: 12000  # Max Willingness-To-Pay
    # High aversion (0.0-1.0 scale). Prefers smaller bids and is more likely to fold.
    risk_aversion: 0.9
    # Low probability of asking a question each round.
    ask_prob: 0.05

  # B2: Makes large, aggressive bids to intimidate others. Aims for a quick win.
  - id: B2_AGGRESSIVE_TRADER
    max_wtp: 15000
    # Low aversion, prefers larger bids (1000) and bidding frequently.
    risk_aversion: 0.1
    ask_prob: 0.0

  # B3: Prefers to gather information before acting.
  - id: B3_ANALYTICAL_BUYER
    max_wtp: 14000
    # Moderate aversion, but heuristic policy will strongly favor asking questions first.
    risk_aversion: 0.6
    # Very high probability of using one of their actions to ask a question.
    ask_prob: 0.85

  # B4: Has a very strict budget and will not exceed it under any circumstances.
  - id: B4_BUDGET_CONSCIOUS
    max_wtp: 11500
    # High aversion. The heuristic policy will force a 'fold' if price > max_wtp.
    risk_aversion: 0.8
    ask_prob: 0.0

  # B5: Prone to anxiety and bidding based on the actions of others.
  - id: B5_FOMO_BIDDER
    max_wtp: 13000
    # Low aversion. Heuristic policy will increase bid probability if many others are active.
    risk_aversion: 0.2
    ask_prob: 0.0
```

---
## **2. Environment (`auction_env.py`)**

Implements the auction dynamics within a `AuctionEnv` class that inherits from `gymnasium.Env`, making it compatible with standard RL libraries.

* **`__init__(config)`**:
    * Initializes internal state variables (e.g., `self.current_price`, `self.round_no`).
    * Defines `self.observation_space` as a `spaces.Dict` and `self.action_space` for each agent.

* **`reset()`**:
    * Resets the environment to its initial state based on the `config`.
    * Returns the initial `observation` dictionary and an empty `info` dictionary.

* **`step(actions)`**:
    * Takes a dictionary of actions from the seller and all active buyers.
    * Processes bids, determines the new high bid and leading bidder.
    * Updates the price, active agent masks, and remaining bid counts.
    * Calculates rewards for all agents.
    * Determines if the episode is `terminated` (auction ended) or `truncated` (max rounds reached).
    * Returns `observation`, `rewards`, `terminated`, `truncated`, `info`.

* **`_get_obs()`**: A helper that compiles the current internal state into the structured `observation` dictionary.

#### **Environment Spaces**

| **Observation Space (`spaces.Dict`)** | Type | Shape | Description |
|:---|:---|:---|:---|
| `price` | `Box(float)` | `(1,)` | Current highest bid price. |
| `round_no` | `Box(int)` | `(1,)` | The current round number. |
| `bids_left` | `Box(int)` | `(5,)` | Remaining bids for each of the 5 buyers. |
| `active_mask` | `Box(int)` | `(5,)` | Binary mask of which buyers are still in the auction. |
| `last_increment`| `Box(float)` | `(1,)` | The size of the last price increase (memory feature). |

| **Action Space** | Values | Description |
|:---|:---|:---|
| **Buyer Action** | `[0, 1, 2, 3]` | `0`: fold, `1`: bid 50k, `2`: bid 100k, `3`: ask question |
| **Seller Action**| `[0, 1, 2]` | `0`: announce next round, `1`: answer question, `2`: close auction |

#### **Reward Calculation**
* **Winning Buyer `i`**: $R_i = \text{max\_wtp}_i - \text{final\_price}$
* **Losing/Folded Buyers**: $R_j = 0$
* **Seller**: $R_s = \text{final\_price} - \text{reserve\_price}$

---
## **3. LLM Wrapper (`llm_wrapper.py`)**

Handles all communication with external LLM APIs.

* **Implementation**: Uses `asyncio` and `httpx.AsyncClient` for efficient, non-blocking I/O.
* **`async call(prompt)`**:
    * Sends the prompt to the configured LLM API (e.g., Gemini).
    * Implements an exponential backoff retry mechanism for `5xx` server errors.
    * Handles API-specific exceptions.
* **`parse_response(response_text)`**:
    * A helper function that parses the raw LLM output. It splits the response string, returning the first word as the `action` and the remainder as `commentary`. This provides a robust method for structured action extraction.

---
## **4. Policies (`policies/`)**

Contains the logic that maps an observation to an action for an agent.

#### `heuristic.py`
* **`HeuristicPolicy` class**: Implements deterministic, rule-based logic.
* **`get_action(state, persona)` method**:
    * Checks if `state['price']` is approaching `persona['max_wtp']`.
    * Uses `persona['risk_aversion']` in a simple probabilistic check to decide whether to bid aggressively or conservatively.
    * Returns a valid action integer based on these rules.

#### `rl_policy.py`
* **`RLPolicy` class**: A wrapper for a trained RL model.
* **`__init__(model_path)`**: Loads a pre-trained model file (e.g., a `.zip` from Stable Baselines 3).
* **`get_action(state)` method**:
    * Directly calls `self.model.predict(state, deterministic=True)`.
    * Returns the action chosen by the neural network policy.

---
## **5. Orchestration (`run.py`)**

The main script that drives the simulation.

* **Implementation**: Uses Python's `argparse` to accept command-line arguments like `--policy [heuristic|rl]` and `--episodes N`.
* **`async def run_episode()`**:
    * Initializes the `AuctionEnv` and the chosen policies.
    * Maintains a `history` object (`collections.deque(maxlen=10)`) to store a log of recent events for rich prompt context.
    * **Main Loop**:
        1.  Get state from `env`.
        2.  Build seller prompt using state and `history`; get seller action via `LLM` or policy.
        3.  Concurrently build prompts for all active buyers; get buyer actions in parallel using `asyncio.gather()`.
        4.  Pass all actions to `env.step()`.
        5.  Log results and update `history`.
        6.  Loop until the episode is `terminated`.

---
## **6. Notebook (`notebook.ipynb`)**

The user-facing component for interactive control and analysis.

* **Implementation**: A standard Jupyter notebook using `ipywidgets` for simple user interaction (e.g., buttons to run different phases).
* **Data Generation**: For batch runs (Phase 1 & 2), the notebook invokes `run.py` using the `subprocess` module to execute the simulation as a separate process and saves results to a `.csv` file.
* **Visualization**: Uses `matplotlib` and `seaborn` to load the `.csv` results and generate key plots:
    * Histogram of final sale prices.
    * Bar chart of average surplus per buyer.
    * Comparison plots between heuristic and RL performance.