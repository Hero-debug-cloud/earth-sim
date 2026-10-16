# Earth Sim: Evolutionary Reinforcement Learning & Cognitive Memory Simulation

A modular implementation of **Evolutionary Reinforcement Learning (ERL)** and **Dual-Process Cognitive Memory** where autonomous agents learn across two distinct time scales:

1. **Inner Loop (Individual Lifetime)**: Agents adapt their PyTorch neural policy networks via **Policy Gradients (REINFORCE)** and temporal backpropagation (BPTT) based on direct environmental interaction.
2. **Outer Loop (Generational Evolution)**: Populations are evaluated across fitness metrics. Selective pressure culls underperforming variants, while elites reproduce via **genetic weight mutations** to seed subsequent generations.

The system incorporates a **Dual-Process Architecture**:
- **System 1 (Sensorimotor / Recurrent Core)**: Gated Recurrent Unit (GRU) networks maintaining continuous temporal hidden states ($h_t \in \mathbb{R}^{16}$) to prevent oscillating deadlocks and navigate complex obstacle perimeters.
- **System 2 (Cognitive Memory Bank)**: Structured tracking of spatial landmarks (home base, water sources, granaries), internal vitals, inventory, and dynamic loop/frustration indicators.

---

## Architecture Overview

| Component | File | Description |
|---|---|---|
| **Phase 2 Environment** | `env_phase2.py` | 14x14 ecological world with Home Base, Water Spring, Timber, Food, and Hazards. |
| **Phase 2 Agent** | `agent_phase2.py` | Dual-process agent with backpack inventory, homeostasis vitals, and goal FSM. |
| **Phase 2 Terminal Runner** | `play_phase2.py` | Live ANSI terminal simulator demonstrating the complete expedition life cycle. |
| **Base Environment** | `env.py` | 12x12 grid simulation with food targets, hazard zones, and step dynamics. |
| **Feedforward Agent** | `agent.py` | Base PyTorch policy network with lifetime gradient descent and genetic mutation operators. |
| **Recurrent Agent** | `agent_recurrent.py` | GRU-powered policy network paired with `VillageMemory` for spatial loop breaking. |
| **Cellular World** | `world.py` | 28x28 cellular automata environment featuring water bodies, terrain types, and creature vitals. |
| **Training Pipeline** | `train.py` | Headless evolutionary trainer for feedforward populations. |
| **Recurrent Trainer** | `train_recurrent.py` | Curriculum-driven trainer exposing recurrent agents to open and trap-heavy maps. |
| **Fortress Benchmarks** | `v3_test.py` | Citadel navigation test benchmarking loop-breaking and obstacle circumnavigation. |
| **Real-time Server** | `server.py` | FastAPI + WebSocket server streaming live multi-agent states and Phase 2 telemetry. |
| **Web Dashboard** | `static/` | Frontend visualizer with views for Population World, Neural Brain, and Living Hero. |

---

## Getting Started

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Installation
Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd earth-sim
uv sync
```
*Or using standard pip:*
```bash
pip install torch fastapi uvicorn websockets numpy
```

---

## Usage

### 1. Phase 2 Living Hero Expedition (Interactive Terminal)
Run the Phase 2 live simulator with real-time vitals, backpack storage, and home deposit loops:
```bash
uv run python play_phase2.py --ticks 300 --delay 0.10
```

### 2. Hero Longevity & Survival Valuation Benchmarks
Evaluate how many days the hero survives across multiple day/night cycles, its stockpile yields, and cause of death:
```bash
# V1: Standard Valuation (Single or Multi-Hero)
uv run python evaluate_hero.py --heroes 10 --max-days 100 --visual

# V2: Ecological Scarcity & Famine Pressure (Limited Bushes, Regrowth Timers & Granary Rationing)
uv run python evaluate_hero_v2.py --heroes 10 --max-days 100 --visual --delay 0.02
```

### 3. Citadel & Fortress Navigation Benchmark
Evaluate agent loop-breaking and perimeter circumnavigation capabilities:
```bash
uv run python v3_test.py
```

### 3. Headless Evolutionary Training
Train a population through genetic selection and lifetime policy gradients:
```bash
# Phase 0: Baseline Feedforward ERL
uv run python train.py

# Phase 1: Recurrent GRU with Obstacle Curriculum
uv run python train_recurrent.py

# Phase 2: Multi-Generational Living Hero (Lifecycle, Harvesting & Homeostasis)
uv run python train_phase2.py --generations 100 --pop 8 --ticks 160
```

### 4. Real-Time Web Visualizer
Launch the interactive simulation dashboard:
```bash
uv run python main.py
```
Then open:
- **Phase 2 Living Hero View**: `http://localhost:8000/phase2`
- **Simulation Arena**: `http://localhost:8000/`
- **Neural Activations & Telemetry**: `http://localhost:8000/neural`

---

## License
MIT
