import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from agent import PolicyNetwork

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def color_weight(w: float, width: int = 7) -> str:
    """Format weight with red/blue/green heat coloring."""
    val_str = f"{w:+6.2f}"
    if w > 1.0:
        return f"{GREEN}{BOLD}{val_str}{RESET}"
    elif w > 0.1:
        return f"{GREEN}{val_str}{RESET}"
    elif w < -1.0:
        return f"{RED}{BOLD}{val_str}{RESET}"
    elif w < -0.1:
        return f"{RED}{val_str}{RESET}"
    else:
        return f"{DIM}{val_str}{RESET}"

def render_bar(val: float, max_val: float = 1.0, width: int = 12) -> str:
    normalized = min(1.0, max(0.0, val / max_val))
    filled = int(normalized * width)
    return f"[{'█' * filled}{' ' * (width - filled)}]"

def inspect_champion_brain():
    ckpt_path = Path("checkpoints/champion.pt")
    history_path = Path("checkpoints/history.json")

    if not ckpt_path.exists():
        print(f"{RED}Error: checkpoints/champion.pt not found. Run training first.{RESET}")
        return

    # Load neural model
    model = PolicyNetwork(in_features=5, hidden_size=16, out_actions=4)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    gen_num = 500
    lineage = "Champion"
    if history_path.exists():
        with open(history_path, "r") as f:
            h = json.load(f)
            gen_num = h.get("total_generations", 500)
            lineage = h.get("dominant_lineage", "Champion")

    w1 = model.fc1.weight.data # [16, 5]
    b1 = model.fc1.bias.data   # [16]
    w2 = model.fc2.weight.data # [4, 16]
    b2 = model.fc2.bias.data   # [4]

    sensor_names = ["Food dx", "Food dy", "Hazard dx", "Hazard dy", "Energy"]
    action_names = ["⬆️ UP", "⬇️ DOWN", "⬅️ LEFT", "➡️ RIGHT"]

    # 1. HEADER
    print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║                     🧠 CHAMPION NEURAL BRAIN ARCHITECTURE                             ║{RESET}")
    print(f"{CYAN}{BOLD}╠═══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"║ {BOLD}Generation:{RESET} {CYAN}{gen_num:<4}{RESET}  │ {BOLD}Lineage:{RESET} {MAGENTA}{lineage[:40]:<40}{RESET}       ║")
    print(f"║ {BOLD}Architecture:{RESET} 5 Sensory Inputs ──► 16 Hidden Neurons (tanh) ──► 4 Motor Actions (Softmax) ║")
    print(f"║ {BOLD}Parameters:{RESET}   164 Synaptic Weights & Biases (Optimized via Evolutionary Policy Gradient)  ║")
    print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════════════════╝{RESET}\n")

    # 2. SENSOR IMPORTANCE / WEIGHT NORMS
    print(f"{BOLD}📊 [1] SENSORY INPUT IMPORTANCE (L2 Weight Norm across 16 Hidden Neurons):{RESET}")
    norms = torch.norm(w1, dim=0)
    max_norm = float(torch.max(norms).item())
    for i, name in enumerate(sensor_names):
        n_val = float(norms[i].item())
        bar = render_bar(n_val, max_norm, 18)
        pct = (n_val / sum(norms.tolist())) * 100.0
        role = "Attractive Food Pull" if i < 2 else ("Repulsive Danger Push" if i < 4 else "Pacing Bias")
        print(f"  • {name:<11} : {bar} {n_val:5.2f} ({pct:4.1f}%) │ {DIM}{role}{RESET}")

    # 3. EFFECTIVE LINEAR SENSITIVITY (JACOBIAN)
    print(f"\n{BOLD}⚡ [2] EFFECTIVE SENSITIVITY MATRIX (How inputs directly push actions):{RESET}")
    x0 = torch.zeros(5)
    x0[4] = 1.0 # Energy=1.0
    h0 = torch.tanh(model.fc1(x0))
    dh = 1.0 - h0**2
    W_eff = w2 @ torch.diag(dh) @ w1 # [4, 5]

    print(f"  {'Action':<10} │ {'Food dx':>8} │ {'Food dy':>8} │ {'Hazard dx':>9} │ {'Hazard dy':>9} │ {'Action Bias':>11}")
    print("  " + "─" * 68)
    for a_idx, act in enumerate(action_names):
        w_row = [color_weight(float(W_eff[a_idx, s].item())) for s in range(4)]
        bias_str = color_weight(float(b2[a_idx].item()))
        print(f"  {act:<10} │ {w_row[0]:>17} │ {w_row[1]:>17} │ {w_row[2]:>18} │ {w_row[3]:>18} │ {bias_str:>20}")

    # 4. NEURAL SIMULATION TEST BENCH
    print(f"\n{BOLD}🔬 [3] LIVE BRAIN THOUGHT SIMULATION ON 3 SCENARIOS:{RESET}")

    scenarios = [
        {
            "name": "Scenario A: Direct Food Sprint (Food is North, Path is 100% Clear)",
            "pos": (5, 5),
            "food": (0, 5),
            "hazards": [(2, 2)],
            "obs": [-0.42, 0.0, 0.0, 0.0, 1.0],
            "desc": "Food is 5 tiles North. No hazards in direct trajectory."
        },
        {
            "name": "Scenario B: Hazard Directly Ahead (Food is North, Hazard is 1 Tile North!)",
            "pos": (5, 5),
            "food": (0, 5),
            "hazards": [(4, 5)], # North is hazard!
            "obs": [-0.42, 0.0, -1.0, 0.0, 1.0],
            "desc": "Food is North, but Hazard is 1 tile North! Reflex guard prevents fatal collision."
        },
        {
            "name": "Scenario C: Diagonal Hazard Detour (Hazard is East, Food is South-East)",
            "pos": (5, 5),
            "food": (8, 8),
            "hazards": [(5, 6)], # East is hazard!
            "obs": [0.25, 0.25, 0.0, 1.0, 1.0],
            "desc": "Hazard blocks the East side directly, food is diagonal South-East."
        }
    ]

    for sc in scenarios:
        obs_t = torch.tensor(sc["obs"], dtype=torch.float32)
        with torch.no_grad():
            h = torch.tanh(model.fc1(obs_t))
            raw_logits = model.fc2(h).clone()

            # Apply Wall Masking & Reflex Guard
            guard_logits = raw_logits.clone()
            ar, ac = sc["pos"]
            moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            reflex_triggered = []
            for act_idx, (dr, dc) in moves.items():
                if (ar + dr, ac + dc) in sc["hazards"]:
                    guard_logits[act_idx] -= 30.0
                    reflex_triggered.append(action_names[act_idx])

            probs = F.softmax(guard_logits / 0.4, dim=-1)

        print(f"\n  {YELLOW}{BOLD}▶ {sc['name']}{RESET}")
        print(f"    {DIM}{sc['desc']}{RESET}")
        print(f"    Sensory Input : Food (dx={sc['obs'][0]:+.2f}, dy={sc['obs'][1]:+.2f}) │ Hazard (dx={sc['obs'][2]:+.2f}, dy={sc['obs'][3]:+.2f})")

        # Hidden layer activations visualization (16 neurons)
        h_vals = h.tolist()
        h_str = " ".join([f"{color_weight(val, 4)}" for val in h_vals])
        print(f"    16 Hidden Neurons : [{h_str}]")

        if reflex_triggered:
            print(f"    {RED}{BOLD}🚨 Pain Reflex Fired:{RESET} Blocked {', '.join(reflex_triggered)} (-30.0 logit penalty to prevent crash)")

        print(f"    Motor Decision Probabilities:")
        top_idx = int(torch.argmax(probs).item())
        for a_idx, act in enumerate(action_names):
            p = float(probs[a_idx].item())
            bar = "█" * int(p * 15)
            marker = f"{GREEN}{BOLD}👉 CHOSEN{RESET}" if a_idx == top_idx else "         "
            raw_val = float(raw_logits[a_idx].item())
            print(f"      {marker} {act:<10} [{bar:<15}] {p*100:5.1f}% │ Raw Logit: {raw_val:+.2f}")

    print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
    print(f"{BOLD}💡 How to observe this live during gameplay:{RESET}")
    print(f"   1. Brain Inspector CLI : {CYAN}uv run python inspect_brain.py{RESET}")
    print(f"   2. Step-by-Step Play   : {CYAN}uv run python play_champion_v2.py{RESET} (Press [Enter] per step)")
    print(f"   3. Auto Play Arena     : {CYAN}uv run python play_champion_v2.py --auto{RESET}")
    print(f"   4. 20-Run Benchmark    : {CYAN}uv run python play_champion_v2.py --runs 20{RESET}")
    print(f"   5. Live ASCII Monitor  : {CYAN}uv run python monitor.py{RESET} (In a second terminal)")
    print(f"   6. Browser Dashboard   : {CYAN}uv run uvicorn server:app --reload --port 8000{RESET}")
    print(f"{CYAN}{BOLD}========================================================================{RESET}\n")

if __name__ == "__main__":
    inspect_champion_brain()
