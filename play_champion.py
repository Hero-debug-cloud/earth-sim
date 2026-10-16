import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from env import Environment
from agent import PolicyNetwork

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def clear_screen():
    print("\033[H\033[J", end="")

def run_champion_demo(target_goals: int = 1, max_steps: int = 100, delay: float = 0.20):
    ckpt_path = Path("checkpoints/champion.pt")
    if not ckpt_path.exists():
        print("Error: checkpoints/champion.pt not found. Run `uv run python train.py` first.")
        return

    # Load champion neural weights
    model = PolicyNetwork(in_features=5, hidden_size=16, out_actions=4)
    weights = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    # Read generation and lineage dynamically from history.json
    history_path = Path("checkpoints/history.json")
    gen_num = 1
    lineage_str = "Champion"
    if history_path.exists():
        import json
        with open(history_path, "r") as f:
            h = json.load(f)
            gen_num = h.get("total_generations", 1)
            lineage_str = h.get("dominant_lineage", "Champion")

    grid_size = 12
    num_food = 8
    num_hazards = 3
    env = Environment(size=grid_size, num_food=num_food, num_hazards=num_hazards)

    # Spawn champion in center
    pos = (grid_size // 2, grid_size // 2)
    food_eaten = 0
    hazards_hit = 0
    action_names = ["Move North (Up)", "Move South (Down)", "Move West (Left)", "Move East (Right)"]
    action_symbols = ["⬆️", "⬇️", "⬅️", "➡️"]

    step = 0
    start_time = time.perf_counter()

    prev_pos = None

    while step < max_steps:
        step += 1
        obs = env.get_observation(pos, agent_energy=1.0)
        state_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            h = torch.tanh(model.fc1(state_t))
            logits = model.fc2(h)

            # ACTION MASKING: Do not allow walking into border walls
            ar, ac = pos
            if ar == 0:             # At top border, forbid North
                logits[0] = -1e9
            if ar == grid_size - 1: # At bottom border, forbid South
                logits[1] = -1e9
            if ac == 0:             # At left border, forbid West
                logits[2] = -1e9
            if ac == grid_size - 1: # At right border, forbid East
                logits[3] = -1e9

            # Reflex Guard (Never voluntarily step on a hazard)
            moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            for act_idx, (dr, dc) in moves.items():
                if (ar + dr, ac + dc) in env.hazards:
                    logits[act_idx] -= 30.0

            # Momentum / Anti-Ping-Pong Suppressor
            if prev_pos is not None:
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) == prev_pos:
                        logits[act_idx] -= 0.5

            # Softmax with temperature tau=0.4 (sharp focus + breaks ping-pong loops)
            probs = F.softmax(logits, dim=-1)
            scaled_probs = F.softmax(logits / 0.4, dim=-1)
            dist = torch.distributions.Categorical(scaled_probs)
            action = int(dist.sample().item())

        prev_pos = pos
        new_pos, reward, ate_food, hit_hazard = env.step_agent(pos, action)

        if ate_food:
            food_eaten += 1
            event_msg = f"{GREEN}{BOLD}🍏 ATE FOOD! (+10 Reward){RESET}"
        elif hit_hazard:
            hazards_hit += 1
            event_msg = f"{RED}{BOLD}⚠️ HIT HAZARD! (-25 Penalty){RESET}"
        else:
            event_msg = f"{DIM}Navigating toward food...{RESET}"

        pos = new_pos

        # --- DRAW TERMINAL ARENA ---
        clear_screen()
        title_text = f"🏆 GENERATION {gen_num} CHAMPION ({lineage_str})"
        print(f"{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║ {title_text:^73} ║{RESET}")
        print(f"{CYAN}{BOLD}╠═══════════════════════════════════════════════════════════════════════════╣{RESET}")
        print(f"║ {BOLD}Step:{RESET} {CYAN}{step:2d}/{max_steps}{RESET} │ {BOLD}Goal:{RESET} {food_eaten}/{target_goals} Foods │ {BOLD}Hazards Hit:{RESET} {RED}{hazards_hit:2d}{RESET} │ {event_msg:<32} ║")
        print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

        # Render Grid
        grid_rows = []
        for r in range(grid_size):
            row_str = "   "
            for c in range(grid_size):
                if (r, c) == pos:
                    row_str += f"{YELLOW}{BOLD}👑{RESET} "
                elif (r, c) in env.foods:
                    row_str += f"{GREEN}🍏{RESET} "
                elif (r, c) in env.hazards:
                    row_str += f"{RED}⚠️ {RESET}"
                else:
                    row_str += f"{DIM}· {RESET} "
            grid_rows.append(row_str)

        print("\n".join(grid_rows))

        # Render Thought Telemetry
        print(f"\n{BOLD}[ BRAIN THOUGHT PROCESS ]{RESET}")
        print(f"  • Nearest Food:   dx={obs[0]:+.2f}, dy={obs[1]:+.2f}")
        print(f"  • Nearest Hazard: dx={obs[2]:+.2f}, dy={obs[3]:+.2f}")
        print(f"  • Action Probabilities:")
        for ai, name in enumerate(action_names):
            p = float(probs[ai].item())
            bar = "█" * int(p * 15)
            marker = f"{YELLOW}{BOLD}👉{RESET}" if ai == action else "  "
            print(f"    {marker} {action_symbols[ai]} {name:<22} [{bar:<15}] {p*100:5.1f}%")

        time.sleep(delay)

        # GOAL CHECK: If goal is achieved, end it simply!
        if food_eaten >= target_goals:
            elapsed = time.perf_counter() - start_time
            print(f"\n{GREEN}{BOLD}═════════════════════════════════════════════════════════════════════════{RESET}")
            print(f"{GREEN}{BOLD}🎉 GOAL ACHIEVED! Champion ate {food_eaten} food in {step} steps ({elapsed:.2f}s).{RESET}")
            print(f"{GREEN}{BOLD}═════════════════════════════════════════════════════════════════════════{RESET}\n")
            return

    print(f"\n{YELLOW}Reached step limit ({max_steps} steps). Food collected: {food_eaten}/{target_goals}.{RESET}\n")

if __name__ == "__main__":
    try:
        # Default goal: eat 1 food and finish immediately!
        run_champion_demo(target_goals=1, max_steps=60, delay=0.18)
    except KeyboardInterrupt:
        print("\nPlayback stopped.")
        sys.exit(0)
