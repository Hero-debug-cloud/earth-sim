import os
import sys
import time
import json
import random
import argparse
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

def clear_screen():
    print("\033[H\033[J", end="")

class ArenaV2:
    """Non-respawning Pac-Man style board: champion must clear ALL food."""
    def __init__(self, size: int = 12, num_food: int = 8, num_hazards: int = 3):
        self.size = size
        self.num_food = num_food
        self.num_hazards = num_hazards
        self.reset()

    def reset(self):
        all_positions = [(r, c) for r in range(self.size) for c in range(self.size)]
        center = (self.size // 2, self.size // 2)
        if center in all_positions:
            all_positions.remove(center)

        random.shuffle(all_positions)
        self.hazards = all_positions[:self.num_hazards]
        self.foods = all_positions[self.num_hazards:self.num_hazards + self.num_food]
        self.total_food_target = len(self.foods)

    def get_observation(self, pos: tuple[int, int]) -> np.ndarray:
        ar, ac = pos

        # Target nearest remaining food
        if self.foods:
            food_dists = [abs(fr - ar) + abs(fc - ac) for fr, fc in self.foods]
            min_idx = int(np.argmin(food_dists))
            best_f = self.foods[min_idx]
            food_dx = np.clip((best_f[0] - ar) / float(self.size), -1.0, 1.0)
            food_dy = np.clip((best_f[1] - ac) / float(self.size), -1.0, 1.0)
        else:
            best_f = None
            food_dx, food_dy = 0.0, 0.0

        # Target nearest hazard with Inverse Proximity & Amplified Gain (Option A)
        # At distance 1: repulsion intensity = 3.5 (overpowers food attraction gradient)
        # Decays to 0.0 at distance >= 4 (Safe zone)
        if self.hazards:
            hazard_dists = [abs(hr - ar) + abs(hc - ac) for hr, hc in self.hazards]
            min_h = int(np.argmin(hazard_dists))
            best_h = self.hazards[min_h]
            dr = best_h[0] - ar
            dc = best_h[1] - ac
            dist = abs(dr) + abs(dc)
            if 0 < dist <= 3:
                intensity = (4.0 - dist) / 3.0  # dist 1 -> 1.0, dist 2 -> 0.67, dist 3 -> 0.33
                hazard_dx = np.clip((dr / float(dist)) * intensity, -1.0, 1.0)
                hazard_dy = np.clip((dc / float(dist)) * intensity, -1.0, 1.0)
            else:
                hazard_dx, hazard_dy = 0.0, 0.0
        else:
            best_h = None
            hazard_dx, hazard_dy = 0.0, 0.0

        return np.array([food_dx, food_dy, hazard_dx, hazard_dy, 1.0], dtype=np.float32), best_f, best_h

    def step(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool, bool]:
        ar, ac = pos
        moves = {
            0: (-1, 0), # Up
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (0, 1)   # Right
        }
        dr, dc = moves.get(action, (0, 0))
        nr = max(0, min(self.size - 1, ar + dr))
        nc = max(0, min(self.size - 1, ac + dc))
        new_pos = (nr, nc)

        ate_food = False
        hit_hazard = False

        if new_pos in self.foods:
            self.foods.remove(new_pos) # Permanent clearance!
            ate_food = True

        if new_pos in self.hazards:
            hit_hazard = True

        return new_pos, ate_food, hit_hazard

def play_v2(interactive: bool = True, max_steps: int = 150):
    ckpt_path = Path("checkpoints/champion.pt")
    history_path = Path("checkpoints/history.json")

    if not ckpt_path.exists():
        print(f"{RED}Error: checkpoints/champion.pt not found. Run training first.{RESET}")
        return

    # Load neural model
    model = PolicyNetwork(in_features=5, hidden_size=16, out_actions=4)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    # Load metadata
    gen_num = 562
    lineage = "Champion"
    if history_path.exists():
        with open(history_path, "r") as f:
            h = json.load(f)
            gen_num = h.get("total_generations", 562)
            lineage = h.get("dominant_lineage", "Champion")

    arena = ArenaV2(size=12, num_food=8, num_hazards=3)
    pos = (arena.size // 2, arena.size // 2)

    food_collected = 0
    hazards_hit = 0
    step = 0
    start_time = time.perf_counter()

    action_names = ["Move North (Up)", "Move South (Down)", "Move West (Left)", "Move East (Right)"]
    action_symbols = ["⬆️", "⬇️", "⬅️", "➡️"]

    prev_pos = None
    auto_mode = not interactive

    while step < max_steps:
        step += 1
        obs, target_food, nearest_hazard = arena.get_observation(pos)
        state_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            h = torch.tanh(model.fc1(state_t))
            logits = model.fc2(h)

            # Border wall masking
            ar, ac = pos
            if ar == 0: logits[0] = -1e9
            if ar == arena.size - 1: logits[1] = -1e9
            if ac == 0: logits[2] = -1e9
            if ac == arena.size - 1: logits[3] = -1e9

            # Option B: Action Reflex Guard (Pain Receptor - Never voluntarily step on a hazard)
            moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            for act_idx, (dr, dc) in moves.items():
                if (ar + dr, ac + dc) in arena.hazards:
                    logits[act_idx] -= 30.0

            # Momentum / Anti-Ping-Pong Suppressor (mild bias to break 2-tile loops)
            if prev_pos is not None:
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) == prev_pos:
                        logits[act_idx] -= 0.5

            probs = F.softmax(logits, dim=-1)
            scaled_probs = F.softmax(logits / 0.4, dim=-1)
            dist = torch.distributions.Categorical(scaled_probs)
            action = int(dist.sample().item())

        prev_pos = pos
        new_pos, ate_food, hit_hazard = arena.step(pos, action)
        if hit_hazard:
            hazards_hit += 1
            pos = new_pos
            clear_screen()
            title = f"🏆 CHAMPION V2: ALL-FOOD HUNT (Gen {gen_num})"
            print(f"{RED}{BOLD}╔═══════════════════════════════════════════════════════════════════════════╗{RESET}")
            print(f"{RED}{BOLD}║ {title:^73} ║{RESET}")
            print(f"{RED}{BOLD}╠═══════════════════════════════════════════════════════════════════════════╣{RESET}")
            print(f"║ {BOLD}Step:{RESET} {CYAN}{step:3d}/{max_steps}{RESET} │ {BOLD}Cleared:{RESET} {YELLOW}{food_collected}/{arena.total_food_target}{RESET} │ {BOLD}Hazards Hit:{RESET} {RED}{hazards_hit:2d}{RESET} │ {RED}{BOLD}💥 FATAL HAZARD COLLISION!{RESET:<20} ║")
            print(f"{RED}{BOLD}╚═══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

            # Draw Grid with death marker 💥
            for r in range(arena.size):
                row_str = "   "
                for c in range(arena.size):
                    if (r, c) == pos:
                        row_str += f"{RED}{BOLD}💥{RESET} "
                    elif (r, c) in arena.foods:
                        row_str += f"{GREEN}🍏{RESET} "
                    elif (r, c) in arena.hazards:
                        row_str += f"{RED}⚠️ {RESET}"
                    else:
                        row_str += f"{DIM}· {RESET} "
                print(row_str)

            print(f"\n{RED}{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}")
            print(f"{RED}{BOLD}💀 GAME OVER: Champion touched a hazard ⚠️ at step {step}! (PERMADEATH){RESET}")
            print(f"{RED}{BOLD}• Foods Cleared: {food_collected}/{arena.total_food_target}  |  Final Rating: 💀 Perished{RESET}")
            print(f"{RED}{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}\n")
            return

        if ate_food:
            food_collected += 1
            event_msg = f"{GREEN}{BOLD}🍏 FOOD CLEARED! ({food_collected}/{arena.total_food_target}){RESET}"
        else:
            event_msg = f"{DIM}Pursuing food target...{RESET}"

        pos = new_pos

        # --- DRAW TERMINAL SCREEN ---
        clear_screen()
        remaining = len(arena.foods)
        title = f"🏆 CHAMPION V2: ALL-FOOD HUNT (Gen {gen_num})"

        print(f"{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║ {title:^73} ║{RESET}")
        print(f"{CYAN}{BOLD}╠═══════════════════════════════════════════════════════════════════════════╣{RESET}")
        print(f"║ {BOLD}Step:{RESET} {CYAN}{step:3d}/{max_steps}{RESET} │ {BOLD}Cleared:{RESET} {GREEN}{food_collected}/{arena.total_food_target}{RESET} ({remaining} left) │ {BOLD}Hazards Hit:{RESET} {RED}{hazards_hit:2d}{RESET} │ {event_msg:<30} ║")
        print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

        # Draw Grid
        grid_rows = []
        for r in range(arena.size):
            row_str = "   "
            for c in range(arena.size):
                if (r, c) == pos:
                    row_str += f"{YELLOW}{BOLD}👑{RESET} "
                elif (r, c) in arena.foods:
                    row_str += f"{GREEN}🍏{RESET} "
                elif (r, c) in arena.hazards:
                    row_str += f"{RED}⚠️ {RESET}"
                else:
                    row_str += f"{DIM}· {RESET} "
            grid_rows.append(row_str)

        print("\n".join(grid_rows))

        # Thought Telemetry
        print(f"\n{BOLD}[ BRAIN THOUGHT PROCESS ]{RESET}")
        target_info = f"at ({target_food[0]}, {target_food[1]}) | dx={obs[0]:+.2f}, dy={obs[1]:+.2f}" if target_food else "ALL CLEARED!"
        if nearest_hazard:
            h_dist = abs(nearest_hazard[0] - pos[0]) + abs(nearest_hazard[1] - pos[1])
            h_alarm = f"{RED}{BOLD}🚨 DANGER! (repulsion={abs(obs[2])+abs(obs[3]):.2f}, dist={h_dist}){RESET}" if h_dist <= 3 else f"{DIM}Safe (dist={h_dist}){RESET}"
            hazard_info = f"at ({nearest_hazard[0]}, {nearest_hazard[1]}) | {h_alarm}"
        else:
            hazard_info = "None"
        print(f"  • Target Food:    {target_info}")
        print(f"  • Nearest Hazard: {hazard_info}")
        print(f"  • Action Probabilities:")
        for ai, name in enumerate(action_names):
            p = float(probs[ai].item())
            bar = "█" * int(p * 15)
            marker = f"{YELLOW}{BOLD}👉{RESET}" if ai == action else "  "
            print(f"    {marker} {action_symbols[ai]} {name:<22} [{bar:<15}] {p*100:5.1f}%")

        # VICTORY CHECK: Did champion collect ALL the food?
        if len(arena.foods) == 0:
            elapsed = time.perf_counter() - start_time
            print(f"\n{GREEN}{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}")
            print(f"{GREEN}{BOLD}🎉 VICTORY! CHAMPION CLEARED ALL {arena.total_food_target} FOODS!{RESET}")
            print(f"{GREEN}{BOLD}• Total Steps: {step}  |  Hazards Hit: {hazards_hit}  |  Time: {elapsed:.2f}s{RESET}")
            score_rating = "S+ (Flawless)" if hazards_hit == 0 and step <= 45 else ("A (Master)" if hazards_hit == 0 else "B (Survived)")
            print(f"{GREEN}{BOLD}• Final Rating: {score_rating}{RESET}")
            print(f"{GREEN}{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}\n")
            return

        # Interactive Step Control
        if not auto_mode:
            print(f"\n{BOLD}[Controls]:{RESET} Press {CYAN}[Enter]{RESET} for next step  │  {YELLOW}[a]{RESET} for Auto-play  │  {RED}[q]{RESET} to Quit")
            user_input = input(">> ").strip().lower()
            if user_input == "q":
                print("\nExiting V2 arena.")
                return
            elif user_input == "a":
                auto_mode = True
        else:
            time.sleep(0.16)

    print(f"\n{YELLOW}Reached step limit ({max_steps} steps). Final score: {food_collected}/{arena.total_food_target} food.{RESET}\n")

def evaluate_champion(num_runs: int = 20, max_steps: int = 150):
    ckpt_path = Path("checkpoints/champion.pt")
    history_path = Path("checkpoints/history.json")

    if not ckpt_path.exists():
        print(f"{RED}Error: checkpoints/champion.pt not found. Run training first.{RESET}")
        return

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

    print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║         🔬 20-EPISODE BENCHMARK REPORT: GENERATION {gen_num:<4} CHAMPION                       ║{RESET}")
    print(f"{CYAN}{BOLD}╠═══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"║ {BOLD}Lineage:{RESET} {lineage[:50]:<50} ║")
    print(f"║ {BOLD}Protocol:{RESET} 20 unique randomized 12x12 boards (8 foods, 3 hazards, center spawn)          ║")
    print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════════════════╝{RESET}\n")

    results = []
    print(f"{BOLD}{'RUN':>4} │ {'FOOD':>7} │ {'STEPS':>6} │ {'HAZARDS':>8} │ {'EFFICIENCY':>13} │ {'GRADE':<15}{RESET}")
    print("─" * 70)

    for run_idx in range(1, num_runs + 1):
        arena = ArenaV2(size=12, num_food=8, num_hazards=3)
        pos = (arena.size // 2, arena.size // 2)
        food_collected = 0
        hazards_hit = 0
        step = 0

        prev_pos = None

        while step < max_steps:
            step += 1
            obs, _, _ = arena.get_observation(pos)
            state_t = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                h = torch.tanh(model.fc1(state_t))
                logits = model.fc2(h)

                ar, ac = pos
                if ar == 0: logits[0] = -1e9
                if ar == arena.size - 1: logits[1] = -1e9
                if ac == 0: logits[2] = -1e9
                if ac == arena.size - 1: logits[3] = -1e9

                # Option B: Action Reflex Guard (Pain Receptor - Never voluntarily step on a hazard)
                moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) in arena.hazards:
                        logits[act_idx] -= 30.0

                # Momentum / Anti-Ping-Pong Suppressor (mild bias to break 2-tile loops)
                if prev_pos is not None:
                    for act_idx, (dr, dc) in moves.items():
                        if (ar + dr, ac + dc) == prev_pos:
                            logits[act_idx] -= 0.5

                scaled_probs = F.softmax(logits / 0.4, dim=-1)
                dist = torch.distributions.Categorical(scaled_probs)
                action = int(dist.sample().item())

            prev_pos = pos
            new_pos, ate_food, hit_hazard = arena.step(pos, action)
            if hit_hazard:
                hazards_hit += 1
                pos = new_pos
                break  # 💀 PERMADEATH: Episode immediately ends!

            if ate_food:
                food_collected += 1

            pos = new_pos

            if len(arena.foods) == 0:
                break

        cleared = (len(arena.foods) == 0)
        eff = round(step / max(food_collected, 1), 1)

        if hazards_hit > 0:
            grade = "💀 Perished"
            grade_color = RED + BOLD
        elif cleared and step <= 45:
            grade = "S+ (Flawless)"
            grade_color = GREEN + BOLD
        elif cleared:
            grade = "A (Master)"
            grade_color = GREEN
        else:
            grade = "C (Timeout)"
            grade_color = YELLOW

        res = {
            "run": run_idx,
            "cleared": cleared,
            "food": food_collected,
            "total_food": arena.total_food_target,
            "steps": step,
            "hazards": hazards_hit,
            "efficiency": eff,
            "grade": grade
        }
        results.append(res)

        food_str = f"{food_collected}/{arena.total_food_target}"
        eff_str = f"{eff:4.1f} steps/f"
        print(f"#{run_idx:02d} │ {food_str:>7} │ {step:6d} │ {hazards_hit:8d} │ {eff_str:>13} │ {grade_color}{grade:<15}{RESET}")

    # Summary Statistics
    total_runs = len(results)
    survived_runs = sum(1 for r in results if r["hazards"] == 0)
    cleared_runs = sum(1 for r in results if r["cleared"])
    perished_runs = sum(1 for r in results if r["hazards"] > 0)
    total_hazards = sum(r["hazards"] for r in results)

    cleared_steps = [r["steps"] for r in results if r["cleared"]]
    avg_steps = sum(cleared_steps) / float(len(cleared_steps)) if cleared_steps else 0.0
    min_steps = min(cleared_steps) if cleared_steps else 0
    max_steps_res = max(cleared_steps) if cleared_steps else 0

    s_plus_count = sum(1 for r in results if "S+" in r["grade"])
    a_count = sum(1 for r in results if "A " in r["grade"])
    perished_count = sum(1 for r in results if "Perished" in r["grade"])
    c_count = sum(1 for r in results if "Timeout" in r["grade"])

    print("─" * 70)
    print(f"\n{BOLD}📊 AGGREGATE PERFORMANCE METRICS (N={total_runs} Episodes - Permadeath):{RESET}")
    print(f"  • {BOLD}Survival Rate (0 Hazards):{RESET} {GREEN}{survived_runs}/{total_runs} ({survived_runs/total_runs*100:.1f}%){RESET}")
    print(f"  • {BOLD}Clearance Success Rate:{RESET}    {GREEN}{cleared_runs}/{total_runs} ({cleared_runs/total_runs*100:.1f}%){RESET}")
    print(f"  • {BOLD}Permadeath Mortality:{RESET}      {RED if perished_runs > 0 else GREEN}{perished_runs}/{total_runs} ({perished_runs/total_runs*100:.1f}% perished){RESET}")
    if cleared_runs > 0:
        print(f"  • {BOLD}Average Steps to Clear:{RESET}    {avg_steps:.1f} steps (Min: {min_steps}, Max: {max_steps_res})")
        print(f"  • {BOLD}Average Efficiency:{RESET}        {avg_steps/8.0:.2f} steps per food collected")

    print(f"\n{BOLD}🏅 GRADE DISTRIBUTION:{RESET}")
    print(f"  • {GREEN}{BOLD}S+ (Flawless, ≤45 steps, Survived):{RESET} {s_plus_count:2d} ({s_plus_count/total_runs*100:5.1f}%)")
    print(f"  • {GREEN}A  (Master, >45 steps, Survived):  {RESET} {a_count:2d} ({a_count/total_runs*100:5.1f}%)")
    print(f"  • {RED}{BOLD}💀 Perished (Fatal Hazard Hit):     {RESET} {perished_count:2d} ({perished_count/total_runs*100:5.1f}%)")
    if c_count > 0:
        print(f"  • {YELLOW}C  (Timeout, step limit reached):   {RESET} {c_count:2d} ({c_count/total_runs*100:5.1f}%)")
    print()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Champion V2: All-Food Clearance Arena")
    parser.add_argument("--auto", "-a", action="store_true", help="Run in continuous auto-play mode instead of step-by-step")
    parser.add_argument("--steps", "-s", type=int, default=150, help="Max steps limit (default: 150)")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of benchmark runs (default: 1 for game, 20+ for report)")
    args = parser.parse_args()

    try:
        if args.runs > 1:
            evaluate_champion(num_runs=args.runs, max_steps=args.steps)
        else:
            play_v2(interactive=not args.auto, max_steps=args.steps)
    except KeyboardInterrupt:
        print("\nArena exited.")
        sys.exit(0)
