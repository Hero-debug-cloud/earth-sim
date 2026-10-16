import os
import sys
import time
import json
import argparse
from typing import Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from agent import PolicyNetwork
from agent_recurrent import RecurrentAgent, RecurrentPolicyNetwork

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

class V3FortressArena:
    """Specialized arena for V3 Test: Food in center, surrounded by hazards."""
    def __init__(self, size: int = 12, scenario: str = "sealed", spawn_pos: tuple[int, int] = (1, 1)):
        self.size = size
        self.scenario = scenario
        self.spawn_pos = spawn_pos
        self.center = (size // 2 - 1, size // 2 - 1) # (5, 5) on 12x12
        self.foods = [self.center]
        self.total_food_target = 1

        cr, cc = self.center
        if scenario == "sealed":
            # 8 hazards completely boxing in the food
            self.hazards = [
                (cr - 1, cc - 1), (cr - 1, cc), (cr - 1, cc + 1),
                (cr,     cc - 1),               (cr,     cc + 1),
                (cr + 1, cc - 1), (cr + 1, cc), (cr + 1, cc + 1),
            ]
        elif scenario == "gate":
            # North tile (cr - 1, cc) is left open as a narrow breach!
            self.hazards = [
                (cr - 1, cc - 1),               (cr - 1, cc + 1),
                (cr,     cc - 1),               (cr,     cc + 1),
                (cr + 1, cc - 1), (cr + 1, cc), (cr + 1, cc + 1),
            ]
        elif scenario == "moat":
            # Radius 2 perimeter with one narrow gate
            self.hazards = []
            for r in range(cr - 2, cr + 3):
                for c in range(cc - 2, cc + 3):
                    if (r in (cr - 2, cr + 2) or c in (cc - 2, cc + 2)) and not (r == cr - 2 and c == cc):
                        self.hazards.append((r, c))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def get_observation(self, pos: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int] | None, tuple[int, int] | None]:
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

        # Target nearest hazard with Inverse Proximity (Option A)
        if self.hazards:
            hazard_dists = [abs(hr - ar) + abs(hc - ac) for hr, hc in self.hazards]
            min_h = int(np.argmin(hazard_dists))
            best_h = self.hazards[min_h]
            dr = best_h[0] - ar
            dc = best_h[1] - ac
            dist = abs(dr) + abs(dc)
            if 0 < dist <= 3:
                intensity = (4.0 - dist) / 3.0
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
            self.foods.remove(new_pos)
            ate_food = True

        if new_pos in self.hazards:
            hit_hazard = True

        return new_pos, ate_food, hit_hazard

def run_v3_test(scenario: str = "sealed", interactive: bool = True, fps: float = 8.0, max_steps: Optional[int] = None, recurrent: bool = False, patience: int = 50):
    ckpt_path = Path("checkpoints/champion_v3_gru.pt" if recurrent else "checkpoints/champion.pt")
    history_path = Path("checkpoints/history_recurrent.json" if recurrent else "checkpoints/history.json")

    recurrent_agent = None
    feedforward_model = None

    if recurrent:
        recurrent_agent = RecurrentAgent(0, patience_ticks=patience)
        if ckpt_path.exists():
            recurrent_agent.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            recurrent_agent.model.eval()
        recurrent_agent.reset_memory()
    else:
        if not ckpt_path.exists():
            print(f"{RED}Error: checkpoints/champion.pt not found. Train champion first!{RESET}")
            return
        feedforward_model = PolicyNetwork(in_features=5, hidden_size=16, out_actions=4)
        feedforward_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        feedforward_model.eval()

    gen_num = 500
    lineage = "Champion"
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                h = json.load(f)
                gen_num = h.get("total_generations", 500)
                lineage = h.get("dominant_lineage", "Champion")
        except Exception:
            pass

    arena = V3FortressArena(size=12, scenario=scenario, spawn_pos=(1, 1))
    pos = arena.spawn_pos
    prev_pos = None

    action_names = ["Move North (Up)", "Move South (Down)", "Move West (Left)", "Move East (Right)"]
    action_symbols = ["⬆️", "⬇️", "⬅️", "➡️"]
    moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    step = 0
    food_eaten = False
    hit_hazard = False
    retreat_steps = 0

    scenario_labels = {
        "sealed": "🔒 LOCKED CITADEL (8 Hazards Enclosing Center Food - Impossible Trap)",
        "gate":   "🚪 FORTRESS GATE (7 Hazards with 1 North Opening - Breach Challenge)",
        "moat":   "🏰 PERIMETER MOAT (Radius-2 Outer Wall with Single Breach)"
    }

    clear_screen()
    model_type_str = f"{GREEN}System 1 Recurrent GRU + System 2 VillageMemory{RESET}" if recurrent else f"{CYAN}Feedforward MLP (Stateless){RESET}"
    print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║                     🔬 V3 EXPERIMENTAL STRESS TEST BENCH                              ║{RESET}")
    print(f"{CYAN}{BOLD}╠═══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"║ {BOLD}Scenario:{RESET} {YELLOW}{scenario_labels.get(scenario, scenario):<68}{RESET} ║")
    print(f"║ {BOLD}Brain Type:{RESET} {model_type_str:<76} ║")
    print(f"║ {BOLD}Step Limit:{RESET} {YELLOW}{'Unlimited (Agent Decides When to Give Up)' if max_steps is None else f'{max_steps} steps':<68}{RESET} ║")
    print(f"║ {BOLD}Champion:{RESET} Generation {CYAN}{gen_num}{RESET} │ Lineage: {MAGENTA}{lineage[:30]}{RESET}                                     ║")
    print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════════════════╝{RESET}\n")

    if interactive:
        input(f"{YELLOW}Press [Enter] to launch the test...{RESET}")

    while max_steps is None or step < max_steps:
        step += 1
        obs, target_food, nearest_hazard = arena.get_observation(pos)

        reflex_blocked = []
        frustration_val = 0.0

        if recurrent:
            ctx_obs = recurrent_agent.memory.get_context_vector(
                pos=pos,
                target_food_dx=obs[0],
                target_food_dy=obs[1],
                hazard_dx=obs[2],
                hazard_dy=obs[3],
                energy=obs[4]
            )
            with torch.no_grad():
                action = recurrent_agent.select_action(
                    ctx_obs,
                    pos=pos,
                    grid_size=arena.size,
                    hazards=arena.hazards,
                    temperature=0.3
                )
            frustration_val = recurrent_agent.memory.compute_frustration()
            # Check reflex blocks for telemetry
            ar, ac = pos
            for act_idx, (dr, dc) in moves.items():
                if (ar + dr, ac + dc) in arena.hazards:
                    reflex_blocked.append(action_symbols[act_idx])
        else:
            state_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                h = torch.tanh(feedforward_model.fc1(state_t))
                raw_logits = feedforward_model.fc2(h).clone()
                logits = raw_logits.clone()

                # Border wall masking
                ar, ac = pos
                if ar == 0: logits[0] = -1e9
                if ar == arena.size - 1: logits[1] = -1e9
                if ac == 0: logits[2] = -1e9
                if ac == arena.size - 1: logits[3] = -1e9

                # Reflex Guard (Pain receptor)
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) in arena.hazards:
                        logits[act_idx] -= 30.0
                        reflex_blocked.append(action_symbols[act_idx])

                # Anti-Ping-Pong / Momentum bias
                if prev_pos is not None:
                    for act_idx, (dr, dc) in moves.items():
                        if (ar + dr, ac + dc) == prev_pos:
                            logits[act_idx] -= 0.5

                probs = F.softmax(logits / 0.4, dim=-1)
                action = int(torch.argmax(probs).item())

        prev_pos = pos
        new_pos, ate_food, hit_hazard = arena.step(pos, action)
        if recurrent:
            recurrent_agent.memory.record_step(
                new_pos,
                food_pos=arena.foods[0] if arena.foods else None,
                hazards=arena.hazards,
                ate_food=ate_food,
                hit_hazard=hit_hazard
            )

        clear_screen()
        status_bar = f"{scenario_labels.get(scenario, scenario)[:65]}"
        step_str = f"{step}" if max_steps is None else f"{step}/{max_steps}"
        print(f"{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║ {status_bar:^73} ║{RESET}")
        print(f"{CYAN}{BOLD}╠═══════════════════════════════════════════════════════════════════════════╣{RESET}")
        status_txt = '🍏 CLEARED!' if ate_food else ('💀 DEAD!' if hit_hazard else ('🏳️ GIVING UP' if (recurrent and recurrent_agent.memory.abandon_target) else 'Active Hunting'))
        print(f"║ {BOLD}Step:{RESET} {CYAN}{step_str:<7}{RESET} │ {BOLD}Pos:{RESET} ({pos[0]:2d},{pos[1]:2d}) │ {BOLD}Food Target:{RESET} {YELLOW}(5,5){RESET} │ {BOLD}Status:{RESET} {status_txt:<18} ║")
        print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

        # Draw Arena
        for r in range(arena.size):
            row_str = "    "
            for c in range(arena.size):
                if (r, c) == new_pos and hit_hazard:
                    row_str += f"{RED}{BOLD}💥{RESET} "
                elif (r, c) == new_pos and ate_food:
                    row_str += f"{GREEN}{BOLD}🏆{RESET} "
                elif (r, c) == new_pos:
                    row_str += f"{YELLOW}{BOLD}👑{RESET} "
                elif (r, c) in arena.foods:
                    row_str += f"{GREEN}🍏{RESET} "
                elif (r, c) in arena.hazards:
                    row_str += f"{RED}⚠️ {RESET}"
                else:
                    row_str += f"{DIM}· {RESET} "
            print(row_str)

        # Brain Telemetry
        print(f"\n{BOLD}[ 🧠 CHAMPION'S THOUGHT PROCESS ]{RESET}")
        food_dist = abs(arena.center[0] - new_pos[0]) + abs(arena.center[1] - new_pos[1])
        print(f"  • Visual Senses  : Food at {arena.center} (dist={food_dist}) │ dx={obs[0]:+.2f}, dy={obs[1]:+.2f}")

        if recurrent:
            f_pct = int(frustration_val * 10)
            f_bar = "█" * f_pct + "░" * (10 - f_pct)
            f_color = RED if frustration_val >= 0.4 else (YELLOW if frustration_val > 0.1 else GREEN)
            h_norm = float(torch.norm(recurrent_agent.hx).item())
            print(f"  • Village Memory : Frustration {f_color}[{f_bar}] {frustration_val*100:3.0f}%{RESET} │ GRU State Norm: {h_norm:.2f} │ Goal: {recurrent_agent.memory.current_goal}")
            if frustration_val >= 0.4:
                print(f"  • {YELLOW}{BOLD}LOOP-BREAKER   : ⚡ ACTIVE (Suppressed ping-pong axis to force orthogonal flank escape!){RESET}")

        if nearest_hazard:
            h_dist = abs(nearest_hazard[0] - pos[0]) + abs(nearest_hazard[1] - pos[1])
            danger_str = f"{RED}{BOLD}🚨 IMMINENT DANGER! (dist={h_dist}){RESET}" if h_dist <= 2 else f"{YELLOW}Alarm (dist={h_dist}){RESET}"
            print(f"  • Hazard Radar   : Nearest Hazard at {nearest_hazard} │ {danger_str}")

        if reflex_blocked:
            print(f"  • {RED}{BOLD}PAIN REFLEX    : 🚫 BLOCKED {', '.join(reflex_blocked)} (-30.0 logit penalty to prevent fatal crash!){RESET}")
        else:
            print(f"  • {DIM}PAIN REFLEX    : No immediate collision on current adjacent tiles{RESET}")

        if not recurrent:
            print(f"  • Motor Probabilities:")
            for ai, name in enumerate(action_names):
                p = float(probs[ai].item())
                bar = "█" * int(p * 15)
                marker = f"{GREEN}{BOLD}👉 CHOSEN{RESET}" if ai == action else "         "
                print(f"    {marker} {action_symbols[ai]} {name:<20} [{bar:<15}] {p*100:5.1f}%")

        if hit_hazard:
            print(f"\n{RED}{BOLD}💀 CATASTROPHIC COLLISION! The champion succumbed to temptation and stepped on ⚠️!{RESET}\n")
            return

        if ate_food:
            print(f"\n{GREEN}{BOLD}🎉 MASTERCLASS BREACH! The champion navigated the hazards and claimed the center food!{RESET}\n")
            return

        # Check Autonomous Give-Up Completion
        if recurrent and recurrent_agent.memory.abandon_target:
            retreat_steps += 1
            h_dists = [abs(hr - new_pos[0]) + abs(hc - new_pos[1]) for hr, hc in arena.hazards]
            min_h_dist = min(h_dists) if h_dists else 99

            # Safely disengaged to distance >= 3 from all hazards after deciding to give up
            if min_h_dist >= 3 and retreat_steps >= 4:
                print(f"\n{YELLOW}{BOLD}════════════════════════════════════════════════════════════════════════════{RESET}")
                print(f"{YELLOW}{BOLD}🏳️ MISSION ABORTED: THE AGENT AUTONOMOUSLY DECIDED TO GIVE UP!{RESET}")
                print(f"  • {BOLD}Cognitive Reason:{RESET} {recurrent_agent.memory.abandon_reason}")
                print(f"  • {BOLD}Decision Outcome:{RESET} {GREEN}{BOLD}Champion safely broke off pursuit and retreated to safety at {new_pos}!{RESET}")
                print(f"  • {BOLD}Total Steps Taken:{RESET} {step} steps | {GREEN}0 hazard collisions{RESET}.")
                print(f"  • {BOLD}Life Preserved:{RESET} Recognized an impossible trap and prioritized survival over reckless feeding.")
                print(f"{YELLOW}{BOLD}════════════════════════════════════════════════════════════════════════════{RESET}\n")
                return

        pos = new_pos

        if interactive:
            user_input = input(f"\n{DIM}[Enter] next step, or 'q' to quit, 'a' for auto-play: {RESET}").strip().lower()
            if user_input == 'q':
                break
            elif user_input == 'a':
                interactive = False
        else:
            time.sleep(1.0 / fps)

    # If loop finishes without eating or dying:
    print(f"\n{CYAN}{BOLD}════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{YELLOW}{BOLD}🛡️ RESULT: PERIMETER SURVIVAL PATROL!{RESET}")
    print(f"The champion approached the food fortress, recognized the ⚠️ hazard cage,")
    print(f"and {GREEN}{BOLD}REFUSED to commit suicide{RESET} to reach the center apple!")
    print(f"It safely orbited the fortress perimeter, prioritizing survival over reckless feeding.")
    print(f"{CYAN}{BOLD}════════════════════════════════════════════════════════════════════════════{RESET}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Test: Food in center, surrounded by hazards.")
    parser.add_argument("--scenario", choices=["sealed", "gate", "moat"], default="sealed",
                        help="sealed: completely walled trap; gate: 1 breach opening; moat: radius-2 wall")
    parser.add_argument("--recurrent", action="store_true", help="Use System 1 Recurrent GRU + System 2 VillageMemory")
    parser.add_argument("--auto", action="store_true", help="Run in continuous auto-play mode")
    parser.add_argument("--fps", type=float, default=6.0, help="Frames per second in auto mode")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step limit (default: unlimited, runs until agent decides to give up)")
    parser.add_argument("--patience", type=int, default=50, help="Probing ticks before evaluating give up (default: 50)")
    args = parser.parse_args()

    run_v3_test(scenario=args.scenario, interactive=not args.auto, fps=args.fps, max_steps=args.max_steps, recurrent=args.recurrent, patience=args.patience)
