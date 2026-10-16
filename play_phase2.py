#!/usr/bin/env python3
import sys
import time
import argparse
from typing import Tuple, List
import numpy as np
import torch

from env_phase2 import Phase2Environment
from agent_phase2 import Phase2Agent

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def clear_screen():
    print("\033[H\033[J", end="")

def make_bar(value: float, max_val: float = 100.0, length: int = 12, fill_char: str = "█", empty_char: str = "░", color: str = GREEN) -> str:
    ratio = max(0.0, min(1.0, value / max_val))
    filled_len = int(round(ratio * length))
    empty_len = length - filled_len
    return f"{color}{fill_char * filled_len}{DIM}{empty_char * empty_len}{RESET} {value:5.1f}%"

def render_phase2(
    env: Phase2Environment,
    agent: Phase2Agent,
    agent_pos: Tuple[int, int],
    tick: int,
    action_name: str,
    recent_events: List[str]
):
    clear_screen()
    size = env.size
    mem = agent.memory

    # Time of Day calculation (100 ticks per day)
    day_tick = tick % 100
    day_num = (tick // 100) + 1
    if day_tick < 25:
        tod = f"{YELLOW}🌅 Morning{RESET}"
    elif day_tick < 65:
        tod = f"{CYAN}☀️ Afternoon{RESET}"
    elif day_tick < 85:
        tod = f"{MAGENTA}🌆 Dusk{RESET}"
    else:
        tod = f"{BLUE}🌙 Night{RESET}"

    # Goal Badge with appropriate color
    goal = mem.current_goal
    if goal == "HYDRATE":
        goal_badge = f"{CYAN}{BOLD}💧 HYDRATE (Seeking Water){RESET}"
    elif goal == "RETURN_HOME":
        goal_badge = f"{YELLOW}{BOLD}⛺ RETURNING HOME (Haul Full / Fatigue){RESET}"
    elif goal == "REST":
        goal_badge = f"{BLUE}{BOLD}💤 RESTING IN SHELTER{RESET}"
    elif goal == "SURVIVE":
        goal_badge = f"{RED}{BOLD}🚨 SURVIVE (Critical Vitals!){RESET}"
    else:
        goal_badge = f"{GREEN}{BOLD}🍏 FORAGING & HARVESTING{RESET}"

    print(f"{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║  🏕️  PHASE 2: THE HERO'S EXPEDITION & HOMEOSTASIS SIMULATOR                      ║{RESET}")
    print(f"{BOLD}║  Day {day_num} • {tod} • Tick: {tick:<4} • Action: {action_name:<16}              ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}")

    # Render Board rows alongside HUD panels
    grid_lines = []
    for r in range(size):
        row_str = "  "
        for c in range(size):
            pt = (r, c)
            if pt == agent_pos:
                if goal == "REST" and pt == env.home_pos:
                    row_str += "😴 "
                else:
                    row_str += f"{BOLD}{YELLOW}🧙{RESET} "
            elif pt == env.home_pos:
                row_str += "⛺ "
            elif pt in env.water_tiles:
                row_str += "💧 "
            elif pt in env.foods:
                row_str += "🍏 "
            elif pt in env.trees:
                row_str += "🌲 "
            elif pt in env.hazards:
                row_str += "⚠️ "
            else:
                row_str += f"{DIM}· {RESET}"
        grid_lines.append(row_str)

    # Prepare HUD lines
    hud_lines = [
        f"{BOLD}🎯 Active Goal:{RESET} {goal_badge}",
        f"{BOLD}📍 Current Pos:{RESET} {agent_pos}  ->  {BOLD}Target:{RESET} {mem.target_pos}",
        "",
        f"{BOLD}─── Biological Vitals (System 2) ───────────────────────────{RESET}",
        f"  ❤️  Health : {make_bar(mem.vitals['health'], 100.0, 10, color=RED if mem.vitals['health'] < 40 else GREEN)}",
        f"  💧 Thirst : {make_bar(mem.vitals['thirst'], 100.0, 10, color=RED if mem.vitals['thirst'] >= 70 else CYAN)}",
        f"  🍽️  Hunger : {make_bar(mem.vitals['hunger'], 100.0, 10, color=RED if mem.vitals['hunger'] >= 70 else YELLOW)}",
        f"  ⚡ Energy : {make_bar(mem.vitals['energy'], 100.0, 10, color=RED if mem.vitals['energy'] <= 25 else GREEN)}",
        "",
        f"{BOLD}─── Inventory & Village Stockpile ──────────────────────────{RESET}",
        f"  🎒 Backpack ({mem.carried_count}/{mem.backpack_capacity}): [{'🍏 ' * mem.backpack['food']}{'🪵 ' * mem.backpack['wood']}{'░ ' * (mem.backpack_capacity - mem.carried_count)}]",
        f"  ⛺ Home Stockpile : {GREEN}🍏 {env.stockpile['food']} Food{RESET}  |  {YELLOW}🪵 {env.stockpile['wood']} Wood{RESET}",
        "",
        f"{BOLD}─── System 1 Working Memory (GRU) ──────────────────────────{RESET}",
        f"  🧠 Memory Norm ||h_t|| : {torch.norm(agent.hx).item():5.2f}  |  Frustration: {mem.compute_frustration()*100:4.1f}%",
        f"  ⚠️ Hazards Memorized  : {len(mem.hazards_seen)} locations",
        "",
        f"{BOLD}─── Recent Cognitive Events ────────────────────────────────{RESET}"
    ]

    for ev in recent_events[-3:]:
        hud_lines.append(f"  • {ev}")
    while len(hud_lines) < size:
        hud_lines.append("")

    for i in range(max(len(grid_lines), len(hud_lines))):
        g_line = grid_lines[i] if i < len(grid_lines) else " " * (size * 3 + 2)
        h_line = hud_lines[i] if i < len(hud_lines) else ""
        print(f"{g_line}   │  {h_line}")

    print(f"\n{DIM}Controls: Press Ctrl+C to stop simulation.{RESET}")

def main():
    parser = argparse.ArgumentParser(description="Phase 2: The Hero's Life Cycle Simulator")
    parser.add_argument("--ticks", type=int, default=300, help="Total simulation ticks (default: 300)")
    parser.add_argument("--delay", type=float, default=0.10, help="Tick delay in seconds (default: 0.10)")
    parser.add_argument("--deterministic", action="store_true", help="Always choose highest-probability action")
    args = parser.parse_args()

    # Initialize Environment & Agent
    env = Phase2Environment(size=14, num_food=7, num_trees=7, num_hazards=4, home_pos=(1, 1))
    agent = Phase2Agent(
        agent_id=1,
        in_features=6,
        hidden_size=16,
        home_pos=env.home_pos,
        backpack_capacity=5,
        ckpt_path="checkpoints/champion_phase2.pt" if Path("checkpoints/champion_phase2.pt").exists() else "checkpoints/champion_v3_gru.pt"
    )

    agent_pos = env.home_pos
    action_names = ["North ⬆️", "South ⬇️", "West ⬅️", "East ➡️"]
    recent_events = ["Simulation initialized. Hero awoke at Home Base ⛺"]

    try:
        for tick in range(1, args.ticks + 1):
            # 1. Observation based on current target pos
            obs = env.get_observation(
                agent_pos=agent_pos,
                target_pos=agent.memory.target_pos,
                energy=agent.memory.vitals["energy"] / 100.0,
                frustration=agent.memory.compute_frustration()
            )

            # 2. Action selection with System 1 GRU + reflexive guards
            # If resting at home and already recovering, don't move
            if agent.memory.current_goal == "REST" and agent_pos == env.home_pos and agent.memory.vitals["energy"] < 95.0:
                action = -1
                action_name = "Sleeping 💤"
                new_pos = agent_pos
                event = None
                hit_hazard = False
            else:
                action = agent.select_action(
                    obs=obs,
                    pos=agent_pos,
                    grid_size=env.size,
                    hazards=env.hazards,
                    water_tiles=env.water_tiles,
                    temperature=0.35,
                    deterministic=args.deterministic
                )
                action_name = action_names[action]
                new_pos, event, hit_hazard = env.step(agent_pos, action)

            # 3. Step System 2 Homeostasis & Vitals
            new_events = agent.memory.update_homeostasis(
                current_pos=new_pos,
                env=env,
                step_event=event,
                hit_hazard=hit_hazard
            )
            recent_events.extend(new_events)
            agent_pos = new_pos

            # 4. Render Frame
            render_phase2(
                env=env,
                agent=agent,
                agent_pos=agent_pos,
                tick=tick,
                action_name=action_name,
                recent_events=recent_events
            )

            # 5. Check if agent perished
            if agent.memory.vitals["health"] <= 0.0:
                print(f"\n{RED}{BOLD}💀 The hero has fallen! Health reached 0.{RESET}")
                break

            time.sleep(args.delay)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Simulation interrupted by user.{RESET}")

    print(f"\n{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}📊 EXPEDITION SUMMARY:{RESET}")
    print(f"  • Total Ticks Lived    : {tick}")
    print(f"  • Village Food Granary : {env.stockpile['food']} 🍏")
    print(f"  • Village Lumber Depot : {env.stockpile['wood']} 🪵")
    print(f"  • Remaining Backpack   : {agent.memory.backpack['food']} 🍏, {agent.memory.backpack['wood']} 🪵")
    print(f"  • Final Health         : {agent.memory.vitals['health']:.1f}%")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}\n")

if __name__ == "__main__":
    main()
