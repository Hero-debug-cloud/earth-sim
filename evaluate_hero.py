#!/usr/bin/env python3
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from env_phase2 import Phase2Environment
from agent_phase2 import Phase2Agent

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def render_meter(val: float, max_val: float = 100.0, length: int = 8, color: str = GREEN) -> str:
    ratio = max(0.0, min(1.0, val / max_val))
    filled = int(round(ratio * length))
    empty = length - filled
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET} {val:4.0f}%"

def get_tod_name(tick: int, ticks_per_day: int = 100) -> Tuple[int, str]:
    day_num = ((tick - 1) // ticks_per_day) + 1
    day_tick = (tick - 1) % ticks_per_day
    if day_tick < 25:
        return day_num, "Morning 🌅"
    elif day_tick < 65:
        return day_num, "Afternoon ☀️"
    elif day_tick < 85:
        return day_num, "Dusk 🌆"
    else:
        return day_num, "Night 🌙"

# ==============================================================================
# SINGLE HERO EVALUATION TRIAL
# ==============================================================================

def run_single_survival_trial(
    agent: Phase2Agent,
    max_days: int = 10,
    ticks_per_day: int = 100,
    grid_size: int = 14,
    seed: int = 42,
    visual: bool = False,
    delay: float = 0.05
) -> Dict[str, Any]:
    """
    Simulates a single hero across multiple day/night cycles until death or max_days.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = Phase2Environment(size=grid_size, num_food=7, num_trees=7, num_hazards=4, home_pos=(1, 1))
    agent.reset_state()
    agent_pos = env.home_pos

    total_ticks = max_days * ticks_per_day
    cause_of_death = "Survived Target Duration (Immortal Elder)"
    times_drank = 0
    times_eaten = 0
    rest_cycles = 0

    action_names = ["North ⬆️", "South ⬇️", "West ⬅️", "East ➡️"]

    for tick in range(1, total_ticks + 1):
        day_num, tod_name = get_tod_name(tick, ticks_per_day)

        obs = env.get_observation(
            agent_pos=agent_pos,
            target_pos=agent.memory.target_pos,
            energy=agent.memory.vitals["energy"] / 100.0,
            frustration=agent.memory.compute_frustration()
        )

        if agent.memory.current_goal == "REST" and agent_pos == env.home_pos and agent.memory.vitals["energy"] < 95.0:
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
                temperature=0.25,
                deterministic=True
            )
            action_name = action_names[action]
            new_pos, event, hit_hazard = env.step(agent_pos, action)

        old_thirst = agent.memory.vitals["thirst"]
        old_hunger = agent.memory.vitals["hunger"]
        old_energy = agent.memory.vitals["energy"]

        events = agent.memory.update_homeostasis(
            current_pos=new_pos,
            env=env,
            step_event=event,
            hit_hazard=hit_hazard
        )
        agent_pos = new_pos

        if agent.memory.vitals["thirst"] < 5.0 and old_thirst >= 50.0:
            times_drank += 1
        if agent.memory.vitals["hunger"] < old_hunger:
            times_eaten += 1
        if agent.memory.vitals["energy"] >= 95.0 and old_energy < 40.0:
            rest_cycles += 1

        if visual and (tick % 2 == 0 or hit_hazard):
            print(f"\033[H\033[J", end="")
            print(f"{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗{RESET}")
            print(f"{BOLD}║  🏕️  HERO LONGEVITY BENCHMARK: LIVE SURVIVAL TRACKER                             ║{RESET}")
            print(f"{BOLD}║  Day: {day_num:2d}/{max_days} • {tod_name:<14} • Tick: {tick:4d} • Pos: {str(agent_pos):<8}            ║{RESET}")
            print(f"{BOLD}╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}")
            print(f"  ❤️  Health  : {render_meter(agent.memory.vitals['health'], color=GREEN if agent.memory.vitals['health'] > 40 else RED)}")
            print(f"  💧 Thirst  : {render_meter(agent.memory.vitals['thirst'], color=RED if agent.memory.vitals['thirst'] > 70 else CYAN)}")
            print(f"  🍽️  Hunger  : {render_meter(agent.memory.vitals['hunger'], color=RED if agent.memory.vitals['hunger'] > 70 else YELLOW)}")
            print(f"  ⚡ Energy  : {render_meter(agent.memory.vitals['energy'], color=RED if agent.memory.vitals['energy'] < 25 else GREEN)}")
            print(f"  🎯 Goal    : {BOLD}{agent.memory.current_goal}{RESET}")
            print(f"  🎒 Backpack: {agent.memory.backpack['food']} 🍏, {agent.memory.backpack['wood']} 🪵  |  ⛺ Stockpile: {env.stockpile['food']} 🍏, {env.stockpile['wood']} 🪵")
            if events:
                print(f"  📢 Event   : {events[-1]}")
            time.sleep(delay)

        if agent.memory.vitals["health"] <= 0.0:
            if agent.memory.vitals["thirst"] >= 90.0:
                cause_of_death = "Dehydration (Lethal Thirst)"
            elif agent.memory.vitals["hunger"] >= 90.0:
                cause_of_death = "Starvation (No Food)"
            elif hit_hazard:
                cause_of_death = "Hazard Trap (Mortality)"
            else:
                cause_of_death = "Fatigue & Physical Collapse"
            break

    days_survived = tick / float(ticks_per_day)

    return {
        "seed": seed,
        "ticks_lived": tick,
        "days_survived": round(days_survived, 2),
        "survived_full_duration": (tick == total_ticks and agent.memory.vitals["health"] > 0),
        "cause_of_death": cause_of_death,
        "food_deposited": env.stockpile["food"],
        "wood_deposited": env.stockpile["wood"],
        "total_deposited": env.stockpile["food"] + env.stockpile["wood"],
        "times_drank": times_drank,
        "times_eaten": times_eaten,
        "rest_cycles": rest_cycles,
        "final_health": round(agent.memory.vitals["health"], 1)
    }

# ==============================================================================
# MULTI-HERO VILLAGE EXPEDITION (10 HEROES SIMULTANEOUSLY)
# ==============================================================================

def run_multi_hero_trial(
    num_heroes: int = 10,
    max_days: int = 100,
    ticks_per_day: int = 100,
    grid_size: int = 14,
    seed: int = 42,
    visual: bool = True,
    delay: float = 0.03,
    ckpt_path: str = "checkpoints/champion_phase2.pt"
) -> Dict[str, Any]:
    """
    Simulates a village society with N heroes living simultaneously on the same map,
    sharing the natural resources and accumulating a massive communal stockpile.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Scale resources appropriately for multi-hero village society
    num_food = max(8, num_heroes)
    num_trees = max(8, num_heroes)
    env = Phase2Environment(
        size=grid_size,
        num_food=num_food,
        num_trees=num_trees,
        num_hazards=4,
        home_pos=(1, 1),
        water_center=(7, 7)
    )

    # Initialize N Heroes
    heroes: List[Phase2Agent] = []
    hero_positions: Dict[int, Tuple[int, int]] = {}
    hero_alive: Dict[int, bool] = {}
    hero_stats: Dict[int, Dict[str, Any]] = {}

    for i in range(1, num_heroes + 1):
        agent = Phase2Agent(
            agent_id=i,
            in_features=6,
            hidden_size=16,
            home_pos=env.home_pos,
            backpack_capacity=5,
            ckpt_path=ckpt_path if Path(ckpt_path).exists() else "checkpoints/champion_v3_gru.pt"
        )
        agent.reset_state()
        heroes.append(agent)
        hero_positions[i] = env.home_pos
        hero_alive[i] = True
        hero_stats[i] = {
            "days_survived": 0.0,
            "cause_of_death": "Survived 100 Days (Immortal Patriarch)",
            "deposited_food": 0,
            "deposited_wood": 0,
            "times_drank": 0,
            "times_eaten": 0,
            "rest_cycles": 0,
            "final_health": 100.0
        }

    total_ticks = max_days * ticks_per_day
    action_names = ["North ⬆️", "South ⬇️", "West ⬅️", "East ➡️"]

    for tick in range(1, total_ticks + 1):
        day_num, tod_name = get_tod_name(tick, ticks_per_day)
        any_alive = False

        for agent in heroes:
            hid = agent.id
            if not hero_alive[hid]:
                continue
            any_alive = True

            pos = hero_positions[hid]
            obs = env.get_observation(
                agent_pos=pos,
                target_pos=agent.memory.target_pos,
                energy=agent.memory.vitals["energy"] / 100.0,
                frustration=agent.memory.compute_frustration()
            )

            # Rest if at home with low energy
            if agent.memory.current_goal == "REST" and pos == env.home_pos and agent.memory.vitals["energy"] < 95.0:
                action_name = "Sleeping 💤"
                new_pos = pos
                event = None
                hit_hazard = False
            else:
                action = agent.select_action(
                    obs=obs,
                    pos=pos,
                    grid_size=env.size,
                    hazards=env.hazards,
                    water_tiles=env.water_tiles,
                    temperature=0.35,
                    deterministic=False
                )
                action_name = action_names[action]
                new_pos, event, hit_hazard = env.step(pos, action)

            old_thirst = agent.memory.vitals["thirst"]
            old_hunger = agent.memory.vitals["hunger"]
            old_energy = agent.memory.vitals["energy"]
            old_stock_f = env.stockpile["food"]
            old_stock_w = env.stockpile["wood"]

            # Step individual homeostasis
            events = agent.memory.update_homeostasis(
                current_pos=new_pos,
                env=env,
                step_event=event,
                hit_hazard=hit_hazard
            )
            hero_positions[hid] = new_pos

            # Track deposits made by this specific hero
            dep_f = env.stockpile["food"] - old_stock_f
            dep_w = env.stockpile["wood"] - old_stock_w
            if dep_f > 0 or dep_w > 0:
                hero_stats[hid]["deposited_food"] += dep_f
                hero_stats[hid]["deposited_wood"] += dep_w

            if agent.memory.vitals["thirst"] < 5.0 and old_thirst >= 50.0:
                hero_stats[hid]["times_drank"] += 1
            if agent.memory.vitals["hunger"] < old_hunger:
                hero_stats[hid]["times_eaten"] += 1
            if agent.memory.vitals["energy"] >= 95.0 and old_energy < 40.0:
                hero_stats[hid]["rest_cycles"] += 1

            # Check individual mortality
            if agent.memory.vitals["health"] <= 0.0:
                hero_alive[hid] = False
                hero_stats[hid]["days_survived"] = round(tick / float(ticks_per_day), 2)
                if agent.memory.vitals["thirst"] >= 90.0:
                    hero_stats[hid]["cause_of_death"] = "Dehydration (Lethal Thirst)"
                elif agent.memory.vitals["hunger"] >= 90.0:
                    hero_stats[hid]["cause_of_death"] = "Starvation (No Food)"
                elif hit_hazard:
                    hero_stats[hid]["cause_of_death"] = "Hazard Trap (Mortality)"
                else:
                    hero_stats[hid]["cause_of_death"] = "Physical Collapse"

        # Update stats for living heroes
        for hid in hero_stats:
            if hero_alive[hid]:
                hero_stats[hid]["days_survived"] = round(tick / float(ticks_per_day), 2)
                hero_stats[hid]["final_health"] = round(heroes[hid - 1].memory.vitals["health"], 1)

        # Visual Rendering of Multi-Hero Village
        alive_count = sum(1 for a in hero_alive.values() if a)
        if visual and (tick % 2 == 0 or tick == 1 or alive_count == 0):
            print("\033[H\033[J", end="")

            print(f"{BOLD}╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
            print(f"{BOLD}║  🏕️  VILLAGE MULTI-HERO SIMULATOR: {num_heroes} HEROES EXPEDITION & ENDURANCE BENCHMARK                         ║{RESET}")
            print(f"{BOLD}║  Day {day_num:3d}/{max_days} • {tod_name:<14} • Tick: {tick:5d} • Active: {alive_count:2d}/{num_heroes} 🛡️ • Stockpile: 🍏 {env.stockpile['food']:3d} | 🪵 {env.stockpile['wood']:3d}    ║{RESET}")
            print(f"{BOLD}╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝{RESET}")

            # Map rows
            occupied_map: Dict[Tuple[int, int], List[int]] = {}
            for hid, pos in hero_positions.items():
                if hero_alive[hid]:
                    occupied_map.setdefault(pos, []).append(hid)

            grid_lines = []
            for r in range(grid_size):
                row_str = " "
                for c in range(grid_size):
                    pt = (r, c)
                    occupants = occupied_map.get(pt, [])
                    if pt == env.home_pos:
                        if len(occupants) > 1:
                            row_str += f"{BOLD}{YELLOW}⛺{len(occupants)}{RESET}"
                        elif len(occupants) == 1:
                            row_str += "⛺🧙"
                        else:
                            row_str += "⛺ "
                    elif len(occupants) > 1:
                        row_str += f"{BOLD}{CYAN}👥{len(occupants)}{RESET}"
                    elif len(occupants) == 1:
                        h_agent = heroes[occupants[0] - 1]
                        if h_agent.memory.current_goal == "REST":
                            row_str += "😴 "
                        else:
                            row_str += f"{BOLD}{YELLOW}🧙{RESET} "
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

            # Roster Table
            roster_lines = [
                f"{BOLD}{'HERO':<5} | {'HEALTH':<14} | {'THIRST':<14} | {'ENERGY':<14} | {'PACK':<6} | {'GOAL':<14}{RESET}",
                f"{DIM}{'-' * 80}{RESET}"
            ]
            for agent in heroes[:10]:
                hid = agent.id
                if not hero_alive[hid]:
                    status_line = f"{RED}Hero {hid:02d} | DEAD (Day {hero_stats[hid]['days_survived']:.1f}) - {hero_stats[hid]['cause_of_death'][:28]}{RESET}"
                    roster_lines.append(status_line)
                    continue

                mem = agent.memory
                v = mem.vitals
                h_meter = render_meter(v["health"], color=GREEN if v["health"] > 40 else RED)
                t_meter = render_meter(v["thirst"], color=RED if v["thirst"] > 70 else CYAN)
                e_meter = render_meter(v["energy"], color=RED if v["energy"] < 25 else GREEN)
                pack_str = f"{'🍏' * mem.backpack['food']}{'🪵' * mem.backpack['wood']}"
                pack_disp = f"{pack_str:<5}"

                goal_color = CYAN if mem.current_goal == "HYDRATE" else (YELLOW if mem.current_goal == "RETURN_HOME" else (BLUE if mem.current_goal == "REST" else GREEN))
                goal_disp = f"{goal_color}{mem.current_goal:<12}{RESET}"

                roster_lines.append(f"H{hid:02d}   | {h_meter} | {t_meter} | {e_meter} | {pack_disp} | {goal_disp}")

            # Print combined side-by-side
            for idx in range(max(len(grid_lines), len(roster_lines))):
                g_part = grid_lines[idx] if idx < len(grid_lines) else " " * (grid_size * 3 + 2)
                r_part = roster_lines[idx] if idx < len(roster_lines) else ""
                print(f"{g_part}  │ {r_part}")

            print(f"\n{DIM}Controls: Press Ctrl+C to halt valuation benchmark.{RESET}")
            time.sleep(delay)

        if not any_alive:
            print(f"\n{RED}{BOLD}💀 All {num_heroes} heroes have fallen on Day {day_num}.{RESET}")
            break

    # Summary Calculations
    survived_full_count = sum(1 for a in hero_alive.values() if a)
    days_survived_list = [s["days_survived"] for s in hero_stats.values()]
    avg_days_survived = float(np.mean(days_survived_list))

    total_food_deposited = env.stockpile["food"]
    total_wood_deposited = env.stockpile["wood"]

    print("\n" + f"{BOLD}══════════════════════════════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}🏆 MULTI-HERO VILLAGE SURVIVAL SCORECARD ({num_heroes} HEROES • {max_days} DAYS HORIZON){RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"  • Survival Rate        : {BOLD}{(survived_full_count / num_heroes) * 100:5.1f}%{RESET} ({survived_full_count}/{num_heroes} heroes survived the full {max_days} days)")
    print(f"  • Mean Longevity       : {BOLD}{avg_days_survived:5.2f} Days{RESET} (Range: {min(days_survived_list):.1f}d to {max(days_survived_list):.1f}d)")
    print(f"  • Village Granary      : {GREEN}{total_food_deposited} 🍏 Food{RESET} gathered and stockpiled")
    print(f"  • Village Lumber Depot : {YELLOW}{total_wood_deposited} 🪵 Wood{RESET} gathered and stockpiled")
    print(f"  • Total Village Wealth : {BOLD}{total_food_deposited + total_wood_deposited} Resources{RESET} delivered to Home Base ⛺")
    print("-" * 106)
    print(f"{BOLD}{'RANK':<5} | {'HERO':<8} | {'DAYS SURVIVED':<14} | {'FOOD DEP':<10} | {'WOOD DEP':<10} | {'OUTCOME'}{RESET}")
    print("-" * 106)

    # Sort heroes by days survived then deposits
    sorted_heroes = sorted(
        hero_stats.items(),
        key=lambda item: (item[1]["days_survived"], item[1]["deposited_food"] + item[1]["deposited_wood"]),
        reverse=True
    )

    for rank, (hid, s) in enumerate(sorted_heroes, 1):
        color = GREEN if s["days_survived"] >= max_days else (YELLOW if s["days_survived"] >= max_days * 0.5 else RED)
        print(f"#{rank:<4} | Hero {hid:02d}  | {s['days_survived']:5.1f} / {max_days} d  | {s['deposited_food']:6d} 🍏   | {s['deposited_wood']:6d} 🪵   | {color}{s['cause_of_death']}{RESET}")

    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════════════════════════════{RESET}\n")

    return {
        "num_heroes": num_heroes,
        "max_days": max_days,
        "survived_full_count": survived_full_count,
        "avg_days_survived": avg_days_survived,
        "total_food_deposited": total_food_deposited,
        "total_wood_deposited": total_wood_deposited
    }

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hero Longevity & Survival Valuation Benchmark")
    parser.add_argument("--heroes", type=int, default=1, help="Number of heroes to send simultaneously into the village (e.g. 10)")
    parser.add_argument("--trials", type=int, default=10, help="Number of evaluation trials across diverse seeds (when heroes=1, default: 10)")
    parser.add_argument("--max-days", type=int, default=10, help="Target days to survive per trial (default: 10 days = 1,000 ticks)")
    parser.add_argument("--ticks-per-day", type=int, default=100, help="Ticks per day cycle (default: 100)")
    parser.add_argument("--visual", action="store_true", help="Watch survival in real time with visual ASCII map and HUD")
    parser.add_argument("--delay", type=float, default=0.03, help="Visual mode step delay in seconds (default: 0.03s)")
    parser.add_argument("--ckpt", type=str, default="", help="Custom checkpoint path to evaluate")
    args = parser.parse_args()

    # Determine checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif Path("checkpoints/champion_phase2.pt").exists():
        ckpt_path = Path("checkpoints/champion_phase2.pt")
    else:
        ckpt_path = Path("checkpoints/champion_v3_gru.pt")

    # If user wants multi-hero mode (e.g. 10 heroes simultaneously)
    if args.heroes > 1:
        run_multi_hero_trial(
            num_heroes=args.heroes,
            max_days=args.max_days,
            ticks_per_day=args.ticks_per_day,
            grid_size=14,
            seed=42,
            visual=args.visual,
            delay=args.delay,
            ckpt_path=str(ckpt_path)
        )
        return

    # Single-hero multi-trial mode
    print(f"\n{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}📊 HERO SURVIVAL & HOMEOSTASIS VALUATION BENCHMARK{RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"• Evaluating Checkpoint : {CYAN}{ckpt_path}{RESET}")
    print(f"• Evaluation Trials     : {args.trials} distinct environment seeds")
    print(f"• Endurance Horizon     : {args.max_days} Days ({args.max_days * args.ticks_per_day} ticks per trial)")
    print(f"• Day/Night Metric      : {args.ticks_per_day} ticks/day (Morning 🌅, Afternoon ☀️, Dusk 🌆, Night 🌙)")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}\n")

    agent = Phase2Agent(
        agent_id=1,
        in_features=6,
        hidden_size=16,
        home_pos=(1, 1),
        backpack_capacity=5,
        ckpt_path=str(ckpt_path) if ckpt_path.exists() else None
    )

    results: List[Dict[str, Any]] = []

    print(f"{'TRIAL':>6} | {'SEED':>5} | {'DAYS SURVIVED':>14} | {'FOOD DEP':>9} | {'WOOD DEP':>9} | {'TOTAL DEP':>10} | {'OUTCOME'}")
    print("-" * 92)

    start_eval_time = time.perf_counter()

    for t in range(1, args.trials + 1):
        is_visual = args.visual and (t == 1)
        res = run_single_survival_trial(
            agent=agent,
            max_days=args.max_days,
            ticks_per_day=args.ticks_per_day,
            seed=1000 + t * 37,
            visual=is_visual,
            delay=args.delay
        )
        results.append(res)

        days_str = f"{res['days_survived']:4.1f} / {args.max_days} d"
        outcome_color = GREEN if res["survived_full_duration"] else (YELLOW if res["days_survived"] >= args.max_days * 0.5 else RED)

        print(f"Trial {t:2d} | {res['seed']:5d} | {days_str:>14} | {res['food_deposited']:7d} 🍏 | {res['wood_deposited']:7d} 🪵 | {res['total_deposited']:8d} 📦 | {outcome_color}{res['cause_of_death']}{RESET}")

    total_eval_time = time.perf_counter() - start_eval_time

    days_list = [r["days_survived"] for r in results]
    deposits_list = [r["total_deposited"] for r in results]
    food_list = [r["food_deposited"] for r in results]
    wood_list = [r["wood_deposited"] for r in results]
    survived_count = sum(1 for r in results if r["survived_full_duration"])
    survival_rate = (survived_count / float(args.trials)) * 100.0

    avg_days = float(np.mean(days_list))
    std_days = float(np.std(days_list))
    max_days_lived = float(np.max(days_list))
    min_days_lived = float(np.min(days_list))

    avg_deposits = float(np.mean(deposits_list))
    avg_food = float(np.mean(food_list))
    avg_wood = float(np.mean(wood_list))

    print("\n" + f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}🏆 HERO VALUATION REPORT & SURVIVAL SCORECARD{RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"  • Survival Rate        : {BOLD}{survival_rate:5.1f}%{RESET} ({survived_count}/{args.trials} trials survived full {args.max_days} days)")
    print(f"  • Longevity (Days)     : {BOLD}{avg_days:4.2f} ± {std_days:4.2f} Days{RESET} (Range: {min_days_lived:.1f}d to {max_days_lived:.1f}d)")
    print(f"  • Average Stockpile    : {BOLD}{avg_deposits:4.1f} items / lifetime{RESET} ({avg_food:.1f} 🍏 Food, {avg_wood:.1f} 🪵 Wood)")
    print(f"  • Hydration Frequency  : {np.mean([r['times_drank'] for r in results]):.1f} lake visits per expedition")
    print(f"  • Autonomous Eating    : {np.mean([r['times_eaten'] for r in results]):.1f} backpack snacks consumed")
    print(f"  • Evaluation Speed     : {len(results) * args.max_days * args.ticks_per_day / total_eval_time:.0f} ticks/second ({total_eval_time:.2f}s total)")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}\n")

if __name__ == "__main__":
    main()
