#!/usr/bin/env python3
import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from env_phase2 import Phase2Environment
from agent_phase2 import Phase2Agent

def render_bar(val: float, max_val: float = 10.0, width: int = 14) -> str:
    filled = int(min(1.0, max(0.0, val / max_val)) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def create_training_environment(grid_size: int = 14) -> Phase2Environment:
    """Creates a diverse ecological training world."""
    return Phase2Environment(
        size=grid_size,
        num_food=random.randint(6, 8),
        num_trees=random.randint(6, 8),
        num_hazards=random.randint(3, 5),
        home_pos=(1, 1),
        water_center=(random.randint(6, 8), random.randint(6, 8))
    )

def run_phase2_training(
    num_generations: int = 100,
    population_size: int = 8,
    ticks_per_gen: int = 160,
    grid_size: int = 14,
    resume: bool = True
):
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    pop_ckpt_path = checkpoints_dir / "population_phase2.pt"
    history_path = checkpoints_dir / "history_phase2.json"
    champion_path = checkpoints_dir / "champion_phase2.pt"
    seed_gru_path = checkpoints_dir / "champion_v3_gru.pt"

    population: List[Phase2Agent] = []
    start_gen = 1
    history_best_deposits = []
    history_avg_deposits = []

    # Resume from Phase 2 checkpoint if available, or seed from Phase 1 champion
    if resume and pop_ckpt_path.exists():
        try:
            ckpt = torch.load(pop_ckpt_path, weights_only=True)
            saved_agents = ckpt.get("agents", [])
            start_gen = ckpt.get("generation", 0) + 1

            for sa in saved_agents[:population_size]:
                agent = Phase2Agent(agent_id=sa["id"], ckpt_path=None)
                agent.model.load_state_dict(sa["weights"])
                agent.lineage = sa.get("lineage", f"Hero_A{sa['id']}")
                agent.generation = sa.get("generation", 1)
                population.append(agent)

            while len(population) < population_size:
                population.append(Phase2Agent(agent_id=len(population), ckpt_path=str(seed_gru_path) if seed_gru_path.exists() else None))

            if history_path.exists():
                with open(history_path, "r") as f:
                    h = json.load(f)
                    history_best_deposits = h.get("history_best_deposits", [])
                    history_avg_deposits = h.get("history_avg_deposits", [])

            print(f"\n🔄 Resuming Phase 2 Evolution from Generation {start_gen}")
        except Exception as e:
            print(f"Notice: Initializing fresh population ({e})")
            population = [Phase2Agent(agent_id=i, ckpt_path=str(seed_gru_path) if seed_gru_path.exists() else None) for i in range(population_size)]
    else:
        population = [Phase2Agent(agent_id=i, ckpt_path=str(seed_gru_path) if seed_gru_path.exists() else None) for i in range(population_size)]

    end_gen = start_gen + num_generations - 1

    print("\n" + "=" * 88)
    print("🏕️  PHASE 2: HYBRID ERL MULTI-GENERATIONAL HERO TRAINING (LIFECYCLE & HOMEOSTASIS)")
    print("=" * 88)
    print(f"• Population Size   : {population_size} Autonomous Villager Agents")
    print(f"• Generations       : {num_generations} (Gen {start_gen} -> {end_gen})")
    print(f"• Ticks/Generation  : {ticks_per_gen} steps (Inner Loop: BPTT Lifetime Learning)")
    print(f"• Ecological World  : 14x14 Grid (Home ⛺, Water 💧, Food 🍏, Timber 🌲, Hazards ⚠️)")
    print(f"• Fitness Objective : Stockpile Deposits (🍏+🪵) + Quenched Thirst + Survival Longevity")
    print(f"• Outer Loop Policy : Elitism (Top 2) + Generational Culling + Gaussian Weight Mutation")
    print("=" * 88)

    total_start_time = time.perf_counter()
    total_steps_simulated = 0

    print(f"{'GEN':>5} | {'AVG DEPOSITS':>12} {'PROGRESS':^16} | {'CHAMPION (DEP/HARV)':>20} | {'SURVIVAL':>9} | {'SPEED':>9} | {'TOP LINEAGE'}")
    print("-" * 92)

    for gen in range(start_gen, end_gen + 1):
        gen_start_time = time.perf_counter()
        env = create_training_environment(grid_size=grid_size)

        for agent in population:
            agent.reset_state()
            agent.fitness = 0.0
            agent.deposited_food = 0
            agent.deposited_wood = 0
            agent.items_harvested = 0
            agent.waters_quenched = 0
            agent.hazards_hit = 0

        # Spawning: Agents awaken at Home Base
        positions = {agent.id: env.home_pos for agent in population}
        alive = {agent.id: True for agent in population}

        # --- SIMULATION OF ONE GENERATION LIFETIME ---
        for tick in range(ticks_per_gen):
            any_alive = False
            for agent in population:
                if not alive[agent.id]:
                    continue
                any_alive = True
                pos = positions[agent.id]

                obs = env.get_observation(
                    agent_pos=pos,
                    target_pos=agent.memory.target_pos,
                    energy=agent.memory.vitals["energy"] / 100.0,
                    frustration=agent.memory.compute_frustration()
                )

                # Action selection
                if agent.memory.current_goal == "REST" and pos == env.home_pos and agent.memory.vitals["energy"] < 95.0:
                    action = -1
                    new_pos = pos
                    step_event = None
                    hit_hazard = False
                else:
                    action = agent.select_action(
                        obs=obs,
                        pos=pos,
                        grid_size=env.size,
                        hazards=env.hazards,
                        water_tiles=env.water_tiles,
                        temperature=0.35
                    )
                    new_pos, step_event, hit_hazard = env.step(pos, action)

                # Prior distance to active target
                tr, tc = agent.memory.target_pos
                prev_dist = abs(tr - pos[0]) + abs(tc - pos[1])
                new_dist = abs(tr - new_pos[0]) + abs(tc - new_pos[1])

                # Shaped lifetime step reward
                step_reward = 0.0
                if prev_dist > new_dist:
                    step_reward += 0.25  # Approaching target
                elif prev_dist < new_dist:
                    step_reward -= 0.15

                # Stockpile before update
                old_food_stock = env.stockpile["food"]
                old_wood_stock = env.stockpile["wood"]
                old_thirst = agent.memory.vitals["thirst"]

                # Step System 2 Homeostasis
                events = agent.memory.update_homeostasis(
                    current_pos=new_pos,
                    env=env,
                    step_event=step_event,
                    hit_hazard=hit_hazard
                )
                positions[agent.id] = new_pos

                # Event Rewards
                if step_event == "FOOD_HARVESTED":
                    step_reward += 4.0
                    agent.items_harvested += 1
                elif step_event == "WOOD_HARVESTED":
                    step_reward += 4.0
                    agent.items_harvested += 1

                # Quenched thirst bonus
                if agent.memory.vitals["thirst"] < 5.0 and old_thirst >= 65.0:
                    step_reward += 8.0
                    agent.waters_quenched += 1

                # Deposited haul bonus (+18 per item delivered to stockpile)
                deposited_food = env.stockpile["food"] - old_food_stock
                deposited_wood = env.stockpile["wood"] - old_wood_stock
                if deposited_food > 0 or deposited_wood > 0:
                    delivered_count = deposited_food + deposited_wood
                    step_reward += 18.0 * delivered_count
                    agent.deposited_food += deposited_food
                    agent.deposited_wood += deposited_wood

                # Hazard damage penalty
                if hit_hazard:
                    step_reward -= 25.0
                    agent.hazards_hit += 1

                # High frustration stagnation penalty
                frustration = agent.memory.compute_frustration()
                if frustration >= 0.5:
                    step_reward -= 0.3 * frustration

                # Record lifetime reward
                agent.record_reward(step_reward)

                # Death check
                if agent.memory.vitals["health"] <= 0.0:
                    alive[agent.id] = False

            if not any_alive:
                break

        total_steps_simulated += ticks_per_gen * population_size

        # --- INNER LOOP: LIFETIME RL LEARNING (BPTT) ---
        for agent in population:
            agent.train_rl_lifetime()

        # --- OUTER LOOP: EVOLUTIONARY SELECTION ---
        for a in population:
            total_deposited = a.deposited_food + a.deposited_wood
            death_penalty = 60.0 if a.memory.vitals["health"] <= 0.0 else 0.0
            a.fitness = (
                (a.deposited_food * 25.0) +
                (a.deposited_wood * 25.0) +
                (a.items_harvested * 4.0) +
                (a.waters_quenched * 12.0) +
                (a.memory.vitals["health"] * 0.2) -
                (a.hazards_hit * 80.0) -
                death_penalty
            )

        population.sort(key=lambda a: a.fitness, reverse=True)

        best_agent = population[0]
        best_deposits = best_agent.deposited_food + best_agent.deposited_wood
        avg_deposits = sum(a.deposited_food + a.deposited_wood for a in population) / float(population_size)
        survived_count = sum(1 for a in population if a.memory.vitals["health"] > 0.0)

        history_best_deposits.append(best_deposits)
        history_avg_deposits.append(avg_deposits)

        # Genetic Reproduction: Top 2 Elites clone & mutate to replace bottom 2
        num_elites = 2
        elites = population[:num_elites]
        new_offspring = []
        for i, elite in enumerate(elites):
            child_id = population[-1 - i].id
            child = elite.clone_and_mutate(new_id=child_id, mutation_rate=0.12, mutation_scale=0.15)
            new_offspring.append(child)

        population[-num_elites:] = new_offspring

        gen_duration = time.perf_counter() - gen_start_time
        gen_throughput = int((ticks_per_gen * population_size) / max(gen_duration, 1e-5))
        bar = render_bar(avg_deposits, max_val=8.0)

        lineage_str = best_agent.lineage
        if len(lineage_str) > 22:
            lineage_str = lineage_str[:9] + "..." + lineage_str[-10:]

        champ_stats = f"{best_deposits:2d} dep ({best_agent.items_harvested:2d} hrv)"

        print(f"G{gen:03d}  | {avg_deposits:5.2f} {bar} | {champ_stats:>20} | {survived_count:2d}/{population_size} live | {gen_throughput:5d} st/s | {lineage_str}")

        # Checkpoint every 20 generations or final
        if gen % 20 == 0 or gen == end_gen:
            # 1. Save Phase 2 Champion model
            torch.save(best_agent.model.state_dict(), champion_path)

            # 2. Save Population
            pop_data = {
                "generation": gen,
                "agents": [
                    {
                        "id": a.id,
                        "weights": a.model.state_dict(),
                        "lineage": a.lineage,
                        "generation": a.generation
                    }
                    for a in population
                ]
            }
            torch.save(pop_data, pop_ckpt_path)

            # 3. Save History
            hist_data = {
                "total_generations": gen,
                "dominant_lineage": best_agent.lineage,
                "history_best_deposits": history_best_deposits,
                "history_avg_deposits": history_avg_deposits
            }
            with open(history_path, "w") as f:
                json.dump(hist_data, f, indent=2)

    total_time = time.perf_counter() - total_start_time
    print("-" * 92)
    print(f"✅ Phase 2 Evolutionary Training Complete: {num_generations} Generations in {total_time:.2f}s ({total_steps_simulated / total_time:.0f} steps/sec)")
    print(f"💾 Phase 2 Champion Saved to: {champion_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Generation Phase 2 Evolutionary Training")
    parser.add_argument("--generations", type=int, default=50, help="Generations to train (default: 50)")
    parser.add_argument("--pop", type=int, default=8, help="Population size (default: 8)")
    parser.add_argument("--ticks", type=int, default=160, help="Ticks per generation (default: 160)")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh (do not resume)")
    args = parser.parse_args()

    run_phase2_training(
        num_generations=args.generations,
        population_size=args.pop,
        ticks_per_gen=args.ticks,
        resume=not args.fresh
    )
