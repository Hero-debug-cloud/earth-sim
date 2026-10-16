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

from env import Environment
from agent import Agent

def render_bar(val: float, max_val: float = 12.0, width: int = 14) -> str:
    filled = int(min(1.0, max(0.0, val / max_val)) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def save_live_telemetry(champion: Agent, env: Environment, pos: tuple[int, int], gen: int, avg_food: float):
    obs = env.get_observation(pos, agent_energy=1.0)
    state_t = torch.tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        h = torch.tanh(champion.model.fc1(state_t))
        logits = champion.model.fc2(h)
        probs = F.softmax(logits, dim=-1)

    action_idx = int(torch.argmax(probs).item())
    action_names = ["Move Up", "Move Down", "Move Left", "Move Right"]
    sensor_names = ["Food dx", "Food dy", "Hazard dx", "Hazard dy", "Energy"]

    telemetry = {
        "gen": gen,
        "lineage": champion.lineage,
        "best_food": champion.food_eaten,
        "best_hazards": getattr(champion, "hazards_hit", 0),
        "best_fitness": round(champion.fitness, 1),
        "avg_food": round(avg_food, 1),
        "sensors": [{"name": sensor_names[i], "val": round(float(obs[i]), 3)} for i in range(len(obs))],
        "hidden": [round(float(h[j].item()), 3) for j in range(len(h))],
        "actions": [{"name": action_names[m], "prob": round(float(probs[m].item()), 3)} for m in range(len(probs))],
        "chosen_action": action_names[action_idx],
        "weights_fc1": champion.model.fc1.weight.detach().numpy().round(3).tolist(),
        "weights_fc2": champion.model.fc2.weight.detach().numpy().round(3).tolist()
    }

    try:
        with open("live_state.json", "w") as f:
            json.dump(telemetry, f)
    except Exception:
        pass

def run_hybrid_training(
    num_generations: int = 500,
    population_size: int = 8,
    ticks_per_gen: int = 150,
    grid_size: int = 12,
    num_food: int = 8,
    num_hazards: int = 3,
    resume: bool = True
):
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    pop_ckpt_path = checkpoints_dir / "population_latest.pt"
    history_path = checkpoints_dir / "history.json"

    env = Environment(size=grid_size, num_food=num_food, num_hazards=num_hazards)

    # Resume or start fresh
    population = []
    start_gen = 1
    history_best = []
    history_avg = []

    if resume and pop_ckpt_path.exists():
        try:
            ckpt = torch.load(pop_ckpt_path, weights_only=True)
            saved_agents = ckpt.get("agents", [])
            start_gen = ckpt.get("generation", 0) + 1

            for sa in saved_agents[:population_size]:
                agent = Agent(agent_id=sa["id"])
                agent.model.load_state_dict(sa["weights"])
                agent.lineage = sa.get("lineage", f"A{sa['id']}")
                agent.generation = sa.get("generation", 1)
                population.append(agent)

            # Fill remaining if population_size increased
            while len(population) < population_size:
                population.append(Agent(agent_id=len(population)))

            if history_path.exists():
                with open(history_path, "r") as f:
                    h = json.load(f)
                    history_best = h.get("history_best", [])
                    history_avg = h.get("history_avg", [])

            print(f"\n🔄 Resuming evolution from Generation {start_gen} (Loaded gene pool from checkpoints/)")
        except Exception as e:
            print(f"⚠️ Could not resume from checkpoint ({e}). Starting fresh.")
            population = [Agent(agent_id=i) for i in range(population_size)]
            start_gen = 1
    else:
        population = [Agent(agent_id=i) for i in range(population_size)]
        start_gen = 1
        history_best = []
        history_avg = []
        print(f"\n🌱 Starting FRESH evolution run from Generation 1 (Clean slate)")

    end_gen = start_gen + num_generations - 1

    print("\n" + "=" * 78)
    print("🧬 HYBRID EVOLUTIONARY REINFORCEMENT LEARNING (ERL) RUN")
    print(f"• Generations: {start_gen} ──► {end_gen} ({num_generations} total gens)")
    print(f"• Population: {population_size} agents  |  World: {grid_size}x{grid_size}  |  Food items: {num_food}")
    print(f"• Inner Loop: PyTorch Policy Gradient (REINFORCE with weight decay)")
    print(f"• Outer Loop: Darwinian Permadeath Selection (food*10 - hazard*100)")
    print("• Live Telemetry: Streaming to second terminal via live_state.json")
    print("=" * 78)

    total_start_time = time.perf_counter()
    total_steps_simulated = 0

    print(f"{'GEN':>5} | {'AVG FOOD':>8} {'PROGRESS':^16} | {'BEST (F/H)':>12} | {'SURVIVAL':>9} | {'THROUGHPUT':>10} | {'TOP LINEAGE'}")
    print("-" * 84)

    for gen in range(start_gen, end_gen + 1):
        gen_start_time = time.perf_counter()
        env.reset_world()

        for agent in population:
            agent.fitness = 0.0
            agent.food_eaten = 0
            agent.hazards_hit = 0

        alive = {agent.id: True for agent in population}

        # Fair spawn: never spawn directly on hazard
        free_tiles = [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) not in env.hazards]
        positions = {
            agent.id: random.choice(free_tiles)
            for agent in population
        }

        prev_positions = {agent.id: None for agent in population}

        # --- SIMULATION OF ONE GENERATION (WITH PERMADEATH) ---
        for tick in range(ticks_per_gen):
            any_alive = False
            for agent in population:
                if not alive[agent.id]:
                    continue
                any_alive = True
                pos = positions[agent.id]
                prev_pos = prev_positions[agent.id]
                obs = env.get_observation(pos, agent_energy=1.0)
                action = agent.select_action(
                    obs,
                    pos=pos,
                    grid_size=grid_size,
                    hazards=env.hazards,
                    prev_pos=prev_pos,
                    temperature=0.5
                )

                new_pos, reward, ate_food, hit_hazard = env.step_agent(pos, action)
                prev_positions[agent.id] = pos
                positions[agent.id] = new_pos
                agent.record_reward(reward)

                if ate_food:
                    agent.food_eaten += 1
                if hit_hazard:
                    agent.hazards_hit += 1
                    alive[agent.id] = False  # 💀 PERMADEATH: Instant termination on hazard collision!

            if not any_alive:
                break

        total_steps_simulated += ticks_per_gen * population_size

        # --- INNER LOOP: REINFORCEMENT LEARNING ---
        for agent in population:
            agent.train_rl_lifetime()

        # --- OUTER LOOP: EVOLUTIONARY SELECTION ---
        # Permadeath Fitness:
        # Living agents score full food points. Dead agents suffer severe penalty (-100.0) and are culled.
        for a in population:
            a.fitness = (a.food_eaten * 10.0) - (a.hazards_hit * 100.0)

        population.sort(key=lambda a: a.fitness, reverse=True)

        best_agent = population[0]
        avg_food = sum(a.food_eaten for a in population) / float(population_size)
        survived_count = sum(1 for a in population if a.hazards_hit == 0)

        history_best.append(best_agent.food_eaten)
        history_avg.append(avg_food)

        # Stream state for second terminal
        save_live_telemetry(best_agent, env, positions[best_agent.id], gen, avg_food)

        # Selection & Mutation
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
        bar = render_bar(avg_food, max_val=12.0)

        # Print progress with food, hazards, and survival count
        best_stats = f"{best_agent.food_eaten}f/{best_agent.hazards_hit}h"
        surv_stats = f"{survived_count}/{population_size} live"
        print(
            f"{gen:5d} | "
            f"{avg_food:7.1f} {bar} | "
            f"{best_stats:>12} | "
            f"{surv_stats:>9} | "
            f"{gen_throughput:8d} st/s | "
            f"{best_agent.lineage}"
        )

        # Periodic checkpoint save every 25 generations
        if gen % 25 == 0:
            torch.save(best_agent.model.state_dict(), checkpoints_dir / "champion.pt")

    total_time = time.perf_counter() - total_start_time
    overall_throughput = int(total_steps_simulated / max(total_time, 1e-5))

    # Save Final Checkpoints
    champion = population[0]
    torch.save(champion.model.state_dict(), checkpoints_dir / "champion.pt")

    pop_checkpoint = {
        "generation": end_gen,
        "agents": [
            {
                "id": a.id,
                "lineage": a.lineage,
                "generation": a.generation,
                "food_eaten": a.food_eaten,
                "hazards_hit": a.hazards_hit,
                "fitness": round(a.fitness, 2),
                "weights": a.model.state_dict()
            }
            for a in population
        ]
    }
    torch.save(pop_checkpoint, checkpoints_dir / "population_latest.pt")

    history_data = {
        "total_generations": end_gen,
        "total_steps": len(history_avg) * ticks_per_gen * population_size,
        "execution_time_sec": round(total_time, 2),
        "dominant_lineage": champion.lineage,
        "peak_score": max(history_best),
        "history_best": history_best,
        "history_avg": [round(x, 2) for x in history_avg]
    }
    with open(checkpoints_dir / "history.json", "w") as f:
        json.dump(history_data, f, indent=2)

    print("=" * 78)
    print("📊 500-GENERATION BENCHMARK COMPLETE")
    print("-" * 78)
    init_avg = history_avg[0]
    final_avg = history_avg[-1]
    improvement = ((final_avg - init_avg) / max(init_avg, 1e-5)) * 100.0

    print(f"• Completed Up To:        Generation {end_gen}")
    print(f"• Total Environment Steps: {total_steps_simulated:,}")
    print(f"• Execution Time:         {total_time:.2f} seconds ({overall_throughput:,} steps/sec)")
    print(f"• Starting Avg:           {init_avg:.1f} food")
    print(f"• Final Gen Avg:          {final_avg:.1f} food  ({improvement:+.1f}% overall)")
    print(f"• Peak Champion Score:    {max(history_best)} food pellets in 150 ticks")
    print(f"• Dominant Lineage:       {champion.lineage}")

    print("\n💾 SAVED LEARNING TO DISK:")
    print(f"  • {checkpoints_dir / 'champion.pt'} (Updated 500-Gen Neural Weights)")
    print(f"  • {checkpoints_dir / 'population_latest.pt'} (Gene Pool at Gen {end_gen})")
    print(f"  • {checkpoints_dir / 'history.json'}")
    print("=" * 78 + "\n")

    return population

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid ERL Training")
    parser.add_argument("--generations", "-g", type=int, default=500, help="Number of generations to train (default: 500)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh from Gen 1 instead of resuming")
    args = parser.parse_args()

    run_hybrid_training(num_generations=args.generations, resume=not args.fresh)
