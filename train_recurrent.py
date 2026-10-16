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
from agent_recurrent import RecurrentAgent, VillageMemory

def render_bar(val: float, max_val: float = 12.0, width: int = 14) -> str:
    filled = int(min(1.0, max(0.0, val / max_val)) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def save_recurrent_telemetry(champion: RecurrentAgent, env: Environment, pos: tuple[int, int], gen: int, avg_food: float):
    obs = env.get_observation(pos, agent_energy=1.0)
    ctx_obs = champion.memory.get_context_vector(
        pos=pos,
        target_food_dx=obs[0],
        target_food_dy=obs[1],
        hazard_dx=obs[2],
        hazard_dy=obs[3],
        energy=obs[4]
    )
    state_t = torch.tensor(ctx_obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, new_h = champion.model(state_t, champion.hx)
        probs = F.softmax(logits.squeeze(0), dim=-1)

    action_idx = int(torch.argmax(probs).item())
    action_names = ["Move Up", "Move Down", "Move Left", "Move Right"]
    sensor_names = ["Food dx", "Food dy", "Hazard dx", "Hazard dy", "Energy", "Frustration"]

    telemetry = {
        "gen": gen,
        "lineage": champion.lineage,
        "type": "Recurrent (GRU)",
        "best_food": champion.food_eaten,
        "best_hazards": getattr(champion, "hazards_hit", 0),
        "best_fitness": round(champion.fitness, 1),
        "avg_food": round(avg_food, 1),
        "sensors": [{"name": sensor_names[i], "val": round(float(ctx_obs[i]), 3)} for i in range(len(ctx_obs))],
        "hidden_gru": [round(float(new_h[0, j].item()), 3) for j in range(new_h.shape[1])],
        "actions": [{"name": action_names[m], "prob": round(float(probs[m].item()), 3)} for m in range(len(probs))],
        "chosen_action": action_names[action_idx]
    }

    try:
        with open("live_state.json", "w") as f:
            json.dump(telemetry, f)
    except Exception:
        pass

def create_curriculum_env(grid_size: int = 12, num_food: int = 8, num_hazards: int = 3) -> Environment:
    """
    Curriculum generator:
    - 60% Standard open boards (random foods & hazards)
    - 20% Cul-de-sac / U-trap boards (hazard pockets requiring backing out)
    - 20% Fortress / Moat boards (surrounded center food with single gate)
    """
    env = Environment(size=grid_size, num_food=num_food, num_hazards=num_hazards)
    roll = random.random()

    if roll < 0.20:
        # 1. Cul-de-sac U-trap
        tr, tc = random.randint(3, 8), random.randint(3, 8)
        env.foods[0] = (tr, tc)
        # Hazard walls on North, West, East
        env.hazards = [(tr - 1, tc), (tr, tc - 1), (tr, tc + 1)]
    elif roll < 0.40:
        # 2. Fortress Gate board
        cr, cc = grid_size // 2, grid_size // 2
        env.foods[0] = (cr, cc)
        gate_dir = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        env.hazards = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) != (0, 0) and (dr, dc) != gate_dir:
                    env.hazards.append((cr + dr, cc + dc))

    return env

def run_recurrent_training(
    num_generations: int = 500,
    population_size: int = 8,
    ticks_per_gen: int = 140,
    grid_size: int = 12,
    num_food: int = 8,
    num_hazards: int = 3,
    resume: bool = True
):
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    pop_ckpt_path = checkpoints_dir / "population_recurrent.pt"
    history_path = checkpoints_dir / "history_recurrent.json"
    champion_path = checkpoints_dir / "champion_v3_gru.pt"

    population: list[RecurrentAgent] = []
    start_gen = 1
    history_best = []
    history_avg = []

    if resume and pop_ckpt_path.exists():
        try:
            ckpt = torch.load(pop_ckpt_path, weights_only=True)
            saved_agents = ckpt.get("agents", [])
            start_gen = ckpt.get("generation", 0) + 1

            for sa in saved_agents[:population_size]:
                agent = RecurrentAgent(agent_id=sa["id"])
                agent.model.load_state_dict(sa["weights"])
                agent.lineage = sa.get("lineage", f"RA{sa['id']}")
                agent.generation = sa.get("generation", 1)
                population.append(agent)

            while len(population) < population_size:
                population.append(RecurrentAgent(agent_id=len(population)))

            if history_path.exists():
                with open(history_path, "r") as f:
                    h = json.load(f)
                    history_best = h.get("history_best", [])
                    history_avg = h.get("history_avg", [])

            print(f"\n🔄 Resuming Recurrent Evolution from Generation {start_gen}")
        except Exception as e:
            print(f"Notice: starting fresh recurrent population ({e})")
            population = [RecurrentAgent(agent_id=i) for i in range(population_size)]
    else:
        population = [RecurrentAgent(agent_id=i) for i in range(population_size)]

    end_gen = start_gen + num_generations - 1

    print("\n" + "=" * 80)
    print("🧠 PHASE 1: HYBRID ERL RECURRENT MEMORY (GRU) EVOLUTIONARY TRAINING")
    print("=" * 80)
    print(f"• Population Size   : {population_size} Recurrent Agents (GRU Memory)")
    print(f"• Generations       : {num_generations} (Gen {start_gen} -> {end_gen})")
    print(f"• Ticks/Generation  : {ticks_per_gen} steps (BPTT Lifetime Learning)")
    print(f"• Curriculum Layout : 60% Open Arena | 20% Cul-de-Sac Traps | 20% Fortress Gates")
    print(f"• Permadeath Policy : Hazard Hit = Instant Mortality + Culling (-100 penalty)")
    print(f"• Loop Frustration  : Dynamic Reversal Penalty & Orthogonal Flank Breaker")
    print("=" * 80)

    total_start_time = time.perf_counter()
    total_steps_simulated = 0

    print(f"{'GEN':>5} | {'AVG FOOD':>8} {'PROGRESS':^16} | {'BEST (F/H)':>12} | {'SURVIVAL':>9} | {'THROUGHPUT':>10} | {'TOP LINEAGE'}")
    print("-" * 84)

    for gen in range(start_gen, end_gen + 1):
        gen_start_time = time.perf_counter()

        # Create curriculum board for this generation
        env = create_curriculum_env(grid_size=grid_size, num_food=num_food, num_hazards=num_hazards)

        for agent in population:
            agent.reset_memory()
            agent.fitness = 0.0
            agent.food_eaten = 0
            agent.hazards_hit = 0

        alive = {agent.id: True for agent in population}

        # Spawn safely away from hazards
        free_tiles = [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) not in env.hazards]
        positions = {agent.id: random.choice(free_tiles) for agent in population}

        # --- SIMULATION OF ONE GENERATION ---
        for tick in range(ticks_per_gen):
            any_alive = False
            for agent in population:
                if not alive[agent.id]:
                    continue
                any_alive = True
                pos = positions[agent.id]

                obs = env.get_observation(pos, agent_energy=1.0)
                ctx_obs = agent.memory.get_context_vector(
                    pos=pos,
                    target_food_dx=obs[0],
                    target_food_dy=obs[1],
                    hazard_dx=obs[2],
                    hazard_dy=obs[3],
                    energy=obs[4]
                )

                action = agent.select_action(
                    ctx_obs,
                    pos=pos,
                    grid_size=grid_size,
                    hazards=env.hazards,
                    temperature=0.4
                )

                new_pos, reward, ate_food, hit_hazard = env.step_agent(pos, action)
                agent.memory.record_step(new_pos, ate_food=ate_food, hit_hazard=hit_hazard)
                positions[agent.id] = new_pos

                # Stagnation penalty for persistent looping (>0.6 frustration)
                frustration = agent.memory.compute_frustration()
                if frustration >= 0.6:
                    reward -= 0.3 * frustration

                agent.record_reward(reward)

                if ate_food:
                    agent.food_eaten += 1
                if hit_hazard:
                    agent.hazards_hit += 1
                    alive[agent.id] = False  # Permadeath

            if not any_alive:
                break

        total_steps_simulated += ticks_per_gen * population_size

        # --- INNER LOOP: REINFORCEMENT LEARNING (BPTT) ---
        for agent in population:
            agent.train_rl_lifetime()

        # --- OUTER LOOP: EVOLUTIONARY SELECTION ---
        for a in population:
            a.fitness = (a.food_eaten * 10.0) - (a.hazards_hit * 100.0)

        population.sort(key=lambda a: a.fitness, reverse=True)

        best_agent = population[0]
        avg_food = sum(a.food_eaten for a in population) / float(population_size)
        survived_count = sum(1 for a in population if a.hazards_hit == 0)

        history_best.append(best_agent.food_eaten)
        history_avg.append(avg_food)

        # Save telemetry
        save_recurrent_telemetry(best_agent, env, positions[best_agent.id], gen, avg_food)

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

        lineage_str = best_agent.lineage
        if len(lineage_str) > 28:
            lineage_str = lineage_str[:12] + "..." + lineage_str[-13:]

        # Log progress every generation
        print(f"G{gen:03d}  | {avg_food:5.1f} {bar} | {best_agent.food_eaten:2d}F / {best_agent.hazards_hit:1d}H | {survived_count:2d}/{population_size} live | {gen_throughput:5d} step/s | {lineage_str}")

        # Checkpoint every 25 generations or final
        if gen % 25 == 0 or gen == end_gen:
            # 1. Save Recurrent Champion model
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
                "history_best": history_best,
                "history_avg": history_avg
            }
            with open(history_path, "w") as f:
                json.dump(hist_data, f, indent=2)

    total_time = time.perf_counter() - total_start_time
    print("-" * 84)
    print(f"✅ Recurrent Training Complete: {num_generations} Generations in {total_time:.2f}s ({total_steps_simulated / total_time:.0f} steps/sec)")
    print(f"💾 Champion Saved to: {champion_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Recurrent GRU Agent with Hybrid ERL")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to train")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh (do not resume)")
    parser.add_argument("--pop", type=int, default=8, help="Population size")
    parser.add_argument("--ticks", type=int, default=140, help="Ticks per generation")
    args = parser.parse_args()

    run_recurrent_training(
        num_generations=args.generations,
        population_size=args.pop,
        ticks_per_gen=args.ticks,
        resume=not args.fresh
    )
