#!/usr/bin/env python3
import copy
import time
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from world import CivilizationWorld
from agent import CivilizationAgent

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
CHAMPION_PATH = CKPT_DIR / "champion.pt"
POPULATION_PATH = CKPT_DIR / "civilization_pool.pt"
HISTORY_PATH = CKPT_DIR / "history.json"


def evaluate_agent(
    agent: CivilizationAgent,
    world: CivilizationWorld,
    ticks_per_gen: int = 150
) -> float:
    """Evaluates an agent over an expedition lifetime in the dynamic world."""
    agent.reset_state()
    agent.food_deposited = 0
    agent.wood_deposited = 0
    pos = world.home_pos

    for _ in range(ticks_per_gen):
        # Ecological progression & hazard motion
        world.tick_ecology()

        ar, ac = pos
        # Determine target based on cognitive drive
        if agent.memory.current_goal in ("RETURN_HOME", "REST", "SURVIVE"):
            target_pos = world.home_pos
        elif agent.memory.current_goal == "HYDRATE":
            water_dists = [abs(wr - ar) + abs(wc - ac) for wr, wc in world.water_tiles]
            target_pos = world.water_tiles[int(np.argmin(water_dists))]
        else:
            available = world.ripe_food_positions + world.mature_tree_positions
            if available:
                dists = [abs(tr - ar) + abs(tc - ac) for tr, tc in available]
                target_pos = available[int(np.argmin(dists))]
            else:
                target_pos = world.home_pos

        agent.memory.target_pos = target_pos
        obs = world.get_observation(
            agent_pos=pos,
            target_pos=target_pos,
            energy=agent.memory.vitals["energy"] / 100.0,
            frustration=agent.memory.compute_frustration()
        )

        action = agent.select_action(
            obs=obs,
            pos=pos,
            grid_size=world.size,
            hazards=world.hazards,
            water_tiles=world.water_tiles,
            temperature=0.35
        )

        old_stock_f = world.stockpile["food"]
        old_stock_w = world.stockpile["wood"]

        new_pos, event, hit_hazard = world.step_agent(pos, action)
        events = agent.memory.update_homeostasis(
            current_pos=new_pos,
            world=world,
            step_event=event,
            hit_hazard=hit_hazard
        )
        pos = new_pos

        dep_f = world.stockpile["food"] - old_stock_f
        dep_w = world.stockpile["wood"] - old_stock_w
        if dep_f > 0 or dep_w > 0:
            agent.food_deposited += max(0, dep_f)
            agent.wood_deposited += max(0, dep_w)

        # Immediate step reward
        reward = 0.0
        if dep_f > 0 or dep_w > 0:
            reward += 15.0 * (dep_f + dep_w)
        if event == "HARVEST_FOOD":
            reward += 3.0
        elif event == "HARVEST_WOOD":
            reward += 2.5
        if hit_hazard:
            reward -= 10.0
        agent.record_reward(reward)

        if agent.memory.vitals["health"] <= 0.0:
            break

    # Inner-loop lifetime learning
    agent.train_rl_lifetime()

    # Fitness composite: Survival days + Resources deposited + Health maintained
    total_deposited = agent.food_deposited + agent.wood_deposited
    health_bonus = agent.memory.vitals["health"] * 0.1
    fitness = (total_deposited * 25.0) + health_bonus
    agent.fitness = float(fitness)
    return fitness


def train_civilization(
    generations: int = 50,
    population_size: int = 16,
    ticks_per_gen: int = 150,
    num_hazards: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """Runs evolutionary training loop and hot-updates the champion checkpoint."""
    world = CivilizationWorld(size=14, num_bushes=5, num_trees=5, num_hazards=num_hazards)

    # Initialize population
    population: List[CivilizationAgent] = []
    base_ckpt = CHAMPION_PATH if CHAMPION_PATH.exists() else None

    for i in range(population_size):
        agent = CivilizationAgent(
            agent_id=i + 1,
            in_features=6,
            hidden_size=16,
            home_pos=world.home_pos,
            backpack_capacity=5,
            ckpt_path=str(base_ckpt) if base_ckpt else None
        )
        population.append(agent)

    if verbose:
        print(f"\n🌱 Initiating Evolutionary Training: {generations} Generations | Population: {population_size} | World Size: 14x14")
        print(f"   Moving Hazards: {num_hazards} | Base Checkpoint: {base_ckpt or 'Random Initialization'}\n")

    history_best: List[float] = []
    history_avg: List[float] = []

    for gen in range(1, generations + 1):
        world.reset_world()

        # Evaluate each agent in independent world episodes
        scores = []
        for agent in population:
            score = evaluate_agent(agent, world, ticks_per_gen)
            scores.append(score)

        # Sort by fitness
        population.sort(key=lambda a: a.fitness, reverse=True)
        best_agent = population[0]
        avg_score = float(np.mean(scores))

        history_best.append(round(best_agent.fitness, 2))
        history_avg.append(round(avg_score, 2))

        if verbose and (gen % 5 == 0 or gen == 1 or gen == generations):
            print(f"Gen {gen:3d}/{generations} | Best Fitness: {best_agent.fitness:6.1f} | Avg: {avg_score:6.1f} | Top Deposits: {best_agent.food_deposited + best_agent.wood_deposited:2d} | Lineage: {best_agent.lineage[:30]}")

        # Elite Selection & Mutation
        elites = population[:4]
        new_pop: List[CivilizationAgent] = []
        for idx, e in enumerate(elites, 1):
            clone_e = CivilizationAgent(
                agent_id=idx,
                in_features=e.in_features,
                hidden_size=e.hidden_size,
                home_pos=e.memory.home_pos,
                backpack_capacity=e.memory.backpack_capacity
            )
            clone_e.model.load_state_dict({k: v.clone() for k, v in e.model.state_dict().items()})
            clone_e.lineage = e.lineage
            new_pop.append(clone_e)

        while len(new_pop) < population_size:
            parent = random.choice(elites)
            child = parent.clone_and_mutate(new_id=len(new_pop) + 1, mutation_rate=0.15, mutation_scale=0.18)
            new_pop.append(child)

        population = new_pop

    # Save champion checkpoint
    torch.save(population[0].model.state_dict(), CHAMPION_PATH)
    torch.save([a.model.state_dict() for a in population], POPULATION_PATH)

    # Save history
    hist_data = {
        "generations_trained": generations,
        "best_fitness": history_best[-1] if history_best else 0.0,
        "history_best": history_best,
        "history_avg": history_avg,
        "updated_at": time.time()
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist_data, f, indent=2)

    if verbose:
        print(f"\n🏆 Training Complete! Checkpoint saved to: {CHAMPION_PATH}\n")

    return hist_data


def main():
    parser = argparse.ArgumentParser(description="Train Civilization Champion Manually")
    parser.add_argument("--generations", type=int, default=30, help="Number of evolutionary generations")
    parser.add_argument("--population", type=int, default=16, help="Population pool size")
    parser.add_argument("--ticks", type=int, default=150, help="Ticks per generation episode")
    parser.add_argument("--hazards", type=int, default=3, help="Number of moving hazard beasts")
    args = parser.parse_args()

    train_civilization(
        generations=args.generations,
        population_size=args.population,
        ticks_per_gen=args.ticks,
        num_hazards=args.hazards,
        verbose=True
    )


if __name__ == "__main__":
    main()
