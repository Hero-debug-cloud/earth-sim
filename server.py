import asyncio
import json
import random
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import torch
import torch.nn.functional as F

from env import Environment
from agent import Agent
from env_phase2 import Phase2Environment
from agent_phase2 import Phase2Agent
from evaluate_hero_v2 import ScarcityPhase2Environment

app = FastAPI(title="Hybrid ERL Living Simulation")

# Global simulation parameters
GRID_SIZE = 12
NUM_FOOD = 8
NUM_HAZARDS = 3
POPULATION_SIZE = 8
TICKS_PER_GEN = 120

# Color palette for lineages
LINEAGE_COLORS = [
    "#38bdf8", "#4ade80", "#f43f5e", "#fbbf24",
    "#a855f7", "#ec4899", "#2dd4bf", "#fb923c"
]

# Initialize world and population
env = Environment(size=GRID_SIZE, num_food=NUM_FOOD, num_hazards=NUM_HAZARDS)
population = [Agent(agent_id=i) for i in range(POPULATION_SIZE)]

ckpt_path = Path("checkpoints/champion.pt")
if ckpt_path.exists():
    try:
        population[0].model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        population[0].lineage = "Champion-G500"
    except Exception as e:
        print(f"Warning: could not load champion checkpoint: {e}")

positions = {a.id: (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)) for a in population}
prev_positions = {}

current_gen = 1
current_tick = 0
is_paused = False
tick_speed = 6.0  # ticks/sec

# Static files setup
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, message: str):
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

def get_champion_telemetry(champion: Agent, pos: tuple[int, int]):
    """Extract neural activations and weights of current #1 agent for visualization."""
    obs = env.get_observation(pos, agent_energy=1.0)
    state_t = torch.tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        h = torch.tanh(champion.model.fc1(state_t))
        logits = champion.model.fc2(h).clone()

        # Wall masking
        ar, ac = pos
        if ar == 0: logits[0] = -1e9
        if ar == GRID_SIZE - 1: logits[1] = -1e9
        if ac == 0: logits[2] = -1e9
        if ac == GRID_SIZE - 1: logits[3] = -1e9

        # Reflex guard
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for act_idx, (dr, dc) in moves.items():
            if (ar + dr, ac + dc) in env.hazards:
                logits[act_idx] -= 30.0

        probs = F.softmax(logits / 0.4, dim=-1)

    action_names = ["Move Up", "Move Down", "Move Left", "Move Right"]
    sensor_names = ["Food Dist X", "Food Dist Y", "Hazard Dist X", "Hazard Dist Y", "Energy"]
    action_idx = int(torch.argmax(probs).item())

    w01 = champion.model.fc1.weight.detach().numpy().tolist() # shape (16, 5)
    w12 = champion.model.fc2.weight.detach().numpy().tolist() # shape (4, 16)

    return {
        "champion_id": champion.id,
        "lineage": champion.lineage,
        "layers": [
            {
                "name": "Sensory Input",
                "nodes": [{"name": sensor_names[i], "val": round(float(obs[i]), 3)} for i in range(len(obs))]
            },
            {
                "name": "Hidden Layer",
                "nodes": [{"name": f"H-{j}", "val": round(float(h[j].item()), 3)} for j in range(len(h))]
            },
            {
                "name": "Motor Actions",
                "nodes": [
                    {
                        "name": action_names[m],
                        "val": round(float(probs[m].item()), 3),
                        "chosen": (m == action_idx)
                    }
                    for m in range(len(probs))
                ]
            }
        ],
        "weights": [w01, w12],
        "chosen_action": action_idx,
        "action_name": action_names[action_idx]
    }

def step_simulation():
    global current_gen, current_tick, population, positions, prev_positions

    current_tick += 1

    # Step all agents
    for agent in population:
        pos = positions[agent.id]
        obs = env.get_observation(pos, agent_energy=1.0)
        action = agent.select_action(
            obs,
            pos=pos,
            grid_size=GRID_SIZE,
            hazards=env.hazards,
            prev_pos=prev_positions.get(agent.id),
            temperature=0.4
        )

        new_pos, reward, ate_food, hit_hazard = env.step_agent(pos, action)
        prev_positions[agent.id] = pos
        positions[agent.id] = new_pos
        agent.record_reward(reward)
        if ate_food:
            agent.food_eaten += 1
        if hit_hazard:
            agent.hazards_hit += 1

    # Check End of Generation
    if current_tick >= TICKS_PER_GEN:
        current_tick = 0
        current_gen += 1

        # 1. Inner Loop: Lifetime Reinforcement Learning
        for agent in population:
            agent.train_rl_lifetime()

        # 2. Outer Loop: Evolutionary Selection
        population.sort(key=lambda a: a.food_eaten, reverse=True)

        num_elites = 2
        elites = population[:num_elites]
        new_offspring = []
        for i, elite in enumerate(elites):
            child_id = population[-1 - i].id
            child = elite.clone_and_mutate(new_id=child_id, mutation_rate=0.15, mutation_scale=0.20)
            new_offspring.append(child)

        population[-num_elites:] = new_offspring

        # Reset for new generation
        env.reset_world()
        for a in population:
            a.food_eaten = 0
            a.fitness = 0.0
            positions[a.id] = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    # Champion agent is current leader
    champion = max(population, key=lambda a: a.food_eaten)
    champ_pos = positions[champion.id]

    # Package state
    state = {
        "type": "state_update",
        "gen": current_gen,
        "tick": current_tick,
        "max_ticks": TICKS_PER_GEN,
        "grid_size": GRID_SIZE,
        "foods": env.foods,
        "hazards": env.hazards,
        "agents": [
            {
                "id": a.id,
                "r": positions[a.id][0],
                "c": positions[a.id][1],
                "food_eaten": a.food_eaten,
                "lineage": a.lineage,
                "color": LINEAGE_COLORS[a.id % len(LINEAGE_COLORS)],
                "is_champion": (a.id == champion.id)
            }
            for a in population
        ],
        "stats": {
            "top_food": champion.food_eaten,
            "avg_food": round(sum(a.food_eaten for a in population) / float(len(population)), 1),
            "top_lineage": champion.lineage
        },
        "brain": get_champion_telemetry(champion, champ_pos),
        "is_paused": is_paused,
        "tick_speed": tick_speed
    }
    return state

async def simulation_loop():
    global is_paused, tick_speed
    while True:
        if not is_paused and len(manager.active_connections) > 0:
            state = step_simulation()
            await manager.broadcast(json.dumps(state))
        delay = 1.0 / max(1.0, min(tick_speed, 30.0))
        await asyncio.sleep(delay)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())
    asyncio.create_task(phase2_simulation_loop())
    asyncio.create_task(scarcity_simulation_loop())

@app.get("/")
async def get_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/neural")
async def get_neural():
    return FileResponse(STATIC_DIR / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_paused, tick_speed, population, positions, current_gen, current_tick
    await manager.connect(websocket)

    # Initial state
    champion = max(population, key=lambda a: a.food_eaten)
    champ_pos = positions[champion.id]
    state = {
        "type": "state_update",
        "gen": current_gen,
        "tick": current_tick,
        "max_ticks": TICKS_PER_GEN,
        "grid_size": GRID_SIZE,
        "foods": env.foods,
        "hazards": env.hazards,
        "agents": [
            {
                "id": a.id,
                "r": positions[a.id][0],
                "c": positions[a.id][1],
                "food_eaten": a.food_eaten,
                "lineage": a.lineage,
                "color": LINEAGE_COLORS[a.id % len(LINEAGE_COLORS)],
                "is_champion": (a.id == champion.id)
            }
            for a in population
        ],
        "stats": {
            "top_food": champion.food_eaten,
            "avg_food": round(sum(a.food_eaten for a in population) / float(len(population)), 1),
            "top_lineage": champion.lineage
        },
        "brain": get_champion_telemetry(champion, champ_pos),
        "is_paused": is_paused,
        "tick_speed": tick_speed
    }
    await websocket.send_text(json.dumps(state))

    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")

            if action == "toggle_pause":
                is_paused = not is_paused
            elif action == "step_once":
                state = step_simulation()
                await manager.broadcast(json.dumps(state))
            elif action == "set_speed":
                tick_speed = float(cmd.get("speed", 6.0))
            elif action == "reset":
                env.reset_world()
                population = [Agent(agent_id=i) for i in range(POPULATION_SIZE)]
                positions = {a.id: (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)) for a in population}
                current_gen = 1
                current_tick = 0

            # Broadcast update
            state = step_simulation() if action == "step_once" else {
                "type": "state_update",
                "gen": current_gen,
                "tick": current_tick,
                "max_ticks": TICKS_PER_GEN,
                "grid_size": GRID_SIZE,
                "foods": env.foods,
                "hazards": env.hazards,
                "agents": [
                    {
                        "id": a.id,
                        "r": positions[a.id][0],
                        "c": positions[a.id][1],
                        "food_eaten": a.food_eaten,
                        "lineage": a.lineage,
                        "color": LINEAGE_COLORS[a.id % len(LINEAGE_COLORS)],
                        "is_champion": (a.id == champion.id)
                    }
                    for a in population
                ],
                "stats": {
                    "top_food": champion.food_eaten,
                    "avg_food": round(sum(a.food_eaten for a in population) / float(len(population)), 1),
                    "top_lineage": champion.lineage
                },
                "brain": get_champion_telemetry(champion, champ_pos),
                "is_paused": is_paused,
                "tick_speed": tick_speed
            }
            await manager.broadcast(json.dumps(state))

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==============================================================================
# PHASE 2: LIVING HERO & HOMEOSTASIS SIMULATION
# ==============================================================================

p2_env = Phase2Environment(size=14, num_food=7, num_trees=7, num_hazards=4, home_pos=(1, 1))
p2_agent = Phase2Agent(
    agent_id=1,
    in_features=6,
    hidden_size=16,
    home_pos=p2_env.home_pos,
    backpack_capacity=5,
    ckpt_path="checkpoints/champion_phase2.pt" if Path("checkpoints/champion_phase2.pt").exists() else "checkpoints/champion_v3_gru.pt"
)
p2_pos = p2_env.home_pos
p2_tick = 0
p2_is_paused = False
p2_tick_speed = 6.0
p2_action_name = "Awakening at Home Base ⛺"
p2_recent_events: list[str] = ["Expedition initialized. Hero awoke at Home Base ⛺"]

class Phase2ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, message: str):
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                pass

p2_manager = Phase2ConnectionManager()

def get_phase2_state() -> dict:
    global p2_env, p2_agent, p2_pos, p2_tick, p2_is_paused, p2_tick_speed, p2_action_name, p2_recent_events
    mem = p2_agent.memory
    day_tick = p2_tick % 100
    day_num = (p2_tick // 100) + 1
    if day_tick < 25:
        tod = f"Day {day_num} · 🌅 Morning"
    elif day_tick < 65:
        tod = f"Day {day_num} · ☀️ Afternoon"
    elif day_tick < 85:
        tod = f"Day {day_num} · 🌆 Dusk"
    else:
        tod = f"Day {day_num} · 🌙 Night"

    return {
        "type": "phase2_update",
        "tick": p2_tick,
        "time_of_day": tod,
        "grid_size": p2_env.size,
        "home_pos": list(p2_env.home_pos),
        "agent_pos": list(p2_pos),
        "target_pos": list(mem.target_pos),
        "action_name": p2_action_name,
        "goal": mem.current_goal,
        "vitals": {
            "health": round(mem.vitals["health"], 1),
            "thirst": round(mem.vitals["thirst"], 1),
            "hunger": round(mem.vitals["hunger"], 1),
            "energy": round(mem.vitals["energy"], 1),
        },
        "backpack": {
            "food": mem.backpack["food"],
            "wood": mem.backpack["wood"]
        },
        "backpack_capacity": mem.backpack_capacity,
        "stockpile": {
            "food": p2_env.stockpile["food"],
            "wood": p2_env.stockpile["wood"]
        },
        "water_tiles": [list(pt) for pt in p2_env.water_tiles],
        "foods": [list(pt) for pt in p2_env.foods],
        "trees": [list(pt) for pt in p2_env.trees],
        "hazards": [list(pt) for pt in p2_env.hazards],
        "events": p2_recent_events[-4:],
        "is_paused": p2_is_paused,
        "tick_speed": p2_tick_speed
    }

def step_phase2_simulation() -> dict:
    global p2_env, p2_agent, p2_pos, p2_tick, p2_action_name, p2_recent_events
    p2_tick += 1
    action_names = ["North ⬆️", "South ⬇️", "West ⬅️", "East ➡️"]

    obs = p2_env.get_observation(
        agent_pos=p2_pos,
        target_pos=p2_agent.memory.target_pos,
        energy=p2_agent.memory.vitals["energy"] / 100.0,
        frustration=p2_agent.memory.compute_frustration()
    )

    if p2_agent.memory.current_goal == "REST" and p2_pos == p2_env.home_pos and p2_agent.memory.vitals["energy"] < 95.0:
        p2_action_name = "Sleeping in Shelter 💤"
        new_pos = p2_pos
        event = None
        hit_hazard = False
    else:
        action = p2_agent.select_action(
            obs=obs,
            pos=p2_pos,
            grid_size=p2_env.size,
            hazards=p2_env.hazards,
            water_tiles=p2_env.water_tiles,
            temperature=0.35
        )
        p2_action_name = action_names[action]
        new_pos, event, hit_hazard = p2_env.step(p2_pos, action)

    events = p2_agent.memory.update_homeostasis(
        current_pos=new_pos,
        env=p2_env,
        step_event=event,
        hit_hazard=hit_hazard
    )
    p2_pos = new_pos
    if events:
        p2_recent_events.extend(events)

    if p2_agent.memory.vitals["health"] <= 0:
        p2_recent_events.append("💀 Hero perished! Re-spawning at Home Base...")
        p2_agent.reset_state()
        p2_pos = p2_env.home_pos
        p2_agent.memory.vitals["health"] = 100.0
        p2_agent.memory.vitals["energy"] = 100.0
        p2_agent.memory.vitals["thirst"] = 0.0
        p2_agent.memory.vitals["hunger"] = 0.0

    return get_phase2_state()

async def phase2_simulation_loop():
    while True:
        if not p2_is_paused and len(p2_manager.active_connections) > 0:
            state = step_phase2_simulation()
            await p2_manager.broadcast(json.dumps(state))
        delay = 1.0 / max(1.0, min(p2_tick_speed, 30.0))
        await asyncio.sleep(delay)

@app.get("/phase2")
async def get_phase2():
    return FileResponse(STATIC_DIR / "phase2.html")

@app.websocket("/ws/phase2")
async def phase2_websocket_endpoint(websocket: WebSocket):
    global p2_is_paused, p2_tick_speed, p2_env, p2_agent, p2_pos, p2_tick, p2_recent_events
    await p2_manager.connect(websocket)

    # Send initial state
    await websocket.send_text(json.dumps(get_phase2_state()))

    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")

            if action == "toggle_pause":
                p2_is_paused = not p2_is_paused
            elif action == "step_once":
                state = step_phase2_simulation()
                await p2_manager.broadcast(json.dumps(state))
                continue
            elif action == "set_speed":
                p2_tick_speed = float(cmd.get("speed", 6.0))
            elif action == "reset":
                p2_env.reset_world()
                p2_agent.reset_state()
                p2_pos = p2_env.home_pos
                p2_tick = 0
                p2_agent.memory.vitals["health"] = 100.0
                p2_agent.memory.vitals["energy"] = 100.0
                p2_agent.memory.vitals["thirst"] = 0.0
                p2_agent.memory.vitals["hunger"] = 0.0
                p2_recent_events = ["Simulation reset. Hero awoke at Home Base ⛺"]

            await p2_manager.broadcast(json.dumps(get_phase2_state()))

    except WebSocketDisconnect:
        p2_manager.disconnect(websocket)


# ==============================================================================
# SCARCITY & FAMINE: 200-HERO COHORT SIMULATION
# ==============================================================================

class ScarcityConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, message: str):
        for conn in list(self.active_connections):
            try:
                await conn.send_text(message)
            except Exception:
                self.disconnect(conn)

scarcity_manager = ScarcityConnectionManager()

scarcity_env: ScarcityPhase2Environment | None = None
scarcity_heroes: list[Phase2Agent] = []
scarcity_positions: dict[int, tuple[int, int]] = {}
scarcity_alive: dict[int, bool] = {}
scarcity_stats: dict[int, dict] = {}
scarcity_tick: int = 0
scarcity_max_days: int = 10
scarcity_ticks_per_day: int = 100
scarcity_num_heroes: int = 200
scarcity_is_paused: bool = False
scarcity_tick_speed: float = 8.0
scarcity_recent_events: list[str] = []
scarcity_latest_frame: dict | None = None
scarcity_external_stream_until: float = 0.0

def init_scarcity_simulation(num_heroes: int = 200, max_days: int = 10, num_bushes: int = 4, num_trees: int = 4):
    global scarcity_env, scarcity_heroes, scarcity_positions, scarcity_alive, scarcity_stats
    global scarcity_tick, scarcity_max_days, scarcity_num_heroes, scarcity_recent_events, scarcity_latest_frame

    scarcity_num_heroes = num_heroes
    scarcity_max_days = max_days
    scarcity_tick = 0
    scarcity_recent_events = [f"Colony initialized: {num_heroes} heroes spawned under severe resource scarcity."]

    scarcity_env = ScarcityPhase2Environment(
        size=14,
        num_bushes=num_bushes,
        num_trees=num_trees,
        num_hazards=4,
        food_respawn_cooldown=35,
        tree_respawn_cooldown=65,
        home_pos=(1, 1),
        water_center=(7, 7)
    )

    ckpt_path = "checkpoints/champion_phase2.pt" if Path("checkpoints/champion_phase2.pt").exists() else "checkpoints/champion_v3_gru.pt"
    scarcity_heroes = []
    scarcity_positions = {}
    scarcity_alive = {}
    scarcity_stats = {}

    for i in range(1, num_heroes + 1):
        agent = Phase2Agent(
            agent_id=i,
            in_features=6,
            hidden_size=16,
            home_pos=scarcity_env.home_pos,
            backpack_capacity=5,
            ckpt_path=ckpt_path
        )
        agent.reset_state()
        scarcity_heroes.append(agent)
        scarcity_positions[i] = scarcity_env.home_pos
        scarcity_alive[i] = True
        scarcity_stats[i] = {
            "days_survived": 0.0,
            "cause_of_death": None,
            "food_deposited": 0,
            "wood_deposited": 0,
            "rations_eaten": 0
        }

    scarcity_latest_frame = get_scarcity_state()
    return scarcity_latest_frame

def get_scarcity_tod_name(tick: int, ticks_per_day: int = 100) -> tuple[int, str]:
    day_num = ((tick - 1) // ticks_per_day) + 1 if tick > 0 else 1
    day_tick = (tick - 1) % ticks_per_day if tick > 0 else 0
    if day_tick < 25:
        return day_num, "Morning 🌅"
    elif day_tick < 65:
        return day_num, "Afternoon ☀️"
    elif day_tick < 85:
        return day_num, "Dusk 🌆"
    else:
        return day_num, "Night 🌙"

def get_scarcity_state() -> dict:
    global scarcity_env, scarcity_heroes, scarcity_positions, scarcity_alive, scarcity_stats
    global scarcity_tick, scarcity_max_days, scarcity_num_heroes, scarcity_recent_events
    global scarcity_is_paused, scarcity_tick_speed

    if scarcity_env is None:
        init_scarcity_simulation(scarcity_num_heroes, scarcity_max_days)

    day_num, tod_name = get_scarcity_tod_name(scarcity_tick, scarcity_ticks_per_day)
    alive_count = sum(1 for a in scarcity_alive.values() if a)

    alive_agents = [a for a in scarcity_heroes if scarcity_alive[a.id]]
    if alive_agents:
        avg_health = round(sum(a.memory.vitals["health"] for a in alive_agents) / len(alive_agents), 1)
        avg_hunger = round(sum(a.memory.vitals["hunger"] for a in alive_agents) / len(alive_agents), 1)
        avg_thirst = round(sum(a.memory.vitals["thirst"] for a in alive_agents) / len(alive_agents), 1)
        avg_energy = round(sum(a.memory.vitals["energy"] for a in alive_agents) / len(alive_agents), 1)
    else:
        avg_health = avg_hunger = avg_thirst = avg_energy = 0.0

    starvation_deaths = sum(1 for s in scarcity_stats.values() if s.get("cause_of_death") == "Starvation")
    dehydration_deaths = sum(1 for s in scarcity_stats.values() if s.get("cause_of_death") == "Dehydration")
    hazard_deaths = sum(1 for s in scarcity_stats.values() if s.get("cause_of_death") == "Hazard Trap")

    heroes_payload = []
    for agent in scarcity_heroes:
        hid = agent.id
        pos = scarcity_positions[hid]
        alive = scarcity_alive[hid]
        heroes_payload.append({
            "id": hid,
            "r": pos[0],
            "c": pos[1],
            "alive": alive,
            "vitals": {
                "health": round(agent.memory.vitals["health"], 1),
                "hunger": round(agent.memory.vitals["hunger"], 1),
                "thirst": round(agent.memory.vitals["thirst"], 1),
                "energy": round(agent.memory.vitals["energy"], 1),
            },
            "backpack": {
                "food": agent.memory.backpack["food"],
                "wood": agent.memory.backpack["wood"]
            },
            "goal": agent.memory.current_goal,
            "days_survived": scarcity_stats[hid]["days_survived"],
            "cause_of_death": scarcity_stats[hid]["cause_of_death"],
            "food_dep": scarcity_stats[hid]["food_deposited"],
            "wood_dep": scarcity_stats[hid]["wood_deposited"],
            "rations": scarcity_stats[hid]["rations_eaten"]
        })

    bushes_payload = [
        {
            "pos": list(b["pos"]),
            "is_ripe": b["is_ripe"],
            "timer": b["timer"],
            "max_timer": scarcity_env.food_cooldown
        }
        for b in scarcity_env.bushes
    ]
    trees_payload = [
        {
            "pos": list(t["pos"]),
            "is_mature": t["is_mature"],
            "timer": t["timer"],
            "max_timer": scarcity_env.tree_cooldown
        }
        for t in scarcity_env.tree_entries
    ]

    return {
        "type": "scarcity_update",
        "tick": scarcity_tick,
        "day_num": day_num,
        "max_days": scarcity_max_days,
        "tod_name": tod_name,
        "grid_size": scarcity_env.size,
        "home_pos": list(scarcity_env.home_pos),
        "water_tiles": [list(pt) for pt in scarcity_env.water_tiles],
        "hazards": [list(pt) for pt in scarcity_env.hazards],
        "bushes": bushes_payload,
        "trees": trees_payload,
        "stockpile": {
            "food": scarcity_env.stockpile["food"],
            "wood": scarcity_env.stockpile["wood"]
        },
        "emergency_rations": scarcity_env.emergency_rations_eaten,
        "total_heroes": scarcity_num_heroes,
        "alive_count": alive_count,
        "heroes": heroes_payload,
        "stats": {
            "survival_rate": round((alive_count / max(1, scarcity_num_heroes)) * 100, 1),
            "avg_health": avg_health,
            "avg_hunger": avg_hunger,
            "avg_thirst": avg_thirst,
            "avg_energy": avg_energy,
            "starvation_deaths": starvation_deaths,
            "dehydration_deaths": dehydration_deaths,
            "hazard_deaths": hazard_deaths
        },
        "recent_events": scarcity_recent_events[-6:],
        "is_paused": scarcity_is_paused,
        "tick_speed": scarcity_tick_speed
    }

def step_scarcity_simulation() -> dict:
    global scarcity_env, scarcity_heroes, scarcity_positions, scarcity_alive, scarcity_stats
    global scarcity_tick, scarcity_max_days, scarcity_num_heroes, scarcity_recent_events, scarcity_latest_frame

    if scarcity_env is None:
        init_scarcity_simulation(scarcity_num_heroes, scarcity_max_days)

    scarcity_tick += 1
    scarcity_env.tick_ecology()

    for agent in scarcity_heroes:
        hid = agent.id
        if not scarcity_alive[hid]:
            continue

        pos = scarcity_positions[hid]
        ar, ac = pos

        if agent.memory.current_goal in ("RETURN_HOME", "REST", "SURVIVE"):
            agent.memory.target_pos = scarcity_env.home_pos
        elif agent.memory.current_goal == "HYDRATE":
            water_dists = [abs(wr - ar) + abs(wc - ac) for wr, wc in scarcity_env.water_tiles]
            agent.memory.target_pos = scarcity_env.water_tiles[int(np.argmin(water_dists))]
        else:
            available_targets = scarcity_env.ripe_food_positions + scarcity_env.mature_tree_positions
            if available_targets:
                dists = [abs(tr - ar) + abs(tc - ac) for tr, tc in available_targets]
                agent.memory.target_pos = available_targets[int(np.argmin(dists))]
            else:
                agent.memory.target_pos = scarcity_env.home_pos

        obs = scarcity_env.get_observation(
            agent_pos=pos,
            target_pos=agent.memory.target_pos,
            energy=agent.memory.vitals["energy"] / 100.0,
            frustration=agent.memory.compute_frustration()
        )

        if agent.memory.current_goal == "REST" and pos == scarcity_env.home_pos and agent.memory.vitals["energy"] < 95.0:
            new_pos = pos
            event = None
            hit_hazard = False
        else:
            action = agent.select_action(
                obs=obs,
                pos=pos,
                grid_size=scarcity_env.size,
                hazards=scarcity_env.hazards,
                water_tiles=scarcity_env.water_tiles,
                temperature=0.35,
                deterministic=False
            )
            new_pos, event, hit_hazard = scarcity_env.step(pos, action)

        old_stock_f = scarcity_env.stockpile["food"]
        old_stock_w = scarcity_env.stockpile["wood"]

        events = agent.memory.update_homeostasis(
            current_pos=new_pos,
            env=scarcity_env,
            step_event=event,
            hit_hazard=hit_hazard
        )
        scarcity_positions[hid] = new_pos

        if scarcity_env.is_home(new_pos) and agent.memory.vitals["hunger"] >= 70.0 and agent.memory.backpack["food"] == 0:
            if scarcity_env.stockpile["food"] > 0:
                scarcity_env.stockpile["food"] -= 1
                scarcity_env.emergency_rations_eaten += 1
                scarcity_stats[hid]["rations_eaten"] += 1
                agent.memory.vitals["hunger"] = max(0.0, agent.memory.vitals["hunger"] - 50.0)
                events.append(f"🍲 H{hid:02d} ate emergency ration from Granary")

        dep_f = scarcity_env.stockpile["food"] - old_stock_f
        dep_w = scarcity_env.stockpile["wood"] - old_stock_w
        if dep_f > 0 or dep_w > 0:
            scarcity_stats[hid]["food_deposited"] += max(0, dep_f)
            scarcity_stats[hid]["wood_deposited"] += max(0, dep_w)

        if events:
            scarcity_recent_events.extend([f"[H{hid:02d}] {ev}" for ev in events])

        if agent.memory.vitals["health"] <= 0.0:
            scarcity_alive[hid] = False
            scarcity_stats[hid]["days_survived"] = round(scarcity_tick / float(scarcity_ticks_per_day), 2)
            if agent.memory.vitals["hunger"] >= 88.0:
                scarcity_stats[hid]["cause_of_death"] = "Starvation"
            elif agent.memory.vitals["thirst"] >= 88.0:
                scarcity_stats[hid]["cause_of_death"] = "Dehydration"
            elif hit_hazard:
                scarcity_stats[hid]["cause_of_death"] = "Hazard Trap"
            else:
                scarcity_stats[hid]["cause_of_death"] = "Exhaustion"
            scarcity_recent_events.append(f"💀 Hero H{hid:02d} perished ({scarcity_stats[hid]['cause_of_death']})")

    for hid in scarcity_stats:
        if scarcity_alive[hid]:
            scarcity_stats[hid]["days_survived"] = round(scarcity_tick / float(scarcity_ticks_per_day), 2)

    scarcity_latest_frame = get_scarcity_state()
    return scarcity_latest_frame

async def scarcity_simulation_loop():
    global scarcity_is_paused, scarcity_tick_speed, scarcity_external_stream_until, scarcity_latest_frame
    while True:
        now = time.time()
        if now > scarcity_external_stream_until:
            if not scarcity_is_paused and len(scarcity_manager.active_connections) > 0:
                state = step_scarcity_simulation()
                await scarcity_manager.broadcast(json.dumps(state))
        delay = 1.0 / max(1.0, min(scarcity_tick_speed, 40.0))
        await asyncio.sleep(delay)

@app.get("/scarcity")
async def get_scarcity():
    return FileResponse(STATIC_DIR / "scarcity.html")

@app.get("/famine")
async def get_famine():
    return FileResponse(STATIC_DIR / "scarcity.html")

@app.get("/api/scarcity/health")
async def scarcity_health():
    return {"status": "ok", "mode": "scarcity", "heroes": scarcity_num_heroes}

@app.post("/api/scarcity/frame")
async def push_scarcity_frame(request: Request):
    global scarcity_latest_frame, scarcity_external_stream_until
    frame_data = await request.json()
    scarcity_latest_frame = frame_data
    scarcity_external_stream_until = time.time() + 3.0
    await scarcity_manager.broadcast(json.dumps(frame_data))
    return {"status": "accepted"}

@app.websocket("/ws/scarcity")
async def scarcity_websocket_endpoint(websocket: WebSocket):
    global scarcity_is_paused, scarcity_tick_speed, scarcity_latest_frame
    await scarcity_manager.connect(websocket)

    if scarcity_latest_frame is None:
        scarcity_latest_frame = get_scarcity_state()
    await websocket.send_text(json.dumps(scarcity_latest_frame))

    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")

            if action == "toggle_pause":
                scarcity_is_paused = not scarcity_is_paused
            elif action == "step_once":
                state = step_scarcity_simulation()
                await scarcity_manager.broadcast(json.dumps(state))
                continue
            elif action == "set_speed":
                scarcity_tick_speed = float(cmd.get("speed", 8.0))
            elif action == "reset":
                nh = int(cmd.get("heroes", 200))
                md = int(cmd.get("max_days", 10))
                init_scarcity_simulation(num_heroes=nh, max_days=md)

            await scarcity_manager.broadcast(json.dumps(get_scarcity_state()))
    except WebSocketDisconnect:
        scarcity_manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)



