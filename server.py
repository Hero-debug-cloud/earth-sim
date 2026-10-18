import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from world import CivilizationWorld
from agent import CivilizationAgent
from train import train_civilization, CHAMPION_PATH

app = FastAPI(title="Civilization Platform")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ==============================================================================
# PERSISTENT SIMULATION STATE
# ==============================================================================

GRID_SIZE = 14
NUM_HEROES = 8
world = CivilizationWorld(size=GRID_SIZE, num_bushes=5, num_trees=5, num_hazards=3)

heroes: List[CivilizationAgent] = []
hero_positions: Dict[int, Tuple[int, int]] = {}
for i in range(1, NUM_HEROES + 1):
    agent = CivilizationAgent(
        agent_id=i,
        in_features=6,
        hidden_size=16,
        home_pos=world.home_pos,
        backpack_capacity=5,
        ckpt_path=str(CHAMPION_PATH) if CHAMPION_PATH.exists() else None
    )
    heroes.append(agent)
    hero_positions[i] = world.home_pos

current_tick: int = 0
current_generation: int = 1
is_paused: bool = False
tick_speed: float = 6.0  # ticks/sec
selected_hero_id: int = 1
last_ckpt_mtime: float = CHAMPION_PATH.stat().st_mtime if CHAMPION_PATH.exists() else 0.0
is_evolving: bool = False
recent_events: List[str] = ["Civilization founded at Home Base ⛺"]


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
        for conn in list(self.active_connections):
            try:
                await conn.send_text(message)
            except Exception:
                self.disconnect(conn)

manager = ConnectionManager()


def check_and_reload_checkpoint():
    """Hot-reloads neural weights if train.py created or updated champion.pt."""
    global last_ckpt_mtime, heroes
    if CHAMPION_PATH.exists():
        mtime = CHAMPION_PATH.stat().st_mtime
        if mtime > last_ckpt_mtime:
            last_ckpt_mtime = mtime
            try:
                state_dict = torch.load(CHAMPION_PATH, weights_only=True)
                for h in heroes:
                    h.model.load_state_dict(state_dict)
                recent_events.append("🔄 Hot-reloaded newly trained champion weights!")
            except Exception as e:
                pass


def get_hero_brain_telemetry(hero: CivilizationAgent, pos: Tuple[int, int]) -> Dict[str, Any]:
    """Extracts neural sensory inputs, GRU hidden state activations, and action distribution."""
    ar, ac = pos
    target_pos = hero.memory.target_pos
    energy = hero.memory.vitals["energy"] / 100.0
    frustration = hero.memory.compute_frustration()

    obs = world.get_observation(pos, target_pos, energy, frustration)
    state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        next_h = hero.model.gru(state_t, hero.hx)
        logits = hero.model.fc_out(next_h).squeeze(0)

        # Apply boundary mask
        if ar == 0: logits[0] = -1e9
        if ar == GRID_SIZE - 1: logits[1] = -1e9
        if ac == 0: logits[2] = -1e9
        if ac == GRID_SIZE - 1: logits[3] = -1e9

        # Reflex guard
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        blocked = set(world.hazards)
        if hero.memory.current_goal != "HYDRATE":
            blocked.update(world.water_tiles)
        for act_idx, (dr, dc) in moves.items():
            if (ar + dr, ac + dc) in blocked:
                logits[act_idx] -= 35.0

        probs = F.softmax(logits / 0.35, dim=-1).numpy().tolist()
        hidden_vals = next_h.squeeze(0).numpy().tolist()

    action_names = ["North ⬆️", "South ⬇️", "West ⬅️", "East ➡️"]
    sensor_names = ["Target DX", "Target DY", "Hazard DX", "Hazard DY", "Energy", "Frustration"]
    chosen_action = int(np.argmax(probs))

    # Weight matrices for visual synapses
    w_gru = hero.model.gru.weight_ih.detach().numpy()[:8, :6].tolist()
    w_out = hero.model.fc_out.weight.detach().numpy()[:, :8].tolist()

    return {
        "hero_id": hero.id,
        "lineage": hero.lineage,
        "sensors": [{"name": sensor_names[i], "val": round(float(obs[i]), 3)} for i in range(len(obs))],
        "hidden": [{"name": f"H{j}", "val": round(float(hidden_vals[j]), 3)} for j in range(len(hidden_vals))],
        "actions": [{"name": action_names[m], "val": round(float(probs[m]), 3), "chosen": (m == chosen_action)} for m in range(len(probs))],
        "chosen_action": chosen_action,
        "chosen_name": action_names[chosen_action],
        "weights": [w_gru, w_out]
    }


def get_world_state() -> Dict[str, Any]:
    global current_tick, current_generation, world, heroes, hero_positions
    global is_paused, tick_speed, selected_hero_id, is_evolving, recent_events

    day_num, tod_name = world.get_tod_name()
    selected_hero = next((h for h in heroes if h.id == selected_hero_id), heroes[0])
    selected_pos = hero_positions.get(selected_hero.id, world.home_pos)

    heroes_payload = []
    for h in heroes:
        pos = hero_positions.get(h.id, world.home_pos)
        heroes_payload.append({
            "id": h.id,
            "r": pos[0],
            "c": pos[1],
            "vitals": {
                "health": round(h.memory.vitals["health"], 1),
                "hunger": round(h.memory.vitals["hunger"], 1),
                "thirst": round(h.memory.vitals["thirst"], 1),
                "energy": round(h.memory.vitals["energy"], 1),
            },
            "backpack": {
                "food": h.memory.backpack["food"],
                "wood": h.memory.backpack["wood"]
            },
            "goal": h.memory.current_goal,
            "lineage": h.lineage,
            "is_selected": (h.id == selected_hero.id)
        })

    bushes_payload = [
        {"pos": list(b["pos"]), "is_ripe": b["is_ripe"], "timer": b["timer"], "max_timer": world.food_cooldown}
        for b in world.bushes
    ]
    trees_payload = [
        {"pos": list(t["pos"]), "is_mature": t["is_mature"], "timer": t["timer"], "max_timer": world.tree_cooldown}
        for t in world.tree_entries
    ]

    return {
        "type": "world_update",
        "tick": current_tick,
        "generation": current_generation,
        "day_num": day_num,
        "tod_name": tod_name,
        "grid_size": world.size,
        "home_pos": list(world.home_pos),
        "water_tiles": [list(pt) for pt in world.water_tiles],
        "hazards": [list(h.pos) for h in world.hazard_entities],
        "bushes": bushes_payload,
        "trees": trees_payload,
        "stockpile": {
            "food": world.stockpile["food"],
            "wood": world.stockpile["wood"]
        },
        "emergency_rations": world.emergency_rations_eaten,
        "heroes": heroes_payload,
        "brain": get_hero_brain_telemetry(selected_hero, selected_pos),
        "events": recent_events[-6:],
        "is_paused": is_paused,
        "tick_speed": tick_speed,
        "is_evolving": is_evolving
    }


def step_civilization():
    """Advances one tick of the persistent world."""
    global current_tick, current_generation, world, heroes, hero_positions, recent_events

    current_tick += 1
    world.tick_ecology()
    check_and_reload_checkpoint()

    for agent in heroes:
        hid = agent.id
        pos = hero_positions[hid]
        ar, ac = pos

        # Dynamic target mapping
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

        new_pos, event, hit_hazard = world.step_agent(pos, action)
        events = agent.memory.update_homeostasis(
            current_pos=new_pos,
            world=world,
            step_event=event,
            hit_hazard=hit_hazard
        )
        hero_positions[hid] = new_pos

        if events:
            recent_events.extend([f"[Hero {hid}] {ev}" for ev in events])

        # Respawn hero if perished
        if agent.memory.vitals["health"] <= 0.0:
            recent_events.append(f"💀 Hero {hid} perished! Reborn at Home Base ⛺")
            agent.reset_state()
            hero_positions[hid] = world.home_pos


async def civilization_persistent_loop():
    """Runs 24/7 on the server regardless of UI connection status."""
    global is_paused, tick_speed
    while True:
        if not is_paused:
            try:
                step_civilization()
                if len(manager.active_connections) > 0:
                    await manager.broadcast(json.dumps(get_world_state()))
            except Exception as e:
                print(f"Civilization step error: {e}", flush=True)
        delay = 1.0 / max(1.0, min(tick_speed, 40.0))
        await asyncio.sleep(delay)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(civilization_persistent_loop())


@app.get("/")
async def get_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_paused, tick_speed, selected_hero_id, is_evolving, current_generation, world, heroes
    await manager.connect(websocket)
    await websocket.send_text(json.dumps(get_world_state()))

    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")

            if action == "toggle_pause":
                is_paused = not is_paused
            elif action == "step_once":
                step_civilization()
                await manager.broadcast(json.dumps(get_world_state()))
                continue
            elif action == "set_speed":
                tick_speed = float(cmd.get("speed", 6.0))
            elif action == "select_hero":
                selected_hero_id = int(cmd.get("id", 1))
            elif action == "evolve_generations":
                num_gens = int(cmd.get("generations", 10))
                is_evolving = True
                await manager.broadcast(json.dumps(get_world_state()))

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, train_civilization, num_gens, 16, 150, 3, False)

                current_generation += num_gens
                is_evolving = False
                check_and_reload_checkpoint()
                recent_events.append(f"🧬 Evolutionary adaptation complete! Advanced {num_gens} generations.")
            elif action == "reset":
                world.reset_world()
                for h in heroes:
                    h.reset_state()
                    hero_positions[h.id] = world.home_pos
                recent_events.append("🔄 World reset by user command.")

            await manager.broadcast(json.dumps(get_world_state()))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
