import time
import argparse
import random
import sys
import json
import queue
import threading
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

from agent_phase2 import Phase2Agent, Phase2VillageMemory

# Web Streaming Queue & Worker
frame_queue: queue.Queue = queue.Queue(maxsize=3)
_sender_thread: Optional[threading.Thread] = None
_sender_running: bool = False

def _frame_sender_loop():
    while _sender_running:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if frame is None:
            break
        try:
            payload = json.dumps(frame).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:8000/api/scarcity/frame",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=0.15) as resp:
                pass
        except Exception:
            pass
        finally:
            frame_queue.task_done()

def start_web_streamer():
    global _sender_thread, _sender_running
    if _sender_thread is not None and _sender_thread.is_alive():
        return
    _sender_running = True
    _sender_thread = threading.Thread(target=_frame_sender_loop, daemon=True)
    _sender_thread.start()

def stop_web_streamer():
    global _sender_running
    _sender_running = False
    try:
        frame_queue.put_nowait(None)
    except Exception:
        pass

def queue_web_frame(frame: dict):
    if not _sender_running:
        return
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        try:
            _ = frame_queue.get_nowait()
            frame_queue.task_done()
            frame_queue.put_nowait(frame)
        except Exception:
            pass

def ensure_web_server_active(port: int = 8000) -> Optional[subprocess.Popen]:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/scarcity/health")
        with urllib.request.urlopen(req, timeout=0.3) as resp:
            if resp.status == 200:
                return None
    except Exception:
        pass

    cmd = [
        sys.executable, "-m", "uvicorn", "server:app",
        "--host", "0.0.0.0", "--port", str(port),
        "--log-level", "warning"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(25):
            time.sleep(0.1)
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/api/scarcity/health")
                with urllib.request.urlopen(req, timeout=0.2) as resp:
                    if resp.status == 200:
                        return proc
            except Exception:
                pass
        return proc
    except Exception:
        return None

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
# SCARCITY ECOLOGICAL WORLD (LIMITED RESOURCES & REGROWTH COOLDOWNS)
# ==============================================================================

class ScarcityPhase2Environment:
    """
    Ecological World with Strict Resource Scarcity and Regrowth Cooldowns:
    - Limited Bush Locations: When picked, bushes become barren (🌱) for a cooldown duration.
    - Limited Timber Trees: When chopped, become stumps (🪵) and take time to regrow.
    - Water Lake: Fixed natural hydration source.
    - Village Stockpile: Central communal reserve at Home Base (⛺) with emergency famine rationing.
    """
    def __init__(
        self,
        size: int = 14,
        num_bushes: int = 4,
        num_trees: int = 4,
        num_hazards: int = 4,
        food_respawn_cooldown: int = 35,
        tree_respawn_cooldown: int = 65,
        home_pos: Tuple[int, int] = (1, 1),
        water_center: Tuple[int, int] = (7, 7)
    ):
        self.size = size
        self.home_pos = home_pos
        self.water_center = water_center
        self.num_bushes = num_bushes
        self.num_trees = num_trees
        self.num_hazards = num_hazards
        self.food_cooldown = food_respawn_cooldown
        self.tree_cooldown = tree_respawn_cooldown

        self.stockpile = {"food": 0, "wood": 0}
        self.emergency_rations_eaten = 0

        # Build map
        self.reset_world()

    def reset_world(self):
        wr, wc = self.water_center
        self.water_tiles = [
            (wr + dr, wc + dc)
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0)]
            if 0 <= wr + dr < self.size and 0 <= wc + dc < self.size
        ]

        hr, hc = self.home_pos
        home_zone = {
            (hr + dr, hc + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if 0 <= hr + dr < self.size and 0 <= hc + dc < self.size
        }

        blocked = set(self.water_tiles) | home_zone
        all_coords = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in blocked
        ]
        random.shuffle(all_coords)

        self.hazards = all_coords[:self.num_hazards]
        all_coords = all_coords[self.num_hazards:]

        # Fixed bush locations: {pos: (r, c), is_ripe: bool, timer: int}
        bush_coords = all_coords[:self.num_bushes]
        self.bushes = [{"pos": pt, "is_ripe": True, "timer": 0} for pt in bush_coords]
        all_coords = all_coords[self.num_bushes:]

        # Fixed tree locations: {pos: (r, c), is_mature: bool, timer: int}
        tree_coords = all_coords[:self.num_trees]
        self.tree_entries = [{"pos": pt, "is_mature": True, "timer": 0} for pt in tree_coords]

    @property
    def foods(self) -> List[Tuple[int, int]]:
        return [b["pos"] for b in self.bushes if b["is_ripe"]]

    @property
    def ripe_food_positions(self) -> List[Tuple[int, int]]:
        return self.foods

    @property
    def trees(self) -> List[Tuple[int, int]]:
        return [t["pos"] for t in self.tree_entries if t["is_mature"]]

    @property
    def mature_tree_positions(self) -> List[Tuple[int, int]]:
        return self.trees

    def tick_ecology(self):
        """Advances ecological regrowth timers."""
        for b in self.bushes:
            if not b["is_ripe"]:
                b["timer"] -= 1
                if b["timer"] <= 0:
                    b["is_ripe"] = True
                    b["timer"] = 0

        for t in self.tree_entries:
            if not t["is_mature"]:
                t["timer"] -= 1
                if t["timer"] <= 0:
                    t["is_mature"] = True
                    t["timer"] = 0

    def is_water(self, pos: Tuple[int, int]) -> bool:
        return pos in self.water_tiles

    def is_adjacent_to_water(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in self.water_tiles:
                return True
        return False

    def is_home(self, pos: Tuple[int, int]) -> bool:
        return pos == self.home_pos

    def get_observation(
        self,
        agent_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        energy: float = 1.0,
        frustration: float = 0.0
    ) -> np.ndarray:
        ar, ac = agent_pos
        tr, tc = target_pos

        target_dx = float(np.clip((tr - ar) / float(self.size), -1.0, 1.0))
        target_dy = float(np.clip((tc - ac) / float(self.size), -1.0, 1.0))

        all_obstacles = list(self.hazards) + list(self.water_tiles)
        if all_obstacles:
            dists = [abs(hr - ar) + abs(hc - ac) for hr, hc in all_obstacles]
            min_idx = int(np.argmin(dists))
            best_obstacle = all_obstacles[min_idx]
            dr = best_obstacle[0] - ar
            dc = best_obstacle[1] - ac
            dist = abs(dr) + abs(dc)
            if 0 < dist <= 3:
                intensity = (4.0 - dist) / 3.0
                hazard_dx = float(np.clip((dr / float(dist)) * intensity, -1.0, 1.0))
                hazard_dy = float(np.clip((dc / float(dist)) * intensity, -1.0, 1.0))
            else:
                hazard_dx, hazard_dy = 0.0, 0.0
        else:
            hazard_dx, hazard_dy = 0.0, 0.0

        return np.array([
            target_dx,
            target_dy,
            hazard_dx,
            hazard_dy,
            energy,
            frustration
        ], dtype=np.float32)

    def step(self, pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], Optional[str], bool]:
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves.get(action, (0, 0))

        nr = max(0, min(self.size - 1, pos[0] + dr))
        nc = max(0, min(self.size - 1, pos[1] + dc))
        new_pos = (nr, nc)

        event = None
        hit_hazard = False

        if new_pos in self.hazards:
            event = "HAZARD_HIT"
            hit_hazard = True
        else:
            # Check ripe bushes
            for b in self.bushes:
                if b["is_ripe"] and b["pos"] == new_pos:
                    b["is_ripe"] = False
                    b["timer"] = self.food_cooldown
                    event = "FOOD_HARVESTED"
                    break

            # Check mature trees
            if event is None:
                for t in self.tree_entries:
                    if t["is_mature"] and t["pos"] == new_pos:
                        t["is_mature"] = False
                        t["timer"] = self.tree_cooldown
                        event = "WOOD_HARVESTED"
                        break

        return new_pos, event, hit_hazard

# ==============================================================================
# MULTI-HERO SCARCITY & FAMINE SURVIVAL SIMULATION
# ==============================================================================

def run_scarcity_simulation(
    num_heroes: int = 10,
    max_days: int = 100,
    ticks_per_day: int = 100,
    grid_size: int = 14,
    num_bushes: int = 4,
    num_trees: int = 4,
    food_respawn_cooldown: int = 35,
    tree_respawn_cooldown: int = 65,
    seed: int = 42,
    visual: bool = True,
    delay: float = 0.02,
    ckpt_path: str = "checkpoints/champion_phase2.pt"
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    server_proc = None
    if visual:
        server_proc = ensure_web_server_active(8000)
        start_web_streamer()
        print(f"\n{CYAN}{BOLD}🌐 Live Web Dashboard running at: http://localhost:8000/scarcity{RESET}\n")

    env = ScarcityPhase2Environment(
        size=grid_size,
        num_bushes=num_bushes,
        num_trees=num_trees,
        num_hazards=4,
        food_respawn_cooldown=food_respawn_cooldown,
        tree_respawn_cooldown=tree_respawn_cooldown,
        home_pos=(1, 1),
        water_center=(7, 7)
    )

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
            "cause_of_death": f"Survived Full {max_days} Days (Immortal Survivor)",
            "food_deposited": 0,
            "wood_deposited": 0,
            "times_drank": 0,
            "times_eaten": 0,
            "rations_eaten": 0,
            "final_health": 100.0
        }

    total_ticks = max_days * ticks_per_day
    action_names = ["North ⬆️", "South ⬇️", "West ⬅️", "East ➡️"]
    recent_events_feed: List[str] = [f"Simulation initiated: {num_heroes} heroes spawned under strict resource scarcity."]

    for tick in range(1, total_ticks + 1):
        day_num, tod_name = get_tod_name(tick, ticks_per_day)

        # 1. Advance Ecological Regrowth Timers
        env.tick_ecology()

        any_alive = False

        # 2. Step Each Living Hero
        for agent in heroes:
            hid = agent.id
            if not hero_alive[hid]:
                continue
            any_alive = True

            pos = hero_positions[hid]

            # Dynamic Target Selection for Scarcity
            ar, ac = pos
            if agent.memory.current_goal in ("RETURN_HOME", "REST", "SURVIVE"):
                agent.memory.target_pos = env.home_pos
            elif agent.memory.current_goal == "HYDRATE":
                water_dists = [abs(wr - ar) + abs(wc - ac) for wr, wc in env.water_tiles]
                agent.memory.target_pos = env.water_tiles[int(np.argmin(water_dists))]
            else: # FORAGE: Target only RIPE bushes or MATURE trees
                available_targets = env.ripe_food_positions + env.mature_tree_positions
                if available_targets:
                    dists = [abs(tr - ar) + abs(tc - ac) for tr, tc in available_targets]
                    agent.memory.target_pos = available_targets[int(np.argmin(dists))]
                else:
                    # All natural resources depleted! Return home to wait/rest
                    agent.memory.target_pos = env.home_pos

            obs = env.get_observation(
                agent_pos=pos,
                target_pos=agent.memory.target_pos,
                energy=agent.memory.vitals["energy"] / 100.0,
                frustration=agent.memory.compute_frustration()
            )

            # Rest if at Home Base with low stamina
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
            old_stock_f = env.stockpile["food"]
            old_stock_w = env.stockpile["wood"]

            # Step Individual Homeostasis
            events = agent.memory.update_homeostasis(
                current_pos=new_pos,
                env=env,
                step_event=event,
                hit_hazard=hit_hazard
            )
            hero_positions[hid] = new_pos

            # Communal Village Granary Emergency Rationing
            # If hero is severely starving (hunger >= 70%) and has no personal backpack food,
            # but has made it to Home Base: can consume 1 food from the Village Stockpile!
            if env.is_home(new_pos) and agent.memory.vitals["hunger"] >= 70.0 and agent.memory.backpack["food"] == 0:
                if env.stockpile["food"] > 0:
                    env.stockpile["food"] -= 1
                    env.emergency_rations_eaten += 1
                    hero_stats[hid]["rations_eaten"] += 1
                    agent.memory.vitals["hunger"] = max(0.0, agent.memory.vitals["hunger"] - 50.0)
                    events.append(f"🍲 H{hid:02d} ate 1 emergency ration from Village Granary (Stockpile: {env.stockpile['food']})")

            # Track deposits
            dep_f = env.stockpile["food"] - old_stock_f
            dep_w = env.stockpile["wood"] - old_stock_w
            if dep_f > 0 or dep_w > 0:
                hero_stats[hid]["food_deposited"] += max(0, dep_f)
                hero_stats[hid]["wood_deposited"] += max(0, dep_w)

            if agent.memory.vitals["thirst"] < 5.0 and old_thirst >= 50.0:
                hero_stats[hid]["times_drank"] += 1
            if agent.memory.vitals["hunger"] < old_hunger:
                hero_stats[hid]["times_eaten"] += 1

            if events:
                recent_events_feed.extend([f"[H{hid:02d}] {ev}" for ev in events])

            # Check Mortality
            if agent.memory.vitals["health"] <= 0.0:
                hero_alive[hid] = False
                hero_stats[hid]["days_survived"] = round(tick / float(ticks_per_day), 2)
                if agent.memory.vitals["hunger"] >= 88.0:
                    hero_stats[hid]["cause_of_death"] = "Starvation (Resource Scarcity / Famine)"
                elif agent.memory.vitals["thirst"] >= 88.0:
                    hero_stats[hid]["cause_of_death"] = "Dehydration (Lethal Thirst)"
                elif hit_hazard:
                    hero_stats[hid]["cause_of_death"] = "Hazard Trap (Mortality)"
                else:
                    hero_stats[hid]["cause_of_death"] = "Physical Collapse & Exhaustion"
                recent_events_feed.append(f"💀 Hero H{hid:02d} perished on Day {day_num} due to {hero_stats[hid]['cause_of_death']}!")

        for hid in hero_stats:
            if hero_alive[hid]:
                hero_stats[hid]["days_survived"] = round(tick / float(ticks_per_day), 2)
                hero_stats[hid]["final_health"] = round(heroes[hid - 1].memory.vitals["health"], 1)

        alive_count = sum(1 for a in hero_alive.values() if a)
        ripe_bushes_count = len(env.ripe_food_positions)
        mature_trees_count = len(env.mature_tree_positions)

        # Visual Mode
        if visual and (tick % 2 == 0 or tick == 1 or alive_count == 0):
            print("\033[H\033[J", end="")

            famine_banner = f"{RED}{BOLD}⚠️ FAMINE ALERT: 0 APPLES ON THE ENTIRE MAP!{RESET}" if ripe_bushes_count == 0 else f"{GREEN}Ecology Stable ({ripe_bushes_count} ripe, {mature_trees_count} trees){RESET}"

            print(f"{BOLD}╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
            print(f"{BOLD}║  🏕️  SCARCITY SIMULATOR V2: {num_heroes} HEROES • HIGH FAMINE PRESSURE (COOLDOWN RESPONSES)                   ║{RESET}")
            print(f"{BOLD}║  Day {day_num:3d}/{max_days} • {tod_name:<14} • Active: {alive_count:2d}/{num_heroes} 🛡️ • Granary: 🍏 {env.stockpile['food']:3d} | 🪵 {env.stockpile['wood']:3d} • {famine_banner:<10}  ║{RESET}")
            print(f"{BOLD}║  🌐 Web Dashboard: {CYAN}http://localhost:8000/scarcity{RESET}{BOLD}                                                         ║{RESET}")
            print(f"{BOLD}╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝{RESET}")

            # Emit frame to Web Dashboard
            alive_heroes_list = [heroes[hid - 1] for hid in range(1, num_heroes + 1) if hero_alive[hid]]
            frame_payload = {
                "type": "scarcity_update",
                "tick": tick,
                "day_num": day_num,
                "max_days": max_days,
                "tod_name": tod_name,
                "grid_size": grid_size,
                "home_pos": list(env.home_pos),
                "water_tiles": [list(pt) for pt in env.water_tiles],
                "hazards": [list(pt) for pt in env.hazards],
                "bushes": [{"pos": list(b["pos"]), "is_ripe": b["is_ripe"], "timer": b["timer"], "max_timer": env.food_cooldown} for b in env.bushes],
                "trees": [{"pos": list(t["pos"]), "is_mature": t["is_mature"], "timer": t["timer"], "max_timer": env.tree_cooldown} for t in env.tree_entries],
                "stockpile": {"food": env.stockpile["food"], "wood": env.stockpile["wood"]},
                "emergency_rations": env.emergency_rations_eaten,
                "total_heroes": num_heroes,
                "alive_count": alive_count,
                "heroes": [
                    {
                        "id": hid,
                        "r": hero_positions[hid][0],
                        "c": hero_positions[hid][1],
                        "alive": hero_alive[hid],
                        "vitals": {
                            "health": round(heroes[hid - 1].memory.vitals["health"], 1),
                            "hunger": round(heroes[hid - 1].memory.vitals["hunger"], 1),
                            "thirst": round(heroes[hid - 1].memory.vitals["thirst"], 1),
                            "energy": round(heroes[hid - 1].memory.vitals["energy"], 1),
                        },
                        "backpack": {
                            "food": heroes[hid - 1].memory.backpack["food"],
                            "wood": heroes[hid - 1].memory.backpack["wood"]
                        },
                        "goal": heroes[hid - 1].memory.current_goal,
                        "days_survived": hero_stats[hid]["days_survived"],
                        "cause_of_death": hero_stats[hid]["cause_of_death"] if not hero_alive[hid] else None,
                        "food_dep": hero_stats[hid]["food_deposited"],
                        "wood_dep": hero_stats[hid]["wood_deposited"],
                        "rations": hero_stats[hid]["rations_eaten"]
                    }
                    for hid in range(1, num_heroes + 1)
                ],
                "stats": {
                    "survival_rate": round((alive_count / max(1, num_heroes)) * 100, 1),
                    "avg_health": round(float(np.mean([h.memory.vitals["health"] for h in alive_heroes_list])), 1) if alive_heroes_list else 0.0,
                    "avg_hunger": round(float(np.mean([h.memory.vitals["hunger"] for h in alive_heroes_list])), 1) if alive_heroes_list else 0.0,
                    "avg_thirst": round(float(np.mean([h.memory.vitals["thirst"] for h in alive_heroes_list])), 1) if alive_heroes_list else 0.0,
                    "avg_energy": round(float(np.mean([h.memory.vitals["energy"] for h in alive_heroes_list])), 1) if alive_heroes_list else 0.0,
                    "starvation_deaths": sum(1 for s in hero_stats.values() if "Starvation" in s.get("cause_of_death", "")),
                    "dehydration_deaths": sum(1 for s in hero_stats.values() if "Dehydration" in s.get("cause_of_death", "")),
                    "hazard_deaths": sum(1 for s in hero_stats.values() if "Hazard" in s.get("cause_of_death", ""))
                },
                "recent_events": recent_events_feed[-6:],
                "is_paused": False,
                "tick_speed": round(1.0 / max(0.001, delay), 1)
            }
            queue_web_frame(frame_payload)

            # Map rendering
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
                    # Check Bush status
                    elif any(b["pos"] == pt for b in env.bushes):
                        b = next(b for b in env.bushes if b["pos"] == pt)
                        row_str += "🍏 " if b["is_ripe"] else f"{DIM}🌱 {RESET}"
                    # Check Tree status
                    elif any(t["pos"] == pt for t in env.tree_entries):
                        t = next(t for t in env.tree_entries if t["pos"] == pt)
                        row_str += "🌲 " if t["is_mature"] else f"{DIM}🪵 {RESET}"
                    elif pt in env.hazards:
                        row_str += "⚠️ "
                    else:
                        row_str += f"{DIM}· {RESET}"
                grid_lines.append(row_str)

            # Roster lines
            roster_lines = [
                f"{BOLD}{'HERO':<5} | {'HEALTH':<13} | {'HUNGER':<13} | {'THIRST':<13} | {'PACK':<5} | {'GOAL':<12}{RESET}",
                f"{DIM}{'-' * 74}{RESET}"
            ]
            for agent in heroes[:10]:
                hid = agent.id
                if not hero_alive[hid]:
                    status_line = f"{RED}H{hid:02d}   | DEAD (Day {hero_stats[hid]['days_survived']:4.1f}) - {hero_stats[hid]['cause_of_death'][:26]}{RESET}"
                    roster_lines.append(status_line)
                    continue

                mem = agent.memory
                v = mem.vitals
                h_meter = render_meter(v["health"], color=GREEN if v["health"] > 40 else RED)
                hu_meter = render_meter(v["hunger"], color=RED if v["hunger"] >= 70 else YELLOW)
                t_meter = render_meter(v["thirst"], color=RED if v["thirst"] >= 70 else CYAN)
                pack_str = f"{'🍏' * mem.backpack['food']}{'🪵' * mem.backpack['wood']}"
                pack_disp = f"{pack_str:<5}"

                goal_color = CYAN if mem.current_goal == "HYDRATE" else (YELLOW if mem.current_goal == "RETURN_HOME" else (BLUE if mem.current_goal == "REST" else GREEN))
                goal_disp = f"{goal_color}{mem.current_goal:<12}{RESET}"

                roster_lines.append(f"H{hid:02d}   | {h_meter} | {hu_meter} | {t_meter} | {pack_disp} | {goal_disp}")

            # Events Feed lines (Bottom or Right)
            for idx in range(max(len(grid_lines), len(roster_lines))):
                g_part = grid_lines[idx] if idx < len(grid_lines) else " " * (grid_size * 3 + 2)
                r_part = roster_lines[idx] if idx < len(roster_lines) else ""
                print(f"{g_part}  │ {r_part}")

            print(f"\n{BOLD}📢 Recent Ecological & Village Events:{RESET}")
            for ev in recent_events_feed[-3:]:
                print(f"  • {ev}")

            print(f"\n{DIM}Controls: Press Ctrl+C to stop simulation.{RESET}")
            time.sleep(delay)

        if not any_alive:
            print(f"\n{RED}{BOLD}💀 COMPLETE EXTINCTION: All {num_heroes} heroes have perished on Day {day_num} due to severe famine.{RESET}")
            break

    # Final Scorecard
    survived_full = sum(1 for a in hero_alive.values() if a)
    days_list = [s["days_survived"] for s in hero_stats.values()]

    print("\n" + f"{BOLD}══════════════════════════════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}🏆 SCARCITY & FAMINE SURVIVAL SCORECARD ({num_heroes} HEROES • {max_days} DAYS HORIZON){RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"  • Survival Rate        : {BOLD}{(survived_full / num_heroes) * 100:5.1f}%{RESET} ({survived_full}/{num_heroes} survived the 100-day famine horizon)")
    print(f"  • Average Longevity    : {BOLD}{np.mean(days_list):5.2f} Days{RESET} (Range: {min(days_list):.1f}d to {max(days_list):.1f}d)")
    print(f"  • Village Granary      : {GREEN}{env.stockpile['food']} 🍏 Food remaining{RESET}")
    print(f"  • Village Lumber Depot : {YELLOW}{env.stockpile['wood']} 🪵 Wood remaining{RESET}")
    print(f"  • Emergency Rations    : {CYAN}{env.emergency_rations_eaten} 🍲 meals eaten from granary to stave off starvation{RESET}")
    print("-" * 106)
    print(f"{BOLD}{'RANK':<5} | {'HERO':<8} | {'DAYS SURVIVED':<14} | {'FOOD DEP':<10} | {'WOOD DEP':<10} | {'OUTCOME'}{RESET}")
    print("-" * 106)

    sorted_heroes = sorted(
        hero_stats.items(),
        key=lambda item: (item[1]["days_survived"], item[1]["food_deposited"] + item[1]["wood_deposited"]),
        reverse=True
    )

    for rank, (hid, s) in enumerate(sorted_heroes, 1):
        color = GREEN if s["days_survived"] >= max_days else (YELLOW if s["days_survived"] >= max_days * 0.5 else RED)
        print(f"#{rank:<4} | Hero {hid:02d}  | {s['days_survived']:5.1f} / {max_days} d  | {s['food_deposited']:6d} 🍏   | {s['wood_deposited']:6d} 🪵   | {color}{s['cause_of_death']}{RESET}")

    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════════════════════════════{RESET}\n")

    if visual:
        stop_web_streamer()

    return {
        "survived_full": survived_full,
        "avg_days": np.mean(days_list),
        "emergency_rations": env.emergency_rations_eaten
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Heroes under Resource Scarcity & Famine Cooldowns (V2)")
    parser.add_argument("--heroes", type=int, default=10, help="Number of heroes to send simultaneously (default: 10)")
    parser.add_argument("--max-days", type=int, default=100, help="Target days horizon (default: 100 days = 10,000 ticks)")
    parser.add_argument("--bushes", type=int, default=4, help="Total food bushes on map (default: 4 for extreme competition)")
    parser.add_argument("--trees", type=int, default=4, help="Total timber trees on map (default: 4)")
    parser.add_argument("--food-cooldown", type=int, default=35, help="Ticks for a bush to regrow fruit after being picked (default: 35)")
    parser.add_argument("--tree-cooldown", type=int, default=65, help="Ticks for a tree stump to regrow timber (default: 65)")
    parser.add_argument("--visual", action="store_true", help="Render live map, depleted bushes (🌱), stumps (🪵), and roster")
    parser.add_argument("--delay", type=float, default=0.02, help="Visual mode delay per tick (default: 0.02s)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/champion_phase2.pt", help="Checkpoint path")
    args = parser.parse_args()

    run_scarcity_simulation(
        num_heroes=args.heroes,
        max_days=args.max_days,
        ticks_per_day=100,
        grid_size=14,
        num_bushes=args.bushes,
        num_trees=args.trees,
        food_respawn_cooldown=args.food_cooldown,
        tree_respawn_cooldown=args.tree_cooldown,
        seed=42,
        visual=args.visual,
        delay=args.delay,
        ckpt_path=args.ckpt
    )

if __name__ == "__main__":
    main()
