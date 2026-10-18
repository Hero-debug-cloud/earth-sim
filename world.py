import random
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class HazardBeast:
    """A dynamic wandering hazard entity that patrols the world."""
    def __init__(self, pos: Tuple[int, int], move_interval: int = 2):
        self.pos = pos
        self.move_interval = move_interval
        self.tick_counter = 0

    def step(self, grid_size: int, blocked: set[Tuple[int, int]]) -> Tuple[int, int]:
        self.tick_counter += 1
        if self.tick_counter < self.move_interval:
            return self.pos

        self.tick_counter = 0
        r, c = self.pos
        possible_moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in blocked:
                possible_moves.append((nr, nc))

        if possible_moves:
            self.pos = random.choice(possible_moves)
        return self.pos


class CivilizationWorld:
    """
    Unified Persistent World:
    - Home Base ⛺ (1, 1): Shelter and central granary stockpile
    - Water Lake 💧: Natural hydration spring
    - Berry Bushes 🍏: Harvestable food with regrowth cooldowns (barren: 🌱)
    - Timber Trees 🌲: Harvestable wood with regrowth cooldowns (stump: 🪵)
    - Dynamic Moving Hazards ⚠️: Wandering predator entities
    - Day/Night Cycle & Passive Ecology
    """
    def __init__(
        self,
        size: int = 14,
        num_bushes: int = 5,
        num_trees: int = 5,
        num_hazards: int = 3,
        food_cooldown: int = 35,
        tree_cooldown: int = 60,
        hazard_speed: int = 2,
        home_pos: Tuple[int, int] = (1, 1),
        water_center: Tuple[int, int] = (7, 7)
    ):
        self.size = size
        self.home_pos = home_pos
        self.water_center = water_center
        self.num_bushes = num_bushes
        self.num_trees = num_trees
        self.num_hazards = num_hazards
        self.food_cooldown = food_cooldown
        self.tree_cooldown = tree_cooldown
        self.hazard_speed = hazard_speed

        self.stockpile = {"food": 0, "wood": 0}
        self.emergency_rations_eaten = 0
        self.tick_count = 0

        self.reset_world()

    def reset_world(self):
        # 1. Water Lake
        wr, wc = self.water_center
        self.water_tiles = [
            (wr + dr, wc + dc)
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0)]
            if 0 <= wr + dr < self.size and 0 <= wc + dc < self.size
        ]

        # 2. Home Shelter Zone
        hr, hc = self.home_pos
        self.home_zone = {
            (hr + dr, hc + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if 0 <= hr + dr < self.size and 0 <= hc + dc < self.size
        }

        blocked = set(self.water_tiles) | self.home_zone
        available_coords = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in blocked
        ]
        random.shuffle(available_coords)

        # 3. Dynamic Moving Hazards
        hazard_coords = available_coords[:self.num_hazards]
        available_coords = available_coords[self.num_hazards:]
        self.hazard_entities = [HazardBeast(pos=pt, move_interval=self.hazard_speed) for pt in hazard_coords]

        # 4. Bushes (Food)
        bush_coords = available_coords[:self.num_bushes]
        available_coords = available_coords[self.num_bushes:]
        self.bushes = [{"pos": pt, "is_ripe": True, "timer": 0} for pt in bush_coords]

        # 5. Trees (Wood)
        tree_coords = available_coords[:self.num_trees]
        self.tree_entries = [{"pos": pt, "is_mature": True, "timer": 0} for pt in tree_coords]

    @property
    def hazards(self) -> List[Tuple[int, int]]:
        return [h.pos for h in self.hazard_entities]

    @property
    def ripe_food_positions(self) -> List[Tuple[int, int]]:
        return [b["pos"] for b in self.bushes if b["is_ripe"]]

    @property
    def mature_tree_positions(self) -> List[Tuple[int, int]]:
        return [t["pos"] for t in self.tree_entries if t["is_mature"]]

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

    def tick_ecology(self):
        """Advances ecological regrowth and moves dynamic hazards."""
        self.tick_count += 1

        # Regrow bushes
        for b in self.bushes:
            if not b["is_ripe"]:
                b["timer"] -= 1
                if b["timer"] <= 0:
                    b["is_ripe"] = True
                    b["timer"] = 0

        # Regrow trees
        for t in self.tree_entries:
            if not t["is_mature"]:
                t["timer"] -= 1
                if t["timer"] <= 0:
                    t["is_mature"] = True
                    t["timer"] = 0

        # Move hazards (cannot enter home base or lake)
        impassable_for_hazards = set(self.water_tiles) | {self.home_pos}
        for h in self.hazard_entities:
            h.step(self.size, impassable_for_hazards)

    def get_observation(
        self,
        agent_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        energy: float = 1.0,
        frustration: float = 0.0
    ) -> np.ndarray:
        """
        Sensory input tensor for the recurrent policy network:
        [target_dx, target_dy, hazard_dx, hazard_dy, energy, frustration]
        """
        ar, ac = agent_pos
        tr, tc = target_pos

        target_dx = float(np.clip((tr - ar) / float(self.size), -1.0, 1.0))
        target_dy = float(np.clip((tc - ac) / float(self.size), -1.0, 1.0))

        # Dynamic distance to nearest moving hazard
        current_hazards = self.hazards
        if current_hazards:
            dists = [abs(hr - ar) + abs(hc - ac) for hr, hc in current_hazards]
            closest_h = current_hazards[int(np.argmin(dists))]
            hazard_dx = float(np.clip((closest_h[0] - ar) / float(self.size), -1.0, 1.0))
            hazard_dy = float(np.clip((closest_h[1] - ac) / float(self.size), -1.0, 1.0))
        else:
            hazard_dx, hazard_dy = 0.0, 0.0

        return np.array([
            target_dx,
            target_dy,
            hazard_dx,
            hazard_dy,
            float(np.clip(energy, 0.0, 1.0)),
            float(np.clip(frustration, 0.0, 1.0))
        ], dtype=np.float32)

    def step_agent(self, current_pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], Optional[str], bool]:
        """
        Executes hero movement, bounds checking, water blocking, and tile interactions.
        Returns: (new_pos, event_string, hit_hazard_bool)
        """
        r, c = current_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves.get(action, (0, 0))
        nr, nc = r + dr, c + dc

        # Wall clamp
        nr = max(0, min(self.size - 1, nr))
        nc = max(0, min(self.size - 1, nc))
        new_pos = (nr, nc)

        # Deep water is impassable
        if self.is_water(new_pos):
            new_pos = current_pos

        # Check collision with dynamic hazards
        hit_hazard = new_pos in self.hazards

        # Interactions
        event = None

        # Check bush harvest
        for b in self.bushes:
            if b["pos"] == new_pos and b["is_ripe"]:
                b["is_ripe"] = False
                b["timer"] = self.food_cooldown
                event = "HARVEST_FOOD"
                break

        # Check tree chop
        if not event:
            for t in self.tree_entries:
                if t["pos"] == new_pos and t["is_mature"]:
                    t["is_mature"] = False
                    t["timer"] = self.tree_cooldown
                    event = "HARVEST_WOOD"
                    break

        return new_pos, event, hit_hazard

    def get_tod_name(self, ticks_per_day: int = 100) -> Tuple[int, str]:
        day_num = (self.tick_count // ticks_per_day) + 1
        day_tick = self.tick_count % ticks_per_day
        if day_tick < 25:
            return day_num, "Morning 🌅"
        elif day_tick < 65:
            return day_num, "Afternoon ☀️"
        elif day_tick < 85:
            return day_num, "Dusk 🌆"
        else:
            return day_num, "Night 🌙"
