import random
from typing import Dict, List, Tuple, Optional
import numpy as np

class Phase2Environment:
    """
    Phase 2 Simulation Environment:
    - 14x14 Grid with multi-tile ecological terrain
    - Home Base (⛺) with safe shelter and village stockpile
    - Permanent Water Source / Lake (💧) for thirst quenching
    - Food / Berry Bushes (🍏) that provide food
    - Trees (🌲) that provide harvestable wood
    - Natural Hazard zones (⚠️)
    """
    def __init__(
        self,
        size: int = 14,
        num_food: int = 6,
        num_trees: int = 6,
        num_hazards: int = 4,
        home_pos: Tuple[int, int] = (1, 1),
        water_center: Tuple[int, int] = (7, 7)
    ):
        self.size = size
        self.home_pos = home_pos
        self.num_food = num_food
        self.num_trees = num_trees
        self.num_hazards = num_hazards
        self.water_center = water_center

        # Village Communal Stockpile at Home Base
        self.stockpile: Dict[str, int] = {
            "food": 0,
            "wood": 0
        }

        # Resource lists
        self.water_tiles: List[Tuple[int, int]] = []
        self.hazards: List[Tuple[int, int]] = []
        self.foods: List[Tuple[int, int]] = []
        self.trees: List[Tuple[int, int]] = []

        self.reset_world()

    def reset_world(self):
        """Construct the ecological layout."""
        # 1. Define Water Spring / Lake Cluster around water_center
        wr, wc = self.water_center
        water_offsets = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0)]
        self.water_tiles = [
            (wr + dr, wc + dc)
            for dr, dc in water_offsets
            if 0 <= wr + dr < self.size and 0 <= wc + dc < self.size
        ]

        # Reserved / protected safe tiles around Home Base
        hr, hc = self.home_pos
        home_zone = {
            (hr + dr, hc + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if 0 <= hr + dr < self.size and 0 <= hc + dc < self.size
        }

        # Available free tiles
        blocked = set(self.water_tiles) | home_zone
        all_coords = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in blocked
        ]
        random.shuffle(all_coords)

        # Place Hazards
        self.hazards = all_coords[:self.num_hazards]
        all_coords = all_coords[self.num_hazards:]

        # Place Food Bushes
        self.foods = all_coords[:self.num_food]
        all_coords = all_coords[self.num_food:]

        # Place Timber Trees
        self.trees = all_coords[:self.num_trees]

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

    def respawn_food(self, idx: int):
        occupied = set(self.water_tiles) | set(self.hazards) | set(self.foods) | set(self.trees) | {self.home_pos}
        available = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in occupied
        ]
        if available:
            self.foods[idx] = random.choice(available)

    def respawn_tree(self, idx: int):
        occupied = set(self.water_tiles) | set(self.hazards) | set(self.foods) | set(self.trees) | {self.home_pos}
        available = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in occupied
        ]
        if available:
            self.trees[idx] = random.choice(available)

    def get_observation(
        self,
        agent_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        energy: float = 1.0,
        frustration: float = 0.0
    ) -> np.ndarray:
        """
        Sensory input for System 1 recurrent controller:
        0: Target dx (-1.0 to 1.0)
        1: Target dy (-1.0 to 1.0)
        2: Hazard dx (-1.0 to 1.0, amplified inverse-distance repulsion)
        3: Hazard dy (-1.0 to 1.0)
        4: Normalized energy (0.0 to 1.0)
        5: Frustration index (0.0 to 1.0)
        """
        ar, ac = agent_pos
        tr, tc = target_pos

        target_dx = float(np.clip((tr - ar) / float(self.size), -1.0, 1.0))
        target_dy = float(np.clip((tc - ac) / float(self.size), -1.0, 1.0))

        # Hazards radar: combine hazards and deep water into lethal obstacles
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
        """
        Takes move action:
        0: North (-1, 0)
        1: South (+1, 0)
        2: West  (0, -1)
        3: East  (0, +1)
        
        Returns:
            new_pos: Tuple[int, int]
            event: Optional[str] ('FOOD_HARVESTED', 'WOOD_HARVESTED', 'HAZARD_HIT', None)
            hit_hazard: bool
        """
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves.get(action, (0, 0))

        nr = max(0, min(self.size - 1, pos[0] + dr))
        nc = max(0, min(self.size - 1, pos[1] + dc))
        new_pos = (nr, nc)

        event = None
        hit_hazard = False

        # Check hazards (including falling into deep water)
        if new_pos in self.hazards:
            event = "HAZARD_HIT"
            hit_hazard = True
        elif new_pos in self.foods:
            event = "FOOD_HARVESTED"
            idx = self.foods.index(new_pos)
            self.respawn_food(idx)
        elif new_pos in self.trees:
            event = "WOOD_HARVESTED"
            idx = self.trees.index(new_pos)
            self.respawn_tree(idx)

        return new_pos, event, hit_hazard
