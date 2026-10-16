import random
import numpy as np
from enum import IntEnum

class TileType(IntEnum):
    DIRT = 0
    WATER = 1
    GRASS = 2
    TREE = 3
    FIRE = 4
    ASH = 5
    ROCK = 6

class Creature:
    def __init__(self, x: int, y: int, generation: int = 1):
        self.x = x
        self.y = y
        self.generation = generation
        self.health = 100.0
        self.thirst = 15.0   # 0 = not thirsty, 100 = dying of thirst
        self.energy = 100.0  # 100 = full energy, 0 = exhausted
        self.last_action = f"Gen {generation} Spawned"
        self.is_alive = True
        self.dead_ticks = 0

    def update_vitals(self):
        if not self.is_alive:
            self.dead_ticks += 1
            return
        # Passive metabolism
        self.thirst = min(100.0, self.thirst + 0.4)
        self.energy = max(0.0, self.energy - 0.2)

        # Health consequences
        if self.thirst >= 85.0:
            self.health -= 2.0
        if self.energy <= 5.0:
            self.health -= 1.0

        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            self.last_action = f"Gen {self.generation} Died"
            self.dead_ticks = 0

class World:
    def __init__(self, size: int = 28):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.fuel = np.zeros((size, size), dtype=int)
        self.moisture = np.zeros((size, size), dtype=float)
        self.ticks = 0
        self.creature = Creature(size // 2, size // 2)
        self.generate_initial_terrain()

    def generate_initial_terrain(self):
        """Build initial earth layout: lake, soil, rocks, grass, and forest."""
        self.grid.fill(TileType.DIRT)
        self.fuel.fill(0)
        self.moisture.fill(0.0)

        center = self.size // 2

        # Create a winding river or lake in the center
        for r in range(self.size):
            for c in range(self.size):
                dist_center = np.sqrt((r - center)**2 + (c - center)**2)
                # Lake in center
                if dist_center < 4.5:
                    self.grid[r, c] = TileType.WATER
                # Meandering river
                elif abs(c - (center + int(np.sin(r / 3.0) * 4))) <= 1:
                    self.grid[r, c] = TileType.WATER

        # Place some rock clusters
        for _ in range(6):
            rx, ry = random.randint(2, self.size - 3), random.randint(2, self.size - 3)
            if self.grid[rx, ry] != TileType.WATER:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if random.random() < 0.6 and 0 <= rx+dx < self.size and 0 <= ry+dy < self.size:
                            if self.grid[rx+dx, ry+dy] != TileType.WATER:
                                self.grid[rx+dx, ry+dy] = TileType.ROCK

        # Calculate initial moisture from water
        self._spread_moisture()

        # Grow grass and trees where soil is moist
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == TileType.DIRT and self.moisture[r, c] > 0.2:
                    if random.random() < 0.7:
                        self.grid[r, c] = TileType.GRASS
                        if self.moisture[r, c] > 0.5 and random.random() < 0.4:
                            self.grid[r, c] = TileType.TREE
                            self.fuel[r, c] = random.randint(40, 80)

        # Spawn guaranteed active natural campfire / wildfires from the start!
        veg_locs = np.argwhere((self.grid == TileType.TREE) | (self.grid == TileType.GRASS))
        if len(veg_locs) > 0:
            for _ in range(2):
                idx = random.randint(0, len(veg_locs) - 1)
                fx, fy = int(veg_locs[idx][0]), int(veg_locs[idx][1])
                for dfx in (-1, 0, 1):
                    for dfy in (-1, 0, 1):
                        nr, nc = fx + dfx, fy + dfy
                        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != TileType.WATER:
                            self.grid[nr, nc] = TileType.FIRE
                            self.fuel[nr, nc] = random.randint(80, 150)

        # Place creature on safe grass/dirt tile away from fire
        for _ in range(100):
            cx, cy = random.randint(2, self.size - 3), random.randint(2, self.size - 3)
            if self.grid[cx, cy] in (TileType.GRASS, TileType.DIRT) and self.grid[cx, cy] != TileType.FIRE:
                self.creature = Creature(cx, cy)
                break

    def _spread_moisture(self):
        """Moisture radiates outward from water tiles."""
        self.moisture.fill(0.0)
        water_locs = np.argwhere(self.grid == TileType.WATER)
        for wr, wc in water_locs:
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    nr, nc = wr + dr, wc + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        dist = np.sqrt(dr*dr + dc*dc)
                        if dist <= 4.0:
                            self.moisture[nr, nc] = max(self.moisture[nr, nc], 1.0 - (dist / 4.5))

    def spark_fire(self, r: int, c: int):
        """Ignite a 3x3 wildfire cluster with high fuel so it blazes noticeably."""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.grid[nr, nc] != TileType.WATER and self.grid[nr, nc] != TileType.ROCK:
                        self.grid[nr, nc] = TileType.FIRE
                        self.fuel[nr, nc] = random.randint(60, 120)

    def trigger_lightning(self):
        """Random lightning strike igniting a forest fire."""
        tree_locs = np.argwhere(self.grid == TileType.TREE)
        if len(tree_locs) > 0:
            target = tree_locs[random.randint(0, len(tree_locs) - 1)]
            self.spark_fire(int(target[0]), int(target[1]))
            return int(target[0]), int(target[1])
        return None

    def add_water(self, r: int, c: int):
        """Add water tile cluster."""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    self.grid[nr, nc] = TileType.WATER
                    self.fuel[nr, nc] = 0
        self._spread_moisture()

    def plant_tree(self, r: int, c: int):
        """Plant a cluster of trees."""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.grid[nr, nc] in (TileType.DIRT, TileType.GRASS):
                        self.grid[nr, nc] = TileType.TREE
                        self.fuel[nr, nc] = random.randint(40, 80)

    def step(self, creature_action: int):
        """Advance the world by one tick with elemental physics and creature action."""
        self.ticks += 1
        new_grid = self.grid.copy()
        new_fuel = self.fuel.copy()

        # 1. Elemental Physics & Cellular Automata
        for r in range(self.size):
            for c in range(self.size):
                tile = self.grid[r, c]

                # Fire Physics
                if tile == TileType.FIRE:
                    new_fuel[r, c] -= 1
                    # Check neighboring tiles to spread fire
                    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            target = self.grid[nr, nc]
                            # Extinguished by water
                            if target == TileType.WATER:
                                new_grid[r, c] = TileType.ASH
                                new_fuel[r, c] = 0
                                break
                            # Spread to tree or grass
                            if target == TileType.TREE and random.random() < 0.35:
                                new_grid[nr, nc] = TileType.FIRE
                                new_fuel[nr, nc] = random.randint(10, 20)
                            elif target == TileType.GRASS and random.random() < 0.20:
                                new_grid[nr, nc] = TileType.FIRE
                                new_fuel[nr, nc] = random.randint(4, 8)

                    # Exhausted fuel turns to ash
                    if new_fuel[r, c] <= 0:
                        new_grid[r, c] = TileType.ASH

                # Ash slowly decomposes into fertile soil / grass
                elif tile == TileType.ASH:
                    if random.random() < 0.04:
                        new_grid[r, c] = TileType.GRASS if self.moisture[r, c] > 0.2 else TileType.DIRT

                # Grass & Tree growth near moisture
                elif tile == TileType.DIRT:
                    if self.moisture[r, c] > 0.3 and random.random() < 0.03:
                        new_grid[r, c] = TileType.GRASS
                elif tile == TileType.GRASS:
                    if self.moisture[r, c] > 0.6 and random.random() < 0.005:
                        new_grid[r, c] = TileType.TREE
                        new_fuel[r, c] = random.randint(15, 25)

        self.grid = new_grid
        self.fuel = new_fuel

        # 2. Creature Sensory Feedback and Execution
        self._execute_creature_action(creature_action)
        self.creature.update_vitals()

        # Check environmental hazards on creature
        cx, cy = self.creature.x, self.creature.y
        if self.grid[cx, cy] == TileType.FIRE:
            self.creature.health -= 25.0
            self.creature.last_action += " (Burned by Fire!)"

        # Generational Auto-Respawn
        if not self.creature.is_alive and self.creature.dead_ticks >= 6:
            new_gen = self.creature.generation + 1
            safe_spots = np.argwhere(((self.grid == TileType.GRASS) | (self.grid == TileType.DIRT)) & (self.grid != TileType.FIRE))
            if len(safe_spots) > 0:
                spot = safe_spots[random.randint(0, len(safe_spots) - 1)]
                self.creature = Creature(int(spot[0]), int(spot[1]), generation=new_gen)
                self.creature.last_action = f"Gen {new_gen} Reborn! 🌟"

    def _execute_creature_action(self, action: int):
        """
        0: Idle/Rest
        1: Move North (-1, 0)
        2: Move South (1, 0)
        3: Move West (0, -1)
        4: Move East (0, 1)
        5: Drink Water
        6: Eat Berries / Forage
        """
        if not self.creature.is_alive:
            return

        actions = ["Rest", "Move North", "Move South", "Move West", "Move East", "Drink Water", "Forage / Eat"]
        self.creature.last_action = actions[action] if action < len(actions) else "Idle"

        moves = {
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }

        if action in moves:
            dr, dc = moves[action]
            nx, ny = self.creature.x + dr, self.creature.y + dc
            if 0 <= nx < self.size and 0 <= ny < self.size:
                # Don't walk directly into deep water or rocks without swim/climb
                if self.grid[nx, ny] != TileType.ROCK:
                    self.creature.x = nx
                    self.creature.y = ny
                    self.creature.energy = max(0.0, self.creature.energy - 0.5)

        elif action == 0:  # Rest
            self.creature.energy = min(100.0, self.creature.energy + 2.0)

        elif action == 5:  # Drink
            # Check adjacent tiles or current tile for water
            cx, cy = self.creature.x, self.creature.y
            has_water = self.grid[cx, cy] == TileType.WATER
            for dx, dy in ((-1,0), (1,0), (0,-1), (0,1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx, ny] == TileType.WATER:
                    has_water = True
                    break
            if has_water:
                self.creature.thirst = max(0.0, self.creature.thirst - 35.0)
                self.creature.last_action = "Drank Water 💧"
            else:
                self.creature.last_action = "Tried to Drink (No water nearby)"

        elif action == 6:  # Forage / Eat
            cx, cy = self.creature.x, self.creature.y
            if self.grid[cx, cy] in (TileType.GRASS, TileType.TREE):
                self.creature.energy = min(100.0, self.creature.energy + 15.0)
                self.creature.health = min(100.0, self.creature.health + 5.0)
                self.creature.last_action = "Ate Wild Flora 🌱"

    def get_creature_sensors(self) -> np.ndarray:
        """
        Sensory inputs for the neural network (8 normalized features):
        0: Thirst (0.0 to 1.0)
        1: Energy (0.0 to 1.0)
        2: Health (0.0 to 1.0)
        3: Water proximity direction X (-1.0 to 1.0)
        4: Water proximity direction Y (-1.0 to 1.0)
        5: Fire danger proximity (0.0 = safe, 1.0 = adjacent fire)
        6: Food/Tree proximity direction X (-1.0 to 1.0)
        7: Food/Tree proximity direction Y (-1.0 to 1.0)
        """
        cx, cy = self.creature.x, self.creature.y
        thirst_norm = self.creature.thirst / 100.0
        energy_norm = self.creature.energy / 100.0
        health_norm = self.creature.health / 100.0

        # Find nearest water
        water_coords = np.argwhere(self.grid == TileType.WATER)
        if len(water_coords) > 0:
            dists = np.sqrt((water_coords[:, 0] - cx)**2 + (water_coords[:, 1] - cy)**2)
            idx = np.argmin(dists)
            closest_w = water_coords[idx]
            water_dx = np.clip((closest_w[0] - cx) / 10.0, -1.0, 1.0)
            water_dy = np.clip((closest_w[1] - cy) / 10.0, -1.0, 1.0)
        else:
            water_dx, water_dy = 0.0, 0.0

        # Find nearest fire
        fire_coords = np.argwhere(self.grid == TileType.FIRE)
        if len(fire_coords) > 0:
            dists = np.sqrt((fire_coords[:, 0] - cx)**2 + (fire_coords[:, 1] - cy)**2)
            min_dist = np.min(dists)
            fire_danger = max(0.0, 1.0 - (min_dist / 6.0))
        else:
            fire_danger = 0.0

        # Find nearest tree/food
        tree_coords = np.argwhere(self.grid == TileType.TREE)
        if len(tree_coords) > 0:
            dists = np.sqrt((tree_coords[:, 0] - cx)**2 + (tree_coords[:, 1] - cy)**2)
            idx = np.argmin(dists)
            closest_t = tree_coords[idx]
            food_dx = np.clip((closest_t[0] - cx) / 10.0, -1.0, 1.0)
            food_dy = np.clip((closest_t[1] - cy) / 10.0, -1.0, 1.0)
        else:
            food_dx, food_dy = 0.0, 0.0

        return np.array([
            thirst_norm,
            energy_norm,
            health_norm,
            water_dx,
            water_dy,
            fire_danger,
            food_dx,
            food_dy
        ], dtype=np.float32)

    def get_state_summary(self):
        """Summary for frontend."""
        unique, counts = np.unique(self.grid, return_counts=True)
        counts_dict = dict(zip(unique.tolist(), counts.tolist()))
        return {
            "tick": self.ticks,
            "grid_size": self.size,
            "grid": self.grid.tolist(),
            "creature": {
                "x": self.creature.x,
                "y": self.creature.y,
                "generation": self.creature.generation,
                "health": round(self.creature.health, 1),
                "thirst": round(self.creature.thirst, 1),
                "energy": round(self.creature.energy, 1),
                "last_action": self.creature.last_action,
                "is_alive": self.creature.is_alive
            },
            "stats": {
                "water": counts_dict.get(int(TileType.WATER), 0),
                "grass": counts_dict.get(int(TileType.GRASS), 0),
                "tree": counts_dict.get(int(TileType.TREE), 0),
                "fire": counts_dict.get(int(TileType.FIRE), 0),
                "ash": counts_dict.get(int(TileType.ASH), 0),
                "rock": counts_dict.get(int(TileType.ROCK), 0),
                "dirt": counts_dict.get(int(TileType.DIRT), 0)
            }
        }
