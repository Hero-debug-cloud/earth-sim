import random
import numpy as np

class Environment:
    def __init__(self, size: int = 12, num_food: int = 8, num_hazards: int = 3):
        self.size = size
        self.num_food = num_food
        self.num_hazards = num_hazards
        self.food_positions = []
        self.hazard_positions = []
        self.reset_world()

    def reset_world(self):
        """Place food and hazards at random unique positions."""
        all_positions = [(r, c) for r in range(self.size) for c in range(self.size)]
        random.shuffle(all_positions)

        self.hazards = all_positions[:self.num_hazards]
        self.foods = all_positions[self.num_hazards:self.num_hazards + self.num_food]

    def respawn_food(self, idx: int):
        """Respawn a food item after being eaten."""
        occupied = set(self.foods + self.hazards)
        available = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in occupied]
        if available:
            self.foods[idx] = random.choice(available)

    def get_observation(self, agent_pos: tuple[int, int], agent_energy: float) -> np.ndarray:
        """
        Sensory input for agent (5 features):
        0: dx to nearest food (-1.0 to 1.0)
        1: dy to nearest food (-1.0 to 1.0)
        2: dx to nearest hazard (-1.0 to 1.0)
        3: dy to nearest hazard (-1.0 to 1.0)
        4: Normalized energy (0.0 to 1.0)
        """
        ar, ac = agent_pos

        # Nearest Food
        food_dists = [abs(fr - ar) + abs(fc - ac) for fr, fc in self.foods]
        min_f_idx = int(np.argmin(food_dists))
        best_f = self.foods[min_f_idx]
        food_dx = np.clip((best_f[0] - ar) / float(self.size), -1.0, 1.0)
        food_dy = np.clip((best_f[1] - ac) / float(self.size), -1.0, 1.0)

        # Nearest Hazard with Inverse Proximity & Amplified Gain (Option A)
        # At distance 1: repulsion intensity = 3.5 (overpowers food attraction gradient)
        # Decays to 0.0 at distance >= 4 (Safe zone)
        if self.hazards:
            hazard_dists = [abs(hr - ar) + abs(hc - ac) for hr, hc in self.hazards]
            min_h_idx = int(np.argmin(hazard_dists))
            best_h = self.hazards[min_h_idx]
            dr = best_h[0] - ar
            dc = best_h[1] - ac
            dist = abs(dr) + abs(dc)
            if 0 < dist <= 3:
                intensity = (4.0 - dist) / 3.0  # dist 1 -> 1.0, dist 2 -> 0.67, dist 3 -> 0.33
                hazard_dx = np.clip((dr / float(dist)) * intensity, -1.0, 1.0)
                hazard_dy = np.clip((dc / float(dist)) * intensity, -1.0, 1.0)
            else:
                hazard_dx, hazard_dy = 0.0, 0.0
        else:
            hazard_dx, hazard_dy = 0.0, 0.0

        return np.array([food_dx, food_dy, hazard_dx, hazard_dy, agent_energy], dtype=np.float32)

    def step_agent(self, agent_pos: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool, bool]:
        """
        Takes action:
        0: Up, 1: Down, 2: Left, 3: Right
        Returns (new_pos, reward, ate_food, hit_hazard)
        """
        ar, ac = agent_pos
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        dr, dc = moves.get(action, (0, 0))
        nr = max(0, min(self.size - 1, ar + dr))
        nc = max(0, min(self.size - 1, ac + dc))
        new_pos = (nr, nc)
        # Distance to nearest food before and after move
        old_dist = min([abs(fr - ar) + abs(fc - ac) for fr, fc in self.foods])
        new_dist = min([abs(fr - nr) + abs(fc - nc) for fr, fc in self.foods])

        # Step penalty + distance improvement reward
        reward = (old_dist - new_dist) * 0.4 - 0.05
        ate_food = False
        hit_hazard = False

        # Check Food
        if new_pos in self.foods:
            f_idx = self.foods.index(new_pos)
            reward += 10.0
            ate_food = True
            self.respawn_food(f_idx)

        # Check Hazard: Heavy penalty so ramming through hazard for food is a massive net loss
        if new_pos in self.hazards:
            reward -= 25.0
            hit_hazard = True

        return new_pos, reward, ate_food, hit_hazard
