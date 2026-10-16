import copy
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent_recurrent import RecurrentPolicyNetwork
from env_phase2 import Phase2Environment

class Phase2VillageMemory:
    """
    System 2 Cognitive Memory Store:
    - Biological Vitals: Health, Thirst, Hunger, Energy
    - Backpack Inventory: Food & Wood (Capacity: 5)
    - Spatial Mental Map: Home Base, Water Wells, Hazards seen
    - Hierarchical Goal State Machine: SURVIVE, HYDRATE, FORAGE, RETURN_HOME, DEPOSIT, REST
    - Spatial Frustration & Loop-Detection Engine
    """
    def __init__(
        self,
        agent_id: int,
        home_pos: Tuple[int, int] = (1, 1),
        backpack_capacity: int = 5,
        buffer_size: int = 8,
        frustration_threshold: float = 0.50
    ):
        self.agent_id = agent_id
        self.home_pos = home_pos
        self.backpack_capacity = backpack_capacity
        self.buffer_size = buffer_size
        self.frustration_threshold = frustration_threshold

        # Spatial Mental Map
        self.landmarks: Dict[str, Optional[Tuple[int, int]]] = {
            "home": home_pos,
            "water_well": None
        }
        self.hazards_seen: Set[Tuple[int, int]] = set()

        # Backpack Inventory
        self.backpack: Dict[str, int] = {
            "food": 0,
            "wood": 0
        }

        # Biological Vitals (0.0 - 100.0)
        self.vitals: Dict[str, float] = {
            "health": 100.0,
            "hunger": 0.0,   # 0 = full, 100 = starving
            "thirst": 0.0,   # 0 = hydrated, 100 = dying of thirst
            "energy": 100.0  # 100 = rested, 0 = exhausted
        }

        # Goal State Machine
        self.current_goal: str = "FORAGE"
        self.target_pos: Tuple[int, int] = home_pos
        self.last_event: Optional[str] = None
        self.events_log: List[str] = []

        # Rolling Buffer for Frustration & Loop-Breaking
        self.recent_positions: List[Tuple[int, int]] = []

    @property
    def carried_count(self) -> int:
        return self.backpack["food"] + self.backpack["wood"]

    @property
    def is_backpack_full(self) -> bool:
        return self.carried_count >= self.backpack_capacity

    def compute_frustration(self) -> float:
        """Calculates stagnation / oscillation score (0.0 to 1.0)."""
        if len(self.recent_positions) < 4:
            return 0.0
        window = self.recent_positions[-6:]
        unique_tiles = len(set(window))
        total_steps = len(window)
        raw = 1.0 - (float(unique_tiles - 1) / float(total_steps - 1))
        return float(np.clip(raw, 0.0, 1.0))

    def update_homeostasis(
        self,
        current_pos: Tuple[int, int],
        env: Phase2Environment,
        step_event: Optional[str] = None,
        hit_hazard: bool = False
    ) -> List[str]:
        """
        Step biological vitals, manage backpack harvesting, handle consumption,
        deposit haul at Home Base, rest/sleep, and update cognitive goals.
        """
        events: List[str] = []

        # 1. Spatial Tracking
        self.recent_positions.append(current_pos)
        if len(self.recent_positions) > self.buffer_size:
            self.recent_positions.pop(0)

        if hit_hazard:
            self.hazards_seen.add(current_pos)
            self.vitals["health"] = max(0.0, self.vitals["health"] - 15.0)
            events.append("⚠️ Took damage from hazard (-15 HP)")

        # 2. Passive Metabolism
        self.vitals["thirst"] = min(100.0, self.vitals["thirst"] + 0.55)
        self.vitals["hunger"] = min(100.0, self.vitals["hunger"] + 0.35)
        self.vitals["energy"] = max(0.0, self.vitals["energy"] - 0.28)

        # 3. Handle Resource Harvesting
        if step_event == "FOOD_HARVESTED":
            if not self.is_backpack_full:
                self.backpack["food"] += 1
                events.append(f"🍏 Harvested 1 Food (Backpack: {self.carried_count}/{self.backpack_capacity})")
        elif step_event == "WOOD_HARVESTED":
            if not self.is_backpack_full:
                self.backpack["wood"] += 1
                events.append(f"🪵 Harvested 1 Wood (Backpack: {self.carried_count}/{self.backpack_capacity})")

        # 4. Water Well Detection & Thirst Quenching
        if env.is_adjacent_to_water(current_pos) or env.is_water(current_pos):
            self.landmarks["water_well"] = current_pos
            if self.vitals["thirst"] > 10.0:
                old_thirst = self.vitals["thirst"]
                self.vitals["thirst"] = 0.0
                self.vitals["health"] = min(100.0, self.vitals["health"] + 2.0)
                events.append(f"💧 Quenched thirst at water spring ({old_thirst:.0f}% -> 0%)")

        # 5. Autonomous Eating from Backpack
        if self.vitals["hunger"] >= 50.0 and self.backpack["food"] > 0:
            self.backpack["food"] -= 1
            old_hunger = self.vitals["hunger"]
            self.vitals["hunger"] = max(0.0, self.vitals["hunger"] - 45.0)
            events.append(f"🍽️ Consumed 1 apple from backpack (Hunger: {old_hunger:.0f}% -> {self.vitals['hunger']:.0f}%)")

        # 6. Home Base Interactions (Deposit & Rest)
        if env.is_home(current_pos):
            # Deposit Haul
            deposited_food = self.backpack["food"]
            deposited_wood = self.backpack["wood"]
            if deposited_food > 0 or deposited_wood > 0:
                env.stockpile["food"] += deposited_food
                env.stockpile["wood"] += deposited_wood
                self.backpack["food"] = 0
                self.backpack["wood"] = 0
                events.append(f"📦 Deposited haul at Home: +{deposited_food} 🍏, +{deposited_wood} 🪵 (Stockpile: {env.stockpile['food']}F, {env.stockpile['wood']}W)")

            # Rest & Sleep
            if self.vitals["energy"] < 98.0:
                old_energy = self.vitals["energy"]
                self.vitals["energy"] = min(100.0, self.vitals["energy"] + 4.5)
                events.append(f"💤 Sleeping in shelter (Energy: {old_energy:.0f}% -> {self.vitals['energy']:.0f}%)")

        # 7. Vitals Degradation Penalties
        if self.vitals["thirst"] >= 90.0:
            self.vitals["health"] = max(0.0, self.vitals["health"] - 2.0)
            events.append("⚠️ Severe dehydration! Taking damage")
        if self.vitals["hunger"] >= 90.0:
            self.vitals["health"] = max(0.0, self.vitals["health"] - 1.5)
            events.append("⚠️ Severe starvation! Taking damage")
        if self.vitals["energy"] <= 5.0:
            self.vitals["health"] = max(0.0, self.vitals["health"] - 1.0)
            events.append("⚠️ Physical exhaustion! Taking damage")

        # 8. Hierarchical Goal State Machine
        old_goal = self.current_goal

        if self.vitals["health"] <= 25.0:
            self.current_goal = "SURVIVE"
        elif self.vitals["thirst"] >= 65.0:
            self.current_goal = "HYDRATE"
        elif self.is_backpack_full or self.vitals["energy"] <= 25.0:
            self.current_goal = "RETURN_HOME"
        elif env.is_home(current_pos) and self.vitals["energy"] < 95.0:
            self.current_goal = "REST"
        else:
            self.current_goal = "FORAGE"

        if self.current_goal != old_goal:
            events.append(f"🎯 Goal Shift: {old_goal} -> {self.current_goal}")

        # 9. Compute Target Coordinates for System 1 Steering
        ar, ac = current_pos
        if self.current_goal in ("RETURN_HOME", "REST", "SURVIVE"):
            self.target_pos = self.home_pos
        elif self.current_goal == "HYDRATE":
            # Target nearest water tile
            if env.water_tiles:
                water_dists = [abs(wr - ar) + abs(wc - ac) for wr, wc in env.water_tiles]
                best_w = env.water_tiles[int(np.argmin(water_dists))]
                self.target_pos = best_w
            else:
                self.target_pos = self.home_pos
        else: # FORAGE
            # Alternate or select nearest resource (food or tree)
            candidates = list(env.foods) + list(env.trees)
            if candidates:
                dists = [abs(cr - ar) + abs(cc - ac) for cr, cc in candidates]
                best_cand = candidates[int(np.argmin(dists))]
                self.target_pos = best_cand
            else:
                self.target_pos = self.home_pos

        if events:
            self.last_event = events[-1]
            self.events_log.extend(events)

        return events

class Phase2Agent:
    """
    Phase 2 Autonomous Villager Agent:
    - System 1: Recurrent Policy Network (GRU, h_t in R^16)
    - System 2: Phase 2 Structured Cognitive Memory & Homeostasis Engine
    """
    def __init__(
        self,
        agent_id: int = 1,
        in_features: int = 6,
        hidden_size: int = 16,
        lr: float = 0.005,
        home_pos: Tuple[int, int] = (1, 1),
        backpack_capacity: int = 5,
        ckpt_path: Optional[str] = "checkpoints/champion_v3_gru.pt"
    ):
        self.id = agent_id
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.lr = lr
        self.home_pos = home_pos

        self.model = RecurrentPolicyNetwork(in_features, hidden_size, out_actions=4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.hx = torch.zeros(1, hidden_size, dtype=torch.float32)

        # Lifetime RL trajectory buffers
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

        # Generational Evolutionary Metrics
        self.fitness: float = 0.0
        self.deposited_food: int = 0
        self.deposited_wood: int = 0
        self.items_harvested: int = 0
        self.waters_quenched: int = 0
        self.hazards_hit: int = 0
        self.generation: int = 1
        self.lineage: str = f"Hero_A{agent_id}"

        # System 2 Cognitive Memory
        self.memory = Phase2VillageMemory(
            agent_id=agent_id,
            home_pos=home_pos,
            backpack_capacity=backpack_capacity
        )

        # Load trained GRU weights if available
        if ckpt_path:
            try:
                self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            except Exception as e:
                print(f"Notice: Initializing fresh GRU weights ({e})")

    def reset_state(self):
        """Reset temporal state, vitals, and recent memory for a fresh expedition."""
        self.hx = torch.zeros(1, self.hidden_size, dtype=torch.float32)
        self.log_probs = []
        self.rewards = []
        self.memory.recent_positions = []
        self.memory.backpack = {"food": 0, "wood": 0}
        self.memory.vitals = {
            "health": 100.0,
            "hunger": 0.0,
            "thirst": 0.0,
            "energy": 100.0
        }
        self.memory.current_goal = "FORAGE"

    def select_action(
        self,
        obs: np.ndarray,
        pos: Tuple[int, int],
        grid_size: int = 14,
        hazards: Optional[List[Tuple[int, int]]] = None,
        water_tiles: Optional[List[Tuple[int, int]]] = None,
        temperature: float = 0.35,
        deterministic: bool = False
    ) -> int:
        """
        Compute action using GRU hidden state, wall boundary masks,
        reflex obstacle avoidance, and frustration-powered loop breaking.
        """
        state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, self.hx = self.model(state_t, self.hx)
        logits = logits.squeeze(0)

        ar, ac = pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # 1. Wall Masking
        if ar == 0: logits[0] = -1e9
        if ar == grid_size - 1: logits[1] = -1e9
        if ac == 0: logits[2] = -1e9
        if ac == grid_size - 1: logits[3] = -1e9

        # 2. Hazard & Deep Water Reflex Guard
        blocked_tiles = set(hazards or [])
        # Only treat water as impassable if not intending to drink
        if self.memory.current_goal != "HYDRATE" and water_tiles:
            blocked_tiles.update(water_tiles)

        for act_idx, (dr, dc) in moves.items():
            if (ar + dr, ac + dc) in blocked_tiles:
                logits[act_idx] -= 35.0

        # 3. Frustration-Powered Loop Breaking (System 2 Interlock)
        frustration = self.memory.compute_frustration()
        if len(self.memory.recent_positions) >= 2:
            prev_pos = self.memory.recent_positions[-1]
            reversal_penalty = 0.5 + 4.0 * frustration
            for act_idx, (dr, dc) in moves.items():
                if (ar + dr, ac + dc) == prev_pos:
                    logits[act_idx] -= reversal_penalty

            if frustration > 0.4 and len(self.memory.recent_positions) >= 4:
                recent = self.memory.recent_positions[-4:]
                rows = [r for r, c in recent]
                cols = [c for r, c in recent]

                if len(set(rows)) == 1 and len(set(cols)) <= 2:
                    logits[2] -= 3.0 * frustration
                    logits[3] -= 3.0 * frustration
                elif len(set(cols)) == 1 and len(set(rows)) <= 2:
                    logits[0] -= 3.0 * frustration
                    logits[1] -= 3.0 * frustration

        if deterministic:
            return int(torch.argmax(logits).item())

        scaled_probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        dist = Categorical(scaled_probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def record_reward(self, reward: float):
        self.rewards.append(reward)

    def train_rl_lifetime(self, gamma: float = 0.95):
        """
        Lifetime Backpropagation Through Time (BPTT).
        Updates both the GRU cell weights and output policy head.
        """
        if len(self.rewards) == 0 or len(self.log_probs) == 0:
            return

        discounted_returns = []
        running_add = 0.0
        for r in reversed(self.rewards):
            running_add = r + gamma * running_add
            discounted_returns.insert(0, running_add)

        returns_t = torch.tensor(discounted_returns, dtype=torch.float32)
        if len(returns_t) > 1 and returns_t.std() > 1e-5:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns_t):
            policy_loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Detach hidden state and clear episode buffers
        self.hx = self.hx.detach()
        self.log_probs = []
        self.rewards = []

    def clone_and_mutate(
        self,
        new_id: int,
        mutation_rate: float = 0.12,
        mutation_scale: float = 0.15
    ) -> "Phase2Agent":
        """
        Outer Loop: Genetic reproduction with Gaussian weight mutation.
        Clones GRU weights, linear projection head, and memory landmarks.
        """
        child = Phase2Agent(
            agent_id=new_id,
            in_features=self.in_features,
            hidden_size=self.hidden_size,
            lr=self.lr,
            home_pos=self.home_pos,
            backpack_capacity=self.memory.backpack_capacity,
            ckpt_path=None
        )
        child.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        child.generation = self.generation + 1
        child.lineage = f"{self.lineage}->G{child.generation}"

        # Mutate weights
        with torch.no_grad():
            for param in child.model.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_scale
                param.add_(mask * noise)

        return child

