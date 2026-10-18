import copy
from typing import Dict, Any, List, Tuple, Set, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from world import CivilizationWorld


class RecurrentPolicyNetwork(nn.Module):
    """
    System 1: Recurrent Neural Policy Core.
    Uses a Gated Recurrent Unit (GRU) to process temporal sensory vectors
    and produce motor action logits.
    """
    def __init__(self, in_features: int = 6, hidden_size: int = 16, out_actions: int = 4):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_actions = out_actions

        self.gru = nn.GRUCell(in_features, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_actions)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_h = self.gru(x, h)
        logits = self.fc_out(next_h)
        return logits, next_h


class CivilizationMemory:
    """
    System 2: Structured Cognitive Memory Store.
    Tracks biological vitals, backpack inventory, spatial landmarks,
    and a frustration-based loop detection engine.
    """
    def __init__(
        self,
        agent_id: int,
        home_pos: Tuple[int, int] = (1, 1),
        backpack_capacity: int = 5,
        buffer_size: int = 8
    ):
        self.agent_id = agent_id
        self.home_pos = home_pos
        self.backpack_capacity = backpack_capacity
        self.buffer_size = buffer_size

        self.landmarks: Dict[str, Optional[Tuple[int, int]]] = {
            "home": home_pos,
            "water_well": None
        }
        self.hazards_seen: Set[Tuple[int, int]] = set()

        self.backpack: Dict[str, int] = {"food": 0, "wood": 0}
        self.vitals: Dict[str, float] = {
            "health": 100.0,
            "hunger": 0.0,
            "thirst": 0.0,
            "energy": 100.0
        }

        self.current_goal: str = "FORAGE"
        self.target_pos: Tuple[int, int] = home_pos
        self.recent_positions: List[Tuple[int, int]] = []

    @property
    def carried_count(self) -> int:
        return self.backpack["food"] + self.backpack["wood"]

    @property
    def is_backpack_full(self) -> bool:
        return self.carried_count >= self.backpack_capacity

    def compute_frustration(self) -> float:
        """Calculates stagnation / oscillation score between 0.0 and 1.0."""
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
        world: CivilizationWorld,
        step_event: Optional[str] = None,
        hit_hazard: bool = False
    ) -> List[str]:
        """Steps biological metabolism, manages backpack items, and handles goal transitions."""
        events: List[str] = []

        self.recent_positions.append(current_pos)
        if len(self.recent_positions) > self.buffer_size:
            self.recent_positions.pop(0)

        # Hazard damage
        if hit_hazard:
            self.vitals["health"] = max(0.0, self.vitals["health"] - 25.0)
            events.append("⚠️ Ambushed by moving hazard (-25 HP)")

        # Passive biological burn
        self.vitals["thirst"] = min(100.0, self.vitals["thirst"] + 0.55)
        self.vitals["hunger"] = min(100.0, self.vitals["hunger"] + 0.35)
        self.vitals["energy"] = max(0.0, self.vitals["energy"] - 0.28)

        # Hydration at water lake
        if world.is_adjacent_to_water(current_pos) or world.is_water(current_pos):
            if self.vitals["thirst"] > 10.0:
                self.vitals["thirst"] = 0.0
                events.append("💧 Quenched thirst at water lake")

        # Harvesting into backpack
        if step_event == "HARVEST_FOOD":
            if not self.is_backpack_full:
                self.backpack["food"] += 1
                events.append(f"🍏 Picked ripe apple ({self.carried_count}/{self.backpack_capacity})")
            else:
                events.append("🎒 Backpack full! Cannot pick food")
        elif step_event == "HARVEST_WOOD":
            if not self.is_backpack_full:
                self.backpack["wood"] += 1
                events.append(f"🪵 Chopped timber ({self.carried_count}/{self.backpack_capacity})")
            else:
                events.append("🎒 Backpack full! Cannot chop timber")

        # Auto-eating in the field when hungry
        if self.vitals["hunger"] >= 60.0 and self.backpack["food"] > 0:
            self.backpack["food"] -= 1
            self.vitals["hunger"] = max(0.0, self.vitals["hunger"] - 45.0)
            self.vitals["health"] = min(100.0, self.vitals["health"] + 15.0)
            events.append("🍎 Ate food ration from backpack")

        # Home Base depot and shelter
        if world.is_home(current_pos):
            # Deposit goods into central granary
            if self.backpack["food"] > 0 or self.backpack["wood"] > 0:
                f_dep = self.backpack["food"]
                w_dep = self.backpack["wood"]
                world.stockpile["food"] += f_dep
                world.stockpile["wood"] += w_dep
                self.backpack["food"] = 0
                self.backpack["wood"] = 0
                events.append(f"⛺ Deposited {f_dep} 🍏 and {w_dep} 🪵 into Village Granary")

            # Emergency granary meal if starving
            if self.vitals["hunger"] >= 70.0 and world.stockpile["food"] > 0:
                world.stockpile["food"] -= 1
                world.emergency_rations_eaten += 1
                self.vitals["hunger"] = max(0.0, self.vitals["hunger"] - 50.0)
                events.append("🍲 Ate emergency meal from Village Granary")

            # Rest and recharge stamina in shelter
            if self.vitals["energy"] < 95.0:
                self.vitals["energy"] = min(100.0, self.vitals["energy"] + 15.0)
                self.vitals["health"] = min(100.0, self.vitals["health"] + 5.0)
                events.append("💤 Resting in Home Shelter")

        # Vital decay if starving or dehydrated
        if self.vitals["hunger"] >= 88.0:
            self.vitals["health"] = max(0.0, self.vitals["health"] - 1.2)
        if self.vitals["thirst"] >= 88.0:
            self.vitals["health"] = max(0.0, self.vitals["health"] - 2.0)
        if self.vitals["energy"] <= 0.0:
            self.vitals["health"] = max(0.0, self.vitals["health"] - 0.8)

        # Update Cognitive Goal
        if self.vitals["health"] < 35.0 or self.vitals["hunger"] > 80.0 or self.vitals["thirst"] > 80.0:
            self.current_goal = "SURVIVE"
        elif self.vitals["thirst"] >= 50.0:
            self.current_goal = "HYDRATE"
        elif self.is_backpack_full or self.current_goal == "RETURN_HOME":
            if current_pos == self.home_pos and not self.is_backpack_full:
                self.current_goal = "FORAGE"
            else:
                self.current_goal = "RETURN_HOME"
        elif self.vitals["energy"] <= 25.0 or (self.current_goal == "REST" and self.vitals["energy"] < 90.0):
            self.current_goal = "REST"
        else:
            self.current_goal = "FORAGE"

        return events


class CivilizationAgent:
    """
    Unified Dual-Process Hero Agent:
    - System 1: Recurrent Policy Network (GRU)
    - System 2: Cognitive Memory & Survival Drives
    """
    def __init__(
        self,
        agent_id: int,
        in_features: int = 6,
        hidden_size: int = 16,
        home_pos: Tuple[int, int] = (1, 1),
        backpack_capacity: int = 5,
        ckpt_path: Optional[str] = None
    ):
        self.id = agent_id
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.lineage = f"Hero_{agent_id:02d}"

        self.model = RecurrentPolicyNetwork(in_features, hidden_size, out_actions=4)
        self.memory = CivilizationMemory(agent_id, home_pos, backpack_capacity)

        self.hx = torch.zeros(1, hidden_size)
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.fitness: float = 0.0
        self.food_deposited: int = 0
        self.wood_deposited: int = 0

        if ckpt_path and Path(ckpt_path).exists():
            try:
                self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            except Exception as e:
                pass

    def reset_state(self):
        """Resets recurrent hidden state, buffers, and vitals for a new trial."""
        self.hx = torch.zeros(1, self.hidden_size)
        self.rewards = []
        self.log_probs = []
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
        """Selects next directional step with wall masks, reflex obstacle guards, and frustration loop breaking."""
        state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, self.hx = self.model(state_t, self.hx)
        logits = logits.squeeze(0)

        ar, ac = pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # Wall masking
        if ar == 0: logits[0] = -1e9
        if ar == grid_size - 1: logits[1] = -1e9
        if ac == 0: logits[2] = -1e9
        if ac == grid_size - 1: logits[3] = -1e9

        # Reflex guard: Avoid walking directly into known hazards or deep water
        blocked_tiles = set(hazards or [])
        if self.memory.current_goal != "HYDRATE" and water_tiles:
            blocked_tiles.update(water_tiles)

        for act_idx, (dr, dc) in moves.items():
            if (ar + dr, ac + dc) in blocked_tiles:
                logits[act_idx] -= 35.0

        # Frustration loop breaking
        frustration = self.memory.compute_frustration()
        if len(self.memory.recent_positions) >= 2:
            prev_pos = self.memory.recent_positions[-1]
            reversal_penalty = 0.5 + 4.0 * frustration
            for act_idx, (dr, dc) in moves.items():
                if (ar + dr, ac + dc) == prev_pos:
                    logits[act_idx] -= reversal_penalty

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
        """Reinforcement learning lifetime update via backpropagation through time."""
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns_t):
            policy_loss.append(-log_prob * G)

        if policy_loss:
            optimizer.zero_grad()
            total_loss = torch.stack(policy_loss).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        self.rewards = []
        self.log_probs = []

    def clone_and_mutate(self, new_id: int, mutation_rate: float = 0.15, mutation_scale: float = 0.18) -> "CivilizationAgent":
        """Spawns an offspring with cloned neural weights and random mutations."""
        child = CivilizationAgent(
            agent_id=new_id,
            in_features=self.in_features,
            hidden_size=self.hidden_size,
            home_pos=self.memory.home_pos,
            backpack_capacity=self.memory.backpack_capacity
        )
        child.model.load_state_dict({k: v.clone() for k, v in self.model.state_dict().items()})
        child.lineage = f"{self.lineage}->G"

        with torch.no_grad():
            for param in child.model.parameters():
                mask = (torch.rand_like(param) < mutation_rate).float()
                noise = torch.randn_like(param) * mutation_scale
                param.add_(mask * noise)

        return child
