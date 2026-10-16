import copy
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class VillageMemory:
    """
    System 2: Structured Cognitive Memory Store for an autonomous villager.
    Tracks spatial landmarks, social interactions, inventory, vitals,
    and a rolling position buffer for spatial loop/frustration detection.
    """
    def __init__(
        self,
        agent_id: int,
        home_pos: Tuple[int, int] = (1, 1),
        buffer_size: int = 8,
        patience_ticks: int = 50,
        frustration_threshold: float = 0.60
    ):
        self.agent_id = agent_id
        self.buffer_size = buffer_size
        self.patience_ticks = patience_ticks
        self.frustration_threshold = frustration_threshold

        # 1. Spatial Landmarks (Mental Map)
        self.landmarks: Dict[str, Optional[Tuple[int, int]]] = {
            "home": home_pos,
            "granary": None,
            "water_well": None
        }
        self.hazards_seen: Set[Tuple[int, int]] = set()

        # 2. Social Registry (Villager relationships)
        # agent_id -> {"trust": float, "affinity": float, "last_seen_tick": int}
        self.social_registry: Dict[int, Dict[str, Any]] = {}

        # 3. Inventory & Resources
        self.inventory: Dict[str, int] = {
            "food": 0,
            "wood": 0,
            "water": 0
        }

        # 4. Vitals & Drives
        self.vitals: Dict[str, float] = {
            "health": 100.0,
            "energy": 100.0,
            "thirst": 0.0,
            "hunger": 0.0
        }
        self.current_goal: str = "FORAGE"

        # 5. Spatial Rolling Buffer (Loop / Stagnation Detector)
        self.recent_positions: List[Tuple[int, int]] = []

        # 6. Cognitive Assessment & Autonomous Give-Up Engine
        self.target_investigation_ticks: int = 0
        self.abandon_target: bool = False
        self.abandon_reason: str = ""
        self.perimeter_hazards_discovered: Set[Tuple[int, int]] = set()

    def record_step(
        self,
        pos: Tuple[int, int],
        food_pos: Optional[Tuple[int, int]] = None,
        hazards: Optional[List[Tuple[int, int]]] = None,
        ate_food: bool = False,
        hit_hazard: bool = False
    ):
        """Record the new position and cognitively evaluate target feasibility."""
        self.recent_positions.append(pos)
        if len(self.recent_positions) > self.buffer_size:
            self.recent_positions.pop(0)

        if ate_food:
            self.inventory["food"] += 1
            self.target_investigation_ticks = 0
            self.abandon_target = False
            self.current_goal = "FORAGE"

        if hit_hazard:
            self.hazards_seen.add(pos)

        # Cognitive Autonomous Give-Up Logic (Requires High Persistence)
        if food_pos and hazards and not self.abandon_target:
            fr, fc = food_pos
            ar, ac = pos
            dist_to_food = abs(fr - ar) + abs(fc - ac)

            # If agent is probing near the food (dist <= 3)
            if dist_to_food <= 3:
                self.target_investigation_ticks += 1

                # Check which cardinal and diagonal entries to food are hazards
                cardinals = [(fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)]
                all_surrounding = [
                    (fr + dr, fc + dc)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                ]
                for c_pos in all_surrounding:
                    if c_pos in hazards:
                        self.perimeter_hazards_discovered.add(c_pos)

                # Thorough reconnaissance: must have probed for at least half of patience_ticks
                min_recon_done = self.target_investigation_ticks >= (self.patience_ticks // 2)

                # Condition 1: All surrounding entrances proven lethal AND thorough recon completed
                all_blocked = (len(self.perimeter_hazards_discovered) >= len(all_surrounding)) and min_recon_done

                # Condition 2: High frustration sustained across full patience limit
                frustration = self.compute_frustration()
                exhaustion = (self.target_investigation_ticks >= self.patience_ticks and frustration >= self.frustration_threshold)

                if all_blocked:
                    self.abandon_target = True
                    self.current_goal = "RETREAT_HOME"
                    self.abandon_reason = f"Exhaustive Reconnaissance Complete: All {len(all_surrounding)} surrounding perimeter tiles verified lethal after {self.target_investigation_ticks} steps. 100% death trap."
                elif exhaustion:
                    self.abandon_target = True
                    self.current_goal = "RETREAT_HOME"
                    self.abandon_reason = f"Patience Limit Reached: Persistently probed perimeter for {self.target_investigation_ticks} steps (Frustration: {frustration*100:.0f}%). Aborting to preserve life."

    def compute_frustration(self) -> float:
        """
        Quantifies spatial stagnation/oscillation:
        - Free movement across diverse tiles -> 0.0
        - Oscillating between 2 tiles (A <-> B) -> ~0.67
        - Stuck on 1 tile -> 1.0
        """
        if len(self.recent_positions) < 4:
            return 0.0

        window = self.recent_positions[-6:]
        unique_tiles = len(set(window))
        total_steps = len(window)

        # Frustration is 1.0 when unique_tiles is 1, and 0.0 when all tiles are unique
        raw_frustration = 1.0 - (float(unique_tiles - 1) / float(total_steps - 1))
        return float(np.clip(raw_frustration, 0.0, 1.0))

    def get_context_vector(
        self,
        pos: Tuple[int, int],
        target_food_dx: float,
        target_food_dy: float,
        hazard_dx: float,
        hazard_dy: float,
        energy: float = 1.0
    ) -> np.ndarray:
        """
        Encodes physical senses + internal memory state into a 6-dimensional feature vector.
        When target is abandoned, vector redirects agent back toward Home Base landmark (1, 1).
        """
        frustration = self.compute_frustration()

        if self.abandon_target:
            home_r, home_c = self.landmarks.get("home", (1, 1))
            eff_food_dx = float(np.clip((home_r - pos[0]) / 12.0, -1.0, 1.0))
            eff_food_dy = float(np.clip((home_c - pos[1]) / 12.0, -1.0, 1.0))
        else:
            eff_food_dx = target_food_dx
            eff_food_dy = target_food_dy

        return np.array([
            eff_food_dx,
            eff_food_dy,
            hazard_dx,
            hazard_dy,
            energy,
            frustration
        ], dtype=np.float32)

class RecurrentPolicyNetwork(nn.Module):
    """
    System 1: Recurrent Neural Core using a Gated Recurrent Unit (GRU).
    Computes updated temporal hidden state h_t and motor logits.
    """
    def __init__(self, in_features: int = 6, hidden_size: int = 16, out_actions: int = 4):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_actions = out_actions

        self.gru = nn.GRUCell(in_features, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_actions)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, in_features]
        h: [batch, hidden_size]
        Returns: (logits [batch, out_actions], next_h [batch, hidden_size])
        """
        next_h = self.gru(x, h)
        logits = self.fc_out(next_h)
        return logits, next_h

class RecurrentAgent:
    """
    Dual-Process Villager Agent:
    - System 1: Recurrent Policy Network (GRU)
    - System 2: Village Cognitive Memory Store
    """
    def __init__(
        self,
        agent_id: int,
        in_features: int = 6,
        hidden_size: int = 16,
        lr: float = 0.005,
        home_pos: Tuple[int, int] = (1, 1),
        patience_ticks: int = 50,
        frustration_threshold: float = 0.60
    ):
        self.id = agent_id
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.patience_ticks = patience_ticks
        self.frustration_threshold = frustration_threshold

        self.model = RecurrentPolicyNetwork(in_features, hidden_size, out_actions=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # System 2: Village Memory Store
        self.memory = VillageMemory(
            agent_id=agent_id,
            home_pos=home_pos,
            patience_ticks=patience_ticks,
            frustration_threshold=frustration_threshold
        )

        # Recurrent hidden state h_t (initialized to zeros)
        self.hx = torch.zeros(1, hidden_size, dtype=torch.float32)

        # Lifetime trajectory records for Reinforcement Learning (BPTT)
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

        # Evolution metrics
        self.fitness: float = 0.0
        self.food_eaten: int = 0
        self.hazards_hit: int = 0
        self.generation: int = 1
        self.lineage: str = f"RA{agent_id}"

    def reset_memory(self):
        """Reset temporal state at the start of a new episode / lifetime."""
        self.hx = torch.zeros(1, self.hidden_size, dtype=torch.float32)
        self.log_probs = []
        self.rewards = []
        self.memory.recent_positions = []

    def select_action(
        self,
        obs: np.ndarray,
        pos: Optional[Tuple[int, int]] = None,
        grid_size: int = 12,
        hazards: Optional[List[Tuple[int, int]]] = None,
        temperature: float = 0.4,
        deterministic: bool = False
    ) -> int:
        """
        Step forward through GRU, apply wall masking, reflex guard,
        and frustration-guided loop-breaking, then sample action.
        """
        state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # [1, in_features]

        # Step through GRU cell
        logits, self.hx = self.model(state_t, self.hx)
        logits = logits.squeeze(0) # [4]

        # Moves mapping: 0=Up, 1=Down, 2=Left, 3=Right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        if pos is not None:
            ar, ac = pos

            # 1. Border Wall Masking
            if ar == 0: logits[0] = -1e9
            if ar == grid_size - 1: logits[1] = -1e9
            if ac == 0: logits[2] = -1e9
            if ac == grid_size - 1: logits[3] = -1e9

            # 2. Reflex Guard (Pain receptor)
            if hazards:
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) in hazards:
                        logits[act_idx] -= 30.0

            # 3. Frustration-Powered Loop Breaking (System 2 Memory Interlock)
            frustration = self.memory.compute_frustration()
            if len(self.memory.recent_positions) >= 2:
                prev_pos = self.memory.recent_positions[-1]

                # Dynamic Momentum Penalty: scales up drastically when oscillating
                reversal_penalty = 0.5 + 4.0 * frustration
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) == prev_pos:
                        logits[act_idx] -= reversal_penalty

                # Active Loop Breakout: When trapped in an oscillation (>0.4 frustration),
                # penalize the entire oscillating axis to force an orthogonal flank escape!
                if frustration > 0.4 and len(self.memory.recent_positions) >= 4:
                    recent = self.memory.recent_positions[-4:]
                    rows = [r for r, c in recent]
                    cols = [c for r, c in recent]

                    # Horizontal ping-pong detected (same row, alternating columns)
                    if len(set(rows)) == 1 and len(set(cols)) <= 2:
                        # Penalize Left/Right oscillation, favor North/South flank
                        logits[2] -= 3.0 * frustration # Suppress Left
                        logits[3] -= 3.0 * frustration # Suppress Right

                    # Vertical ping-pong detected (same column, alternating rows)
                    elif len(set(cols)) == 1 and len(set(rows)) <= 2:
                        logits[0] -= 3.0 * frustration # Suppress Up
                        logits[1] -= 3.0 * frustration # Suppress Down

        if deterministic:
            return int(torch.argmax(logits).item())

        scaled_probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        dist = Categorical(scaled_probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def record_reward(self, reward: float):
        self.rewards.append(reward)
        self.fitness += reward

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

        # Gradient clipping to ensure stable BPTT through GRU
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Detach hidden state and clear episode buffers
        self.hx = self.hx.detach()
        self.log_probs = []
        self.rewards = []

    def clone_and_mutate(self, new_id: int, mutation_rate: float = 0.15, mutation_scale: float = 0.20) -> "RecurrentAgent":
        """
        Outer Loop: Genetic reproduction with Gaussian weight mutation.
        Clones GRU weights, linear projection head, and memory landmarks.
        """
        child = RecurrentAgent(
            agent_id=new_id,
            in_features=self.in_features,
            hidden_size=self.hidden_size,
            home_pos=self.memory.landmarks.get("home", (1, 1))
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
