import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, in_features: int = 5, hidden_size: int = 16, out_actions: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        logits = self.fc2(h)
        return logits

class Agent:
    def __init__(self, agent_id: int, in_features: int = 5, hidden_size: int = 16, lr: float = 0.01):
        self.id = agent_id
        self.model = PolicyNetwork(in_features, hidden_size, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Lifetime memory for Policy Gradient (RL)
        self.log_probs = []
        self.rewards = []

        # Fitness metrics for Evolution
        self.fitness = 0.0
        self.food_eaten = 0
        self.hazards_hit = 0
        self.generation = 1
        self.lineage = f"A{agent_id}"

    def select_action(
        self,
        obs: np.ndarray,
        pos: tuple[int, int] | None = None,
        grid_size: int = 12,
        hazards: list[tuple[int, int]] | None = None,
        prev_pos: tuple[int, int] | None = None,
        temperature: float = 0.5,
        deterministic: bool = False
    ) -> int:
        """Forward pass with border wall masking, reflex guard, momentum, and temperature sampling."""
        state_t = torch.tensor(obs, dtype=torch.float32)
        logits = self.model(state_t)

        if pos is not None:
            ar, ac = pos
            # 1. Border wall masking
            if ar == 0: logits[0] = -1e9
            if ar == grid_size - 1: logits[1] = -1e9
            if ac == 0: logits[2] = -1e9
            if ac == grid_size - 1: logits[3] = -1e9

            moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

            # 2. Reflex Guard (Pain receptor - heavily penalize stepping onto hazards)
            if hazards:
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) in hazards:
                        logits[act_idx] -= 30.0

            # 3. Momentum / Anti-Ping-Pong Suppressor (mild bias to break 2-tile loops)
            if prev_pos is not None:
                for act_idx, (dr, dc) in moves.items():
                    if (ar + dr, ac + dc) == prev_pos:
                        logits[act_idx] -= 0.5

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
        Inner Loop: Reinforcement Learning update (REINFORCE).
        Updates the agent's policy based on its own lived experiences.
        """
        if len(self.rewards) == 0:
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
        self.optimizer.step()

        # Clear episode memory
        self.log_probs = []
        self.rewards = []

    def clone_and_mutate(self, new_id: int, mutation_rate: float = 0.15, mutation_scale: float = 0.20) -> "Agent":
        """
        Outer Loop: Evolutionary reproduction with genetic weight mutation.
        """
        child = Agent(new_id)
        child.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        child.generation = self.generation + 1
        child.lineage = f"{self.lineage}->G{child.generation}"

        # Mutate a fraction of weights with Gaussian noise
        with torch.no_grad():
            for param in child.model.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_scale
                param.add_(mask * noise)

        return child
