import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SENSOR_NAMES = [
    "Thirst",
    "Energy",
    "Health",
    "Water Dist X",
    "Water Dist Y",
    "Fire Danger",
    "Food Dist X",
    "Food Dist Y"
]

ACTION_NAMES = [
    "Rest",
    "Move North",
    "Move South",
    "Move West",
    "Move East",
    "Drink Water",
    "Forage / Eat"
]

class NeuralBrain(nn.Module):
    def __init__(self, in_features=8, h1=10, h2=8, out_features=7):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_features)

        self._initialize_primal_pathways()

    def _initialize_primal_pathways(self):
        """
        Initializes neural weights with primal survival pathways:
        - Thirst drives drinking & pathfinding to water
        - Fire danger drives flight & inhibits rest
        - Low energy drives foraging
        """
        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.15)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.15)
            nn.init.normal_(self.fc3.weight, mean=0.0, std=0.15)
            self.fc1.bias.fill_(0.0)
            self.fc2.bias.fill_(0.0)
            self.fc3.bias.fill_(0.0)

            # Pathway 1: Thirst (Sensor 0) -> H1-0 -> H2-0 -> Drink Water (Action 5)
            self.fc1.weight[0, 0] = 2.5
            self.fc2.weight[0, 0] = 2.5
            self.fc3.weight[5, 0] = 3.0

            # Pathway 2: Fire Danger (Sensor 5) -> H1-1 -> H2-1 -> Suppress Rest (Action 0), Excite Move
            self.fc1.weight[1, 5] = 3.5
            self.fc2.weight[1, 1] = 3.0
            self.fc3.weight[0, 1] = -4.0  # Strongly suppress Rest
            self.fc3.weight[1, 1] = 2.0   # Flee North/South
            self.fc3.weight[3, 1] = 2.0   # Flee West/East

            # Pathway 3: Water Seek North: Water Dist X < 0 (Sensor 3) -> H1-2 -> H2-2 -> Move North (Action 1)
            self.fc1.weight[2, 3] = -2.2
            self.fc1.weight[2, 0] = 1.2   # only when thirsty
            self.fc2.weight[2, 2] = 2.0
            self.fc3.weight[1, 2] = 2.5

            # Pathway 4: Water Seek South: Water Dist X > 0 (Sensor 3) -> H1-3 -> H2-3 -> Move South (Action 2)
            self.fc1.weight[3, 3] = 2.2
            self.fc1.weight[3, 0] = 1.2   # only when thirsty
            self.fc2.weight[3, 3] = 2.0
            self.fc3.weight[2, 3] = 2.5

            # Pathway 5: Water Seek West: Water Dist Y < 0 (Sensor 4) -> H1-4 -> H2-4 -> Move West (Action 3)
            self.fc1.weight[4, 4] = -2.2
            self.fc1.weight[4, 0] = 1.2
            self.fc2.weight[4, 4] = 2.0
            self.fc3.weight[3, 4] = 2.5

            # Pathway 6: Water Seek East: Water Dist Y > 0 (Sensor 4) -> H1-5 -> H2-5 -> Move East (Action 4)
            self.fc1.weight[5, 4] = 2.2
            self.fc1.weight[5, 0] = 1.2
            self.fc2.weight[5, 5] = 2.0
            self.fc3.weight[4, 5] = 2.5

            # Pathway 7: Low Energy (Sensor 1 negative) -> H1-6 -> H2-6 -> Forage/Eat (Action 6)
            self.fc1.weight[6, 1] = -2.0
            self.fc2.weight[6, 6] = 2.0
            self.fc3.weight[6, 6] = 2.5

    def forward_with_trace(self, input_vec: np.ndarray):
        """
        Runs the neural network and returns full telemetry:
        - Layer-by-layer activations
        - Synaptic weights (layer_idx, from_neuron, to_neuron, weight)
        - Action decision & probabilities
        """
        x_tensor = torch.tensor(input_vec, dtype=torch.float32)

        # Forward pass tracking intermediate activations
        z1 = self.fc1(x_tensor)
        a1 = torch.tanh(z1)

        z2 = self.fc2(a1)
        a2 = torch.tanh(z2)

        logits = self.fc3(a2)
        probs = F.softmax(logits, dim=-1)

        action_idx = int(torch.argmax(probs).item())

        # Extract weights for visualization
        w01 = self.fc1.weight.detach().numpy().tolist() # shape (h1, in_features)
        w12 = self.fc2.weight.detach().numpy().tolist() # shape (h2, h1)
        w23 = self.fc3.weight.detach().numpy().tolist() # shape (out, h2)

        # Format layer nodes and values for frontend
        telemetry = {
            "layers": [
                {
                    "name": "Sensory Layer",
                    "nodes": [
                        {"name": SENSOR_NAMES[i], "val": round(float(input_vec[i]), 3)}
                        for i in range(len(input_vec))
                    ]
                },
                {
                    "name": "Hidden Layer 1",
                    "nodes": [
                        {"name": f"H1-{j}", "val": round(float(a1[j].item()), 3)}
                        for j in range(len(a1))
                    ]
                },
                {
                    "name": "Hidden Layer 2",
                    "nodes": [
                        {"name": f"H2-{k}", "val": round(float(a2[k].item()), 3)}
                        for k in range(len(a2))
                    ]
                },
                {
                    "name": "Action Motor Layer",
                    "nodes": [
                        {
                            "name": ACTION_NAMES[m],
                            "val": round(float(probs[m].item()), 3),
                            "chosen": (m == action_idx)
                        }
                        for m in range(len(probs))
                    ]
                }
            ],
            "weights": [
                w01, # from Sensory to H1
                w12, # from H1 to H2
                w23  # from H2 to Action
            ],
            "chosen_action": action_idx,
            "action_name": ACTION_NAMES[action_idx]
        }

        return action_idx, telemetry
