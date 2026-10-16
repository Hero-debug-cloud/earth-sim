import json
from pathlib import Path
import torch
from agent import PolicyNetwork

def inspect():
    ckpt_dir = Path("checkpoints")
    champ_path = ckpt_dir / "champion.pt"
    history_path = ckpt_dir / "history.json"

    if not champ_path.exists():
        print("No champion.pt found yet. Run `uv run python train.py` to train and save one.")
        return

    print("=" * 60)
    print("🧠 INSPECTING SAVED CHAMPION BRAIN")
    print("=" * 60)

    # Load history metadata
    if history_path.exists():
        with open(history_path, "r") as f:
            h = json.load(f)
        print(f"• Dominant Lineage:  {h.get('dominant_lineage')}")
        print(f"• Generations Run:   {h.get('total_generations')}")
        print(f"• Peak Food Record:  {h.get('peak_score')} foods in 1 lifetime")
        print(f"• Fitness Progress:  Gen 1 ({h['history_avg'][0]}) -> Final ({h['history_avg'][-1]})")

    # Load neural weights into model
    weights = torch.load(champ_path, weights_only=True)
    model = PolicyNetwork(in_features=5, hidden_size=16, out_actions=4)
    model.load_state_dict(weights)
    model.eval()

    print("\n📦 PyTorch Neural Network Weights Loaded:")
    for name, param in model.named_parameters():
        print(f"  • {name:<12} Shape: {list(param.shape)}  Mean: {param.mean().item():+.3f}  Std: {param.std().item():.3f}")

    print("=" * 60)

if __name__ == "__main__":
    inspect()
