import os
import sys
import time
import json

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def clear_screen():
    print("\033[H\033[J", end="")

def format_val(val: float, width: int = 5) -> str:
    color = GREEN if val > 0.05 else (RED if val < -0.05 else DIM)
    sign = "+" if val > 0 else ("-" if val < 0 else " ")
    return f"{color}{sign}{abs(val):.2f}{RESET}"

def render_meter(prob: float, width: int = 10) -> str:
    filled = int(prob * width)
    return f"[{'█' * filled}{' ' * (width - filled)}]"

def main():
    state_file = "live_state.json"
    last_mod_time = 0

    clear_screen()
    print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
    print(f"{CYAN}{BOLD}             🧠 NEURAL BRAIN LIVE MONITOR (TERMINAL 2)                  {RESET}")
    print(f"{CYAN}{BOLD}========================================================================{RESET}")
    print(f"{YELLOW}Waiting for training to start in Terminal 1 (`uv run python train.py`)...{RESET}\n")

    while True:
        try:
            if os.path.exists(state_file):
                mtime = os.path.getmtime(state_file)
                if mtime > last_mod_time:
                    last_mod_time = mtime
                    with open(state_file, "r") as f:
                        data = json.load(f)

                    clear_screen()
                    gen = data.get("gen", 1)
                    lineage = data.get("lineage", "A0")
                    best_food = data.get("best_food", 0)
                    avg_food = data.get("avg_food", 0.0)
                    sensors = data.get("sensors", [])
                    hidden = data.get("hidden", [])
                    actions = data.get("actions", [])
                    chosen = data.get("chosen_action", "")
                    w1 = data.get("weights_fc1", [])
                    w2 = data.get("weights_fc2", [])

                    print(f"{CYAN}{BOLD}╔════════════════════════════════════════════════════════════════════════════╗{RESET}")
                    print(f"{CYAN}{BOLD}║                 🧠 CHAMPION NEURAL BRAIN TELEMETRY                         ║{RESET}")
                    print(f"{CYAN}{BOLD}╠════════════════════════════════════════════════════════════════════════════╣{RESET}")
                    print(f"║ {BOLD}Generation:{RESET} {CYAN}{gen:2d}{RESET}  │ {BOLD}Top Lineage:{RESET} {MAGENTA}{lineage:<14}{RESET} │ {BOLD}Best Food:{RESET} {YELLOW}{best_food:2d}{RESET} │ {BOLD}Avg Food:{RESET} {GREEN}{avg_food:.1f}{RESET}  ║")
                    print(f"{CYAN}{BOLD}╚════════════════════════════════════════════════════════════════════════════╝{RESET}\n")

                    # --- ARCHITECTURE LAYER ACTIVATIONS ---
                    print(f"{BOLD}[ 1. SENSORS ]{RESET}            {BOLD}[ 2. HIDDEN LAYER (16) ]{RESET}             {BOLD}[ 3. MOTOR ACTIONS ]{RESET}")

                    lines = []
                    num_rows = max(len(sensors), len(actions), 8)
                    for r in range(num_rows):
                        # Col 1: Sensory input
                        if r < len(sensors):
                            s = sensors[r]
                            s_str = f"{s['name']:<10} {format_val(s['val'])} ──►"
                        else:
                            s_str = " " * 20

                        # Col 2: Hidden neuron pair
                        h_idx1 = r
                        h_idx2 = r + 8
                        h1_str = f"H{h_idx1:02d}:{format_val(hidden[h_idx1])}" if h_idx1 < len(hidden) else ""
                        h2_str = f"H{h_idx2:02d}:{format_val(hidden[h_idx2])}" if h_idx2 < len(hidden) else ""
                        h_str = f"{h1_str}  {h2_str} ──►"

                        # Col 3: Motor action
                        if r < len(actions):
                            act = actions[r]
                            is_chosen = (act["name"] == chosen)
                            marker = f"{YELLOW}{BOLD}👉{RESET}" if is_chosen else "  "
                            bar = render_meter(act['prob'], 8)
                            act_color = YELLOW if is_chosen else RESET
                            act_str = f"{marker} {act_color}{act['name']:<10}{RESET} {bar} {act['prob']*100:4.1f}%"
                        else:
                            act_str = ""

                        lines.append(f"  {s_str}   {h_str:<23}  {act_str}")

                    print("\n".join(lines))

                    # --- SYNAPSE WEIGHTS SAMPLE (Sensory -> First 8 Hidden) ---
                    print(f"\n{BOLD}[ 4. SYNAPSE WEIGHTS (Sensory Inputs ──► Hidden Neurons H00-H07) ]{RESET}")
                    print(f"{DIM}  {'Sensor':<10}  {'H00':>5}  {'H01':>5}  {'H02':>5}  {'H03':>5}  {'H04':>5}  {'H05':>5}  {'H06':>5}  {'H07':>5}{RESET}")
                    if len(w1) >= 8:
                        sensor_labels = ["Food dx", "Food dy", "Hazard dx", "Hazard dy", "Energy"]
                        for si, s_name in enumerate(sensor_labels):
                            row_weights = [format_val(w1[hi][si]) for hi in range(min(8, len(w1)))]
                            print(f"  {s_name:<10}  " + "  ".join(row_weights))

                    # --- MOTOR WEIGHTS (Hidden -> Motor Actions) ---
                    print(f"\n{BOLD}[ 5. MOTOR WEIGHTS (Hidden Neurons H00-H07 ──► Actions) ]{RESET}")
                    print(f"{DIM}  {'Action':<10}  {'H00':>5}  {'H01':>5}  {'H02':>5}  {'H03':>5}  {'H04':>5}  {'H05':>5}  {'H06':>5}  {'H07':>5}{RESET}")
                    action_labels = ["Move Up", "Move Down", "Move Left", "Move Right"]
                    if len(w2) >= 4:
                        for ai, a_name in enumerate(action_labels):
                            row_weights = [format_val(w2[ai][hi]) for hi in range(min(8, len(w2[ai])))]
                            print(f"  {a_name:<10}  " + "  ".join(row_weights))

                    print(f"\n{DIM}Streaming live updates... (Press Ctrl+C to exit){RESET}")

            time.sleep(0.08)
        except KeyboardInterrupt:
            print("\nExiting monitor.")
            sys.exit(0)
        except Exception:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
