# TIER: trivial
# Do nothing clever: T=0 forever (pure single-bit-flip hill-climbing). Every trap
# block gets driven straight to its u=0 consolation value and sits there.
import sys, json

def main():
    inst = json.load(sys.stdin)
    steps = inst["steps"]
    policy = {
        "T0": 0.0,
        "alpha": 1.0,
        "window": steps,
        "stagnation_window": steps + 1,
        "accept_floor": 0.0,
        "reheat_factor": 1.0,
        "restart_mode": "reheat",
        "max_events": 0,
    }
    print(json.dumps(policy))

if __name__ == "__main__":
    main()
