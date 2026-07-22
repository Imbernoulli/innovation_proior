# TIER: invalid
# Deliberately malformed: restart_mode is not one of the allowed literals, and
# several numeric fields are out of their declared ranges. The evaluator's
# strict validator must reject this and score every instance 0.0.
import sys, json

def main():
    _ = json.load(sys.stdin)
    policy = {
        "T0": -5.0,                 # out of range (must be >= 0)
        "alpha": 1.7,                # out of range (must be in (0,1])
        "window": 0,                  # out of range (must be >= 5)
        "stagnation_window": -1,      # out of range (must be >= 1)
        "accept_floor": 3.0,          # out of range (must be in [0,1])
        "reheat_factor": 0.1,         # out of range (must be >= 1.0)
        "restart_mode": "teleport",   # not a valid literal
        "max_events": -3,             # out of range
    }
    print(json.dumps(policy))

if __name__ == "__main__":
    main()
