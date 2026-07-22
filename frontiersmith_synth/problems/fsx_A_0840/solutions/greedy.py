# TIER: greedy
# The "obvious" textbook recipe: pick one plausible initial temperature from the
# only structural hint that's public (the largest block size), then decay it
# ONCE, smoothly, across the whole step budget down to a near-zero final
# temperature. No restarts, no reaction to what actually happens during the
# run. This is exactly the single monotone geometric cooling curve every SA
# tutorial teaches first -- it has no way to notice, mid-run, that its initial
# guess was wrong for THIS instance's hidden reward scale, and no way to try
# again with a different guess.
import sys, json

def main():
    inst = json.load(sys.stdin)
    blocks = inst["blocks"]
    steps = inst["steps"]
    T0_max = inst["T0_max"]

    max_block = max(b["size"] for b in blocks) if blocks else 4
    T0 = min(T0_max, 0.65 * max_block)
    Tf = 0.02
    window = 300
    n_windows = max(1, steps // window)
    alpha = (Tf / T0) ** (1.0 / n_windows) if T0 > 0 else 1.0

    policy = {
        "T0": T0,
        "alpha": alpha,
        "window": window,
        "stagnation_window": steps + 1,   # never counts as "stagnant" -> never reheats
        "accept_floor": 0.0,              # never counts as "stuck" -> never reheats
        "reheat_factor": 1.0,
        "restart_mode": "reheat",
        "max_events": 0,
    }
    print(json.dumps(policy))

if __name__ == "__main__":
    main()
