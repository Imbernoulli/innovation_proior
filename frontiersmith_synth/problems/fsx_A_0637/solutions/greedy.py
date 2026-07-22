# TIER: greedy
"""Obvious "textbook safety margin" heuristic: assume a single generic buffer
(18% of each zone's carrying capacity) is a safe floor everywhere, and harvest
everything above it every step. This is a reasonable-looking recipe -- it beats
doing nothing by a wide margin -- but it (a) sits far from each zone's true
max-growth point K/2, so it leaves a lot of sustainable surplus unharvested even
when it's safe, and (b) uses ONE static floor for every zone even though the
problem only guarantees collapse thresholds stay below 45% of K: any zone whose
hidden threshold happens to sit above 18% of K gets driven straight through its
invisible floor and collapses, killing that zone's future yield for the rest of
the episode."""
import sys, json

FLOOR_FRAC = 0.18


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    zones = inst["zones"]
    Z = len(zones)
    S = [z["S0"] for z in zones]
    harvest = []
    for t in range(T):
        row = []
        for zi in range(Z):
            K = zones[zi]["K"]; r = zones[zi]["r"]
            floor = FLOOR_FRAC * K
            h = max(0.0, S[zi] - floor)
            row.append(h)
            S_after = S[zi] - h
            growth = r * S_after * (1.0 - S_after / K)   # candidate's own (optimistic) model
            S[zi] = max(0.0, min(K, S_after + growth))
        harvest.append(row)
    print(json.dumps({"harvest": harvest}))


if __name__ == "__main__":
    main()
