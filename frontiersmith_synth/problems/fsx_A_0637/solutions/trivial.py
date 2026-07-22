# TIER: trivial
"""Trivial reference: skim a tiny fixed fraction of the current (self-tracked)
stock from every zone every step. Never targets any level, never reasons about
regeneration or collapse -- just a constant-rate cull. Safe (skim is small) but
leaves almost all of the sustainable surplus on the table."""
import sys, json

SKIM = 0.015


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    zones = inst["zones"]
    Z = len(zones)
    S = [z["S0"] for z in zones]          # candidate's own (approximate) tracking
    harvest = []
    for t in range(T):
        row = []
        for zi in range(Z):
            h = SKIM * S[zi]
            row.append(h)
            # candidate's naive self-model of what happens next (no collapse knowledge)
            K = zones[zi]["K"]; r = zones[zi]["r"]
            S_after = S[zi] - h
            growth = r * S_after * (1.0 - S_after / K)
            S[zi] = max(0.0, min(K, S_after + growth))
        harvest.append(row)
    print(json.dumps({"harvest": harvest}))


if __name__ == "__main__":
    main()
