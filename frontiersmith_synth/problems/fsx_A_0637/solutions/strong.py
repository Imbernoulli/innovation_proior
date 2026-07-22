# TIER: strong
"""Insight: hold every zone at its OWN maximum-growth point (K/2) and harvest
only the surplus above it -- treating K/2 as a hard floor no smarter than the
worst-case collapse-threshold guarantee (thresholds never exceed 0.45*K, so
0.5*K is provably safe in EVERY zone, regardless of the exact hidden value).

Per zone, per step: harvest = max(0, S - K/2).
  - If S is already above K/2 (a zone that started rich), harvest the excess
    immediately -- a one-time bonus catch that also moves the zone onto its
    steady operating point.
  - If S is below K/2 (a zone that started depleted), harvest NOTHING and let
    it regenerate; regeneration is fastest precisely as it approaches K/2, so
    this converges quickly without ever again touching the invisible floor.
  - Once a zone is at/near K/2, harvesting the surplus every step captures
    close to that zone's per-step maximum sustainable yield (r*K/4), forever,
    with the stock never approaching the region where collapse thresholds
    could plausibly live."""
import sys, json


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
            target = 0.5 * K
            h = max(0.0, S[zi] - target)
            row.append(h)
            S_after = S[zi] - h
            growth = r * S_after * (1.0 - S_after / K)
            S[zi] = max(0.0, min(K, S_after + growth))
        harvest.append(row)
    print(json.dumps({"harvest": harvest}))


if __name__ == "__main__":
    main()
