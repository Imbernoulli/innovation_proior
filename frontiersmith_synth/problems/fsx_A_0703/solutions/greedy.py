# TIER: greedy
# Keep the reservoir full for water security: react to a FIXED target of 80% of
# capacity, releasing only the excess above it (capped by Rmax), and never a drop more.
# This looks safe on paper -- the level is never allowed to run low -- but it is
# completely oblivious to the fact that the inflow record mixes a slow seasonal drift
# with sudden storm bursts: because the target never moves, the reservoir is usually
# sitting near 80% full, so any storm whose excess exceeds Rmax above that slack
# overtops it on the very day it lands. It never looks ahead in the given inflow array
# to preemptively lower the level before a storm it could easily have seen coming.
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]; cap = inst["cap"]; Rmax = inst["Rmax"]
    inflow = inst["inflow"]
    target = 0.8 * cap
    L = inst["L0"]
    out = []
    for t in range(T):
        avail = L + inflow[t]
        rel = max(0.0, avail - target)
        rel = min(rel, Rmax)
        out.append(rel)
        raw = avail - rel
        L = cap if raw > cap else raw
    print(json.dumps({"releases": out}))


if __name__ == "__main__":
    main()
