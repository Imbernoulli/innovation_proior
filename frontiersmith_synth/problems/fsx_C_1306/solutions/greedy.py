# TIER: greedy
"""Obvious first-instinct policy: keep the reservoir near a fixed high
target level for supply security, reacting only to TODAY's forecast value
(trusted at face value, no skill weighting, no lookahead beyond the
current day). This is exactly the "keep it full" rule-curve trap: it never
proactively drains ahead of a pulse it cannot yet see on today's forecast,
and even when today's forecast does show a spike it can only react with a
single day's worth of release-rate headroom."""
import sys, json

TARGET_FRAC = 0.92


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]; C = inst["capacity"]; Dmin = inst["dead_storage"]
    Rmax = inst["r_max"]; demand = inst["demand"]; forecast = inst["forecast"]

    target = TARGET_FRAC * C
    S_est = inst["init_storage"]
    release = []
    for t in range(T):
        inflow_guess = forecast[t]           # trust today's forecast at face value
        proj = S_est + inflow_guess
        rel = min(demand[t], max(0.0, proj - Dmin))
        proj_after_demand = proj - rel
        if proj_after_demand > target:
            rel += proj_after_demand - target
        rel = max(0.0, min(rel, Rmax, proj - Dmin))
        release.append(rel)
        S_est = proj - rel
        if S_est < Dmin: S_est = Dmin
        if S_est > C: S_est = C

    print(json.dumps({"release": release}))


if __name__ == "__main__":
    main()
