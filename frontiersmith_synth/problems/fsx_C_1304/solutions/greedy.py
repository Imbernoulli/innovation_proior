# TIER: greedy
"""Greedy candidate: the obvious first attempt. Track moisture with a SINGLE
bucket (no surface/root split, no percolation lag), and each day top it up to
one FIXED target fraction of root-zone capacity -- the same target every day,
regardless of crop stage, tomorrow's forecast, or today's electricity price.
This is exactly "irrigate to a fixed moisture target," the standard approach
the problem statement calls out as wasteful."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    p = inst["params"]
    Cr = p["Cr"]
    maxirr = p["max_irrig_per_day"]
    rain = inst["rain"]
    et = inst["et"]
    theta_target = 0.65

    M = p["R0"] + p["S0"]
    irr = []
    for t in range(T):
        M += rain[t]
        add = max(0.0, min(maxirr, theta_target * Cr - M))
        M += add
        M -= et[t]
        if M < 0.0:
            M = 0.0
        if M > Cr:
            M = Cr
        irr.append(add)

    print(json.dumps({"irrig": irr}))


if __name__ == "__main__":
    main()
