#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of the algae-fuel-pond-fleet problem to stdout.
Deterministic: all randomness is seeded from testId only.
"""
import sys
import random


def main():
    test_id = int(sys.argv[1])
    rnd = random.Random(1000003 * test_id + 7919)

    # Difficulty / contention ladder. Later tests (idx>=3, i.e. testId>=4) are the
    # "trap" cases: P grows, the horizon grows only mildly, and the per-step feed
    # cap barely grows with P, so naive simultaneous growth starves every pond.
    p_list = [3, 4, 5, 7, 7, 9, 9, 11, 12, 13]
    t_list = [12, 15, 18, 20, 22, 25, 28, 32, 36, 40]
    cbase_list = [1.60, 1.40, 1.30, 1.05, 0.95, 0.85, 0.78, 0.70, 0.66, 0.62]

    idx = test_id - 1
    if idx < 0 or idx >= len(p_list):
        idx = idx % len(p_list)
    P = p_list[idx]
    T = t_list[idx]
    # Cap scales sub-linearly with P: enough to keep a small handful of ponds
    # busy, nowhere near enough for all P to grow at once without starving.
    C = round(cbase_list[idx] * (1.0 + 0.22 * (P - 3)), 4)

    lines = [f"{P} {T}", f"{C:.4f}"]
    for _p in range(P):
        a = round(rnd.uniform(1.5, 4.0), 4)      # growth coefficient
        b0 = round(rnd.uniform(1.0, 3.0), 4)     # initial biomass
        e0 = round(rnd.uniform(0.6, 1.4), 4)     # base harvest efficiency
        decay = round(rnd.uniform(0.72, 0.98), 4)  # per-step age-decay factor
        # activation threshold: a pond only converts feed into biomass on a step
        # where it receives AT LEAST tau of the line's flow; thinner shares keep
        # the culture alive (no loss) but add zero biomass that step.
        tau = round(C * rnd.uniform(0.30, 0.62), 4)
        lines.append(f"{a:.4f} {b0:.4f} {e0:.4f} {decay:.4f} {tau:.4f}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
