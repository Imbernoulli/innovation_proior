#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE instance for the low-glare lighting problem.

testId 1..10 is a small->large difficulty ladder. Everything is seeded by testId
only, so generation is fully deterministic. Output:
    line 1: N
    line 2: N ceilings u_i (3 decimals)
"""
import sys, random


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(1000 + tid)

    # N grows with the ladder: 30, 40, ..., 120 (medium scale).
    N = 20 + 10 * tid

    # Ceiling profile. A mix of regimes so no single fixed shape is optimal:
    #  - later tests get more irregular / spiky ceilings (adversarial),
    #  - a handful of tightly-capped segments force non-uniform shapes.
    u = []
    for i in range(N):
        base = rng.uniform(0.3, 1.5)
        if tid >= 6 and rng.random() < 0.25:
            # occasional very tight cap (regulatory dark zone)
            base = rng.uniform(0.05, 0.3)
        if tid >= 8 and rng.random() < 0.15:
            # occasional bright allowance
            base = rng.uniform(1.5, 2.0)
        u.append(round(base, 3))

    out = [str(N), " ".join("%.3f" % x for x in u)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
