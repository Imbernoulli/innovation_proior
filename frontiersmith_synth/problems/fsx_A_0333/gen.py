#!/usr/bin/env python3
# gen.py <testId> : print ONE instance of "Metro Catchment Packing" to stdout.
# Instance = N fixed platform locations inside the unit-square city.
# Difficulty ladder (testId 1..10): N grows and platforms get denser/harder.
# Fully deterministic: all randomness seeded from testId only.
import sys, random


def gen(test_id):
    rng = random.Random(1000 + 97 * test_id)

    # N grows with difficulty: 8 -> 26
    N = 8 + 2 * (test_id - 1)

    # Higher testId -> platforms packed into a smaller sub-region (denser, less
    # headroom above the baseline), and a smaller enforced separation.
    frac = min(1.0, 0.35 + 0.07 * test_id)      # side of usable sub-region
    lo = 0.5 - frac / 2.0
    hi = 0.5 + frac / 2.0
    margin = 0.015                               # keep platforms off the wall
    lo = max(lo, margin)
    hi = min(hi, 1.0 - margin)

    min_sep = 0.5 * frac / (N ** 0.5) * 0.9      # rejection-sampling separation

    pts = []
    attempts = 0
    while len(pts) < N and attempts < 20000:
        attempts += 1
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        ok = True
        for (px, py) in pts:
            if (px - x) ** 2 + (py - y) ** 2 < min_sep ** 2:
                ok = False
                break
        if ok:
            pts.append((x, y))

    # fallback: relax separation if rejection stalled
    while len(pts) < N:
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        dup = any(abs(px - x) < 1e-4 and abs(py - y) < 1e-4 for px, py in pts)
        if not dup:
            pts.append((x, y))

    out = [str(N)]
    for (x, y) in pts:
        out.append("%.9f %.9f" % (x, y))
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <testId>\n")
        sys.exit(1)
    sys.stdout.write(gen(int(sys.argv[1])))


if __name__ == "__main__":
    main()
