#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance of the cold-chain depot dispersion problem.

testId 1..10 is a difficulty ladder: more depots and more warm-zones as testId grows.
All randomness is seeded from testId only (bit-for-bit deterministic).

Instance format (stdin of the solver):
    n
    c
    <c lines: cx cy r>         # forbidden "warm zone" open disks

The generator GUARANTEES that the horizontal baseline row
    (x_i, 0.5),  x_i = (i+0.5)/n
is disk-free (every warm-zone centre keeps |cy-0.5| >= r + 0.03), so the
checker's baseline construction is always feasible.
"""
import sys, random


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(20260701 + 7919 * tid)

    n = 12 + tid                      # 13 .. 22 depots
    c = min(6, 2 + tid // 2)          # 2 .. 6 warm zones

    disks = []
    tries = 0
    while len(disks) < c and tries < 5000:
        tries += 1
        r = round(rng.uniform(0.04, 0.10), 4)
        cx = round(rng.uniform(0.15, 0.85), 4)
        # keep the disk clear of the midline row: |cy - 0.5| >= r + 0.03
        lo, hi = 0.10, 0.5 - r - 0.03
        if rng.random() < 0.5:
            lo, hi = 0.5 + r + 0.03, 0.90
        if hi <= lo:
            continue
        cy = round(rng.uniform(lo, hi), 4)
        # avoid near-duplicate / heavily-overlapping disks
        ok = True
        for (px, py, pr) in disks:
            if (px - cx) ** 2 + (py - cy) ** 2 < (pr + r) ** 2 * 0.25:
                ok = False
                break
        if ok:
            disks.append((cx, cy, r))

    out = [str(n), str(len(disks))]
    for (cx, cy, r) in disks:
        out.append("%.4f %.4f %.4f" % (cx, cy, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
