#!/usr/bin/env python3
# gen.py -- prints ONE instance of the data-center cooling diffuser-packing problem.
# Usage: python3 gen.py <testId>   (testId = 1..10, difficulty ladder small->large)
# Deterministic: all randomness seeded from testId only.
#
# Instance format (stdout):
#   line 1: N W H K
#   next K lines: x0 y0 x1 y1   (axis-aligned server-rack keep-out rectangle)
import sys, random


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    rng = random.Random(90000 + tid)

    # difficulty ladder: N grows 10 -> ~37, room grows / gets more racks.
    N = 10 + (tid - 1) * 3            # 10,13,16,...,37
    W = round(1.0 + 0.05 * (tid - 1), 4)   # 1.00 .. 1.45
    H = round(0.60 + 0.04 * (tid - 1), 4)  # 0.60 .. 0.96
    K = (tid + 1) // 2               # 1,1,2,2,3,3,4,4,5,5 racks

    # place K non-overlapping racks inside the room, leaving generous free space.
    racks = []
    tries = 0
    while len(racks) < K and tries < 4000:
        tries += 1
        rw = round(rng.uniform(0.08, 0.20), 4)
        rh = round(rng.uniform(0.08, 0.20), 4)
        x0 = round(rng.uniform(0.05, W - rw - 0.05), 4)
        y0 = round(rng.uniform(0.05, H - rh - 0.05), 4)
        x1 = round(x0 + rw, 4)
        y1 = round(y0 + rh, 4)
        ok = True
        for (ax0, ay0, ax1, ay1) in racks:
            # keep a small gap between racks
            if not (x1 < ax0 - 0.03 or ax1 < x0 - 0.03 or
                    y1 < ay0 - 0.03 or ay1 < y0 - 0.03):
                ok = False
                break
        if ok:
            racks.append((x0, y0, x1, y1))

    out = ["%d %s %s %d" % (N, repr(W), repr(H), len(racks))]
    for (x0, y0, x1, y1) in racks:
        out.append("%s %s %s %s" % (repr(x0), repr(y0), repr(x1), repr(y1)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
