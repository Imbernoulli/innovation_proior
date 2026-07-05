#!/usr/bin/env python3
# gen.py -- prints ONE instance of the drone-delivery-swarm safety-bubble packing problem.
# Usage: python3 gen.py <testId>   (testId = 1..10, difficulty ladder small->large)
# Deterministic: all randomness seeded from testId only.
#
# Setting: a circular delivery airspace (a disk of radius R centred at (cx,cy)) contains K
# fixed circular RESTRICTED-AIRSPACE / no-fly zones. We must position N drone "safety bubbles"
# (circles) inside the airspace, each avoiding every no-fly zone and every other bubble, so that
# the TOTAL bubble radius (sum of radii) is as large as possible.
#
# Instance format (stdout):
#   line 1: N cx cy R K
#   next K lines: zx zy zr   (no-fly-zone circle: centre (zx,zy), radius zr)
import sys, random


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    rng = random.Random(133000 + tid)

    # difficulty ladder (LARGE scale): N grows 20 -> 56; airspace grows; more no-fly zones.
    N = 20 + (tid - 1) * 4                 # 20,24,28,...,56
    R = round(1.0 + 0.12 * (tid - 1), 4)   # 1.00 .. 2.08
    cx = round(R, 4)                       # centre so the airspace sits in [0,2R]x[0,2R]
    cy = round(R, 4)
    K = (tid + 1) // 2                      # 1,1,2,2,3,3,4,4,5,5 no-fly zones

    # place K non-overlapping no-fly zones well inside the airspace, leaving free room.
    zones = []
    tries = 0
    while len(zones) < K and tries < 6000:
        tries += 1
        zr = round(rng.uniform(0.10 * R, 0.22 * R), 4)
        # sample a centre whose whole zone lies inside the airspace disk with margin
        ang = rng.uniform(0, 6.283185307179586)
        rad = rng.uniform(0.0, max(0.0, R - zr - 0.20 * R))
        zx = round(cx + rad * __import__("math").cos(ang), 4)
        zy = round(cy + rad * __import__("math").sin(ang), 4)
        ok = True
        for (ax, ay, ar) in zones:
            d = ((zx - ax) ** 2 + (zy - ay) ** 2) ** 0.5
            if d < zr + ar + 0.05 * R:   # keep a gap between zones
                ok = False
                break
        if ok:
            zones.append((zx, zy, zr))

    out = ["%d %s %s %s %d" % (N, repr(cx), repr(cy), repr(R), len(zones))]
    for (zx, zy, zr) in zones:
        out.append("%s %s %s" % (repr(zx), repr(zy), repr(zr)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
