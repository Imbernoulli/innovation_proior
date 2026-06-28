#!/usr/bin/env python3
"""Instance generator for ale-15 "Continuous Facility Layout".

Usage:  python3 gen.py SEED  > instance.txt

Instance format (stdout):
    line 1:  N W H
    next N lines:  w h        (integer width and height of rectangle i, 1-based)

Semantics: N axis-aligned facility footprints (rectangles) must be placed inside
a W x H container so that the bottom-left corner of every rectangle has integer
coordinates and the whole rectangle stays inside the container. We minimise an
energy = OVERLAP_WEIGHT * (total pairwise overlap area)
       + DISPERSION_WEIGHT * (sum of squared distance of each rect centre from
                              the mean centre).
The container is sized so that the rectangles CAN be laid out with little or no
overlap (total rectangle area is a controlled fraction of the container area),
but a careless placement overlaps heavily, so the layout problem is non-trivial.

Everything is a deterministic function of SEED.
"""
import sys
import random


def gen(seed: int):
    rng = random.Random(seed * 2654435761 + 911)

    # Number of rectangles. Moderate N so an O(N^2) overlap recompute is the
    # lever the spatial-hash innovation removes.
    N = rng.randint(120, 200)

    # Rectangle sizes: a mix of small and large footprints.
    rects = []
    total_area = 0
    for _ in range(N):
        # roughly log-uniform side lengths in [4, 60]
        w = int(round(4 * (1.5 ** rng.uniform(0, 6.5))))
        h = int(round(4 * (1.5 ** rng.uniform(0, 6.5))))
        w = max(4, min(120, w))
        h = max(4, min(120, h))
        rects.append((w, h))
        total_area += w * h

    # Container: choose a near-square container whose area is ~ total_area / fill
    # with a packing fraction "fill" in (0,1). fill < 1 leaves slack so a good
    # layout can avoid almost all overlap; the tension is dispersion vs overlap.
    fill = rng.uniform(0.35, 0.55)
    area = total_area / fill
    # aspect ratio near 1
    aspect = rng.uniform(0.8, 1.25)
    Wf = (area * aspect) ** 0.5
    Hf = area / Wf
    W = int(round(Wf))
    H = int(round(Hf))

    # Make sure every single rectangle fits in the container with room to move.
    maxw = max(r[0] for r in rects)
    maxh = max(r[1] for r in rects)
    W = max(W, maxw + 10)
    H = max(H, maxh + 10)

    return N, W, H, rects


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py SEED\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    N, W, H, rects = gen(seed)
    out = [f"{N} {W} {H}"]
    for (w, h) in rects:
        out.append(f"{w} {h}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
