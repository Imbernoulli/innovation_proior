#!/usr/bin/env python3
"""gen.py <testId> -- print ONE instance ("d M") for the sentinel-grid
low-discrepancy pointset problem. testId 1..10 is a difficulty ladder
(small 2D up to large 5D). Seeded by testId only; fully deterministic."""
import sys

# (d, M) difficulty ladder: small -> large / higher dimension.
LADDER = [
    (2, 16),
    (2, 32),
    (2, 64),
    (3, 40),
    (3, 80),
    (3, 150),
    (4, 120),
    (4, 220),
    (5, 180),
    (5, 300),
]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t <= len(LADDER):
        d, M = LADDER[t - 1]
    else:
        # deterministic extension beyond the base ladder
        d = 2 + ((t - 1) % 4)
        M = 16 + (t * 31) % 285
    print("%d %d" % (d, M))


if __name__ == "__main__":
    main()
