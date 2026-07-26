#!/usr/bin/env python3
"""gen.py <testId> -- publishes the complete small-domain value table of a hidden
integer law F(x,y), for x,y in [0,M]. The law itself (its algebraic form and the
per-test constants k,p) is NEVER printed -- only the table. testId selects the
per-instance secret constants (k,p) and the magnitude scale used later (by the
checker only) for held-out extrapolation testing.
"""
import sys

M = 14

# testId -> (k, p, held_out_scale). Fixed, small, deterministic. This table is
# duplicated verbatim in counter.py so the checker can independently regenerate
# ground truth without ever reading this file at run time (solutions run sandboxed
# and cannot see this source either).
PARAMS = {
    1: (3, 2, 300),
    2: (3, 5, 800),
    3: (3, 7, 3000),
    4: (5, 3, 9000),
    5: (3, 4, 30000),
    6: (3, 9, 90000),
    7: (2, 6, 250000),
    8: (3, 1, 500000),
    9: (7, 8, 1000000),
    10: (3, 3, 1000000),
}


def F(x, y, k, p):
    num = 3 * x * y * (x + y)
    den = k * x * y + p
    return num % den


def main():
    testId = int(sys.argv[1])
    if testId not in PARAMS:
        testId = ((testId - 1) % 10) + 1
    k, p, _scale = PARAMS[testId]
    lines = [f"{testId} {M}"]
    for x in range(M + 1):
        row = [str(F(x, y, k, p)) for y in range(M + 1)]
        lines.append(" ".join(row))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
