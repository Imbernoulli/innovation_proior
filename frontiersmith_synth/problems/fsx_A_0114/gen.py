#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance of the lunar-habitat Heilbronn problem.

testId 1..10 is a difficulty ladder: the number of habitat modules `n` grows with
testId, and the shape of the triangular building plot varies (seeded by testId only,
so generation is bit-for-bit deterministic).

Instance format (stdin of the solver):
    n
    Ax Ay
    Bx By
    Cx Cy

`n` habitat modules must be placed inside the triangular plot with corners A, B, C.
The objective (see statement.md) is to MAXIMIZE the minimum triangle area over all
triples of modules -- a Heilbronn-triangle extremal point configuration.
"""
import sys, random


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(20260702 + 8117 * tid)

    n = 6 + tid                       # 7 .. 16 modules

    # Triangular plot: base edge A--B fixed on the ground line, apex C varies so the
    # plot shape (and hence the extremal configuration) differs from test to test.
    Ax, Ay = 0.0, 0.0
    Bx, By = 1.0, 0.0
    Cx = round(rng.uniform(0.20, 0.65), 4)
    Cy = round(rng.uniform(0.70, 1.00), 4)

    out = [str(n),
           "%.4f %.4f" % (Ax, Ay),
           "%.4f %.4f" % (Bx, By),
           "%.4f %.4f" % (Cx, Cy)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
