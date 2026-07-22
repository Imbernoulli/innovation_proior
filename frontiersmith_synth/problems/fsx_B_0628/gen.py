import sys

# gen.py <testId>  -- prints ONE endurance-race pacing instance to stdout.
#
# Instance format (whitespace separated):
#   line 1: L k                 (laps, size of the intensity grid)
#   line 2: base a p b q P       (lap-time / wear / pit-cost coefficients)
#   line 3: g_0 g_1 ... g_{k-1}  (intensity grid, strictly ascending, all > 0)
#
# The lap-time model is
#     lap_time_i = base + a * wear_i**p + b / x_i
# where wear_i is the wear at the START of lap i, and after the lap
#     wear <- wear + x_i**q .
# A pit BEFORE a lap pays P seconds and resets wear to 0.
#
# The trap regime: p in ~2.0-2.3 makes the wear penalty strongly CONVEX and q<2
# makes accumulation concave in intensity, so a single constant pace with evenly
# spaced pits (the textbook stint answer) is far from a stint-position-aware
# ramp.  Difficulty (L and nonconvexity) grows with testId; tests 4..10 are
# engineered so the constant-pace greedy lands far from the strong pacing.

SPECS = {
 1:  (10, 3.0, 2.0, 1.8,  9.0, 1.5, 6.0, [0.5, 0.8, 1.1, 1.5, 2.0]),
 2:  (12, 3.0, 2.5, 2.0, 10.0, 1.6, 6.0, [0.5, 0.8, 1.1, 1.5, 2.0]),
 3:  (14, 2.0, 3.0, 2.0, 11.0, 1.6, 7.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
 4:  (16, 2.0, 3.0, 2.2, 12.0, 1.7, 6.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
 5:  (18, 2.0, 3.0, 2.2, 13.0, 1.8, 6.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
 6:  (20, 2.0, 3.0, 2.2, 12.0, 1.8, 7.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
 7:  (22, 2.5, 3.0, 2.3, 13.0, 1.8, 6.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
 8:  (24, 2.0, 3.0, 2.2, 14.0, 1.8, 6.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
 9:  (28, 4.0, 3.0, 2.2, 12.0, 1.8, 7.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
 10: (32, 5.0, 3.0, 2.2, 12.0, 1.8, 8.0, [0.5, 0.7, 1.0, 1.4, 1.9, 2.5]),
}

def main():
    tid = int(sys.argv[1])
    L, base, a, p, b, q, P, grid = SPECS[tid]
    k = len(grid)
    out = []
    out.append("%d %d" % (L, k))
    out.append("%s %s %s %s %s %s" % (repr(base), repr(a), repr(p), repr(b), repr(q), repr(P)))
    out.append(" ".join(repr(g) for g in grid))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
