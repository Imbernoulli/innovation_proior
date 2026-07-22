#!/usr/bin/env python3
# gen.py <testId>  -> prints ONE instance to stdout.
# Instance = n pre-ordered wires described by a planted partial order (a union of
# contiguous already-sorted "runs"), plus a depth weight alpha.
#
# Format (stdout):
#   line 1:  n  E  alpha
#   next E lines: "a b"   meaning: on EVERY valid input, value(wire a) <= value(wire b)
#                          (a and b are consecutive members of an already-sorted run)
#
# testId 1..10 = difficulty ladder. Deterministic (depends only on testId).
import sys

# (n, run_lengths, alpha).  run lengths partition [0,n) into contiguous sorted blocks.
# product(run_len+1) = number of consistent 0-1 inputs (kept small for the checker).
CASES = {
    1:  (6,  [3, 3],          0.10),
    2:  (8,  [4, 4],          0.15),
    3:  (10, [5, 5],          0.20),
    4:  (12, [4, 4, 4],       0.10),
    5:  (14, [7, 7],          0.25),
    6:  (16, [8, 8],          0.20),
    7:  (18, [6, 6, 6],       0.15),
    8:  (20, [5, 5, 5, 5],    0.10),
    9:  (22, [11, 11],        0.30),
    10: (24, [8, 8, 8],       0.20),
}

def main():
    tid = int(sys.argv[1])
    if tid not in CASES:
        # clamp into range for robustness
        tid = ((tid - 1) % 10) + 1
    n, runs, alpha = CASES[tid]
    edges = []
    s = 0
    for L in runs:
        for k in range(L - 1):
            edges.append((s + k, s + k + 1))
        s += L
    out = []
    out.append("%d %d %.4f" % (n, len(edges), alpha))
    for (a, b) in edges:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
