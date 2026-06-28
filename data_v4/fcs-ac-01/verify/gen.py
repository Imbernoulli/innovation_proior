#!/usr/bin/env python3
"""
Random small-case generator for "Parity-Invariant Reachability".

Usage: gen.py <seed>

Produces a board pair (A, B) on a small R x C grid. With ~50% probability B is a
genuine permutation of the same tile multiset (so the answer may be YES or NO,
exercising the invariant); the rest of the time B is produced by actually walking
some random legal moves from A (guaranteeing a reachable YES instance), so the
test set is not dominated by NO answers. Occasionally we also emit boards whose
multisets differ -> trivially NO (the equal-multiset precondition is part of the
contract but the solution must still print NO, never crash).

Board sizes are kept tiny (R*C <= 9) so the BFS oracle terminates quickly.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Pick a small GENUINE 2D board (R >= 2 and C >= 2), keeping R*C <= 9 so the
    # BFS oracle terminates quickly. Degenerate 1-wide strips are excluded by the
    # problem contract (the parity invariant is exact only for true 2D boards).
    shapes = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]
    R, C = rng.choice(shapes)
    n = R * C
    tiles = list(range(n))  # 0 is the blank

    A = tiles[:]
    rng.shuffle(A)

    mode = rng.randint(0, 3)
    if mode == 0:
        # Reachable-by-construction: walk random legal moves from A.
        B = A[:]
        z = B.index(0)
        r, c = divmod(z, C)
        steps = rng.randint(0, 40)
        for _ in range(steps):
            opts = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    opts.append((nr, nc))
            nr, nc = rng.choice(opts)
            nz = nr * C + nc
            B[z], B[nz] = B[nz], B[z]
            z, r, c = nz, nr, nc
    elif mode == 3:
        # Different multiset -> trivially NO. Bump one tile value out of range.
        B = tiles[:]
        rng.shuffle(B)
        idx = rng.randrange(n)
        B[idx] = n + rng.randint(0, 3)  # value not in {0..n-1}
    else:
        # Independent random permutation of the same multiset.
        B = tiles[:]
        rng.shuffle(B)

    out = [f"{R} {C}"]
    for row in range(R):
        out.append(" ".join(str(A[row * C + col]) for col in range(C)))
    for row in range(R):
        out.append(" ".join(str(B[row * C + col]) for col in range(C)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
