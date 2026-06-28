#!/usr/bin/env python3
"""
Deterministic local scorer for ale-33: Grid Wire Routing.

Usage:
    python3 score.py <instance_file> <solution_file>
prints a single float: the score (0.0 if infeasible).

Scoring rule (as stated in context.md):
  * Read the instance (H, W, n, and the n endpoint pairs).
  * Read the solution.  The solution gives, for each of the n pairs IN INPUT
    ORDER, one path as a cell list:
        L  r_0 c_0  r_1 c_1  ...  r_{L-1} c_{L-1}
    meaning the path visits L cells, in order.  Its edge length is L-1.
  * FEASIBILITY (all of the following must hold; otherwise score 0.0):
      - the file parses (n path records, each with L >= 1 cells and 2L coords);
      - every cell is inside the grid (0 <= r < H, 0 <= c < W);
      - within a path, every consecutive pair of cells differs by exactly one
        in r or c (a unit rectilinear step) -- no diagonals, no jumps, no
        repeated cell within the same path;
      - the path's first cell equals endpoint 1 of its pair and its last cell
        equals endpoint 2 (it actually connects the right terminals);
      - NO grid cell is used by more than one wire (vertex-disjoint paths,
        endpoints included).  A single shared cell floors the score to 0.
  * If feasible, the raw cost is the total wire length
        total = sum over pairs of (L_k - 1).
    We normalise against the sum of independent shortest-path lengths, i.e. the
    Manhattan distance of each pair (the unconstrained lower bound that ignores
    all conflicts):
        LB = sum over pairs of |r1-r2| + |c1-c2|.
    The score is
        score = LB / total.
    Because the disjoint routing can only be longer than the unconstrained
    shortest paths, total >= LB, so score <= 1.0; higher (closer to 1) is
    better.  An infeasible output (any broken path or shared cell) scores 0.0.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it))
    W = int(next(it))
    n = int(next(it))
    pairs = []
    for _ in range(n):
        r1 = int(next(it)); c1 = int(next(it))
        r2 = int(next(it)); c2 = int(next(it))
        pairs.append((r1, c1, r2, c2))
    return H, W, n, pairs


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    inst, soln = sys.argv[1], sys.argv[2]
    H, W, n, pairs = read_instance(inst)

    # Lower bound = sum of Manhattan distances (independent shortest paths).
    LB = 0
    for (r1, c1, r2, c2) in pairs:
        LB += abs(r1 - r2) + abs(c1 - c2)
    if LB <= 0:
        LB = 1  # guard (cannot happen: endpoints of a pair differ)

    # Parse solution: n path records.
    try:
        with open(soln) as f:
            toks = f.read().split()
        it = iter(toks)
        paths = []
        for _ in range(n):
            L = int(next(it))
            if L < 1:
                print(0.0)
                return
            cells = []
            for _c in range(L):
                r = int(next(it)); c = int(next(it))
                cells.append((r, c))
            paths.append(cells)
    except Exception:
        print(0.0)
        return

    used = {}          # cell -> wire id that owns it
    total = 0
    for k, cells in enumerate(paths):
        L = len(cells)
        # endpoints must match this pair (start = ep1, end = ep2)
        r1, c1, r2, c2 = pairs[k]
        if cells[0] != (r1, c1) or cells[-1] != (r2, c2):
            print(0.0)
            return
        seen_in_path = set()
        prev = None
        for (r, c) in cells:
            if not (0 <= r < H and 0 <= c < W):
                print(0.0)
                return
            if (r, c) in seen_in_path:   # path revisits a cell
                print(0.0)
                return
            seen_in_path.add((r, c))
            if prev is not None:
                dr = abs(r - prev[0]); dc = abs(c - prev[1])
                if dr + dc != 1:         # not a unit rectilinear step
                    print(0.0)
                    return
            prev = (r, c)
        # disjointness across wires
        for cell in cells:
            if cell in used:
                print(0.0)
                return
            used[cell] = k
        total += (L - 1)

    if total < LB:
        # Should be impossible (LB is a true lower bound); guard against a
        # malformed instance.  Treat as a perfect score.
        total = LB
    if total <= 0:
        total = 1

    score = LB / float(total)
    print(f"{score:.6f}")


if __name__ == "__main__":
    main()
