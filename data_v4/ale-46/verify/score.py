#!/usr/bin/env python3
"""Deterministic scorer for the Number-Link (Flow-Free style) maximum-connections
problem.

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score.

Scoring rule (matches context.md):
  * The instance defines a grid H x W and K colored endpoint pairs.
  * A solution claims to connect C of the pairs.  Each claimed path is a list of
    grid cells.
  * The path for pair `id` is VALID iff:
      - `id` is a valid pair index in [0, K) and is claimed at most once;
      - the path has >= 2 cells;
      - its first and last cells are exactly the two endpoints of pair `id`
        (in either order);
      - every cell lies inside the grid (0 <= r < H, 0 <= c < W);
      - consecutive cells differ by exactly one step (|dr| + |dc| == 1), i.e.
        unit rectilinear moves;
      - the path does not revisit a cell (a simple path).
  * GLOBAL feasibility: across ALL claimed paths, no grid cell is used by more
    than one path.  Endpoint cells count as used by their own path; an endpoint
    cell of one pair may NOT appear on another pair's path.
  * If ANY of the above is violated -- a malformed line, an out-of-range id, a
    duplicate id, a non-unit step, an out-of-bounds cell, a wrong endpoint, a
    self-revisit, or a shared cell between two paths -- the whole solution is
    INFEASIBLE and the score is floored to 0.
  * Otherwise the score is C, the number of pairs connected.

This is a hard floor: a single shared cell or invalid path makes the score 0,
exactly as stated in the candidate's local scoring rule.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); K = int(next(it))
    pairs = []
    for _ in range(K):
        ra = int(next(it)); ca = int(next(it))
        rb = int(next(it)); cb = int(next(it))
        pairs.append(((ra, ca), (rb, cb)))
    return H, W, K, pairs


def score(instance_file, solution_file):
    H, W, K, pairs = read_instance(instance_file)

    with open(solution_file) as f:
        toks = f.read().split()

    # Empty / no output -> 0 valid pairs (feasible, score 0).
    if not toks:
        return 0

    it = iter(toks)
    try:
        C = int(next(it))
    except StopIteration:
        return 0

    if C < 0 or C > K:
        return 0

    used = {}          # cell -> pair id that uses it
    seen_ids = set()

    for _ in range(C):
        try:
            pid = int(next(it))
            L = int(next(it))
        except StopIteration:
            return 0  # malformed: fewer paths than claimed
        if pid < 0 or pid >= K:
            return 0
        if pid in seen_ids:
            return 0
        seen_ids.add(pid)
        if L < 2:
            return 0
        cells = []
        for _k in range(L):
            try:
                r = int(next(it)); c = int(next(it))
            except StopIteration:
                return 0
            cells.append((r, c))

        # In-bounds and unit-step + no self-revisit.
        local_seen = set()
        for idx, (r, c) in enumerate(cells):
            if not (0 <= r < H and 0 <= c < W):
                return 0
            if (r, c) in local_seen:
                return 0  # self-revisit -> not a simple path
            local_seen.add((r, c))
            if idx > 0:
                pr, pc = cells[idx - 1]
                if abs(r - pr) + abs(c - pc) != 1:
                    return 0  # not a unit rectilinear step

        # Endpoints must match the pair's endpoints (either orientation).
        a, b = pairs[pid]
        start, end = cells[0], cells[-1]
        if not ((start == a and end == b) or (start == b and end == a)):
            return 0

        # Global conflict check: no cell shared with another path.
        for cell in cells:
            if cell in used:
                return 0  # shared cell -> infeasible -> floor 0
            used[cell] = pid

    # No extra records or garbage may follow the declared C paths.
    try:
        next(it)
        return 0
    except StopIteration:
        pass

    return len(seen_ids)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance_file> <solution_file>\n")
        sys.exit(1)
    s = score(sys.argv[1], sys.argv[2])
    print(s)


if __name__ == "__main__":
    main()
