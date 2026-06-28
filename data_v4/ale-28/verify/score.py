#!/usr/bin/env python3
"""Deterministic local scorer for "Sensor Placement for Coverage + Connectivity".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). HIGHER is better.

Scoring rule (see context.md "Evaluation settings"):
  * Instance: an H x W grid; cell (i, j) has demand d[i][j] >= 0. At most k
    sensors, each of radius r. lam is the connectivity penalty per extra
    component.
  * SOLUTION format (stdout of the solver):
        s
        i_0 j_0
        i_1 j_1
        ...
        i_{s-1} j_{s-1}
    s is the number of placed sensors; each following line is the (row, col) of a
    sensor (cell centre). 0 <= i < H, 0 <= j < W. Placing fewer than k sensors is
    allowed (s may be 0).

  * FEASIBILITY (any violation -> score 0):
      - the file parses as the integer s followed by exactly s well-formed
        "i j" lines (no trailing garbage);
      - 0 <= s <= k   (placing MORE than k sensors floors the score to 0);
      - every (i, j) is in range: 0 <= i < H and 0 <= j < W;
      - the sensor positions are DISTINCT (no two sensors on the same cell).
    If any of these fail the solution is INFEASIBLE and scores 0.

  * OBJECTIVE (higher is better) of a feasible solution:
        covered = sum of d[a][b] over every cell (a, b) that is within Euclidean
                  distance r of AT LEAST ONE placed sensor, i.e.
                  (a - i)^2 + (b - j)^2 <= r^2 for some sensor (i, j)
                  (each covered cell counted ONCE -- it is the demand of the UNION
                  of the coverage disks).
        Build the connectivity graph on the placed sensors: sensors u, v are
        linked iff (i_u - i_v)^2 + (j_u - j_v)^2 <= (2r)^2 (their disks touch or
        overlap). Let C be the number of connected components (C = 0 if s = 0).
        objective = covered - lam * max(0, C - 1)
    One component is "free"; each additional disconnected component costs lam.

  * SCORE (higher better), normalized against a deterministic grid-spaced
    placement baseline the scorer recomputes itself:
        score = round(1_000_000 * solver_objective / max(1, baseline_objective))
    The grid baseline scores ~1_000_000; a better placement scores more.
    A feasible solution whose objective is <= 0 scores 0 (it did no better than
    placing nothing). INFEASIBLE -> 0.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it))
    W = int(next(it))
    k = int(next(it))
    r = int(next(it))
    lam = int(next(it))
    d = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            d[i][j] = int(next(it))
    return H, W, k, r, lam, d


def read_solution(path, H, W, k):
    """Parse + fully validate. Return list of (i, j) sensors, or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    it = iter(toks)
    try:
        s = int(next(it))
    except (StopIteration, ValueError):
        return None
    if s < 0 or s > k:           # MORE than k sensors -> infeasible
        return None
    sensors = []
    seen = set()
    for _ in range(s):
        try:
            i = int(next(it))
            j = int(next(it))
        except (StopIteration, ValueError):
            return None
        if i < 0 or i >= H or j < 0 or j >= W:
            return None
        if (i, j) in seen:        # duplicate position -> infeasible
            return None
        seen.add((i, j))
        sensors.append((i, j))
    # reject trailing garbage
    if next(it, None) is not None:
        return None
    return sensors


# --------------------------------------------------------------------- objective
def covered_demand(sensors, H, W, r, d):
    r2 = r * r
    covered = [[False] * W for _ in range(H)]
    for (si, sj) in sensors:
        a0 = max(0, si - r)
        a1 = min(H - 1, si + r)
        b0 = max(0, sj - r)
        b1 = min(W - 1, sj + r)
        for a in range(a0, a1 + 1):
            da = (a - si) * (a - si)
            for b in range(b0, b1 + 1):
                if da + (b - sj) * (b - sj) <= r2:
                    covered[a][b] = True
    total = 0
    for a in range(H):
        row = covered[a]
        drow = d[a]
        for b in range(W):
            if row[b]:
                total += drow[b]
    return total


def num_components(sensors, r):
    s = len(sensors)
    if s == 0:
        return 0
    link2 = (2 * r) * (2 * r)
    parent = list(range(s))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u in range(s):
        iu, ju = sensors[u]
        for v in range(u + 1, s):
            iv, jv = sensors[v]
            if (iu - iv) ** 2 + (ju - jv) ** 2 <= link2:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
    roots = set(find(x) for x in range(s))
    return len(roots)


def objective(sensors, H, W, r, lam, d):
    cov = covered_demand(sensors, H, W, r, d)
    C = num_components(sensors, r)
    return cov - lam * max(0, C - 1)


# ------------------------------------------------------- baseline: grid placement
def baseline_objective(H, W, k, r, lam, d):
    """Deterministic grid-spaced placement.

    Lay down a regular lattice of candidate positions spaced ~2r apart (so
    neighbours are linked), walk them in row-major order, and take the first k.
    This is the natural "spread sensors evenly" baseline: it tends to be
    connected (one component) but ignores where the demand actually is, so it
    leaves a lot of high-demand cells uncovered.
    """
    step = max(1, 2 * r)
    cand = []
    i = r
    while i < H:
        j = r
        while j < W:
            cand.append((i, j))
            j += step
        i += step
    if not cand:
        cand = [(min(r, H - 1), min(r, W - 1))]
    sensors = cand[:k]
    # dedup just in case (positions are distinct by construction)
    seen = set()
    uniq = []
    for p in sensors:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return objective(uniq, H, W, r, lam, d)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    H, W, k, r, lam, d = read_instance(sys.argv[1])

    sensors = read_solution(sys.argv[2], H, W, k)
    if sensors is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    obj = objective(sensors, H, W, r, lam, d)
    if obj <= 0:
        print(0)
        return

    base = baseline_objective(H, W, k, r, lam, d)
    score = int(round(1_000_000.0 * obj / max(1, base)))
    print(score)


if __name__ == "__main__":
    main()
