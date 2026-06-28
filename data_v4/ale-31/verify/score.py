#!/usr/bin/env python3
"""Deterministic local scorer for "Balanced Districting" (graph partition).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is an H x W grid of POPULATIONS p[r][c] (each >= 1) and a target
    number of districts K. A SOLUTION assigns every cell a district id in
    [0, K-1] (row-major, H*W ids). The partition is FEASIBLE iff:
      (a) the output parses as exactly H*W integers, each in [0, K-1];
      (b) every district id 0..K-1 is used by at least one cell (no empty
          district);
      (c) every district is a single 4-connected region (the cells with a given
          id form one orthogonally-connected blob).
  * COST of a feasible partition is
        cost = imbalance + LAMBDA * boundary
    where, with avg = (total population) / K,
        imbalance = sum over districts d of | pop(d) - avg |   (an L1 deviation)
        boundary  = number of unordered 4-adjacent cell pairs whose two cells
                    lie in DIFFERENT districts (the total cut length).
    LAMBDA is a fixed weight (see LAMBDA below). Lower cost is better.
  * FEASIBILITY FLOOR: any parse error, an id out of range, an empty district, or
    a disconnected district makes the solution INFEASIBLE and the score is 0.
  * The reference partition is the STRIPE partition: cut the grid into K
    contiguous horizontal bands of (as-equal-as-possible) numbers of ROWS, each
    band one district. This is always feasible (each band is a connected
    rectangle of rows, K <= H*W and we clamp to <= H bands by splitting columns
    when K > H -- see stripe_cost). Its cost is cost_ref.
  * SCORE = round(1_000_000 * cost_ref / cost_solver) for a feasible partition
    with cost_solver > 0, and a generous full-credit cap when cost_solver == 0
    (a perfectly balanced, zero-cut partition, essentially never reachable). The
    stripe reference scores ~1_000_000; a better partition scores strictly more;
    a worse-but-feasible one scores less but stays positive. Infeasible -> 0.

The scorer is self-contained and deterministic: it recomputes the stripe
reference itself, so the baseline is reproducible and solver-independent.
"""
import sys

LAMBDA = 100  # weight of the boundary (cut-length) penalty in the cost


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it))
    W = int(next(it))
    K = int(next(it))
    grid = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            grid[r][c] = int(next(it))
    return H, W, K, grid


def read_solution(path, H, W, K):
    """Return a flat list of H*W district ids in [0,K-1], or None on error."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != H * W:
        return None
    ids = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            return None
        if v < 0 or v >= K:
            return None
        ids.append(v)
    return ids


def district_populations(H, W, K, grid, assign):
    pops = [0] * K
    for r in range(H):
        base = r * W
        for c in range(W):
            pops[assign[base + c]] += grid[r][c]
    return pops


def all_districts_connected(H, W, K, assign):
    """True iff every used id forms exactly one 4-connected region and every id
    0..K-1 is used at least once."""
    # count cells per id; every id must be non-empty
    cnt = [0] * K
    for v in assign:
        cnt[v] += 1
    for k in range(K):
        if cnt[k] == 0:
            return False
    # BFS flood per id from its first cell; must reach all cells of that id
    seen = [False] * (H * W)
    # find a representative cell for each id
    first = [-1] * K
    for idx, v in enumerate(assign):
        if first[v] < 0:
            first[v] = idx
    for k in range(K):
        start = first[k]
        stack = [start]
        seen[start] = True
        reached = 0
        while stack:
            cur = stack.pop()
            reached += 1
            r, c = divmod(cur, W)
            # 4 neighbours
            if r > 0:
                nb = cur - W
                if not seen[nb] and assign[nb] == k:
                    seen[nb] = True
                    stack.append(nb)
            if r < H - 1:
                nb = cur + W
                if not seen[nb] and assign[nb] == k:
                    seen[nb] = True
                    stack.append(nb)
            if c > 0:
                nb = cur - 1
                if not seen[nb] and assign[nb] == k:
                    seen[nb] = True
                    stack.append(nb)
            if c < W - 1:
                nb = cur + 1
                if not seen[nb] and assign[nb] == k:
                    seen[nb] = True
                    stack.append(nb)
        if reached != cnt[k]:
            return False  # district k split into multiple components
    return True


def boundary_count(H, W, assign):
    """Number of unordered 4-adjacent cell pairs in different districts."""
    b = 0
    for r in range(H):
        base = r * W
        for c in range(W):
            v = assign[base + c]
            if c + 1 < W and assign[base + c + 1] != v:
                b += 1
            if r + 1 < H and assign[base + c + W] != v:
                b += 1
    return b


def partition_cost(H, W, K, grid, assign):
    total = sum(sum(row) for row in grid)
    avg = total / K
    pops = district_populations(H, W, K, grid, assign)
    imbalance = sum(abs(p - avg) for p in pops)
    boundary = boundary_count(H, W, assign)
    return imbalance + LAMBDA * boundary


def stripe_assignment(H, W, K):
    """Reference partition: K contiguous bands.

    If K <= H, split the rows into K bands of near-equal height (band b covers a
    contiguous block of rows). If K > H (more districts than rows), fall back to
    a row-major contiguous chunking of the H*W cells into K near-equal contiguous
    runs -- still feasible because each run is a contiguous row-major block whose
    cells are 4-connected within the snake order... to stay strictly safe we
    instead chunk by columns within a single-row regime. For the instance sizes
    here (K <= 10 <= H), the band case is what is used; the fallback only guards
    the edge case.
    """
    N = H * W
    assign = [0] * N
    if K <= H:
        # K horizontal bands of rows
        for r in range(H):
            band = (r * K) // H
            if band >= K:
                band = K - 1
            for c in range(W):
                assign[r * W + c] = band
        # guard: ensure every band non-empty (true since K<=H, each row maps to a
        # band and bands are contiguous), but double check
        used = set(assign)
        if len(used) == K:
            return assign
    # fallback: contiguous row-major chunks of equal size (boustrophedon to keep
    # each chunk 4-connected). Chunk i gets cells [i*N//K, (i+1)*N//K).
    order = []
    for r in range(H):
        cols = range(W) if r % 2 == 0 else range(W - 1, -1, -1)
        for c in cols:
            order.append(r * W + c)
    for i in range(K):
        lo = (i * N) // K
        hi = ((i + 1) * N) // K
        for j in range(lo, hi):
            assign[order[j]] = i
    return assign


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    H, W, K, grid = read_instance(sys.argv[1])

    # reference (stripe) cost -- recomputed by the scorer, solver-independent
    ref_assign = stripe_assignment(H, W, K)
    assert all_districts_connected(H, W, K, ref_assign), "stripe ref must be feasible"
    cost_ref = partition_cost(H, W, K, grid, ref_assign)

    assign = read_solution(sys.argv[2], H, W, K)
    if assign is None:
        print(0)  # parse error / wrong count / id out of range -> infeasible
        return

    if not all_districts_connected(H, W, K, assign):
        print(0)  # empty or disconnected district -> infeasible
        return

    cost = partition_cost(H, W, K, grid, assign)
    if cost <= 0:
        # perfectly balanced AND zero cut: unreachable for K>=2 on a connected
        # grid, but give full+ credit rather than divide by zero.
        print(2_000_000)
        return

    score = int(round(1_000_000.0 * cost_ref / cost))
    print(score)


if __name__ == "__main__":
    main()
