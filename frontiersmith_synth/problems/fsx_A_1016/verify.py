#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for stick-weld-routing (fsx_A_1016).

Validates the submitted polyline (a lattice walk A->B avoiding obstacles, no revisits),
computes its "developed length" (straight steps + bend allowances), finds the OPTIMAL
partition of that developed length into stock sticks of length L via DP (welds*W + waste),
and scores it against the checker's own naive-heuristic baseline path (run through the
SAME optimal-partition DP), minimization convention:
    sc = min(1000, 100*B/F);  print Ratio: sc/1000
Any feasibility violation -> Ratio: 0.0.
"""
import sys

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split("\n")
    idx = 0
    R, C = map(int, toks[idx].split()); idx += 1
    grid = []
    for _ in range(R):
        grid.append(toks[idx]); idx += 1
    Ar, Ac, Br, Bc = map(int, toks[idx].split()); idx += 1
    L, a, W = map(int, toks[idx].split()); idx += 1
    return R, C, grid, (Ar, Ac), (Br, Bc), L, a, W


def naive_baseline_path(R, C, grid, A, B):
    """Deterministic naive heuristic walk (DFS w/ a fixed greedy direction preference,
    backtracking on dead ends). This is the checker's OWN trivial construction (mirrors
    solutions/trivial.py bit-for-bit) -- used only to compute the calibration baseline B."""
    def blocked(r, c):
        return not (0 <= r < R and 0 <= c < C) or grid[r][c] == '#'

    def pref_order(cur, tgt):
        dr = tgt[0] - cur[0]; dc = tgt[1] - cur[1]
        cand = []
        if abs(dr) >= abs(dc):
            cand.append((1 if dr > 0 else -1, 0) if dr != 0 else (0, 1 if dc >= 0 else -1))
            cand.append((0, 1 if dc >= 0 else -1) if dc != 0 else (1 if dr >= 0 else -1, 0))
        else:
            cand.append((0, 1 if dc > 0 else -1) if dc != 0 else (1 if dr >= 0 else -1, 0))
            cand.append((1 if dr >= 0 else -1, 0) if dr != 0 else (0, 1 if dc >= 0 else -1))
        for d in DIRS:
            if d not in cand:
                cand.append(d)
        return cand

    visited = {A}
    path = [A]
    cur = A
    guard = 0
    max_guard = R * C * 6 + 50
    while cur != B and guard < max_guard:
        guard += 1
        moved = False
        for dr, dc in pref_order(cur, B):
            nr, nc = cur[0] + dr, cur[1] + dc
            if not blocked(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                path.append((nr, nc))
                cur = (nr, nc)
                moved = True
                break
        if not moved:
            if len(path) <= 1:
                return None
            path.pop()
            cur = path[-1]
    if cur != B:
        return None
    return path


def path_to_devlen(path, a):
    """Given a lattice polyline (list of (r,c)), return (T, blobs) where T is the developed
    length (unit steps + bend allowances) and blobs is a sorted list of (start,end) intervals
    (in developed-length coordinates) that a stick cut may NOT fall strictly inside."""
    D = len(path) - 1
    directions = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(D)]
    cum = 0
    blobs = []
    for i in range(D):
        cum += 1
        if i < D - 1 and directions[i] != directions[i + 1]:
            blobs.append((cum, cum + a))
            cum += a
    return cum, blobs


def optimal_partition_cost(T, blobs, L, W):
    """DP: min welds*W + waste over partitions of [0,T] into pieces of length <= L, no cut
    point may land strictly inside a blob (bend can't be split across a weld)."""
    if T == 0:
        return 0.0
    INF = float("inf")

    def invalid_cut(p):
        for s, e in blobs:
            if s < p < e:
                return True
            if s > p:
                break
        return False

    dp = [INF] * (T + 1)
    dp[0] = 0.0
    for i in range(T + 1):
        if dp[i] == INF:
            continue
        hi = min(i + L, T)
        for j in range(i + 1, hi + 1):
            if j != T and invalid_cut(j):
                continue
            waste = L - (j - i)
            cost = dp[i] + waste + (W if j < T else 0)
            if cost < dp[j]:
                dp[j] = cost
    return dp[T]


def fail(msg):
    print("INFEASIBLE: %s  Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]
    R, C, grid, A, B, L, a, W = read_instance(in_path)

    try:
        with open(out_path, "r") as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")

    toks = raw.split()
    if not toks:
        fail("empty output")

    try:
        K = int(toks[0])
    except (ValueError, OverflowError):
        fail("K not an integer")

    # generous DoS-safe cap on path length (legitimate solutions never need anywhere near this;
    # bounds the DP's O(T*L) work even under an adversarial huge-token attack)
    max_pts = 6 * R * C + 20
    if K < 1 or K > max_pts:
        fail("K out of range")
    if len(toks) < 1 + 2 * K:
        fail("not enough coordinate tokens")

    pts = []
    for i in range(K):
        try:
            r = int(toks[1 + 2 * i]); c = int(toks[2 + 2 * i])
        except (ValueError, OverflowError):
            fail("non-integer coordinate")
        if not (0 <= r < R and 0 <= c < C):
            fail("coordinate out of grid bounds")
        pts.append((r, c))

    if pts[0] != A:
        fail("path does not start at A")
    if pts[-1] != B:
        fail("path does not end at B")

    for i, (r, c) in enumerate(pts):
        if grid[r][c] == '#':
            fail("path passes through an obstacle at step %d" % i)
        if i > 0:
            pr, pc = pts[i - 1]
            if abs(r - pr) + abs(c - pc) != 1:
                fail("step %d is not a unit orthogonal move" % i)

    if K == 1:
        fail("A and B coincide with a single-point path (degenerate instance)")

    T, blobs = path_to_devlen(pts, a)
    F = optimal_partition_cost(T, blobs, L, W)
    if F == float("inf"):
        fail("no valid stick partition exists for this path (should not happen)")

    base_path = naive_baseline_path(R, C, grid, A, B)
    if base_path is None:
        fail("internal: checker baseline path-finding failed (generator bug)")
    Tb, blobsb = path_to_devlen(base_path, a)
    B_cost = optimal_partition_cost(Tb, blobsb, L, W)

    sc = min(1000.0, 100.0 * B_cost / max(1e-9, F))
    ratio = sc / 1000.0
    print("dev_len=%d bends=%d cost=%.6f baseline_cost=%.6f Ratio: %.6f" %
          (T, len(blobs), F, B_cost, ratio))


if __name__ == "__main__":
    main()
