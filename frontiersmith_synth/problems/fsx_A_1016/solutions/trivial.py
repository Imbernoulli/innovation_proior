# TIER: trivial
"""Naive deterministic wall-hugging walk: greedily step toward B (larger-delta axis first,
fixed tie-break), backtrack (DFS) on dead ends. Ignores L/a/W entirely -- this is exactly
the checker's own internal calibration baseline, reproduced bit-for-bit."""
import sys

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def read_instance():
    data = sys.stdin.read().split("\n")
    idx = 0
    R, C = map(int, data[idx].split()); idx += 1
    grid = []
    for _ in range(R):
        grid.append(data[idx]); idx += 1
    Ar, Ac, Br, Bc = map(int, data[idx].split()); idx += 1
    L, a, W = map(int, data[idx].split()); idx += 1
    return R, C, grid, (Ar, Ac), (Br, Bc), L, a, W


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


def naive_baseline_path(R, C, grid, A, B):
    def blocked(r, c):
        return not (0 <= r < R and 0 <= c < C) or grid[r][c] == '#'

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


def main():
    R, C, grid, A, B, L, a, W = read_instance()
    path = naive_baseline_path(R, C, grid, A, B)
    if path is None:
        path = [A, B]  # should never happen; harmless fallback
    out = [str(len(path))]
    for r, c in path:
        out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
