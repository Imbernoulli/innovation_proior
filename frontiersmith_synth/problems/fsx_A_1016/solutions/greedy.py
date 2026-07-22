# TIER: greedy
"""The obvious first move: plain BFS shortest path from A to B avoiding obstacles. Minimizes
raw geometric length only -- completely ignorant of L/a/W, so it happily crams many bends
into a short span whenever that's what shortest-path takes (the trap)."""
import sys
from collections import deque

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def bfs_shortest_path(R, C, grid, A, B):
    prev = {A: None}
    q = deque([A])
    while q:
        cur = q.popleft()
        if cur == B:
            break
        for dr, dc in DIRS:
            nr, nc = cur[0] + dr, cur[1] + dc
            np = (nr, nc)
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == '.' and np not in prev:
                prev[np] = cur
                q.append(np)
    if B not in prev:
        return None
    path = []
    cur = B
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def main():
    R, C, grid, A, B, L, a, W = read_instance()
    path = bfs_shortest_path(R, C, grid, A, B)
    if path is None:
        path = [A, B]
    out = [str(len(path))]
    for r, c in path:
        out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
