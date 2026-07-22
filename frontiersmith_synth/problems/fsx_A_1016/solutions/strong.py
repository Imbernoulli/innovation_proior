# TIER: strong
"""The insight: don't minimize geometric path length -- minimize the ACTUAL stick-partition
cost directly, by running a shortest-path search over an augmented state
    (cell, incoming direction, remaining capacity of the current stock stick)
where a straight step consumes 1 unit of capacity, a direction change (bend) additionally
consumes the bend allowance 'a' (forced early weld if the current stick can't fit it), and a
stick may also be cut VOLUNTARILY at any time (paying a weld + the leftover waste) to set up
a clean full-length stick for what's coming. This is exactly the checker's own optimal
stick-partition DP, fused with the routing search, so the path it returns is jointly optimal
for (route shape, stick partition) -- it will happily take a slightly longer route around an
obstacle cluster if that lets bends land on stick-length-congruent spacing, beating the
geodesic on total welds+waste even though it is not the shortest path."""
import sys, heapq

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


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


def solve(R, C, grid, A, B, L, a, W):
    def blocked(r, c):
        return not (0 <= r < R and 0 <= c < C) or grid[r][c] == '#'

    start = (A[0], A[1], -1, L)
    dist = {start: 0.0}
    prev = {start: None}
    pq = [(0.0, start)]
    best = None  # (finalized_cost, state)

    while pq:
        d, state = heapq.heappop(pq)
        if d > dist.get(state, float("inf")) + 1e-9:
            continue
        r, c, dir_idx, cap = state
        if (r, c) == B:
            fin = d + cap  # leftover capacity of the final, in-progress stick is wasted
            if best is None or fin < best[0] - 1e-9:
                best = (fin, state)
        # voluntary/forced weld: cut the current stick now, start a fresh full-length one
        # (the stick's unused remaining capacity `cap` is the wasted offcut)
        if dir_idx != -1 and cap < L:
            ns = (r, c, dir_idx, L)
            nd = d + W + cap
            if nd < dist.get(ns, float("inf")) - 1e-9:
                dist[ns] = nd; prev[ns] = (state, None)
                heapq.heappush(pq, (nd, ns))
        # move to a free neighbor
        for ndi, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if blocked(nr, nc):
                continue
            bend = dir_idx != -1 and ndi != dir_idx
            need = 1 + (a if bend else 0)
            if cap < need:
                continue
            ns = (nr, nc, ndi, cap - need)
            nd = d
            if nd < dist.get(ns, float("inf")) - 1e-9:
                dist[ns] = nd; prev[ns] = (state, (nr, nc))
                heapq.heappush(pq, (nd, ns))

    if best is None:
        return None
    _, state = best
    pts = []
    cur = state
    while cur is not None:
        entry = prev.get(cur)
        if entry is None:
            break
        pstate, pt = entry
        if pt is not None:
            pts.append(pt)
        cur = pstate
    pts.append(A)
    pts.reverse()
    return pts


def main():
    R, C, grid, A, B, L, a, W = read_instance()
    path = solve(R, C, grid, A, B, L, a, W)
    if path is None:
        path = [A, B]
    out = [str(len(path))]
    for r, c in path:
        out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
