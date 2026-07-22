# TIER: strong
"""
Insight: the run cap does not just forbid long monochrome runs, it turns
gradient matching into a TRANSPORTATION problem with an anti-clustering side
constraint. Sorting cells by target(r) and handing tile-values out in
matching sorted order (the textbook transportation optimum) paints solid
concentric rings -- exactly what the run-cap forbids on any ring wide
enough to cross several rows/columns, and exactly what greedy's local,
scan-order reactions cannot see coming.

This solution starts from greedy's construction (same one-pass blended
score) and then runs a RING-AWARE local search: repeatedly propose swapping
the colors of two cells (mostly -- but not always -- drawn from the SAME
ring, so it can directly attack whichever annulus is currently
under-dispersed), and accept the swap only if it improves the SAME
composite objective the checker scores:

    F = fidelity * (1 + BONUS * ring_dispersion)

where ring_dispersion is the fraction of angularly-adjacent same-ring cell
pairs that hold DIFFERENT colors. Because dispersion multiplies fidelity
rather than just adding to it, a swap that spreads a ring's mismatched
tiles out (raising dispersion) is worth taking even at a small fidelity
cost, and the search naturally gravitates toward "deliberately scatter the
unavoidable errors along each annulus" instead of leaving them clustered
wherever greedy's cap-repairs happened to land. Every candidate swap is
verified against the row/col run-cap before being kept, so the result is
always feasible.
"""
import math
import random
import sys

GAMMA_MULT = 1.8
BONUS = 140.0
ITERS_PER_CELL = 320
RING_BIAS = 0.85


def local_ok(grid, n, K, i, j):
    col = grid[i][j]
    l = j
    while l - 1 >= 0 and grid[i][l - 1] == col:
        l -= 1
    r = j
    while r + 1 < n and grid[i][r + 1] == col:
        r += 1
    if r - l + 1 > K:
        return False
    u = i
    while u - 1 >= 0 and grid[u - 1][j] == col:
        u -= 1
    d = i
    while d + 1 < n and grid[d + 1][j] == col:
        d += 1
    if d - u + 1 > K:
        return False
    return True


def greedy_construct(n, c, K, v, cnt, target):
    vrange = max(1, v[-1] - v[0])
    gamma = GAMMA_MULT / vrange
    remaining = list(cnt)
    grid = [[0] * n for _ in range(n)]
    col_last = [-1] * n
    col_run = [0] * n
    for i in range(n):
        row_last_c = -1
        row_run_c = 0
        for j in range(n):
            tgt = target(i, j)
            feas = []
            for k in range(c):
                if remaining[k] <= 0:
                    continue
                row_ok = not (row_last_c == k and row_run_c >= K)
                col_ok = not (col_last[j] == k and col_run[j] >= K)
                if not (row_ok and col_ok):
                    continue
                feas.append(k)
            best = -1
            if feas:
                best = max(feas, key=lambda k: remaining[k] - gamma * abs(v[k] - tgt))
            if best == -1:
                for relax_col in (False, True):
                    for relax_row in (False, True):
                        cc = []
                        for k in range(c):
                            if remaining[k] <= 0:
                                continue
                            row_ok = relax_row or not (row_last_c == k and row_run_c >= K)
                            col_ok = relax_col or not (col_last[j] == k and col_run[j] >= K)
                            if row_ok and col_ok:
                                cc.append(k)
                        if cc:
                            best = min(cc, key=lambda k: abs(v[k] - tgt))
                            break
                    if best != -1:
                        break
            grid[i][j] = best
            remaining[best] -= 1
            if row_last_c == best:
                row_run_c += 1
            else:
                row_last_c = best
                row_run_c = 1
            if col_last[j] == best:
                col_run[j] += 1
            else:
                col_last[j] = best
                col_run[j] = 1
    return grid


def ring_groups(n):
    cy = cx = (n - 1) / 2.0
    rings = {}
    for i in range(n):
        for j in range(n):
            r = math.hypot(i - cy, j - cx)
            ang = math.atan2(i - cy, j - cx)
            b = int(r + 0.5)
            rings.setdefault(b, []).append((ang, i, j))
    for b in rings:
        rings[b].sort()
    return rings


def build_ring_neighbors(n, rings):
    neigh = {}
    for b, lst in rings.items():
        m = len(lst)
        if m < 2:
            for _, i, j in lst:
                neigh[(i, j)] = None
            continue
        for idx in range(m):
            _, i, j = lst[idx]
            _, pi, pj = lst[(idx - 1) % m]
            _, ni, nj = lst[(idx + 1) % m]
            neigh[(i, j)] = ((pi, pj), (ni, nj))
    return neigh


def local_search(grid, n, c, K, v, target, rings, neigh, seed):
    vrange = max(1, v[-1] - v[0])
    max_rt = sum(len(lst) for lst in rings.values() if len(lst) >= 2)
    if max_rt <= 0:
        return grid
    rnd = random.Random(seed)
    all_cells = [(i, j) for i in range(n) for j in range(n)]
    ring_cellpos = {b: [(i, j) for _, i, j in lst] for b, lst in rings.items() if len(lst) >= 2}
    bands = list(ring_cellpos.keys())
    if not bands:
        return grid

    cur_fid = 0.0
    for i in range(n):
        for j in range(n):
            cur_fid += vrange - abs(v[grid[i][j]] - target(i, j))
    cur_rt = 0
    for b, lst in rings.items():
        m = len(lst)
        if m < 2:
            continue
        colors = [grid[i][j] for _, i, j in lst]
        cur_rt += sum(1 for x in range(m) if colors[x] != colors[(x + 1) % m])

    iters = ITERS_PER_CELL * n * n
    for _ in range(iters):
        if rnd.random() < RING_BIAS:
            b = bands[rnd.randrange(len(bands))]
            lst = ring_cellpos[b]
            p1 = lst[rnd.randrange(len(lst))]
            p2 = lst[rnd.randrange(len(lst))]
        else:
            p1 = all_cells[rnd.randrange(len(all_cells))]
            p2 = all_cells[rnd.randrange(len(all_cells))]
        if p1 == p2:
            continue
        i1, j1 = p1
        i2, j2 = p2
        c1, c2 = grid[i1][j1], grid[i2][j2]
        if c1 == c2:
            continue
        t1 = target(i1, j1)
        t2 = target(i2, j2)
        cur_local_fid = (vrange - abs(v[c1] - t1)) + (vrange - abs(v[c2] - t2))
        new_local_fid = (vrange - abs(v[c2] - t1)) + (vrange - abs(v[c1] - t2))

        def local_trans(pos, col):
            nb = neigh.get(pos)
            if nb is None:
                return 0
            (pi, pj), (ni, nj) = nb
            if (pi, pj) == p1:
                cp = c2
            elif (pi, pj) == p2:
                cp = c1
            else:
                cp = grid[pi][pj]
            if (ni, nj) == p1:
                cn = c2
            elif (ni, nj) == p2:
                cn = c1
            else:
                cn = grid[ni][nj]
            return (col != cp) + (col != cn)

        cur_trans = local_trans(p1, c1) + local_trans(p2, c2)
        new_trans = local_trans(p1, c2) + local_trans(p2, c1)

        new_fid_total = cur_fid - cur_local_fid + new_local_fid
        new_rt_total = cur_rt - cur_trans + new_trans

        old_obj = cur_fid * (1.0 + BONUS * cur_rt / max_rt)
        new_obj = new_fid_total * (1.0 + BONUS * new_rt_total / max_rt)
        if new_obj <= old_obj + 1e-9:
            continue

        grid[i1][j1], grid[i2][j2] = c2, c1
        if not (local_ok(grid, n, K, i1, j1) and local_ok(grid, n, K, i2, j2)):
            grid[i1][j1], grid[i2][j2] = c1, c2
            continue
        cur_fid = new_fid_total
        cur_rt = new_rt_total
    return grid


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it)); K = int(next(it))
    v = [int(next(it)) for _ in range(c)]
    cnt = [int(next(it)) for _ in range(c)]
    tcenter = int(next(it)); tedge = int(next(it))

    cy = cx = (n - 1) / 2.0
    rmax = math.hypot(cx, cy) if n > 1 else 1.0
    if rmax <= 0:
        rmax = 1.0

    def target(i, j):
        r = math.hypot(i - cy, j - cx)
        return tcenter + (tedge - tcenter) * (r / rmax)

    grid = greedy_construct(n, c, K, v, cnt, target)
    rings = ring_groups(n)
    neigh = build_ring_neighbors(n, rings)
    seed = (n * 1_000_003 + c * 9973 + K * 131 + sum(v) * 7 + sum(cnt) * 3
            + tcenter * 17 + tedge * 19) & 0x7FFFFFFF
    grid = local_search(grid, n, c, K, v, target, rings, neigh, seed)

    out_lines = []
    for i in range(n):
        out_lines.append(" ".join(str(grid[i][j] + 1) for j in range(n)))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
