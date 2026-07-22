# TIER: strong
import sys, random


def build_common(R, C, base, counters, hint_map):
    def partner(r):
        return (r + R // 2) % R

    def scope_cells(r, scope):
        if scope == "ROW":
            return [(r, cc) for cc in range(C)]
        rp = partner(r)
        return [(r, cc) for cc in range(C)] + [(rp, cc) for cc in range(C)]

    for k in range(len(counters)):
        r, scope = counters[k][0], counters[k][4]
        counters[k][5] = scope_cells(r, scope)

    affected = {}
    for k in range(len(counters)):
        for cell in counters[k][5]:
            affected.setdefault(cell, []).append(k)

    free_cells = [(r, c) for r in range(R) for c in range(C) if (r, c) not in hint_map]
    return affected, free_cells


def total_weight(grid, counters):
    total = 0
    for (r, c, d, w, scope, cells) in counters:
        actual = 0
        for (rr, cc) in cells:
            if grid[rr][cc] == d:
                actual += 1
        if grid[r][c] == actual:
            total += w
    return total


def coordinate_descent(grid, R, C, base, counters, hint_map, affected, free_cells, rng, sweeps):
    K = len(counters)
    current_count = [0] * K
    for k in range(K):
        r, c, d, w, scope, cells = counters[k]
        cnt = 0
        for (rr, cc) in cells:
            if grid[rr][cc] == d:
                cnt += 1
        current_count[k] = cnt
    current_satisfied = [grid[counters[k][0]][counters[k][1]] == current_count[k] for k in range(K)]

    for _sweep in range(sweeps):
        order = free_cells[:]
        rng.shuffle(order)
        improved = False
        for (r, c) in order:
            aff = affected.get((r, c))
            if not aff:
                continue
            cur_val = grid[r][c]
            best_val = cur_val
            best_delta = 0
            for v in range(base):
                if v == cur_val:
                    continue
                delta = 0
                for k in aff:
                    rk, ck, dk, wk, scope_k, cellsk = counters[k]
                    old_count = current_count[k]
                    new_count = old_count + (1 if v == dk else 0) - (1 if cur_val == dk else 0)
                    new_target = v if (rk, ck) == (r, c) else grid[rk][ck]
                    new_sat = (new_target == new_count)
                    old_sat = current_satisfied[k]
                    if new_sat and not old_sat:
                        delta += wk
                    elif old_sat and not new_sat:
                        delta -= wk
                if delta > best_delta:
                    best_delta = delta
                    best_val = v
            if best_val != cur_val:
                for k in aff:
                    rk, ck, dk, wk, scope_k, cellsk = counters[k]
                    old_count = current_count[k]
                    new_count = old_count + (1 if best_val == dk else 0) - (1 if cur_val == dk else 0)
                    current_count[k] = new_count
                grid[r][c] = best_val
                for k in aff:
                    rk, ck, dk, wk, scope_k, cellsk = counters[k]
                    new_sat = (grid[rk][ck] == current_count[k])
                    if new_sat != current_satisfied[k]:
                        current_satisfied[k] = new_sat
                improved = True
        if not improved:
            break
    return grid


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it)); base = int(next(it))
    K = int(next(it))
    counters = []  # (r, c, d, w, scope, cells)
    for _ in range(K):
        r = int(next(it)); c = int(next(it)); d = int(next(it)); w = int(next(it))
        scope = next(it)
        counters.append([r, c, d, w, scope, None])
    H = int(next(it))
    hint_map = {}
    for _ in range(H):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        hint_map[(r, c)] = v

    affected, free_cells = build_common(R, C, base, counters, hint_map)

    seed = (R * 1000003 + C * 9187 + base * 733 + K * 131 + H * 17) & 0x7fffffff

    # exploit the "one edit flips exactly two tallies" invariant via O(1) incremental
    # coordinate descent (monotone non-decreasing), from several deterministic starting
    # fills / sweep orders, and keep the best converged tableau -- this both escapes
    # shallow local fixpoints and never regresses within a run (unlike synchronous
    # recount-and-rewrite, which can 2-cycle and land anywhere).
    seed_rng = random.Random(seed + 99991)
    starts = [
        [[c % base for c in range(C)] for r in range(R)],
        [[(base - 1 - (c % base)) for c in range(C)] for r in range(R)],
        [[0 for c in range(C)] for r in range(R)],
        None,  # random fill, generated below
        None,  # random fill, generated below
    ]
    best_grid = None
    best_score = -1
    for si, start in enumerate(starts):
        if start is None:
            init_rng = random.Random(seed + si * 7919)
            grid = [[init_rng.randrange(base) for _ in range(C)] for _ in range(R)]
        else:
            grid = [row[:] for row in start]
        for (r, c), v in hint_map.items():
            grid[r][c] = v
        rng = random.Random(seed + si * 104729)
        grid = coordinate_descent(grid, R, C, base, counters, hint_map, affected, free_cells, rng, sweeps=12)
        sc = total_weight(grid, counters)
        if sc > best_score:
            best_score = sc
            best_grid = grid

    out_lines = [" ".join(map(str, row)) for row in best_grid]
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
