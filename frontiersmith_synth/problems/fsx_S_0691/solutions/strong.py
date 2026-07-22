# TIER: strong
# The insight: the target is exactly the fixed point of a local RELAXATION invariant --
# "my ring value = 1 + the smallest ring value among my non-wall, non-empty neighbors"
# (the seed cell is a self-sustaining source and always stays at its own state). This
# is literally shortest-path (BFS) distance from the seed, reconstructed by the local
# rule itself instead of being copied from a memorized trace. Because it is an
# invariant satisfied EVERYWHERE in the target (not just along one growth trace), it
# is self-stabilizing under ANY asynchronous sweep order (Bellman-Ford-style
# relaxation is confluent) and it heals wounds: surviving tissue around a hole still
# carries correct distances, and reapplying the SAME rule re-derives the missing
# interior from local context alone, regardless of the wound's shape or position --
# a genuinely reconstructable coordinate system, not a frozen copy.
#
# The table format only supports a fixed "default" fallback (no live formula), so we
# harvest the formula's outputs over: (a) the actual growth trace from the seed, and
# (b) many synthetic "cut a hole, let it heal" drills at varied sizes/positions -- to
# give the emitted table broad practical coverage of what real (hidden) wounds will
# look like, without needing to enumerate the full state space.
import sys, json


def main():
    inst = json.load(sys.stdin)
    H, W = inst["H"], inst["W"]
    seed = tuple(inst["seed"])
    seed_state = inst["seed_state"]
    K = inst["K"]
    wall_state = inst["wall_state"]
    walls = set(tuple(w) for w in inst["walls"])
    n_growth = inst["n_growth_ticks"]
    n_repair = inst.get("n_repair_ticks", n_growth)
    max_entries = inst["max_table_entries"]
    target = inst["target"]

    table = {}

    def formula(self_s, n, s, e, w):
        if self_s == seed_state:
            return seed_state
        best = None
        for v in (n, s, e, w):
            if 1 <= v <= K - 2:
                if best is None or v < best:
                    best = v
        if best is not None and best + 1 <= K - 2:
            return best + 1
        return 0

    def nb(grid, r, c):
        n = grid[r - 1][c] if r - 1 >= 0 else wall_state
        s = grid[r + 1][c] if r + 1 < H else wall_state
        e = grid[r][c + 1] if c + 1 < W else wall_state
        w = grid[r][c - 1] if c - 1 >= 0 else wall_state
        return n, s, e, w

    def record(grid, r, c):
        if len(table) >= max_entries:
            return
        self_s = grid[r][c]
        n, s, e, w = nb(grid, r, c)
        key = "%d,%d,%d,%d,%d" % (self_s, n, s, e, w)
        if key not in table:
            table[key] = formula(self_s, n, s, e, w)

    def init_grid():
        g = [[0] * W for _ in range(H)]
        for (r, c) in walls:
            g[r][c] = wall_state
        g[seed[0]][seed[1]] = seed_state
        return g

    cells = [(r, c) for r in range(H) for c in range(W) if (r, c) not in walls]

    def run_ticks(grid, ticks, rng_seed):
        rstate = [rng_seed]

        def rnd(n):
            rstate[0] = (rstate[0] * 1103515245 + 12345) & 0x7fffffff
            return rstate[0] % n

        for _t in range(ticks):
            order = cells[:]
            for i in range(len(order) - 1, 0, -1):
                j = rnd(i + 1)
                order[i], order[j] = order[j], order[i]
            for (r, c) in order:
                record(grid, r, c)
                self_s = grid[r][c]
                n, s, e, w = nb(grid, r, c)
                grid[r][c] = formula(self_s, n, s, e, w)
        return grid

    grid = init_grid()
    grid = run_ticks(grid, n_growth, 777)

    mask_cells = [(r, c) for r in range(H) for c in range(W)
                  if 0 < target[r][c] < wall_state]

    drill_sizes = [(2, 2), (3, 3), (2, 3), (3, 2), (4, 3), (3, 4), (4, 4), (2, 4), (4, 2), (4, 5), (5, 4)]
    rstate2 = [999331]

    def rnd2(n):
        rstate2[0] = (rstate2[0] * 1103515245 + 12345) & 0x7fffffff
        return rstate2[0] % n

    n_drills = 30
    for d in range(n_drills):
        if not mask_cells or len(table) >= max_entries:
            break
        hh, ww = drill_sizes[d % len(drill_sizes)]
        cr, cc = mask_cells[rnd2(len(mask_cells))]
        r0 = max(0, min(H - hh, cr - hh // 2))
        c0 = max(0, min(W - ww, cc - ww // 2))
        gcopy = [row[:] for row in grid]
        for r in range(r0, r0 + hh):
            for c in range(c0, c0 + ww):
                if (r, c) not in walls:
                    gcopy[r][c] = 0
        run_ticks(gcopy, n_repair, 555 + d * 97)

    print(json.dumps({"table": table, "default": "stay"}))


main()
