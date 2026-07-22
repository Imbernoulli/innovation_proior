# TIER: greedy
# The obvious recipe: simulate ONE forward growth trace from the seed, and for every
# dead cell that activates, memorize "this exact 5-tuple -> the neighbor state I
# copied from, plus one" (first alive neighbor found in a FIXED N,S,E,W priority
# order). Once a cell is alive it is left to the "stay" default forever (freeze --
# never revisit an already-alive cell). This usually reproduces the target shape on
# the FIRST growth pass (a monotone single-source wave typically only offers the
# truly-nearest neighbor at the moment a cell first activates). But it is not an
# invariant: it never re-derives a cell's value from its neighbors, so when a wound
# reopens INSIDE the body, the boundary of the hole offers MULTIPLE already-settled
# neighbors of different distances simultaneously, in a pattern the single forward
# trace never saw -- "first neighbor in fixed order" is frequently not the *nearest*
# one, and unseen patterns silently fall back to "stay" (no regrowth at all).
import sys, json


def main():
    inst = json.load(sys.stdin)
    H, W = inst["H"], inst["W"]
    seed = tuple(inst["seed"])
    seed_state = inst["seed_state"]
    K = inst["K"]
    wall_state = inst["wall_state"]
    walls = set(tuple(w) for w in inst["walls"])
    n_ticks = inst["n_growth_ticks"]
    max_entries = inst["max_table_entries"]

    grid = [[0] * W for _ in range(H)]
    for (r, c) in walls:
        grid[r][c] = wall_state
    grid[seed[0]][seed[1]] = seed_state

    table = {}

    state = [123456789]

    def rnd(n):
        state[0] = (state[0] * 1103515245 + 12345) & 0x7fffffff
        return state[0] % n

    cells = [(r, c) for r in range(H) for c in range(W) if (r, c) not in walls]

    def nb(r, c):
        n = grid[r - 1][c] if r - 1 >= 0 else wall_state
        s = grid[r + 1][c] if r + 1 < H else wall_state
        e = grid[r][c + 1] if c + 1 < W else wall_state
        w = grid[r][c - 1] if c - 1 >= 0 else wall_state
        return n, s, e, w

    for _t in range(n_ticks):
        order = cells[:]
        for i in range(len(order) - 1, 0, -1):
            j = rnd(i + 1)
            order[i], order[j] = order[j], order[i]
        for (r, c) in order:
            self_s = grid[r][c]
            if self_s != 0:
                continue  # already alive (or wall): frozen, rely on "stay" default
            n, s, e, w = nb(r, c)
            key = "%d,%d,%d,%d,%d" % (self_s, n, s, e, w)
            nxt = None
            for cand in (n, s, e, w):  # fixed priority order N,S,E,W
                if 1 <= cand <= K - 2:
                    nxt = cand + 1
                    break
            if nxt is not None and nxt <= K - 2:
                grid[r][c] = nxt
                if key not in table and len(table) < max_entries:
                    table[key] = nxt
            else:
                if key not in table and len(table) < max_entries:
                    table[key] = 0

    print(json.dumps({"table": table, "default": "stay"}))


main()
