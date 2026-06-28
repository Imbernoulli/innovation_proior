#!/usr/bin/env python3
"""
Instance generator for "Grid Wire Routing" (ale-v2-07).

Usage:  python3 gen.py SEED  > instance.txt

Instance format (stdin of the solver):
    H W K
    H lines, each a string of length W over {'.', '#'}   (# = blocked obstacle cell)
    K lines, each: r1 c1 r2 c2   (the two terminals of net i, 0-indexed, on '.' cells)

Guarantees:
  - 2 <= H, W ; grid not tiny.
  - All 2*K terminals are distinct free ('.') cells.
  - Obstacle density is moderate so the grid stays mostly connected, but the
    nets genuinely contend for cells (this is what makes ordering matter).
The instance is intentionally over-subscribed: K is large enough that a naive
sequential router cannot route all nets, so the heuristic has room to win.
"""
import sys, random


def gen(seed: int):
    rng = random.Random(seed * 1000003 + 12345)

    # Grid size: vary with seed for diversity but keep within a fast budget.
    H = rng.randint(28, 40)
    W = rng.randint(28, 40)

    # Obstacle density: 8%..18%.
    dens = rng.uniform(0.08, 0.18)
    grid = [['.' for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if rng.random() < dens:
                grid[r][c] = '#'

    free = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']
    rng.shuffle(free)

    # Number of nets: chosen so the instance is OVER-SUBSCRIBED but mostly
    # routable -- a strong router fits most nets while a naive one strands a
    # meaningful fraction. Cells used per net ~ Manhattan distance (typ. ~ (H+W)/2),
    # so we scale K to roughly  ratio * free_cells / typ_path_len  with ratio<1.
    typ_path = max(6, (H + W) // 2)
    capacity = len(free) / typ_path
    ratio = rng.uniform(0.85, 1.25)            # mild over-subscription
    K = int(round(ratio * capacity))
    K = max(8, min(K, len(free) // 2))

    nets = []
    used = set()
    idx = 0
    # Pick terminals; encourage non-trivial separation so routes have length.
    free_list = free[:]
    attempts = 0
    while len(nets) < K and attempts < 50 * K:
        attempts += 1
        a = free_list[idx % len(free_list)]
        idx += 1
        if a in used:
            continue
        # second terminal: pick one reasonably far away
        best = None
        for _ in range(12):
            b = free_list[rng.randrange(len(free_list))]
            if b in used or b == a:
                continue
            d = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if best is None or d > best[0]:
                best = (d, b)
        if best is None:
            continue
        b = best[1]
        used.add(a)
        used.add(b)
        nets.append((a, b))

    K = len(nets)

    out = []
    out.append(f"{H} {W} {K}")
    for r in range(H):
        out.append(''.join(grid[r]))
    for (a, b) in nets:
        out.append(f"{a[0]} {a[1]} {b[0]} {b[1]}")
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sys.stdout.write(gen(seed))
