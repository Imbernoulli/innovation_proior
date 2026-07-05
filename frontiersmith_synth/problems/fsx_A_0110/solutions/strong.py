# TIER: strong
# Multi-seed region growing + add-only simulated annealing.
#
#   1) Grow a greedy region (best positive-gain annexation) from EACH science
#      hotspot seed, not just the global-best cell -- the most valuable footprint
#      often sits around a different hotspot cluster.  This already dominates the
#      single-seed greedy baseline (that run is one of the seeds).
#   2) From the best region found, run seeded ADD-ONLY simulated annealing: the
#      region only ever grows (so 4-connectivity is preserved for free), but the
#      annealer will accept small NEGATIVE-gain annexations with probability
#      exp(gain/T).  This lets it pay a short "bridge" of mildly-negative ice to
#      reach a nearby hotspot cluster that pure greedy would never touch, while a
#      best-net snapshot keeps the strongest footprint seen along the way.
#
# The loose positive-mass bound (which ignores connectivity and the perimeter
# penalty) keeps even this normalized score strictly below 1.0 -> real headroom.
import sys, json, math

inst = json.load(sys.stdin)
H = inst["H"]
W = inst["W"]
grid = inst["grid"]
pc = inst["perim_cost"]


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return rnd


rnd = _rng(20240110)


def neighbours(r, c):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            yield nr, nc


def net_of(region):
    total = 0
    perim = 0
    for (r, c) in region:
        total += grid[r][c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or (nr, nc) not in region:
                perim += 1
    return total - pc * perim


def grow(seed):
    """Greedy positive-gain region growing from a single seed cell."""
    chosen = {seed}
    frontier = set(nb for nb in neighbours(*seed))
    net = grid[seed[0]][seed[1]] - 4 * pc
    while frontier:
        best_gain = 0
        best_cell = None
        for (r, c) in frontier:
            k = sum(1 for nb in neighbours(r, c) if nb in chosen)
            gain = grid[r][c] - pc * (4 - 2 * k)
            if gain > best_gain:
                best_gain = gain
                best_cell = (r, c)
        if best_cell is None:
            break
        chosen.add(best_cell)
        net += best_gain
        frontier.discard(best_cell)
        for nb in neighbours(*best_cell):
            if nb not in chosen:
                frontier.add(nb)
    return chosen, net


# ---- 1) multi-seed greedy ----
seeds = [(r, c) for r in range(H) for c in range(W) if grid[r][c] > 0]
seeds.sort(key=lambda rc: grid[rc[0]][rc[1]], reverse=True)
if not seeds:
    # no positive cells: fall back to global-best single cell
    br = bc = 0
    best = None
    for r in range(H):
        for c in range(W):
            if best is None or grid[r][c] > best:
                best = grid[r][c]
                br, bc = r, c
    seeds = [(br, bc)]
seeds = seeds[:14]

best_region = None
best_net = None
for s in seeds:
    reg, net = grow(s)
    if best_net is None or net > best_net:
        best_net = net
        best_region = set(reg)


# ---- 2) add-only simulated annealing restarts from the best region ----
def anneal(base_region, base_net, steps, t0):
    global best_region, best_net
    chosen = set(base_region)
    net = base_net
    frontier = set()
    for cell in chosen:
        for nb in neighbours(*cell):
            if nb not in chosen:
                frontier.add(nb)
    flist = list(frontier)
    for i in range(steps):
        if not flist:
            break
        T = t0 * (1.0 - i / steps) + 1e-3
        j = int(rnd() * len(flist))
        r, c = flist[j]
        if (r, c) in chosen:
            # stale frontier entry (a cell can be queued more than once) -> drop
            flist[j] = flist[-1]
            flist.pop()
            continue
        k = sum(1 for nb in neighbours(r, c) if nb in chosen)
        gain = grid[r][c] - pc * (4 - 2 * k)
        accept = gain > 0 or rnd() < math.exp(gain / T)
        if accept:
            chosen.add((r, c))
            net += gain
            # swap-remove from flist
            flist[j] = flist[-1]
            flist.pop()
            for nb in neighbours(r, c):
                if nb not in chosen:
                    flist.append(nb)
            if net > best_net:
                best_net = net
                best_region = set(chosen)


restart_steps = min(2200, 3 * H * W)
for t0 in (6.0, 4.0, 2.5):
    anneal(best_region, best_net, restart_steps, t0)

print(json.dumps({"cells": [[r, c] for (r, c) in best_region]}))
