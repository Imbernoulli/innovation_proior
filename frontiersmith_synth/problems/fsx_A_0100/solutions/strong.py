# TIER: strong
# Multi-start local search over connected trenches.
#   * Seed from each of the top-K yield cells (one per hotspot region).
#   * From each seed run best-improvement region growing, then a single-cell
#     ADD/REMOVE hill-climb (removals must preserve connectivity) -- this both
#     absorbs profitable neighbours and trims cells whose shoring cost outweighs
#     their yield, and can bridge two hotspots when the bridge pays for itself.
#   * A short seeded random perturbation ("kick") escapes local optima.
# Keep the best connected trench found.  The loose all-artifacts bound keeps the
# normalised score below 1.0, so this still leaves headroom.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
grid = inst["grid"]
lam = inst["lam"]

NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))


def inb(r, c):
    return 0 <= r < N and 0 <= c < N


def value(dug):
    total = 0
    perim = 0
    for (r, c) in dug:
        total += grid[r][c]
        for dr, dc in NEI:
            if (r + dr, c + dc) not in dug:
                perim += 1
    return total - lam * perim


def delta_add(dug, r, c):
    dv = grid[r][c]
    dp = 0
    for dr, dc in NEI:
        if (r + dr, c + dc) in dug:
            dp -= 1
        else:
            dp += 1
    return dv - lam * dp


def delta_remove(dug, r, c):
    f_in = 0
    f_out = 0
    for dr, dc in NEI:
        if (r + dr, c + dc) in dug:
            f_in += 1
        else:
            f_out += 1
    return -grid[r][c] - lam * (f_in - f_out)


def connected_without(dug, cell):
    rest = dug - {cell}
    if not rest:
        return False
    start = next(iter(rest))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEI:
            nb = (r + dr, c + dc)
            if nb in rest and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(rest)


def frontier_of(dug):
    fr = set()
    for (r, c) in dug:
        for dr, dc in NEI:
            nb = (r + dr, c + dc)
            if inb(*nb) and nb not in dug:
                fr.add(nb)
    return fr


def hill_climb(dug, max_moves=4000):
    dug = set(dug)
    moves = 0
    while moves < max_moves:
        best_gain = 1e-9
        best_move = None  # ("add"/"rem", cell)
        for (r, c) in frontier_of(dug):
            g = delta_add(dug, r, c)
            if g > best_gain:
                best_gain = g
                best_move = ("add", (r, c))
        if len(dug) > 1:
            for (r, c) in list(dug):
                g = delta_remove(dug, r, c)
                if g > best_gain and connected_without(dug, (r, c)):
                    best_gain = g
                    best_move = ("rem", (r, c))
        if best_move is None:
            break
        kind, cell = best_move
        if kind == "add":
            dug.add(cell)
        else:
            dug.discard(cell)
        moves += 1
    return dug


# deterministic RNG seeded from the instance content
_seed = (sum((r * 131 + c * 17 + (grid[r][c] & 1023))
             for r in range(N) for c in range(N)) + N * 1000003 + lam * 7) & ((1 << 64) - 1)


def rnd(lo, hi):
    global _seed
    _seed = (_seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return lo + (_seed >> 17) % (hi - lo + 1)


# candidate seed cells: the highest-yield cells
allc = sorted(((grid[r][c], r, c) for r in range(N) for c in range(N)), reverse=True)
seeds = [(r, c) for (_, r, c) in allc[:6]]

best_set = None
best_val = None


def consider(dug):
    global best_set, best_val
    if not dug:
        return
    v = value(dug)
    if best_val is None or v > best_val:
        best_val = v
        best_set = set(dug)


for s in seeds:
    cur = hill_climb({s})
    consider(cur)
    # a few random kicks + re-climb
    for _ in range(12):
        work = set(cur)
        fr = list(frontier_of(work))
        if not fr:
            break
        for _k in range(rnd(1, 3)):
            if fr:
                idx = rnd(0, len(fr) - 1)
                work.add(fr[idx])
                fr = list(frontier_of(work))
        work = hill_climb(work)
        consider(work)
        if best_val is not None and value(work) >= best_val:
            cur = work

if best_set is None:
    best_set = {seeds[0]}

print(json.dumps({"cells": [[r, c] for (r, c) in best_set]}))
