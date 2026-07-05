# TIER: strong
# Best-of-N seeded restarts + add/remove local search.
#   * Seed set = the top-value stacks.  From each seed, grow greedily (same rule
#     as the greedy tier), so the strong cordon is always >= the greedy cordon.
#   * Then, over many SEEDED restarts, run local search: sweep the frontier adding
#     every profit-increasing cell, and peel off any interior cell whose removal
#     increases profit while keeping the zone connected.  Randomised restart seeds
#     and frontier orderings (deterministic RNG) let it escape the single-seed
#     basin the greedy rule is stuck in.
# Still far below the optimistic per-cell bound the evaluator normalises against,
# so it leaves headroom.  All randomness is seeded -> deterministic score.
import sys, json, random

inst = json.load(sys.stdin)
H, W, lam = inst["H"], inst["W"], inst["lam"]
grid = inst["grid"]
rnd = random.Random(1234567)


def nbrs(i, j):
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W:
            yield ni, nj


def perim_profit(S):
    per = 0
    tot = 0
    for (i, j) in S:
        tot += grid[i][j]
        for (ni, nj) in nbrs(i, j):
            if (ni, nj) not in S:
                per += 1
        # boundary-of-yard edges also count
        per += 4 - sum(1 for _ in nbrs(i, j))
    return tot - lam * per


def grow(seed):
    S = {seed}
    frontier = {}
    for n in nbrs(*seed):
        frontier[n] = 1
    while frontier:
        best_delta = 0
        best_c = None
        for c, a in frontier.items():
            delta = grid[c[0]][c[1]] - lam * (4 - 2 * a)
            if delta > best_delta:
                best_delta = delta
                best_c = c
        if best_c is None:
            break
        S.add(best_c)
        del frontier[best_c]
        for n in nbrs(*best_c):
            if n not in S:
                frontier[n] = frontier.get(n, 0) + 1
    return S


def connected(S):
    if not S:
        return False
    start = next(iter(S))
    seen = {start}
    stack = [start]
    while stack:
        i, j = stack.pop()
        for n in nbrs(i, j):
            if n in S and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(S)


# candidate seeds = top-value stacks
cells_sorted = sorted(((grid[i][j], (i, j)) for i in range(H) for j in range(W)),
                      reverse=True)
seeds = [c for _, c in cells_sorted[:max(6, (H * W) // 12)]]

best_S = None
best_obj = None
for sc in seeds:
    S = grow(sc)
    o = perim_profit(S)
    if best_obj is None or o > best_obj:
        best_obj = o
        best_S = set(S)

for _ in range(30):
    sc = rnd.choice(seeds)
    S = set(grow(sc))
    improved = True
    rounds = 0
    while improved and rounds < 6:
        improved = False
        rounds += 1
        frontier = {}
        for (i, j) in S:
            for (ni, nj) in nbrs(i, j):
                if (ni, nj) not in S:
                    frontier[(ni, nj)] = frontier.get((ni, nj), 0) + 1
        order = list(frontier.items())
        rnd.shuffle(order)
        for c, a in order:
            if grid[c[0]][c[1]] - lam * (4 - 2 * a) > 0:
                S.add(c)
                improved = True
        for c in list(S):
            if len(S) <= 1:
                break
            a = sum(1 for n in nbrs(*c) if n in S)
            if -(grid[c[0]][c[1]] - lam * (4 - 2 * a)) > 0:
                S2 = set(S)
                S2.discard(c)
                if S2 and connected(S2):
                    S = S2
                    improved = True
    o = perim_profit(S)
    if o > best_obj:
        best_obj = o
        best_S = set(S)

print(json.dumps({"cells": [[i, j] for (i, j) in best_S]}))
