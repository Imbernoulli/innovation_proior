# TIER: greedy
# Hottest-first incremental placement.  Sort jobs by descending load, then drop each
# job onto the currently-free slot that adds the LEAST squared over-temperature given
# the deposits already placed.  This discounts overlap (it "sees" the heat already in
# a slot's 3x3 footprint) and prefers strong-vent slots, so hot jobs spread apart and
# onto cooling -- clearing the row-major baseline comfortably.  It is myopic, though:
# it never revisits an early placement, so it leaves penalty on the table that swap /
# relocation local search (the strong tier) can still recover.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; J = inst["j"]; loads = inst["loads"]; cool = inst["cool"]
kernel = inst["kernel"]

dep = [[0] * n for _ in range(n)]


def delta_place(r, c, w):
    """Increase in sum-of-squared-overheat from adding a job of load w at (r,c)."""
    d = 0
    for dr in (-1, 0, 1):
        rr = r + dr
        if rr < 0 or rr >= n:
            continue
        krow = kernel[dr + 1]
        for dc in (-1, 0, 1):
            cc = c + dc
            if cc < 0 or cc >= n:
                continue
            add = w * krow[dc + 1]
            old = dep[rr][cc] - cool[rr][cc]
            oldp = old * old if old > 0 else 0
            new = old + add
            newp = new * new if new > 0 else 0
            d += newp - oldp
    return d


def commit(r, c, w):
    for dr in (-1, 0, 1):
        rr = r + dr
        if rr < 0 or rr >= n:
            continue
        krow = kernel[dr + 1]
        for dc in (-1, 0, 1):
            cc = c + dc
            if cc < 0 or cc >= n:
                continue
            dep[rr][cc] += w * krow[dc + 1]


order = sorted(range(J), key=lambda jj: (-loads[jj], jj))
used = [[False] * n for _ in range(n)]
place = [None] * J

for jj in order:
    w = loads[jj]
    best = None; best_d = None
    for r in range(n):
        for c in range(n):
            if used[r][c]:
                continue
            d = delta_place(r, c, w)
            if best_d is None or d < best_d or (d == best_d and (r, c) < best):
                best_d = d; best = (r, c)
    r, c = best
    used[r][c] = True
    place[jj] = [r, c]
    commit(r, c, w)

print(json.dumps({"place": place}))
