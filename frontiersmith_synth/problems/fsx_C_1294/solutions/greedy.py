# TIER: greedy
# Value-greedy / biggest-first: every year, cut the `quota` currently highest-stage
# cells on the board (ties broken by row, then column). This is the obvious first
# instinct -- always take the most valuable trees available -- and it wins big in
# the opening years. But the maturity threshold for seeding equals the value cap S,
# so every cut of a top-value tree is also a cut of a seed source. Once a region's
# stage-S trees are gone, dispersal there stops forever (growth alone cannot
# manufacture a new seed source), so the plan's yield collapses over the back half
# of the horizon on the dispersal-limited instances -- it never looks ahead to the
# network it is destroying.
import sys, json


def neighbors_within(r, c, R, n):
    r0, r1 = max(0, r - R), min(n - 1, r + R)
    c0, c1 = max(0, c - R), min(n - 1, c + R)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr == r and cc == c:
                continue
            yield rr, cc


def step_world(grid, n, R, S, minseed):
    for rr in range(n):
        for cc in range(n):
            if 1 <= grid[rr][cc] < S:
                grid[rr][cc] += 1
    newly = []
    for rr in range(n):
        for cc in range(n):
            if grid[rr][cc] == 0:
                cnt = 0
                for (nr, nc) in neighbors_within(rr, cc, R, n):
                    if grid[nr][nc] >= S:
                        cnt += 1
                if cnt >= minseed:
                    newly.append((rr, cc))
    for (rr, cc) in newly:
        grid[rr][cc] = 1


inst = json.load(sys.stdin)
n = inst["n"]; R = inst["radius"]; S = inst["s_max"]
minseed = inst["min_seed"]; Q = inst["quota"]; T = inst["horizon"]
grid = [row[:] for row in inst["grid"]]

harvests = []
for _t in range(T):
    cells = [(grid[r][c], r, c) for r in range(n) for c in range(n) if grid[r][c] >= 1]
    cells.sort(key=lambda x: (-x[0], x[1], x[2]))
    picks = [[r, c] for (_, r, c) in cells[:Q]]
    harvests.append(picks)
    for (r, c) in picks:
        grid[r][c] = 0
    step_world(grid, n, R, S, minseed)

print(json.dumps({"harvests": harvests}))
