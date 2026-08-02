# TIER: trivial
# Positional FIFO harvest: each year cut the first `quota` harvestable cells found
# in row-major scan order, completely ignoring value and regrowth. This is exactly
# the evaluator's weak reference plan, so it reproduces the 0.1 anchor.
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
    picks = []
    for r in range(n):
        for c in range(n):
            if len(picks) >= Q:
                break
            if grid[r][c] >= 1:
                picks.append([r, c])
        if len(picks) >= Q:
            break
    harvests.append(picks)
    for (r, c) in picks:
        grid[r][c] = 0
    step_world(grid, n, R, S, minseed)

print(json.dumps({"harvests": harvests}))
