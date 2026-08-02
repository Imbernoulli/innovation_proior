# TIER: strong
# Seed-network-preserving harvest. Still cuts biggest-first by value, but before
# committing to cut a mature (stage == S) cell it checks: does any not-yet-mature
# cell within dispersal `radius` depend on THIS tree as one of its last remaining
# seed sources (i.e. would its count of surviving stage-S neighbours drop below
# `min_seed` once this cut, and any other cuts already committed this year, are
# applied)? If so, the cut is UNSAFE and is skipped in favour of the next-best
# available cell that year.
#
# This is the genuine insight the family points at: instead of maximizing this
# year's yield, maintain the invariant "every regenerating cell keeps enough
# spatially-reachable mature neighbours" -- a small, spatially-spread set of
# mature trees is kept standing forever as permanent infrastructure, and
# everything else is rotated through around it. Because the network keeps
# reseeding the stand, the plan can keep cutting near-maximum-value trees for the
# WHOLE horizon instead of just the opening years.
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
    mature_set = set((r, c) for r in range(n) for c in range(n) if grid[r][c] >= S)
    # at-risk cells: non-mature cells whose current mature-neighbour count is
    # already at or below the threshold (losing one more source is dangerous)
    at_risk = {}
    for r in range(n):
        for c in range(n):
            if grid[r][c] < S:
                cnt = sum(1 for (nr, nc) in neighbors_within(r, c, R, n) if (nr, nc) in mature_set)
                # only the exact boundary matters: cnt > minseed has slack (losing one
                # still clears the bar) and cnt < minseed is already a lost cause this
                # round (protecting a neighbour cannot rescue it), so only cnt==minseed
                # cells are genuinely one-cut-away from losing dispersal coverage.
                if cnt == minseed:
                    at_risk[(r, c)] = cnt

    cells = [(grid[r][c], r, c) for r in range(n) for c in range(n) if grid[r][c] >= 1]
    cells.sort(key=lambda x: (-x[0], x[1], x[2]))

    picks = []
    removed_mature = set()
    for (_val, r, c) in cells:
        if len(picks) >= Q:
            break
        if (r, c) in mature_set:
            unsafe = False
            for (nr, nc) in neighbors_within(r, c, R, n):
                if (nr, nc) in at_risk:
                    remaining = sum(
                        1 for (n2r, n2c) in neighbors_within(nr, nc, R, n)
                        if (n2r, n2c) in mature_set
                        and (n2r, n2c) not in removed_mature
                        and (n2r, n2c) != (r, c)
                    )
                    if remaining < minseed:
                        unsafe = True
                        break
            if unsafe:
                continue
            removed_mature.add((r, c))
        picks.append([r, c])

    harvests.append(picks)
    for (r, c) in picks:
        grid[r][c] = 0
    step_world(grid, n, R, S, minseed)

print(json.dumps({"harvests": harvests}))
