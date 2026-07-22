# TIER: greedy
# The textbook move: generic Latin-square constraint propagation. Repeatedly find an
# empty cell whose row-missing-set intersect column-missing-set has exactly one
# candidate, and fill it (classic "naked single" propagation, the same trick a Sudoku
# solver uses). This is a genuinely reasonable algorithm and does well while the ledger
# is lightly damaged. But it only ever reasons LOCALLY about one row and one column at
# a time -- it has no notion that the WHOLE table is forced by a handful of far-away
# cells (the quadrangle-criterion / group-isotopy structure). Once erasure is heavy,
# most empty cells have SEVERAL locally-consistent candidates and propagation stalls;
# the fallback below just grabs the smallest still-consistent value with no way to tell
# which one is actually correct, so it fills roughly half the destroyed cells wrong on
# the harder instances.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
grid = [row[:] for row in inst["grid"]]

changed = True
while changed:
    changed = False
    row_missing = [set(range(n)) - set(x for x in grid[i] if x is not None) for i in range(n)]
    col_missing = [set(range(n)) - set(grid[i][j] for i in range(n) if grid[i][j] is not None) for j in range(n)]
    for i in range(n):
        for j in range(n):
            if grid[i][j] is None:
                cand = row_missing[i] & col_missing[j]
                if len(cand) == 1:
                    v = next(iter(cand))
                    grid[i][j] = v
                    row_missing[i].discard(v)
                    col_missing[j].discard(v)
                    changed = True

# fallback for cells propagation couldn't pin down: smallest value still locally
# consistent with its row and column (no backtracking, no global reasoning)
for i in range(n):
    for j in range(n):
        if grid[i][j] is None:
            used_row = set(x for x in grid[i] if x is not None)
            used_col = set(grid[k][j] for k in range(n) if grid[k][j] is not None)
            cand = sorted(set(range(n)) - used_row - used_col)
            if cand:
                grid[i][j] = cand[0]
            else:
                rem = sorted(set(range(n)) - used_row)
                grid[i][j] = rem[0] if rem else 0

print(json.dumps({"grid": grid}))
