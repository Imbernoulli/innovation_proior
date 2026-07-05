# TIER: strong
# Minimum-Remaining-Values (most-constrained-cell-first) fill.  Repeatedly pick
# the empty cell with the fewest available phases (>0), assign the phase that is
# rarest among the cell's row+column peers, and update.  MRV ordering avoids the
# dead-ends that trip up the naive row-major greedy, so it validly fills more
# cells on dense instances.
import sys


def popcount(x):
    return bin(x).count("1")


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it))
    grid = [[int(next(it)) for _ in range(N)] for _ in range(N)]

    row_used = [0] * N
    col_used = [0] * N
    for i in range(N):
        for j in range(N):
            v = grid[i][j]
            if v != -1:
                row_used[i] |= (1 << v)
                col_used[j] |= (1 << v)

    full = (1 << N) - 1
    out = [row[:] for row in grid]

    empties = [(i, j) for i in range(N) for j in range(N) if grid[i][j] == -1]

    remaining = set(empties)
    while remaining:
        best = None
        best_avail = None
        best_cnt = None
        for (i, j) in remaining:
            avail = full & ~row_used[i] & ~col_used[j]
            c = popcount(avail)
            if c == 0:
                continue
            # tie-break deterministically by (count, i, j)
            if best is None or c < best_cnt or (c == best_cnt and (i, j) < best):
                best = (i, j)
                best_cnt = c
                best_avail = avail
        if best is None:
            break
        i, j = best
        # choose the available phase whose bit is lowest (deterministic)
        s = (best_avail & (-best_avail)).bit_length() - 1
        bit = 1 << s
        row_used[i] |= bit
        col_used[j] |= bit
        out[i][j] = s
        remaining.discard(best)
        # drop cells that just became unfillable to keep the loop moving
        # (they will be re-checked lazily; cheap for these sizes)

    w = sys.stdout.write
    for i in range(N):
        w(" ".join(str(x) for x in out[i]) + "\n")


if __name__ == "__main__":
    main()
