# TIER: trivial
# Reproduces the checker's internal baseline: on stripe cells (i+j)%5==0 that
# are empty, place the cyclic phase (i+j)%N iff it stays unique in row & column.
import sys

MOD_BASE = 3


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

    out = [row[:] for row in grid]
    for i in range(N):
        for j in range(N):
            if grid[i][j] != -1:
                continue
            if (i + j) % MOD_BASE != 0:
                continue
            s = (i + j) % N
            bit = 1 << s
            if (row_used[i] & bit) or (col_used[j] & bit):
                continue
            row_used[i] |= bit
            col_used[j] |= bit
            out[i][j] = s

    w = sys.stdout.write
    for i in range(N):
        w(" ".join(str(x) for x in out[i]) + "\n")


if __name__ == "__main__":
    main()
