# TIER: greedy
# Single row-major pass: for each empty cell place the SMALLEST phase that is
# not yet used in its row or column.  Fills many cells but can paint itself into
# dead-ends near the end of rows/columns.
import sys


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
    for i in range(N):
        for j in range(N):
            if grid[i][j] != -1:
                continue
            avail = full & ~row_used[i] & ~col_used[j]
            if avail == 0:
                continue
            s = (avail & (-avail)).bit_length() - 1   # smallest available
            bit = 1 << s
            row_used[i] |= bit
            col_used[j] |= bit
            out[i][j] = s

    w = sys.stdout.write
    for i in range(N):
        w(" ".join(str(x) for x in out[i]) + "\n")


if __name__ == "__main__":
    main()
