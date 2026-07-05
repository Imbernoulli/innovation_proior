# TIER: greedy
"""Row-major greedy: for each empty cell, build the smallest block that is legal
(not already in the row, not in the column, not forbidden here). No lookahead,
so it dead-ends and leaves cells that could have been filled by a smarter order."""
import sys


def parse():
    tok = sys.stdin.read().split()
    p = 0
    n = int(tok[p]); p += 1
    grid = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            t = tok[p]; p += 1
            grid[i][j] = None if t == "." else int(t)
    forb = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            t = tok[p]; p += 1
            if t != "-":
                forb[i][j] = set(int(x) for x in t.split(","))
    return n, grid, forb


def main():
    n, grid, forb = parse()
    board = [[grid[i][j] for j in range(n)] for i in range(n)]
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if v is not None:
                rows[i].add(v); cols[j].add(v)
    for i in range(n):
        for j in range(n):
            if board[i][j] is not None:
                continue
            for v in range(n):
                if v in rows[i] or v in cols[j] or v in forb[i][j]:
                    continue
                board[i][j] = v; rows[i].add(v); cols[j].add(v)
                break
    out = []
    for i in range(n):
        out.append(" ".join("." if board[i][j] is None else str(board[i][j]) for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
