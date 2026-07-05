# TIER: greedy
"""Naive row-major fill: scan cells in reading order, drop in the smallest depot
that fits the row and column right now. No lookahead, so early choices trap later
cells and leave gaps."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    g = [[int(toks[idx + i * N + j]) for j in range(N)] for i in range(N)]

    rows = [set() for _ in range(N)]
    cols = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            v = g[i][j]
            if v != 0:
                rows[i].add(v)
                cols[j].add(v)

    for i in range(N):
        for j in range(N):
            if g[i][j] == 0:
                for s in range(1, N + 1):
                    if s not in rows[i] and s not in cols[j]:
                        g[i][j] = s
                        rows[i].add(s)
                        cols[j].add(s)
                        break

    out = []
    for i in range(N):
        out.append(' '.join(str(g[i][j]) for j in range(N)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
