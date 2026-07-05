# TIER: greedy
"""Row-major smallest-channel greedy. Scan cells in row-major order; for each
empty cell tune it to the smallest channel that violates no row/column/district
rule (given current state). Never backtracks -> dead-ends and leaves many towers
empty. Beats the "tune nothing" baseline but is far from optimal."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    reg = [[int(next(it)) for _ in range(n)] for _ in range(n)]
    grid = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            tok = next(it)
            grid[i][j] = None if tok in (".", "-1") else int(tok)

    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    regs = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v is not None:
                rows[i].add(v); cols[j].add(v); regs[reg[i][j]].add(v)

    for i in range(n):
        for j in range(n):
            if grid[i][j] is not None:
                continue
            d = reg[i][j]
            for v in range(n):
                if v not in rows[i] and v not in cols[j] and v not in regs[d]:
                    grid[i][j] = v
                    rows[i].add(v); cols[j].add(v); regs[d].add(v)
                    break

    out = []
    for i in range(n):
        out.append(" ".join("." if grid[i][j] is None else str(grid[i][j])
                            for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
