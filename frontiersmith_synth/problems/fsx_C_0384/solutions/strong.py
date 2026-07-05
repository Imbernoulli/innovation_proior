# TIER: strong
"""Most-constrained-variable backtracking search for the largest restricted-Latin
completion. Orders empty cells by fewest legal candidates (MRV), searches for a
full completion, and always keeps the best partial board seen (so it never scores
below a decent greedy fallback even when a full completion is not found within the
node budget)."""
import sys
sys.setrecursionlimit(100000)


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

    empties = [(i, j) for i in range(n) for j in range(n) if board[i][j] is None]

    best = {"cnt": sum(1 for i in range(n) for j in range(n) if board[i][j] is not None),
            "board": [row[:] for row in board]}
    nodes = {"c": 0}
    NODE_CAP = 300000

    def cands(i, j):
        f = forb[i][j]; ri = rows[i]; cj = cols[j]
        return [v for v in range(n) if v not in ri and v not in cj and v not in f]

    def snapshot():
        cnt = sum(1 for i in range(n) for j in range(n) if board[i][j] is not None)
        if cnt > best["cnt"]:
            best["cnt"] = cnt
            best["board"] = [row[:] for row in board]

    def dfs(remaining):
        if nodes["c"] > NODE_CAP or best["cnt"] >= n * n:
            return
        nodes["c"] += 1
        if not remaining:
            snapshot()
            return
        # MRV: choose the empty cell with fewest candidates
        bi = -1; bc = None; bn = 10**9
        for idx, (i, j) in enumerate(remaining):
            cs = cands(i, j)
            if len(cs) < bn:
                bn = len(cs); bi = idx; bc = cs
                if bn <= 1:
                    break
        i, j = remaining[bi]
        rest = remaining[:bi] + remaining[bi + 1:]
        if bc:
            for v in bc:
                board[i][j] = v; rows[i].add(v); cols[j].add(v)
                dfs(rest)
                board[i][j] = None; rows[i].discard(v); cols[j].discard(v)
                if best["cnt"] >= n * n or nodes["c"] > NODE_CAP:
                    break
        # also try leaving this cell empty (allows completing the rest when this
        # cell is a dead-end); record the best partial along the way
        board[i][j] = None
        snapshot()
        dfs(rest)

    dfs(empties)
    out_board = best["board"]
    out = []
    for i in range(n):
        out.append(" ".join("." if out_board[i][j] is None else str(out_board[i][j]) for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
