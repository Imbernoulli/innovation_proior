# TIER: strong
"""MRV backtracking for a full gerechte (jigsaw Latin-square) completion, with a
deterministic node budget and best-partial recording.

Strategy:
  * candidate sets per empty cell = channels absent from its row, column, district;
  * always branch on the empty cell with the FEWEST candidates (most-constrained
    variable) -- this fills the tight corners first and reaches deeper than the
    naive row-major greedy;
  * whenever the search dead-ends (or the node budget is hit), greedily fill the
    remaining empties and record the assignment if it beats the best so far;
  * a full solution (all n*n towers tuned) short-circuits immediately.

Deterministic: no wall-clock or randomness; the node budget bounds the work."""
import sys

NODE_BUDGET = 40000


def solve(n, reg, grid):
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    regs = [set() for _ in range(n)]
    empties = []
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v is None:
                empties.append((i, j))
            else:
                rows[i].add(v); cols[j].add(v); regs[reg[i][j]].add(v)

    best = {"count": -1, "grid": None}

    def greedy_record():
        # complete current partial state greedily, count fills, keep if best
        g = [row[:] for row in grid]
        r2 = [set(s) for s in rows]
        c2 = [set(s) for s in cols]
        d2 = [set(s) for s in regs]
        cnt = sum(1 for i in range(n) for j in range(n) if g[i][j] is not None)
        for (i, j) in empties:
            if g[i][j] is not None:
                continue
            d = reg[i][j]
            for v in range(n):
                if v not in r2[i] and v not in c2[j] and v not in d2[d]:
                    g[i][j] = v
                    r2[i].add(v); c2[j].add(v); d2[d].add(v)
                    cnt += 1
                    break
        if cnt > best["count"]:
            best["count"] = cnt
            best["grid"] = g

    greedy_record()  # baseline == plain greedy; search only improves on this

    state = {"nodes": 0, "done": False}

    def candidates(i, j):
        d = reg[i][j]
        return [v for v in range(n)
                if v not in rows[i] and v not in cols[j] and v not in regs[d]]

    def pick_cell():
        best_cell, best_cand, best_len = None, None, 999
        for (i, j) in empties:
            if grid[i][j] is not None:
                continue
            cs = candidates(i, j)
            if len(cs) < best_len:
                best_len, best_cell, best_cand = len(cs), (i, j), cs
                if best_len == 0:
                    break
        return best_cell, best_cand

    def bt(remaining):
        if state["done"]:
            return
        if remaining == 0:
            # full solution
            best["count"] = n * n
            best["grid"] = [row[:] for row in grid]
            state["done"] = True
            return
        state["nodes"] += 1
        if state["nodes"] > NODE_BUDGET:
            greedy_record()
            state["done"] = True
            return
        cell, cs = pick_cell()
        if cell is None:
            return
        if not cs:
            greedy_record()  # dead-end: salvage a partial fill
            return
        i, j = cell
        d = reg[i][j]
        for v in cs:
            grid[i][j] = v
            rows[i].add(v); cols[j].add(v); regs[d].add(v)
            bt(remaining - 1)
            grid[i][j] = None
            rows[i].discard(v); cols[j].discard(v); regs[d].discard(v)
            if state["done"]:
                return

    n_empty = sum(1 for i in range(n) for j in range(n) if grid[i][j] is None)
    try:
        sys.setrecursionlimit(10000)
        bt(n_empty)
    except RecursionError:
        greedy_record()
    return best["grid"]


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

    g = solve(n, reg, grid)
    if g is None:
        g = grid
    out = []
    for i in range(n):
        out.append(" ".join("." if g[i][j] is None else str(g[i][j])
                            for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
