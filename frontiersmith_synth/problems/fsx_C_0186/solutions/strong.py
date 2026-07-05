# TIER: strong
# Maximize the number of validly installed turbines.
#  (1) MRV backtracking search for a FULL Latin completion (bounded node budget),
#      recording the best partial hit at dead-ends;
#  (2) two greedy fallbacks (row-major and most-constrained-variable);
#  (3) emit whichever candidate installs the most turbines.
# Guaranteed >= the row-major greedy baseline, and usually much better.
import sys
sys.setrecursionlimit(100000)


def read_grid():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    vals = tok[1:1 + n * n]
    G = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            t = vals[idx]; idx += 1
            G[i][j] = -1 if t == '.' else int(t)
    return n, G


def count_filled(S, n):
    return sum(1 for i in range(n) for j in range(n) if S[i][j] != -1)


def make_sets(S, n):
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if S[i][j] != -1:
                rows[i].add(S[i][j])
                cols[j].add(S[i][j])
    return rows, cols


def rowmajor_fill(G, n):
    S = [row[:] for row in G]
    rows, cols = make_sets(S, n)
    for i in range(n):
        for j in range(n):
            if S[i][j] == -1:
                for v in range(n):
                    if v not in rows[i] and v not in cols[j]:
                        S[i][j] = v; rows[i].add(v); cols[j].add(v)
                        break
    return S


def mrv_fill(G, n):
    S = [row[:] for row in G]
    rows, cols = make_sets(S, n)
    while True:
        pick = None; pb = n + 1
        for i in range(n):
            for j in range(n):
                if S[i][j] == -1:
                    a = [v for v in range(n) if v not in rows[i] and v not in cols[j]]
                    if a and len(a) < pb:
                        pb = len(a); pick = (i, j, a)
                        if pb == 1:
                            break
            if pick is not None and pb == 1:
                break
        if pick is None:
            break
        i, j, a = pick
        v = a[0]
        S[i][j] = v; rows[i].add(v); cols[j].add(v)
    return S


def backtrack_full(G, n, budget=15000):
    S = [row[:] for row in G]
    rows, cols = make_sets(S, n)
    empties = [(i, j) for i in range(n) for j in range(n) if S[i][j] == -1]
    best = {"S": None, "c": count_filled(S, n)}
    nodes = {"k": 0}

    def record():
        c = count_filled(S, n)
        if c > best["c"]:
            best["c"] = c
            best["S"] = [row[:] for row in S]

    def dfs():
        nodes["k"] += 1
        if nodes["k"] > budget:
            return None  # give up on full completion
        # MRV over remaining empties
        cand = None; cb = n + 1
        for (i, j) in empties:
            if S[i][j] == -1:
                a = [v for v in range(n) if v not in rows[i] and v not in cols[j]]
                if len(a) < cb:
                    cb = len(a); cand = (i, j, a)
                    if cb == 0:
                        break
        if cand is None:
            return [row[:] for row in S]  # full solution
        i, j, a = cand
        if not a:
            record()
            return False  # dead-end for full completion
        for v in a:
            S[i][j] = v; rows[i].add(v); cols[j].add(v)
            res = dfs()
            if res is None:
                S[i][j] = -1; rows[i].discard(v); cols[j].discard(v)
                return None  # budget exhausted, propagate up
            if res is not False:
                return res  # full solution found
            S[i][j] = -1; rows[i].discard(v); cols[j].discard(v)
        record()
        return False

    res = dfs()
    if isinstance(res, list):
        return res
    return best["S"]  # best partial found (may be None)


def main():
    n, G = read_grid()
    candidates = [rowmajor_fill(G, n), mrv_fill(G, n)]
    bt = backtrack_full(G, n)
    if bt is not None:
        candidates.append(bt)
    best = max(candidates, key=lambda S: count_filled(S, n))

    lines = []
    for i in range(n):
        lines.append(" ".join("." if best[i][j] == -1 else str(best[i][j])
                              for j in range(n)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
