# TIER: strong
# Minimum-remaining-values (MRV) fill with several seeded restarts: at each step
# stock the most-constrained empty slot using the least-used SKU, break ties by a
# per-restart random order, and keep the best (largest) feasible completion found.
# The row-major greedy pass is included as one candidate, so strong >= greedy.
import sys, random


def read():
    data = [ln.split() for ln in sys.stdin.read().splitlines()]
    data = [t for t in data if t]
    N = int(data[0][0])
    P = [[int(x) for x in data[1 + i]] for i in range(N)]
    allowed = []
    idx = 1 + N
    for _ in range(N * N):
        row = data[idx]; idx += 1
        k = int(row[0])
        allowed.append([int(x) for x in row[1:1 + k]])
    return N, P, allowed


def init_used(N, P):
    rowset = [set() for _ in range(N)]
    colset = [set() for _ in range(N)]
    diaset = [set() for _ in range(N)]
    for r in range(N):
        for c in range(N):
            v = P[r][c]
            if v:
                rowset[r].add(v); colset[c].add(v); diaset[(r + c) % N].add(v)
    return rowset, colset, diaset


def row_major(N, P, allowed):
    S = [row[:] for row in P]
    rowset, colset, diaset = init_used(N, P)
    for r in range(N):
        for c in range(N):
            if S[r][c]:
                continue
            g = (r + c) % N
            for s in sorted(allowed[r * N + c]):
                if s not in rowset[r] and s not in colset[c] and s not in diaset[g]:
                    S[r][c] = s
                    rowset[r].add(s); colset[c].add(s); diaset[g].add(s)
                    break
    return S


def mrv_fill(N, P, allowed, rnd, usage_bias):
    S = [row[:] for row in P]
    rowset, colset, diaset = init_used(N, P)
    empties = [(r, c) for r in range(N) for c in range(N) if not S[r][c]]
    remaining = set(empties)
    changed = True
    while changed:
        changed = True
        # pick the empty slot with fewest feasible SKUs
        best = None
        best_opts = None
        best_key = None
        order = list(remaining)
        rnd.shuffle(order)
        for (r, c) in order:
            g = (r + c) % N
            opts = [s for s in allowed[r * N + c]
                    if s not in rowset[r] and s not in colset[c] and s not in diaset[g]]
            if not opts:
                continue
            key = len(opts)
            if best is None or key < best_key:
                best = (r, c); best_opts = opts; best_key = key
                if key == 1:
                    break
        if best is None:
            break
        r, c = best
        g = (r + c) % N
        # least-used SKU (spread load), tie-broken by seeded shuffle
        rnd.shuffle(best_opts)
        s = min(best_opts, key=lambda x: usage_bias[x])
        S[r][c] = s
        usage_bias[s] += 1
        rowset[r].add(s); colset[c].add(s); diaset[g].add(s)
        remaining.discard((r, c))
    return S


def count(N, S):
    return sum(1 for r in range(N) for c in range(N) if S[r][c])


def main():
    N, P, allowed = read()
    best = row_major(N, P, allowed)
    best_c = count(N, best)
    for t in range(10):
        rnd = random.Random(1234567 + 101 * t)
        usage = [0] * (N + 1)
        S = mrv_fill(N, P, allowed, rnd, usage)
        c = count(N, S)
        if c > best_c:
            best, best_c = S, c
    out = "\n".join(" ".join(str(best[r][c]) for c in range(N)) for r in range(N))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
