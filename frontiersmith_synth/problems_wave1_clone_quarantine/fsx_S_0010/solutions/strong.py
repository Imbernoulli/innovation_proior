# TIER: strong
import sys, random

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1

    clauses = []
    for _ in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        lits = [int(d[idx + j]) for j in range(k)]
        idx += k
        clauses.append((w, lits))

    W = [w for (w, _) in clauses]
    lit_lists = [lits for (_, lits) in clauses]

    occ = [[] for _ in range(n + 1)]
    for ci, lits in enumerate(lit_lists):
        for l in lits:
            occ[abs(l)].append((ci, l > 0))

    rng = random.Random(1234567 + n * 131 + m)

    def compute_num_true(x):
        nt = [0] * m
        for ci, lits in enumerate(lit_lists):
            c = 0
            for l in lits:
                if (l > 0 and x[abs(l)] == 1) or (l < 0 and x[abs(l)] == 0):
                    c += 1
            nt[ci] = c
        return nt

    def cleared_weight(nt):
        s = 0
        for ci in range(m):
            if nt[ci] > 0:
                s += W[ci]
        return s

    def flip_delta(x, nt, v):
        delta = 0
        newv = 1 - x[v]
        for (ci, pos) in occ[v]:
            cur_true = (x[v] == 1) == pos
            aft_true = (newv == 1) == pos
            if cur_true == aft_true:
                continue
            if cur_true and not aft_true:
                if nt[ci] == 1:
                    delta -= W[ci]
            else:
                if nt[ci] == 0:
                    delta += W[ci]
        return delta

    def apply_flip(x, nt, v):
        newv = 1 - x[v]
        for (ci, pos) in occ[v]:
            cur_true = (x[v] == 1) == pos
            aft_true = (newv == 1) == pos
            if cur_true and not aft_true:
                nt[ci] -= 1
            elif aft_true and not cur_true:
                nt[ci] += 1
        x[v] = newv

    def hill_climb(x):
        nt = compute_num_true(x)
        improved = True
        sweeps = 0
        while improved and sweeps < 40:
            improved = False
            sweeps += 1
            order = list(range(1, n + 1))
            rng.shuffle(order)
            for v in order:
                if flip_delta(x, nt, v) > 0:
                    apply_flip(x, nt, v)
                    improved = True
        return x, cleared_weight(nt)

    best_x = None
    best_val = -1

    # restart 0: from all-retrograde (>= what greedy explores)
    x0 = [0] * (n + 1)
    xr, val = hill_climb(x0)
    if val > best_val:
        best_val = val; best_x = xr[:]

    # restart from all-prograde
    x1 = [0] + [1] * n
    xr, val = hill_climb(x1)
    if val > best_val:
        best_val = val; best_x = xr[:]

    # random restarts (bounded for time)
    restarts = 8 if m <= 800 else 5
    for _ in range(restarts):
        xr = [0] + [rng.randint(0, 1) for _ in range(n)]
        xr, val = hill_climb(xr)
        if val > best_val:
            best_val = val; best_x = xr[:]

    sys.stdout.write(" ".join(str(best_x[v]) for v in range(1, n + 1)) + "\n")

main()
