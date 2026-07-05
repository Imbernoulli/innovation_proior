# TIER: strong
import sys, random

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1

    W = [0] * m
    T = [0] * m
    lit_lists = [None] * m
    for ci in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        t = int(d[idx]); idx += 1
        lits = [int(d[idx + j]) for j in range(k)]
        idx += k
        W[ci] = w; T[ci] = t; lit_lists[ci] = lits

    occ = [[] for _ in range(n + 1)]
    for ci in range(m):
        for l in lit_lists[ci]:
            occ[abs(l)].append((ci, l > 0))

    rng = random.Random(778899 + n * 131 + m)

    def compute_c(x):
        c = [0] * m
        for ci in range(m):
            cnt = 0
            for l in lit_lists[ci]:
                if (l > 0 and x[abs(l)] == 1) or (l < 0 and x[abs(l)] == 0):
                    cnt += 1
            c[ci] = cnt
        return c

    def cleared_weight(c):
        s = 0
        for ci in range(m):
            if c[ci] >= T[ci]:
                s += W[ci]
        return s

    def flip_delta(x, c, v):
        newv = 1 - x[v]
        delta = 0
        for (ci, pos) in occ[v]:
            cur_true = (x[v] == 1) == pos
            aft_true = (newv == 1) == pos
            if cur_true == aft_true:
                continue
            if cur_true and not aft_true:
                if c[ci] == T[ci]:
                    delta -= W[ci]
            else:
                if c[ci] == T[ci] - 1:
                    delta += W[ci]
        return delta

    def apply_flip(x, c, v):
        newv = 1 - x[v]
        for (ci, pos) in occ[v]:
            cur_true = (x[v] == 1) == pos
            aft_true = (newv == 1) == pos
            if cur_true and not aft_true:
                c[ci] -= 1
            elif aft_true and not cur_true:
                c[ci] += 1
        x[v] = newv

    def hill_climb(x):
        c = compute_c(x)
        improved = True
        sweeps = 0
        while improved and sweeps < 40:
            improved = False
            sweeps += 1
            order = list(range(1, n + 1))
            rng.shuffle(order)
            for v in order:
                if flip_delta(x, c, v) > 0:
                    apply_flip(x, c, v)
                    improved = True
        return x, cleared_weight(c)

    best_x = None
    best_val = -1

    # restart 0: from all-production (>= what greedy explores)
    xr, val = hill_climb([0] * (n + 1))
    if val > best_val:
        best_val = val; best_x = xr[:]

    # restart from all-injection
    xr, val = hill_climb([0] + [1] * n)
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
