# TIER: strong
import sys, random

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    D = int(d[idx]); idx += 1

    W = [0] * m
    pairs_of = [None] * m
    for ci in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        pr = []
        for _ in range(k):
            v = int(d[idx]); a = int(d[idx + 1]); idx += 2
            pr.append((v, a))
        W[ci] = w
        pairs_of[ci] = pr

    occ = [[] for _ in range(n + 1)]
    for ci in range(m):
        for (v, a) in pairs_of[ci]:
            occ[v].append((ci, a))

    rng = random.Random(987654321 + n * 131 + m * 7 + D)

    def compute_sat(x):
        sat = [0] * m
        for ci in range(m):
            c = 0
            for (v, a) in pairs_of[ci]:
                if x[v] == a:
                    c += 1
            sat[ci] = c
        return sat

    def cleared_weight(sat):
        s = 0
        for ci in range(m):
            if sat[ci] > 0:
                s += W[ci]
        return s

    def best_config(x, sat, v):
        cur = x[v]
        cand = [0] * D
        for (ci, a) in occ[v]:
            others = sat[ci] - (1 if x[v] == a else 0)
            w = W[ci]
            if others > 0:
                for cfg in range(D):
                    cand[cfg] += w
            else:
                cand[a] += w
        bv = cur; bg = 0
        for cfg in range(D):
            g = cand[cfg] - cand[cur]
            if g > bg:
                bg = g; bv = cfg
        return bv, bg

    def apply_cfg(x, sat, v, newv):
        cur = x[v]
        if cur == newv:
            return
        for (ci, a) in occ[v]:
            if a == cur:
                sat[ci] -= 1
            if a == newv:
                sat[ci] += 1
        x[v] = newv

    def hill_climb(x):
        sat = compute_sat(x)
        improved = True
        sweeps = 0
        while improved and sweeps < 40:
            improved = False
            sweeps += 1
            order = list(range(1, n + 1))
            rng.shuffle(order)
            for v in order:
                bv, bg = best_config(x, sat, v)
                if bg > 0:
                    apply_cfg(x, sat, v, bv)
                    improved = True
        return x, cleared_weight(sat)

    best_x = None
    best_val = -1

    # restart 0: from all-default
    x0 = [0] * (n + 1)
    xr, val = hill_climb(x0)
    if val > best_val:
        best_val = val; best_x = xr[:]

    # criticality-aware init: for each module pick config demanded by its heaviest touching clause
    xc = [0] * (n + 1)
    for v in range(1, n + 1):
        bw = -1; ba = 0
        for (ci, a) in occ[v]:
            if W[ci] > bw:
                bw = W[ci]; ba = a
        xc[v] = ba
    xr, val = hill_climb(xc)
    if val > best_val:
        best_val = val; best_x = xr[:]

    # random restarts (bounded for time)
    restarts = 10 if m <= 700 else 6
    for _ in range(restarts):
        xr = [0] + [rng.randint(0, D - 1) for _ in range(n)]
        xr, val = hill_climb(xr)
        if val > best_val:
            best_val = val; best_x = xr[:]

    sys.stdout.write(" ".join(str(best_x[v]) for v in range(1, n + 1)) + "\n")

main()
