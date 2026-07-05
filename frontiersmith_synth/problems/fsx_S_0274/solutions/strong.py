# TIER: strong
import sys, random

def parse(d):
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    D = int(d[idx]); idx += 1
    W = [0] * m
    lits = [None] * m
    for c in range(m):
        w = int(d[idx]); idx += 1
        k = int(d[idx]); idx += 1
        ll = []
        for _ in range(k):
            v = int(d[idx]); a = int(d[idx + 1]); idx += 2
            ll.append((v, a))
        W[c] = w
        lits[c] = ll
    return n, m, D, W, lits

def main():
    d = sys.stdin.buffer.read().split()
    n, m, D, W, lits = parse(d)

    occ = [[] for _ in range(n + 1)]
    for c in range(m):
        for (v, a) in lits[c]:
            occ[v].append((c, a))

    rng = random.Random(9871 + n * 131 + m * 7 + D)

    def compute_num_true(x):
        nt = [0] * m
        for c in range(m):
            cnt = 0
            for (v, a) in lits[c]:
                if x[v] == a:
                    cnt += 1
            nt[c] = cnt
        return nt

    def cleared(nt):
        s = 0
        for c in range(m):
            if nt[c] > 0:
                s += W[c]
        return s

    def best_channel(x, nt, v):
        aold = x[v]
        const_part = 0
        bonus = [0] * (D + 1)
        for (c, a) in occ[v]:
            base = nt[c] - (1 if aold == a else 0)
            if base > 0:
                const_part += W[c]
            else:
                bonus[a] += W[c]
        best_a = aold
        best_val = bonus[aold]
        for a in range(1, D + 1):
            if bonus[a] > best_val:
                best_val = bonus[a]; best_a = a
        gain = bonus[best_a] - bonus[aold]
        return best_a, gain

    def apply(x, nt, v, newa):
        aold = x[v]
        if newa == aold:
            return
        for (c, a) in occ[v]:
            if aold == a:
                nt[c] -= 1
            if newa == a:
                nt[c] += 1
        x[v] = newa

    def hill_climb(x):
        nt = compute_num_true(x)
        sweeps = 0
        improved = True
        while improved and sweeps < 40:
            improved = False
            sweeps += 1
            order = list(range(1, n + 1))
            rng.shuffle(order)
            for v in order:
                na, gain = best_channel(x, nt, v)
                if gain > 0:
                    apply(x, nt, v, na)
                    improved = True
        return x, cleared(nt)

    best_x = None
    best_val = -1

    # restart 0: from all channel 1 (dominates the greedy baseline)
    x0 = [1] * (n + 1)
    xr, val = hill_climb(x0)
    if val > best_val:
        best_val = val; best_x = xr[:]

    # random restarts (bounded for time)
    restarts = 12 if m <= 800 else 7
    for _ in range(restarts):
        xr = [0] + [rng.randint(1, D) for _ in range(n)]
        xr, val = hill_climb(xr)
        if val > best_val:
            best_val = val; best_x = xr[:]

    sys.stdout.write(" ".join(str(best_x[v]) for v in range(1, n + 1)) + "\n")

main()
