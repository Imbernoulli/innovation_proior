# TIER: greedy
"""The obvious recipe: every idle truck immediately hauls a full payload from
whichever stocked mine gives the best ore-per-fuel trip rate, the refinery
runs flat out, and capacity is never upgraded (reinvesting fuel 'wastes'
export). Per-trip efficient; blind to compounding, concavity and net-negative
hauls."""
import sys
import math


def read_instance():
    tok = sys.stdin.read().split()
    it = iter(tok)
    T = int(next(it)); K = int(next(it)); M = int(next(it))
    F0 = float(next(it)); C0 = float(next(it)); Cmax = float(next(it))
    cu = float(next(it)); beta = float(next(it)); v = float(next(it))
    P = float(next(it)); kap = float(next(it))
    d = []; S = []
    for _ in range(M):
        d.append(float(next(it))); S.append(float(next(it)))
    return dict(T=T, K=K, M=M, F0=F0, C0=C0, Cmax=Cmax, cu=cu, beta=beta,
                v=v, P=P, kap=kap, d=d, S=S)


def main():
    inst = read_instance()
    T = inst['T']; K = inst['K']; M = inst['M']
    C = inst['C0']; F = inst['F0']; ore = 0.0
    stock = inst['S'][:]
    free = [1] * K
    arr = [[] for _ in range(T + 2)]
    P = inst['P']; kap = inst['kap']; beta = inst['beta']
    tau = [max(1, int(math.ceil(2.0 * inst['d'][i] / inst['v'] - 1e-12)))
           for i in range(M)]
    trips = []
    runs = []
    for t in range(1, T + 1):
        for (i, p) in arr[t]:
            ore += p
        for j in range(K):
            if free[j] > t:
                continue
            best = None
            for i in range(M):
                if stock[i] <= 1e-9:
                    continue
                if t + tau[i] > T:
                    continue
                p = min(P, stock[i])
                cost = kap * inst['d'][i] * (2.0 + p / P)
                if cost > F + 1e-9:
                    continue
                rate = p / cost
                if best is None or rate > best[0] + 1e-18:
                    best = (rate, i, p, cost)
            if best is not None:
                _, i, p, cost = best
                F -= cost
                stock[i] -= p
                rday = t + tau[i]
                free[j] = rday
                trips.append((t, j, i, p))
                arr[rday].append((i, p))
        q = min(ore, C)
        if q > 0.0:
            ore -= q
            F += beta * C * q / (q + C)
            runs.append((t, q))
    out = []
    out.append("0")                       # nU: greedy never reinvests
    out.append(str(len(trips)))
    for (t, j, i, p) in trips:
        out.append("%d %d %d %.6f" % (t, j, i, p))
    out.append(str(len(runs)))
    for (t, q) in runs:
        out.append("%d %.6f" % (t, q))
    sys.stdout.write("\n".join(out) + "\n")


main()
