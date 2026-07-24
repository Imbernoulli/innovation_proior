# TIER: trivial
"""Do-little baseline: truck 0 shuttles the nearest stocked mine at half
payload, half-throttle refining, never upgrades. Reproduces the checker's
internal baseline construction exactly."""
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


def tau_of(inst, i):
    return max(1, int(math.ceil(2.0 * inst['d'][i] / inst['v'] - 1e-12)))


def main():
    inst = read_instance()
    T = inst['T']; C = inst['C0']; F = inst['F0']; ore = 0.0
    stock = inst['S'][:]
    free = 1
    arr = [[] for _ in range(T + 2)]
    P = inst['P']; kap = inst['kap']; beta = inst['beta']
    order = sorted(range(inst['M']), key=lambda i: (inst['d'][i], i))
    trips = []
    runs = []
    for t in range(1, T + 1):
        for (i, p) in arr[t]:
            ore += p
        if free <= t:
            for i in order:
                if stock[i] > 1e-9:
                    p = min(P / 2.0, stock[i])
                    cost = kap * inst['d'][i] * (2.0 + p / P)
                    if cost <= F + 1e-9:
                        F -= cost
                        stock[i] -= p
                        free = t + tau_of(inst, i)
                        trips.append((t, 0, i, p))
                        if free <= T:
                            arr[free].append((i, p))
                    break
        q = min(ore, C / 2.0)
        if q > 0.0:
            ore -= q
            F += beta * C * q / (q + C)
            runs.append((t, q))
    out = []
    out.append("0")                       # nU: never upgrade
    out.append(str(len(trips)))
    for (t, j, i, p) in trips:
        out.append("%d %d %d %.6f" % (t, j, i, p))
    out.append(str(len(runs)))
    for (t, q) in runs:
        out.append("%d %.6f" % (t, q))
    sys.stdout.write("\n".join(out) + "\n")


main()
