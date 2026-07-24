# TIER: strong
"""Insight: the depot is autocatalytic -- fuel makes fuel. Early decisions are
compounding-rate decisions, so the early objective is GROWTH of the refinery
(marginal capacity value vs its fuel price), not per-trip efficiency; late
decisions are HARVEST decisions (haul only ore that is worth more fuel than it
burns, spread the run level along the concave conversion curve). The phase
boundary is not assumed: it is located by simulating a grid of candidate
switch days and keeping the best final tank.

Deterministic: pure function of the input, no randomness, no wall time."""
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


def simulate(inst, t_star, theta=3.0):
    T = inst['T']; K = inst['K']; M = inst['M']
    F = inst['F0']; C = inst['C0']; ore = 0.0
    stock = inst['S'][:]
    free = [1] * K
    arr = [[] for _ in range(T + 2)]
    P = inst['P']; kap = inst['kap']; beta = inst['beta']
    cu = inst['cu']; Cmax = inst['Cmax']
    d = inst['d']
    tau = [max(1, int(math.ceil(2.0 * d[i] / inst['v'] - 1e-12)))
           for i in range(M)]
    dmin = min(d)
    reserve = kap * dmin * 2.5          # keep one cheap bootstrap loop fuelable
    in_transit = 0.0
    ups = []; trips = []; runs = []
    for t in range(1, T + 1):
        for (i, p) in arr[t]:
            ore += p
            in_transit -= p
        days_left = T - t + 1
        qplan = min(C, (ore + in_transit) / days_left)
        yhat = beta * C / (qplan + C) if C > 0.0 else 0.0
        remcap = C * days_left - (ore + in_transit)
        growth = (t <= t_star)
        pipeline_full = growth and (ore + in_transit) >= theta * qplan + P
        # ---- dispatch idle trucks ----
        if not pipeline_full:
            for j in range(K):
                if free[j] > t:
                    continue
                best = None
                for i in range(M):
                    if stock[i] <= 1e-9:
                        continue
                    if t + tau[i] > T:
                        continue
                    p = min(P, stock[i], remcap)
                    if p < 1e-3:
                        continue
                    cost = kap * d[i] * (2.0 + p / P)
                    gain = p * yhat - cost
                    if gain <= 0.0:
                        continue
                    rate = gain / tau[i]
                    if best is None or rate > best[0] + 1e-18:
                        best = (rate, i, p, cost)
                if best is None:
                    continue
                _, i, p, cost = best
                if cost > F - 1e-9:
                    continue
                F -= cost
                stock[i] -= p
                remcap -= p
                in_transit += p
                rday = t + tau[i]
                free[j] = rday
                trips.append((t, j, i, p))
                arr[rday].append((i, p))
                if growth and (ore + in_transit) >= theta * qplan + P:
                    break                       # pipeline target met for today
        # ---- reinvest: grow capacity while its marginal yield beats price ----
        if growth:
            while C < Cmax - 1e-9:
                qbar = min(C, max(qplan, 1e-9))
                if C <= qbar + 1e-9:
                    marg = beta / 2.0           # capacity-bound: d(beta*C/2)/dC
                else:
                    marg = beta * (qbar / (qbar + C)) ** 2
                if marg * (T - t) <= cu * 1.01:
                    break
                step = min(1.0, Cmax - C)
                cost = cu * step
                if F - cost < reserve:
                    break
                F -= cost
                C += step
                remcap += step * days_left
                ups.append((t, step))
        # ---- refine along the concave curve (spread, don't floor it) ----
        q = min(ore, qplan)
        if q > 0.0:
            ore -= q
            F += beta * C * q / (q + C)
            runs.append((t, q))
    return F, ups, trips, runs


def main():
    inst = read_instance()
    T = inst['T']
    best_F = -1.0
    best = None
    fracs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for fr in fracs:
        t_star = int(round(fr * T))
        F, ups, trips, runs = simulate(inst, t_star)
        if F > best_F + 1e-12:
            best_F = F
            best = (ups, trips, runs)
    ups, trips, runs = best
    out = []
    out.append(str(len(ups)))
    for (t, u) in ups:
        out.append("%d %.6f" % (t, u))
    out.append(str(len(trips)))
    for (t, j, i, p) in trips:
        out.append("%d %d %d %.6f" % (t, j, i, p))
    out.append(str(len(runs)))
    for (t, q) in runs:
        out.append("%d %.6f" % (t, q))
    sys.stdout.write("\n".join(out) + "\n")


main()
