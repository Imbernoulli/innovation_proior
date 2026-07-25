#!/usr/bin/env python3
"""verify.py -- deterministic checker for 'Ashline Refinery Bootstrap' (fsx_A_1102).

Usage: python3 verify.py <in> <out> <ans>     (ans ignored)

Simulates the fuel/ore/capacity ledgers day by day. Any feasibility violation
prints Ratio: 0.0. Otherwise prints Ratio: min(1, 0.1 * F / B) where F is the
final fuel stock and B is the checker's own do-little baseline construction.
"""
import sys
import math


def fail(msg):
    sys.stdout.write("infeasible: %s\n" % msg)
    sys.stdout.write("Ratio: 0.0\n")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        tok = f.read().split()
    it = iter(tok)

    def nxt():
        return next(it)

    T = int(nxt()); K = int(nxt()); M = int(nxt())
    F0 = float(nxt()); C0 = float(nxt()); Cmax = float(nxt())
    cu = float(nxt()); beta = float(nxt()); v = float(nxt())
    P = float(nxt()); kap = float(nxt())
    d = []; S = []
    for _ in range(M):
        d.append(float(nxt())); S.append(float(nxt()))
    return dict(T=T, K=K, M=M, F0=F0, C0=C0, Cmax=Cmax, cu=cu, beta=beta,
                v=v, P=P, kap=kap, d=d, S=S)


def tau_of(inst, i):
    # round-trip duration in whole days (truck+ore return at start of day t+tau)
    return max(1, int(math.ceil(2.0 * inst['d'][i] / inst['v'] - 1e-12)))


def baseline(inst):
    """Do-little reference: truck 0 shuttles the nearest stocked mine at half
    payload, refinery runs at half throttle, capacity is never upgraded."""
    T = inst['T']; C = inst['C0']; F = inst['F0']; ore = 0.0
    stock = inst['S'][:]
    free = 1
    arr = [[] for _ in range(T + 2)]
    P = inst['P']; kap = inst['kap']; beta = inst['beta']
    order = sorted(range(inst['M']), key=lambda i: (inst['d'][i], i))
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
                        if free <= T:
                            arr[free].append((i, p))
                    break
        q = min(ore, C / 2.0)
        if q > 0.0:
            ore -= q
            F += beta * C * q / (q + C)
    return F


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inst = read_instance(sys.argv[1])
    T = inst['T']; K = inst['K']; M = inst['M']

    # ---- parse participant output: strict schema, finite floats only ----
    try:
        with open(sys.argv[2]) as f:
            tok = f.read().split()
    except OSError:
        fail("cannot read output")
    if not tok:
        fail("empty output")
    vals = []
    for x in tok:
        try:
            y = float(x)
        except ValueError:
            fail("non-numeric token %r" % x)
        if not math.isfinite(y):
            fail("non-finite token")
        vals.append(y)

    pos = 0

    def rd():
        nonlocal pos
        if pos >= len(vals):
            fail("truncated output")
        y = vals[pos]; pos += 1
        return y

    def rd_int(lo, hi):
        y = rd()
        if y != math.floor(y):
            fail("non-integer field")
        y = int(y)
        if y < lo or y > hi:
            fail("index out of range")
        return y

    MAXREC = 1000000
    nU = rd_int(0, MAXREC)
    ups = []
    for _ in range(nU):
        t = rd_int(1, T); u = rd()
        if u < 0.0:
            fail("negative upgrade")
        ups.append((t, u))
    nR = rd_int(0, MAXREC)
    trips = []
    for _ in range(nR):
        t = rd_int(1, T); j = rd_int(0, K - 1); i = rd_int(0, M - 1); p = rd()
        if p <= 0.0:
            fail("non-positive payload")
        trips.append((t, j, i, p))
    nQ = rd_int(0, MAXREC)
    runs = []
    for _ in range(nQ):
        t = rd_int(1, T); q = rd()
        if q < 0.0:
            fail("negative run level")
        runs.append((t, q))
    if pos != len(vals):
        fail("trailing tokens")

    # ---- group directives by day, preserving listed order within a day ----
    U = [[] for _ in range(T + 2)]
    for (t, u) in ups:
        U[t].append(u)
    R = [[] for _ in range(T + 2)]
    for (t, j, i, p) in trips:
        R[t].append((j, i, p))
    Q = [[] for _ in range(T + 2)]
    for (t, q) in runs:
        Q[t].append(q)

    # ---- simulate (order: arrivals, upgrades, departures, refining) ----
    TOL = 1e-6
    F = inst['F0']; C = inst['C0']; ore = 0.0
    stock = inst['S'][:]
    free = [1] * K
    arr = [[] for _ in range(T + 2)]
    P = inst['P']; kap = inst['kap']; beta = inst['beta']
    cu = inst['cu']; Cmax = inst['Cmax']
    for t in range(1, T + 1):
        for (i, p) in arr[t]:
            ore += p
        for u in U[t]:
            F -= cu * u
            if F < -TOL * (1.0 + abs(cu * u)):
                fail("fuel overdrawn by upgrade on day %d" % t)
            C += u
            if C > Cmax + 1e-9:
                fail("capacity exceeds Cmax on day %d" % t)
        for (j, i, p) in R[t]:
            if free[j] > t:
                fail("truck %d not idle on day %d" % (j, t))
            if p > P + 1e-9:
                fail("payload exceeds truck capacity on day %d" % t)
            if p > stock[i] + TOL:
                fail("payload exceeds ore stock at mine %d on day %d" % (i, t))
            cost = kap * inst['d'][i] * (2.0 + p / P)
            F -= cost
            if F < -TOL * (1.0 + abs(cost)):
                fail("fuel overdrawn by trip on day %d" % t)
            stock[i] = max(0.0, stock[i] - p)
            rday = t + tau_of(inst, i)
            free[j] = rday
            if rday <= T:
                arr[rday].append((i, p))
        q = sum(Q[t])
        if q > 0.0:
            cap = min(ore, C)
            if q > cap + TOL:
                fail("run level exceeds ore/capacity on day %d" % t)
            q = min(q, cap)
            ore -= q
            if C > 0.0:
                F += beta * C * q / (q + C)

    F = max(0.0, F)
    B = baseline(inst)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    sys.stdout.write("final_fuel %.6f baseline %.6f\n" % (F, B))
    sys.stdout.write("Ratio: %.6f\n" % (sc / 1000.0))


if __name__ == "__main__":
    main()
