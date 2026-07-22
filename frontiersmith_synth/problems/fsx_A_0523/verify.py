#!/usr/bin/env python3
# verify.py <in> <out> <ans>   -- deterministic scorer for the boiler-dispatch problem.
# Reads the instance from <in>, the participant schedule from <out> (T lines x K outputs).
# On ANY feasibility violation prints `Ratio: 0.0` and exits 0.
# Otherwise: F = total fuel burned; B = fuel of the checker's own all-online proportional
# reference; minimization score = min(1000, 100*B/F)/1000.
import sys, math


def read_instance(path):
    toks = open(path).read().split()
    i = 0
    T = int(toks[i]); i += 1
    K = int(toks[i]); i += 1
    D = [float(toks[i + j]) for j in range(T)]; i += T
    units = []
    for _ in range(K):
        C = float(toks[i]); pmin = float(toks[i + 1]); c = float(toks[i + 2])
        a = float(toks[i + 3]); b = float(toks[i + 4]); x = float(toks[i + 5])
        ramp = float(toks[i + 6]); mu = int(float(toks[i + 7])); md = int(float(toks[i + 8]))
        i += 9
        units.append((C, pmin, c, a, b, x, ramp, mu, md))
    return T, K, D, units


def fuel(o, u):
    if o <= 1e-12:
        return 0.0
    C, pmin, c, a, b, x = u[0], u[1], u[2], u[3], u[4], u[5]
    xr = o / C
    return c + a * o * (1.0 + b * (xr - x) ** 2)


def baseline(T, K, D, units):
    total = sum(u[0] for u in units)
    B = 0.0
    for t in range(T):
        for u in units:
            share = D[t] * u[0] / total
            o = share if share > u[1] else u[1]
            B += fuel(o, u)
    return B


def fail(msg):
    print("INFEASIBLE: " + msg)
    print("Ratio: 0.000000")
    sys.exit(0)


def main():
    T, K, D, units = read_instance(sys.argv[1])
    toks = open(sys.argv[2]).read().split()
    if len(toks) != T * K:
        fail("expected %d numbers, got %d" % (T * K, len(toks)))
    O = [[0.0] * K for _ in range(T)]
    idx = 0
    for t in range(T):
        for k in range(K):
            try:
                v = float(toks[idx])
            except ValueError:
                fail("non-numeric output")
            if not math.isfinite(v):
                fail("non-finite output")
            O[t][k] = v
            idx += 1

    # bounds + on/off legality
    for t in range(T):
        for k in range(K):
            u = units[k]; o = O[t][k]
            if o < -1e-6:
                fail("negative output t=%d k=%d" % (t, k))
            if o > 1e-9:
                if o < u[1] - 1e-4:
                    fail("output below pmin t=%d k=%d" % (t, k))
                if o > u[0] + 1e-4:
                    fail("output above capacity t=%d k=%d" % (t, k))

    # demand must be met (over-production allowed, it just wastes fuel)
    for t in range(T):
        if sum(O[t]) < D[t] - 1e-4:
            fail("unmet demand t=%d" % t)

    # ramp between consecutive online steps
    for k in range(K):
        ramp = units[k][6]
        for t in range(1, T):
            if O[t][k] > 1e-9 and O[t - 1][k] > 1e-9:
                if abs(O[t][k] - O[t - 1][k]) > ramp + 1e-4:
                    fail("ramp violation t=%d k=%d" % (t, k))

    # minimum up / down times (boilers start OFF before t=0)
    for k in range(K):
        mu, md = units[k][7], units[k][8]
        on = [O[t][k] > 1e-9 for t in range(T)]
        t = 0
        while t < T:
            v = on[t]; s = t
            while t < T and on[t] == v:
                t += 1
            L = t - s
            touches_end = (t == T)
            touches_start = (s == 0)
            if v:
                if L < mu and not touches_end:
                    fail("min-up violation k=%d" % k)
            else:
                if (not touches_start) and (not touches_end) and L < md:
                    fail("min-down violation k=%d" % k)

    F = 0.0
    for t in range(T):
        for k in range(K):
            F += fuel(O[t][k], units[k])
    if F <= 1e-9:
        fail("degenerate zero-fuel schedule")
    B = baseline(T, K, D, units)
    sc = min(1000.0, 100.0 * B / F)
    print("fuel=%.4f reference=%.4f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
