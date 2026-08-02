#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the self-assembly
scheduling problem. Prints "... Ratio: <float in [0,1]>" and exits 0.

Reference baseline B: the checker's own trivial construction -- enable EVERY
bond option simultaneously at one single, fixed, late time step
(round(0.94 * T), clamped to [1, T]).  This is exactly what solutions/trivial.py
reproduces (giving it Ratio == 0.1 up to floating point).
"""
import sys
import math

BASELINE_FRAC = 0.94


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    M = int(toks[idx]); idx += 1
    Tmax = int(toks[idx]); idx += 1
    theta0 = int(toks[idx]); idx += 1
    bonds = []
    for _ in range(M):
        u = int(toks[idx]); idx += 1
        v = int(toks[idx]); idx += 1
        s = int(toks[idx]); idx += 1
        typ = toks[idx]; idx += 1
        bonds.append((u, v, s, typ))
    return N, M, Tmax, theta0, bonds


def theta(t, theta0, Tmax):
    if Tmax <= 1:
        return 0
    return theta0 - (theta0 * (t - 1)) // (Tmax - 1)


def simulate(N, M, Tmax, theta0, bonds, enable_times):
    """Deterministic annealing-with-irreversible-freeze simulation.  Returns the
    time-averaged fraction of target (`type == 'T'`) bond strength that is
    formed and currently stable, in [0, 1]."""
    occ = [-1] * N
    formed = [False] * M
    target_total = sum(s for (_, _, s, typ) in bonds if typ == 'T')
    if target_total <= 0:
        target_total = 1
    total = 0.0
    for t in range(1, Tmax + 1):
        th = theta(t, theta0, Tmax)
        # (1) formation attempts, fixed input-order arbitration
        for j in range(M):
            if formed[j]:
                continue
            if enable_times[j] > t:
                continue
            u, v, s, typ = bonds[j]
            if occ[u] == -1 and occ[v] == -1:
                occ[u] = j
                occ[v] = j
                formed[j] = True
        # (2) reversibility check -- bonds weaker than the current
        #     temperature break; bonds at/above it are frozen forever
        for j in range(M):
            if not formed[j]:
                continue
            u, v, s, typ = bonds[j]
            if s < th:
                formed[j] = False
                occ[u] = -1
                occ[v] = -1
        cur = 0
        for j in range(M):
            if formed[j] and bonds[j][3] == 'T':
                cur += bonds[j][2]
        total += cur / target_total
    return total / Tmax


def baseline_schedule(M, Tmax):
    t = max(1, min(Tmax, int(round(Tmax * BASELINE_FRAC))))
    return [t] * M


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad invocation)")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, M, Tmax, theta0, bonds = read_instance(in_path)

    try:
        with open(out_path) as f:
            toks = f.read().split()
    except OSError:
        print("Ratio: 0.0 (no output)")
        return 0

    if len(toks) < M:
        print("Ratio: 0.0 (expected %d tokens, got %d)" % (M, len(toks)))
        return 0

    enable = []
    for i in range(M):
        tok = toks[i]
        try:
            v = int(tok)
        except ValueError:
            print("Ratio: 0.0 (non-integer token %r)" % tok)
            return 0
        if not math.isfinite(v):
            print("Ratio: 0.0 (non-finite)")
            return 0
        if not (1 <= v <= Tmax):
            print("Ratio: 0.0 (enable time out of [1,%d])" % Tmax)
            return 0
        enable.append(v)

    F = simulate(N, M, Tmax, theta0, bonds, enable)
    B = simulate(N, M, Tmax, theta0, bonds, baseline_schedule(M, Tmax))
    if B <= 0:
        B = 1e-9
    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
