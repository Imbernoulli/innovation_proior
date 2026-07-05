#!/usr/bin/env python3
# verify.py <in> <out> <ans>
# Deterministic scorer for the aquarium pipe-packing (max sum of radii) problem.
# Prints "... Ratio: <x>" on the final line; exits 0.
import sys
import math

TOL = 1e-6

def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    N = int(toks[0])
    W = float(toks[1]); H = float(toks[2]); rmax = float(toks[3])
    return N, W, H, rmax

def read_output(path, N):
    # Return list of (x,y,r) or None on any schema/finiteness violation.
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        return None
    if len(toks) != 3 * N:
        return None
    vals = []
    for tk in toks:
        try:
            v = float(tk)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        vals.append(v)
    circ = []
    for i in range(N):
        x = vals[3 * i]; y = vals[3 * i + 1]; r = vals[3 * i + 2]
        circ.append((x, y, r))
    return circ

def feasible(circ, W, H, rmax):
    for (x, y, r) in circ:
        if not (r > 0.0):
            return False
        if r > rmax + TOL:
            return False
        if x - r < -TOL or x + r > W + TOL:
            return False
        if y - r < -TOL or y + r > H + TOL:
            return False
    n = len(circ)
    for i in range(n):
        xi, yi, ri = circ[i]
        for j in range(i + 1, n):
            xj, yj, rj = circ[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < ri + rj - TOL:
                return False
    return True

def baseline_sum(N, W, H, rmax):
    # Trivial single-row packing capped at rmax; always feasible, positive.
    # N equal circles centered along the horizontal midline, spaced W/N apart.
    r = min(0.5 * W / N, 0.5 * H, rmax)
    return N * r

def main():
    inp, out = sys.argv[1], sys.argv[2]
    N, W, H, rmax = read_instance(inp)
    B = baseline_sum(N, W, H, rmax)
    circ = read_output(out, N)
    if circ is None or not feasible(circ, W, H, rmax):
        print("infeasible layout Ratio: 0.0")
        return
    F = sum(r for (_, _, r) in circ)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
