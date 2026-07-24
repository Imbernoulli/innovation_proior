#!/usr/bin/env python3
# Deterministic checker for radiative-hotspot-sinks (format C, MINIMIZE the
# steady-state hotspot temperature).  CLI: python3 verify.py <in> <out> <ans>
# (ans ignored).  Prints "... Ratio: <r>" with r in [0,1]; any feasibility
# breach -> Ratio: 0.0.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import math

import numpy as np


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


# ---- nonlinear steady-state solve ------------------------------------------
# g_i = a_i T_i^4 + sum_j c_ij (T_i - T_j)  on free nodes; sinks clamped to 0.
# Strictly convex energy -> unique non-negative fixed point; damped Newton.
def steady_state(n, g, a, eu, ev, ec, sink_mask, T0=None):
    free = [i for i in range(n) if not sink_mask[i]]
    T_full = np.zeros(n)
    if not free:
        return T_full
    pos = [-1] * n
    for j, i in enumerate(free):
        pos[i] = j
    nf = len(free)
    A = np.zeros((nf, nf))
    for u, v, c in zip(eu, ev, ec):
        pu, pv = pos[u], pos[v]
        if pu >= 0:
            A[pu, pu] += c
        if pv >= 0:
            A[pv, pv] += c
        if pu >= 0 and pv >= 0:
            A[pu, pv] -= c
            A[pv, pu] -= c
    gf = np.array([g[i] for i in free])
    af = np.array([a[i] for i in free])
    if T0 is None:
        T = np.maximum((gf / af) ** 0.25, 1e-3)
    else:
        T = np.maximum(np.array([T0[i] for i in free]), 1e-6)
    tol = 1e-10 * (1.0 + float(np.max(gf)))
    for _ in range(80):
        F = gf - af * T ** 4 - A.dot(T)
        r = float(np.max(np.abs(F)))
        if r < tol:
            break
        J = A + np.diag(4.0 * af * T ** 3)
        d = np.linalg.solve(J, F)
        t = 1.0
        Tn = None
        for _ls in range(40):
            cand = T + t * d
            if float(np.min(cand)) <= 0.0:
                t *= 0.5
                continue
            Fn = gf - af * cand ** 4 - A.dot(cand)
            if float(np.max(np.abs(Fn))) < r:
                Tn = cand
                break
            t *= 0.5
        if Tn is None:
            Tn = T + t * d           # accept the fully damped step
        T = Tn
    for j, i in enumerate(free):
        T_full[i] = T[j]
    return T_full


def read_instance(path):
    it = open(path).read().split()
    p = 0
    n = int(it[p]); m = int(it[p + 1]); k = int(it[p + 2]); p += 3
    g = [float(it[p + i]) for i in range(n)]; p += n
    a = [float(it[p + i]) for i in range(n)]; p += n
    eu, ev, ec = [], [], []
    for _ in range(m):
        u = int(it[p]); v = int(it[p + 1]); c = float(it[p + 2]); p += 3
        eu.append(u); ev.append(v); ec.append(c)
    return n, m, k, g, a, eu, ev, ec


def parse_int(tok):
    # strict integer token: no floats, no nan/inf, no whitespace tricks
    if not tok or any(ch not in "+-0123456789" for ch in tok):
        return None
    try:
        v = int(tok)
    except Exception:
        return None
    return v


def main():
    try:
        n, m, k, g, a, eu, ev, ec = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")

    # ---- participant output -------------------------------------------------
    try:
        ot = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not ot:
        fail("empty output")
    M = parse_int(ot[0])
    if M is None:
        fail("bad M")
    if M < 0 or M > k:
        fail("M out of range (0..k)")
    if len(ot) != 1 + M:
        fail("token count != 1 + M")

    sink_mask = [False] * n
    for j in range(M):
        v = parse_int(ot[1 + j])
        if v is None:
            fail("non-integer sink index %d" % j)
        if v < 0 or v >= n:
            fail("sink index %d out of range" % j)
        if sink_mask[v]:
            fail("duplicate sink index")
        sink_mask[v] = True

    # ---- objective ----------------------------------------------------------
    T = steady_state(n, g, a, eu, ev, ec, sink_mask)
    F_obj = float(np.max(T))
    if not math.isfinite(F_obj):
        fail("non-finite temperature")

    # ---- internal baseline: sinks at junctions 0..k-1 -----------------------
    bmask = [False] * n
    for i in range(min(k, n)):
        bmask[i] = True
    Tb = steady_state(n, g, a, eu, ev, ec, bmask)
    B = float(np.max(Tb))
    if not math.isfinite(B) or B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-9, F_obj))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F_obj, B, sc / 1000.0))


if __name__ == "__main__":
    main()
