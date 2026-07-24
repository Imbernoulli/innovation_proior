# TIER: strong
# The insight: with T^4 radiation the steady state is a global fixed point in
# which a junction's temperature is set by how little heat can LEAVE (thin
# conductive necks, weak radiators), not by how much is generated.  So we
# solve the actual nonlinear steady state and sink the EMERGENT hottest
# junction, re-solving after each sink; hotspots migrate as sinks land, which
# automatically balances budget between the loud core and the trapped pockets.
# A short swap-polish pass then repairs any early placement that later
# re-solves reveal to be wasted.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys

import numpy as np


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
            Tn = T + t * d
        T = Tn
    for j, i in enumerate(free):
        T_full[i] = T[j]
    return T_full


def hottest(T, sink_mask, count=1):
    order = sorted((i for i in range(len(T)) if not sink_mask[i]),
                   key=lambda i: (-T[i], i))
    return order[:count]


def main():
    it = sys.stdin.read().split()
    p = 0
    n = int(it[p]); m = int(it[p + 1]); k = int(it[p + 2]); p += 3
    g = [float(it[p + i]) for i in range(n)]; p += n
    a = [float(it[p + i]) for i in range(n)]; p += n
    eu, ev, ec = [], [], []
    for _ in range(m):
        eu.append(int(it[p])); ev.append(int(it[p + 1])); ec.append(float(it[p + 2])); p += 3

    sink_mask = [False] * n
    sinks = []
    T = steady_state(n, g, a, eu, ev, ec, sink_mask)
    for _ in range(k):
        h = hottest(T, sink_mask)[0]
        sink_mask[h] = True
        sinks.append(h)
        T = steady_state(n, g, a, eu, ev, ec, sink_mask, T0=T)
    best = float(np.max(T))

    # best-swap local search: move a sink onto one of the hottest free
    # junctions whenever doing so lowers the peak; repeat to a fixed point.
    for _round in range(6):
        improved = False
        for s in list(sinks):
            for h in hottest(T, sink_mask, count=4):
                sink_mask[s] = False
                sink_mask[h] = True
                T2 = steady_state(n, g, a, eu, ev, ec, sink_mask, T0=T)
                b2 = float(np.max(T2))
                if b2 < best - 1e-9:
                    sinks[sinks.index(s)] = h
                    T = T2
                    best = b2
                    improved = True
                    break
                sink_mask[s] = True
                sink_mask[h] = False
        if not improved:
            break

    out = [str(len(sinks))] + [str(i) for i in sorted(sinks)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
