#!/usr/bin/env python3
# Deterministic checker for "Valve Balancing on a Looped Water Trunk" (format C).
# CLI: python3 verify.py <in> <out> <ans>  (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
import sys, math

X_MAX = 1000.0
TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def solve_heads(n_hub, Hsrc, edges, R):
    H = [0.0] * (n_hub + 2)
    H[0] = Hsrc
    H[n_hub + 1] = 0.0
    for k in range(1, n_hub + 1):
        H[k] = Hsrc * (1.0 - k / (n_hub + 1))
    eps = 1e-9
    for _ in range(200):
        res = [0.0] * n_hub
        J = [[0.0] * n_hub for _ in range(n_hub)]
        for idx, (u, v, r) in enumerate(edges):
            Ru = R[idx]
            d = H[u] - H[v]
            ad = abs(d) if abs(d) > eps else eps
            q = math.copysign(math.sqrt(ad / Ru), d) if d != 0 else 0.0
            g = 1.0 / (2.0 * math.sqrt(ad * Ru))
            if 1 <= u <= n_hub:
                res[u - 1] += q
                J[u - 1][u - 1] += g
                if 1 <= v <= n_hub:
                    J[u - 1][v - 1] -= g
            if 1 <= v <= n_hub:
                res[v - 1] -= q
                J[v - 1][v - 1] += g
                if 1 <= u <= n_hub:
                    J[v - 1][u - 1] -= g
        for i in range(n_hub):
            J[i][i] += 1e-9
        dH = _solve_linear(J, [-x for x in res])
        maxstep = max((abs(x) for x in dH), default=0.0)
        damp = 1.0 if maxstep < 40.0 else 40.0 / maxstep
        for k in range(n_hub):
            H[k + 1] += damp * dH[k]
        if max((abs(x) for x in res), default=0.0) < 1e-10:
            break
    return H


def _solve_linear(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-14:
            continue
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / pv
            if f == 0.0:
                continue
            for cc in range(c, n + 1):
                M[r][cc] -= f * M[c][cc]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][cc] * x[cc] for cc in range(r + 1, n))
        x[r] = s / M[r][r] if abs(M[r][r]) > 1e-14 else 0.0
    return x


def edge_flows(H, edges, R):
    qs = []
    for idx, (u, v, r) in enumerate(edges):
        d = H[u] - H[v]
        ad = abs(d) if abs(d) > 1e-9 else 1e-9
        q = math.copysign(math.sqrt(ad / R[idx]), d) if d != 0 else 0.0
        qs.append(q)
    return qs


def total_F(n_hub, Hsrc, edges, cap, r_base, x, outlets, targets, lam):
    R = [r_base[i] + (x[i] if cap[i] else 0.0) for i in range(len(edges))]
    H = solve_heads(n_hub, Hsrc, edges, R)
    qs = edge_flows(H, edges, R)
    outq = {}
    for idx, (u, v, r) in enumerate(edges):
        if cap[idx] == 0:
            outq[u] = qs[idx]
    mism = math.sqrt(sum((outq[j] - targets[j]) ** 2 for j in outlets) / len(outlets))
    m = sum(1 for i in range(len(edges)) if cap[i] and x[i] > 1e-6)
    return mism + lam * m


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    n_hub = int(itoks[p]); p += 1
    K = int(itoks[p]); p += 1
    n_edges = int(itoks[p]); p += 1
    Hsrc = float(itoks[p]); p += 1
    lam = float(itoks[p]); p += 1
    outlets = [int(itoks[p + i]) for i in range(K)]; p += K
    targets = {}
    for i in range(K):
        targets[outlets[i]] = float(itoks[p + i])
    p += K
    edges = []
    cap = []
    for _ in range(n_edges):
        u = int(itoks[p]); v = int(itoks[p + 1]); r = float(itoks[p + 2]); c = int(itoks[p + 3])
        p += 4
        edges.append((u, v, r))
        cap.append(c)
    r_base = [r for (u, v, r) in edges]

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != n_edges:
        fail("expected exactly %d numbers, got %d" % (n_edges, len(otoks)))

    x = []
    for i, tok in enumerate(otoks):
        try:
            v = float(tok)
        except Exception:
            fail("unparsable value at position %d" % i)
        if not math.isfinite(v):
            fail("non-finite value at position %d" % i)
        if v < -TOL:
            fail("negative extra resistance at pipe %d" % i)
        v = max(0.0, v)
        if v > X_MAX + 1e-6:
            fail("extra resistance at pipe %d exceeds %.1f" % (i, X_MAX))
        if cap[i] == 0 and v > TOL:
            fail("valve installed on non-valve-capable pipe %d" % i)
        x.append(v)

    F = total_F(n_hub, Hsrc, edges, cap, r_base, x, outlets, targets, lam)
    x0 = [0.0] * n_edges
    B = total_F(n_hub, Hsrc, edges, cap, r_base, x0, outlets, targets, lam)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
