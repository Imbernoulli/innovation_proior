#!/usr/bin/env python3
# Generator for "Valve Balancing on a Looped Water Trunk" (format C).
# `python3 gen.py <testId>` prints ONE instance to stdout. testId 1..10 is a
# fixed size/difficulty ladder; all randomness is seeded from testId only.
import sys, random, math

X_MAX = 1000.0

LADDER = [
    dict(n_hub=3,  n_cross=0, K=1),
    dict(n_hub=4,  n_cross=0, K=1),
    dict(n_hub=5,  n_cross=0, K=2),
    dict(n_hub=5,  n_cross=1, K=2),
    dict(n_hub=6,  n_cross=1, K=3),
    dict(n_hub=7,  n_cross=1, K=3),
    dict(n_hub=8,  n_cross=2, K=4),
    dict(n_hub=9,  n_cross=2, K=4),
    dict(n_hub=10, n_cross=2, K=5),
    dict(n_hub=12, n_cross=3, K=5),
]
# reduction-only targets: a valve can only ADD resistance, so it can only
# throttle flow down from the natural (all-open) level, never boost it above.
FACTORS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.68]


def solve_heads(n_hub, Hsrc, edges, R):
    """Newton-Raphson on the quadratic (Hazen/Darcy-style) node-head equations.
    edges: list of (u, v, r_effective) with u,v in 0..n_hub+1 (0=source fixed
    at Hsrc, n_hub+1=atmosphere fixed at 0). Returns head vector length n_hub+2."""
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
    """Plain Gaussian elimination with partial pivoting (n small, <=12)."""
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


def build_instance(test_id):
    spec = LADDER[test_id - 1]
    n_hub, n_cross, K = spec["n_hub"], spec["n_cross"], spec["K"]
    rng = random.Random(1000 + test_id)
    r_T = [round(rng.uniform(1.0, 2.0), 3) for _ in range(n_hub)]
    r_P = [round(rng.uniform(4.0, 7.0), 3) for _ in range(n_hub)]

    edges = []
    cap = []
    for i in range(1, n_hub + 1):
        edges.append((i - 1, i, r_T[i - 1])); cap.append(1)
    for i in range(1, n_hub + 1):
        edges.append((i - 1, i, r_P[i - 1])); cap.append(1)

    crossties = []
    tries = 0
    while len(crossties) < n_cross and tries < 300:
        tries += 1
        a = rng.randint(0, n_hub - 2)
        b = rng.randint(a + 2, n_hub)
        if (a, b) in crossties:
            continue
        crossties.append((a, b))
    for (a, b) in crossties:
        r_X = round(rng.uniform(1.3, 2.3), 3)
        edges.append((a, b, r_X)); cap.append(1)

    outlets = set(rng.sample(range(1, n_hub), K - 1)) if K > 1 else set()
    outlets.add(n_hub)          # the far end always drains -> no dead trunk tail
    outlets = sorted(outlets)
    r_O = []
    for j in outlets:
        r_O.append(round(rng.uniform(1.5, 4.0), 3))
        edges.append((j, n_hub + 1, r_O[-1])); cap.append(0)

    Hsrc = 100.0
    lambda_cost = round(rng.uniform(0.02, 0.08), 3)

    r_base = [r for (u, v, r) in edges]
    H0 = solve_heads(n_hub, Hsrc, edges, r_base)
    q0 = edge_flows(H0, edges, r_base)
    natural = {}
    for idx, (u, v, r) in enumerate(edges):
        if cap[idx] == 0:
            natural[u] = q0[idx]

    targets = {}
    for j in outlets:
        f = FACTORS[rng.randrange(len(FACTORS))]
        targets[j] = round(natural[j] * f, 4)

    return n_hub, K, edges, cap, Hsrc, lambda_cost, outlets, targets


def main():
    test_id = int(sys.argv[1])
    n_hub, K, edges, cap, Hsrc, lambda_cost, outlets, targets = build_instance(test_id)
    n_edges = len(edges)
    out = []
    out.append(f"{n_hub} {K} {n_edges}")
    out.append(f"{Hsrc:.6f} {lambda_cost:.6f}")
    out.append(" ".join(str(j) for j in outlets))
    out.append(" ".join(f"{targets[j]:.6f}" for j in outlets))
    for idx, (u, v, r) in enumerate(edges):
        out.append(f"{u} {v} {r:.6f} {cap[idx]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
