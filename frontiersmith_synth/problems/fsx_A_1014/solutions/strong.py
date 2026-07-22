# TIER: strong
# The insight: (1) find, via max-flow/min-cut over the valve-capable pipes,
# the minimal source-to-outlet CUTSET that can influence each under-target
# outlet -- pipes that lie on no such cutset are structurally irrelevant and
# never get touched; (2) greedily set-cover the active outlets with as few
# distinct cutset pipes as possible (a shared upstream pipe that gates several
# outlets at once is preferred over one valve per outlet); (3) solve the TRUE
# nonlinear (quadratic) network exactly at every step with Newton's method,
# and drive the chosen valves with Gauss-Newton least squares whose Jacobian
# is RE-LINEARIZED around the current operating point every iteration (finite
# differences through a fresh nonlinear solve, not a one-shot linear guess);
# (4) backward-prune any chosen valve whose removal (after re-optimizing the
# rest) does not hurt the objective, since an ineffective valve only costs
# lambda_cost for nothing.
import sys, math


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


def outlet_flows(n_hub, Hsrc, edges, cap, r_base, xvec, outlets):
    R = [r_base[i] + (xvec[i] if cap[i] else 0.0) for i in range(len(edges))]
    H = solve_heads(n_hub, Hsrc, edges, R)
    qs = edge_flows(H, edges, R)
    out = {}
    for idx, (u, v, r) in enumerate(edges):
        if cap[idx] == 0:
            out[u] = qs[idx]
    return out


def objective(n_hub, Hsrc, edges, cap, r_base, xvec, outlets, targets, lam):
    qs = outlet_flows(n_hub, Hsrc, edges, cap, r_base, xvec, outlets)
    mism = math.sqrt(sum((qs[j] - targets[j]) ** 2 for j in outlets) / len(outlets))
    m = sum(1 for i in range(len(edges)) if cap[i] and xvec[i] > 1e-6)
    return mism + lam * m


def min_cut_edges(n_hub, edges, cap, src, dst):
    """Edmonds-Karp min-cut treating cap=1 pipes as capacity-1 cuttable arcs
    (both directions) and cap=0 pipes as effectively uncuttable (huge cap)."""
    BIG = 10 ** 6
    nnodes = n_hub + 2
    C = [[0] * nnodes for _ in range(nnodes)]
    for idx, (u, v, r) in enumerate(edges):
        c = 1 if cap[idx] else BIG
        C[u][v] += c
        C[v][u] += c
    F = [[0] * nnodes for _ in range(nnodes)]

    def bfs():
        parent = [-1] * nnodes
        parent[src] = src
        q = [src]
        qi = 0
        while qi < len(q):
            u = q[qi]; qi += 1
            for v in range(nnodes):
                if C[u][v] - F[u][v] > 0 and parent[v] == -1:
                    parent[v] = u
                    if v == dst:
                        return parent
                    q.append(v)
        return None

    while True:
        parent = bfs()
        if parent is None:
            break
        v = dst
        pf = float("inf")
        path = []
        while v != src:
            u = parent[v]
            pf = min(pf, C[u][v] - F[u][v])
            path.append((u, v))
            v = u
        for (u, v) in path:
            F[u][v] += pf
            F[v][u] -= pf

    seen = [False] * nnodes
    seen[src] = True
    stack = [src]
    while stack:
        u = stack.pop()
        for v in range(nnodes):
            if not seen[v] and C[u][v] - F[u][v] > 0:
                seen[v] = True
                stack.append(v)
    cutedges = set()
    for idx, (u, v, r) in enumerate(edges):
        if cap[idx] and seen[u] != seen[v]:
            cutedges.add(idx)
    return cutedges


def gauss_newton(n_hub, Hsrc, edges, cap, r_base, outlets, targets, lam, chosen, active):
    chosen_list = sorted(chosen)
    xmap = {e: 0.0 for e in chosen_list}
    n_edges = len(edges)
    if not chosen_list or not active:
        xf = [0.0] * n_edges
        return xf, objective(n_hub, Hsrc, edges, cap, r_base, xf, outlets, targets, lam)
    for _ in range(25):
        xf = [xmap.get(i, 0.0) for i in range(n_edges)]
        qs = outlet_flows(n_hub, Hsrc, edges, cap, r_base, xf, outlets)
        resid = [targets[j] - qs[j] for j in active]
        if max(abs(v) for v in resid) < 1e-4:
            break
        # Jacobian via finite differences: re-linearize around the CURRENT xf
        # (the operating point) each iteration.
        Jac = [[0.0] * len(chosen_list) for _ in active]
        for ci, e in enumerate(chosen_list):
            delta = max(1.0, xmap[e] * 0.05 + 0.5)
            xp = dict(xmap); xp[e] += delta
            xpf = [xp.get(i, 0.0) for i in range(n_edges)]
            qsp = outlet_flows(n_hub, Hsrc, edges, cap, r_base, xpf, outlets)
            for ai, j in enumerate(active):
                Jac[ai][ci] = (qsp[j] - qs[j]) / delta
        dx = _lstsq(Jac, resid)
        maxstep = max((abs(v) for v in dx), default=0.0)
        damp = 1.0 if maxstep < 60.0 else 60.0 / maxstep
        for ci, e in enumerate(chosen_list):
            xmap[e] = min(1000.0, max(0.0, xmap[e] + damp * dx[ci]))
    xf = [xmap.get(i, 0.0) for i in range(n_edges)]
    return xf, objective(n_hub, Hsrc, edges, cap, r_base, xf, outlets, targets, lam)


def _lstsq(Jac, resid):
    """Solve normal equations (J^T J) dx = J^T r for a small, possibly
    rectangular system, with light Tikhonov regularization for stability."""
    m = len(Jac); n = len(Jac[0]) if m else 0
    if n == 0:
        return []
    JT = [[Jac[r][c] for r in range(m)] for c in range(n)]
    A = [[sum(JT[i][k] * Jac[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    for i in range(n):
        A[i][i] += 1e-6
    b = [sum(JT[i][k] * resid[k] for k in range(m)) for i in range(n)]
    return _solve_linear(A, b)


def main():
    tok = sys.stdin.read().split()
    p = 0
    n_hub = int(tok[p]); p += 1
    K = int(tok[p]); p += 1
    n_edges = int(tok[p]); p += 1
    Hsrc = float(tok[p]); p += 1
    lam = float(tok[p]); p += 1
    outlets = [int(tok[p + i]) for i in range(K)]; p += K
    targets = {}
    for i in range(K):
        targets[outlets[i]] = float(tok[p + i])
    p += K
    edges = []
    cap = []
    for _ in range(n_edges):
        u = int(tok[p]); v = int(tok[p + 1]); r = float(tok[p + 2]); c = int(tok[p + 3])
        p += 4
        edges.append((u, v, r))
        cap.append(c)
    r_base = [r for (u, v, r) in edges]

    natural = outlet_flows(n_hub, Hsrc, edges, cap, r_base, [0.0] * n_edges, outlets)
    active = [j for j in outlets if abs(natural[j] - targets[j]) > 0.02]

    if not active:
        print(" ".join(["0.0"] * n_edges))
        return

    cand = {j: min_cut_edges(n_hub, edges, cap, 0, j) for j in active}
    chosen = set()
    remaining = set(active)
    while remaining:
        counts = {}
        for j in remaining:
            for e in cand[j]:
                counts[e] = counts.get(e, 0) + 1
        if not counts:
            break
        best_e = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        chosen.add(best_e)
        remaining = {j for j in remaining if best_e not in cand[j]}

    xf, F_before = gauss_newton(n_hub, Hsrc, edges, cap, r_base, outlets, targets, lam, chosen, active)

    changed = True
    cur = set(chosen)
    while changed:
        changed = False
        for e in sorted(cur):
            trial = cur - {e}
            xf2, F2 = gauss_newton(n_hub, Hsrc, edges, cap, r_base, outlets, targets, lam, trial, active)
            if F2 <= F_before + 1e-9:
                cur, xf, F_before = trial, xf2, F2
                changed = True
                break

    print(" ".join(f"{v:.6f}" for v in xf))


if __name__ == "__main__":
    main()
