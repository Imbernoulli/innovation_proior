# TIER: greedy
"""The obvious first approach: fit the FIRST excitation pattern only. Scan components in
netlist order; report the first single component (or, failing that, the first pair) whose
resistance can be tuned to reproduce that one pattern's readings almost exactly. This is a
real, useful heuristic -- it just never checks whether the explanation also survives the
OTHER excitation pattern(s), so it can lock onto a component that only looks responsible
because a single measurement is under-determined."""
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    test_id = int(next(it))
    n_terminal = int(next(it)); n_edges = int(next(it))
    edges = []
    for _ in range(n_edges):
        u = int(next(it)); v = int(next(it)); R = float(next(it))
        edges.append([u, v, R])
    n_shown = int(next(it))
    shown = []
    for _ in range(n_shown):
        s = int(next(it)); g = int(next(it)); Q = float(next(it))
        Vs = float(next(it)); Vg = float(next(it))
        shown.append((s, g, Q, Vs, Vg))
    n_nodes = max(max(u, v) for (u, v, _R) in edges) + 1
    n_nodes = max(n_nodes, n_terminal)
    return test_id, n_nodes, edges, shown


def solve_nodal_float(n, edges, R_list, s, g, Q):
    G = [[0.0] * n for _ in range(n)]
    for (u, v, _r0), R in zip(edges, R_list):
        cond = 1.0 / R
        G[u][u] += cond; G[v][v] += cond
        G[u][v] -= cond; G[v][u] -= cond
    I = [0.0] * n
    if s != g:
        I[s] += Q
        I[g] -= Q
    idxs = [i for i in range(n) if i != 0]
    m = len(idxs)
    A = [[G[idxs[i]][idxs[j]] for j in range(m)] for i in range(m)]
    b = [I[idxs[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            continue
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        pv = A[col][col]
        for r in range(m):
            if r != col and A[r][col] != 0:
                factor = A[r][col] / pv
                for c2 in range(col, m):
                    A[r][c2] -= factor * A[col][c2]
                b[r] -= factor * b[col]
    x = [0.0] * m
    for i in range(m):
        if abs(A[i][i]) > 1e-12:
            x[i] = b[i] / A[i][i]
    V = [0.0] * n
    for i, idx in enumerate(idxs):
        V[idx] = x[i]
    return V


GRID = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0,
        2.5, 3.0, 3.7, 4.5, 5.5, 7, 9, 12, 16, 22, 30]
FIT_TOL = 0.006


def rms_error(n, edges, override, patterns):
    R = [e[2] for e in edges]
    for idx, v in override.items():
        R[idx] = v
    tot = 0.0
    cnt = 0
    for (s, g, Q, Vs, Vg) in patterns:
        V = solve_nodal_float(n, edges, R, s, g, Q)
        tot += (V[s] - Vs) ** 2 + (V[g] - Vg) ** 2
        cnt += 2
    return (tot / max(1, cnt)) ** 0.5


def optimize_1var(n, edges, edge_idx, patterns, fixed=None):
    fixed = fixed or {}
    R0 = edges[edge_idx][2]
    best_v = R0
    best_e = rms_error(n, edges, dict(fixed), patterns)
    for f in GRID:
        v = R0 * f
        ov = dict(fixed); ov[edge_idx] = v
        e = rms_error(n, edges, ov, patterns)
        if e < best_e:
            best_e, best_v = e, v
    lo, hi = best_v * 0.5, best_v * 2.0
    for _ in range(30):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        ov1 = dict(fixed); ov1[edge_idx] = m1
        ov2 = dict(fixed); ov2[edge_idx] = m2
        e1 = rms_error(n, edges, ov1, patterns)
        e2 = rms_error(n, edges, ov2, patterns)
        if e1 < e2:
            hi = m2
        else:
            lo = m1
    v = (lo + hi) / 2
    ov = dict(fixed); ov[edge_idx] = v
    e = rms_error(n, edges, ov, patterns)
    if e < best_e:
        best_e, best_v = e, v
    return best_v, best_e


def optimize_2var(n, edges, a, b, patterns, rounds=3):
    va = edges[a][2] * 1.0
    vb = edges[b][2] * 1.0
    for _ in range(rounds):
        va, _ = optimize_1var(n, edges, a, patterns, fixed={b: vb})
        vb, _ = optimize_1var(n, edges, b, patterns, fixed={a: va})
    e = rms_error(n, edges, {a: va, b: vb}, patterns)
    return va, vb, e


def main():
    _test_id, n, edges, shown = read_instance()
    n_edges = len(edges)
    p1 = shown[:1]  # ONLY the first excitation pattern -- the trap

    for e in range(n_edges):
        v, err = optimize_1var(n, edges, e, p1)
        if err < FIT_TOL:
            print(1)
            print(e, "%.6f" % v)
            return

    for a in range(n_edges):
        for b in range(a + 1, n_edges):
            va, vb, err = optimize_2var(n, edges, a, b, p1)
            if err < FIT_TOL:
                print(2)
                print(a, "%.6f" % va)
                print(b, "%.6f" % vb)
                return

    print(0)


if __name__ == "__main__":
    main()
