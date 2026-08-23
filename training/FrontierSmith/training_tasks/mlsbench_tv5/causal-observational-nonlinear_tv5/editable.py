# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (compute-frugal, anytime): every quantity below is a
# closed-form statistic obtained in O(1) passes over X -- correlation
# matrices on raw values, ranks, and squares, plus third-order cross
# moments. Directed-edge claims are sorted by confidence and admitted
# greedily under an acyclicity check, so any prefix of the stream is a
# coherent partial DAG. The cheap direction cues and the stream cutoff
# are the intended headroom; iterative fitting is out of scope.
# =====================================================================
def _reaches(admitted: np.ndarray, a: int, b: int) -> bool:
    """True iff b is reachable from a in the currently admitted digraph
    (admitted[u, v] = True means edge u -> v). Tiny BFS; d is small."""
    frontier = [a]
    seen = np.zeros(admitted.shape[0], dtype=bool)
    while frontier:
        u = frontier.pop()
        if u == b:
            return True
        for v in np.nonzero(admitted[u])[0]:
            if not seen[v]:
                seen[v] = True
                frontier.append(v)
    return False


def run_causal_discovery(X: np.ndarray, edge_factor: float = 1.2) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    Single-pass statistics -> one confidence-ranked list of candidate
    arrows -> greedy acyclic admission, stopping at a dependence floor
    or after ``edge_factor * d`` arrows, whichever comes first.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    Rk = np.argsort(np.argsort(Z, axis=0), axis=0).astype(np.float64)
    Rk = (Rk - Rk.mean(axis=0)) / (Rk.std(axis=0) + 1e-12)
    Z2 = Z * Z
    Z2 = (Z2 - Z2.mean(axis=0)) / (Z2.std(axis=0) + 1e-12)

    P = (Z.T @ Z) / n          # Pearson
    S = (Rk.T @ Rk) / n        # Spearman
    Q = (Z2.T @ Z) / n         # Q[i, j] = corr(z_i^2, z_j): nonlinearity cue
    T = (Z.T @ (Z * Z2)) / n   # T[i, j] ~ E[z_i z_j^3]: tail/skew cue

    dep = np.maximum(np.abs(P), np.abs(S))
    floor = 2.6 / np.sqrt(n)

    # direction score for i -> j: positive favors i as the cause
    cand = []
    for i in range(d):
        for j in range(i + 1, d):
            if dep[i, j] <= floor:
                continue
            cue_nl = np.abs(Q[i, j]) - np.abs(Q[j, i])
            cue_mo = np.abs(T[i, j]) - np.abs(T[j, i])
            score = cue_nl + 0.5 * cue_mo
            a, b = (i, j) if score >= 0 else (j, i)      # a -> b
            conf = dep[i, j] * (1.0 + min(1.0, abs(score)))
            cand.append((conf, a, b))
    cand.sort(reverse=True)

    admitted = np.zeros((d, d), dtype=bool)   # admitted[u, v]: u -> v
    max_edges = int(edge_factor * d)
    n_admitted = 0
    for conf, a, b in cand:
        if n_admitted >= max_edges:
            break
        if _reaches(admitted, b, a):          # would close a cycle: skip
            continue
        admitted[a, b] = True
        n_admitted += 1

    B = np.zeros((d, d))
    for a in range(d):
        for b in np.nonzero(admitted[a])[0]:
            B[b, a] = 1.0                     # a -> b in causal-learn form
    return B
# =====================================================================
