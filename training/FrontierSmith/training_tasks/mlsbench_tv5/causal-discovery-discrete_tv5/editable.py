from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (recall recovery without densification): raw
# association thresholds starve edges that touch high-cardinality
# variables -- the test's degrees of freedom explode and real edges
# vanish. This scaffold therefore admits edges on a STANDARDIZED
# chi-square scale, z = (chi2 - df) / sqrt(2 df), which puts a 2x2
# table and an 11x8 table on the same footing, and adds a minimal
# degree-floor rescue: any variable left isolated gets its single
# best-z partner back. A per-node cap keeps the rescue from turning
# into densification. Headroom: category pooling/coarsening, smarter
# power-aware statistics, rescue guided by the rest of the graph.
# =====================================================================
def _std_chi2_matrix(X: np.ndarray) -> np.ndarray:
    """Standardized chi-square z-score for every variable pair.

    Under independence chi2 has mean df and variance 2*df, so z is
    approximately cardinality-comparable -- the core device of this
    variant's recall objective.
    """
    n, d = X.shape
    card = X.max(axis=0).astype(int) + 1
    Z = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            ki, kj = int(card[i]), int(card[j])
            tab = np.bincount(X[:, i] * kj + X[:, j],
                              minlength=ki * kj).reshape(ki, kj).astype(float)
            exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0).sum())
            df = max(1, (ki - 1) * (kj - 1))
            Z[i, j] = Z[j, i] = (chi2 - df) / np.sqrt(2.0 * df)
    return Z


def run_causal_discovery(X: np.ndarray,
                         z_thresh: float = 8.0,
                         max_degree: int = 6) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    ``z_thresh`` is the admission bar on the standardized scale (NOT on
    raw association strength -- that is the whole point). ``max_degree``
    caps reported neighbours per node so recall-chasing cannot densify
    the output. The isolated-node rescue below is the crudest possible
    version of "targeted recall recovery"; replace it with something
    that consults the surrounding structure.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    X = np.asarray(X, dtype=np.int64)
    X = X - X.min(axis=0)
    d = X.shape[1]
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    Z = _std_chi2_matrix(X)
    keep = np.zeros((d, d), dtype=bool)

    # Degree-capped admission, strongest pairs first.
    iu = np.triu_indices(d, k=1)
    deg = np.zeros(d, dtype=int)
    for k in np.argsort(Z[iu])[::-1]:
        i, j = int(iu[0][k]), int(iu[1][k])
        if Z[i, j] < z_thresh:
            break
        if deg[i] < max_degree and deg[j] < max_degree:
            keep[i, j] = keep[j, i] = True
            deg[i] += 1
            deg[j] += 1

    # Rescue pass: no variable in these networks is truly isolated, so
    # an empty neighbourhood is read as test starvation, not absence.
    for i in range(d):
        if deg[i] == 0:
            j = int(np.argmax(Z[i]))
            if Z[i, j] > 0 and not keep[i, j]:
                keep[i, j] = keep[j, i] = True
                deg[i] += 1
                deg[j] += 1

    for i in range(d):
        for j in range(i + 1, d):
            if keep[i, j]:
                # Undirected placeholder output; orientation inherits
                # whatever recall the skeleton manages to recover.
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
