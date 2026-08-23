from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (scale-flat recovery): the task score is a geometric
# mean over five networks spanning 5 to 76 nodes, so the WEAKEST
# (usually the largest) network caps everything. One fixed configuration
# for all inputs; adapt only through statistics computed from X.
# Orientations must be earned -- a wrong arrow costs arrow precision AND
# SHD -- so the placeholder below claims no orientations at all.
# =====================================================================
def _pairwise_association(X: np.ndarray) -> np.ndarray:
    """Cramer's V for every variable pair (the skeleton evidence).

    Plain, uncorrected V. Replace or extend freely (G-test, BDeu local
    score deltas, conditional pruning) -- this is the evidence channel
    the scale-flat objective is supposed to improve.
    """
    n, d = X.shape
    card = X.max(axis=0).astype(int) + 1
    V = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            ki, kj = int(card[i]), int(card[j])
            tab = np.bincount(X[:, i] * kj + X[:, j], minlength=ki * kj)
            tab = tab.reshape(ki, kj).astype(float)
            rs, cs = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
            exp = rs @ cs / n
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0).sum())
            denom = n * max(1, min(ki, kj) - 1)
            V[i, j] = V[j, i] = float(np.sqrt(chi2 / denom))
    return V


def run_causal_discovery(X: np.ndarray, edges_per_node: float = 1.2) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    ``edges_per_node`` is the reporting budget: at most edges_per_node * d
    edges are emitted, so the output stays sparse at every scale instead
    of densifying quadratically on the 56- and 76-node networks. The
    budget (not a per-network alpha) is this variant's scale-adaptation
    hook; tune or replace the statistic behind it, never the identity of
    the input.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    X = np.asarray(X, dtype=np.int64)
    X = X - X.min(axis=0)  # ensure 0-based codes for the contingency count
    d = X.shape[1]
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    V = _pairwise_association(X)
    budget = int(round(edges_per_node * d))
    iu = np.triu_indices(d, k=1)
    order = np.argsort(V[iu])[::-1][:budget]
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        if V[i, j] <= 0.0:
            break
        # Undirected (TAIL-TAIL): the placeholder never orients. Earning
        # orientations (v-structures + propagation rules) is the headroom.
        g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
