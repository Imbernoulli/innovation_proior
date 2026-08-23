from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (stability selection): an edge enters the reported
# graph only if it is re-selected across resampled versions of the
# data. The placeholder wires the loop -- half-sample replicates of a
# correlation-threshold base learner, then a selection-frequency bar --
# but the base learner is deliberately weak and nothing downstream
# consumes the frequencies yet. Stronger base learners, smarter
# resampling, and frequency-aware orientation are the headroom. The
# resampling design and the bar are fixed once for all five regimes.
# =====================================================================
def run_causal_discovery(X: np.ndarray, n_resamples: int = 12,
                         keep_frac: float = 0.75,
                         corr_thresh: float = 0.25) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    Each replicate fits on a random half of the rows; an edge is
    reported (undirected) when selected in >= keep_frac of replicates.
    The RNG is seeded so repeated calls give the same graph.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    rng = np.random.default_rng(0)
    half = max(d + 2, n // 2)
    counts = np.zeros((d, d))
    for _ in range(n_resamples):
        rows = rng.choice(n, size=min(half, n), replace=False)
        C = np.corrcoef(X[rows], rowvar=False)
        np.fill_diagonal(C, 0.0)
        counts += (np.abs(C) > corr_thresh)

    freq = counts / float(n_resamples)
    for i in range(d):
        for j in range(i + 1, d):
            if freq[i, j] >= keep_frac:
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
