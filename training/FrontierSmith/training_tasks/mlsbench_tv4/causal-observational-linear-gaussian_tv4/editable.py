from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (hub-aware discovery for scale-free graphs): under
# preferential attachment the children of a hub are pairwise correlated
# through it, so degree-blind thresholding turns stars into cliques.
# The placeholder estimates a degree profile from the correlation
# matrix and re-tests every candidate pair conditional on its highest-
# degree common neighbour (the suspected hub). Staying harmless on
# flat-degree ER graphs, choosing richer conditioning sets, and adding
# hub-aware orientation are the headroom.
# =====================================================================
def run_causal_discovery(X: np.ndarray, screen_thresh: float = 0.2,
                         keep_thresh: float = 0.1) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    ``screen_thresh`` builds the candidate skeleton and the degree
    proxy; ``keep_thresh`` is applied to the hub-deconfounded partial
    correlation. Edges are emitted undirected.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    C = np.clip(np.corrcoef(X, rowvar=False), -0.999999, 0.999999)
    np.fill_diagonal(C, 0.0)
    cand = np.abs(C) > screen_thresh
    degree = cand.sum(axis=0)  # crude hubness proxy from the screen

    for i in range(d):
        for j in range(i + 1, d):
            if not cand[i, j]:
                continue
            common = np.flatnonzero(cand[i] & cand[j])
            common = common[(common != i) & (common != j)]
            keep = True
            if common.size:
                k = int(common[np.argmax(degree[common])])  # suspected hub
                den = ((1.0 - C[i, k] ** 2) * (1.0 - C[j, k] ** 2)) ** 0.5
                pc = 0.0 if den <= 1e-12 else (C[i, j] - C[i, k] * C[j, k]) / den
                keep = abs(pc) > keep_thresh
            if keep:
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
