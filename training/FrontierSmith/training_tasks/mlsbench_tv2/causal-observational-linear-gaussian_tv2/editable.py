from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (orientation-first recovery): arrowheads are the
# deliverable. Every ARROW endpoint must trace to a detected unshielded
# collider or to a Meek propagation step licensed by one -- no decree
# orientations. The collider detector below is the naive part: it flags
# i -> k <- j whenever conditioning on k STRENGTHENS the i,j association
# (the classic collider signature) with no error control, and there is
# no Meek pass at all. Making detection reliable at n=400/noise=2.5 and
# propagating soundly is this variant's headroom.
# =====================================================================
def _partial_corr(C: np.ndarray, i: int, j: int, k: int) -> float:
    """First-order partial correlation r(i,j | k) from a correlation matrix."""
    den = ((1.0 - C[i, k] ** 2) * (1.0 - C[j, k] ** 2)) ** 0.5
    if den <= 1e-12:
        return 0.0
    return float((C[i, j] - C[i, k] * C[j, k]) / den)


def run_causal_discovery(X: np.ndarray, adj_thresh: float = 0.25,
                         collider_margin: float = 0.05) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    Placeholder: correlation-threshold skeleton, then a single collider
    sweep over unshielded triples. Conflicting arrowheads back off to a
    tail-tail edge, because in this variant a wrong arrow costs more
    than a missing one.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 0.0)
    C = np.clip(C, -0.999999, 0.999999)
    A = np.abs(C) > adj_thresh

    # arrow[i, k] == True  means an arrowhead at k on the i -- k edge.
    arrow = np.zeros((d, d), dtype=bool)
    for k in range(d):
        nb = np.flatnonzero(A[k])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                i, j = int(nb[a]), int(nb[b])
                if A[i, j]:
                    continue  # shielded triple: no collider information
                if abs(_partial_corr(C, i, j, k)) > abs(C[i, j]) + collider_margin:
                    arrow[i, k] = True
                    arrow[j, k] = True

    for i in range(d):
        for j in range(i + 1, d):
            if not A[i, j]:
                continue
            head_i, head_j = arrow[j, i], arrow[i, j]
            if head_i and head_j:  # conflicting claims: refuse both
                head_i = head_j = False
            g.add_edge(Edge(nodes[i], nodes[j],
                            Endpoint.ARROW if head_i else Endpoint.TAIL,
                            Endpoint.ARROW if head_j else Endpoint.TAIL))
    return g
