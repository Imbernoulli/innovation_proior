from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (multiplicity-aware false-positive control): the
# admission rule must budget errors over the WHOLE reported edge set
# (FDR-style), never test-by-test. The placeholder runs a Benjamini-
# Hochberg step-up over Fisher-z p-values of the MARGINAL correlations,
# so indirect associations still leak through; upgrading the evidence
# to conditional tests while keeping the set-level guarantee is the
# variant's headroom. Edges are left undirected on purpose: orientation
# belongs on top of an FP-controlled skeleton, not before it.
# =====================================================================
def run_causal_discovery(X: np.ndarray, fdr_q: float = 0.05) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    ``fdr_q`` is the target false-discovery rate for the reported
    adjacency set. It is the ONLY knob, and it is a rate rather than a
    correlation cutoff, so a single setting serves every regime.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from scipy.stats import norm

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    C = np.clip(np.corrcoef(X, rowvar=False), -0.999999, 0.999999)
    iu, ju = np.triu_indices(d, k=1)
    z = np.abs(np.arctanh(C[iu, ju])) * np.sqrt(max(n - 3, 1))
    pvals = 2.0 * norm.sf(z)

    # Benjamini-Hochberg step-up on the m = d*(d-1)/2 candidate pairs.
    m = pvals.size
    order = np.argsort(pvals)
    passed = np.flatnonzero(pvals[order] <= fdr_q * np.arange(1, m + 1) / m)
    n_admit = int(passed[-1]) + 1 if passed.size else 0

    for idx in order[:n_admit]:
        i, j = int(iu[idx]), int(ju[idx])
        g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
