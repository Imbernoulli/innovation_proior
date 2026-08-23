from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (SNR-calibrated decisions): the noisy dense setting
# (noise 2.5, n=400) and the hub-dense scale-free graph are what cap the
# geometric mean. Every threshold must be a function of statistics
# estimated from X (n, d, noise level, correlation strength) -- never a
# frozen constant or a branch keyed to the input's shape.
# =====================================================================
def _shrinkage_intensity(n: int, d: int) -> float:
    """Continuous regularization knob: more shrinkage when d/n is large.

    This is the variant's intended adaptation channel -- replace with a
    data-driven estimate (Ledoit-Wolf, residual-variance calibration,
    stability selection) rather than a per-scenario constant.
    """
    return float(min(0.9, max(0.05, d / max(n, 1))))


def run_causal_discovery(X: np.ndarray, base_alpha: float = 0.01) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    Placeholder: a partial-correlation skeleton whose Fisher-z critical
    value uses an effective sample size and a shrunk precision matrix, so
    ONE configuration covers n=400/noise=2.5 and n=2000/noise=1.0. It
    emits undirected edges only -- deliberately weak on the arrow
    metrics; orientations have to be earned by the agent.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from scipy.stats import norm

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    C = np.corrcoef(X, rowvar=False)
    lam = _shrinkage_intensity(n, d)
    P = np.linalg.pinv((1.0 - lam) * C + lam * np.eye(d))
    s = np.sqrt(np.abs(np.diag(P)))
    s = np.where(s > 0, s, 1.0)
    pcorr = -P / np.outer(s, s)
    np.fill_diagonal(pcorr, 0.0)
    pcorr = np.clip(pcorr, -0.999999, 0.999999)

    # Fisher z with a degrees-of-freedom correction for conditioning on
    # the remaining d-2 variables; degrades gracefully at small n.
    eff = max(n - (d - 2) - 3, 8)
    z = np.abs(np.arctanh(pcorr)) * np.sqrt(eff)
    crit = float(norm.ppf(1.0 - base_alpha / 2.0))

    for i in range(d):
        for j in range(i + 1, d):
            if z[i, j] > crit:
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
