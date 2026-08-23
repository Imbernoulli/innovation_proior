from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (audit-grade admission): a false edge is billed
# twice (adjacency precision AND SHD), a missed edge once, so nothing
# enters the report without surviving vetting. Scaffold pipeline:
#   admit  -- demanding marginal association bar, then
#   audit  -- re-test each admitted pair conditional on its strongest
#             shared neighbour; drop the pair if the association
#             collapses (classic common-cause signature).
# Output stays undirected here. Headroom: multi-conditioner audits,
# FDR-style threshold calibration, recall growth from the vetted core,
# and orientations that meet the same evidentiary bar.
# =====================================================================
def _pair_stat(x: np.ndarray, y: np.ndarray) -> float:
    """Cramer's V between two 0-based integer columns."""
    n = x.shape[0]
    if n == 0:
        return 0.0
    kx, ky = int(x.max()) + 1, int(y.max()) + 1
    tab = np.bincount(x * ky + y, minlength=kx * ky).reshape(kx, ky).astype(float)
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0).sum())
    return float(np.sqrt(chi2 / (n * max(1, min(kx, ky) - 1))))


def _conditional_stat(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      min_stratum: int = 30) -> float:
    """Size-weighted Cramer's V of (x, y) within the strata of z.

    The single-conditioner audit: if this collapses relative to the
    marginal statistic, the x-y association is plausibly routed through
    z and the pair should NOT be reported as an edge.
    """
    total, weight = 0.0, 0.0
    for v in np.unique(z):
        m = z == v
        cnt = int(m.sum())
        if cnt < min_stratum:
            continue
        total += cnt * _pair_stat(x[m], y[m])
        weight += cnt
    return total / weight if weight > 0 else 0.0


def run_causal_discovery(X: np.ndarray,
                         admit_thresh: float = 0.22,
                         survive_frac: float = 0.55) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    ``admit_thresh`` is the marginal admission bar (deliberately high:
    precision is the binding channel). ``survive_frac`` is the audit
    rule -- an admitted pair is kept only if its conditional statistic
    given the strongest shared neighbour retains at least this fraction
    of the marginal value. Both constants are placeholders for a
    calibrated false-discovery argument.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    X = np.asarray(X, dtype=np.int64)
    X = X - X.min(axis=0)
    d = X.shape[1]
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    V = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            V[i, j] = V[j, i] = _pair_stat(X[:, i], X[:, j])

    admitted = [(i, j) for i in range(d) for j in range(i + 1, d)
                if V[i, j] >= admit_thresh]

    for i, j in admitted:
        # Strongest shared neighbour = the most dangerous common cause.
        strength = np.minimum(V[i], V[j])
        strength[[i, j]] = 0.0
        k = int(np.argmax(strength))
        keep = True
        if strength[k] >= admit_thresh:
            cond = _conditional_stat(X[:, i], X[:, j], X[:, k])
            keep = cond >= survive_frac * V[i, j]
        if keep:
            # Undirected: arrowheads must meet a bar this scaffold does
            # not yet implement, so it declines to claim any.
            g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
