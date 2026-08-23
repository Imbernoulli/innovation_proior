from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (anytime under a compute envelope): a valid graph
# must exist early and only improve; ALL heavy work is metered against
# an explicit wall-clock budget the algorithm enforces on itself.
# Scaffold stages:
#   triage -- rank all pairs by a cheap association score computed on a
#             row subsample (fast even at 76 nodes);
#   refine -- while budget remains, re-score the top-ranked candidates
#             on the FULL sample, best first.
# Admission then reads refined scores where available and triage scores
# otherwise, so stopping at ANY point still yields a coherent output.
# Headroom: budgeted conditional tests and orientation, smarter
# effort allocation (hubs, ambiguous pairs), operation-count metering.
# =====================================================================
def _v_stat(x: np.ndarray, y: np.ndarray) -> float:
    """Cramer's V on two 0-based integer columns (the unit of work)."""
    n = x.shape[0]
    kx, ky = int(x.max()) + 1, int(y.max()) + 1
    tab = np.bincount(x * ky + y, minlength=kx * ky).reshape(kx, ky).astype(float)
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0).sum())
    return float(np.sqrt(chi2 / (n * max(1, min(kx, ky) - 1))))


def run_causal_discovery(X: np.ndarray,
                         time_budget_s: float = 120.0,
                         triage_rows: int = 1500,
                         edge_thresh: float = 0.20) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    ``time_budget_s`` is the self-imposed envelope for the refinement
    stage; the triage pass is cheap by construction (``triage_rows``
    caps its sample). Per-pair effort thus scales DOWN automatically as
    the number of candidate pairs grows -- that is the only form of
    size adaptation this variant permits. The scaffold logs its budget
    accounting so envelope compliance is auditable from stdout.
    """
    import time
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    t0 = time.time()
    X = np.asarray(X, dtype=np.int64)
    X = X - X.min(axis=0)
    n, d = X.shape
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    # --- stage 1: triage on a row subsample -------------------------------
    rows = np.random.RandomState(0).choice(n, size=min(n, triage_rows),
                                           replace=False)
    Xs = X[rows]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    score = {p: _v_stat(Xs[:, p[0]], Xs[:, p[1]]) for p in pairs}
    order = sorted(pairs, key=lambda p: -score[p])

    # --- stage 2: budgeted full-sample refinement, best first -------------
    refined = 0
    for p in order:
        if time.time() - t0 > time_budget_s:
            break
        if score[p] < 0.5 * edge_thresh:
            break  # remaining candidates are hopeless; save the budget
        score[p] = _v_stat(X[:, p[0]], X[:, p[1]])
        refined += 1

    print(f"[anytime] d={d} pairs={len(pairs)} refined={refined} "
          f"elapsed={time.time() - t0:.1f}s budget={time_budget_s}s",
          flush=True)

    # --- emit: whatever the budget bought, as an always-valid graph -------
    for i, j in pairs:
        if score[(i, j)] >= edge_thresh:
            # Orientation is future budgeted work; undirected for now.
            g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
