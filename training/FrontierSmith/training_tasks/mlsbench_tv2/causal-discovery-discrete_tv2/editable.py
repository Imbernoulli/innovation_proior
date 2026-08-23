from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (orientation is the deliverable): arrow precision,
# arrow recall AND SHD all react to edge directions, so a graph that
# never orients forfeits a whole scored channel. This scaffold ships a
# minimal two-stage pipeline: (1) thresholded-association skeleton,
# (2) one crude collider rule -- orient i -> k <- j when i-k and j-k
# are edges but i and j look marginally independent. Both stages are
# deliberately naive; the headroom is certifying colliders with real
# conditioning-set evidence and Meek-style propagation while keeping
# arrow precision from sagging below adjacency precision.
# =====================================================================
def _assoc_matrix(X: np.ndarray) -> np.ndarray:
    """Cramer's-V-style association for every pair.

    Serves two roles here: skeleton admission evidence AND the marginal
    (in)dependence signal that the naive collider rule reads. Upgrading
    this shared evidence channel (conditional tests, score deltas)
    upgrades both stages at once.
    """
    n, d = X.shape
    card = X.max(axis=0).astype(int) + 1
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            ki, kj = int(card[i]), int(card[j])
            tab = np.bincount(X[:, i] * kj + X[:, j],
                              minlength=ki * kj).reshape(ki, kj).astype(float)
            exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0).sum())
            A[i, j] = A[j, i] = float(np.sqrt(chi2 / (n * max(1, min(ki, kj) - 1))))
    return A


def run_causal_discovery(X: np.ndarray,
                         edge_thresh: float = 0.18,
                         indep_thresh: float = 0.07) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph

    ``edge_thresh`` admits skeleton edges; ``indep_thresh`` is the bar
    under which an unshielded triple i-k-j is read as a collider and k
    receives arrowheads from both sides. The collider call is UNSHIELDED
    + MARGINAL only -- no conditioning sets, no propagation, no conflict
    resolution beyond "orient only when exactly one direction is
    endorsed". That conservatism is the starting point, not the goal:
    earned arrows are where this variant's score lives.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    X = np.asarray(X, dtype=np.int64)
    X = X - X.min(axis=0)  # 0-based codes for the contingency counts
    d = X.shape[1]
    nodes = [GraphNode(f"X{i + 1}") for i in range(d)]
    g = GeneralGraph(nodes)

    A = _assoc_matrix(X)
    adj = A >= edge_thresh
    np.fill_diagonal(adj, False)

    # Collider endorsements: into[i, k] means some unshielded triple
    # endorsed the arrowhead i -> k.
    into = np.zeros((d, d), dtype=bool)
    for k in range(d):
        nb = np.flatnonzero(adj[k])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                i, j = int(nb[a]), int(nb[b])
                if not adj[i, j] and A[i, j] < indep_thresh:
                    into[i, k] = True
                    into[j, k] = True

    for i in range(d):
        for j in range(i + 1, d):
            if not adj[i, j]:
                continue
            if into[i, j] and not into[j, i]:
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
            elif into[j, i] and not into[i, j]:
                g.add_edge(Edge(nodes[j], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
            else:
                # Conflicting or absent endorsements: stay undirected.
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return g
