**Problem.** Recover the discrete network's structure — as much as observational data identifies — from
integer-coded cases, with a clean probabilistic criterion computable from counts, in a space far too
large to enumerate. This is the floor of the ladder: a score plus a greedy local search, no ordering,
no CI tests, no equivalence-class machinery.

**Key idea.** Score a DAG by the BDeu marginal likelihood (Dirichlet-multinomial, closed-form Gamma
ratios from cell counts, with a built-in complexity penalty), which is **decomposable** so a single-arc
edit costs one or two local recomputes, and **score-equivalent** so it gives Markov-equivalent
structures equal value. Search the full DAG space by best-improvement hill-climbing over single-arc
**add / delete / reverse** moves, each guarded by a DFS acyclicity check (no ordering is assumed, so
acyclicity must be enforced explicitly). Reverse is kept as an atomic move because delete-then-add would
pass through a worse intermediate the greedy step would refuse. Start from the empty graph; stop at the
local optimum where no single-arc edit improves the (lower-is-better) score.

**Why it is the weakest rung.** It searches DAG space, not equivalence-class space, so it wastes moves
on covered-edge reversals that change the DAG but not the class, and it stalls in shallow optima — worst
on the larger, denser networks. It does no CI reasoning and uses a single fixed start (the empty graph)
with no restarts. Because observational data identifies only the equivalence class, the learned DAG is
collapsed to its CPDAG with `dag2cpdag` before return.

**Hyperparameters.** BDeu via `LocalScoreClass(local_score_fun=local_score_BDeu, parameters=None)` —
equivalent sample size `N' = 1`, structure prior `1`, cardinalities `r_i` auto-inferred from the data.
Improvement tolerance `ε = 1e-6`. Empty-graph initialization; best-improvement (steepest single edit).

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BDeu
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    N = X.shape[1]
    # Pass parameters=None so local_score_BDeu auto-computes r_i_map from data
    score_func = LocalScoreClass(
        data=X, local_score_fun=local_score_BDeu, parameters=None
    )

    nodes = [GraphNode(f"X{i + 1}") for i in range(N)]
    adj = np.zeros((N, N), dtype=int)

    # Cache local scores (one per node)
    local_scores = np.zeros(N)
    for j in range(N):
        local_scores[j] = score_func.score(j, [])

    def _has_path(src, tgt):
        """DFS check: is there a directed path from src to tgt in adj?"""
        visited = set()
        stack = [src]
        while stack:
            node = stack.pop()
            if node == tgt:
                return True
            if node in visited:
                continue
            visited.add(node)
            for c in np.where(adj[node] == 1)[0]:
                if int(c) not in visited:
                    stack.append(int(c))
        return False

    # Greedy hill-climbing: add / delete / reverse
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_op = None

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                if adj[i, j] == 0 and adj[j, i] == 0:
                    # --- Try ADD i -> j (only if no cycle) ---
                    if not _has_path(j, i):
                        pj_new = sorted(
                            np.where(adj[:, j] == 1)[0].tolist() + [i]
                        )
                        new_sj = score_func.score(j, pj_new)
                        delta = new_sj - local_scores[j]
                        if delta < best_delta - 1e-6:
                            best_delta = delta
                            best_op = ("add", i, j)

                elif adj[i, j] == 1:
                    # --- Try DELETE i -> j ---
                    pj_new = [
                        p for p in np.where(adj[:, j] == 1)[0] if p != i
                    ]
                    new_sj = score_func.score(j, sorted(pj_new))
                    delta = new_sj - local_scores[j]
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best_op = ("delete", i, j)

                    # --- Try REVERSE i -> j  to  j -> i ---
                    adj[i, j] = 0  # temporarily remove
                    if not _has_path(i, j):
                        pj_del = sorted(
                            np.where(adj[:, j] == 1)[0].tolist()
                        )
                        new_sj = score_func.score(j, pj_del)
                        pi_new = sorted(
                            np.where(adj[:, i] == 1)[0].tolist() + [j]
                        )
                        new_si = score_func.score(i, pi_new)
                        delta = (
                            (new_sj - local_scores[j])
                            + (new_si - local_scores[i])
                        )
                        if delta < best_delta - 1e-6:
                            best_delta = delta
                            best_op = ("reverse", i, j)
                    adj[i, j] = 1  # restore

        if best_op is not None:
            op_type, i, j = best_op
            if op_type == "add":
                adj[i, j] = 1
            elif op_type == "delete":
                adj[i, j] = 0
            elif op_type == "reverse":
                adj[i, j] = 0
                adj[j, i] = 1
            # Recompute affected local scores
            local_scores[j] = score_func.score(
                j, sorted(np.where(adj[:, j] == 1)[0].tolist())
            )
            if op_type == "reverse":
                local_scores[i] = score_func.score(
                    i, sorted(np.where(adj[:, i] == 1)[0].tolist())
                )
            improved = True

    # Build GeneralGraph from learned DAG
    G = GeneralGraph(nodes)
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                G.add_directed_edge(nodes[i], nodes[j])

    G = dag2cpdag(G)
    return G
```
