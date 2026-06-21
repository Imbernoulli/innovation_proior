We need to recover the structure of a discrete Bayesian network from a database of complete, integer-coded cases. The graph is hidden, and the space of directed acyclic graphs is super-exponential, so exhaustive scoring is impossible. A principled criterion must rest on the marginal likelihood of the data given a structure, not on an ad-hoc fit measure, and it must be computable from simple counts. Constraint-based methods avoid scoring altogether but hinge on many conditional-independence tests that become unreliable when conditioning sets grow and data are finite. Greedy parent-selection methods like K2 are fast and effective, yet they require a correct topological ordering of the variables, which is exactly the kind of prior knowledge one usually lacks for purely observational data. Without an ordering, acyclicity is no longer free, and the search must wander the full DAG space while explicitly guarding against cycles.

The method I propose is Hill-Climbing (HC). It combines a decomposable, score-equivalent Bayesian-Dirichlet (BDeu) score with greedy best-improvement local search over single-arc edits in the space of DAGs. Because the score is a sum of per-node family terms, adding or deleting an arc changes only one local term, and reversing an arc changes only two. This makes each candidate move cheap to evaluate. The search starts from the empty graph and repeatedly applies the single add, delete, or reverse that most improves the lower-is-better score while keeping the graph acyclic, stopping when no single-arc edit improves the score. Reverse is treated as an atomic move so the search can flip an arc without passing through a worse intermediate that delete-then-add would require. Since observational data identifies only the Markov equivalence class, not the individual DAG, the learned DAG is collapsed to a CPDAG before returning.

The score is the marginal likelihood p(D | B_S) integrated over the conditional probability table parameters against a Dirichlet prior. Under multinomial sampling, parameter independence, complete data, parameter modularity, and Dirichlet priors, the integral collapses to ratios of Gamma functions over the observed cell counts. Score equivalence holds when the Dirichlet pseudocounts come from an equivalent-sample-size construction with a uniform prior joint, giving α_ijk = N' / (r_i q_i) for equivalent sample size N'. The implementation uses N' = 1 and a default structure prior, so the metric is BDeu. The log-domain score is numerically stable and the local score routine returns the negative log-score so lower is better.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BDeu
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    N = X.shape[1]
    # Decomposable BDeu local score; lower is better.
    # parameters=None auto-detects cardinalities and uses N' = 1.
    score_func = LocalScoreClass(
        data=X, local_score_fun=local_score_BDeu, parameters=None
    )

    nodes = [GraphNode(f"X{i + 1}") for i in range(N)]
    adj = np.zeros((N, N), dtype=int)  # adj[i, j] == 1 means i -> j

    local_scores = np.zeros(N)
    for j in range(N):
        local_scores[j] = score_func.score(j, [])

    def _has_path(src, tgt):
        """DFS: is there a directed path src -> ... -> tgt in adj?"""
        visited, stack = set(), [src]
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

    improved = True
    while improved:
        improved = False
        best_delta, best_op = 0.0, None

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                if adj[i, j] == 0 and adj[j, i] == 0:
                    # ADD i -> j (legal iff it closes no cycle)
                    if not _has_path(j, i):
                        pj_new = sorted(np.where(adj[:, j] == 1)[0].tolist() + [i])
                        delta = score_func.score(j, pj_new) - local_scores[j]
                        if delta < best_delta - 1e-6:
                            best_delta, best_op = delta, ("add", i, j)

                elif adj[i, j] == 1:
                    # DELETE i -> j
                    pj_new = [p for p in np.where(adj[:, j] == 1)[0] if p != i]
                    delta = score_func.score(j, sorted(pj_new)) - local_scores[j]
                    if delta < best_delta - 1e-6:
                        best_delta, best_op = delta, ("delete", i, j)

                    # REVERSE i -> j to j -> i (both Pi_j and Pi_i change)
                    adj[i, j] = 0
                    if not _has_path(i, j):
                        pj_del = sorted(np.where(adj[:, j] == 1)[0].tolist())
                        pi_new = sorted(np.where(adj[:, i] == 1)[0].tolist() + [j])
                        delta = (
                            (score_func.score(j, pj_del) - local_scores[j])
                            + (score_func.score(i, pi_new) - local_scores[i])
                        )
                        if delta < best_delta - 1e-6:
                            best_delta, best_op = delta, ("reverse", i, j)
                    adj[i, j] = 1

        if best_op is not None:
            op, i, j = best_op
            if op == "add":
                adj[i, j] = 1
            elif op == "delete":
                adj[i, j] = 0
            elif op == "reverse":
                adj[i, j] = 0
                adj[j, i] = 1
            local_scores[j] = score_func.score(
                j, sorted(np.where(adj[:, j] == 1)[0].tolist())
            )
            if op == "reverse":
                local_scores[i] = score_func.score(
                    i, sorted(np.where(adj[:, i] == 1)[0].tolist())
                )
            improved = True

    G = GeneralGraph(nodes)
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                G.add_directed_edge(nodes[i], nodes[j])
    return dag2cpdag(G)
```
