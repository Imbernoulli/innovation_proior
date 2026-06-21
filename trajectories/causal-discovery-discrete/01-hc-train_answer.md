The scaffold hands me an empty graph on the right number of nodes and a single function to fill, so the floor I have to start from is the simplest thing that is still a real structure-learning method: score a candidate DAG and walk to a better one by single-arc edits. There is no variable ordering given, no conditional-independence machinery, no equivalence-class bookkeeping yet — and that is deliberate. The classical alternatives all fall short here. A marginal-likelihood score (Cooper & Herskovits 1992; Heckerman, Geiger & Chickering 1995) is a criterion but not a search, and enumerating all DAGs is impossible (the count is super-exponential — hundreds of digits at 76 nodes). The exact optimum exists only in the at-most-one-parent regime as a maximum-weight tree (Chow & Liu 1968), too poor to be useful; with two or more parents the problem is NP-hard. Greedy parent selection from a known topological order (K2) makes acyclicity free but presupposes the very parent/child directions I am trying to learn — wrong order in, reversed arcs out — and observational data hands me no such order. So the floor must be a heuristic search over the full DAG space, paying for the dropped ordering with an explicit acyclicity check at every move.

I propose hill-climbing DAG search scored by the BDeu marginal likelihood. The score comes from the right probabilistic question: between two structures $B_S$ over the discrete variables, which does the data prefer? By Bayes, $p(B_S \mid D) \propto p(B_S)\,p(D \mid B_S)$, and the normalizer $p(D)$ is the same constant for every structure, so it cancels in any comparison — I only ever need the relative joint. The load-bearing term is $p(D \mid B_S)$, the probability of the data given the *structure*, not given particular CPT numbers. The honest way to handle the uncertain table entries $\Theta$ is to integrate them out, $p(D \mid B_S) = \int p(D \mid B_S, \Theta)\,p(\Theta \mid B_S)\,d\Theta$ — an *average* over parameter settings, not the maximized likelihood of one best fit. That distinction does real work: averaging over $\Theta$ automatically discounts a structure with many parents, because its prior mass is spread thin over a high-dimensional $\Theta$, so the marginal likelihood carries a built-in Occam penalty with no explicit regularizer.

The integral collapses under the standard assumptions — i.i.d. multinomial sampling, parameter independence across families, complete data, and a Dirichlet prior. The Dirichlet is conjugate to the multinomial, so each family integral is a ratio of Dirichlet normalizers, i.e. a ratio of Gammas. With $r_i$ the number of states of $X_i$, $q_i = \prod_{p \in \Pi_i} r_p$ the number of parent configurations, counts $N_{ijk}$ (records with $X_i = k$ and parents in configuration $j$), $N_{ij} = \sum_k N_{ijk}$, and Dirichlet pseudocounts $\alpha$, the closed-form score per node sums over the observed parent configurations
$$s(X_i \mid \Pi_i) = \sum_j \Big[\mathrm{lgamma}(\alpha_{ij}) - \mathrm{lgamma}(N_{ij} + \alpha_{ij}) + \sum_k \big(\mathrm{lgamma}(N_{ijk} + \alpha_{ijk}) - \mathrm{lgamma}(\alpha_{ijk})\big)\Big].$$
I work in the log domain (products of Gammas underflow on any real database), so the total is a *sum* over nodes, $\sum_i s(X_i \mid \Pi_i)$. That sum shape is the single most important property I have: the score is **decomposable**, every term depending only on $X_i$ and the counts conditioned on its parents. Change the structure by touching only the parents of one node and every term stays fixed except that node's — to compare the new structure to the old I recompute one local term, not the whole graph. That is the lever that makes searching a super-exponential space thinkable.

Score equivalence pins down the pseudocounts. With unit pseudocounts ($\alpha_{ijk} = 1$, the K2 special case) the score is *not* generally score-equivalent — isomorphic structures can get different values, and the search would chase a within-class orientation carrying no information. Tying the pseudocounts to a uniform prior over the whole domain and a single scalar $N'$ (the equivalent sample size) gives $\alpha_{ijk} = N' / (r_i q_i)$ and $\alpha_{ij} = N' / q_i$: a uniform pseudocount per cell, scaled by $N'$. This is BDeu ("u" for uniform), the unique BD score with score equivalence, needing no elicited prior — just the scalar $N'$. In the scaffold this is exactly `LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)`; passing `parameters=None` auto-infers the cardinalities $r_i$ from the data and uses the canonical defaults ($N' = 1$, structure prior $1$). One convention to mind: the harness's `local_score_BDeu` returns a quantity where *lower is better*, so the climb looks for the most-negative delta, and "improving" reads as $\Delta < \text{best} - \varepsilon$ with $\varepsilon = 10^{-6}$ to avoid thrashing on numerically-zero ties.

With the score fixed, the search is best-improvement hill-climbing over single-arc moves. For any ordered pair $(i, j)$: if there is no arc I can **add** $i \to j$; if there is an arc $i \to j$ I can **delete** it or **reverse** it to $j \to i$. Add, delete, reverse make any DAG reachable from any other, so the move set is complete. Reverse is kept as an atomic move even though it is a delete followed by an add, because the intermediate graph (neither arc) can score worse than both endpoints; a greedy step-by-step search would never take the first step, so reverse is what lets me cross that valley and fix an arc oriented backwards in one scored unit. By decomposability, add or delete $i \to j$ changes only $X_j$'s parent set — one local recompute, $\Delta = s(X_j \mid \Pi_j^{\text{new}}) - s(X_j \mid \Pi_j^{\text{old}})$ — while reverse changes the parents of both endpoints, so it is two local recomputes summed. I cache one local score per node and refresh only the touched node(s) after an accepted move. The instant the ordering is dropped, acyclicity stops being free, so each candidate add or reverse is guarded by a DFS reachability test: adding $i \to j$ is illegal iff $i$ is already reachable from $j$; reversing $i \to j$ is tested by temporarily removing the arc and asking whether $j$ is still reachable from $i$ (if so, $j \to i$ would close a cycle), then restoring it. The loop scans every eligible move, remembers the single best improving one, applies it, updates the cached local score(s), and stops when no single-arc edit improves the score — a local optimum.

Finally, observational data identifies only the Markov equivalence class: $X \to Y \to Z$, $X \leftarrow Y \to Z$, and $X \leftarrow Y \leftarrow Z$ all assert exactly $X \perp\!\!\!\perp Z \mid Y$ and nothing else, and a score-equivalent score gives them equal value. So wherever the climb lands, covered-edge orientations are arbitrary; the object I can honestly claim is the class, not the particular DAG. After climbing I collapse the learned DAG to its CPDAG with `dag2cpdag` — keeping an edge directed only where every member of the class agrees — before returning. Returning the raw DAG would overclaim orientations the data never determined and be penalized on the arrow metrics. I am clear-eyed about what this floor cannot do: DAG space is riddled with local optima, and best single-arc improvement from one empty-graph start carries no guarantee of reaching the global best. Density will hurt most (the right correction is often a coordinated change of several arcs no single edit can reach), and orientation will be the soft spot, because a climb that lands in the wrong basin leaves compelled arrows that `dag2cpdag` cannot rescue. I expect it decent on the small networks and degrading sharply with size — which is exactly the failure the next rung must address by changing the *search space*.

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
