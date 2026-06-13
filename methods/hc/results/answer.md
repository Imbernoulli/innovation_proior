# Hill-Climbing structure search with the BDeu score, distilled

Hill-Climbing (HC) is a score-based Bayesian-network structure learning method for discrete
data. It scores a DAG by the marginal likelihood of the data under a Bayesian-Dirichlet prior
(the BDeu score), and searches the space of DAGs by greedy best-improvement local search: at
each step it applies the single arc addition, deletion, or reversal that most improves the
lower-is-better score, subject to the result staying acyclic, stopping at a local optimum.
Because the score is decomposable, each candidate move costs only one or two local recomputes;
because the score is score-equivalent, observational data identifies only the Markov
equivalence class, so the learned DAG is collapsed to a CPDAG before being returned.

## Problem it solves

Recover the structure of a discrete Bayesian network — or as much of it as observational data
can identify — from a database of complete, integer-coded cases, with a clean probabilistic
criterion computable from counts, that scales past the point where structures can be enumerated.

## Key idea

1. **Score = marginal likelihood (BDe / BDeu).** Treat the structure as uncertain and rank
   structures by `p(B_S, D) = p(B_S) p(D | B_S)`. Integrate the table parameters out against a
   Dirichlet prior (conjugate to the multinomial), under multinomial sampling, parameter
   independence, complete data, parameter modularity, and Dirichlet priors. The marginal
   likelihood is closed-form:

   ```
   p(D, B_S) = p(B_S) · ∏_i ∏_{j=1}^{q_i} {
       [ Γ(α_ij) / Γ(α_ij + N_ij) ]
       · ∏_{k=1}^{r_i} [ Γ(α_ijk + N_ijk) / Γ(α_ijk) ]
   }
   ```

   `N_ijk` = #cases with `x_i = k` and parent-config `Π_i = j`; `N_ij = Σ_k N_ijk`;
   `r_i` = #states of `x_i`; `q_i = ∏_{p∈Π_i} r_p` = #parent configs; `α` = Dirichlet
   pseudocounts. Integrating over parameters builds in an automatic complexity penalty.

2. **BDeu pseudocounts make the score equivalent.** Score equivalence (isomorphic /
   Markov-equivalent structures get equal scores) holds when the pseudocounts come from an
   equivalent-sample-size construction `α_ijk = N' · p(x_i=k, Π_i=j | prior)`. Taking the prior
   joint uniform gives `α_ijk = N' / (r_i q_i)` and `α_ij = N' / q_i`, needing only the scalar
   equivalent sample size `N'` (the implementation defaults to `N' = 1`). This is BDeu:
   Bayesian-Dirichlet equivalent uniform. (Unit pseudocounts `α_ijk = 1` give the K2 score,
   which is **not** score-equivalent.)

3. **Decomposability.** `log p(D, B_S) = Σ_i s(x_i | Π_i)`, a sum of per-node family scores.
   Adding/deleting an arc `i→j` changes only `Π_j` (one local recompute); reversing `i→j` to
   `j→i` changes `Π_j` and `Π_i` (two). Cache local scores keyed by (node, sorted parents).

4. **Greedy hill-climbing over DAGs.** Start from the empty graph. Each sweep, evaluate the
   score delta of every eligible add / delete / reverse that keeps the graph acyclic, apply the
   single best improving move, and repeat until no move improves. Add `i→j` is legal iff `i` is
   not reachable from `j`; reverse `i→j` is legal iff, with `i→j` removed, `j` is not reachable
   from `i`. No variable ordering is required (unlike K2), which is why the full DAG space is
   searched with explicit acyclicity checks.

5. **Return the CPDAG.** The DAG's orientation on covered edges is arbitrary within its
   equivalence class, so collapse to the CPDAG (`dag2cpdag`) to report only identifiable
   orientations.

## Local score (log domain)

The per-node BDeu family score actually used also includes the decomposable structure-prior
term used by causallearn. The mathematical sum is over all parent configurations `j`; unobserved
configurations contribute zero in log space, so the implementation only loops over observed
ones:

```
s(x_i | Π_i) =
    |Π_i| log(ρ / (n - 1)) + (n - 1 - |Π_i|) log(1 - ρ / (n - 1))
    + Σ_j [ lgamma(α_ij) − lgamma(N_ij + α_ij)
            + Σ_k ( lgamma(N_ijk + α_ijk) − lgamma(α_ijk) ) ],
   α_ij = N' / q_i,   α_ijk = N' / (r_i q_i),   N' = 1,   ρ = 1.
```

The `causallearn` implementation returns the **negative** of this, so lower is better and the climb
seeks the most-negative delta. `N'` is a gain/smoothing control: smaller `N'` lets data dominate
the prior joint faster.

## Complexity and the local-optimum caveat

Each sweep considers `O(n^2)` moves; each move is one or two cached local recomputes; a local
recompute is one pass of counts over the sample. HC is greedy, so it can get stuck in a local
optimum — no theoretical global guarantee. Standard robustness extensions (not in the core
procedure) are **tabu search** (continue with the best non-worsening move, forbidding recently
visited graphs) and **random restarts** (perturb and re-climb, keep the best), both bolting
onto the same delta machinery.

## Relation to prior methods

- **K2** (Cooper & Herskovits): greedy parent addition exploiting decomposability, but
  requires a correct variable ordering and uses unit pseudocounts (not score-equivalent). HC
  removes the ordering requirement by searching the full DAG space, and uses BDeu.
- **Exact one-parent search**: maximum-branching / Chow–Liu tree finds the optimum in
  polynomial time only under the at-most-one-parent restriction; NP-hard once nodes may have
  several parents — hence the heuristic climb.

## Working code

Fills the `run_causal_discovery` slot of the score-based structure-search harness, using the
decomposable cached BDeu local score, an adjacency matrix with a reachability-based acyclicity
guard, and a final DAG→CPDAG collapse.

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
    # Decomposable BDeu local score (lower is better). parameters=None ->
    # observed cardinalities auto-detected; N' = 1 and structure_prior = 1.
    score_func = LocalScoreClass(
        data=X, local_score_fun=local_score_BDeu, parameters=None
    )

    nodes = [GraphNode(f"X{i + 1}") for i in range(N)]
    adj = np.zeros((N, N), dtype=int)            # adj[i, j] == 1  means i -> j

    # Per-node local scores; total = sum (decomposability). Start: no parents.
    local_scores = np.zeros(N)
    for j in range(N):
        local_scores[j] = score_func.score(j, [])

    def _has_path(src, tgt):
        """DFS: directed path src -> ... -> tgt in adj? (acyclicity guard)."""
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
                    # ADD i -> j  (legal iff i not already reachable from j)
                    if not _has_path(j, i):
                        pj_new = sorted(np.where(adj[:, j] == 1)[0].tolist() + [i])
                        delta = score_func.score(j, pj_new) - local_scores[j]
                        if delta < best_delta - 1e-6:
                            best_delta, best_op = delta, ("add", i, j)

                elif adj[i, j] == 1:
                    # DELETE i -> j  (only Pi_j changes)
                    pj_new = [p for p in np.where(adj[:, j] == 1)[0] if p != i]
                    delta = score_func.score(j, sorted(pj_new)) - local_scores[j]
                    if delta < best_delta - 1e-6:
                        best_delta, best_op = delta, ("delete", i, j)

                    # REVERSE i -> j  to  j -> i  (Pi_j and Pi_i both change)
                    adj[i, j] = 0                      # illegal iff i can still reach j
                    if not _has_path(i, j):
                        pj_del = sorted(np.where(adj[:, j] == 1)[0].tolist())
                        pi_new = sorted(np.where(adj[:, i] == 1)[0].tolist() + [j])
                        delta = ((score_func.score(j, pj_del) - local_scores[j])
                                 + (score_func.score(i, pi_new) - local_scores[i]))
                        if delta < best_delta - 1e-6:
                            best_delta, best_op = delta, ("reverse", i, j)
                    adj[i, j] = 1                      # restore

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
                j, sorted(np.where(adj[:, j] == 1)[0].tolist()))
            if op == "reverse":
                local_scores[i] = score_func.score(
                    i, sorted(np.where(adj[:, i] == 1)[0].tolist()))
            improved = True

    # Observational data identifies only the equivalence class -> CPDAG.
    G = GeneralGraph(nodes)
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                G.add_directed_edge(nodes[i], nodes[j])
    return dag2cpdag(G)
```
