# Greedy Equivalence Search (GES), distilled

GES is a two-phase, score-based, greedy search for the **equivalence class** of the DAG that
generated a set of iid observations. It searches directly over equivalence classes (represented
by completed PDAGs / CPDAGs), not over DAGs: a *forward* phase greedily **inserts** edges until
no insertion improves the score, then a *backward* phase greedily **deletes** edges until no
deletion improves it. With a decomposable, score-equivalent, locally consistent score, GES is
**large-sample optimal** — in the limit it returns the equivalence class of the true structure —
and every candidate move is tested and scored by purely local functions of a node and its
neighbors, so no member DAG of a class is ever enumerated.

## Problem it solves

Recover causal/probabilistic structure from purely observational discrete data. Observationally
only the **Markov equivalence class** (skeleton + v-structures, Verma–Pearl) is identifiable, so
the output is a CPDAG, not a single DAG. The structure space is super-exponential and finding the
optimal structure under a Bayesian score is NP-hard, so the search must be local and greedy — yet
one wants a global guarantee of correctness in the large-sample limit.

## Key idea

1. **Search equivalence classes, not DAGs.** Represent each class by its completed PDAG (directed
   edge for every *compelled* edge, undirected for every *reversible* edge — unique per class).
   DAG-space hill-climbing wastes moves on covered-edge reversals, which change the DAG but not
   the class; class-space search removes that redundancy.
2. **Two phases.** Start from the empty graph. **Forward**: neighbors are all classes reachable by
   adding one edge to *some* member DAG; greedily move to the best while the score improves.
   **Backward**: neighbors are all classes reachable by deleting one edge from some member DAG;
   greedily move to the best while the score improves. Terminate at the backward local maximum.
3. **Large-sample optimality** rests on score properties:
   - *Consistency*: in the limit the score prefers a structure that contains the true `p` over one
     that doesn't, and among containers prefers fewer parameters.
   - *Local consistency* (a consequence, via decomposability): adding `X_i -> X_j` raises the
     score iff `X_j` is dependent on `X_i` given its current parents.
   - The **forward** phase reaches a class containing `p` (local consistency + the composition
     axiom of DAG-perfect distributions). The **backward** phase peels off every unnecessary edge
     down to the perfect map; this needs **Meek's conjecture**: if `G <= H` (`H` an independence
     map of `G`), then `G` transforms into `H` by a sequence of covered-edge reversals and single
     edge additions (`<= r + 2m` moves). The constructive step removes common sinks, works from a
     sink `Y` in `H`, finds the unique maximal element `D` of `De_G(Y)` under ancestry in `H`,
     and chooses a maximal child `Z` of `Y` in `G` that has `D` as a descendant. This choice
     matters because, under `G <= H`, that unique maximal element is an ancestor in `H` of every
     node in `De_G(Y)`; it is what keeps each addition/reversal inside the `G <= H` invariant.
     Proving this transformation is what makes GES's optimality unconditional.

## Operators (scored and validity-tested locally on the completed PDAG)

Let `NA_{Y,X}` = neighbors of `Y` (undirected-adjacent) that are also adjacent to `X`; `Pa_Y` =
parents of `Y`; a *semi-directed path* `Y -> X` has each edge undirected or directed away from `Y`.

- **Insert(X, Y, T)** — `X`, `Y` non-adjacent; `T` ⊆ neighbors of `Y` not adjacent to `X`. Add
  `X -> Y`; orient each `T in T` as `T -> Y`.
  - *Valid* iff (1) `NA_{Y,X} ∪ T` is a clique **and** (2) every semi-directed path from `Y` to
    `X` contains a node in `NA_{Y,X} ∪ T`.
  - *Score increase* = `s(Y, NA_{Y,X} ∪ T ∪ Pa_Y ∪ {X}) − s(Y, NA_{Y,X} ∪ T ∪ Pa_Y)`.
- **Delete(X, Y, H)** — `X`, `Y` adjacent (`X−Y` or `X→Y`); `H` ⊆ neighbors of `Y` adjacent to
  `X`. Delete the `X`–`Y` edge; orient each `H in H` as `Y -> H` and `X -> H`.
  - *Valid* iff `NA_{Y,X} \ H` is a clique.
  - *Score increase*: let `P_old = (NA_{Y,X} \ H) ∪ Pa_Y ∪ {X}` in the witness DAG; then the
    increase is `s(Y, P_old \ {X}) − s(Y, P_old)`.

Both score formulas are differences of a single node's **family score** — valid because the score
is decomposable (only `Y`'s family changes) and score-equivalent (the class score change equals
that of any witness member DAG). After an accepted move the PDAG is reconverted to its completed
form (extract a consistent extension via Dor–Tarsi `PDAG-to-DAG`, then re-label compelled vs.
reversible edges via `DAG-to-CPDAG`); this conversion runs **only on a move**, never when scoring
candidates. Pruning: an insert whose `NA_{Y,X} ∪ T` is not a clique fails for every superset of
`T`; a delete valid for `H` is valid for every superset; family scores are cached by
`(node, sorted parents)`.

## The discrete family score: BDeu

For discrete data the family score `s(X_i, Pa_i)` is the Bayesian multinomial **marginal
likelihood** with a uniform Dirichlet prior — decomposable and score-equivalent. With `r_i` states
of `X_i`, `q_i = prod_{p in Pa_i} r_p` parent configurations, counts `N_ijk` (records with
`X_i = k`, parents in config `j`) and `N_ij = sum_k N_ijk`, equivalent sample size `N'`:

```
s_BDeu(X_i, Pa_i) = sum_{j=1}^{q_i} [ log Γ(N'/q_i) − log Γ(N'/q_i + N_ij)
                        + sum_{k=1}^{r_i} ( log Γ(N'/(r_i q_i) + N_ijk) − log Γ(N'/(r_i q_i)) ) ]
                    + structure-prior term.
```

BDeu ("u" for uniform) is the special case of the BDe metric where each joint cell is a priori
equally likely (`N'_ijk = N'/(r_i q_i)`, `N'_ij = N'/q_i`); demanding **likelihood equivalence**
together with parameter independence/modularity forces this Dirichlet form, which is exactly the
score-equivalence GES needs. The canonical BDeu call uses equivalent sample size
`sample_prior = 1` and `structure_prior = 1` by default, inferring the state counts `r_i` from
the data. Larger score = better penalized fit; GES compares differences.

## Final algorithm

```
state E <- empty-graph class (completed PDAG)
# Forward (FES)
repeat:
    over all non-adjacent (X, Y) and all valid Insert(X, Y, T): score by the insert difference
    if best gain > 0: apply it, reconvert E to completed PDAG
    else: break
# Backward (BES)
repeat:
    over all adjacent (X, Y) and all valid Delete(X, Y, H): score by the delete difference
    if best gain > 0: apply it, reconvert E to completed PDAG
    else: break
return E   # estimated equivalence class (CPDAG)
```

## Working code

Faithful to the `causal-learn` implementation. The validity tests, local score differences,
operator applications, and PDAG↔completed-PDAG conversions are the canonical routines.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GESUtils import (
    precompute_graph_info, Combinatorial, find_subset_include,
    check_clique_fast, insert_vc2_fast,
    insert_changed_score_fast, delete_changed_score_fast,
    insert, delete, score_g,
)
from causallearn.utils.PDAG2DAG import pdag2dag        # Dor-Tarsi consistent extension (PDAG -> DAG)
from causallearn.utils.DAG2CPDAG import dag2cpdag      # compelled/reversible labeling (DAG -> CPDAG)
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import local_score_BDeu


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """GES with the BDeu score. Two-phase greedy search over equivalence classes;
    returns the estimated CPDAG as a GeneralGraph."""
    N = X.shape[1]
    maxP = N

    # decomposable, score-equivalent BDeu family score s(X_i, Pa_i)
    score_func = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    parameters = None

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(N)]
    G = GeneralGraph(nodes)                       # empty graph (all independencies)
    score = score_g(X, G, score_func, parameters)
    G = dag2cpdag(pdag2dag(G))                     # completed PDAG of the current class
    cache = {}                                     # (node, sorted parents) -> family score

    # ---------------- Phase 1: forward, Insert(i, j, T) ----------------
    while True:
        best_gain, best = -np.inf, None
        nbrs, adj, pa, semi = precompute_graph_info(G, N)
        for i in range(N):
            for j in range(N):
                if (
                    G.graph[i, j] == 0
                    and G.graph[j, i] == 0
                    and i != j
                    and len(pa[j]) <= maxP
                ):                                       # i, j non-adjacent
                    NA = nbrs[j] & adj[i]                  # NA_{Y,X}
                    subsets = Combinatorial(sorted(nbrs[j] - adj[i]))      # tails not adjacent to i
                    flag = np.zeros(len(subsets))          # prune supersets of a non-clique
                    for k in range(len(subsets)):
                        if flag[k] >= 2:
                            continue
                        T = set(subsets[k])
                        if check_clique_fast(G, NA | T):                   # cond 1: NA ∪ T clique
                            if flag[k] == 0:
                                valid_path = insert_vc2_fast(j, i, NA | T, semi)  # cond 2: semi-path
                            else:
                                valid_path = 1
                            if valid_path:
                                flag[np.where(find_subset_include(subsets[k], subsets) == 1)] = 1
                                gain, desc, cache = insert_changed_score_fast(
                                    X, i, j, subsets[k], NA, pa[j], cache, score_func, parameters)
                                if gain > best_gain:
                                    best_gain, best = gain, desc
                        else:
                            flag[np.where(find_subset_include(subsets[k], subsets) == 1)] = 2
        if best is None or best_gain <= 0:
            break
        G = insert(G, best[0], best[1], best[2])           # add i->j, orient each T as T->j
        G = dag2cpdag(pdag2dag(G))                          # reconvert (only on a move)
        score += best_gain

    # ---------------- Phase 2: backward, Delete(i, j, H) ----------------
    while True:
        best_gain, best = -np.inf, None
        nbrs, adj, pa, semi = precompute_graph_info(G, N)
        for i in range(N):
            for j in range(N):
                if (j in nbrs[i]) or (i in pa[j]):         # i - j  or  i -> j
                    NA = nbrs[j] & adj[i]                   # NA_{Y,X}
                    subsets = Combinatorial(sorted(NA))     # heads H ⊆ NA
                    ok = np.ones(len(subsets))              # prune supersets of a clique-pass
                    for k in range(len(subsets)):
                        H = set(subsets[k])
                        if ok[k] == 1:
                            if check_clique_fast(G, NA - H):              # validity: NA \ H clique
                                ok[np.where(find_subset_include(subsets[k], subsets) == 1)] = 2
                            else:
                                continue
                        # The helper forms P_old=(NA\H)∪Pa∪{i}, then subtracts i:
                        # s(j, P_old\{i}) - s(j, P_old).
                        gain, desc, cache = delete_changed_score_fast(
                            X, i, j, subsets[k], NA, pa[j], cache, score_func, parameters)
                        if gain > best_gain:
                            best_gain, best = gain, desc
        if best is None or best_gain <= 0:
            break
        G = delete(G, best[0], best[1], best[2])            # drop i-j, orient each H as a new head
        G = dag2cpdag(pdag2dag(G))
        score += best_gain

    return G                                                # completed PDAG = estimated CPDAG
```

## Relation to prior methods

- **Constraint-based (PC)**: hard CI-test verdicts that cascade errors; GES optimizes one global,
  score-based objective and has a large-sample optimality guarantee instead.
- **DAG-space hill-climbing**: searches the larger, redundant DAG space (wasting covered-edge
  reversals) and carries no optimality guarantee; GES searches the smaller equivalence-class space.
- **Greedy search over equivalence classes with the earlier insert/delete/reverse operators**:
  efficient but its neighbor connectivity is *not* the "add/delete an edge in some member DAG"
  connectivity that the optimality proof requires; GES's `Insert(X,Y,T)` and `Delete(X,Y,H)`
  operators realize exactly that connectivity, which is why they come with the guarantee.
