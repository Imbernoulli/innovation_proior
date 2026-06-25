**Problem.** GES is the strongest baseline (Child SHD 2.3, Alarm 2.7) but its one residual weakness is a
maximally-**greedy forward phase** that over-adds adjacencies on dense, near-unfaithful networks —
Win95pts SHD 48.7 (worse than BOSS's 38, adjacency precision only 0.80) and Hailfinder 39.7 (arrow
precision 0.56). The backward phase cannot fully remove the excess edges the forward phase inserts.

**Key idea (LGES — Less Greedy Equivalence Search).**
A search that applies *any* score-increasing neighbor — not only the highest-scoring one — still finds
the score's global optimum in the sample limit. That freedom lets the forward phase be **less greedy**:
for each non-adjacent pair `(X, Y)`, enumerate the valid `Insert(X, Y, T)` operators, and **if any one is
score-decreasing, discard all inserts for that pair** (the score has signaled a conditional independence
via a model-score comparison, evidence the pair is non-adjacent in the truth — exactly the over-adding
GES does). Only un-separated pairs keep their score-increasing candidates; apply the global best.
Everything else is GES: identical BDeu score, `Insert`/`Delete` operators, completed-PDAG search, and an
**untouched backward (delete) phase**, so the large-sample MEC-recovery guarantee is preserved. A cheap
"safe" pre-screen (decline a pair when its simplest `T = ∅` insert is non-positive) is provably sound and
backstops the conservative rule.

**Why stronger than GES.** It refuses to insert edges between pairs the score implies independent, so it
avoids the excess adjacencies the GES backward phase may fail to remove — targeting Win95pts/Hailfinder
over-adding at its source — while changing nothing that made GES the strongest baseline. The only delta
from GES is the forward insertion policy: a controlled single-variable ablation of the greedy step.

**Hyperparameters.** Identical BDeu (`sample_prior = 1`, `structure_prior = 1`, cardinalities inferred);
conservative insertion with the `T = ∅` safe screen; GES backward phase verbatim; empty-graph start.

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """LGES-0 (Less Greedy Equivalence Search, conservative insertion).

    Same BDeu score and Insert/Delete operators as GES, with one change to the
    forward phase: a pair (X, Y) whose score implies a conditional independence
    (some valid Insert(X, Y, T) is score-decreasing) is "separated" and gets no
    inserts. The backward (delete) phase is GES's, unchanged. Returns the CPDAG.
    """
    import numpy as np
    from causallearn.utils.GESUtils import (
        precompute_graph_info, Combinatorial, find_subset_include,
        check_clique_fast, insert_vc2_fast,
        insert_changed_score_fast, delete_changed_score_fast,
        insert, delete, score_g,
    )
    from causallearn.utils.PDAG2DAG import pdag2dag       # Dor-Tarsi consistent extension
    from causallearn.utils.DAG2CPDAG import dag2cpdag     # compelled/reversible labeling
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BDeu

    N = X.shape[1]
    maxP = N

    # Decomposable, score-equivalent BDeu family score (sample_prior=1, structure_prior=1)
    score_func = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    parameters = None

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(N)]
    G = GeneralGraph(nodes)                       # empty graph (all independencies)
    score = score_g(X, G, score_func, parameters)
    G = dag2cpdag(pdag2dag(G))                     # completed PDAG of the current class
    cache = {}                                     # (node, sorted parents) -> family score

    # ---------------- Forward: CONSERVATIVE Insert(i, j, T) ----------------
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
                    subsets = Combinatorial(sorted(nbrs[j] - adj[i]))   # tails not adjacent to i
                    flag = np.zeros(len(subsets))          # prune supersets of a non-clique
                    pair_candidates = []                   # score-increasing inserts for THIS pair
                    separated = False                      # some Insert(i,j,T) score-decreasing?
                    for k in range(len(subsets)):
                        if separated:
                            break                          # pair separated: drop all its inserts
                        if flag[k] >= 2:
                            continue
                        T = set(subsets[k])
                        if check_clique_fast(G, NA | T):               # cond 1: NA u T clique
                            if flag[k] == 0:
                                valid_path = insert_vc2_fast(j, i, NA | T, semi)  # cond 2
                            else:
                                valid_path = 1
                            if valid_path:
                                flag[np.where(find_subset_include(subsets[k], subsets) == 1)] = 1
                                gain, desc, cache = insert_changed_score_fast(
                                    X, i, j, subsets[k], NA, pa[j], cache, score_func, parameters)
                                if gain <= 0:
                                    # score implies a conditional independence:
                                    # separate the pair, discard ALL its inserts
                                    separated = True
                                    pair_candidates = []
                                else:
                                    pair_candidates.append((gain, desc))
                        else:
                            flag[np.where(find_subset_include(subsets[k], subsets) == 1)] = 2
                    if not separated:
                        for gain, desc in pair_candidates:
                            if gain > best_gain:
                                best_gain, best = gain, desc
        if best is None or best_gain <= 0:
            break
        G = insert(G, best[0], best[1], best[2])           # add i->j, orient each T as T->j
        G = dag2cpdag(pdag2dag(G))                          # reconvert (only on a move)
        score += best_gain

    # ---------------- Backward: GES Delete(i, j, H) (unchanged) ----------------
    while True:
        best_gain, best = -np.inf, None
        nbrs, adj, pa, semi = precompute_graph_info(G, N)
        for i in range(N):
            for j in range(N):
                if (j in nbrs[i]) or (i in pa[j]):         # i - j  or  i -> j
                    NA = nbrs[j] & adj[i]                   # NA_{Y,X}
                    subsets = Combinatorial(sorted(NA))     # heads H subset of NA
                    ok = np.ones(len(subsets))              # prune supersets of a clique-pass
                    for k in range(len(subsets)):
                        H = set(subsets[k])
                        if ok[k] == 1:
                            if check_clique_fast(G, NA - H):              # validity: NA \ H clique
                                ok[np.where(find_subset_include(subsets[k], subsets) == 1)] = 2
                            else:
                                continue
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
