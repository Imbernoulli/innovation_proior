# Less Greedy Equivalence Search (LGES), distilled

LGES (Ejaz & Bareinboim, NeurIPS 2025) is a variant of Greedy Equivalence Search (GES) that
retains GES's large-sample guarantee — recovering the true Markov equivalence class (CPDAG) in the
sample limit — while improving finite-sample accuracy and runtime. It changes one thing: the
**forward (insertion) phase** is made *less greedy*. Rather than always applying the highest-scoring
`Insert`, LGES declines to insert an edge between any variable pair for which the score already
implies a conditional independence. The score, the `Insert`/`Delete` operators, the
equivalence-class (completed-PDAG) search, and the entire **backward (deletion) phase** are GES's,
unchanged.

## Problem it solves

GES is provably correct in the large-sample limit but, on finite data, (1) struggles to scale
(NP-hard, costly in high dimensions) and (2) often fails to recover the true MEC — its output
contains adjacencies between variables that are non-adjacent in the truth. Those spurious
adjacencies are all introduced by `Insert` operators, so the target is the *choice of which
`Insert` to apply*.

## Key idea

1. **Greedy is not necessary for correctness.** Generalized GES (initialize from any class, apply
   operators in any order, apply *any* score-increasing operator — not just the highest-scoring
   one) still finds the score's global optimum in the sample limit. This frees the search to choose
   *which* score-increasing insertion to take, and to *decline* insertions GES would have made.
2. **A score-decreasing insertion flags a non-adjacency.** For a non-adjacent pair `(X, Y)`,
   `Insert(X, Y, T)` has score change `s(Y, NA∪T∪Pa_Y∪{X}) − s(Y, NA∪T∪Pa_Y)` (one node-family
   difference, by decomposability + score equivalence). By **local consistency** this is negative
   iff `X ⊥ Y` given that conditioning set — a conditional independence read off the score, not a
   thresholded CI test. So a score-decreasing `Insert(X, Y, T)` for some `T` is evidence the pair is
   non-adjacent in the true MEC. GES fails precisely by finding such a `T` yet applying a different
   score-increasing `Insert(X, Y, T')` and committing the spurious adjacency anyway.
3. **CONSERVATIVEINSERT.** At each forward state, for each non-adjacent pair `(X, Y)`, enumerate the
   valid `Insert(X, Y, T)`; **if any is score-decreasing, discard all `Insert(X, Y, *)` for that
   pair** (mark it separated) and move on. Only un-separated pairs keep their score-increasing
   inserts as candidates; apply the global best. Payoffs: *accuracy* (no excess adjacencies the
   backward phase may fail to remove) and *efficiency* (finding one score-decreasing insert stops
   the `T`-enumeration for that pair and shrinks the candidate set).
4. **SAFEINSERT (the provable backstop).** CONSERVATIVEINSERT's soundness rests on a premise that is
   only partially established. SAFEINSERT relaxes the declining condition: decline a pair only when
   its simplest insertion (`T = ∅`, i.e. `G ∪ {X → Y}` lower-scoring than `G` for a witness DAG `G`)
   is score-decreasing. This **provably returns a score-increasing insertion iff one exists**, so
   generalized GES with SAFEINSERT recovers the true MEC in the sample limit, from any initial class
   (Correctness of LGES, Cor. 1). Ship conservative for accuracy with the `T = ∅` safe screen as the
   soundness backstop.
5. **Backward phase = GES, verbatim.** Greedily apply the highest-scoring `Delete(X, Y, H)` until
   none improves. Untouched, so the deletion correctness carries over directly.

## The discrete family score: BDeu

Identical to GES: the uniform-Dirichlet multinomial marginal likelihood — decomposable and
score-equivalent (the unique BD score with score equivalence) — with equivalent sample size
`α = 1` and structure prior `1`, cardinalities inferred from the data. Nothing about the score
changes; the only change from GES is the insertion *selection*, so any improvement is attributable
to the search strategy.

## Final algorithm

```
state E <- empty-graph class (completed PDAG)
# Forward (CONSERVATIVE insertion)
repeat:
    for each non-adjacent pair (X, Y):
        for each valid Insert(X, Y, T):
            if its score change <= 0:        # score implies X _||_ Y
                separate (X, Y); discard all its inserts; break
            else: keep it as a candidate
    if best retained candidate has gain > 0: apply it, reconvert E to completed PDAG
    else: break
# Backward (GES delete, unchanged)
repeat:
    over all adjacent (X, Y) and valid Delete(X, Y, H): score by the delete difference
    if best gain > 0: apply it, reconvert E to completed PDAG
    else: break
return E   # estimated equivalence class (CPDAG)
```

## Working code

Built on the `causal-learn` GES operator primitives so the validity tests, local score
differences, operator applications, and PDAG↔completed-PDAG conversions are the canonical
routines; only the forward *selection* differs from GES (conservative declining + the `T = ∅` safe
screen subsumed by `gain <= 0`), and the backward phase is GES's verbatim.

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
    """LGES-0 with the BDeu score: GES with a conservative forward insertion
    strategy (decline a pair once the score implies a conditional independence)
    and the unchanged GES backward phase. Returns the estimated CPDAG."""
    N = X.shape[1]
    maxP = N

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
                    pair_candidates = []                   # score-increasing inserts for this pair
                    separated = False                      # any Insert(i,j,T) score-decreasing?
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
                                    separated = True       # score implies X _||_ Y: drop the pair
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

## Relation to prior methods

- **GES (Chickering 2002)**: identical score, operators, backward phase, and large-sample
  guarantee; LGES changes only the forward insertion *selection* — declining a pair once the score
  implies an independence — to remove GES's finite-sample spurious adjacencies and cut runtime.
- **PC (Spirtes et al. 2000)**: also uses the "score-decreasing insertion ⇒ conditional
  independence" signal, but as a *model-score comparison* inside a score-based search rather than a
  thresholded CI test, avoiding PC's hard-verdict over-deletion.
- **XGES (Nazaret & Blei 2024)**: complementary scheduling heuristics (prioritize insertions before
  deletions; force deletions with restarts) that LGES reuses as optional variants (LGES, LGES+); the
  core contribution here is the insertion strategy, not the scheduling.
