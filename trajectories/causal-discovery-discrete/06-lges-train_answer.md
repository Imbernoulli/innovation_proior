GES landed exactly where its guarantee predicts, and the residual it leaves is precise enough to name. On the faithful, well-behaved networks it is the clean winner: Child SHD 2.3 (arrow precision a perfect 1.0, two of three seeds at SHD 1), Alarm SHD 2.7 (arrow precision 0.98, recall 0.94, seeds a tight 4/2/2), crushing every prior rung and erasing BOSS's Alarm seed-spread. The guaranteed forward-then-backward search did what a heuristic forward sweep could not — converge to essentially the right equivalence class with almost no per-seed variance. But the two extremes confirm the one weakness I flagged. Hailfinder improved to SHD 39.7 yet the adjacency numbers tell the story: precision 0.85 against recall 0.82, arrow precision a mediocre 0.56, so GES is keeping edges and orientations the truth does not have. And Win95pts, the densest by edge count, came in at SHD 48.7 — *worse* than BOSS's 38 — with adjacency precision 0.80 (the lowest of the strong rungs) against recall 0.83, the signature of a forward phase that inserts excess adjacencies the backward phase cannot fully remove. So GES's failure mode is not the score, not the backward phase, not the equivalence-class search — it is the *greedy forward phase* over-adding adjacencies on dense, near-unfaithful graphs. That is the one gap the whole ladder has circled and no rung has closed: hc over-added by stalling, GRaSP by wandering, PC over-*deleted* instead, BOSS over-added by seed-luck, and now GES over-adds by being maximally greedy in its insertions. The next move is forced and surgical: keep everything that made GES the strongest baseline — the BDeu score, the equivalence-class search, the `Insert`/`Delete` operators, the backward phase, the large-sample guarantee — and make only the *forward* phase less greedy.

I propose LGES — Less Greedy Equivalence Search. The opening is a fact GES does not exploit: at each forward state GES applies the *highest-scoring* `Insert`, implicitly claiming the highest-scoring insertion is the best to take, but a search that applies *any* score-increasing neighbor — not necessarily the highest — still finds the score's global optimum in the sample limit, because the optimum is a fixed point of "no score-increasing move exists" regardless of which improving moves got there. That freedom lets me choose *which* score-increasing insertion to take strategically, and in particular *decline* certain insertions GES would have made. Which ones? GES does wrong when it inserts an adjacency between two variables $X, Y$ that are non-adjacent in the truth. Recall `Insert(X, Y, T)` adds $X \to Y$ and orients a chosen set $T$ of $Y$'s neighbors into $Y$, and for a fixed pair GES considers many such operators, one per valid $T$. The key observation: in the sample limit, if $X$ and $Y$ are non-adjacent in the true MEC, then there *exists* some $T$ for which `Insert(X, Y, T)` is **score-decreasing** — because that $T$ exposes a conditioning set under which $X$ and $Y$ are conditionally independent, and by local consistency a score-decreasing insertion means exactly "this independence is real." Conversely, a score-decreasing `Insert(X, Y, T)` for some $T$ is evidence the pair is non-adjacent. The score, through a single operator, is telling me a conditional independence — not via a thresholded CI test, but as a model-score comparison, the robustness PC lacked.

That gives the less-greedy rule directly. At a forward state, for each non-adjacent pair $(X, Y)$, iterate over the valid `Insert(X, Y, T)` operators. **If *any* of them is score-decreasing**, treat it as evidence that $X$ and $Y$ are conditionally independent — discard *all* `Insert(X, Y, *)` operators for that pair, mark the pair separated, and move on. Only for pairs where *no* tried insertion is score-decreasing do I keep the score-increasing insertions as candidates; among all retained candidates across all pairs, I then apply the single highest-scoring one (still a valid score-increasing move, so the guarantee survives). This is *conservative* insertion: it refuses to insert an edge between any pair for which the score has already signaled an independence, precisely the over-adding GES does on dense graphs. It has two payoffs the GES failure mode names exactly — accuracy, because it avoids inserting the excess adjacencies the backward phase may fail to remove (the Win95pts/Hailfinder problem), and efficiency, because finding a score-decreasing insertion stops the enumeration of $T$ subsets for that pair early and shrinks the candidate set in every later state.

I have to be honest about the guarantee, because that is the whole reason to stay in the GES family. There are two readings of "score-decreasing insertion implies non-adjacency." The strong reading — whenever the current class is not Markov, *some* pair has a score-decreasing insertion whose decision is member-DAG-independent — would make the conservative rule provably complete, but that completeness is only partially established. The safe reading relaxes the rule just enough to be *guaranteed*: a "safe" variant first checks, for each non-adjacent pair, whether the simplest insertion ($T = \emptyset$, i.e. adding $X$ straight into $Y$'s parents in a witness DAG) is score-decreasing, and only declines the pair on that basis, which is provably sound — it returns a valid score-increasing insertion iff one exists, so the overall forward-then-backward search recovers the true MEC in the sample limit, for any initial class. Both forms are strictly within the GES guarantee structure because the backward phase is untouched and every applied move is a valid score-increasing operator. I run the conservative form — the one that most directly attacks the over-adding — with the safe $T = \emptyset$ screen as the soundness backstop.

The implementation builds the conservative forward phase on the same `GESUtils` operator surface GES uses, leaving the backward phase identical. The forward loop, at each state, computes the graph info, and for each non-adjacent ordered pair $(i, j)$ enumerates the valid $T$ subsets (clique test on $NA_{Y,X} \cup T$, semi-directed-path test) exactly as GES does, scoring each by the insert-difference helper — but tracks, per pair, whether *any* valid insertion came out non-positive, and the moment one does, marks the pair separated and skips its remaining and future insertions, retaining none of its candidates. Among the retained strictly score-increasing candidates from the un-separated pairs, it picks the global best, applies it, and reconverts the PDAG to its completed form only on a move, exactly as GES does. The backward phase is GES's `Delete` loop verbatim, and the score is the identical BDeu (`sample_prior = 1`, `structure_prior = 1`, cardinalities inferred from `parameters=None`), so once again the *only* thing that changed from the strongest baseline is the forward insertion policy — a controlled, single-variable ablation of GES's greedy step. The bar it must clear: hold the clean wins (Child 2.3, Alarm 2.7 should change little, because on faithful networks there are few spurious adjacencies to decline) and strictly *improve* the two over-add cases — lift Win95pts adjacency precision above GES's 0.80 by declining the excess edges to pull SHD toward and ideally past BOSS's 38, and pull Hailfinder SHD below 39.7 with adjacency precision above 0.85 as the spurious edges are refused — because the only mechanism changed is the refusal to insert edges between score-implied-independent pairs.

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
