Let me start from what I have and what hurts. I have Greedy Equivalence Search, and it is a beautiful thing: it searches the space of Markov equivalence classes — represented by completed PDAGs — by greedily applying the highest-scoring `Insert` until no insertion improves the score, then the highest-scoring `Delete` until no deletion improves it, and with a decomposable, consistent, score-equivalent score this two-phase climb is *guaranteed*, in the large-sample limit, to land on the true MEC. So I am not searching for a guarantee; I already have one. What hurts is finite samples and scale. With limited data GES routinely fails to recover the true MEC, and the failure has a known signature: GES's output contains *adjacencies between variables that are non-adjacent in the truth*. And it is slow — the problem is NP-hard and GES grinds on high-dimensional graphs. I want to fix both without leaving the family that gives me the guarantee, because the guarantee is the whole reason to be here.

Where do those spurious adjacencies come from? They are introduced by `Insert` operators — the forward phase is the only thing that adds edges. So the lever, if there is one, is the choice of *which* insertion to apply. Let me look hard at the basic assumption GES makes there, because that is the thing I want to question. At each forward state GES enumerates the neighboring classes reachable by one `Insert(X, Y, T)` and applies the *highest-scoring* one. The implicit claim is that the highest-scoring insertion is the best move to take. Is that necessary?

Here is the observation that cracks it open. Greedy hill-climbing finds the global optimum of a function not because it takes the *steepest* step, but because it stops only at a point where *no* improving step exists — and any path of improving steps reaches such a point. So if I relax GES to apply *any* valid score-increasing operator, not just the highest-scoring one, the search still terminates at a class where no score-increasing operator exists, which in the sample limit (by GES's own optimality argument) is the global optimum of the score: the true MEC. Let me make sure I believe this and not just assert it. The forward phase's job is to reach a class with respect to which `P` is Markov; the argument for that — if some member DAG still asserted a false independence, then by the composition axiom there is a *single* variable whose addition raises the score (local consistency), so a score-increasing insertion exists and the search cannot have stopped — never used *which* insertion was applied, only that an improving one exists whenever the current class is not yet Markov. Likewise the backward phase reaches the perfect map by an argument about the existence of an improving deletion, not about its being the best. So generalized GES — initialize anywhere, apply operators in any order, apply *any* score-increasing operator — still recovers the true MEC. Good. That is the freedom I needed: I may choose *which* score-increasing insertion to take strategically, and in particular I may *decline* insertions GES would have greedily made.

Now, which insertions should I decline? The thing GES does wrong is insert an adjacency between a pair `(X, Y)` that is non-adjacent in the true MEC. I want the score itself to tell me when an adjacency is spurious, before I commit it. Recall the operator: `Insert(X, Y, T)` adds `X → Y` and orients a chosen set `T` of `Y`'s undirected neighbors into `Y`; for a fixed pair `(X, Y)` there are many such operators, one per valid `T`, and each corresponds to a witness DAG where `Y`'s parent set is `NA_{Y,X} ∪ T ∪ Pa_Y`. The score change of `Insert(X, Y, T)` is the single node-family difference `s(Y, NA_{Y,X} ∪ T ∪ Pa_Y ∪ {X}) − s(Y, NA_{Y,X} ∪ T ∪ Pa_Y)` — by decomposability only `Y`'s family changes, and by score equivalence the class score change equals this witness-DAG change. Now invoke local consistency on that difference: it is positive iff `X` and `Y` are *dependent* given `NA_{Y,X} ∪ T ∪ Pa_Y`, and negative iff they are *conditionally independent* given that set. So a score-*decreasing* `Insert(X, Y, T)` is the score telling me, through one operator, that `X ⊥ Y` given a particular conditioning set — a conditional independence, expressed as a model-score comparison rather than a thresholded CI test. And in the sample limit, if `X` and `Y` are non-adjacent in the true MEC, then *some* valid `T` exposes exactly such a separating set, so *some* `Insert(X, Y, T)` is score-decreasing. The contrapositive is my signal: a score-decreasing insertion for the pair is evidence the pair is non-adjacent in the truth.

That hands me the less-greedy rule almost directly. At a forward state, for each non-adjacent pair `(X, Y)`, iterate over the valid `Insert(X, Y, T)`. *If any one of them is score-decreasing*, treat that as evidence `X` and `Y` are conditionally independent: discard *all* `Insert(X, Y, *)` for that pair, mark the pair separated, move on. Only for pairs where no tried insertion is score-decreasing do I keep the score-increasing insertions as candidates; among all retained candidates across all pairs I apply the single best (still a valid score-increasing move, so the generalized-GES guarantee is intact). Call this CONSERVATIVEINSERT. Why is this the right shape? Because the failure I am fixing is precisely GES inserting an adjacency for a pair that *has* a separating set — and the example that makes GES fail is exactly this: GES finds a score-decreasing `Insert(X, Y, T)` for some `T` but then applies a *different* score-increasing `Insert(X, Y, T')` and commits the spurious adjacency anyway. CONSERVATIVEINSERT refuses to do that: the moment any `T` says "independent," the pair is off the table. Two payoffs fall out and they match the two things that hurt. Accuracy: I never insert the excess adjacencies the backward phase might fail to remove. Efficiency: finding one score-decreasing insertion lets me stop enumerating `T` subsets for that pair and shrinks the candidate set in every later state.

I have to be honest about the guarantee, because a clever rule that quietly breaks correctness is worse than plain GES. Is CONSERVATIVEINSERT guaranteed to find a score-increasing insertion whenever one is needed? The soundness would rest on a strong premise: that whenever `P` is not Markov with respect to the current class, there exist variables `X, Y` such that `X ⊥̸ Y | Pa_Y` for *every* member DAG — i.e. the pair whose insertion is needed can be identified *without* depending on which member DAG I picked. That premise is the hard part and it is only partially established; so CONSERVATIVEINSERT's soundness is left open. I do not want to ship an algorithm whose correctness is open, so I need a relaxation that is provably safe and that I can fall back to.

The safe relaxation weakens the declining condition to something I *can* prove. Instead of declining a pair whenever *any* `Insert(X, Y, T)` is score-decreasing, decline it only on the basis of the simplest insertion: pick an arbitrary DAG `G` in the current class, and for each non-adjacent pair `(X, Y)` check whether `G ∪ {X → Y}` has a *lower* score than `G` — equivalently whether the `T = ∅` insertion into the witness DAG is score-decreasing. If so, discard all `Insert(X, Y, *)`; otherwise keep the score-increasing ones. Call this SAFEINSERT. The premise it needs is the weaker, provable one: if `P` is not Markov with respect to the class, then for *every* DAG `G ∈ E` there exist `X, Y` with `X ⊥̸ Y | Pa_Y^G` — and unlike the conservative premise, here the choice of `X, Y` *may* depend on `G`, which is exactly what makes it provable. With that, SAFEINSERT returns a valid score-increasing insertion if and only if one exists, so generalized GES with SAFEINSERT recovers the true MEC in the sample limit, from any initial class. So I have a conservative strategy (empirically the most accurate, soundness open) and a safe strategy (provably correct), and the natural design ships both: run conservative for accuracy with the safe `T = ∅` screen as the soundness backstop, and the user can fall back to safe-only if they need the proof.

That settles the forward phase. The backward phase I leave *exactly* as GES's: greedily apply the highest-scoring `Delete(X, Y, H)` until none improves. I do not touch it, both because it is not the source of the over-adding and because keeping it verbatim means the correctness argument for deletions carries over unchanged. So the whole algorithm — Less Greedy Equivalence Search — is GES with one surgical change: the forward insertion *selection*, conservative or safe. (There are further optional heuristics from the XGES line — prioritizing insertions before deletions, forcing deletions with restarts — that bolt on as variants LGES-0/LGES/LGES+, but the core, and the part that buys the accuracy and the guarantee, is the insertion strategy.)

Now the discrete instantiation, since that is the case at hand — integer-coded multinomial data. The score is the same BDeu I would use for GES: the uniform-Dirichlet multinomial marginal likelihood, decomposable and score-equivalent (the unique BD score with score equivalence), with equivalent sample size `α = 1` and structure prior `1`. Nothing about the score changes — that is the point; the only thing that changes from GES is the insertion policy, so any improvement is attributable to the search strategy, not the criterion.

Let me write it as the search I would actually run, built on the GES operator primitives so the operator validity tests, local score differences, and CPDAG conversions are the canonical routines and only the forward selection differs. The forward loop: at each state, precompute neighbors/adjacents/parents/semi-paths; for each non-adjacent ordered pair `(i, j)`, enumerate valid `T` subsets (clique test on `NA ∪ T`, semi-directed-path test) exactly as GES does, scoring each by the insert-difference helper; but track per pair whether *any* valid insertion is score-decreasing, and the moment one is, mark the pair separated and discard its candidates. Among retained strictly-score-increasing candidates from un-separated pairs, apply the global best; reconvert the CPDAG only on a move. The backward loop is GES's `Delete` loop verbatim. Finish by returning the CPDAG.

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
    """LGES-0 (Less Greedy Equivalence Search) with the BDeu score: GES with a
    conservative forward insertion strategy (decline a pair once the score implies
    a conditional independence) and the unchanged GES backward phase. Returns the
    estimated CPDAG as a GeneralGraph."""
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

Let me retrace the chain. GES is already large-sample optimal, so the problem is finite-sample accuracy and scale, and the documented finite-sample defect is spurious adjacencies, all introduced by `Insert`. The basic assumption I questioned is GES's greedy "apply the highest-scoring insertion," and the crack is that correctness needs only "apply *some* score-increasing operator" (generalized GES), so I am free to decline insertions. The score itself flags a spurious adjacency: by local consistency a score-decreasing `Insert(X, Y, T)` means `X ⊥ Y` given a conditioning set, evidence the pair is non-adjacent — and GES fails precisely by finding such a `T` yet applying a different score-increasing `Insert(X, Y, T')` anyway. CONSERVATIVEINSERT closes that hole: decline the whole pair the moment any insertion is score-decreasing, keeping only the score-increasing inserts of un-separated pairs and applying the best — buying accuracy (no excess adjacencies) and efficiency (early stop, smaller candidate set). Its soundness rests on a premise that is only partially established, so SAFEINSERT relaxes the declining condition to the provable `T = ∅` screen, which returns a score-increasing insertion iff one exists and so keeps the sample-limit MEC-recovery guarantee. The backward (delete) phase is GES's, untouched, so the deletion correctness carries over. The discrete score is the same BDeu as GES, so the only change from the strongest prior method is the forward insertion policy — a controlled, single-knob less-greedy modification of GES that targets exactly its over-adding failure mode while preserving its guarantee.
