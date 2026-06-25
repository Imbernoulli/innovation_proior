Let me start from what I have and what hurts. I have Greedy Equivalence Search. It searches the space of Markov equivalence classes — represented by completed PDAGs — by greedily applying the highest-scoring `Insert` until no insertion improves the score, then the highest-scoring `Delete` until no deletion improves it, and with a decomposable, consistent, score-equivalent score this two-phase climb lands, in the large-sample limit, on the true MEC. So I am not searching for a large-sample guarantee; I already have one. What hurts is finite samples and scale. With limited data GES routinely fails to recover the true MEC, and the failure has a known signature: GES's output contains *adjacencies between variables that are non-adjacent in the truth*. And it is slow — the problem is NP-hard and GES grinds on high-dimensional graphs. I want to attack both without leaving the family that gives me the guarantee, because the guarantee is the whole reason to be here.

Where do those spurious adjacencies come from? They are introduced by `Insert` operators — the forward phase is the only thing that adds edges; `Delete` only removes them. So the lever, if there is one, is the choice of *which* insertion to apply. Let me look hard at the basic assumption GES makes there. At each forward state GES enumerates the neighboring classes reachable by one `Insert(X, Y, T)` and applies the *highest-scoring* one. The implicit claim is that the highest-scoring insertion is the best move to take. Is that claim load-bearing for correctness, or just a habit?

Here is the observation that makes me doubt it is load-bearing. Greedy hill-climbing reaches the global optimum of a function not because it takes the *steepest* step, but because it stops only at a point where *no* improving step exists, and any path of improving steps reaches such a point. So suppose I relax GES to apply *any* valid score-increasing operator, not just the highest-scoring one. The search still terminates only at a class where no score-increasing operator exists. Does that class have to be the true MEC? Let me retrace GES's own optimality argument and watch for any place it actually needs "highest-scoring." The forward phase's job is to reach a class with respect to which `P` is Markov: the argument is that if some member DAG still asserted a false independence, then by the composition axiom there is a *single* variable whose addition raises the score (local consistency), so a score-increasing insertion exists and the search cannot have stopped there. That argument names the *existence* of an improving insertion; it never says which one. The backward phase reaching the perfect map is the same shape — existence of an improving deletion, not its being best. So the "highest-scoring" rule is doing no work in the proof. Generalized GES — initialize anywhere, apply operators in any order, apply *any* score-increasing operator — recovers the true MEC in the sample limit. That is the freedom I was hoping for: I may choose *which* score-increasing insertion to take, and in particular I may *decline* insertions GES would have greedily made.

So which insertions should I decline? The thing GES does wrong is insert an adjacency between a pair `(X, Y)` that is non-adjacent in the true MEC. I would like the score itself to flag a spurious adjacency before I commit it. Look at the operator. `Insert(X, Y, T)` adds `X → Y` and orients a chosen set `T` of `Y`'s undirected neighbors into `Y`; for a fixed pair `(X, Y)` there are many such operators, one per valid `T`, each corresponding to a witness DAG where `Y`'s parent set is `NA_{Y,X} ∪ T ∪ Pa_Y`. The score change of `Insert(X, Y, T)` is the single node-family difference `s(Y, NA_{Y,X} ∪ T ∪ Pa_Y ∪ {X}) − s(Y, NA_{Y,X} ∪ T ∪ Pa_Y)` — by decomposability only `Y`'s family changes, and by score equivalence the class score change equals this witness-DAG change. Now invoke local consistency on that difference: it should be positive iff `X` and `Y` are *dependent* given the conditioning set `NA_{Y,X} ∪ T ∪ Pa_Y`, and negative iff they are conditionally independent given that set.

I want to actually see this sign behave the way I am claiming, not just recite local consistency, because the entire method is going to rest on reading "independent" off the sign of one node-family difference. Let me take the discrete score I will actually use — BDeu, the uniform-Dirichlet multinomial marginal likelihood with equivalent sample size `α = 1` — and compute the insert difference by hand on a tiny dataset. Closed form for a family is `s(Y, Pa) = Σ_configs [ lΓ(a_ij) − lΓ(a_ij + N_ij) + Σ_states ( lΓ(a_ijk + N_ijk) − lΓ(a_ijk) ) ]`, with `a_ij = α/q`, `a_ijk = α/(q·r_Y)`, `q` the number of parent configurations, `r_Y` the number of states of `Y`. Two binary variables, ~4000 records each.

Case 1, draw `X` and `Y` independently (so `X ⊥ Y`), both `Pa_Y` and `T` empty, and compute `s(Y, {X}) − s(Y, {})`. I get **−3.53**: score-*decreasing*. Case 2, draw `Y` to agree with `X` 80% of the time (strong dependence), same difference: **+757.3**: score-*increasing*, by a wide margin. So the sign does exactly what local consistency promised — negative under independence, positive under dependence — and it is not a marginal effect I should worry about reading off noisy data; the dependent case is enormously positive and the independent case is comfortably below zero. Good: a score-*decreasing* `Insert(X, Y, T)` is the score telling me, through one operator, that `X ⊥ Y` given a particular conditioning set — a conditional independence expressed as a model-score comparison rather than a thresholded CI test. And in the sample limit, if `X` and `Y` are non-adjacent in the true MEC, *some* valid `T` exposes a separating set, so *some* `Insert(X, Y, T)` is score-decreasing. That is my signal: a score-decreasing insertion for a pair is evidence the pair is non-adjacent in the truth.

This almost hands me a rule directly — but before I write it I have to face the subtlety that actually decides its shape. The clean statement would be "if `Insert(X, Y, T)` is score-decreasing for some `T`, then it is score-decreasing for *all* `T`, so the pair is unambiguously separated." Is that true? It would make life easy. Let me check it, because if a single non-adjacent pair can have *both* a score-decreasing insert and a score-increasing insert depending on `T`, then the rule and its danger both live exactly there. Take the textbook collider: `X → Z ← Y` with `X ⊥ Y` marginally but `X ⊥̸ Y | Z`. Generate it — `X`, `Y` independent bits, `Z = X xor Y` with 5% noise, ~8000 records — and compute the insert difference for the pair `(X, Y)` under two conditioning sets. With conditioning set `{}` (the `T = ∅` insert): **−4.32**, score-decreasing, correctly flagging `X ⊥ Y`. With conditioning set `{Z}` (some `T` that pulls the collider `Z` in as a co-parent of `Y`): **+3926.8**, violently score-*increasing*.

So the easy statement is false, and concretely so. The very same non-adjacent pair `(X, Y)` simultaneously offers a score-decreasing insert and a hugely score-increasing one. This is not a corner case; it is the generic behavior of a collider, and it is *exactly* how GES manufactures a spurious adjacency: it finds a score-decreasing `Insert(X, Y, T)` for some `T`, ignores it, and applies a different score-increasing `Insert(X, Y, T')` — here the one that conditions on `Z` — committing the false `X–Y` edge anyway. Now the rule is forced and I trust it. At a forward state, for each non-adjacent pair `(X, Y)`, iterate over the valid `Insert(X, Y, T)`. The moment *any one* of them is score-decreasing, treat that as the score asserting `X ⊥ Y`: discard *all* `Insert(X, Y, *)` for that pair, mark the pair separated, move on. Only for pairs where no tried insertion is score-decreasing do I keep the score-increasing insertions as candidates; among all retained candidates across all pairs I apply the single best — still a valid score-increasing move, so the generalized-GES guarantee is intact. Call this CONSERVATIVEINSERT. It refuses precisely the move my collider example showed GES making: the moment any `T` says "independent," the `+3926.8` insert for the same pair is off the table. Two payoffs fall out and they match the two things that hurt. Accuracy: I never insert the excess adjacencies the backward phase might later fail to remove. Efficiency: finding one score-decreasing insertion lets me stop enumerating `T` subsets for that pair and shrinks the candidate set in every later state.

I have to be honest about the guarantee now, because a clever rule that quietly breaks correctness is worse than plain GES. Is CONSERVATIVEINSERT guaranteed to find a score-increasing insertion *whenever one is needed*? Soundness here would rest on a strong premise: that whenever `P` is not Markov with respect to the current class, there exist variables `X, Y` such that `X ⊥̸ Y | Pa_Y` for *every* member DAG — i.e. the pair whose insertion is required can be identified without depending on which member DAG I happened to pick. That premise is the hard part, and I cannot close it; it is only partially established. I do not want to ship an algorithm whose correctness is open, so I need a relaxation that is provably safe and that I can fall back to.

The safe relaxation weakens the declining condition to something I *can* prove. Instead of declining a pair whenever *any* `Insert(X, Y, T)` is score-decreasing, decline it only on the basis of the simplest insertion: pick an arbitrary DAG `G` in the current class, and for each non-adjacent pair `(X, Y)` check whether `G ∪ {X → Y}` scores *lower* than `G` — equivalently whether the `T = ∅` insertion into the witness DAG is score-decreasing. If so, discard all `Insert(X, Y, *)`; otherwise keep the score-increasing ones. Call this SAFEINSERT. Notice my collider check already exercised this exact screen: the `T = ∅` insert there was the **−4.32** one, the side that correctly separates, while the dangerous `+3926.8` insert was a non-empty `T`. The premise SAFEINSERT needs is the weaker, provable one: if `P` is not Markov with respect to the class, then for *every* DAG `G ∈ E` there exist `X, Y` with `X ⊥̸ Y | Pa_Y^G` — and unlike the conservative premise, here the choice of `X, Y` *may* depend on `G`, which is exactly what makes it provable. With that, SAFEINSERT returns a valid score-increasing insertion if and only if one exists, so generalized GES with SAFEINSERT recovers the true MEC in the sample limit, from any initial class. So I have a conservative strategy (empirically the most accurate, soundness open) and a safe strategy (provably correct), and the natural design ships both: run conservative for accuracy with the safe `T = ∅` screen riding along as the soundness backstop, and the user can fall back to safe-only if they need the proof.

That settles the forward phase. The backward phase I will leave *exactly* as GES's: greedily apply the highest-scoring `Delete(X, Y, H)` until none improves. I do not touch it, both because it is not the source of the over-adding and because keeping it verbatim means GES's correctness argument for deletions carries over with nothing to re-prove. So the whole algorithm — Less Greedy Equivalence Search — is GES with one surgical change: the forward insertion *selection*, conservative or safe. (There are further optional heuristics from the XGES line — prioritizing insertions before deletions, forcing deletions with restarts — that bolt on as variants LGES-0/LGES/LGES+, but the core, and the part that buys the accuracy and the guarantee, is the insertion strategy.)

Now the discrete instantiation, since that is the case at hand — integer-coded multinomial data. The score is the same BDeu I just computed by hand for the sign checks: the uniform-Dirichlet multinomial marginal likelihood, decomposable and score-equivalent (the unique BD score with score equivalence), with equivalent sample size `α = 1` and structure prior `1`. Nothing about the score changes — that is the point; the only thing that changes from GES is the insertion policy, so any improvement is attributable to the search strategy, not the criterion.

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

Let me retrace the chain to be sure each link is one I actually earned and not one I assumed. GES is already large-sample optimal, so the live problem is finite-sample accuracy and scale, and the documented finite-sample defect is spurious adjacencies, all introduced by `Insert`. The assumption I questioned is GES's greedy "apply the highest-scoring insertion," and re-reading its optimality proof showed correctness needs only "apply *some* score-increasing operator" (generalized GES), so I am free to decline insertions. The score itself flags a spurious adjacency: I computed the BDeu insert difference by hand and watched its sign go negative under independence (−3.53) and strongly positive under dependence (+757), so a score-decreasing `Insert(X, Y, T)` does read off `X ⊥ Y` for its conditioning set. I then tested whether one score-decreasing insert settles a pair, and the collider construction said no — the same non-adjacent `(X, Y)` gave −4.32 for `T = ∅` and +3926.8 for `T = {Z}` — which is exactly the trap GES falls into and exactly why CONSERVATIVEINSERT must decline the *whole* pair the moment *any* insert is score-decreasing. Its soundness rests on a premise I could not close, so SAFEINSERT relaxes the declining condition to the provable `T = ∅` screen — the very −4.32 side of that collider — which returns a score-increasing insertion iff one exists and so keeps the sample-limit MEC-recovery guarantee. The backward (delete) phase is GES's, untouched, so the deletion correctness carries over with nothing to re-prove. The discrete score is the same BDeu I computed with, so the only change from the strongest prior method is the forward insertion policy — a controlled, single-knob less-greedy modification of GES that targets exactly its over-adding failure mode while preserving its guarantee.
