We are given a data matrix of $m$ iid records over $n$ variables drawn from an unknown distribution $P(V)$, and we want the Markov equivalence class of the DAG that generated it — the CPDAG that names that class. The natural starting point is Greedy Equivalence Search, and it is a beautiful thing: it searches the space of equivalence classes (each represented by its unique completed PDAG) by greedily applying the highest-scoring edge insertion until none improves the score, then the highest-scoring edge deletion until none improves it, and with a decomposable, consistent, score-equivalent score this two-phase climb is *guaranteed*, in the large-sample limit, to land on the true MEC. So I am not searching for a guarantee — I already have one. What hurts is finite samples and scale. With limited data GES routinely fails to recover the true MEC, and the failure has a known signature: GES's output contains adjacencies between variables that are *non-adjacent* in the truth. And it is slow — the problem is NP-hard, and GES grinds on high-dimensional graphs. The constraint I impose on myself is to fix both *without leaving the GES family*, because the family — same decomposable score, same `Insert`/`Delete` operators, same equivalence-class search — is the whole reason the correctness machinery exists. Constraint-based alternatives like PC (Spirtes, Glymour & Scheines 2000) share the asymptotic guarantee but rely on thresholded CI tests with hard verdicts, and methods such as max-min hill-climbing (Tsamardinos et al. 2006) or NOTEARS (Zheng et al. 2018) carry no such large-sample guarantee at all. None of those is the lever I want; the lever is inside GES.

Where do the spurious adjacencies come from? They are introduced by `Insert` operators — the forward phase is the only thing that adds edges. So the lever, if there is one, is the choice of *which* insertion to apply, and the assumption I want to question is GES's implicit claim that the highest-scoring insertion is the best move. It is not necessary. Greedy hill-climbing finds a global optimum not because it takes the steepest step but because it halts only where *no* improving step exists, and any path of improving steps reaches such a point. So if I relax GES to apply *any* valid score-increasing operator rather than the best one, the search still terminates at a class admitting no score-increasing operator, which in the sample limit is the score's global optimum: the true MEC. The optimality argument never used *which* improving operator was applied, only that one exists whenever the current class is not yet Markov — for the forward phase (reach a class with respect to which $P$ is Markov) and equally for the backward phase (reach the perfect map). This is Generalized GES: initialize from any class, apply operators in any order, apply any score-increasing operator. It still recovers the true MEC, and it is exactly the freedom I needed — I may choose strategically which score-increasing insertion to take, and in particular I may *decline* insertions GES would have greedily made.

I propose **LGES — Less Greedy Equivalence Search**: GES with one surgical change to the forward insertion *selection*, declining to insert an edge between any pair for which the score already implies a conditional independence, with everything else — score, operators, the entire backward deletion phase — left exactly as GES's. The pivot is that the score itself can tell me when an adjacency is spurious before I commit it. Recall the operator: `Insert(X, Y, T)` adds $X \to Y$ and orients a chosen subset $T$ of $Y$'s undirected neighbors (those not adjacent to $X$) into $Y$; for a fixed pair $(X,Y)$ there are many such operators, one per valid $T$, each corresponding to a witness DAG in which $Y$'s parent set is $NA_{Y,X} \cup T \cup Pa_Y$. By decomposability only $Y$'s family changes, and by score equivalence the class score change equals that witness-DAG change, so the score change of `Insert(X, Y, T)` is the single node-family difference
$$\Delta = s\!\left(Y,\; NA_{Y,X} \cup T \cup Pa_Y \cup \{X\}\right) - s\!\left(Y,\; NA_{Y,X} \cup T \cup Pa_Y\right).$$
Now invoke local consistency — the property that decomposability plus consistency hand me (Chickering 2002): in the sample limit, $\Delta > 0$ iff $X$ and $Y$ are *dependent* given $NA_{Y,X} \cup T \cup Pa_Y$, and $\Delta < 0$ iff $X \perp Y$ given that set. So a score-*decreasing* `Insert(X, Y, T)` is the score telling me, through one operator, that $X \perp Y$ holds for a particular conditioning set — a conditional independence read off a model-score comparison rather than a thresholded CI test. And in the sample limit, if $X$ and $Y$ are genuinely non-adjacent in the true MEC, then *some* valid $T$ exposes exactly such a separating set, so *some* `Insert(X, Y, T)` is score-decreasing. The contrapositive is my signal: a score-decreasing insertion for the pair is evidence the pair is non-adjacent in the truth. Crucially, a single score-decreasing $T$ does not make *all* of the pair's insertions score-decreasing — and this is precisely GES's failure mode. GES may find a score-decreasing `Insert(X, Y, T)` and then, ignoring it, apply a *different* score-increasing `Insert(X, Y, T')` and commit the spurious adjacency anyway.

That hands me the less-greedy rule, which I call CONSERVATIVEINSERT. At each forward state, for each non-adjacent pair $(X, Y)$, iterate over the valid `Insert(X, Y, T)`; *the moment any one of them is score-decreasing*, treat it as evidence $X \perp Y$, mark the pair separated, discard *all* `Insert(X, Y, *)` for that pair, and move on. Only pairs for which no tried insertion was score-decreasing keep their score-increasing insertions as candidates; among all retained candidates across all pairs I apply the single best — still a valid score-increasing move, so the Generalized-GES guarantee of termination at the global optimum is intact. This closes exactly the hole that makes GES fail: the moment any $T$ says "independent," the pair is off the table, so I never even consider the spurious $T'$. Two payoffs fall out, matching the two things that hurt. Accuracy: I never insert the excess adjacencies the backward phase might fail to remove. Efficiency: finding one score-decreasing insertion lets me stop enumerating $T$ subsets for that pair and shrinks the candidate set in every later state.

I have to be honest about the guarantee, because a clever rule that quietly breaks correctness is worse than plain GES. CONSERVATIVEINSERT's soundness would rest on a strong premise — that whenever $P$ is not Markov with respect to the current class, there exist $X, Y$ with $X \not\perp Y \mid Pa_Y$ for *every* member DAG, i.e. the needed pair can be identified without depending on which member DAG I picked. That premise is only partially established, so CONSERVATIVEINSERT's soundness is left open. I therefore add a provably safe relaxation to fall back to: SAFEINSERT. It weakens the declining condition to something I can prove. Rather than declining a pair when *any* $T$ is score-decreasing, decline it only on the basis of the simplest insertion: pick an arbitrary DAG $G$ in the current class and check whether $G \cup \{X \to Y\}$ scores *lower* than $G$ — equivalently whether the $T = \emptyset$ insertion is score-decreasing. The premise this needs is the weaker, provable one: if $P$ is not Markov with respect to the class, then for *every* DAG $G \in E$ there exist $X, Y$ with $X \not\perp Y \mid Pa_Y^G$ — and here the choice of $X, Y$ may depend on $G$, which is exactly what makes it provable. With that, SAFEINSERT returns a valid score-increasing insertion if and only if one exists, so Generalized GES with SAFEINSERT recovers the true MEC in the sample limit, from any initial class. The natural design ships both: run conservative for accuracy with the safe $T = \emptyset$ screen as the soundness backstop (in code, the $T=\emptyset$ subset is enumerated and its $\Delta \le 0$ verdict is subsumed by the same `gain <= 0` separation rule), and a user who needs the proof can fall back to safe-only.

The backward phase I leave *exactly* as GES's: greedily apply the highest-scoring `Delete(X, Y, H)` until none improves. I do not touch it, both because it is not the source of the over-adding and because keeping it verbatim means the deletion correctness argument carries over unchanged. There are further optional scheduling heuristics from the XGES line (Nazaret & Blei 2024) — prioritizing insertions before deletions, forcing deletions with restarts — that bolt on as variants LGES/LGES+, but the core, and the part that buys the accuracy and the guarantee, is the insertion strategy. For the discrete case at hand — integer-coded multinomial data — the score is the same BDeu I would use for GES: the uniform-Dirichlet multinomial marginal likelihood, decomposable and score-equivalent (the unique BD score with score equivalence), with equivalent sample size $\alpha = 1$ and structure prior $1$. Nothing about the score changes; that is the point. The only change from GES is the insertion policy, so any improvement is attributable to the search strategy, not the criterion.

I write it as the search I would actually run, built on the `causal-learn` GES operator primitives so that the operator validity tests, local score differences, and CPDAG conversions are the canonical routines and only the forward selection differs. The forward loop precomputes neighbors, adjacents, parents and semi-paths; for each non-adjacent ordered pair it enumerates valid $T$ subsets (the clique test on $NA \cup T$ and the semi-directed-path test, exactly as GES does), scores each by the insert-difference helper, tracks per pair whether any valid insertion is score-decreasing, and the moment one is, marks the pair separated and discards its candidates. Among retained strictly-score-increasing candidates from un-separated pairs it applies the global best, reconverting the CPDAG only on a move. The backward loop is GES's `Delete` loop verbatim, returning the completed PDAG as the estimated CPDAG.

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
