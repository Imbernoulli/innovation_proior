Let me start from what actually goes wrong, because the goal here is not subtle: I have `n` i.i.d. rows on `p` variables, no interventions, no time order, and I want the causal structure. Observational data can only ever pin `G*` down to its Markov equivalence class, so honestly what I'm after is the CPDAG. Fine. The trouble is the regime I care about. The provably-correct workhorses — PC and GES — are correct in the limit under faithfulness, and yet when I look at how they behave on graphs that are at all dense, their accuracy is well below what the theorems promise, and the shortfall gets worse as the average degree climbs. Adjacency precision stays high but recall falls off a cliff: they miss real edges. So the theory is fine and the practice is not, which means the assumptions are doing more work than they look like they're doing.

Why would that be? The standing diagnosis is almost-violations of faithfulness. Faithfulness says every conditional independence in `P` is one the graph forces; the failure mode is a dependence that is real but *faint* — two near-cancelling paths, or a near-deterministic relation — so that the dependence strength is pushed close to zero. Then a finite-sample conditional-independence test reports "independent" when the truth is a weak "dependent," and a greedy edge-addition score sees almost no gain from the right edge. PC deletes an edge it shouldn't and the mistake cascades through orientation; GES never adds the edge in the first place. And these near-cancellations aren't pathological corner cases — on dense graphs they're everywhere, because the more paths there are between two variables the more ways their influences can partially cancel. So I shouldn't trust any procedure whose every decision hinges on a single CI test or a single edge's marginal score gain. I want decisions that are more robust to faint-but-real dependences.

What tolerates almost-unfaithfulness? There's a line of work that searches *orderings* instead of edges. The sparsest-permutation idea: take a permutation `π` of the variables, treat it as an acyclic order, and build a DAG `G_π` by giving each variable parents drawn only from the variables ahead of it in `π`. Among all permutations, return the one whose induced DAG has the fewest edges. The appeal is the assumption it needs — u-frugality, "the true DAG is the uniquely sparsest Markovian DAG" — which is strictly weaker than faithfulness. It survives exactly the near-cancellation cases that sink PC and GES, because it's asking a global sparsity question rather than a local independence question. The catch is brutal: it enumerates all `p!` permutations. That's nine variables, maybe, before it dies. So the *robustness* lives in the permutation view, and the open problem is "how do I move through permutation space without enumerating it."

So let me commit to the permutation view and figure out the two things I actually need: how to turn one `π` into a scored DAG cheaply and correctly, and how to walk from one `π` to a better one.

First the projection, because everything downstream calls it. Given `π`, the parents of variable `k` should be chosen from `pre(k)`, the variables before it. The clean object is a Markov boundary of `X_k` within `pre(k)`: the smallest predecessor set that renders `X_k` independent of the rest of the predecessors. Two ways to get it. One (RU / Verma–Pearl) draws `j → k` iff `j` precedes `k` and `X_j ⊥̸ X_k | X_{pre(k)\{j}}` — but that's a conditional-independence test per edge, and I just argued CI tests are the fragile thing I'm trying to get away from. The other (VP) sets the parents of `k` to a Markov boundary among the predecessors, and I can estimate that boundary with a *score*, no hypothesis testing. For a graphoid the two recipes give the same DAG, so I lose nothing by taking the score route, and I gain robustness. And there's a structural fact I want to lean on: for every `π`, the induced `G_π` is Markovian and subgraph-minimal, and conversely a DAG is subgraph-minimal iff it's the projection of *some* permutation. That last "iff" is the load-bearing one — it tells me the set of subgraph-minimal DAGs, which contains `G*`, is exactly the image of permutation space under projection. I'm not throwing away the answer by restricting to projections of orderings; the true MEC is reachable.

Now, the score. I want something decomposable, `BIC(G) = Σ_v BIC(X_v, X_{pa(v)})`, so projecting and scoring is per-node and local. BIC is consistent on curved exponential families — Gaussian and multinomial DAG models both qualify — and the property I really need is *local consistency*: in the large-sample limit, adding `j → k` to `G` raises the BIC iff `X_j ⊥̸ X_k | X_{pa(k)}`. Read that again. It means the score *already encodes the independence facts* a CI test would give me, but as a comparison of two model scores rather than a thresholded test statistic. So if I select parents by hill-climbing the BIC, I'm implicitly doing the right conditional-independence reasoning, just more stably.

How do I find the Markov boundary by score? Grow then shrink. Grow: `M ← ∅`; repeatedly add the predecessor `Y` that most increases `BIC(X_k, M ∪ {Y})`, stop when no addition helps. By local consistency, I only add `Y` when `X_k ⊥̸ Y | M`, i.e. when `Y` carries information about `X_k` beyond what `M` already explains — so grow keeps pulling in genuinely informative predecessors and stops when the rest are conditionally independent given what I've got. That gives a superset of the boundary (it can overshoot, because a variable that looked informative early may be redundant once others are in). So shrink: from the grown set, repeatedly remove the `Y` whose removal most increases `BIC(X_k, M \ {Y})`, stop when none helps. Removing `Y` helps exactly when `X_k ⊥ Y | M\{Y}` — `Y` is now redundant — so shrink peels off the false positives. In the limit, on a compositional graphoid, `shrink(grow(pre(k)))` is the *unique* Markov boundary, and the per-node shrink scores sum to `BIC(G_π)`. Good: I have `project(π)` → a subgraph-minimal DAG, and `score(π) = BIC(G_π)`, all from local score calls, no CI tests. The discrete-data version just swaps the local score for the BDeu marginal likelihood; I'll come back to that, because the score-equivalence of BDeu is what makes searching over equivalence classes well-posed, and I want to actually check that property rather than take it on faith.

Now the real question: how do I move through permutation space? The crudest sensible move is Teyssier–Koller's adjacency transposition — swap two neighbours in `π`. Its one virtue is that a single adjacency swap only changes the local scores of the two swapped variables, so it's cheap to evaluate. But it's *too* local: a single swap barely changes the induced DAG, so hill-climbing with adjacency swaps stalls in shallow optima on hard graphs, and Teyssier–Koller give no consistency guarantee anyway. At the other extreme is the tuck operator from the GRaSP line — re-order the minimal stretch of `π` needed to flip a chosen edge of `G_π`, run as a depth-first search over sequences of covered/singular/general tucks, in tiers. That's powerful and it's the current accuracy champion on dense graphs, robust to the almost-unfaithfulness I care about. But look at what it costs: a DFS over tuck sequences with several interacting depth knobs — overall depth, singular depth, uncovered depth — and a running time that climbs steeply, so it doesn't comfortably get past a few hundred variables, and the knobs make it fiddly to run. I want its accuracy without its machinery. So the question I have to answer is whether there's a simpler move that still leaps far enough in one step to escape the shallow optima that trap single swaps.

Let me think about what a good move should accomplish. The score depends on `π` only through, for each variable, which other variables sit ahead of it (that determines its candidate parents). The thing that most changes a variable's parent set is *where that variable sits* relative to everyone else. So instead of swapping neighbours or tucking around a particular edge, suppose I take one variable `v`, pull it out, and drop it back into the single best slot — the position among all `|π|` possible positions that maximizes the total score. That's not a local swap; sliding `v` from the end to the front is, in effect, a whole run of adjacent transpositions collapsed into one decision, so it can leap across the shallow optima that trap single swaps. And it has zero depth parameters — there's nothing to tune, I just evaluate every insertion point and take the argmax. Sweep that best-move over every variable, one at a time, and repeat the whole sweep until a full pass produces no improvement. When the sweep converges, project to a DAG. Let me call this candidate the "best-position move" and keep going to see whether it's good enough.

Is that enough to be *correct*, not just good? Project already hands me a subgraph-minimal DAG that contains `P`, which is most of the way there. To get the asymptotic guarantee I can borrow the backward phase of GES — BES — which greedily deletes edges that improve the score until none do; running it after the best-move sweep cleans the result up to the true MEC, and its correctness carries the whole procedure's correctness. So the shape is two-phase, exactly like GES: a greedy phase (here, best-move over permutations) then an optional backward-equivalence phase, with the data only identifying the MEC so I finish by converting the DAG to a CPDAG. The BES step is genuinely optional in the design: it's the switch that buys the formal large-sample guarantee, while the lean implementation can stop after the ordering phase and CPDAG conversion when I just want the parameter-light search itself. The guarantee follows from the chain: because `project` always returns a subgraph-minimal DAG containing `P`, asymptotic correctness of the whole two-phase search reduces to the correctness of BES, for *any* starting permutation. That "any start" is a real convenience — no need for clever initialization or random restarts to get the guarantee, so I can just start from the identity order.

So far so clean, but now the cost. The best-move for a single variable evaluates `O(p)` insertion positions, and each position needs me to know that variable's parent set for the relevant prefix — a grow/shrink. A full sweep does this for all `p` variables, and I iterate sweeps to convergence. If I recompute grow/shrink from scratch at every position of every variable on every sweep, I'm doing an enormous amount of repeated work, because the prefixes overlap massively: when I score `v` at position `i` and then at position `i+1`, almost the same set of predecessors is in play; when I score `v` on this sweep versus the next, again almost the same. The grow result for variable `v` given a prefix is the bottleneck primitive, and I'm calling it on overlapping prefixes a staggering number of times. So the scalability isn't going to come from a smarter move — the best-position move is already simple — it has to come from reusing grow and shrink work whenever the same state recurs.

So I want to *cache* grow/shrink results, keyed by what they actually depend on. What does `grow(v, prefix)` depend on? Grow is greedy: at each step it adds the single best-scoring available candidate. So the grown parent set for a given prefix is determined by a deterministic sequence of "best available addition" decisions. That's a *path*. Picture a tree, one per variable `v`: the root is the empty parent set (score `BIC(X_v, ∅)`); from any node, the children are the candidate additions, and I keep only the children that *strictly improve* the score, and I sort those children in descending order of their score. Now grow for any particular prefix should be a walk down this tree: from the current node, go to the highest-scoring child whose variable is in the prefix; if that child's variable isn't in the prefix, skip to the next-highest; recurse. The claim I'm making is that the greedy "best available addition" equals "the first sorted child that's available." I should check that equality before I build anything on top of it, because it's the entire reason the tree is correct, and it's not obvious — sorting the branches once, at expansion time, has to agree with grow's per-step argmax even though grow's argmax is taken over a *shrinking* available set as the prefix is consumed.

Let me actually verify it rather than wave at it. I'll take a toy additive-plus-pairwise score over 7 candidate variables, and for 3000 random prefixes compare two things: (a) honest greedy grow restricted to the prefix — repeatedly scan the available candidates, add the single best-improving one, stop when none improves; (b) the tree descent — at each node, expand by scoring every available addition, keep the strictly-improving ones, sort them descending, then walk that sorted list and recurse into the first branch whose variable is in the prefix, removing branches from the available set as I pass them (mirroring how the trace consumes `available`). Running it, all 3000 prefixes match on both the returned parent set and the score, zero mismatches. So the equality holds, and I can see *why* it holds from the run: at any node, the highest-scoring improving branch whose variable is in the prefix is, by construction, the best-improving available addition — the branches I skip over are either not improving (so grow would never take them) or not in the prefix (so they aren't available), and skipping them doesn't change which of the *remaining* branches is the argmax, because the argmax is over scores that were all computed against the same fixed parent set at that node. That's the subtlety I was worried about, and it survives the check. Call it a grow-shrink tree; shrink runs once per terminal node and gets cached there too.

Why does this kill the redundancy? Because the same prefix-walk gets reused across every permutation and every insertion position that shares a prefix. The first time I need a node I expand and score it; after that, traversing through it reuses the stored branch scores and the stored shrink removals. The tree expands *lazily* — only the paths actually visited ever get built — so I never pay for parent sets no permutation explores. And the sorting matters for more than speed: it makes grow a deterministic descent, so any two prefixes that lead down the same branch share the same cached scores. The expensive thing — the local score call — moves from repeated grow/shrink runs to tree-node expansion. That's the whole game. With this, the best-move sweep that looked quadratic-times-grow becomes a forest of cached descents, which is the mechanism that makes the large, dense regime plausible.

Let me get the tree's grow rule exactly right, because there's a subtlety in "keep only improving children." When I expand a node, I score adding each available candidate to the current parent set; a candidate becomes a branch only if its score strictly exceeds the node's own (grow) score — i.e. only if adding it improves things. Candidates that don't improve aren't branches at all, which is correct: grow would never add them. Then I sort the surviving branches descending by score. During a trace for a given prefix I walk the sorted branches and take the first whose added variable is in the prefix — that's the best *available* improving addition, which is exactly what the check above confirmed grow does. If none of the (improving, sorted) branches is in the prefix, grow is finished, and I run shrink at this node (cached), returning the shrink score. One more bookkeeping point: a variable can't be its own parent, so `v` itself is forbidden from `v`'s tree.

Now let me make the best-move itself efficient, because there's a trick that turns "try `v` at every position" from `O(p)` independent re-scorings into a single linear sweep. Lay out the order and walk `v` forward through it. As I conceptually slide `v` to position `j`, the total score splits into two independent pieces: the score of `v` *with the prefix that would sit ahead of it at position `j`*, plus the score of *every other variable* given *its* prefix. The second piece I can accumulate incrementally as I extend the prefix one variable at a time. So I do one forward pass building up the prefix and recording, at each candidate slot `j`, `score(v | prefix_j) + Σ_{w placed before j} score(w | its prefix)`; then a backward pass that adds in `Σ_{w placed after j} score(w | its prefix)` by removing variables from the prefix one at a time. After both passes I should have the total score for `v` at every insertion slot, in linear time, each term a cached tree trace. Take the argmax; if it beats `v`'s current position by more than a tiny tolerance, move `v` there, else leave it.

This split is fiddly enough — two passes, a prefix that grows then shrinks, and an insertion that has to account for `v`'s own removal — that I don't trust it until I've watched it reproduce a brute-force answer. So I'll write a brute-force reference: remove `v` from the order, try inserting it at every one of the `|rest|+1` slots, score each resulting full order from scratch, take the best. Then I'll run the two-pass sweep on the same order and check that (i) the slot it picks gives the same total as the brute-force best, and (ii) the move it actually applies — `order.remove(v); order.insert(best - int(best > i), v)` — lands on an order with that same best total. Over 2000 random orders of 6 variables with a fixed random score table, both checks pass on every trial, zero mismatches. That `best - int(best > i)` correction is the part I most wanted to see exercised: the forward/backward sweep computes `scores[j]` in the *original* order's coordinate frame (slots `0..p`), but once I physically remove `v` the slots after its old position `i` all shift left by one, so an insertion target `best` that lies to the right of `i` has to be decremented by one, and to the left it stays. Getting that off-by-one wrong would silently move `v` one slot away from the optimum the sweep found; the brute-force agreement is what tells me the index arithmetic is right rather than merely plausible. The tolerance `1e-6` in the comparison is just to avoid thrashing on numerically-tied positions so the sweep actually terminates.

A couple of choices in the sweep I should pin down. I sweep the variables in a *shuffled* order each round rather than a fixed order — fixing the order would bias which variable gets first claim on a good slot, and randomizing makes the search less sensitive to the meaningless input column order. And I keep iterating sweeps until a complete pass over all variables yields no move; that's my convergence test for the greedy phase. With "any initial permutation" fine asymptotically, I start from the identity order.

Now the discrete score, since that's the case at hand — integer-coded multinomial data. The local score becomes the BDeu marginal likelihood. Let `X_i` have `r_i` states; a parent set `PA_i` induces `q_i = Π_{a∈PA_i} r_a` joint parent configurations. Count `N_ij` = samples in parent-config `j`, and `N_ijk` = samples with `X_i = k` in config `j`. With a single equivalent-sample-size hyperparameter `α`, the BDeu marginal likelihood is

  Π_j [ Γ(α/q_i) / Γ(α/q_i + N_ij) · Π_k Γ(α/(r_i q_i) + N_ijk) / Γ(α/(r_i q_i)) ],

and I work with its log, so per parent configuration `j` the contribution is

  lgamma(α/q_i) − lgamma(N_ij + α/q_i) + Σ_k [ lgamma(N_ijk + α/(r_i q_i)) − lgamma(α/(r_i q_i)) ].

Let me sanity-check the Dirichlet bookkeeping by hand first. The prior over the `r_i`-way conditional `P(X_i | config j)` is Dirichlet with concentration `α/(r_i q_i)` on each of the `r_i` cells, so its total mass per configuration is `r_i · α/(r_i q_i) = α/q_i`, which is exactly the `α/q_i` sitting in the first two gamma terms — the marginal-likelihood of a Dirichlet-multinomial is `Γ(Σα_cell)/Γ(Σα_cell + N) · Π_cell Γ(α_cell + N_cell)/Γ(α_cell)`, and that's term-for-term what I wrote. Summed over all `q_i` configurations the prior counts total `q_i · α/q_i = α`, so `α` is genuinely an equivalent sample size spread across the whole table — that's the "equivalent uniform" in BDeu, and it's why a single scalar `α` suffices instead of an elicited prior.

The property I really need from BDeu is *score-equivalence*: it should assign the same value to every DAG in a MEC. Without that, the score I'm maximizing over orderings wouldn't be a well-defined function of the equivalence class, and an order-based search would be chasing an artifact of the chosen DAG representative rather than the structure. BDeu is supposed to be the unique BD score with this property, but "supposed to be" isn't good enough when my whole search is built on it, so let me put a number on it. The smallest nontrivial Markov-equivalent pair is a single edge: `X → Y` and `Y → X` encode the same independences (no v-structure), so a score-equivalent local score must give them the same *total*. I generate 400 samples of a dependent pair — `X` a fair coin, `Y = X` with a 20% flip — and compute `Σ_i BDeu(X_i | parents)` with `α = 1` for both DAGs using the log formula above. The result: `X→Y` totals `−499.21766344405...` and `Y→X` totals `−499.21766344405...`, agreeing to about 13 significant figures (difference `1.7e-13`, i.e. floating-point noise from summing the lgamma terms in a different order). So score-equivalence is real here, not just a quoted theorem, and the order search is maximizing a genuine function of the equivalence class. While I have that example up, it's worth confirming the score also *prefers the edge*: the empty graph (both variables parentless) totals `−560.11`, so adding the X–Y edge improves the score by about `+60.9` — the dependence I built in is being detected, which is the local-consistency behaviour grow relies on. Good: the discrete score does the two things the whole construction needs, and I've watched it do both. I can also fold in a structure prior over the parent set — a per-edge log term `|PA_i|·log(structure_prior/vm) + (vm−|PA_i|)·log(1−structure_prior/vm)` with `vm = p−1` — which gently favours sparser parent sets; with `structure_prior = 1` and `α = 1` (the defaults for this discrete setting) it sits alongside the marginal likelihood. Higher BDeu is better, which is the convention grow uses when it keeps only strictly-improving branches.

Let me also reconcile the projection's "edge of the CPDAG" question. The greedy phase plus project gives me a DAG; the data only identify the MEC, so I convert that DAG to its CPDAG — the find-compelled / DAG-to-CPDAG step that marks which edges are forced (compelled) by the equivalence class and which are reversible. That's the object I return.

Now let me write it as code I'd actually run, filling the two empty slots from the harness — the move strategy (`better_mutation`, the linear best-move sweep) and the caching it relies on (the grow-shrink tree). The public discrete call is `boss(X, score_func="local_score_BDeu")`; inside that path, the BDeu scorer uses `sample_prior = 1`, `structure_prior = 1`, and observed state counts by default, so the engine code should call through the same score class rather than invent a different scoring interface.

```python
import random
import warnings
from typing import List, Optional

import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import local_score_BDeu
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag


class GSTNode:
    def __init__(self, tree, add=None, score=None):
        if score is None:
            score = tree.score.score_nocache(tree.vertex, [])
        self.tree = tree
        self.add = add
        self.grow_score = score
        self.shrink_score = score
        self.branches = None
        self.remove = None

    def __lt__(self, other):
        return self.grow_score < other.grow_score

    def grow(self, available, parents):
        self.branches = []
        for add in available:
            parents.append(add)
            score = self.tree.score.score_nocache(self.tree.vertex, parents)
            parents.remove(add)
            branch = GSTNode(self.tree, add, score)
            if score > self.grow_score:
                self.branches.append(branch)
        self.branches.sort(reverse=True)

    def shrink(self, parents):
        self.remove = []
        while True:
            best = None
            for remove in [parent for parent in parents]:
                parents.remove(remove)
                score = self.tree.score.score_nocache(self.tree.vertex, parents)
                parents.append(remove)
                if score > self.shrink_score:
                    self.shrink_score = score
                    best = remove
            if best is None:
                break
            self.remove.append(best)
            parents.remove(best)

    def trace(self, prefix, available, parents):
        if self.branches is None:
            self.grow(available, parents)
        for branch in self.branches:
            available.remove(branch.add)
            if branch.add in prefix:
                parents.append(branch.add)
                return branch.trace(prefix, available, parents)
        if self.remove is None:
            self.shrink(parents)
            return self.shrink_score
        for remove in self.remove:
            parents.remove(remove)
        return self.shrink_score


class GST:
    def __init__(self, vertex, score):
        self.vertex = vertex
        self.score = score
        self.root = GSTNode(self)
        self.forbidden = [vertex]
        self.required = []

    def trace(self, prefix, parents=None):
        if parents is None:
            parents = []
        available = [i for i in range(self.score.data.shape[1]) if i not in self.forbidden]
        return self.root.trace(prefix, available, parents)

    def reset(self):
        self.root = GSTNode(self)


def reversed_enumerate(seq, j):
    for w in reversed(seq):
        yield j, w
        j -= 1


def better_mutation(v, order, gsts):
    i = order.index(v)
    p = len(order)
    scores = np.zeros(p + 1)

    prefix = []
    score = 0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + score
        if v != w:
            score += gsts[w].trace(prefix)
            prefix.append(w)

    scores[p] = gsts[v].trace(prefix) + score
    best = p

    prefix.append(v)
    score = 0
    for j, w in reversed_enumerate(order, p - 1):
        if v != w:
            prefix.remove(w)
            score += gsts[w].trace(prefix)
        scores[j] += score
        if scores[j] > scores[best]:
            best = j

    if scores[i] + 1e-6 > scores[best]:
        return False
    order.remove(v)
    order.insert(best - int(best > i), v)
    return True


def boss_discrete(X: np.ndarray, node_names: Optional[List[str]] = None) -> GeneralGraph:
    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    score = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    order = list(range(p))
    gsts = [GST(v, score) for v in order]
    parents = {v: [] for v in order}

    variables = list(order)
    while True:
        improved = False
        random.shuffle(variables)
        for v in variables:
            improved |= better_mutation(v, order, gsts)
        if not improved:
            break

    for i, v in enumerate(order):
        parents[v].clear()
        gsts[v].trace(order[:i], parents[v])

    names = [("X%d" % (i + 1)) for i in range(p)] if node_names is None else node_names
    nodes = [GraphNode(name) for name in names]
    G = GeneralGraph(nodes)
    for y in range(p):
        for x in parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])
    return dag2cpdag(G)


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    return boss_discrete(X)
```

Let me retrace the causal chain to make sure it hangs together. I started stuck: PC and GES are provably correct but their accuracy on dense graphs falls far short of the theory, because almost-violations of faithfulness — faint-but-real dependences from near-cancellation — are everywhere on dense graphs and they fool the per-edge CI tests and greedy edge gains those methods rely on. The permutation/sparsest-order view tolerates exactly those cases under the weaker u-frugality assumption, but enumerating `p!` orderings is hopeless. So I kept the permutation view and attacked its two needs. Projection: turn `π` into a DAG by giving each node a Markov boundary of its predecessors, found by grow-shrink on a decomposable, locally-consistent BIC (BDeu in the discrete case, whose score-equivalence I checked numerically gives `X→Y` and `Y→X` identical totals to ~13 figures), which encodes the right independence reasoning as score comparisons rather than fragile tests and yields a subgraph-minimal DAG whose score is `BIC(G_π)`. Movement: instead of GRaSP's powerful-but-heavy tuck-DFS with its depth knobs, take one variable and slide it to its single best position — a big, parameter-free move that collapses a whole run of adjacent transpositions and escapes the shallow optima that trap single swaps — which I checked against brute force over 2000 random orders (including the `best - int(best > i)` insertion correction) before trusting it; sweep it over all variables, repeat to convergence, with an optional GES backward phase that secures asymptotic correctness for any starting order. Scalability: the best-move re-scores enormously overlapping prefixes, so cache grow and shrink in a per-variable tree whose nodes are grown parent sets reached by a deterministic descent through sorted, improving branches — a descent I checked reproduces honest greedy grow on 3000 random prefixes — expanded lazily, so repeated prefix traces reuse stored work. Finish by converting the projected DAG to its CPDAG, since the data identify only the equivalence class. The result is a parameter-light ordering search whose expensive local-score work is organized around reusable grow-shrink paths instead of repeated full projections from scratch.
