PC's numbers tell me exactly what I feared, and the precision/recall split is the lever. On the two small sparse graphs PC's adjacency precision is fine (1.0 on ER10, 0.857 on ER20) but its *recall* is already sagging (0.857, 0.585), and on the larger and denser scenarios it collapses: SF50 adjacency recall 0.51, SF50-Hard 0.30, ER20-Noisy a catastrophic 0.246 with adjacency precision still 0.85. That split is diagnostic — PC is not adding junk, it is *missing real edges*, and it misses more as density and noise climb, exactly where a single thresholded Fisher-z test deletes a true edge whenever the partial correlation of a real-but-faint dependence sits near zero. The arrow numbers are even worse (SF50 arrow recall 0.19, ER20-Noisy 0.044) because they sit downstream of a broken skeleton, and the SHD line — 8, 31, 93, 143, 69 — says the same in one number. PC's correctness theorem is true; the finite-sample assumption it rests on, faithfulness, breaks exactly where edges crowd together. So I do not want a faster faithful algorithm. I want decisions robust to faint-but-real dependences, and I want them to scale to the fifty-node graphs where PC fell apart.

I propose **GRaSP** (Greedy Relaxations of the Sparsest Permutation). What tolerates almost-unfaithfulness is the line of work that searches *orderings* instead of edges. Take a permutation $\pi$ of the variables, treat it as an acyclic order, and build a DAG $G_\pi$ by giving each variable parents drawn only from the variables ahead of it in $\pi$; among all permutations, return the one whose induced DAG has the fewest edges. The assumption this needs — u-frugality, "the true DAG is the uniquely sparsest Markovian DAG" — is strictly weaker than faithfulness, because it asks a *global sparsity* question rather than a *local independence* one. A faint dependence that hides an edge just makes the true graph look sparser, and sparsity is exactly what is being optimized, so the search survives precisely the near-cancellations that sink PC. The catch is brutal: enumerating all $p!$ permutations is dead at nine variables. So the robustness lives in the permutation view, and my whole problem becomes how to move through permutation space without enumerating it.

Two things are needed: how to turn one $\pi$ into a scored DAG cheaply, and how to walk to a better $\pi$. For the projection, the parents of variable $k$ should be its Markov boundary within $\mathrm{pre}(k)$, the smallest predecessor set rendering $X_k$ independent of the rest of the predecessors. I could draw $j \to k$ whenever a CI test says they are dependent given the other predecessors — but that puts me right back into the fragile CI regime that sank PC. The escape is to use a *score*. I want a decomposable local score, $\mathrm{score}(G_\pi) = \sum_v \mathrm{score}(X_v, \mathrm{Pa}(v))$, with *local consistency*: in the large-sample limit, adding $j \to k$ raises the score iff $X_j \not\perp X_k \mid \mathrm{Pa}(k)$. That property means the score already encodes the independence facts a CI test would give, but as a comparison of two model scores rather than a thresholded statistic — the same information, far more stable. For this linear Gaussian task the right such score is the Gaussian SEM-BIC, `local_score_BIC_from_cov`, with the complexity multiplier `lambda_value = 2` — a heavier penalty than textbook BIC's 1, which on dense scale-free graphs guards against over-adding — and it reads the data only through the covariance, so per-family scoring is a cheap linear-algebra call.

Finding each Markov boundary by score is grow-then-shrink. Grow: start $M = \emptyset$ and repeatedly add the predecessor that most improves $\mathrm{score}(X_k, M \cup \{Y\})$, stopping when nothing helps — by local consistency this only pulls in $Y$ when $X_k \not\perp Y \mid M$, and overshoots into a superset of the boundary. Shrink: from the grown set, repeatedly remove the $Y$ whose removal most improves the score, peeling off redundancies, since removing $Y$ helps exactly when $X_k \perp Y \mid M \setminus \{Y\}$. In the limit this returns the unique Markov boundary, and the per-node shrink scores sum to $\mathrm{score}(G_\pi)$, so I have `project(π)` and `score(π)` entirely from local score calls and no CI tests. There is also a structural guarantee I lean on: every $G_\pi$ is Markovian and subgraph-minimal, and a DAG is subgraph-minimal iff it is the projection of *some* permutation, so the true MEC stays reachable inside the permutation image.

Now the move. The crudest option is Teyssier-Koller's adjacent transposition — swap two neighbors in $\pi$ — whose one virtue is that a single swap only changes the local scores of the two swapped variables. But it is *too* local: a single swap barely changes the induced DAG, so hill-climbing with adjacent swaps stalls in shallow optima on exactly the hard dense graphs I care about, and it has no consistency guarantee. I need a move that leaps far enough in one step to escape shallow optima while staying tied to the equivalence-class structure a correctness argument would need. The handle is Chickering's covered edges. An edge $j \to k$ is *covered* when $j$ and $k$ share all their other parents, $\mathrm{Pa}(j) = \mathrm{Pa}(k) \setminus \{j\}$; reversing a covered edge keeps the DAG in the same MEC, and any two Markov-equivalent DAGs differing in $k$ orientations are linked by exactly $k$ covered-edge reversals. So a search whose moves are covered-edge reversals can traverse an entire equivalence class — the navigation a correctness proof rides.

The clean realization performs a covered-edge reversal *entirely in permutation space*, with no detour through the DAG. Take $\pi$ and a covered edge $j \to k$ in $G_\pi$, and write $\pi = \langle \delta_1, j, \delta_2, k, \delta_3 \rangle$. I want a new permutation whose induced DAG is $G_\pi$ with $j \to k$ reversed. Slide $k$ left past each vertex of $\delta_2$: sliding past $i$ is an adjacent transposition that leaves $G_\pi$ unchanged exactly when $i$ is not a parent of $k$. Here "covered" earns its keep — if $j \to k$ is covered, $k$'s parents are $j$ plus $j$'s parents, all of which precede $j$, so no vertex of $\delta_2$ is a parent of $k$. So I can slide $k$ all the way to just after $j$, every step DAG-preserving, and the single adjacent transposition swapping the now-adjacent $j$ and $k$ flips $j \to k$ to $k \to j$ and, because reversing a covered edge stays in the MEC, lands the DAG with exactly that one arrow reversed. That operation is the **tuck**: in general, split $\delta_2$ into $\gamma$ (the ancestors of $k$, which must travel with $k$) and $\gamma^c$ (the rest), and $\mathrm{tuck}(\pi, j, k) = \langle \delta_1, \gamma, k, j, \gamma^c, \delta_3 \rangle$. A single tuck can fuse a reversal with edge deletions, so it is strictly more efficient than a Chickering DAG-walk, and it never increases the edge count nor loses an independence.

The search is a depth-first traversal driven by tucks. From an initial permutation I scan candidate edges and tuck them. A tuck that *strictly improves* the score is accepted and the search restarts from the improved order; a tuck that is *score-neutral* (a within-MEC reversal) is explored one level deeper, because a neutral move can set up a later strictly improving one — this is the relaxation that lets the search cross plateaus that trap a pure hill-climber. At the DFS root I may tuck *any* parent edge of a variable; deeper in the recursion I restrict to *covered* tucks only, which keep me inside the current MEC while I look for an exit, and a flip-set `history` of reversed pairs ensures the within-MEC wandering cannot loop forever. The depth bound matters: unbounded DFS carries the full correctness theorem, but in finite samples a shallow `depth = 3` captures essentially all the benefit cheaply, so that is what I run.

What makes the per-tuck re-scoring affordable is the Grow-Shrink Tree. A tuck only perturbs a contiguous block of the permutation — the vertices between positions $j$ and $i$ — so only those families change which predecessors they see; everything else is untouched. I cache each vertex's grow/shrink work in a `GST` keyed by the available predecessors, and after a tuck I re-derive just the affected block against the cached trie rather than rerunning grow-shrink from scratch — one `GST` per variable, built once over the score. The order object carries the working permutation, each vertex's parents, its cached local score, and a running edge count; its `__init__` seeds each vertex from `score.score(y, [])`, and here a `causal-learn` convention is worth naming because it is in the literal fill: that setup score is *negated* (`-score.score(y, [])`), while the values that actually drive grow, shrink, and the DFS accept/reject are the higher-is-better numbers returned by `GST.trace(...)`. I keep that exactly as the harness exposes it; the negation is a bookkeeping quirk of the initializer, not the scoring used for moves. When the DFS converges — a full pass with no improving or productive-neutral tuck — I read the DAG off the final order and convert it with `dag2cpdag`, since the data identify only the MEC. Relative to the general method, there is no explicit Backward-Equivalence cleanup phase bolted on and no tier-by-tier escalation knob: the within-MEC/covered-tuck structure of the DFS is doing the work the tier relaxation would do, and I accept the lean fill as the operational form.

The delta from PC is total. Instead of thousands of independent thresholded Fisher-z verdicts, every decision is now a comparison of decomposable model scores; instead of a fixed skeleton that an early wrong deletion poisons, the search is a plateau-crossing DFS of covered/general tucks over orderings, robust under an assumption weaker than the faithfulness PC's recall violated. The falsifiable claims are on recall and on the dense scenarios: the two small sparse graphs, where PC was already near-perfect on precision, should snap to essentially zero SHD; the dense and noisy rows where PC collapsed — SF50, SF50-Hard, and especially ER20-Noisy (PC adjacency recall 0.246, SHD 69) — are the real test, and I expect adjacency recall to lift sharply there. Where I am not sure GRaSP wins cleanly is the very densest, lowest-sample regime: ER20-Noisy's 400 samples at noise 2.5 strain even a robust score, and the tuck-DFS's plateau-crossing can wander on a graph where the sparse-Markov razor is near its own limit. If anything survives as the next failure mode, I expect it to be that row, not the scale-free ones.

```python
class _GraspOrder:
    def __init__(self, p, score):
        self.order = list(range(p))
        self.parents = {}
        self.local_scores = {}
        self.edges = 0
        import random
        random.shuffle(self.order)
        for i in range(p):
            y = self.order[i]
            self.parents[y] = []
            self.local_scores[y] = -score.score(y, [])

    def get(self, i): return self.order[i]
    def set(self, i, y): self.order[i] = y
    def index(self, y): return self.order.index(y)
    def insert(self, i, y): self.order.insert(i, y)
    def pop(self, i=-1): return self.order.pop(i)
    def get_parents(self, y): return self.parents[y]
    def set_parents(self, y, yp): self.parents[y] = yp
    def get_local_score(self, y): return self.local_scores[y]
    def set_local_score(self, y, s): self.local_scores[y] = s
    def get_edges(self): return self.edges
    def set_edges(self, e): self.edges = e
    def bump_edges(self, b): self.edges += b
    def len(self): return len(self.order)


def _grasp_get_ancestors(y, ancestors, order):
    ancestors.append(y)
    for x in order.get_parents(y):
        if x not in ancestors:
            _grasp_get_ancestors(x, ancestors, order)


def _grasp_tuck(i, j, order):
    ancestors = []
    _grasp_get_ancestors(order.get(i), ancestors, order)
    shift = 0
    for k in range(j + 1, i + 1):
        if order.get(k) in ancestors:
            order.insert(j + shift, order.pop(k))
            shift += 1


def _grasp_update(i, j, order, gsts):
    edge_bump = 0
    old_score = 0
    new_score = 0
    for k in range(j, i + 1):
        z = order.get(k)
        z_parents = order.get_parents(z)
        edge_bump -= len(z_parents)
        old_score += order.get_local_score(z)
        z_parents.clear()
        candidates = [order.get(l) for l in range(0, k)]
        local_score = gsts[z].trace(candidates, z_parents)
        order.set_local_score(z, local_score)
        edge_bump += len(z_parents)
        new_score += local_score
    return edge_bump, new_score - old_score


def _grasp_dfs(depth, flipped, history, order, gsts):
    import random
    cache = [{}, {}, {}, 0]
    indices = list(range(order.len()))
    random.shuffle(indices)
    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y)
        random.shuffle(y_parents)
        for x in y_parents:
            covered = set([x] + order.get_parents(x)) == set(y_parents)
            if len(history) > 0 and not covered:
                continue
            j = order.index(x)
            for k in range(j, i + 1):
                z = order.get(k)
                cache[0][k] = z
                cache[1][k] = order.get_parents(z)[:]
                cache[2][k] = order.get_local_score(z)
            cache[3] = order.get_edges()
            _grasp_tuck(i, j, order)
            edge_bump, score_bump = _grasp_update(i, j, order, gsts)
            if score_bump > 1e-6:
                order.bump_edges(edge_bump)
                return True
            if score_bump > -1e-6:
                flipped = flipped ^ set(
                    [
                        tuple(sorted([x, z]))
                        for z in order.get_parents(x)
                        if order.index(z) < i
                    ]
                )
                if len(flipped) > 0 and flipped not in history:
                    history.append(flipped)
                    if depth > 0 and _grasp_dfs(depth - 1, flipped, history, order, gsts):
                        return True
                    del history[-1]
            for k in range(j, i + 1):
                z = cache[0][k]
                order.set(k, z)
                order.set_parents(z, cache[1][k])
                order.set_local_score(z, cache[2][k])
            order.set_edges(cache[3])
    return False


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov
    from causallearn.search.PermutationBased.gst import GST
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    X = X.copy()
    n, p = X.shape
    depth = 3
    parameters = {"lambda_value": 2}
    score = LocalScoreClass(
        data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
    )
    gsts = [GST(i, score) for i in range(p)]

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(p)]
    G = GeneralGraph(nodes)

    order = _GraspOrder(p, score)

    for i in range(p):
        y = order.get(i)
        y_parents = order.get_parents(y)
        candidates = [order.get(j) for j in range(0, i)]
        local_score = gsts[y].trace(candidates, y_parents)
        order.set_local_score(y, local_score)
        order.bump_edges(len(y_parents))

    while _grasp_dfs(depth - 1, set(), [], order, gsts):
        pass

    for y in range(p):
        for x in order.get_parents(y):
            G.add_directed_edge(nodes[x], nodes[y])

    G = dag2cpdag(G)
    return G
```
