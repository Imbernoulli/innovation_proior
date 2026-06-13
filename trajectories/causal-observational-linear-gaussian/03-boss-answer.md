**Problem.** Recover the CPDAG of a linear Gaussian SEM. Keep GRaSP's robust permutation/score view
(no brittle CI tests) but drop its tuck-DFS — whose depth knobs and within-MEC excursions left SF50
arrows soft and let ER20-Noisy wander into a denser-than-truth optimum — for a parameter-light,
strictly-improving search move.

**Key idea.** BOSS (Best Order Score Search). Same order-to-DAG projection as GRaSP: each variable's
parents are its score-selected Markov boundary among its predecessors (grow-shrink). The move is the
**best-position mutation**: pull one variable out of the order and reinsert it at the single slot
that maximizes the total score, computed for all `p + 1` slots in two linear passes over the order
(a forward pass accumulating the moved variable's score plus the prefix's running score, a backward
pass accumulating the suffix's contribution). Sweep this move over all variables in shuffled order
and repeat until a full sweep makes no move; then read off the DAG and convert to a CPDAG.

**Why this fill.** This is the *linear Gaussian* task, so the decomposable score is again
`local_score_BIC_from_cov` with `lambda_value = 2` — *identical to GRaSP's score*, so any gain is
attributable to the search move, not the criterion. The per-variable Grow-Shrink Tree (`GST`) caches
grow/shrink traces so each best-position evaluation is a cached lookup. The move is strictly greedy
(accept only an improvement beyond tolerance `1e-6`), so unlike GRaSP's plateau-crossing DFS it has
no neutral within-MEC excursion to drift denser on — the fix for the ER20-Noisy over-adding. This
fill is the *lean* BOSS: best-position sweep then `dag2cpdag`, with **no** optional
Backward-Equivalence-Search phase.

**Hyperparameters.** `lambda_value = 2`; improvement tolerance `1e-6`; identity initial order `[0,
1, …, p-1]`; variables swept in shuffled order until a full pass moves nothing; output via
`dag2cpdag`.

```python
def _boss_reversed_enumerate(iter_, j):
    for w in reversed(iter_):
        yield j, w
        j -= 1


def _boss_better_mutation(v, order, gsts):
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
    for j, w in _boss_reversed_enumerate(order, p - 1):
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


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    import random
    import sys
    import time
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov
    from causallearn.search.PermutationBased.gst import GST
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    X = X.copy()
    n, p = X.shape
    parameters = {"lambda_value": 2}
    score = LocalScoreClass(
        data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
    )

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(p)]
    G = GeneralGraph(nodes)

    order = list(range(p))
    gsts = [GST(v, score) for v in order]
    parents = {v: [] for v in order}

    variables = list(order)
    while True:
        improved = False
        random.shuffle(variables)
        for v in variables:
            improved |= _boss_better_mutation(v, order, gsts)
        if not improved:
            break

    for i, v in enumerate(order):
        parents[v].clear()
        gsts[v].trace(order[:i], parents[v])

    for y in range(p):
        for x in parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])

    G = dag2cpdag(G)
    return G
```
