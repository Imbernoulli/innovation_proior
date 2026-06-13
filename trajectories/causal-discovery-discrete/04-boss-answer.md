**Problem.** No prior rung is good across the whole size/density range: hc stalls on big graphs, GRaSP
nails medium density (Alarm 3.3) but is a heuristic depth-knobbed tuck DFS that stalls and goes unstable
on the densest networks, PC prunes the skeleton precisely (Win95pts adj-prec 0.93) but over-deletes true
edges (Hailfinder adj-recall 0.52). Keep the score/permutation view's robustness and orientation power,
but replace the fiddly tuck DFS with a clean, parameter-free move plus a guaranteed cleanup.

**Key idea (BOSS — Best Order Score Search).** Same permutation substrate and grow-shrink BDeu projection
as GRaSP. The move: take one variable `v`, pull it out, and insert it at the single **best-scoring
position** among all slots (the argmax over insertion points) — a whole run of adjacent transpositions
collapsed into one decision, big enough to escape shallow optima, with **zero depth parameters**. Sweep
this best-move over every variable (in shuffled order each round) and repeat to convergence; project the
final ordering to a DAG. An optional GES-style backward delete phase secures the large-sample guarantee
for **any** starting permutation, since project always returns a subgraph-minimal DAG containing the
distribution. Cache grow-shrink per variable in a tree (deterministic descent through sorted improving
additions, lazy expansion) so each best-move is a linear sweep of cached traces — the mechanism that
makes the large, dense regime tractable where GRaSP stalled.

**Why over the prior rungs.** Parameter-free leap (no depth knob to stall on) + guaranteed backward
cleanup + cached grow-shrink scalability — so it should finish and explore the 76-node ordering space
where GRaSP stalled, and recover the adjacency recall PC sacrificed by adding edges by score rather than
deleting them by a brittle test. Output DAG converted to a CPDAG.

**The scaffold fill.** A single library call. The harness exposes the BDeu score (`sample_prior = 1`,
`structure_prior = 1`); the best-move sweep, grow-shrink trees, and backward phase are internal — there
is no depth knob to set (that is the point).

**Hyperparameters.** `score_func = "local_score_BDeu"`; `sample_prior = 1.0`, `structure_prior = 1.0`.

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.search.PermutationBased.BOSS import boss

    G = boss(
        X,
        score_func="local_score_BDeu",
        parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    )
    return G
```
