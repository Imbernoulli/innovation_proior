**Problem.** Every prior rung's forward/skeleton phase is a heuristic that can land in the wrong basin
on a given network or seed — BOSS's best-move sweep wobbles seed-to-seed (Alarm 16/24/22, SHD 20.7) and
no rung is uniformly good. Use the search whose forward *and* backward phases carry an outright
large-sample optimality guarantee.

**Key idea (GES — Greedy Equivalence Search).** Same decomposable, score-equivalent BDeu score, now read
through **consistency**: in the limit, adding `X_i → X_j` raises the score iff `X_j` depends on `X_i`
given its current parents (local consistency) — a model-score comparison, not a brittle CI test. Search
**equivalence classes** (completed PDAGs, removing covered-reversal redundancy) in two phases. **Forward:**
from the empty graph, greedily apply the highest-scoring `Insert(X,Y,T)` until none improves — the local
max provably contains the true distribution (composition axiom + local consistency). **Backward:**
greedily apply the highest-scoring `Delete(X,Y,H)` until none improves — provably descends to the true
class (Meek's conjecture: covered reversals + additions bottom out at the perfect map). Operators are
validity-tested and scored by single node-family differences on a witness DAG, so no class member is ever
enumerated; the PDAG is reconverted only on an accepted move. Output is the CPDAG.

**Why it is the strongest baseline.** Unconditional large-sample optimality under faithfulness, no
ordering, no depth knob, no sweep order to depend on — so it removes the per-seed heuristic uncertainty
BOSS exposed, winning where a guaranteed low-variance search beats a heuristic one. Its residual weakness
is a *greedy forward phase* that can over-add adjacencies on the densest, near-unfaithful networks.

**Hyperparameters.** `score_func = "local_score_BDeu"`; `sample_prior = 1.0`, `structure_prior = 1.0`.
Forward (FES) then backward (BES); empty-graph start; library defaults for the operators/conversions.

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.search.ScoreBased.GES import ges

    result = ges(
        X,
        score_func="local_score_BDeu",
        parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    )
    return result["G"]
```
