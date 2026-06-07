# The Algorithmic Lovász Local Lemma (Moser–Tardos)

## Problem

The Lovász Local Lemma says: given a finite family of "bad" events A, each event A determined by a subset vbl(A) of mutually independent random variables, with dependency neighborhood Γ(A) = {B ≠ A : vbl(A) ∩ vbl(B) ≠ ∅}, if there are reals x : A → (0,1) with

    Pr[A] ≤ x(A) · ∏_{B ∈ Γ(A)} (1 − x(B))   for all A,

then Pr[no event occurs] ≥ ∏(1 − x(A)) > 0 — a "good" point exists. But that probability is typically exponentially small, so the existence proof gives no efficient way to *find* the good point. The task is to construct it efficiently, ideally under the same condition.

## Key idea

Do the obvious thing and prove it terminates fast:

> Start from a uniformly random assignment. While some event is violated, pick any violated event A and **resample** vbl(A) — draw fresh independent values for exactly those variables, leaving all others fixed. Repeat until nothing is violated.

Resampling A only changes events in Γ⁺(A). In the recursive SAT view, a returned local correction leaves every clause touched by its recursion tree satisfied and introduces no new violated clauses; long corrections are ruled out by counting the witnesses they would force, not by a monotone potential.

Why it is fast — two layers of analysis:
- **Entropy compression / incompressibility (SAT counting).** With |Γ⁺(C)| ≤ d = 2^{k−5}, a size-u composite witness has at most m(4ed)^u possible encodings, and 4ed = e·2^{k−3} < 2^{k−1}, so this is below m·2^{u(k−1)}. A fixed size-u witness is consistent with the random table with probability 2^{−ku}, so E[X_u] ≤ m·2^{−u}; summing u ≥ log m + 2, with log base 2, gives at most 1/2. A long local correction would force one of these large consistent witnesses, so abort-and-restart succeeds in expected polynomial time.
- **Witness trees + coupling + branching process (the tight argument).** For each resample step build a proper witness tree τ (children labelled from Γ⁺ of the parent, each new vertex attached to the deepest eligible vertex, so every level is an independent set in the dependency graph). A coupling against the fixed random source gives Pr[τ occurs] ≤ ∏_{v} Pr[[v]] ≤ ∏_v x'([v]), where x'(B) := x(B) ∏_{C∈Γ(B)}(1 − x(C)). A Galton–Watson process that adds each child B of v with probability x(B) produces τ with probability p_τ = ((1−x(A))/x(A)) ∏_v x'([v]); since Σ_τ p_τ ≤ 1,

      E[#resamples of A] = E[N_A] ≤ Σ_τ ∏_v x'([v]) = (x(A)/(1−x(A))) Σ_τ p_τ ≤ x(A)/(1−x(A)).

  Total expected resamples ≤ Σ_A x(A)/(1−x(A)). This holds under the *exact* existential LLL condition — no constant lost.

Extensions from the same machinery: a parallel version resampling a maximal independent set of violated events each round terminates in O((1/ε) log Σ x/(1−x)) rounds under Pr[A] ≤ (1−ε)x'(A); a deterministic version with bounded dependency degree enumerates the polynomially-many relevant witness trees and uses conditional expectations to fix a good random table.

## Algorithm (general form)

```
function sequential_lll(P, A):
    for all P in P:  v_P <- random sample of P
    while some A in A is violated under (v_P):
        pick any violated A
        for all P in vbl(A):  v_P <- fresh random sample of P
    return (v_P)
```

## Working code (k-SAT specialization)

```python
import random

def is_clause_violated(clause, assignment):
    # clause: list of (var, sign) literals; satisfied iff some literal is true
    for var, sign in clause:
        if assignment[var] == sign:
            return False
    return True

def violated_clauses(clauses, assignment):
    return [i for i, c in enumerate(clauses) if is_clause_violated(c, assignment)]

def random_assignment(n_vars, rng):
    # one fair coin per variable: a uniform point of the probability space
    return [rng.random() < 0.5 for _ in range(n_vars)]

def search_good_assignment(clauses, n_vars, rng=None, max_resamples=None):
    rng = rng or random.Random()
    assignment = random_assignment(n_vars, rng)
    resamples = 0
    bad = violated_clauses(clauses, assignment)
    while bad:
        i = bad[0]  # any violated clause; the bound is selection-independent
        for var, _sign in clauses[i]:
            # resample exactly vbl(A), leaving every other variable untouched
            assignment[var] = rng.random() < 0.5
        resamples += 1
        if max_resamples is not None and resamples > max_resamples:
            raise RuntimeError("resample budget exhausted")
        bad = violated_clauses(clauses, assignment)
    return assignment, resamples
```

Under the symmetric SAT condition, each k-clause is violated with probability 2^{−k}; when the LLL criterion is satisfied, the loop finds a satisfying assignment with expected resamples bounded by Σ_A x(A)/(1−x(A)).
