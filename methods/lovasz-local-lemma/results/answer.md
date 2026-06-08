# The Lovász Local Lemma

## Problem it solves

Prove that an object avoiding a family of "bad" events `A_1, …, A_n` exists — i.e. that
`P(Ā_1 … Ā_n) > 0` — in the regime where the probabilistic method's two easy modes both fail: the events
are **not** independent (so `∏(1 − P(A_i))` is unavailable) and their probabilities **sum past one** (so the
union bound `Σ P(A_i) < 1` is hopeless). The saving structure is that the dependence is **local**: each `A_i`
is independent of all but a few of the other events.

## Key idea

Stop bounding the raw `P(A_i)` and instead bound the **conditional** probability of a bad event given that
some of the others were avoided — the exact factor in the chain-rule expansion
`P(Ā_1 … Ā_n) = ∏_k (1 − P(A_k | Ā_1 … Ā_{k−1}))`. Prove, by induction on the size of the conditioning set,
that this conditional stays below a per-event threshold `x_i`. Splitting the conditioning into an event's
neighbours and its far events lets **independence collapse the numerator** to `P(A_i)` and lets the **chain
rule telescope the denominator** into a product over neighbours. Local dependence is then enough for positive
probability, even though the union bound dies.

## Dependency graph (the right notion of "local")

`A_i` is *independent of a set* `B_1, …, B_m` if `P(A_i | any Boolean combination of the B_j) = P(A_i)` — this
is **mutual** independence, strictly stronger than pairwise (e.g. `A_i = "x_{i+1}+x_{i+2}=0"` for independent
bits `x_1,x_2,x_3 ∈ Z/2Z` are pairwise independent but not mutually). A graph `G` on the events is a
*dependency graph* if each `A_i` is independent of the set of its non-neighbours.

## General (asymmetric) form

> Let `A_1, …, A_n` have dependency graph `G` with neighbourhoods `N(i)`. If there exist `x_i ∈ [0,1)` with
> `P(A_i) ≤ x_i ∏_{j∈N(i)} (1 − x_j)` for every `i`, then `P(Ā_1 … Ā_n) ≥ ∏_i (1 − x_i) > 0`.

**Proof.** Show `P(A_i | ∩_{j∈S} Ā_j) ≤ x_i` for all `i ∉ S`, by induction on `|S|`.
*Base* `S = ∅`: `P(A_i) ≤ x_i ∏_{j∈N(i)}(1 − x_j) ≤ x_i`.
*Step*: split `S = S_1 ⊔ S_2` with `S_1 = S ∩ N(i)`, `S_2 = S \ S_1`; then
`P(A_i | ∩_{S} Ā_j) = P(A_i ∩_{S_1} Ā_j | ∩_{S_2} Ā_l) / P(∩_{S_1} Ā_j | ∩_{S_2} Ā_l)`.
Numerator `≤ P(A_i | ∩_{S_2} Ā_l) = P(A_i) ≤ x_i ∏_{j∈N(i)}(1 − x_j)` (independence from the far set `S_2`).
Denominator: with `S_1 = {j_1,…,j_r}`, telescope by the chain rule into factors
`P(Ā_{j_t} | Ā_{j_1} … Ā_{j_{t-1}} ∩ ∩_{S_2} Ā_l)`. Each conditioning set has size `< |S|`, so each factor is
`1 − P(A_{j_t} | …) ≥ 1 − x_{j_t}` by induction, and the denominator `≥ ∏_{t}(1 − x_{j_t}) ≥ ∏_{j∈N(i)}(1 − x_j)`.
Ratio `≤ x_i`. Finally `P(Ā_1 … Ā_n) = ∏_k (1 − P(A_k | Ā_1 … Ā_{k−1})) ≥ ∏_k (1 − x_k) > 0`. ∎

## Symmetric form

> If `P(A_i) ≤ p` for all `i`, each `A_i` is independent of all but at most `d` of the others, and
> `e · p · (d + 1) ≤ 1`, then `P(Ā_1 … Ā_n) > 0`.

**Proof.** For `d = 0`, mutual independence gives positivity directly when `p < 1`; otherwise apply the general form with `x_i = 1/(d+1)`. Then
`x_i ∏_{j∈N(i)}(1 − x_j) ≥ (1/(d+1))(1 − 1/(d+1))^d ≥ 1/(e(d+1))`, using `(1 − 1/(d+1))^d ≥ 1/e` (the
decreasing bound `(1 − 1/m)^{m−1} ≥ 1/e` with `m = d+1`). So `e p (d+1) ≤ 1 ⟹ p ≤ 1/(e(d+1))` gives the
hypothesis. The `1/(d+1)` is the maximiser of `x(1−x)^d`; the exact finite value is
`(1/(d+1))(d/(d+1))^d`, and the `e` form is its clean universal lower-bound simplification. ∎

**Variant (different-probability events).** If `P(A_i) < 1/2` and `Σ_{j∈N(i)} P(A_j) ≤ 1/4` for all `i`, take
`x_i = 2P(A_i)`; then `x_i ∏(1 − x_j) ≥ 2P(A_i)(1 − Σ_{j∈N(i)} 2P(A_j)) ≥ 2P(A_i)·(1/2) = P(A_i)`, so
`P(Ā_1 … Ā_n) > 0`.

## Applications

- **Hypergraph 2-colouring (property B).** `k`-uniform hypergraph; colour each vertex red/blue by an
  independent fair coin; `A_f = "edge f monochromatic"`, `P(A_f) = 2^{-(k-1)}`. `A_f` depends only on edges
  meeting `f`. If each edge meets at most `d` others and `e · 2^{-(k-1)} · (d+1) ≤ 1` (i.e. `d+1 ≤ 2^{k-1}/e`),
  the hypergraph is 2-colourable — for **arbitrarily many edges**, replacing the union bound's global cap of
  `2^{k-1}` total edges with a local degree bound. (Corollary: every `k`-uniform `k`-regular hypergraph,
  `d ≤ k(k−1)`, is 2-colourable for `k ≥ 9`.)
- **`k`-SAT.** Uniform random assignment; `A_i = "clause i violated"`, `P(A_i) = 2^{-k}`. Clauses depend only
  when they share a variable. If each clause shares variables with at most `d` others and `d+1 ≤ 2^k/e`, a satisfying
  assignment exists — regardless of the total number of clauses.

## Certificate helper

```python
from math import e, prod

def general_lll_certificate(p, neighbours, x):
    """Certify P(A_i) <= x_i * prod_{j in N(i)} (1 - x_j) for all i.
    True => P(no bad event) >= prod_i (1 - x_i) > 0."""
    for i in range(len(p)):
        rhs = x[i] * prod(1.0 - x[j] for j in neighbours[i])
        if not (0.0 <= x[i] < 1.0 and p[i] <= rhs + 1e-12):
            return False
    return True

def symmetric_lll(p, d):
    """All P(A_i) <= p, each independent of all but <= d others.
    True (e*p*(d+1) <= 1) => a good object provably exists."""
    if d == 0:
        return p < 1.0
    return e * p * (d + 1) <= 1.0

def hypergraph_two_colourable(k, max_edge_intersections):
    # A_f = "edge f monochromatic", P = 2^{-(k-1)}; depends only on meeting edges.
    return symmetric_lll(2.0 ** (-(k - 1)), max_edge_intersections)

def k_regular_two_colourable(k):
    return hypergraph_two_colourable(k, k * (k - 1))   # True for k >= 9

def k_sat_satisfiable(k, max_shared_clauses):
    # A_i = "clause i violated", P = 2^{-k}; depends only on variable-sharing clauses.
    return symmetric_lll(2.0 ** (-k), max_shared_clauses)
```
