# Greedy Maximization of Submodular Functions

## Problem

Maximize a non-negative, monotone, submodular set function f: 2^V -> R>=0
subject to a cardinality constraint |S| <= k. This is NP-hard (it generalizes
maximum coverage), so we seek a polynomial-time algorithm with a provable
worst-case ratio against f(OPT).

- **Monotone:** A contained in B implies f(A) <= f(B).
- **Submodular (diminishing returns):** for A contained in B, e not in B,
  f(A + e) - f(A) >= f(B + e) - f(B). The marginal gain f(e | S) = f(S+e) - f(S)
  is non-increasing as S grows.

## Key idea

Grow S one element at a time, each round adding the element of **maximum marginal
gain**:

    S <- empty
    repeat k times:  S <- S + argmax_{e not in S} [ f(S + e) - f(S) ]

That single rule -- argmax marginal gain -- is the whole method.

## Guarantee

For k = 0 the claim is trivial, so assume k >= 1. Let S_i be the greedy set after
i additions, with S_0 = empty, and let O = OPT, |O| <= k. If O is empty,
monotonicity makes the empty set optimal and the guarantee is immediate.
Otherwise, the greedy choice beats adding any single o in O. Averaging over O and
then weakening the denominator to k is safe because all marginals are
non-negative:

    f(S_{i+1}) - f(S_i) >= (1/k) sum_{o in O} [ f(S_i + o) - f(S_i) ]
                         >= (1/k) [ f(S_i ∪ O) - f(S_i) ]    (submodularity)
                         >= (1/k) [ f(OPT) - f(S_i) ]        (monotonicity).

The submodularity step is the telescoping one: order O as o_1, ..., o_r and write
f(S_i ∪ O) - f(S_i) as the sum of the marginal of o_j after S_i plus the earlier
o's; diminishing returns makes each such marginal no larger than
f(S_i + o_j) - f(S_i).

With gap a_i = f(OPT) - f(S_i) this is a_{i+1} <= (1 - 1/k) a_i, so after k rounds
a_k <= (1 - 1/k)^k a_0. For k = 1 the residue is zero; for k > 1, the
concavity/tangent inequality log(1 - x) <= -x gives (1 - 1/k)^k <= 1/e. Since
a_0 <= f(OPT), this gives

    f(S_greedy) >= [ 1 - (1 - 1/k)^k ] f(OPT) >= (1 - 1/e) f(OPT) ≈ 0.632 f(OPT).

## Tightness

The proof's geometric loss is real for greedy. Split a universe into k blocks
B_1, ..., B_k of size N = k^k. The optimal sets are O_i = B_i. In every block,
carve disjoint slices P_{i,t} of size (N/k)(1 - 1/k)^{t-1}, for t = 1..k, leaving
a remainder R_i of size N(1 - 1/k)^k, and add cross-block sets
G_t = union_i P_{i,t}. Initially every O_i and G_1 has marginal N; after
G_1, ..., G_{t-1} have been selected, every O_i has marginal
N(1 - 1/k)^{t-1} from its future slices plus R_i, and G_t has the same marginal
from one fresh slice in each block. A tie-breaking greedy can therefore select
G_1, ..., G_k, covering

    kN [1 - (1 - 1/k)^k],

while OPT selects O_1, ..., O_k and covers kN. The ratio is exactly
1 - (1 - 1/k)^k, tending to 1 - 1/e; arbitrarily small perturbations remove the
ties without changing the limiting ratio.

The cardinality constraint is the uniform matroid, the favorable special case in
which the averaging step is cleanest. Under a general matroid, feasible exchanges
replace the simple average over k OPT elements, and the same plain greedy rule
guarantees 1/2 instead of 1 - 1/e.

## Code

```python
from itertools import combinations
from math import e


class SetFunction:
    """Value oracle over a ground set of element indices."""
    def __init__(self, ground_set):
        self.ground_set = list(ground_set)

    def value(self, S):
        raise NotImplementedError

    def marginal(self, e, S):
        return self.value(S | {e}) - self.value(S)


class Coverage(SetFunction):
    """f(S) = | union of the subsets indexed by S |."""
    def __init__(self, sets):
        super().__init__(range(len(sets)))
        self.sets = [set(s) for s in sets]

    def value(self, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(covered)

    def marginal(self, e, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(self.sets[e] - covered)


def exact_best(f, k):
    """Exhaustive optimum for tiny instances, used only as a checker."""
    best_S, best_value = set(), f.value(set())
    limit = min(k, len(f.ground_set))
    for r in range(limit + 1):
        for combo in combinations(f.ground_set, r):
            S = set(combo)
            value = f.value(S)
            if value > best_value:
                best_S, best_value = S, value
    return best_S, best_value


def select(f, k):
    """Grow S by repeatedly adding the element with maximum marginal gain."""
    if k < 0:
        raise ValueError("k must be non-negative")

    S = set()
    for _ in range(min(k, len(f.ground_set))):
        best_e, best_gain = None, 0
        for e in f.ground_set:
            if e in S:
                continue
            g = f.marginal(e, S)
            if best_e is None or g > best_gain:
                best_e, best_gain = e, g
        if best_e is None or best_gain <= 0:
            break
        S.add(best_e)
    return S


if __name__ == "__main__":
    sets = [
        {1, 2, 3, 8},
        {1, 2, 3, 4, 5},
        {1, 4, 6, 7},
        {5, 6, 7, 8},
        {2, 3, 9},
    ]
    f = Coverage(sets)
    k = 2

    chosen = select(f, k)
    opt, opt_value = exact_best(f, k)
    chosen_value = f.value(chosen)
    ratio = 1.0 if opt_value == 0 else chosen_value / opt_value

    assert chosen_value >= (1 - 1 / e) * opt_value
    print("selected", sorted(chosen), "value", chosen_value)
    print("optimum ", sorted(opt), "value", opt_value)
    print("ratio   ", round(ratio, 3))
```

Running the block prints the selected set, the brute-force optimum, and the
realized ratio; the assertion checks the promised lower bound on that instance.
