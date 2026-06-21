# The Erdős–Ko–Rado theorem, via the cycle method

## Problem

Among families `F` of `k`-element subsets of `[n] = {1,…,n}` that are **intersecting** (`A ∩ B ≠ ∅` for all `A, B ∈ F`), how large can `|F|` be, and which families are extremal? The question is genuine only for `n ≥ 2k` — when `n < 2k` every two `k`-subsets already meet, so all `\binom{n}{k}` sets form an intersecting family.

## Key idea

The conjectured optimum `\binom{n-1}{k-1}` equals exactly `\tfrac{k}{n}\binom{n}{k}`, a `k/n` share of all `k`-sets — an averaging statement waiting to be made. Arrange `[n]` on a **circle**; call a block of `k` consecutive points an **arc**. The circle is symmetric (rotations and relabelings permute its `n` arcs), and intersection becomes a one-dimensional overlap condition. Two facts then close the problem:

1. **Local bound.** On any single circular arrangement, an intersecting family contains at most `k` of the `n` arcs.
2. **Averaging.** Each `k`-set is an arc in a `k/n` fraction of circular arrangements, so averaging the local bound `≤ k` over all arrangements turns it into the global bound `|F| ≤ \tfrac{k}{n}\binom{n}{k} = \binom{n-1}{k-1}`.

The hypothesis `n ≥ 2k` is not incidental: it is exactly the condition under which the two arcs flanking a chosen arc are disjoint, which is what the local bound rests on.

## Theorem

Let `1 ≤ k` and `n ≥ 2k`, and let `F` be an intersecting family of `k`-element subsets of `[n]`. Then
```
|F| ≤ C(n-1, k-1).
```
- The **star** `S_x = { A ∈ C([n],k) : x ∈ A }` is intersecting and has `|S_x| = C(n-1, k-1)`, so the bound is tight.
- If `n > 2k`, equality holds **only** for stars: every extremal `F` equals some `S_x`.
- If `n = 2k`, the extremal families are exactly the **transversals of the complementary-pair matching** — choose one set out of each disjoint pair `{A, [n]\A}`, of which there are `\tfrac12\binom{2k}{k} = \binom{2k-1}{k-1}`. Stars are special transversals, but most transversals are not stars.

## Proof of the bound

**Local lemma — at most `k` arcs.** Fix a circular arrangement of `[n]`; its arcs are the `n` blocks of `k` consecutive points. If no arc lies in `F`, the count is `0 ≤ k`. Otherwise fix an arc `A = \{a_1,…,a_k\}` (consecutive) in `F`. Any other arc `B ∈ F` satisfies `B ∩ A ≠ ∅` and `B ≠ A`, so `B` slides off `A` at one end: for some internal gap `i ∈ \{1,…,k-1\}`, `B` contains exactly one of the adjacent pair `a_i, a_{i+1}`. Exactly two arcs separate a given gap `i`: the arc `L_i` ending at `a_i` and the arc `R_i` starting at `a_{i+1}`. Each has length `k`, so together they occupy `2k` points; since `n ≥ 2k`, `L_i` and `R_i` are **disjoint**, and an intersecting `F` contains at most one of them. Summing over the `k-1` gaps gives at most `k-1` arcs besides `A`, hence at most `k` arcs in total. (The bound is tight: `k` consecutive arcs through one common point pairwise intersect.)

**Global averaging.** Choose a circular arrangement uniformly at random. A fixed `k`-set `A` is an arc with probability
```
P(A is an arc) = (n · k! · (n-k)!) / n! = n / C(n,k),
```
so by linearity `E[ #{A ∈ F : A is an arc} ] = |F| · n / C(n,k)`. By the local lemma this expectation is `≤ k`, whence
```
|F| · n / C(n,k) ≤ k   ⟹   |F| ≤ (k/n) · C(n,k) = C(n-1, k-1).   ∎
```
Equivalently, as a pure double count over the `(n-1)!` circular arrangements (each `k`-set is an arc in `k!(n-k)!` of them): `|F| · k!(n-k)! ≤ k · (n-1)!`, giving the same `|F| ≤ C(n-1, k-1)`.

## Proof of uniqueness (`n > 2k`)

Let `F` be intersecting with `|F| = C(n-1,k-1)`; suppose `F` is not a star.

- *No saturated direction.* If for some `x, y` every `k`-set containing `x` but not `y` were in `F`, then `F` would be the full star `S_x`: for any `K ∌ x`, since `n > 2k` there are `n-1-k ≥ k` points besides `y` outside `K`, so a `k`-set `L ∋ x`, `y ∉ L`, `L ∩ K = ∅` exists; `L ∈ F` forces `K ∉ F`, so `F ⊆ S_x`, and by cardinality `F = S_x`. Contradiction. Hence for every `x, y` some `k`-set with `x`, without `y`, is missing from `F`.
- *A boundary swap.* There exist `K, K'` with `|K ∩ K'| = k-1`, `K ∈ F`, `K' ∉ F` (else single-element swaps, which connect all of `C([n],k)`, would force `F` to be everything — not intersecting).
- *The contradiction.* Label `K \ K' = \{0\}`, `K' \ K = \{k\}`; pick (by the first point) `K'' ∉ F` with `0 ∈ K''`, `k ∉ K''`. Choose a circular arrangement making `K = [0,k-1]` an arc, with element `k` at position `k` (so the arc `[1,k]` equals `K' ∉ F`) and `K''` realized as the arc on the other side of `K`. The local lemma's equality case forces the `≤ k` arcs of `F` to be `k` *consecutive* arcs through `K`; but each of the `k` such runs contains either `[1,k] = K'` or `K''`, both `∉ F`, so no run survives inside `F` — contradiction. So `F` must be a star.

At `n = 2k` the slack `n-1-k ≥ k` becomes equality and fails, which is exactly why the complementary-pair transversals tie the bound there without being stars.

## Optional numerical sanity check

Pure combinatorics; no implementation is needed for the result. The two load-bearing identities and the bound on small parameters can be confirmed directly:

```python
from itertools import combinations
from math import comb, factorial

def k_subsets(n, k):
    return [frozenset(s) for s in combinations(range(n), k)]

def is_intersecting(family):
    fam = list(family)
    return all(fam[i] & fam[j]
               for i in range(len(fam)) for j in range(i + 1, len(fam)))

def star(n, k, x=0):
    return [s for s in k_subsets(n, k) if x in s]

# Identity 1: the bound is the k/n share of all k-sets.
def check_bound_identity(n, k):
    return comb(n - 1, k - 1) * n == k * comb(n, k)

# Identity 2: a fixed k-set is an arc with probability n / C(n,k).
def check_arc_probability(n, k):
    favorable = n * factorial(k) * factorial(n - k)   # n*k!*(n-k)!
    from fractions import Fraction
    return Fraction(favorable, factorial(n)) == Fraction(n, comb(n, k))

# Brute-force largest intersecting family for tiny (n, k), compared to bound.
def brute_force_max(n, k):
    sets = k_subsets(n, k)
    best = 0
    # exact search via a simple branch over sets, feasible only for small n,k
    def extend(idx, chosen):
        nonlocal best
        best = max(best, len(chosen))
        if len(chosen) + (len(sets) - idx) <= best:
            return
        for j in range(idx, len(sets)):
            if all(sets[j] & c for c in chosen):
                chosen.append(sets[j]); extend(j + 1, chosen); chosen.pop()
    extend(0, [])
    return best

if __name__ == "__main__":
    for (n, k) in [(4, 2), (5, 2), (6, 3)]:   # last is the n = 2k boundary
        assert check_bound_identity(n, k)
        assert check_arc_probability(n, k)
        assert len(star(n, k)) == comb(n - 1, k - 1)
        assert is_intersecting(star(n, k))
        assert brute_force_max(n, k) == comb(n - 1, k - 1)
    print("all checks pass")
```
