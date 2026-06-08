# The Kraft–McMillan inequality

## Problem

Encode the symbols of a source `X` over `X = {x_1,…,x_m}` as strings over a `D`-ary alphabet so that codewords can be concatenated and the message recovered unambiguously, while making the expected length `L = Σ p_i ℓ_i` small. Two questions: (1) **which length profiles `(ℓ_1,…,ℓ_m)` admit a decodable code at all**, and does that depend on whether we demand instant (prefix-free) decoding or only eventual unique decoding? (2) How small can `L` be?

## Key idea

The achievable lengths are governed by a single budget, `Σ_i D^{−ℓ_i} ≤ 1`. **Kraft**: this is necessary and sufficient for a *prefix (instantaneous)* code — proved by a leaf-counting argument on the `D`-ary code tree. **McMillan**: the *same* inequality is already forced by the much weaker requirement of mere *unique decodability* — proved by raising `Σ D^{−ℓ_i}` to the `k`-th power and counting strings (Karush's argument). The two classes therefore realize exactly the same length profiles, so prefix codes lose nothing. Because `D^{−ℓ_i}` then behaves like a (sub-)probability, the budget converts the coding problem into a constrained minimization whose answer is the entropy: `L ≥ H_D(X)`, achievable to within one digit by `ℓ_i = ⌈log_D(1/p_i)⌉` and to within `1/n` per symbol by blocking — the source-coding theorem.

## The theorems, stated and proved

**Theorem 1 (Kraft).** *For any `D`-ary prefix (instantaneous) code with positive integer codeword lengths `ℓ_1,…,ℓ_m`, `Σ_i D^{−ℓ_i} ≤ 1`. Conversely, any positive integer lengths satisfying this admit a prefix code.*

*Proof.* Put the code on the `D`-ary tree (each node has `D` children; a codeword is a root-to-node path). Prefix-free ⇔ no codeword is an ancestor of another. Let `ℓ_max = max_i ℓ_i`. A codeword at depth `ℓ_i` has exactly `D^{ℓ_max−ℓ_i}` descendants at depth `ℓ_max`, and for distinct codewords these descendant sets are disjoint. The total number of depth-`ℓ_max` nodes is `D^{ℓ_max}`, so `Σ_i D^{ℓ_max−ℓ_i} ≤ D^{ℓ_max}`; divide by `D^{ℓ_max}` to get `Σ_i D^{−ℓ_i} ≤ 1`.

For the converse, sort the lengths `ℓ_1≤⋯≤ℓ_m` and number the depth-`ℓ_max` leaves lexicographically. A depth-`ℓ_j` node is an aligned block of `D^{ℓ_max−ℓ_j}` bottom leaves. Before placing `ℓ_j`, the earlier codewords have consumed `U_{j-1}=Σ_{i<j}D^{ℓ_max−ℓ_i}` leaves; because `ℓ_i≤ℓ_j`, this `U_{j-1}` is a multiple of the current block size. Also `U_{j-1}+D^{ℓ_max−ℓ_j}≤Σ_iD^{ℓ_max−ℓ_i}≤D^{ℓ_max}`. Thus the aligned block starting at `U_{j-1}` exists and is disjoint from all earlier blocks; choosing its depth-`ℓ_j` node gives the next codeword. Iterating builds a prefix code. Equivalently, the same construction lays aligned half-open `D`-adic intervals of lengths `D^{−ℓ_i}` end to end in `[0,1)`. The countably-infinite case follows by the disjoint-interval argument and finite partial constructions.* ∎

**Theorem 2 (McMillan).** *The positive integer codeword lengths of any uniquely decodable `D`-ary stream code satisfy `Σ_i D^{−ℓ_i} ≤ 1`. Conversely, any positive integer lengths satisfying it admit a uniquely decodable (indeed prefix) code.*

*Proof (Karush's `k`-th-power counting).* Let `S = Σ_i D^{−ℓ_i}` and `ℓ(x) = |C(x)|`. A zero-length word would make the extension non-injective, because inserting that source symbol into a sequence would not change the channel string, so `ℓ_min≥1`. For any integer `k ≥ 1`,
```
S^k = (Σ_x D^{−ℓ(x)})^k
    = Σ_{x_1,…,x_k} D^{−(ℓ(x_1)+⋯+ℓ(x_k))}
    = Σ_{m'=k·ℓ_min}^{k·ℓ_max} a(m') D^{−m'},
```
where `a(m')` is the number of source `k`-tuples whose encoding has total length `m'`. By unique decodability the encoding map on length-`k` strings is injective, so these are distinct `D`-ary strings of length `m'`, of which there are only `D^{m'}`; hence `a(m') ≤ D^{m'}`. Therefore
```
S^k ≤ Σ_{m'=k·ℓ_min}^{k·ℓ_max} D^{m'} D^{−m'} = (k(ℓ_max−ℓ_min)+1) ≤ k·ℓ_max,
```
where the last inequality uses `ℓ_min≥1`. Hence `S ≤ (k·ℓ_max)^{1/k}`. Letting `k → ∞` gives `(k·ℓ_max)^{1/k} → 1`, and since `S` is independent of `k`, `S = Σ_i D^{−ℓ_i} ≤ 1`. (Equivalently via the generating function `F(x)=Σ_i x^{−|s_i|}`: the coefficient of `x^{−ℓ}` in `F(x)^k` is `≤ D^ℓ`, so `F(D)^k ≤ k·ℓ_max` in the nontrivial case.) The converse is Theorem 1's construction: lengths satisfying the budget yield a prefix code, which is uniquely decodable. The argument extends to infinite alphabets since every finite sub-collection is uniquely decodable, so every partial sum is `≤ 1`.* ∎

**Consequence.** The achievable length sets of uniquely decodable and of prefix codes coincide — both are exactly `{(ℓ_i): Σ_i D^{−ℓ_i} ≤ 1}`. Instant decodability costs nothing.

## Entropy bound (corollary)

**Lower bound.** For any uniquely decodable `D`-ary code, `L = Σ p_i ℓ_i ≥ H_D(X)`. Writing `r_i = D^{−ℓ_i}/c` with `c = Σ_j D^{−ℓ_j}`:
```
L − H_D(X) = Σ p_i log_D( p_i / D^{−ℓ_i} ) = D(p‖r) + log_D(1/c) ≥ 0,
```
since relative entropy `D(p‖r) ≥ 0` and `c ≤ 1` (the budget) gives `log_D(1/c) ≥ 0`. After restricting to the positive-probability alphabet, equality holds iff `D^{−ℓ_i} = p_i` for every symbol, i.e. the source probabilities are `D`-adic.

**Achievability.** For a non-degenerate source, after restricting to the positive-probability alphabet, the Shannon lengths `ℓ_i = ⌈log_D(1/p_i)⌉` are positive and satisfy the budget (`Σ D^{−ℓ_i} ≤ Σ p_i = 1`), hence are realizable, and from `log_D(1/p_i) ≤ ℓ_i < log_D(1/p_i)+1`,
```
H_D(X) ≤ L < H_D(X) + 1.
```
Blocking `n` i.i.d. symbols gives `H_D(X) ≤ L_n < H_D(X) + 1/n → H_D(X)`; for a stationary source, the same bound with `H_D(X_1,\dots,X_n)/n` gives convergence to the `D`-ary entropy rate `H_D(𝒳)`. Thus `H_D(X)` is both the floor and the asymptotic limit of per-symbol description length in the memoryless case.

## Worked example / verifier

```python
from fractions import Fraction
from math import log, ceil

def kraft_sum(lengths, D):
    # S = sum_i D^{-l_i}: exact rational arithmetic keeps the budget exact.
    if D < 2 or any(not isinstance(l, int) or l <= 0 for l in lengths):
        raise ValueError("use D >= 2 and positive integer lengths")
    return sum(Fraction(1, D ** l) for l in lengths)

def kraft_feasible(lengths, D):
    # Necessary AND sufficient for a decodable D-ary code (prefix == UD here).
    return kraft_sum(lengths, D) <= 1

def base_D_digits(index, length, D):
    digits = [0] * length
    for pos in range(length - 1, -1, -1):
        digits[pos] = index % D
        index //= D
    return tuple(digits)

def build_prefix_code(lengths, D):
    # Constructive converse: sorted lengths give aligned depth-l_max leaf blocks.
    if not kraft_feasible(lengths, D):
        raise ValueError("lengths violate the Kraft budget")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    code = [None] * len(lengths)
    l_max = max(lengths, default=0)
    next_leaf = 0
    for i in order:
        block = D ** (l_max - lengths[i])
        assert next_leaf % block == 0      # sorted lengths make the block aligned
        code[i] = base_D_digits(next_leaf // block, lengths[i], D)
        next_leaf += block
    return code

def shannon_lengths(p, D):                 # l_i = ceil(log_D(1/p_i))
    if any(pi <= 0 or pi >= 1 for pi in p):
        raise ValueError("use a non-degenerate positive-probability support")
    return [ceil(log(1.0 / pi, D)) for pi in p]

def entropy_D(p, D):
    return sum(pi * log(1.0 / pi, D) for pi in p if pi > 0)

def expected_length(lengths, p):
    return sum(pi * l for pi, l in zip(p, lengths))

# D = 2, dyadic p = (1/2, 1/4, 1/8, 1/8):
#   shannon_lengths -> (1, 2, 3, 3); kraft_sum = 1 (budget exactly filled);
#   build_prefix_code -> [(0,), (1, 0), (1, 1, 0), (1, 1, 1)];
#   expected_length = 1.75 = H_2(X)  (equality, since p is D-adic).
```
