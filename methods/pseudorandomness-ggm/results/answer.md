# Computational pseudorandomness: from one-way functions to pseudorandom generators and the GGM pseudorandom functions

## Problem

Truly random bits are expensive, and a truly random function on {0,1}^k costs ~k·2^k bits to
specify. We want deterministic, efficiently-computable objects that are "random enough" to substitute
for true randomness in any efficient use: a generator that stretches a short seed into a long
random-looking string, and a short-keyed family of functions indistinguishable from a uniformly
random function even under adaptive queries.

## Key idea

"Random enough" is a **computational** notion. A distribution is *pseudorandom* if no efficient
(probabilistic polynomial-time) test can distinguish it from uniform — **computational
indistinguishability**. Three consequences build the theory:

1. **Yao's theorem (next-bit unpredictability ⟺ indistinguishability).** A generator fools every
   efficient statistical test iff no efficient predictor guesses the next output bit from the prefix
   better than 1/2 + negligible. This reduces a universal claim (all tests) to defeating one
   adversary (a predictor), via a hybrid argument.
2. **Blum–Micali–Yao generator (OWP + hard-core bit ⟹ PRG).** A one-way permutation equipped with a
   *hard-core* bit B (computable from x, unpredictable from f(x)) gives a generator: walk the
   permutation orbit and emit B at each state. The reversed-orbit next-bit reduction proves
   unpredictability, and bit-reversal gives the online output order.
3. **GGM construction (PRG ⟹ PRF).** Reading the PRG's length-doubling as *branching* (one seed → two
   child seeds) and iterating k times yields a depth-k binary tree; its 2^k leaves, addressed by the
   k-bit input, form a pseudorandom function family — exponentially many random-looking functions
   from one short key, each evaluable in k PRG calls.

## Definitions

- **One-way function:** efficient f with Pr[f(A(f(x))) = f(x)] ≤ negl for every efficient A (x
  uniform). A **one-way permutation** is additionally a bijection on {0,1}^n.
- **Hard-core predicate** B of f: B(x) efficient from x, but given only f(x), every efficient A has
  Pr[A(f(x)) = B(x)] ≤ 1/2 + negl.
- **Computational indistinguishability:** ensembles D₁ ≈_c D₂ iff for every efficient A,
  |Pr[A(D₁)=1] − Pr[A(D₂)=1]| ≤ negl.
- **PRG:** efficient G: {0,1}^k → {0,1}^{P(k)}, P(k) > k, with G(U_k) ≈_c U_{P(k)}.
- **Next-bit test:** for every efficient predictor C and every i, Pr[C(b₁…b_i)=b_{i+1}] < 1/2 + negl.
- **PRF / pseudorandom function family:** {f_K : {0,1}^k → {0,1}^k}_{K∈{0,1}^k} with (i) indexing
  (choose f by choosing K uniformly), (ii) poly-time evaluation of f_K(x), (iii) for every efficient
  oracle test, |Pr[A^{f_K}=1] − Pr[A^{R}=1]| ≤ negl with R a uniformly random function.

## Theorem 1 (Yao). G passes all efficient statistical tests ⟺ G is next-bit unpredictable.

*Easy direction.* A predictor is a test (output 1 iff its next-bit guess is correct): advantage
1/2 + ε on G vs 1/2 on random ⟹ G fails a test. So passing all tests ⟹ unpredictable.

*Converse (hybrid).* If test T distinguishes G(U_k) from U_ℓ with advantage ε, define hybrids
exp_i = (first i bits of G(x)) · (last ℓ−i uniform bits); exp_0 is uniform, exp_ℓ is G's output.
Let p_i = Pr[exp_i→1]. Complement T if needed so p_ℓ − p_0 ≥ ε; then some signed gap
p_{j+1} − p_j ≥ ε/ℓ. Predictor A (given first j bits): pick random guess g and random tail r, run T
on (prefix)·g·r; output g if T=1 else 1−g. With z₁/z₂ the strings carrying the correct/flipped
(j+1)-st bit,
Pr[A correct] = 1/2 + (Pr[T(z₁)=1] − Pr[T(z₂)=1])/2
= 1/2 + (Pr[exp_{j+1}→1] − Pr[exp_j→1]) ≥ 1/2 + ε/ℓ.
Non-negligible ε/ℓ contradicts unpredictability. ∎

*Corollary (order-independence).* If G is a PRG so is its bit-reversal G~ (reverse a test's input;
reversed-uniform is uniform). Hence Blum–Micali, which emits along the orbit, may run forward in real
time.

## Theorem 2 (Blum–Micali–Yao). If f is a one-way permutation with hard-core bit B, then
G(x) = f(x)·B(x) (and its iterate) is a pseudorandom generator.

Iterated form: s₀ = seed; for i ≥ 1, b_i = B(s_{i−1}), s_i = f(s_{i−1}); output b₁…b_ℓ.

*Security.* Prove next-bit unpredictability for the reversed orbit b_ℓ…b₁. If C predicts the next
bit after seeing B(f^c(x)), …, B(f(x)), then given f(x) one computes that entire prefix by walking
forward and uses C to predict B(x), contradicting the hard-core property. Because f is a permutation,
the relevant orbit state is exactly uniform and the reduction loses no distributional factor; by
Theorem 1 and bit-reversal, the online output b₁…b_ℓ is pseudorandom. ∎
(For discrete-exp f(s)=g^s mod p, B(s) is the MSB/principal-root bit of the image g^s; hardness
follows from exact discrete-log reconstruction plus random self-reduction and the weak law of large
numbers.)

## Theorem 3 (GGM). If G: {0,1}^k → {0,1}^{2k} is a PRG with halves G₀,G₁, then
F_K(x) = G_{x_k}(G_{x_{k−1}}(…G_{x_1}(K)…)) is a pseudorandom function family.

Indexing and poly-time evaluation (k PRG calls) are immediate. Picture a depth-k binary tree: root
labeled K; an internal node labeled v has children G₀(v), G₁(v); leaf reached by path x is F_K(x).

*Security (hybrid over tree levels, lazy).* Hybrid A_i: place independent uniform labels at all
level-i nodes, derive deeper levels by G, answer x with its leaf label. A_0 = real F_K (random root,
all G below); A_k = truly random function (independent random leaves). Adjacent A_i, A_{i+1} differ by
"two children = G(random parent)" vs "two children = fresh random" at the level-i nodes that queries
actually touch. Level i has 2^i nodes, but a P(k)-query test touches only poly-many nodes, so instantiate
lazily and use at most P(k) independent 2k-bit strings at that level. If p_i=Pr[A_i→1] and
|p_0−p_k|>1/Q(k), a random-level string test has advantage
(1/k)|Σ_i(p_i−p_{i+1})|=(1/k)|p_0−p_k|>1/(kQ(k)); a further per-touched-node hybrid converts the
polynomial list of samples to one PRG challenge if needed. Contradiction. ∎

*Equivalent per-invocation hybrid.* With M = P(k)·k total G-invocations on the query paths, hybrids
H^0=F_K, …, H^M=random function on the queried points replace one G-invocation by a fresh random
doubling at a time; a distinguisher between H^{m−1}, H^m embeds its challenge y∈{0,1}^{2k} as the
m-th doubling (y=G(seed)⟹H^{m−1}, y uniform ⟹ H^m). Random m, telescope ⟹ advantage ε/M, M = poly.

## Theorem 4 (GGM; characterization). A family satisfying indexing + poly-time evaluation cannot be
polynomially inferred ⟺ it passes all efficient statistical tests for functions.

"Polynomially inferred" = an efficient adversary, after adaptive queries, names a fresh point x and
recognizes f(x) among random alternatives with advantage. If a distinguisher T separates the family
from a random function with p^{P(k)}_k−p^0_k>1/Q(k), choose the exam index i ∈ {0,…,P(k)−1} uniformly:
answer T's first i queries from the oracle, use query i+1 as the exam, feed T either f(x) or a random
value, and let T's final bit decide which was used. Then
Pr[guess correct] = (1/(2P(k)))·Σ((1−p^i_k)+p^{i+1}_k) ≥ 1/2 + 1/(2·P(k)·Q(k)).
The reverse direction is immediate by running the inferrer as a statistical test. Thus PRF
indistinguishability and non-inferability are equivalent.

## Net result

OWF ⟹ PRG ⟹ PRF. Under the existence of a one-way function (or permutation), one obtains: a
pseudorandom generator stretching a short seed to any polynomial length, and a pseudorandom function
family giving exponentially many random-looking functions from one short key, each computed on demand
in k generator evaluations — making "a random function" a usable, poly-time object indistinguishable
from the ideal to every efficient adversary.

## Construction (concrete)

```python
# G: {0,1}^k -> {0,1}^{2k}; G0, G1 its left/right k-bit halves.
def G0(s, k): return prg_double(s, k)[:k]
def G1(s, k): return prg_double(s, k)[k:]

# Blum-Micali-Yao PRG: one hard-core bit per step of a one-way permutation.
def prg_bits(seed, length):
    s, out = seed, []
    for _ in range(length):
        out.append(hard_core_bit(s))     # provably-unpredictable bit
        s = one_way_permutation(s)        # advance the orbit (stays uniform)
    return out                            # order-independent: emit forward in real time

# GGM tree: short key K -> random-looking function on k-bit inputs.
def f_K(K, x, k):
    label = K                             # root = key
    for bit in x:                         # walk the k-bit input down the tree
        label = G0(label, k) if bit == 0 else G1(label, k)   # child = one doubling
    return label                          # leaf label = f_K(x); k PRG calls, never 2^k
```
