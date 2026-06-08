# PAC Learning — the method, distilled

## The problem

"Can machines learn?" had no precise model, so it could not be turned into theorems the
way Turing's model turned "what can be computed?" into theorems. Prior formal accounts
each failed a desideratum: Gold's identification-in-the-limit demands *exact* convergence
with no time bound (and makes interesting classes unlearnable); statistical pattern
recognition lacks a complexity-theoretic notion of which concept classes admit a feasibly
deducible recognizer; heuristic AI concept learning gives no provable guarantee. The goal:
a model under which *whole, characterizable* classes of nontrivial concepts are provably
acquirable from examples in a *polynomial* number of steps.

## The key idea — the PAC definition of "learnable"

Represent concepts as Boolean functions over propositional variables. Give the learner a
protocol: **EXAMPLES**, returning a positive example drawn from a fixed but *arbitrary,
unknown* distribution D, and (when needed) **ORACLE**, a membership query. Measure a
hypothesis g's error against the target f as Pr_{x∼D}[g(x) ≠ f(x)] — under the same D that
generates examples. Then:

> A concept class C is **PAC-learnable** if there is an algorithm A such that for **every**
> concept f ∈ C, **every** distribution D, and every ε, δ ∈ (0,1), A draws a number of
> examples polynomial in 1/ε, 1/δ, n and the target size, runs in polynomial time, and
> outputs a hypothesis g with
>
>   Pr[ err_D(g) ≤ ε ] ≥ 1 − δ.

*Probably* (confidence 1 − δ) *approximately correct* (error ≤ ε), *distribution-free*
(for all D). Approximate because exact deduction from polynomially many examples is
impossible even for a single monomial; probabilistic because a random sample can be
unlucky; distribution-free because only the inputs that actually occur (where D has mass)
can matter. This converts "learnability" into a sharp, falsifiable complexity question:
some classes are inside the boundary, others (cryptographically hard functions) provably
outside.

## The combinatorial engine: a sample-complexity bound

**L(h,S)** = smallest number of independent Bernoulli trials, each with success
probability ≥ h⁻¹, such that Pr[fewer than S successes] < h⁻¹.

**Proposition.** For integer S ≥ 1 and real h > 1, **L(h,S) ≤ 2h(S + ln h)** — linear in
both h and S.

*Proof.* Multiplicative Chernoff lower tail: in m trials each with success prob ≥ p, for
k < mp, Pr[≤ k successes] ≤ e^{−mp+k}(mp/k)^k. Put m = 2h(S+ln h), p = h⁻¹, k = S, so
mp = 2S+2ln h, −mp+k = −S−2ln h, mp/k = 2(1+(ln h)/S). The bound becomes
e^{−S−2ln h}·2^S·(1+(ln h)/S)^S. Using (1+1/x)^x < e with x = S/ln h gives
(1+(ln h)/S)^S < e^{ln h} = h, so the whole expression < e^{−S−2ln h}·2^S·h =
(2/e)^S·h^{−1} < h^{−1} (since 2/e < 1, S ≥ 1). ∎

## Finite hypothesis classes — the (1/ε)ln(|H|/δ) bound

If H is finite and the algorithm returns any hypothesis **consistent** with the sample,
then m examples suffice whenever **m ≥ (1/ε) ln(|H|/δ)**.

*Proof.* Fix a "bad" h with err_D(h) > ε. Pr[h consistent with one example] < 1 − ε, so
Pr[h consistent with all m] < (1−ε)^m ≤ e^{−εm}. Union-bounding over the ≤ |H| bad
hypotheses, Pr[some bad h survives] < |H|·e^{−εm}. Setting |H|e^{−εm} ≤ δ gives
**m ≥ (1/ε)(ln|H| + ln(1/δ))**. ∎

## Result 1 — Conjunctions (and k-CNF) are PAC-learnable from examples alone

**Conjunction algorithm.** Start with g = AND of all 2n literals (maximally restrictive).
On each positive example v, delete every literal v does not make true; if v has `*` in
coordinate i, delete both x_i and ¬x_i. Every literal of the true f survives forever, so g
⊇ f's literals — g is always at least as restrictive as f (one-sided error). The only
errors come from a *bad literal* (in g, not in f). Calling a literal bad if
Pr_D[it is not made true] ≥ ε/2n: it survives m examples w.p. ≤ (1−ε/2n)^m ≤ e^{−εm/2n};
union over ≤ 2n literals, failure ≤ 2n·e^{−εm/2n} ≤ δ gives
**m ≥ (2n/ε) ln(2n/δ)**, time O(mn). (Cross-check: |H| = 3^n general / 2^n monotone in the
finite-class bound gives the same n/ε scaling: (1/ε)(n ln 3 + ln(1/δ)).)

**k-CNF (Theorem A).** A conjunction is k = 1. For fixed k there are fewer than (2t)^{k+1}
non-tautological clauses of ≤ k literals. Start with g = product of *all* of them; on each
positive v, delete every clause v fails to satisfy. Let B be the product of all up-to-k clauses
implied by f; no clause of B is ever deleted and B ≡ f, so g stays squeezed between f and
B (one-sided). Error X = D-mass of {v : v⊨f, v⊭g} decreases monotonically; each example
with v⊭g deletes ≥ 1 clause (≤ (2t)^{k+1} deletions possible). Set
**h = max{1/ε, 1/δ}**. With
**L = L(h,(2t)^{k+1}) ≤ 2h((2t)^{k+1}+ln h)** examples, either the error drops below
1/h ≤ ε, or the probability that it stays above 1/h is < 1/h ≤ δ. Polynomial for fixed k;
no oracle needed.

## Result 2 — Monotone DNF is PAC-learnable with a membership oracle (Theorem B)

Unrestricted DNF carries an intractable membership test (deciding whether a partial vector
implies a DNF is the co-NP-hard tautology problem, Cook 1971), so restrict to monotone
DNF, whose concept is always evaluable. Start g ≡ 0. On a positive v with v⊭g, use ORACLE
to strip v to a prime implicant m of f: ignore non-1 coordinates, then for each 1-coordinate,
undetermine it, drop it if ORACLE still says f is implied, and keep it only if ORACLE says
the loosened vector no longer implies f. Add m to g. Each m is a genuine prime
implicant (g ⊆ f, one-sided); there are at most d (the degree) of them, ≤ dt oracle calls.
Error X = D-mass of {w : w⊨f, w⊭g} starts at 1 and decreases; each example with v⊭g adds a
monomial. With **h = max{1/ε, 1/δ}**, **L = L(h,d) ≤ 2h(d+ln h)** examples and ≤ dt oracle
calls give error ≤ ε with failure probability < δ. (For unrestricted DNF under
distributions on total vectors only, one gets the *function*, not the concept — Theorem B′.)

## Conjunction learner code

```python
import math

def num_examples(n, eps, delta):
    # bad-literal union bound: 2n*(1-eps/2n)^m <= delta  =>  m >= (2n/eps)*ln(2n/delta)
    return math.ceil((2 * n / eps) * math.log(2 * n / delta))

def learn_conjunction(EXAMPLES, n, eps, delta):
    m = num_examples(n, eps, delta)
    require_pos = [True] * n   # x_i still required in g
    require_neg = [True] * n   # ~x_i still required in g
    for _ in range(m):
        v = EXAMPLES()                 # vector with f(v) = 1, drawn from D
        for i in range(n):
            if v[i] == 1:
                require_neg[i] = False  # ~x_i is not true here -> drop it
            elif v[i] == 0:
                require_pos[i] = False  # x_i is not true here -> drop it
            else:
                require_pos[i] = False  # '*' forces neither literal
                require_neg[i] = False
    def g(x):
        for i in range(n):
            if require_pos[i] and x[i] != 1: return 0
            if require_neg[i] and x[i] != 0: return 0
        return 1
    return g
```

## Why each choice

- **Approximate (ε):** exact deduction from polynomially many examples is impossible (a
  single monomial cannot be pinned down under an adversarial D).
- **Probably (δ):** the random sample can be unlucky; certainty is unattainable.
- **Distribution-free, error under D:** error is measured under the generating
  distribution; only inputs that occur matter, which both makes the goal achievable and
  dissolves Gold's worst-case impossibility.
- **Polynomial:** "feasible = polynomial" makes learnability a complexity question.
- **EXAMPLES + ORACLE:** calibrates teacher power — enough to give typical examples, not
  enough to encode the target program in a premeditated example sequence.
- **Start maximally restrictive, make monotone progress:** keeps g one-sided (g ⊆ f) and
  gives a bounded, countable notion of progress that L(h,S) converts into a sample bound.
