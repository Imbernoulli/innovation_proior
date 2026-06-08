# No Free Lunch: there is no universally best algorithm

## The problem it solves

General-purpose optimizers and learners are routinely benchmarked on a handful of problems and then declared
"better." The No Free Lunch (NFL) theorems make precise what such a comparison can and cannot license: **with
no assumption about which problems are likely, no search or learning algorithm outperforms any other.**
Averaged uniformly over all objective functions (optimization) or all target concepts (learning), every
algorithm has identical expected performance. Any algorithm that does better than another on some problems
does correspondingly worse on the rest — the wins and losses cancel, weighted by their amounts, by an exact
counting identity over the space of problems. Therefore all real performance comes from **prior assumptions
matching the problem** (inductive bias); it cannot come from the algorithm alone.

## Setup

Finite search space 𝒳 (|𝒳| points), finite cost values 𝒴 (|𝒴| values). A problem is a cost function
f : 𝒳 → 𝒴; the space of all problems is ℱ = 𝒴^𝒳, with |ℱ| = |𝒴|^|𝒳|. An algorithm a maps a *sample*
d_m = {(d_m^x(1), d_m^y(1)), …} of m **distinct** visited points and their costs to a new, previously
**unvisited** point (counting only distinct evaluations removes the algorithm-and-problem-dependent
revisiting artifact). Performance after m steps is any function Φ of the cost sequence d_m^y. Everything is
cast probabilistically: P(d_m^y | f, m, a) is the probability of obtaining cost sequence d_m^y running a on
f for m distinct steps (0/1 for deterministic a; probability lets a prior P(f) carry effective ignorance).

## The theorem and proof

**No Free Lunch (static optimization).** For any cost sequence d_m^y and any m,

  Σ_{f ∈ ℱ} P(d_m^y | f, m, a)  is independent of the algorithm a.

Hence for any performance measure Φ, the uniform average over f of P(Φ(d_m^y) | f, m, a) is the same for
every algorithm: averaged over all problems, all algorithms tie.

*Proof (induction on m).*

- **Base, m = 1.** The first point d_1^x is fixed by a alone, and the only possible cost is f(d_1^x), so
    Σ_f P(d_1^y | f, 1, a) = Σ_f δ(d_1^y, f(d_1^x)).
  Summed over all f, the Kronecker δ pins f at the single point d_1^x; the other |𝒳| − 1 points take any of
  |𝒴| values, so the sum = |𝒴|^{|𝒳|−1}. This count does not depend on which point a chose — independent of a.

- **Inductive step.** Assume Σ_f P(d_m^y | f, m, a) is a-independent for all sequences. Once the first m cost
  values are fixed, the deterministic algorithm recursively fixes the full size-m sample d_m. Factor the
  size-(m+1) sample:
    P(d_{m+1}^y | f, m+1, a) = P(d_{m+1}^y(m+1) | d_m, f, a) · P(d_m^y | f, m, a),
  with P(d_m^y | f, m+1, a) = P(d_m^y | f, m, a). The next point x = a(d_m) is fixed and **unvisited**, and its
  cost is f(x), so
    Σ_f P(d_{m+1}^y | f, m+1, a) = Σ_f δ(d_{m+1}^y(m+1), f(a(d_m))) · P(d_m^y | f, m, a).
  The δ pins f only at the unvisited point a(d_m). The |𝒳| − m − 1 points neither in the sample nor equal to
  a(d_m) are free, contributing |𝒴|^{|𝒳|−m−1}. In the size-m sum, the same compatible partial functions have
  |𝒴|^{|𝒳|−m} full extensions; in the size-(m+1) sum, the new point is pinned and only
  |𝒴|^{|𝒳|−m−1} extensions remain. The ratio is exactly 1/|𝒴|:
    Σ_f P(d_{m+1}^y | f, m+1, a) = (1/|𝒴|) · Σ_f P(d_m^y | f, m, a),
  which is a-independent by hypothesis. ∎

**Why it cancels (the bijection).** Summing over all f, the cost the function hands the algorithm at its next
(unvisited) point is a flat draw over 𝒴, independent of how the algorithm chose that point. Relabeling f's
values on the unseen points is a bijection of ℱ onto itself, giving identical counts for fixed cost sequences.
For a general performance score, the safe statement is weighted cancellation: advantages on one subset of
functions are offset by disadvantages on the rest.

## Geometric meaning: performance is alignment with the prior

Over a prior P(f), performance is an inner product:
  P(d_m^y | m, a) = Σ_f P(d_m^y | m, a, f) P(f) = v⃗ · p⃗,
where v⃗ has components P(d_m^y | m, a, f) (the algorithm) and p⃗ has components P(f) (the prior). For fixed
d_m^y and m, NFL says every deterministic algorithm's v⃗ has the same number of ones — same length, same
projection onto the uniform diagonal 1⃗ — so all v⃗ lie on a cone about the diagonal and differ only in their
tilt. With a uniform p⃗ (the diagonal) every algorithm has identical inner product (NFL). The only way to beat
another algorithm is to tilt v⃗ toward a **non-uniform** P(f): all advantage comes from matching the prior.
Uniform P(f) is a
*tool* exposing this "skeleton" of optimization, not a claim that problems are uniformly distributed.

## Average vs. per-problem: head-to-head minimax distinctions

NFL flattens the **average** over f; it does **not** flatten per-function behavior. For a performance score
where larger is better, define a₁ as head-to-head minimax-superior to a₂ if for some k > 0 there is an f on
which a₁ beats a₂ by k and **no** f on which a₁ loses to a₂ by k. The signed differences sum to zero (NFL),
but their *shape* can be lopsided: the joint distribution of two algorithms' cost histograms over f need not
be symmetric, so from the costs alone one can sometimes infer which algorithm produced which. In the
supervised-learning OTS setting, the random learner has a flat profile over targets (max_f E = min_f E under
homogeneous loss), so it is head-to-head minimax-superior to every other learner. This is a learning result,
not a claim that random search's expected optimization cost is constant over all objective functions.

## The supervised-learning companion

For target f, hypothesis h, training set d, and loss L, define error on **off-training-set (OTS)** inputs
q ∉ d_X only — isolating generalization from memorization. For a uniform prior and zero-one (homogeneous)
loss, the same sum (with the roles of f and h interchangeable in the NFL counting) gives P(c | d) identical
for **every** learning algorithm; weighted target and relabeled-prior averages cancel. In the zero-one OTS
majority-vs-anti-majority calculation, with fixed d, single-valued target φ, no training noise, and candidate
hypotheses restricted to IID error below ε < 1, the restricted sum satisfies
Σ_{h₁,h₂}[E(C_OTS|anti-majority) − E(C_OTS|majority)] < 0, so low training-style fit alone still does not
license OTS generalization. Cross-validation has no assumption-free justification: choosing the **best**
held-out candidate does as well as choosing the **worst** ("anti-cross-validation") over all targets.
Averaging learners reduces variance without changing bias (convexity:
(z − [α+β]/2)² ≤ ½(z−α)² + ½(z−β)²) but does not beat a single fixed learner on the OTS zero-one problem —
variance reduction is not a free lunch.

## Verification

```python
import itertools
from collections import Counter

X = [0, 1, 2, 3]; Y = [0, 1]; m = 3

def all_functions():
    for vals in itertools.product(Y, repeat=len(X)):
        yield dict(zip(X, vals))

def run(algorithm, f, m):
    sample = []
    seen = set()
    for _ in range(m):
        x = algorithm(sample)          # next point, must be unvisited
        assert x in X and x not in seen
        seen.add(x)
        sample.append((x, f[x]))
    return tuple(y for _, y in sample)  # cost sequence d_m^y

def fixed_order(sample):
    return len(sample)
def greedy_then_scan(sample):
    if not sample:
        return 0
    seen = {x for x, _ in sample}
    best_x = min(sample, key=lambda p: p[1])[0]
    for cand in (best_x + 1, best_x - 1, *X):
        if cand in X and cand not in seen:
            return cand

def histogram_over_all_f(algorithm):
    h = Counter()
    for f in all_functions():
        h[run(algorithm, f, m)] += 1
    return h

hist_A = histogram_over_all_f(fixed_order)
hist_B = histogram_over_all_f(greedy_then_scan)
assert hist_A == hist_B
assert set(hist_A.values()) == {len(Y) ** (len(X) - m)}
# Summed over all cost functions, each observed cost-sequence is produced by the
# same number of functions regardless of the algorithm — no free lunch.
```

## The takeaway

There is no universally best optimizer or learner. Performance is conserved across the space of all problems
by exact weighted cancellation, so every algorithm's edge on some problems is paid for on the rest. Generalization is
possible only because real problems are *not* uniformly distributed, and an algorithm helps only insofar as
its inductive bias matches that structure. The bias is not optional — it is the entire source of performance.
