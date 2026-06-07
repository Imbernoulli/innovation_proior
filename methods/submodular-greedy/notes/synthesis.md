# Synthesis

## Problem
Maximize a monotone submodular set function f: 2^V -> R>=0 subject to |S| <= k.
NP-hard (generalizes max coverage / max-k-cover). Want a polynomial-time
algorithm with a *provable* multiplicative guarantee against f(OPT).

## Load-bearing concepts (all grounded)
- Submodularity (diminishing returns): A subset B, e not in B =>
  f(A+e)-f(A) >= f(B+e)-f(B). Equivalent lattice form: f(A)+f(B) >= f(A∪B)+f(A∩B).
  (Dartmouth notes; Jeremy Kun.)
- Monotone: A subset B => f(A) <= f(B).
- Max coverage: f(I) = |union of sets S_j, j in I| is non-neg, monotone, submodular.

## The (1-1/e) proof (Dartmouth, Fisher-Nemhauser-Wolsey)
Let X_i be greedy set at start of round i (X_1 = empty), O = OPT, |O| <= k.
Greedy rule: f(X_{i+1}) - f(X_i) >= f(X_i + o) - f(X_i) for every element, hence
for the best o; averaging over o in O:
  f(X_{i+1}) - f(X_i) >= (1/k) * sum_{o in O} [f(X_i + o) - f(X_i)]
                       >= (1/k) * [f(X_i ∪ O) - f(X_i)]   (submodularity, termwise dominance)
                       >= (1/k) * [f(OPT) - f(X_i)]        (monotonicity, f(X_i ∪ O) >= f(O) = f(OPT))
Rearranged: f(X_{i+1}) >= (1/k) f(OPT) + (1 - 1/k) f(X_i).
Let a_i = f(OPT) - f(X_i). Then a_{i+1} <= (1 - 1/k) a_i, so a_{k+1} <= (1-1/k)^k a_1.
a_1 = f(OPT) - f(empty) <= f(OPT). So f(X_{k+1}) = f(OPT) - a_{k+1}
   >= f(OPT) [1 - (1-1/k)^k] >= (1 - 1/e) f(OPT), using (1-1/k)^k < 1/e (since 1-x <= e^{-x}).

Telescoped form (Dartmouth eq before Thm 2):
f(X_k) >= f(OPT) * (1/k)[1 + (1-1/k) + ... + (1-1/k)^{k-1}] = f(OPT)[1-(1-1/k)^k].

## Submodular set cover (the "ancestor", Wolsey 1982) -- min cost s.t. f(S)=f(V)
Greedy picks j minimizing c(j)/(f(X+j)-f(X)); H_N approximation, N=f(V).
This is the cost-version; cardinality-constrained max is the value-version, same
two-line submodularity+monotonicity engine. Set cover greedy (ln n) is the
special case.

## Matroid generalization (Fisher-Nemhauser-Wolsey Part II, 1978)
Cardinality constraint = uniform matroid. Under a general matroid the same greedy
is only 1/2-approx; cardinality is special and sharpens to 1-1/e. Tight: NWF and
later Feige show no poly algorithm beats 1-1/e for max coverage (value-oracle /
P!=NP).

## Acceleration (Minoux 1978, lazy greedy)
Marginals are monotone non-increasing in S (submodularity), so a previously
computed gain is a valid upper bound. Max-heap of stale gains; re-evaluate top,
if it still tops the heap it is the true argmax. Same output, far fewer oracle
calls.

## Code (apricot maxCoverage + naive/lazy optimizers)
- Oracle f(S), marginal(e,S). MaxCoverage: gain = |set_e \ covered(S)|.
- naive greedy: k rounds, argmax marginal.
- lazy greedy: heap [-gain, e, evaluated_at_size]; refresh stale top, accept fresh top.
Verified: both pick same set, coverage 8 on the toy instance.

## Design decisions -> why
- Greedy by max marginal gain (not random/best-pair/local search): the proof needs
  f(X_{i+1})-f(X_i) >= best single marginal; only max-marginal gives the per-step
  (1/k)(f(OPT)-f(X_i)) drop. Random search has no such handle -> anti-pattern.
- Average over O (not max): turns "greedy beats each single o" into "greedy beats
  the *average* contribution", which submodularity lower-bounds by the *joint*
  f(X_i ∪ O). The averaging is what k enters through.
- a_i recurrence + 1-x<=e^{-x}: the geometric (1-1/k) contraction is exactly what
  produces 1/e; nothing magic about e, it falls out of the contraction.
- Stop at k rounds / non-positive gain: monotone f never wants you to remove;
  cardinality is the only thing stopping you.
- Lazy heap: diminishing returns => stale gain is an upper bound => safe to skip.
