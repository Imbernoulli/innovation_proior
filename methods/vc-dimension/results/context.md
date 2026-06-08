# Context: uniform convergence of empirical frequencies over a class of events

## Research question

I am given a class S of events (equivalently, decision rules / subsets of an input space X), a fixed but
unknown probability measure P on X, and an i.i.d. sample x₁,…,x_l. For a single event A, the empirical
frequency ν_A^(l) = n_A / l (the fraction of sample points falling in A) is what I observe; P_A = P(A) is
what I want to know. The classical law of large numbers tells me ν_A^(l) → P_A in probability for each
*fixed* A.

But in pattern recognition I do not work with a fixed A. I choose an event — a decision rule — *after*
looking at the data, by minimizing the empirical error over the whole class S. The rule I land on is
therefore selected because its empirical frequency was favorable, so its empirical frequency is a biased
estimate of its true probability. The quantity that actually controls whether fitting the sample tells me
anything about the truth is not the deviation of any one event but the *largest* deviation over the entire
class:

    π^(l) = sup_{A ∈ S} | ν_A^(l) − P_A |.

The precise problem: under what conditions on the class S does π^(l) → 0 in probability — and can the
condition be made independent of the unknown distribution P, with an explicit estimate of the rate and of
the sample size l needed to guarantee, with probability ≥ 1 − η, that π^(l) ≤ ε simultaneously over all of
S? A criterion that depends only on S (not on P) would let one certify a learning method before seeing
any data. This is a strengthening of the law of large numbers from one event to a whole class, uniformly.

## Background

**Bernoulli's law of large numbers (1713).** For any event A, any ε > 0, and any P,

    P_l( | ν_A^(l) − P_A | > ε ) → 0   as l → ∞.

The modern proof, via Chebyshev's inequality applied to the binomial count n_A, gives an explicit and
distribution-free rate for a single event:

    P( | ν_A^(l) − P_A | > ε ) ≤ P_A(1 − P_A) / (l ε²) ≤ 1 / (4 l ε²).

This is the bedrock: a single event's frequency concentrates at rate 1/(lε²), uniformly in P (because
P_A(1−P_A) ≤ 1/4 regardless of P_A). The whole difficulty is extending "one event" to "all of S at once."

**Glivenko–Cantelli (1933).** For the special class S of all rays {x ≤ a} on the real line, the empirical
distribution function F_l(a) = ν_{{x≤a}}^(l) converges to the true CDF F(a) = P_{{x≤a}} uniformly:

    P( sup_a | F_l(a) − F(a) | → 0 ) = 1.

So for at least one infinite, naturally ordered class, uniform convergence over the whole class does hold,
and the proof exploits the monotone, totally ordered structure of rays (the events are nested). This is a
proof of concept that "sup over an infinite class" need not blow up — but it leans entirely on the special
order structure and offers no general notion of how "rich" a class may be while still permitting uniform
convergence.

**The finite-class union bound.** If S contains only N decision rules, the path is elementary. Fix a
target risk κ and confidence η. A rule whose true risk exceeds κ classifies a random point correctly with
probability ≤ (1 − κ); the probability it gets all l sample points right is ≤ (1 − κ)^l; the probability
that *some* one of the (at most N) bad rules survives all l points is ≤ N(1 − κ)^l. Demanding
N(1 − κ)^l ≤ η gives a sufficient sample size

    l ≥ ( log N − log η ) / κ

(using −log(1−κ) ≥ κ). The number of rules N enters only through log N. This was the working tool for
finite classes ("full-memory" algorithms that make no training errors). Its fatal limitation: for the
classes that matter — linear decision rules / half-spaces in Rⁿ, the perceptron's hypotheses — there is a
continuum of rules, N = ∞, and the bound is vacuous.

**Diagnostic facts knowable at the time.**
- There exist classes for which uniform convergence genuinely *fails*: take X = [0,1] and S = all
  (open) subsets. On any sample of l distinct points, every one of the 2^l labelings is realized by some
  set in S, so the empirical frequencies can be made arbitrary; sup_{A∈S}|ν_A − P_A| does not vanish.
  Uniform convergence over a class is therefore a real restriction, not automatic.
- The "selection-bias" fallacy. A recurring objection from colleagues was: a probabilistic statement true
  for *every* rule is true for the rule the algorithm happens to choose, so choosing a rule after seeing
  the data changes nothing. This is wrong, and seeing why is the crux. A useful intuition: the probability
  of randomly meeting a person with a given rare condition in a city is tiny, but if you deliberately walk
  into the clinic for that condition the probability is far higher — even though the clinic is in the same
  city. Selecting the empirically-best rule is exactly such a deliberate walk; the chosen rule is special,
  and only a bound that holds *simultaneously over the whole class* covers it.
- The missing words. In a 1965 argument, Khurgin pointed out that demanding a guarantee that holds across
  an entire (e.g. linear) class amounts to "playing on the non-compactness of the unit ball in Hilbert
  space" and is precisely a demand for *uniform convergence*. This named the right object: the deviation
  must be controlled uniformly over S.

**Counting cells cut by hyperplanes (Schläfli 1852; Cover 1965).** A separate combinatorial thread:
how many distinct sign-patterns can r points induce on an n-parameter family of linear threshold
rules, such as the fixed-threshold half-spaces {x : ⟨w,x⟩ ≥ 1} in Rⁿ? Equivalently, as w varies,
the r sample points cut the parameter space Rⁿ by r affine hyperplanes. The count obeys the
Pascal-type recurrence

    Φ(n, r) = Φ(n, r − 1) + Φ(n − 1, r − 1),    Φ(0, r) = 1, Φ(n, 0) = 1,

with closed form Φ(n, r) = Σ_{k=0}^{n} C(r, k), which is polynomial in r of degree n (for r > n) —
sharply smaller than the trivial 2^r. So for linear threshold classes, the number of distinct labelings
of r points can grow polynomially even when the class itself is infinite. This number, attached to a
finite sample rather than to the class's cardinality, is the quantity that might replace N in a union
bound.

## Baselines

- **Bernoulli / Chebyshev, single event.** Core idea: bound the binomial deviation of one fixed event,
  giving ≤ 1/(4lε²), distribution-free. Limitation: it controls one event chosen *before* the sample; it
  says nothing about sup over a class, hence nothing about a rule selected by fitting.

- **Glivenko–Cantelli, ordered class.** Core idea: for nested events (rays), uniform convergence of
  F_l → F follows from monotonicity by a covering argument on finitely many quantiles. Limitation: the
  proof and the result are tied to the total order of the real line; there is no general measure of class
  complexity, and it does not extend to half-spaces, intervals, or arbitrary classes.

- **Finite-class union bound.** Core idea: P(some bad rule survives) ≤ N · (single-rule failure
  probability); sample size scales as log N. Limitation: requires |S| = N < ∞. For the continuum of
  linear rules it is empty. A direct discretization of the continuum (ε-net the parameter space) makes
  N depend on ε and on the ambient dimension in a way that is both crude and distribution-dependent.

## Evaluation settings

The natural testbeds are classes of decision rules / events on an input space X with an unknown P:
- homogeneous and fixed-threshold linear half-spaces in Rⁿ;
- affine half-spaces in Rⁿ (linear thresholds with an intercept);
- rays {x ≤ a} and intervals on the line;
- the pathological class of *all* subsets of [0,1].

The yardsticks are: the tail probability P( π^(l) > ε ) as a function of sample size l; whether it tends
to 0 (and almost surely); the explicit sample size l(ε, η) needed for P( π^(l) > ε ) ≤ η; and whether the
criterion is distribution-free (holds for every P) or distribution-dependent.

## Code framework

The computational pieces already available are empirical frequencies on a sample, the single-event
Chebyshev bound, and a finite-class union bound. The missing piece is the function that, for an infinite
class, plays the role that N plays for a finite one: a measure of how many genuinely different ways the
class can label a sample, and the deviation bound built from it.

```python
import numpy as np

def empirical_frequency(member_indicator, sample):
    """nu_A^(l): fraction of sample points that fall in event A."""
    return np.mean([member_indicator(x) for x in sample])

def single_event_chebyshev_tail(l, eps):
    """Bernoulli/Chebyshev bound for ONE fixed event, distribution-free."""
    return 1.0 / (4.0 * l * eps**2)

def finite_class_sample_size(N, kappa, eta):
    """Union-bound sample size for a finite class of N rules."""
    return (np.log(N) - np.log(eta)) / kappa

def class_capacity(S, l):
    """Effective 'number of distinguishable events' of class S on l points.
    For a finite class this is just |S|; for an infinite class we do not yet
    know what to put here."""
    # TODO: the combinatorial capacity quantity that replaces N
    pass

def uniform_deviation_tail(S, l, eps):
    """P( sup_{A in S} |nu_A - P_A| > eps ): the object we want to bound,
    simultaneously over the whole class, with no dependence on P."""
    # TODO: pass from the infinite class S to its capacity on the sample,
    #       then to a distribution-free tail bound
    pass
```
