# Context: the asymptotics of rare sums beyond the central limit theorem

## Research question

Let `Z_1, Z_2, …` be independent, identically distributed real random variables with common cumulative distribution function `V`, mean `E(Z) = 0`, and variance `E(Z²) = σ² > 0`. Write the normed sum

```
F_n(x) = Prob( (Z_1 + … + Z_n) / (σ√n) ≤ x ).
```

The central limit theorem describes `F_n(x)` for a **fixed** value of `x`. The question here is different: what is the probability that the *unnormed* sum `Z_1 + … + Z_n` is *large*, on the scale of `n` itself rather than `√n`? Equivalently, what is the asymptotic behaviour of the tail probability

```
1 − F_n(x)   as   x → +∞   together with   n → ∞,
```

and in particular when `x` grows like a power of `n` — most sharply when `x ~ √n`, i.e. when the sum sits at order `n`?

In insurance and risk theory the total claims paid by a company over a period are a sum of many independent individual claims. Solvency depends on the probability of a *ruinously large* total — an event far out in the right tail. What is wanted there is the *magnitude* of the probability of a large excess, and the *size of the error* committed by replacing the true distribution with the normal one: a genuine asymptotic estimate of `1 − F_n(x)` in a regime where `x` runs off to infinity together with `n`.

## Background

**The central limit theorem.** In the form sharpened by Lindeberg and by Paul Lévy, the Laplace–Lyapunov theorem states that for each fixed real `x`,

```
F_n(x) → Φ(x) = (1/√(2π)) ∫_{−∞}^{x} e^{−t²/2} dt.
```

This pins down the probability that the normed sum lands within `O(1)` of its mean, on the `√n` scale. Lyapunov supplied a *rate*: under a third-moment hypothesis,

```
| F_n(x) − Φ(x) | ≤ k · (log n) / √n ,     k = (3 / σ³) ∫ |y|³ dV(y),
```

uniformly in `x`. This is an estimate of the difference `F_n − Φ`.

**The tail ratios.** Let `x` vary with `n`. As `x → +∞`, both `F_n(x) → 1` and `Φ(x) → 1`. The natural object becomes the ratio

```
(1 − F_n(x)) / (1 − Φ(x))   (right tail),     F_n(−x) / Φ(−x)   (left tail).
```

**A structural ceiling.** If `V` is supported on a bounded interval `(−μσ, μσ)`, then the sum is supported on `(−μ√n, μ√n)` after norming, so `1 − F_n(x)` is *identically zero* for `x > μ√n`. The target scale is `x ~ √n` (sum of order `n`), which is exactly the "large excess" regime of the risk problem.

**Light tails as the natural arena.** For the right tail to decay at a clean exponential rate in `n`, the summands have tails lighter than any polynomial — concretely, the moment-generating-type integral

```
R(h) = ∫ e^{hy} dV(y)
```

converges for `h` in a neighbourhood of `0` (heavy-tailed sums behave differently, dominated by a single large summand). Its logarithm `log R(h)` is the generating function of the cumulants (semi-invariants) `γ_ν` of `V`: `log R(h) = Σ_{ν≥2} (γ_ν/ν!) h^ν`, with `γ_1 = 0`, `γ_2 = σ²`.

**Prior fragments in the tail.** Several authors had touched pieces of the tail problem:
- For the special case of repeated trials (Bernoulli / binomial), the relative discrepancy `(ν − np)/√(npq)` has a normal limit; its *large* discrepancies were studied by Khinchin (1929) and, via an asymptotic-series refinement `(1 − F_n)/(1 − Φ) = 1 + o(x^{−2s})` valid for `x = o(n^{1/(4s+6)})`, by Smirnov (1933).
- Lévy (1937), again for repeated trials, obtained the logarithmic asymptotic `log(1 − F_n(x)) ∼ log(1 − Φ(x)) ∼ −x²/2`.
- Fréchet (1937) collected results in this direction.

**An actuarial device for tails.** Independently, in collective risk theory, F. Esscher (1932, *Skandinavisk Aktuarietidskrift*) had introduced a re-weighting of a distribution by an exponential factor,

```
f_h(x) = e^{hx} f(x) / ∫ e^{hy} f(y) dy ,
```

as an aid to approximating the tail of an aggregate-claims distribution by a saddle-point / steepest-descent argument. The transformed law is again a bona fide probability distribution; it is concentrated where the original one is rare. This re-weighting existed in the actuarial toolbox as a computational trick for tails.

## Baselines

These are the methods against which a tail asymptotic for sums would be measured.

**1. The Laplace–Lyapunov central limit theorem (with the Lyapunov error bound).**
`F_n(x) → Φ(x)` for fixed `x`; the rescaled sum is asymptotically Gaussian, and `|F_n − Φ| = O(log n / √n)` uniformly. A statement about the *centre* of the distribution, controlling the difference `F_n − Φ`.

**2. Edgeworth / Charlier asymptotic expansions.**
Refine the central limit theorem by expanding the difference `F_n(x) − Φ(x)` in inverse powers of `√n`, with coefficients built from the cumulants of `V` (skewness, kurtosis, …): `F_n(x) = Φ(x) + (correction)/√n + …`. These are additive corrections to the centre.

**3. Khinchin (1929) / Smirnov (1933) / Lévy (1937) — tail results for repeated trials.**
For the binomial (repeated-trials) model, asymptotics of `(1 − F_n)/(1 − Φ)` (Khinchin, Smirnov) and the logarithmic form `log(1 − F_n) ∼ −x²/2` (Lévy). Tied to the two-point `Z`, with `x` growing moderately (e.g. Smirnov requires `x = o(n^{1/(4s+6)})`).

**4. Esscher's exponential re-weighting (1932) as a saddle-point aid.**
Re-weight a claims density by `e^{hx}`, renormalise, and read off a tail estimate by steepest descent around a chosen `h`. A computational device for a *single* aggregate distribution in risk theory.

## Evaluation settings

This is a theoretical result; the natural yardstick is internal consistency and recovery of the known special cases, not a benchmark dataset.

- **Object of study.** The sequence of normed-sum CDFs `F_n(x)` for an i.i.d. sequence with `E(Z) = 0`, `E(Z²) = σ²`, under the light-tail hypothesis `R(h) = ∫ e^{hy} dV(y) < ∞` for `|h| < A`.
- **Regimes, by growing `x`.** Fixed `x` (central limit theorem); `x = o(n^{1/6})`; `x = O(n^{1/6})`; `1 < x = o(√n / log n)` and, under a mild additional regularity hypothesis, `x = o(√n)`; and the large-deviation scale `x = c√n` (sum of order `cσn`).
- **Quantities of interest.** The tail ratios `(1 − F_n(x))/(1 − Φ(x))` and `F_n(−x)/Φ(−x)`; and on the `√n` scale, an asymptotic for `1 − F_n(c√n)` itself.
- **Sanity checks.** Reduce to the central limit theorem in the centre (`x = o(n^{1/6})` ⇒ ratio `→ 1`); recover the binomial tail results of Khinchin, Smirnov, Lévy as the two-point special case; and reproduce, in its homogeneous special case, the actuarial aggregate-claims tail estimate.

## Code framework

No simulation or training is involved; the "code" is the symbolic skeleton of the argument — the named objects that exist before any new method, with one empty slot where the contribution will go. (A tiny numerical sanity check on a known distribution is the only place a computer is natural.)

```python
import numpy as np

# --- Pre-method primitives that already exist -------------------------------

def normal_cdf(x):
    """Φ(x): the central-limit limit CDF."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def mgf(h, sample):
    """R(h) = E[e^{hZ}] from a sample of Z (the moment generating function).
    Finite only for h in a strip |h| < A around 0 (the light-tail hypothesis)."""
    return np.mean(np.exp(h * sample))

def empirical_tail(samples_sum, threshold):
    """Monte-Carlo estimate of Prob(sum > threshold) — the quantity we want to
    describe asymptotically, for sanity checking only."""
    return np.mean(samples_sum > threshold)

# --- The object to be discovered -------------------------------------------

def tail_exponent(c, sample):
    """The exponential rate at which Prob( (Z_1+...+Z_n)/(σ√n) > c√n ) decays in n,
    as a function of the deviation c and the summand distribution.
    TODO: the object we will define here."""
    raise NotImplementedError

# --- A check the discovered object must pass --------------------------------

def central_limit_consistency_check():
    """For small deviations the tail ratio (1 - F_n)/(1 - Φ) must tend to 1.
    The discovered asymptotic has to reduce to this in the appropriate regime."""
    pass
```
