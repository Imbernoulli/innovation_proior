# VC dimension and the uniform convergence of empirical frequencies

## The problem

Choose a decision rule (an event A from a class S of subsets of an input space X) by fitting a finite
i.i.d. sample drawn from an unknown distribution P. Because the rule is chosen *after* seeing the data, a
per-event law of large numbers is useless; what governs whether the fit generalizes is the worst-case
deviation of empirical frequency from true probability over the entire class,

    π^(l) = sup_{A ∈ S} | ν_A^(l) − P_A |,   ν_A^(l) = n_A / l.

The question: for which classes S does π^(l) → 0, and can the condition and rate be made independent of P?

## The key idea

Generalization is not controlled by the *number* of rules (infinite for half-spaces) but by a
combinatorial **capacity** of the class. On any sample of l points a class can only realize finitely many
labelings, so the effective "number of events" is the **growth function**, and its growth rate is pinned
down by the single integer **VC dimension** — the size of the largest set the class can *shatter*. A
finite VC dimension is exactly what makes the worst event's empirical frequency converge to its
probability uniformly, distribution-free.

## The objects

**Index / growth function.** For a sample x₁,…,x_l, let Δ^S(x₁,…,x_l) be the number of distinct labelings
(𝟙_A(x₁),…,𝟙_A(x_l)) ∈ {0,1}^l realized by sets A ∈ S. The growth function is the worst case:

    m^S(l) = max_{x₁,…,x_l} Δ^S(x₁,…,x_l)   (always ≤ 2^l).

**Shattering and VC dimension.** S *shatters* a set of k points if it realizes all 2^k labelings of them
(Δ^S = 2^k there). The VC dimension is the size of the largest shattered set:

    d = VC(S) = max { l : m^S(l) = 2^l }   (the "capacity"; ∞ if no maximum).

When d is finite, the index proof uses n = d + 1, the first sample size where full shattering is no
longer possible. This n is the exponent in the power-growth estimate below.

## The theorems

**Growth-function dichotomy.** Let Φ(n,r) be defined by

    Φ(n,r) = Φ(n,r−1) + Φ(n−1,r−1),    Φ(0,r)=1, Φ(n,0)=1.

Lemma 1 says: if a sample of size r has Δ^S ≥ Φ(n,r), then some n-point subsample is shattered.

*Proof of Lemma 1.* Induct on r. Drop the last point x_r. On the first r−1 points, split the induced
labelings into a type with only one extension to x_r and a type with both extensions. If a is the number
of one-extension labelings and b the number of two-extension labelings, then

    Δ^S(x₁,…,x_{r−1}) = a + b,       Δ^S(x₁,…,x_r) = a + 2b.

Let S″ be the subfamily represented by the two-extension labelings. If S″ shattered n−1 of the first r−1
points, adding x_r would give an n-point shattered set for S. Therefore, under the contradiction hypothesis,
the induction hypothesis gives b < Φ(n−1,r−1); and since S itself shatters no n-point subset among the
first r−1 points, a+b < Φ(n,r−1). Hence

    Δ^S(x₁,…,x_r) = (a+b)+b < Φ(n,r−1)+Φ(n−1,r−1) = Φ(n,r),

contradicting Δ^S ≥ Φ(n,r). ∎

If m^S is not identically 2^r and n is the first r with m^S(r) ≠ 2^r, Lemma 1 implies

    m^S(r) < Φ(n,r) < r^n + 1       for r > n.

Thus the index argument gives degree n=d+1. The sharper Sauer form uses the largest shattered size d
directly:

    m^S(l) ≤ Σ_{k=0}^{d} C(l, k) ≤ (e l / d)^d   (l ≥ d ≥ 1).

There is no intermediate (e.g. 2^{√l}) regime: shattering is the obstruction to polynomial growth.

**Uniform convergence bound.** For l ≥ 2/ε²,

    P( sup_{A∈S} | ν_A^(l) − P_A | > ε ) ≤ 4 m^S(2l) e^{−ε² l / 8},

independent of P. If m^S(l) ≤ l^n + 1, the right side is ≤ 4[(2l)^n + 1] e^{−ε²l/8} → 0.

*Proof (symmetrization + permutation).*
1. **Ghost sample.** Draw 2l points; let ν′,ν″ be the frequencies on the two halves and
   ρ^(l) = sup_A |ν′_A − ν″_A|. For l ≥ 2/ε², P(ρ^(l) ≥ ε/2) ≥ ½P(π^(l) > ε). On the event π > ε, pick
   A₀ with |ν′_{A₀} − P_{A₀}| > ε; Chebyshev gives
   P(|ν″_{A₀} − P_{A₀}| > ε/2) ≤ 4P_{A₀}(1−P_{A₀})/(ε²l) ≤ 1/(ε²l) ≤ 1/2, and then
   |ν′_{A₀} − ν″_{A₀}| > ε/2.
2. **Permutation symmetry.** The 2l points are exchangeable, so average over all (2l)! permutations. The
   sup over S depends only on the ≤ Δ^S(x₁,…,x_{2l}) ≤ m^S(2l) distinct labelings; union-bound over them.
3. **Hypergeometric tail.** For a fixed event with m of the 2l points inside, a random equipartition gives
   |ν′ − ν″| = |2k − m|/l, and
   Γ = P(|2k − m| ≥ εl/2) = Σ_k C(m,k)C(2l−m,l−k)/C(2l,l) ≤ 2 e^{−ε²l/8}
   because this is sampling without replacement with deviation |k−m/2| ≥ εl/4. Hence
   P(ρ^(l) ≥ ε/2) ≤ 2m^S(2l)e^{−ε²l/8}; multiply by 2 from step 1. ∎

**Almost-sure convergence and sample size.** If m^S(l) ≤ l^n + 1, the tails are summable, so by
Borel–Cantelli P(π^(l) → 0) = 1; and P(π^(l) > ε) ≤ η whenever

    l ≥ (16/ε²) ( n log(16 n / ε²) − log(η/4) ).

The capacity plays the role that log N plays for a finite class of N rules, with the index proof using
n=d+1 unless a sharper growth estimate is known.

**Necessary and sufficient condition (distribution-dependent).** With the entropy
H^S(l) = E log₂ Δ^S(x₁,…,x_l) (subadditive, so H^S(l)/l → c),

    uniform convergence of ν to P over S  ⇔  H^S(l)/l → 0.

Requiring uniformity over P as well makes VC(S) < ∞ both necessary and sufficient.

**Recovered special case.** For rays S = {x ≤ a} on R, d = 1 and m^S(l) = l + 1, so uniform convergence
holds a.s.; with A = {x ≤ a}, this is P(sup_a |F_l(a) − F(a)| → 0) = 1 — the Glivenko–Cantelli theorem.

## Worked examples of the capacity

- Homogeneous half-spaces {x : ⟨w,x⟩ ≥ 0} in R^n: VC dimension = n.
- Fixed-threshold linear half-spaces {x : ⟨w,x⟩ ≥ 1} in R^n: the parameter-space cell count is Φ(n,r), and
  the capacity is n.
- Affine half-spaces {x : ⟨w,x⟩ + b ≥ 0} in R^n: VC dimension = n + 1 (exhibit n+1 affinely independent
  points shattered; Radon's lemma shows no n+2 points can be — any n+2 points split into two
  convex-hull-meeting subsets that a half-space cannot separate). More generally, sign-thresholds of a
  k-dimensional vector space of functions have VC dimension ≤ k.
- All subsets of [0,1]: VC dimension = ∞, m^S(l) = 2^l, no uniform convergence.

## Computational formulas

```python
from itertools import product
from math import comb, log, exp

def growth_index(label_fn, S_params, points):
    """Delta^S: number of distinct {0,1}^k labelings sets in S induce on `points`."""
    return len({tuple(int(label_fn(t, x)) for x in points) for t in S_params})

def shatters(label_fn, S_params, points):
    return growth_index(label_fn, S_params, points) == 2 ** len(points)

def vc_dimension(label_fn, S_params, candidate_point_sets):
    """Largest shattered set among the supplied candidate point sets."""
    d = 0
    for pts in candidate_point_sets:
        if shatters(label_fn, S_params, pts):
            d = max(d, len(pts))
    return d

def sauer_bound(d, l):
    """m^S(l) <= sum_{k<=d} C(l,k)  (<= (e l/d)^d for l>=d); 2^l if d is infinite."""
    return 2 ** l if d == float("inf") else sum(comb(l, k) for k in range(d + 1))

def index_growth_bound(n, l):
    """Power-growth bound m^S(l) <= l^n + 1."""
    return l ** n + 1

def vc_uniform_bound_tight(d, l, eps):
    """Distribution-free bound using the sharper Sauer growth estimate."""
    assert l >= 2 / eps**2, "ghost-sample lemma requires l >= 2/eps^2"
    return 4.0 * sauer_bound(d, 2 * l) * exp(-(eps**2) * l / 8.0)

def vc_uniform_bound_index(n, l, eps):
    """Distribution-free bound using m^S(l) <= l^n + 1."""
    assert l >= 2 / eps**2, "ghost-sample lemma requires l >= 2/eps^2"
    return 4.0 * index_growth_bound(n, 2 * l) * exp(-(eps**2) * l / 8.0)

def sample_size_from_power_growth(n, eps, eta):
    """l such that 4[(2l)^n + 1] exp(-eps^2 l/8) <= eta."""
    return (16.0 / eps**2) * (n * log(16.0 * n / eps**2) - log(eta / 4.0))

# rays {x<=a}: vc=1 -> Glivenko-Cantelli
# homogeneous/fixed-threshold half-spaces in R^n: vc=n; affine half-spaces: vc=n+1
# all subsets: vc=inf -> bound never converges.
```
