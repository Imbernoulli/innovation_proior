# Context: distributing number sequences uniformly modulo one

## Research question

Take an infinite sequence of real numbers α₁, α₂, α₃, … and look only at their
fractional parts {αₙ} = αₙ − ⌊αₙ⌋, i.e. reduce them modulo 1 and lay them on a
circle of circumference 1. For a subarc (or subinterval) **a** of length |a|,
let n_a be how many of the first n points fall in a. We want to decide, for a
given sequence, whether the points spread out *evenly*:

> for every subinterval a ⊆ [0,1),  lim_{n→∞} n_a / n = |a|.

A sequence with this property is called **uniformly distributed (gleichmäßig
dicht) mod 1**. The question is sharp and concrete: *which* sequences have it,
and how does one prove it for a given one? The motivating examples come from
celestial mechanics and Diophantine approximation — the angular positions
{nξ} of a body returning at an incommensurable period, the values of a
polynomial p(n) reduced mod 1, the fractional parts {√n}, {log n}, {nξ²}. For
each, the bare definition asks us to control a *counting* limit over arbitrary
intervals, simultaneously for all intervals — a stiff, combinatorial demand.
A usable theory would replace that demand with something one can actually
compute or bound for a concrete sequence.

Why it matters: uniform distribution is the deterministic analogue of "the
sequence behaves like independent uniform samples." It underlies error terms in
analytic number theory (sums Σ e(p(n)) over polynomial phases, which control,
e.g., the growth of the Riemann ζ-function on the line Re s = 1), the
ergodic/statistical-mechanics picture of a point moving on a torus, and
numerical integration by averaging a function over the sequence.

## Background

**The defining notion and its measure-theoretic content.** Saying every arc a
gets its fair share n_a/n → |a| is, by a standard sandwich, the same as saying
that *averaging any reasonable function along the sequence equals integrating it*:

  (1/n) Σ_{h=1}^n f(αₕ)  →  ∫₀¹ f(x) dx.

The bridge is Riemann integrability: a bounded Riemann-integrable f can be
trapped between two step functions f₁ ≤ f ≤ f₂ whose integrals differ by less
than ε, and a step function is a finite combination of interval indicators χ_a.
So the interval-counting statement and the function-averaging statement are
equivalent; the indicator of an interval is the discontinuous, rigid end of a
spectrum whose smooth end is the continuous (and trigonometric) functions.

**Fourier analysis, ca. 1916.** The theory of Fourier series is mature.
Every continuous 1-periodic function is a uniform limit of finite trigonometric
polynomials a₀/2 + Σ_{k=1}^m (a_k cos 2πkx + b_k sin 2πkx) — by Weierstrass's
approximation theorem and, constructively, by Fejér's theorem (Cesàro means of
the Fourier series converge uniformly for continuous f). In complex form the
building blocks are the exponentials x ↦ e^{2πikx}; we abbreviate e(x) =
e^{2πix}. Their integrals over a period are trivial: ∫₀¹ e(kx) dx = 1 if k = 0
and 0 if k ≠ 0.

**What is already known about which sequences distribute evenly.**

- *Linear sequences.* For ξ irrational, the sequence {nξ} is uniformly
  distributed mod 1. This was established around 1909–1910, independently, by
  Bohl, by Sierpiński, and by Weyl, by elementary interval-chasing arguments
  (e.g. Bohr's three-distance / two-interval comparison): one shows that two
  equal-length subintervals capture asymptotically equally many of the points
  1·ξ, 2·ξ, …, nξ. These arguments are real proofs but are tied to the line
  structure of nξ; they do not obviously survive when the dependence on n is
  nonlinear or when several coordinates move at once.

- *Density (not distribution) for several coordinates.* Kronecker's
  approximation theorem (1884), born from the problem of approximate integer
  solution of linear equations and from secular perturbations in astronomy,
  says: if ξ₁,…,ξ_p together with 1 are linearly independent over the rationals,
  then the points (nξ₁,…,nξ_p) mod 1 are *dense* in the p-dimensional torus —
  they come arbitrarily close to every target. Density is qualitative: it says
  the orbit gets close, not how *often* it visits a given region.

**The frontier: polynomial phases.** Beyond the linear case the picture is
fragmentary. Hardy and Littlewood, in a 1912 Cambridge lecture and a 1914 Acta
Mathematica paper ("Some problems of Diophantine approximation"), proved that
for ξ irrational the *quadratic* sequence {n²ξ} is uniformly distributed, by a
delicate argument resting on Cauchy's integral theorem applied to a generating
function. Their underlying motive was analytic: to bound exponential sums of
the form Σ e(n²ξ) and thereby control ζ(1 + ti) = o(log t). Their method is
heavy and special to low degree; for a general polynomial
p(z) = α_q z^q + … + α_1 z + α_0 with some non-constant coefficient irrational,
no clean treatment exists. The continuous analogue is easy by contrast: for any
non-constant polynomial φ, the integral ∫₀ᵗ e(φ(t)) dt stays bounded —
substitute φ(t) = x and integrate by parts, using that ∫|φ″/φ′²| converges — so
the *continuous* average of e(φ) decays like O(1)/t. The trouble is purely that
the discrete sum Σ_{h} e(φ(h)) is not an integral and has no closed form when
deg φ ≥ 2.

## Baselines

**Bohl / Sierpiński / Weyl (1909–1910), linear case.** Claim: ξ irrational ⇒
{nξ} uniformly distributed. Method (Bohr's elementary version): fix a large
integer H; take two equal-length intervals [1] and [2]; choose L so that
α₁ + Lε is congruent mod 1 to a point of [2]; compare the counts n₁, n₂ of the
first n multiples falling in [1] and [2] and show |n₁ − n₂| stays below a fixed
bound C, then ≤ n/H + const, hence (n₁ − n₂)/n → 0; since this holds for any
two equal intervals, each interval gets its fair share.

**Kronecker (1884), approximation theorem.** Claim: 1, ξ₁,…,ξ_p rationally
independent ⇒ {(nξ₁,…,nξ_p)} dense in the torus. Method: approximate integer
solution of linear forms; pigeonhole / continued-fraction style approximation.
The conclusion is that the orbit gets arbitrarily close to every target point.

**Hardy & Littlewood (1912/1914), quadratic case.** Claim: ξ irrational ⇒
{n²ξ} uniformly distributed; underlying goal, bound Σ e(n²ξ) to control
ζ(1+ti). Method: a generating-function / Cauchy-integral-theorem argument
specific to the quadratic phase.

**The continuous integral ∫₀ᵗ e(φ(t)) dt = O(1).** Claim/method: for
non-constant polynomial φ, substitute φ(t)=x and integrate by parts; ∫|φ″/φ′²|
converges, so the integral is bounded.

## Evaluation settings

The natural objects on which any proposed criterion or theorem would be tried
out — the standard yardsticks of the subject:

- **Sequences to test.** {nξ} for ξ irrational and for ξ rational (the latter
  is *not* uniformly distributed — it cycles through finitely many residues);
  {n²ξ}, {n^k ξ}; general polynomial values {p(n)}; {√n}; {log n} and
  {log p_n} (p_n the n-th prime); joint sequences ((nξ₁,…,nξ_p)) on the torus.
- **The yardstick of even spreading.** For a candidate sequence one looks at,
  for arbitrary subintervals [b,c] ⊆ [0,1), the empirical frequency
    #{h ≤ N : {αₕ} ∈ [b,c]} / N
  against the target length c − b; equivalently the star-discrepancy
    D*_N = sup_{0≤c≤1} | #{h ≤ N : {αₕ} ∈ [0,c]} / N − c |,
  whose vanishing as N → ∞ is the quantitative form of uniform distribution.
- **Computable surrogates.** Beyond raw interval counts, any fixed
  Riemann-integrable test function f gives the averaged quantity
    A_N(f) = (1/N) Σ_{h=1}^N f(αₕ)
  to compare against ∫₀¹ f — the empirical mean of f along the sequence, which
  is directly computable for a concrete f and a concrete sequence.
- **Ambient domains.** the circle ℝ/ℤ (one coordinate), the p-dimensional
  torus ℝ^p/ℤ^p, and — for moving-point versions — the continuous-time path of
  a point in a cube/torus.

## Code framework

A pre-method scaffold in terms only of primitives that already exist:
fractional parts, an arc/interval counter, the discrepancy, and a generic
"average a test function along the sequence" routine. The single thing not yet
known is *what condition to test* in order to certify uniform distribution —
that is the empty slot.

```python
import math

def frac(x):
    "fractional part {x} in [0,1)"
    return x - math.floor(x)

def empirical_frequency(seq, N, b, c):
    "fraction of the first N reduced points {seq(n)} landing in [b, c)"
    return sum(1 for n in range(1, N + 1) if b <= frac(seq(n)) < c) / N

def star_discrepancy(seq, N, grid=1000):
    "D*_N = sup_c | #{n<=N: {seq(n)} in [0,c)}/N - c |  (the quantitative gap)"
    pts = sorted(frac(seq(n)) for n in range(1, N + 1))
    d, j = 0.0, 0
    for i in range(grid + 1):
        c = i / grid
        while j < len(pts) and pts[j] < c:
            j += 1
        d = max(d, abs(j / N - c))
    return d

def average_along_sequence(seq, f, N):
    "A_N(f) = (1/N) sum_{h=1}^N f(seq(h)) -- empirical mean of a test function f"
    return sum(f(seq(h)) for h in range(1, N + 1)) / N

def is_uniformly_distributed(seq, N):
    """Decide uniform distribution of {seq(n)} mod 1 from computable data."""
    # TODO: fill in the condition and the argument that certifies it
    pass
```
