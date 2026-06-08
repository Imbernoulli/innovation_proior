# The PAC-Bayes bound

## Problem

Bound the generalization gap of a learner that uses a prior over hypotheses, **distribution-free** (no
assumption that the prior matches the data) and without becoming vacuous merely because the hypothesis class is
continuous. The classical single-hypothesis Occam/union bound charges complexity `−ln P(h)` — the prior mass
of one hypothesis — which is `+∞` for continuous classes, and it needs a union bound to hold over the whole
class.

## Key idea

Do not bound a single selected hypothesis; bound a **randomized predictor**, i.e. a posterior distribution
`Q` over hypotheses (predict by drawing `h ∼ Q`). Then the complexity of one hypothesis, `−ln P(h)`, is
replaced by `KL(Q‖P)`, the Kullback–Leibler divergence of the posterior `Q` from a fixed, data-independent
prior `P`. This term is the exact price, by the **Donsker–Varadhan change-of-measure identity**, of
transporting a single concentration statement proven on the prior side over to `Q`. Because that statement is
proven and frozen *before `Q` is chosen*, the bound holds **for all `Q` simultaneously**, even data-dependent
`Q`. `KL(Q‖P)` can be finite for continuous classes, reduces to `−ln P(h)` when `Q` is a point mass, and the
whole bound is computable from the sample, so it can be reported as a certificate and optimized over `Q` rather
than tied to a singleton hypothesis.

## The theorem

Let `ℓ(h, z) ∈ [0,1]`, sample `S` of size `m` i.i.d. from any `D`, `ℓ(h) = E_{z∼D} ℓ(h,z)`,
`ℓ̂(h, S) = (1/m) Σ_{z∈S} ℓ(h,z)`, `ℓ(Q) = E_{h∼Q} ℓ(h)`, `ℓ̂(Q,S) = E_{h∼Q} ℓ̂(h,S)`,
`KL(Q‖P) = E_{h∼Q} ln(dQ/dP)`.

**Square-root form.** For any prior `P` and any `δ ∈ (0,1)`, with probability `≥ 1 − δ` over `S`,
simultaneously for all posteriors `Q`:

```
ℓ(Q) ≤ ℓ̂(Q, S) + sqrt( ( KL(Q‖P) + ln(1/δ) + ln m + 2 ) / (2m − 1) ).
```

**kl-form.** With `kl(q, p) = q ln(q/p) + (1−q) ln((1−q)/(1−p))` the Bernoulli relative entropy, with
probability `≥ 1 − δ` over `S`, simultaneously for all `Q`:

```
kl( ℓ̂(Q, S),  ℓ(Q) ) ≤ ( KL(Q‖P) + ln((m+1)/δ) ) / m
```

(The sharper exponential-moment refinement replaces `ln(m+1)` by `ln(2√m)` for `m ≥ 8`). Inverting `kl` in
its second argument gives an explicit upper bound
`ℓ(Q) ≤ kl^{-1}( ℓ̂(Q,S); (KL(Q‖P)+ln((m+1)/δ))/m )`. By Pinsker `kl(q,p) ≥ 2(p−q)²`, the kl-form also gives
the explicit corollary `ℓ(Q) − ℓ̂(Q) ≤ sqrt((KL(Q‖P)+ln((m+1)/δ))/(2m))`; the displayed kl-form is tighter
when `ℓ̂(Q)` is small.

## Proof (kl-form; the square-root form follows the same template with `(2m−1)Δ²` in place of `m·kl`)

1. **Concentration on the prior.** For a fixed `h` with `p = ℓ(h)`, in the 0-1 case `m·ℓ̂(h) ∼ Binomial(m, p)`,
   and the `p`-dependence cancels:
   `E_S e^{m·kl(ℓ̂(h),p)} = Σ_{j=0}^m C(m,j)(j/m)^j(1−j/m)^{m−j} ≤ m+1`, using
   `C(m,j)(j/m)^j(1−j/m)^{m−j} = C(m,j) e^{−mH(j/m)} ≤ 1`. For `[0,1]` losses, convexity reduces the moment
   to the Bernoulli case, so the `m+1` bound still applies; the sharper refinement is
   `E_S e^{m·kl(ℓ̂(h),p)} ≤ 2√m` for `m ≥ 8`. (The square-root route instead uses Chernoff
   `Pr(Δ≥x) ≤ 2e^{−2mx²}` and the tail-integral identity:
   `E_S e^{(2m−1)Δ²} ≤ 1 + 4(2m−1)∫_0^∞ x e^{−x²} dx = 4m−1 ≤ 4m`.)
2. **Average over `P`, then Markov.** By Fubini (`P` is data-independent), `E_S E_{h∼P} e^{m·kl} ≤ m+1`, so
   `Pr_S( E_{h∼P} e^{m·kl(ℓ̂(h),ℓ(h))} > (m+1)/δ ) ≤ δ`. Fix any `S` on the complementary event.
3. **Change of measure (Donsker–Varadhan / Gibbs).** With `dP_G ∝ e^{m·kl} dP`, nonnegativity of
   `KL(Q‖P_G)` expands to
   `0 ≤ KL(Q‖P) + ln E_{h∼P} e^{m·kl} − E_{h∼Q}[m·kl]`, i.e.
   `E_{h∼Q} kl(ℓ̂(h),ℓ(h)) ≤ ( KL(Q‖P) + ln((m+1)/δ) ) / m`. (Equivalently, solving
   `max_y Σ Q_i y_i` s.t. `Σ P_i e^{β y_i} ≤ K` by Lagrange/Kuhn–Tucker gives
   `e^{βy_i}=KQ_i/P_i`, equivalently `Q_i ∝ P_i e^{βy_i}`, and value `(KL(Q‖P)+ln K)/β`.)
4. **Jensen.** `kl` is jointly convex, so `kl(E_Q ℓ̂, E_Q ℓ) ≤ E_Q kl(ℓ̂, ℓ)`, yielding the stated bound.
   (Square-root form: `(E_Q Δ)² ≤ E_Q Δ²` plus `ℓ(Q)−ℓ̂(Q) ≤ E_Q|ℓ−ℓ̂| = E_Q Δ`.) ∎

## Optimal posterior

For the square-root objective, write `K = ln(1/δ)+ln m+2` and `γ = 2m−1`. In a finite class, the KKT condition
for an interior optimum is
`λ = ℓ̂_i + (1 + ln(Q_i/P_i))/(2 sqrt(γ(KL(Q‖P)+K)))`, so
`Q_i ∝ P_i e^{−βℓ̂_i}`. Thus the optimum has the Gibbs form
`dQ_β(h) ∝ e^{−β ℓ̂(h,S)} dP(h)` with the self-consistency condition
`β = 2 sqrt((2m−1)(KL(Q_β‖P)+ln(1/δ)+ln m+2))`. As `β→0` it returns the prior; as `β→∞` it concentrates on
the empirical risk minimizer. This is the exponentially-weighted aggregate / Boltzmann weighting.

## Computing the certificate

```python
import numpy as np

def kl_div(Q, P):                      # KL(Q || P): the complexity term
    Q = np.asarray(Q, float); P = np.asarray(P, float)
    if np.any((Q > 0) & (P == 0)):
        return float("inf")
    mask = Q > 0
    return float(np.sum(Q[mask] * np.log(Q[mask] / P[mask])))

def kl_ber(q, p):                      # Bernoulli relative entropy
    q = float(np.clip(q, 0.0, 1.0)); p = float(np.clip(p, 0.0, 1.0))
    if p == 0.0:
        return 0.0 if q == 0.0 else float("inf")
    if p == 1.0:
        return 0.0 if q == 1.0 else float("inf")
    out = 0.0
    if q > 0: out += q * np.log(q / p)
    if q < 1: out += (1 - q) * np.log((1 - q) / (1 - p))
    return float(out)

def kl_inverse(q, c):                  # largest p >= q with kl(q || p) <= c
    q = float(np.clip(q, 0.0, 1.0))
    if c < 0:
        raise ValueError("kl radius must be nonnegative")
    if q >= 1.0 or c == 0:
        return q
    lo, hi = q, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if kl_ber(q, mid) > c else (hi, mid)
    return lo

def pac_bayes_certificate(Q, P, emp_risks, m, delta, form="kl"):
    Q = np.asarray(Q, float); P = np.asarray(P, float)
    KL = kl_div(Q, P); rhat = float(np.dot(Q, emp_risks))
    if form == "sqrt":
        return rhat + np.sqrt((KL + np.log(1/delta) + np.log(m) + 2) / (2*m - 1))
    return kl_inverse(rhat, (KL + np.log((m + 1) / delta)) / m)

def gibbs_posterior(P, emp_risks, beta):
    P = np.asarray(P, float); emp_risks = np.asarray(emp_risks, float)
    with np.errstate(divide="ignore"):
        log_w = np.log(P) - beta * emp_risks
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()
```

The bound holds with probability `≥ 1 − δ` simultaneously for every `Q`, is distribution-free, can remain finite
for continuous classes (`KL(Q‖P) < ∞` where `−ln P(h) = ∞`), and — being empirical — can be minimized over `Q`
to seek a non-vacuous generalization certificate alongside the learned predictor.
