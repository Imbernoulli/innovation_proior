# Cramér's theorem (large deviations)

## The problem it solves

For an i.i.d. sum `S_n = X_1 + … + X_n`, the central limit theorem describes fluctuations of order `√n` around the mean. It says nothing about *large* deviations — the probability that `S_n` reaches order `n`, i.e. that the empirical mean `S_n/n` lands a fixed distance away from `E[X_1]`. That tail probability is exponentially small in `n`; the question is to find its exact exponential rate (and, sharply, its prefactor). This is the quantity that governs rare-but-ruinous events — e.g. an insurer's probability of a catastrophically large total claim.

## The key idea

The central limit theorem and the law of large numbers are sharp only at the *centre* of a distribution. The rare event lives in the tail. So **exponentially tilt** the law of `X` (the Esscher transform) to slide the rare value into the centre, where the central tools apply; the cost of un-tilting is an explicit exponential factor whose rate is the **Legendre transform** of the cumulant generating function. The upper bound on the probability comes for free from Markov's inequality applied to `e^{tS_n}` with the tilt parameter `t` optimized (the Chernoff bound); the matching lower bound comes from the change of measure (tilting); the two coincide by convex duality.

## Statement (logarithmic / large-deviation-principle form)

Let `X_1, X_2, …` be i.i.d. real random variables with **cumulant generating function**

```
Λ(t) = log E[ e^{t X_1} ],
```

finite in a neighbourhood of `t = 0` (the **Cramér condition** — equivalently `E[e^{tX_1}] < ∞` for `|t| < A`, the light-tail hypothesis). Let `m = E[X_1] = Λ'(0)` and define the **rate function** as the Legendre–Fenchel transform

```
Λ*(a) = sup_{t ∈ ℝ} [ t a − Λ(t) ].
```

Then for every `a > m`,

```
lim_{n→∞} (1/n) log P( X_1 + … + X_n ≥ n a ) = − Λ*(a).
```

(Symmetrically, `lim (1/n) log P(S_n ≤ na) = −Λ*(a)` for `a < m`.) More generally the laws of `S_n/n` satisfy a large deviation principle: for closed `C`, `limsup (1/n) log P(S_n/n ∈ C) ≤ −inf_C Λ*`; for open `G`, `liminf (1/n) log P(S_n/n ∈ G) ≥ −inf_G Λ*`.

Properties of `Λ*`: convex, nonnegative, lower-semicontinuous, with `Λ*(m) = 0` and `Λ*(a) > 0` for `a ≠ m`. (`Λ` is convex with `Λ(0)=0`, `Λ'(0)=m`; by Jensen `Λ(t) ≥ tm`, so `tm − Λ(t) ≤ 0` with equality at `t=0`, giving `Λ*(m)=0`.) For `a > m` the supremum is attained at the unique `t* > 0` with `Λ'(t*) = a`, so `Λ*(a) = sup_{t≥0}(ta − Λ(t))`.

## Proof

Fix `a > m` and let `S_n = X_1 + … + X_n`.

**Upper bound (Markov / Chernoff).** For any `t ≥ 0`, the map `s ↦ e^{ts}` is increasing, so `{S_n ≥ na} = {e^{tS_n} ≥ e^{tna}}` and Markov's inequality gives

```
P(S_n ≥ na) ≤ e^{−tna} E[ e^{tS_n} ] = e^{−tna} (E[e^{tX_1}])^n = e^{−n( ta − Λ(t) )}.
```

This holds for every `t ≥ 0`; optimizing,

```
P(S_n ≥ na) ≤ exp[ −n · sup_{t≥0}( ta − Λ(t) ) ] = e^{−n Λ*(a)},
```

since for `a > m` the unrestricted sup equals the sup over `t ≥ 0`. Hence
`limsup_{n→∞} (1/n) log P(S_n ≥ na) ≤ −Λ*(a).`

**Lower bound (exponential change of measure / tilting).** It suffices to bound `P(S_n ∈ [na, na+nδ))` for arbitrarily small `δ > 0`. Assume first that `Λ` is finite in a neighbourhood of `t*` (true when `a` is strictly below the essential supremum of `X`); let `t* > 0` be the unique root of `Λ'(t*) = a`. Define the **tilted law**

```
dμ_{t*}(x) = e^{ t* x − Λ(t*) } dμ(x),
```

a probability measure (`∫ e^{t*x} dμ = e^{Λ(t*)}`) whose mean is

```
∫ x dμ_{t*}(x) = E[X e^{t*X}] / E[e^{t*X}] = Λ'(t*) = a.
```

Rewrite the probability under the tilt:

```
P(S_n ∈ [na, na+nδ)) = ∫_{Σx_i ∈ [na, na+nδ)} ∏_{i} dμ(x_i)
                     = e^{ n Λ(t*) } ∫_{Σx_i ∈ [na, na+nδ)} e^{ −t* Σ x_i } ∏_i dμ_{t*}(x_i).
```

On the event, `Σ x_i < n(a+δ)`, and `t* ≥ 0`, so `e^{−t* Σ x_i} ≥ e^{−t* n(a+δ)}`. Therefore

```
P(S_n ∈ [na, na+nδ)) ≥ e^{ n Λ(t*) } e^{ −t* n(a+δ) } · μ_{t*}^{⊗n}( S_n/n ∈ [a, a+δ) ).
```

Under `μ_{t*}` the i.i.d. variables have mean exactly `a` and finite variance `Λ''(t*)`, so `S_n/n → a`; since `a` is the mean, the one-sided slab `[a, a+δ)` keeps a probability bounded away from `0` (by the central limit theorem `μ_{t*}^{⊗n}( S_n/n ∈ [a, a+δ) ) → 1/2`, and in any case `≥ const > 0`) as `n → ∞`. Taking `(1/n) log`, the constant factor drops out:

```
liminf_{n→∞} (1/n) log P(S_n ≥ na)
   ≥ Λ(t*) − t*(a+δ) = −( t* a − Λ(t*) ) − t* δ = −Λ*(a) − t* δ.
```

Letting `δ → 0` gives `liminf (1/n) log P(S_n ≥ na) ≥ −Λ*(a)`.

**Unbounded support.** If `Λ(t) = ∞` for some `t > 0` (or `t*` not interior), truncate: for `A > 0` let `X^A` be `X` conditioned on `{|X| ≤ A}`, with `Λ_A(t) = log E[e^{tX} 1_{|X|≤A}] − log P(|X|≤A) ≤ Λ(t)`. The bounded-support argument applies to `X^A`, and `Λ_A ↑ Λ` (hence `Λ_A^*(a) ↓ Λ*(a)`) as `A → ∞`; passing to the limit recovers the bound for `X`. ∎

Combining the two bounds: `lim_{n→∞} (1/n) log P(S_n ≥ na) = −Λ*(a)`.

## Sharp (original) form, with the prefactor

The tilting argument can be pushed to recover the exponentially-decaying probability *with its multiplicative prefactor*, not just the rate. Normalize `E[X] = 0`, `E[X²] = σ²`, write `R(h) = E[e^{hX}]` and `log R(h) = Σ_{ν≥2}(γ_ν/ν!) h^ν` (cumulants `γ_ν`). The Esscher-tilted law `dV̄ = e^{hy}dV/R` has mean `m̄ = (d/dh)log R` and variance `σ̄² = d m̄/dh`; `m̄` is continuous and strictly increasing in `h` with `m̄(0)=0`. The exact identity (via characteristic functions)

```
1 − F_n(x) = R^n e^{−h m̄ n} ∫_{(σx − m̄√n)/σ̄}^{∞} e^{−h σ̄ √n y} dF̄_n(y),   F_n(x)=P(S_n ≤ σx√n),
```

with `F̄_n` the normed tilted sum, places the rare value at the centre of `F̄_n` when `σx = m̄√n`, so the central limit theorem applies to `F̄_n`. Under the additional smoothness condition that `V` has a nontrivial absolutely-continuous component (`β > 0` in `V = βU_1 + (1−β)U_2`), the central-limit error is `O(1/√n)` and one obtains: there is `C_1 > 0` such that for `0 < c < C_1`,

```
1 − F_n(c√n) = (1/√n) e^{−α n} [ b_0 + b_1/n + … + b_{k−1}/n^{k−1} + O(1/n^k) ],
   b_0 = 1/( h σ̄ √(2π) ) > 0,
```

where `h` is the unique root of `m̄ = σc` and the rate is

```
α = h m̄ − log R(h)   evaluated at that root,   i.e.   α = sup_h [ h σc − log R(h) ] = Λ*(σc) > 0.
```

The exponential rate is exactly the Legendre transform of the cumulant generating function evaluated at the deviation; the logarithmic form above is its `(1/n) log` shadow.

## Why each piece is there

- **Cramér condition (`Λ` finite near `0`).** Light tails are necessary for an exponential rate; without a finite m.g.f. the rate function and the tilt do not exist, and large deviations of heavy-tailed sums are polynomial (dominated by one large summand).
- **Tilt `dμ_t = e^{tx−Λ(t)}dμ`.** Re-centres the law so the rare value becomes the typical value, importing the law-of-large-numbers/central-limit sharpness into the tail.
- **Tilt parameter = saddle (`Λ'(t*) = a`).** The unique tilt making `a` the new mean; any other tilt leaves the event in the tail.
- **Legendre transform `Λ*`.** Forced by optimizing the free `t` in the Chernoff bound (upper) and equalling the un-tilting cost at the saddle (lower); the two agree because `Λ`, `Λ*` are convex conjugates and the optimum is interior. `Λ*(m)=0`, `Λ*>0` elsewhere.

## Worked example (Bernoulli)

For `X_1 ∈ {0,1}` with `P(X=1) = p`, `Λ(t) = log(1 − p + p e^t)`, and

```
Λ*(a) = a log(a/p) + (1−a) log( (1−a)/(1−p) ),   0 ≤ a ≤ 1,
```

the relative entropy (Kullback–Leibler divergence) of `Bernoulli(a)` from `Bernoulli(p)`. So `P(S_n ≥ na) = e^{−n[ a log(a/p) + (1−a) log((1−a)/(1−p)) ] + o(n)}` for `a > p` — the exponential rate of the binomial tail, recovering the classical Chernoff bound for coin tossing as a special case.

The rate function is the Legendre transform of the cumulant generating function — directly computable from a sample of the summand:

```python
import numpy as np
from scipy.optimize import minimize_scalar

def mgf(t, sample):
    return np.mean(np.exp(t * sample))                      # R(t) = E[e^{tZ}]

def rate_function(a, sample):
    """Λ*(a) = sup_t ( t a − log R(t) ): the exponential rate of P(S_n/n ≈ a)."""
    Lambda = lambda t: np.log(mgf(t, sample))               # Λ(t) = log E[e^{tZ}]
    obj = lambda t: -(t * a - Lambda(t))                    # negate to maximize
    res = minimize_scalar(obj)
    return -res.fun                                          # = sup_t (t a − Λ(t))
```

`rate_function` fills the one empty slot of the pre-method scaffold: the exponential rate at which the sum's tail decays is the Legendre transform of the cumulant generating function, evaluated at the deviation.
