The thing I actually need is a number. An insurance company is the sum of its claims; over a year the net result is `Z_1 + … + Z_n`, a sum of many independent individual outcomes, and the question that keeps me up is not what the *typical* total is but what the probability is of a *catastrophically large* total — the event that wipes out the reserves. So I want `Prob(Z_1 + … + Z_n` large`)`, and "large" here means large on the scale of `n`, not on the scale of `√n`. And I do not just want to know that this is "small"; I want its order of magnitude, because that is what reserves are sized against.

Let me set it up cleanly. The `Z_k` are i.i.d., common CDF `V`, and I will center and scale them so `E(Z) = 0`, `E(Z²) = σ² > 0`. Write `F_n(x) = Prob((Z_1 + … + Z_n)/(σ√n) ≤ x)`. The central limit theorem — Laplace–Lyapunov, in Lindeberg's and Lévy's sharpened form — tells me `F_n(x) → Φ(x)` for each fixed `x`, with `Φ` the standard normal CDF. And there is even a rate: Lyapunov gives `|F_n(x) − Φ(x)| ≤ k·(log n)/√n` uniformly, with `k = (3/σ³)∫|y|³ dV(y)`. That feels like it should settle everything.

It doesn't. Watch what happens when I let `x` grow with `n`, which is exactly what "large total" forces. As `x → +∞`, `F_n(x) → 1` and `Φ(x) → 1`. So the CLT statement `F_n(x) → Φ(x)` becomes `1 → 1`. Empty. The theorem has nothing to say once the deviation runs off to infinity with the sample size. The whole content of the tail has slipped out between my fingers.

So I should not look at `1 − F_n(x)` against zero; I should look at it against its natural Gaussian comparison. The honest object is the *ratio*

```
(1 − F_n(x)) / (1 − Φ(x))     and on the left,  F_n(−x) / Φ(−x).
```

If these ratios stayed near `1`, the Gaussian tail would be the answer; the question is whether they do, and if not, how they blow up. Let me see how far the CLT carries them. I have `|F_n − Φ| = O(log n/√n)`, and the Gaussian tail is `1 − Φ(x) ≈ e^{−x²/2}/(x√(2π))`. So the ratio is `1 + O((log n/√n) · x√(2π) e^{x²/2})`. For that error to stay small I need `e^{x²/2}` to be killed by `√n`, i.e. `x²/2 ≲ ½ log n`, i.e. `|x| ≲ √(log n)` (up to the constant). That's a microscopic window. Past `x` of order `√(log n)` — and certainly anywhere near `x ~ √n`, which is where the *sum* reaches order `n` — the central limit theorem is silent. Subtracting `Φ` from `F_n` controls the centre and learns nothing about the tail ratio. I am chasing the wrong difference.

What's the temptation here? Edgeworth / Charlier expansions — refine the CLT by expanding `F_n − Φ` in powers of `1/√n`, with cumulant coefficients. Those expansions are familiar ground, and they correct the centre beautifully. But stare at what the tail correction would have to be. If it exists at all, the ratio `(1−F_n)/(1−Φ)` deep in the tail is not `1 + small`; whatever it is, it has to undo the entire Gaussian tail `e^{−x²/2}` and replace it with the true tail. That is a *multiplicative*, exponentially large object. An additive expansion of `F_n − Φ` in `1/√n` cannot produce a factor like `e^{(something)·n}`. So expansions of the difference are structurally the wrong frame for the tail. Wall.

Let me also be honest about how far I can even hope to go. Suppose `V` is supported in a bounded interval `(−μσ, μσ)`. Then the sum lives in `(−μ√n, μ√n)` after norming, and `1 − F_n(x)` is *identically zero* for `x > μ√n`. So there is no universal simple asymptotic for arbitrarily large `x`; the right ambition is the scale `x ~ √n`, the sum at order `n`. Good — that is exactly the risk scale. That's the target.

And one more honest restriction. For the right tail to decay at a clean exponential rate in `n`, I'd better have summands whose own tails are lighter than any polynomial. The clean way to say "light tail" is that the integral

```
R(h) = ∫ e^{hy} dV(y)
```

converges for `h` in some interval around `0`, say `|h| < A`. Call this the basic condition. It's a real restriction — heavy-tailed sums are dominated by a single huge summand and behave completely differently — but it is the regularity the tail rate will be built from. Note `log R(h)` is the generating function of the cumulants `γ_ν` of `V`: `log R(h) = Σ_{ν≥2} (γ_ν/ν!) h^ν`, with `γ_1 = 0` because `E(Z)=0`, and `γ_2 = σ²`.

Now I'm stuck on the real difficulty. The CLT only works at the *centre* of a distribution. My event — sum at order `n`, i.e. `x ~ √n` — is far out in the tail of `F_n`. The CLT is a tool that's only sharp where I'm not. How do I bring the only sharp tool I have to bear on a place where it isn't sharp?

Turn the sentence around. The CLT is sharp at the centre. So if I could *move the centre* of the distribution out to where my rare event is, the rare event would become a *typical* event, and the CLT would apply there. I don't get to move the centre of `V` — `V` is the data. But I can study a *different* distribution, one that agrees with `V` in shape but is shifted so that the rare region of `V` is the typical region of the new one. And there is a device sitting right there in the actuarial toolbox for exactly this: Esscher's exponential re-weighting. Esscher, in 1932, re-weighted a claims density by `e^{hx}` to push mass into the tail for a saddle-point estimate. Let me steal that and make it the engine.

Pick a real `h` inside the convergence strip `|h| < A`, and define

```
V̄(x) = (1/R) ∫_{−∞}^{x} e^{hy} dV(y),     R = R(h).
```

This is again a genuine CDF: `e^{hy} ≥ 0`, and dividing by `R = ∫ e^{hy} dV` makes it integrate to `1`. The factor `e^{hy}` is increasing in `y` for `h > 0`, so it transfers mass toward `+∞` — it pushes the centre to the right. Exactly the lever I wanted.

Where is the new centre? Let `Z̄` have CDF `V̄`, with mean `m̄ = E(Z̄)` and variance `σ̄² = E((Z̄−m̄)²)`. Compute:

```
m̄ = (1/R) ∫ y e^{hy} dV(y) = (d/dh) log R = Σ_{ν≥2} (γ_ν/(ν−1)!) h^{ν−1},
```

and

```
σ̄² = d m̄/dh = (d²/dh²) log R = Σ_{ν≥2} (γ_ν/(ν−2)!) h^{ν−2}.
```

These are just cumulant-generating-function derivatives — the tilt's mean and variance are the first and second derivatives of `log R`. Near `h = 0`, `m̄` is continuous, and `σ̄² = d m̄/dh > 0` (it's a variance), so `m̄` is strictly increasing in `h`, with `m̄(0) = 0`. So I have a *dial*: turning `h` up from `0` slides the tilted mean `m̄` continuously and monotonically out to the right. For any target deviation I can set the dial so the tilted mean lands exactly there. That monotonicity is what guarantees the dial has a unique setting for each target — I'll lean on it hard.

Now I need the exact bookkeeping that connects `F_n` (what I want) to `F̄_n` (the tilted normed sum, which the CLT will handle). Work with characteristic functions: `ν(z) = ∫ e^{izy} dV(y)`, and similarly `ν̄, w_n, w̄_n` for `V̄, W_n, W̄_n` (where `W_n` is the CDF of the raw sum `Z_1+…+Z_n`). Treat `z = ξ + iη` as complex; under the basic condition `ν(z)` is holomorphic in the strip `|η| < A`. From the definition of `V̄`,

```
R = ν(−ih),     ν̄(z) = ν(z − ih)/R.
```

(The first because `R = ∫ e^{hy} dV = ∫ e^{i(−ih)y} dV = ν(−ih)`; the second by the same shift inside the integral defining `V̄`.) Independence gives `w_n = ν^n` and `w̄_n = ν̄^n`, so

```
w̄_n(z) = ν̄(z)^n = (1/R^n) ν(z − ih)^n = (1/R^n) w_n(z − ih).
```

Replace `z` by `z + ih`:

```
w_n(z) = R^n w̄_n(z + ih) = w̄_n(z + ih)/w̄_n(ih),
```

using `w̄_n(ih) = (1/R^n) w_n(0) = 1/R^n`. This identity between characteristic functions is equivalent to an identity between the CDFs themselves:

```
W_n(x) = R^n ∫_{−∞}^{x} e^{−hy} dW̄_n(y).
```

(Both sides are CDFs in `x`; their characteristic functions are `w_n(z)` and `R^n w̄_n(z + ih)`, which I just showed are equal, so the CDFs coincide.) This is the exact inversion of the tilt at the level of the `n`-fold sum: it un-tilts `W̄_n` back to `W_n`, and the price of un-tilting is the explicit factor `R^n e^{−hy}` — the same exponential weight, raised to the power that matches `n` summands. *This factor is where the whole rare-event cost will be hiding.*

Translate into the normed variables. `F_n(x) = W_n(σx√n)`, and `F̄_n` is the CDF of `(Z̄_1+…+Z̄_n − m̄n)/(σ̄√n)`. A direct substitution gives

```
1 − F_n(x) = R^n e^{−h m̄ n} ∫_{(σx − m̄√n)/σ̄}^{∞} e^{−h σ̄ √n y} dF̄_n(y).
```

Now I can *see* the strategy work. `F̄_n` is the normed sum of the i.i.d. tilted variables `Z̄`, so the CLT applies to *it*: `F̄_n → Φ`, with Lyapunov error `Q_n(y) = F̄_n(y) − Φ(y)`, `|Q_n| < K (log n)/√n`. The clever part is the lower limit of the integral. If I choose `x` so that `σx = m̄√n`, i.e.

```
x = m̄√n/σ,
```

then the lower limit is `0` — the *centre* of `F̄_n`. The rare event of `F_n` has been dialed into the centre of `F̄_n`. The CLT is now sitting exactly where it's sharp.

Do the integral at `x = m̄√n/σ`:

```
1 − F_n(m̄√n/σ) = R^n e^{−h m̄ n} ∫_0^∞ e^{−h σ̄ √n y} dF̄_n(y).
```

Replace `F̄_n = Φ + Q_n`. The `dΦ` piece is `(1/√(2π)) ∫_0^∞ e^{−hσ̄√n y − y²/2} dy`, a Gaussian-against-exponential integral; the `dQ_n` piece is `O(log n/√n)` after an integration by parts, since `|Q_n| < K log n/√n` and `hσ̄√n ∫_0^∞ e^{−hσ̄√n y} dy = 1`. Keep the dominant term. In the regime `h → 0`, `n → ∞` with `h√n` bounded below, `hσ̄√n = m̄√n/σ + O(h²√n)`, and the integral becomes

```
(1/√(2π)) ∫_0^∞ e^{−(m̄√n/σ) y − y²/2} dy = e^{n m̄²/(2σ²)} [1 − Φ(m̄√n/σ)],
```

by completing the square. Putting it back,

```
(1 − F_n(m̄√n/σ)) / (1 − Φ(m̄√n/σ)) = e^{ n( m̄²/(2σ²) − h m̄ + log R ) } · [1 + O(h log n)].
```

So the entire tail correction is the exponential of `n` times

```
m̄²/(2σ²) − h m̄ + log R.
```

That exponent wants unpacking. Define `z = m̄/σ` (so `x = z√n`). From the cumulant series, `m̄ = Σ γ_ν h^{ν−1}/(ν−1)!`, and inverting gives `h = z/σ − (γ_3/2σ⁴)z² − …`. Substituting into `m̄²/(2σ²) − h m̄ + log R` and using `m̄²/(2σ²) = z²/2`, everything of order `z²` cancels and what survives starts at `z³`:

```
m̄²/(2σ²) − h m̄ + log R = z³ λ(z),     λ(z) = c_0 + c_1 z + c_2 z² + …,
c_0 = γ_3/(6σ³),     c_1 = (σ²γ_4 − 3γ_3²)/(24σ⁶), …
```

So the exponent is `(x³/√n) λ(x/√n)`, and I have my fundamental theorem: for `1 < x = o(√n/log n)`,

```
(1 − F_n(x))/(1 − Φ(x)) = exp[ (x³/√n) λ(x/√n) ] · [ 1 + O(x log n/√n) ],
```

with the symmetric statement `F_n(−x)/Φ(−x) = exp[ −(x³/√n) λ(−x/√n) ]·[1 + O(x log n/√n)]` proved identically. The first coefficient `c_0 = γ_3/6σ³` is the skewness — the tail correction is led by skewness, as it must be: a right-skewed `V` makes large positive sums *more* likely than Gaussian (`c_0 > 0` pushes the ratio up). And there's the consistency check baked in: the exponent is `(x³/√n)λ`, so for `x = o(n^{1/6})` it `→ 0` and the ratio `→ 1` — the CLT regime sits inside, exactly as it should. If `V` is symmetric, `c_0 = 0` and the next nonzero coefficient takes over. Good.

I should pause to confirm `h√n` really is bounded below, or the "centre" trick is a lie. From `h ∼ z/σ = x/(σ√n)` and the assumption `x > 1`, indeed `h√n ∼ x/σ` is bounded below. So the tilt parameter is genuinely sending the rare event to the centre and the CLT error estimate holds.

This already answers the moderate-deviation question and absorbs the prior fragments. The binomial case `Z ∈ {1−p (prob p), −p (prob q)}` is just a particular `V`; my theorem recovers Smirnov's `1 + o(x^{−2s})` (his restriction `x = o(n^{1/(4s+6)})` is a sub-case of mine), Lévy's `log(1−F_n) ∼ −x²/2`, and Khinchin's expansion. The model-specific tail results were all shadows of one mechanism: tilt the law, re-center, use the CLT.

But I haven't reached the real prize — the large-deviation scale `x = c√n`, sum of order `cσn`. Right now my regime is `x = o(√n/log n)`, and the obstruction is that pesky `log n` in the Lyapunov error: it forces `x log n/√n → 0`. Where does the `log n` come from? It's the worst-case error of the CLT for a *general* CDF, which can have nasty oscillation near a lattice. If `V̄` has even a little smoothness, the CLT error improves to `O(1/√n)` with no `log n`. So I add a mild condition: in the decomposition `V = β U_1 + (1−β) U_2` into an absolutely continuous part `U_1` (with a density) and a singular part `U_2`, require `β > 0` — `V` has a nontrivial smooth component. (The tilt `V̄` inherits this.) Then `|Q_n| < K/√n`, the `log n` disappears, the regime opens to `x = o(√n)`, and — crucially — I can now take `x` all the way to order `√n`.

So set `x = c√n` with `0 < c < C_1`, where `σC_1 = lim_{h→A_1} m̄` is the largest tilted mean I can reach by turning the dial up to the edge `A_1` of the convergence strip. Because `m̄` is continuous and strictly increasing in `h` (that monotonic dial again), the equation

```
m̄ = σc
```

has a *unique* root `h` in the strip, with the same sign as `c`. This is the punchline of the whole tilt idea: I choose `h` to be precisely the tilt that makes `σc` the tilted mean. Then the deviation `sum ≈ σc·√n·√n` is the *typical* value under the tilted law, and the CLT/LLN error is genuinely `o(1)`. Redo the same Gaussian-against-exponential integral as before, now with the sharper `|Q_n| < K/√n`: the Gaussian-against-exponential integral now contributes a `1/√n` and a constant prefactor `b_0 = 1/(hσ̄√(2π))`, and I get

```
1 − F_n(c√n) = (1/√n) e^{−α n} [ b_0 + b_1/n + … + b_{k−1}/n^{k−1} + O(1/n^k) ],
```

where the exponential rate is

```
α = h m̄ − log R,     evaluated at the root h of  m̄ = σc.
```

`b_0 = 1/(hσ̄√(2π)) > 0`. There it is: the probability of a sum of order `n` decays like `e^{−αn}`, with an explicit `1/√n` prefactor and an honest leading constant — the *magnitude*, which is what I needed all along.

Now let me really look at `α = h m̄ − log R`, because it's begging to be recognized. Recall `m̄ = (d/dh) log R(h)`, and `h` is chosen as the root of `m̄ = σc`, i.e. `(d/dh) log R(h) = σc`. So `α(c) = h·(σc) − log R(h)` where `h` solves `(d/dh) log R = σc`. That is exactly the recipe for the *Legendre transform* of `log R`: the function

```
Λ*(a) = sup_h [ h a − log R(h) ]
```

attains its supremum at the `h` where `(d/dh) log R(h) = a`, and there equals `h a − log R(h)`. With `a = σc`, `α = Λ*(σc)`. The rate of exponential decay of a rare sum is the Legendre transform of the cumulant generating function of the summand. And `α > 0` always (for `c ≠ 0`): `Λ*(a) ≥ 0` with equality only at `a = E(Z) = 0`, by Jensen — `log R(h) ≥ hE(Z) = 0`, so `h·0 − log R ≤ 0`, hence `Λ*(0) = 0`, and convexity does the rest. The tail rate vanishes only at the mean; everywhere else it's strictly positive. That's the whole risk-theory content distilled into one convex function.

That route — tilt, re-center, CLT, read off the Legendre transform as the sharp rate — gives the prefactor and everything. If I only want the exponential *rate* and am willing to drop the prefactor, there's a leaner argument hiding inside it, and it's worth doing on its own because it pins the rate from above and below separately. Strip to the bare i.i.d. setup: `X_i` i.i.d., `Λ(t) = log E[e^{tX_1}]` (the cumulant generating function — same `log R`), and the claim is, for the empirical mean `S_n = (X_1+…+X_n)/n` and any `a > E[X_1]`,

```
lim_{n→∞} (1/n) log Prob(S_n ≥ a) = − Λ*(a),     Λ*(a) = sup_t [ t a − Λ(t) ].
```

The upper bound first, because it's almost free. For any `t ≥ 0`, the event `{S_n ≥ a}` = `{e^{t·nS_n} ≥ e^{tna}}`, so Markov's inequality gives

```
Prob(S_n ≥ a) ≤ e^{−tna} E[e^{t Σ X_i}] = e^{−tna} (E[e^{tX_1}])^n = e^{−n(ta − Λ(t))}.
```

This holds for *every* `t ≥ 0`, so I'm free to pick the best one — minimize the bound, i.e. maximize the exponent `ta − Λ(t)`:

```
Prob(S_n ≥ a) ≤ exp[ −n sup_{t≥0} (ta − Λ(t)) ] = e^{−n Λ*(a)}.
```

(For `a > E[X_1]` the unconstrained sup over all `t` is attained at some `t > 0`, so `sup_{t≥0} = sup_t = Λ*(a)`; the constraint `t ≥ 0` is non-binding.) Hence `limsup (1/n) log Prob(S_n ≥ a) ≤ −Λ*(a)`. The Legendre transform *emerges* here — I didn't assume it; it's just what optimizing the free tilt parameter `t` in the Chernoff bound produces. That's the exponential-Chebyshev / Chernoff move, and it's exactly the upper half.

The danger with a one-sided bound is that it might be loose — Markov could be throwing away a lot. I need a matching lower bound to prove the rate is sharp, and this is where the tilt earns its keep again, now probabilistically rather than via characteristic functions. The intuition: I can't bound `Prob(S_n ≥ a)` from below directly because the event is rare and the LLN says `S_n → E[X] < a`, so the typical sample never gets there. So change the measure to one where `a` *is* typical. Let `t*` be the (unique, by strict convexity of `Λ`) solution of `Λ'(t*) = a` — the same saddle as before. Define the tilted law

```
dμ_{t*}(x) = e^{t* x − Λ(t*)} dμ(x).
```

It's a probability law (`∫ e^{t* x} dμ = E[e^{t*X}] = e^{Λ(t*)}`, so the normalization is right), and its mean is

```
∫ x dμ_{t*}(x) = E[X e^{t*X}]/E[e^{t*X}] = Λ'(t*) = a.
```

So under `μ_{t*}`, the value `a` is the mean — the rare event has become typical. Now express the original probability as an expectation under the tilt. For a small `δ > 0`, restrict to the slab `{S_n ∈ [a, a + δ)}`:

```
Prob(S_n ∈ [a, a+δ)) = ∫_{S_n ∈ [a,a+δ)} ∏ dμ(x_i)
                     = ∫_{S_n ∈ [a,a+δ)} ∏ e^{−t* x_i + Λ(t*)} dμ_{t*}(x_i)
                     = e^{nΛ(t*)} ∫_{S_n ∈ [a,a+δ)} e^{−t* Σ x_i} dμ_{t*}^{⊗n}.
```

On the slab, `Σ x_i = n S_n ∈ [na, n(a+δ))`, so for `t* ≥ 0` (which holds since `a > E[X]` ⇒ `t* > 0`), `e^{−t* Σ x_i} ≥ e^{−t* n(a+δ)}`. Pull it out:

```
Prob(S_n ∈ [a,a+δ)) ≥ e^{nΛ(t*)} e^{−t* n(a+δ)} · μ_{t*}^{⊗n}( S_n ∈ [a, a+δ) ).
```

And here's the LLN paying off: under `μ_{t*}` the mean is exactly `a`, so `S_n → a`, and the probability `μ_{t*}^{⊗n}(S_n ∈ [a, a+δ)) → ½` (or any constant bounded away from `0` — `a` is the mean, and the tilted variance is finite, so a one-sided slab around the mean has probability bounded below). Taking `(1/n) log`:

```
(1/n) log Prob(S_n ≥ a) ≥ Λ(t*) − t*(a+δ) + (1/n) log(const)
                        → Λ(t*) − t* a − t* δ = −Λ*(a) − t* δ,
```

since `Λ(t*) − t* a = −(t* a − Λ(t*)) = −Λ*(a)` at the saddle. Let `δ → 0`: `liminf (1/n) log Prob(S_n ≥ a) ≥ −Λ*(a)`. The two bounds meet, and the limit is `−Λ*(a)`.

Let me make sure I haven't quietly assumed something false in the lower bound. I used that `μ_{t*}` exists, i.e. `t*` is in the open convergence strip — true when `a` is strictly between the mean and the essential supremum of `X`, which is the interesting range. And I used that the slab has probability bounded below under the tilt; if `X` has unbounded support I should be a touch careful that the tilted distribution genuinely concentrates at `a` — but its mean is `a` and (by differentiating `Λ` twice) its variance `Λ''(t*)` is finite, so Chebyshev under the tilt gives `μ_{t*}^{⊗n}(|S_n − a| < δ) → 1`. If I were nervous about heavy tails making `Λ(t)` infinite, I'd first truncate `X` to `[−A, A]`, run the whole argument for the truncated law (its `Λ_A ≤ Λ`, and `Λ_A → Λ` as `A → ∞`), and let `A → ∞` at the end — that's the standard patch, and it doesn't change the conclusion. So the lower bound is honest.

Step back and look at what controls the whole thing. The exponent that bounds the probability from *above* is `sup_t(ta − Λ(t))`, produced by optimizing the free tilt in Markov. The exponent that bounds it from *below* is `t* a − Λ(t*)` at the special tilt `t*` where the tilted mean is `a`. These are the *same number* — the value of the sup is attained at exactly the `t*` where the derivative vanishes, `Λ'(t*) = a`. The optimization in the upper bound and the re-centering in the lower bound are two faces of one convex-duality fact: `Λ` and `Λ*` are Legendre conjugates, the upper bound optimizes over `t`, the lower bound picks the optimizer, and they coincide because the optimum is interior. That coincidence is *why* the rate is exactly the Legendre transform and not merely bounded by it.

One last sanity pass on the shape of `Λ*`. `Λ(t) = log E[e^{tX}]` is convex (Hölder), `Λ(0) = 0`, `Λ'(0) = E[X]`. Its conjugate `Λ*(a) = sup_t(ta − Λ(t))` is convex and nonnegative, with `Λ*(E[X]) = 0` (the sup at `a = E[X]` is attained at `t = 0`). For `a > E[X]` the optimizer `t* > 0`, so I may restrict the sup to `t ≥ 0` there (and to `t ≤ 0` for `a < E[X]`), which is exactly the restriction the Markov upper bound naturally lived under. Everything is consistent. The rate function is the convex dual of the cumulant generating function, it vanishes only at the mean, and it grows on both sides — the tail of a sum decays like `e^{−nΛ*(a)}`, fastest near the mean and at the explicit Legendre rate elsewhere.

So the final landing — the precise theorem, two equivalent forms, both proven. Sharp asymptotic form (the original, prefactor and all): under the light-tail condition plus a smooth-component condition, for `0 < c < C_1`,

```
1 − F_n(c√n) = (1/√n) e^{−α n} [ b_0 + b_1/n + … ],   b_0 = 1/(hσ̄√(2π)) > 0,
α = h m̄ − log R   at the root h of   m̄ = σc, i.e.  α = Λ*(σc) = sup_h[ hσc − log R(h) ].
```

Logarithmic (large-deviation principle) form: for i.i.d. `X_i` with `Λ(t) = log E[e^{tX_1}]` finite near `0`, and `a > E[X_1]`,

```
lim_{n→∞} (1/n) log Prob( (X_1+…+X_n)/n ≥ a ) = − Λ*(a),   Λ*(a) = sup_t [ t a − Λ(t) ].
```
