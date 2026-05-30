OK, let me think this through from scratch. I'm training a Transformer for translation and I keep running into the same annoying fact: if I just turn on Adam with a constant learning rate, the training loss falls a little and then jams around 10 and never recovers. The fix everyone uses is a learning-rate *warmup* — make the step size tiny for the first few thousand updates, ramp it up, and then it trains fine, loss going below 3. I've seen the same thing in language-model pretraining. But warmup bothers me. It has no derivation. I have to pick how long it lasts and what shape it takes, and the right choice is different for every task; I tune it by hand. So I don't actually understand what it's doing. Before I add yet another knob, I want to know *why Adam needs warmup at all*, and *what exactly warmup fixes* — because if I can name the disease I might be able to write down the cure instead of guessing at a schedule.

So let me stare at the early updates, where the damage happens. There's a concrete clue: if I plot the histogram of absolute gradient values on a log scale over training, then *without* warmup the gradient distribution gets distorted — its mass slides toward small values — within the first ten updates. Ten updates. Whatever goes wrong, goes wrong almost immediately, and then the optimizer is stuck there. So this is an *early-stage* phenomenon, not a slow drift. That tells me to look hard at what Adam's update is made of in its first handful of steps.

Let me write Adam down carefully in a form that separates the two things it computes. Following the generic framing where any of these methods is `θ_t = θ_{t-1} − α_t · m_t · l_t` with a momentum `m_t = φ(g_1,…,g_t)` and an adaptive rate `l_t = ψ(g_1,…,g_t)`, Adam's choices are
`φ = ((1−β₁)Σᵢ β₁^{t−i} gᵢ)/(1−β₁^t)` and `ψ = sqrt( (1−β₂^t) / ((1−β₂)Σᵢ β₂^{t−i} gᵢ²) )`,
both element-wise. The momentum is just a smoothed, bias-corrected gradient — nothing exotic. The whole *adaptive* personality lives in `ψ`: it's one over the square root of a (bias-corrected) exponential moving average of squared gradients. That's the object I should interrogate. Numerically people compute `ψ̂ = sqrt(1−β₂^t)/(ε + sqrt((1−β₂)Σβ₂^{t−i}gᵢ²))` with a tiny `ε=1e-8`, but `ε` is just there to avoid dividing by zero; conceptually `ψ` is `1/sqrt(second-moment-estimate)`.

Now here is the thing that nags at me. That second-moment estimate is a *statistic* — it's an average of squared gradients — and in the first few steps it's an average over almost nothing. At `t=1` it's literally a single squared gradient. So `ψ` at `t=1` is `1/sqrt(g₁²) = 1/|g₁|`. And `g₁` is a random gradient. If `g₁` happens to come out small, `1/|g₁|` is huge, and Adam takes an enormous step in that coordinate. The adaptive rate isn't a fixed schedule that I'm choosing — early on it's a wildly *random* quantity, because it's the reciprocal square root of an estimate built from one or two samples. That smells like the disease: the early adaptive rate has a huge *variance*, and a few of those huge-rate coordinates throw the parameters somewhere bad, which then distorts all the later gradients. The histogram getting wrecked in ten steps is exactly what I'd expect if the first few updates are occasionally gigantic.

Let me make the `t=1` case quantitative, because if I can show the variance is not just large but *infinite* there, the diagnosis is airtight. Model the gradients at the start as i.i.d. `gᵢ ~ N(0, σ²)`. That's defensible: at initialization the weights are sampled mean-zero, so the gradients are roughly mean-zero too. Then `g₁² /σ² ~ χ²₁`, so `σ²/g₁² ~ inverse-χ²₁`, i.e. `1/g₁²` follows a scaled inverse chi-square `Scale-inv-χ²(1, 1/σ²)`. And `ψ = sqrt(1/g₁²)`. What's `Var[ψ]`? I need `E[1/g₁²]` for the variance, and `E[1/g₁²] = E[1/(σ² χ²₁)]` — but the inverse chi-square with one degree of freedom has no finite mean. Concretely the density of `x = 1/g₁²` near `x→∞` behaves like `x^{-3/2} e^{-1/(2σ²x)}`, and `E[√x] = ∫ √x · p(x) dx` picks up `∫^∞ x^{1/2} x^{-3/2} dx = ∫^∞ x^{-1} dx`, which diverges. So even `E[ψ]` is infinite at `t=1`, let alone `Var[ψ]`. The early adaptive rate has *divergent* variance. That's not "a bit noisy" — at the very first step the expected adaptive learning rate is unbounded.

And now warmup falls into place as the obvious palliative, because of a one-line fact: scaling a random variable by `α` scales its variance by `α²`, `Var[αx] = α²Var[x]`. So if the early adaptive rate has enormous (even infinite-in-the-limit) variance, multiplying the whole step by a tiny `α` in those first updates shrinks that variance quadratically. Warmup doesn't fix the rate; it just *mutes* the rate while it's at its most volatile, and lets it through once it has calmed down. So my hypothesis is: warmup is a variance-reduction trick for the adaptive learning rate. That feels right, but a hypothesis about "the cause" needs me to check it can be falsified, so let me think about what experiments would confirm or kill it before I build anything on top.

If the disease is "too few samples in the denominator," then I should be able to cure it two independent ways. One: just *get more samples* before I start trusting the rate. Imagine freezing the parameters and the momentum for the first couple thousand iterations while still accumulating the squared-gradient EMA, then unfreezing. By the time real updates begin, the denominator has seen 2000 samples and its variance is small. If my hypothesis is right, this should converge fine even with no warmup — and the gradient histogram should stay undistorted. Two: directly *cap* the rate's variance. The crude way is to make `ε` non-negligible — say `1e-4` instead of `1e-8`. A big additive constant in the denominator bounds how large `ψ` can get (if `ψ̂` were uniform on some interval, its variance would be `1/(12ε²)`, shrinking fast as `ε` grows), so this should also avoid the blow-up. Both of these are things I can reason about cleanly, and both point at the same lever: reduce the variance of `ψ` early and the failure goes away. The second one, though, comes with a tell — a big `ε` doesn't just cap the variance, it *biases* the rate, distorting Adam's intended per-coordinate scaling, so I'd expect it to avoid catastrophe but still underperform. Which means: a crude variance cap works but pays a price; I want a *principled* way to control the variance of `ψ` — one that pins the variance down without injecting bias. That's the real target.

So I need `Var[ψ]` not just at `t=1` but as a function of how many samples the rate has effectively seen, so I can see exactly how it shrinks and then *rectify* it. Two obstacles. First, the denominator isn't a clean average of `t` squared gradients — it's an *exponential* moving average, weights `β₂^{t−i}`, which is messier than a flat average and actually has *larger* variance than a flat average over the same samples. Second, I need the distribution, not just a point value.

Let me deal with the EMA by approximating it as a flat average — a simple moving average over some effective window. The justification: early on, when `t` is small, the exponential weights `β₂^{t−i}` are all close to each other (they differ only up to `1−β₂^{t−1}`, which is tiny for `β₂` near 1), so the EMA is nearly a flat average anyway. And there's a clean classical correspondence (Nau, 2014) between an EMA and an SMA: they're "the same" when the window length is chosen so the two weightings have the same *center of mass*. So write `ψ²(.) = (1−β₂^t)/((1−β₂)Σβ₂^{t−i}gᵢ²) ≈ (Σ_{i=1}^{f} g²)/f` for some effective window `f`. Now the distribution is easy: if `gᵢ ~ N(0,σ²)`, then `Σ_{i=1}^{f} gᵢ²/σ² ~ χ²_f`, so `f/Σgᵢ² ~ Scale-inv-χ²(f, 1/σ²)`. So I'll model `ψ²(.) ~ Scale-inv-χ²(ρ, 1/σ²)` with `ρ` degrees of freedom standing for "effective number of samples," and I just need to figure out what `ρ` is as a function of `t` and `β₂`.

Let me get `Var[ψ]` as a function of `ρ` first, treating `ρ` as a free parameter; I'll pin it to `t` afterward. Write `x = ψ²` and `τ² = 1/σ²`, so `x ~ Scale-inv-χ²(ρ, τ²)` with density
`p(x) = (τ²ρ/2)^{ρ/2}/Γ(ρ/2) · exp(−ρτ²/(2x)) / x^{1+ρ/2}`.
The mean exists once `ρ>2`: `E[x] = ρ/((ρ−2)σ²) = ρτ²/(ρ−2)`. But I want `Var[ψ] = Var[√x] = E[x] − E[√x]²`, so I need `E[√x]`. Compute it by absorbing the `√x` into the power of `x` in the integrand:
`E[√x] = ∫₀^∞ x^{1/2} p(x) dx`. The `x`-dependent part is `x^{1/2} · x^{−1−ρ/2} e^{−ρτ²/(2x)} = x^{−(ρ+1)/2 − 1/2}…` — cleanest to substitute `u = ρτ²/(2x)` and recognize a Gamma integral. Doing that bookkeeping, the integral converges only when `ρ>4` (the `√x` pulls one extra factor of `x` into the tail, so I need two more degrees of freedom than for `E[x]`), and it gives
`E[√x] = τ√ρ · Γ((ρ−1)/2) / (√2 · Γ(ρ/2))`, for `ρ>4`.
That `ρ>4` is itself a load-bearing fact: below five effective samples, even the *square root's mean* — let alone its variance — isn't well defined. So:
`Var[ψ] = E[x] − E[√x]² = τ²( ρ/(ρ−2) − (ρ/2)·(Γ((ρ−1)/2)/Γ(ρ/2))² )`, for `ρ>4`.
Let me tidy the second term using the Beta function `B(a,b)=Γ(a)Γ(b)/Γ(a+b)`. With `a=b=(ρ−1)/2`, `B((ρ−1)/2,(ρ−1)/2) = Γ((ρ−1)/2)²/Γ(ρ−1)`, so `Γ((ρ−1)/2)² = B(·,·)·Γ(ρ−1)`. Folding the `Γ(ρ−1)/Γ(ρ/2)²` together with the powers of two (Legendre duplication, `Γ(ρ−1)` relates to `Γ((ρ−1)/2)` and `Γ(ρ/2)` with a `2^{ρ−2}` factor) collapses the whole thing to
`Var[ψ] = τ²( ρ/(ρ−2) − ρ·2^{2ρ−5}/π · B((ρ−1)/2,(ρ−1)/2)² )`, for `ρ>4`.

Does this match the `t=1` intuition? At `ρ=1` I'm below the `ρ>4` regime where the formula is even valid, and indeed `E[√x]` diverged there — consistent. Good. Now the qualitative claim I need: `Var[ψ]` must *decrease* as `ρ` grows, i.e. more samples → less variance. Let me convince myself that bracket `h(ρ) = ρ/(ρ−2) − ρ·2^{2ρ−5}/π · B((ρ−1)/2,(ρ−1)/2)²` is decreasing for `ρ≥4`. Differentiate. The first piece `ρ/(ρ−2)` has derivative `−2/(ρ−2)² < 0`. The second piece is subtracted, so I need its derivative to be *non-negative* (a growing thing being subtracted makes `h` fall) — equivalently I need to show the whole derivative is negative. Differentiating `ρ·2^{2ρ−5}·B²` brings down `2^{2ρ−5}B²` (product rule on `ρ`), `ρ·2^{2ρ−5}B²·ln4` (from `d/dρ 2^{2ρ}`), and `2ρ·2^{2ρ−5}B²·(Ψ((ρ−1)/2)−Ψ(ρ−1))` (from `d/dρ B²`, where `Ψ=Γ'/Γ` is the digamma and `B((ρ−1)/2,(ρ−1)/2)` has log-derivative `Ψ((ρ−1)/2)−Ψ(ρ−1)` after the chain rule). Collecting, `h'(ρ)<0` is equivalent, after clearing positive factors, to
`64π/((ρ−2)² 4^ρ B((ρ−1)/2,(ρ−1)/2)²) + 1 + ρ ln4 + 2ρ Ψ((ρ−1)/2) > 2ρ Ψ(ρ−1)`.
Use the Legendre duplication formula in digamma form, `2Ψ(ρ−1) = Ψ((ρ−1)/2) + Ψ(ρ/2) + ln4`, to rewrite the right side, and the `ln4` and one `Ψ((ρ−1)/2)` cancel, leaving
`64π/((ρ−2)² 4^ρ B²) + 1 + ρΨ((ρ−1)/2) − ρΨ(ρ/2) > 0`.
Now I lower-bound the digamma difference with the standard sandwich `ln(x) − 1/(2x) > Ψ(x) > ln(x+0.5) − 1/x`: take the upper bound for `Ψ((ρ−1)/2)`'s positive contribution and the lower for `Ψ(ρ/2)` so that `ρΨ((ρ−1)/2) − ρΨ(ρ/2) ≥ ρ(ln(ρ/2) − 1/(ρ/2−0.5)) − ρln(ρ/2) = −ρ/((ρ−1)/2) = −2ρ/(ρ−1)`, and `1 + (−2ρ/(ρ−1)) = (ρ−1−2ρ)/(ρ−1)`… let me redo that more carefully: with the `+1` it telescopes to `1 − 2/(ρ−1) + …`; the upshot is the non-Beta part is `> 1 − 2/(ρ−1) = (ρ−3)/(ρ−1)`, and combined with `−2/(ρ−1)` bookkeeping the whole expression exceeds `64π/((ρ−2)²4^ρ B²) − 2/(ρ−2)`. So it suffices that `64π/((ρ−2)²4^ρ B²) ≥ 2/(ρ−2)`, i.e. `32π ≥ (ρ−2)4^ρ B((ρ−1)/2,(ρ−1)/2)²`. Rewrite `B((ρ−1)/2,(ρ−1)/2)² = Γ((ρ−1)/2)⁴/Γ(ρ−1)²`, apply Legendre duplication to `Γ(ρ−1)` to turn `4^ρ Γ((ρ−1)/2)⁴/Γ(ρ−1)²` into `16π(ρ−2)Γ((ρ−1)/2)²/Γ(ρ/2)²`, so the claim becomes `(ρ−2)Γ((ρ−1)/2)²/Γ(ρ/2)² ≤ 2`. Finally Gautschi's inequality `Γ(x+1)/Γ(x+s) < (x+1)^{1−s}` gives `Γ((ρ−1)/2)²/Γ(ρ/2)² ≤ ((ρ−1)/2)^{−1} = 2/(ρ−1)`, so `(ρ−2)·2/(ρ−1) = 2(ρ−2)/(ρ−1) < 2`. Done — `Var[ψ]` strictly decreases in `ρ`. So the picture is exactly what the diagnosis predicted: huge variance when `ρ` is small (few samples), monotonically falling as samples accumulate.

Now pin `ρ` to the actual step `t`. I claimed `ρ` = the effective SMA window length `f(t,β₂)`, the window whose flat weighting has the same center of mass as the EMA's `β₂^{t−i}` weighting. Let me actually solve for it. The EMA's normalized weight on sample `i` (at step `t`) is `(1−β₂)β₂^{t−i}/(1−β₂^t)`. Its center of mass, measured in the index `i`, is `Σᵢ i·(1−β₂)β₂^{t−i}/(1−β₂^t)`. The SMA over the most recent `f` samples weights `i = t, t−1, …, t−f+1` uniformly; its center of mass is the average of `(t+1−i)` over `i=1..f`, i.e. `(Σ_{i=1}^{f}(t+1−i))/f`. Set them equal:
`(1−β₂)Σ_{i=1}^t β₂^{t−i}·i / (1−β₂^t) = (Σ_{i=1}^{f}(t+1−i))/f`.
The right side is `(t+1) − (f+1)/2`. For the left, it's cleaner to think in the EMA's natural "age" coordinate. Let me just take the `t→∞` limit first to get the ceiling. As `t→∞`, the EMA weight on age `k=t−i` is `(1−β₂)β₂^k`, `k=0,1,2,…`, summing to 1. The mean age is `E[k] = Σ k(1−β₂)β₂^k = β₂/(1−β₂)`. The SMA's mean age over a window of `f` is `(f−1)/2`. Matching the *index* `t+1−i = k+1`: EMA mean of `k+1` is `1/(1−β₂)`; SMA mean of `(t+1−i)` is `(f+1)/2`. Equate: `(f+1)/2 = 1/(1−β₂)`, so `f = 2/(1−β₂) − 1`. Call this `ρ_∞` — the maximum effective window, reached as `t→∞`. For `β₂=0.999` that's `1999`; for `β₂=0.9` it's `19`.

For finite `t` I have to keep the truncation terms. Solving the full center-of-mass equation (the geometric-sum identities `Σβ₂^{t−i}=（1−β₂^t)/(1−β₂)` and `Σ i β₂^{t−i}` differentiated from it) gives
`f(t,β₂) = 2/(1−β₂) − 1 − 2tβ₂^t/(1−β₂^t)`,
which is exactly `ρ_∞` minus a correction `2tβ₂^t/(1−β₂^t)` that vanishes as `t→∞` (since `β₂^t→0`) and is large when `t` is small. So define
`ρ_t = ρ_∞ − 2tβ₂^t/(1−β₂^t)`,
my estimate of the effective degrees of freedom at step `t`. Let me sanity-check the endpoints. At `t=1`, `ρ_t = 2/(1−β₂)−1 − 2β₂/(1−β₂) = (2 − 1 + β₂ − 2β₂)/(1−β₂) = (1−β₂)/(1−β₂) = 1`. One sample at step one — exactly right, and exactly the divergent-variance case. As `t→∞`, `ρ_t → ρ_∞`. And `ρ_t ≤ ρ_∞` always, since the correction is non-negative. So `ρ_t` slides monotonically from 1 up toward `ρ_∞`, and by the monotonicity theorem `Var[ψ]` slides from its divergent early value down to its floor `Var[ψ]|_{ρ_∞}`.

Now the rectification. I want the adaptive rate to have the *same* variance at every step, so the optimizer's behavior is consistent from the start instead of being dominated by the early blow-up. The natural target is the floor — the minimal variance, achieved at `ρ_∞`; call it `C_var = Var[ψ]|_{ρ=ρ_∞}`. I'll multiply the rate by a scalar `r_t` chosen so `Var[r_t·ψ] = C_var`. Since `Var[r_t ψ] = r_t² Var[ψ]`, that gives
`r_t = sqrt( C_var / Var[ψ]|_{ρ=ρ_t} ) = sqrt( Var[ψ]|_{ρ_∞} / Var[ψ]|_{ρ_t} )`.
Because `Var[ψ]` decreases in `ρ` and `ρ_t ≤ ρ_∞`, the denominator is `≥` the numerator, so `r_t ≤ 1` always, with `r_t → 1` as `t→∞`. So `r_t` is a multiplier that starts small and climbs to one.

But there's a numerical problem if I plug in the exact `Var[ψ]`: that Beta-function expression with `2^{2ρ−5}` and `B((ρ−1)/2,(ρ−1)/2)²` overflows/underflows badly for large `ρ` like 1999 — `4^ρ` is astronomical, the Beta is astronomically small, and computing their product stably is hopeless. I need a clean, stable surrogate for `Var[ψ]`. Use a first-order Taylor (delta-method) expansion of `√` around the mean: `√x ≈ √E[x] + (1/(2√E[x]))(x − E[x])`, so `Var[√x] ≈ Var[x]/(4E[x])`. For `x ~ Scale-inv-χ²(ρ,τ²)` I know `E[x] = ρτ²/(ρ−2)` and `Var[x] = 2ρ²τ⁴/((ρ−2)²(ρ−4))` (the inverse-chi-square variance). Then
`Var[ψ] ≈ Var[x]/(4E[x]) = [2ρ²τ⁴/((ρ−2)²(ρ−4))] / [4·ρτ²/(ρ−2)] = ρτ²/(2(ρ−2)(ρ−4))`,
i.e. with `τ²=1/σ²`,
`Var[ψ] ≈ ρ / (2(ρ−2)(ρ−4)σ²)`.
This is wonderfully simple, it only needs the four arithmetic operations, and it makes the `ρ>4` requirement visible right in the denominator — at `ρ=4` it blows up, below `ρ=4` it goes negative, i.e. the variance is undefined, which matches the exact formula's domain. For large `ρ` it behaves like `1/(2ρσ²) = O(1/ρ)`, so the variance decays like one over the effective sample count — and concretely the variance at small `ρ` is well over a hundred times the variance at `ρ=500`, a huge early excess. (I'd want to check numerically that this first-order form tracks the exact Beta expression as a function of `ρ`; I'd expect them to share the same `O(1/ρ)` shape and to agree closely once `ρ` is past the smallest values, with the approximation loosest right near `ρ=4` where the variance is largest anyway — which is fine, since there I only need it to say "huge," not to be tight.)

Substitute this approximation into `r_t`. The `σ²` and the constant `2` cancel between numerator and denominator, leaving only the `ρ`-dependent factors `ρ/((ρ−2)(ρ−4))`:
`r_t = sqrt( [ρ_∞/((ρ_∞−2)(ρ_∞−4))] / [ρ_t/((ρ_t−2)(ρ_t−4))] ) = sqrt( (ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t) )`.
That's the rectification term. Clean, closed-form, stable, no special functions. And it self-consistently demands `ρ_t > 4` for the square root's argument to be positive and the variance to be finite — which is the same threshold I kept hitting.

So what do I do when `ρ_t ≤ 4`? That's the regime — the first few steps — where the variance isn't merely large, it's *not well-defined* (the integral for `E[√x]` diverges, the approximation goes negative). There's nothing to rectify; the adaptive rate is meaningless there. The honest move is to not use it at all in those steps. Just take the momentum step without the adaptive denominator — i.e. fall back to plain SGD-with-momentum on the bias-corrected first moment, `θ_t = θ_{t-1} − α_t m̂_t`. So the early phase, where Adam was taking those occasionally-gigantic adaptive steps that wrecked the gradient histogram, is replaced by ordinary momentum steps, which have no divergent denominator. Once `ρ_t` crosses 4, the variance becomes tractable, I switch on the adaptive rate, and I damp it by `r_t` so its variance is held at the floor `C_var` from that point. As `t` grows `r_t → 1` and I'm just running Adam again. There's a nice corner case that confirms the logic: if `β₂ ≤ 0.6` then `ρ_∞ = 2/(1−β₂)−1 ≤ 4`, so `ρ_t` never exceeds 4 and the method is *always* in the fallback — it degenerates to SGD with momentum. That's exactly what should happen: with `β₂` that small the second-moment estimate never accumulates enough effective samples to have a well-defined variance, so it's never trustworthy, so you should never use it.

Let me write the adaptive rate explicitly in the form I'll code. `ψ` is the bias-corrected inverse-root second moment: `l_t = sqrt((1−β₂^t)/v_t)` where `v_t` is the raw EMA second moment. The first moment gets its usual bias correction `m̂_t = m_t/(1−β₁^t)`. So the rectified step in the active regime is `θ_t = θ_{t-1} − α_t · r_t · m̂_t · l_t`. Precompute `ρ_∞ = 2/(1−β₂)−1` once. Per step compute `ρ_t = ρ_∞ − 2tβ₂^t/(1−β₂^t)`, branch on `ρ_t > 4`.

Now step back to the warmup question I started with, because the shape of `r_t` answers it. `r_t` rises monotonically from near 0 (just after `ρ_t` passes 4) toward 1 as `t` grows. A linear warmup multiplies the step by `min(t,T_w)/T_w` — which is *also* a multiplier rising from near 0 to 1 over the early phase. So warmup is doing the same thing my `r_t` does: it's a heuristic, hand-shaped approximation to the variance-rectification multiplier. That closes the loop on the original hypothesis — warmup works *because* it reduces the variance of the adaptive learning rate in the early stage, and `r_t` is the principled version of it. Two concrete advantages fall out: I never had to choose a `T_w` (the schedule is determined entirely by `β₂` through `ρ_∞` and `ρ_t`), and unlike warmup, which still feeds the divergent-variance rate through a small multiplier in the very first steps, my rule *deactivates* the adaptive rate entirely while its variance is divergent (`ρ_t ≤ 4`) — it doesn't just shrink the dangerous quantity, it refuses to use it until it's meaningful. That deactivation matters: those first few updates with divergent variance are precisely the ones that distorted the gradient histogram, so they're the most damaging ones to get wrong.

One implementation nuance from doing this for real. The exact threshold from the math is `ρ_t > 4`, but `ρ_t` is itself an *approximation* (EMA-as-SMA, center-of-mass matching), so being a touch conservative near the boundary is sensible — using `ρ_t ≥ 5` as the activation test buys a margin against the approximation error right where the variance estimate is most delicate, at the cost of one extra fallback step. And `ρ_t` only depends on `t` and `β₂`, not on the parameters, so I can cache it in a small ring buffer keyed by `t mod 10` and avoid recomputing the powers every parameter group.

Let me write it as a `torch` optimizer.

```python
import math
import torch
from torch.optim.optimizer import Optimizer

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        self.degenerated_to_sgd = degenerated_to_sgd
        # small cache of (t, rho_t, step_size) keyed by t % 10 — rho_t depends only on t, beta2
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                p_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_fp32)      # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p_fp32)   # v_t
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # EMA second moment v_t and first moment m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                state['step'] += 1
                t = state['step']

                # rho_t (effective SMA length); cached because it depends only on (t, beta2)
                buffered = group['buffer'][t % 10]
                if t == buffered[0]:
                    rho_t, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = t
                    beta2_t = beta2 ** t
                    rho_inf = 2 / (1 - beta2) - 1                      # rho_infinity = 2/(1-b2) - 1
                    rho_t = rho_inf - 2 * t * beta2_t / (1 - beta2_t)  # rho_t = rho_inf - 2 t b2^t/(1-b2^t)
                    buffered[1] = rho_t
                    if rho_t >= 5:  # variance tractable (rho_t > 4); >=5 is a margin since rho_t is approximate
                        # r_t * sqrt(1-b2^t) / (1-b1^t): rectification x bias corrections, x lr at use
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (rho_t - 4) / (rho_inf - 4)
                            * (rho_t - 2) / rho_t
                            * rho_inf / (rho_inf - 2)
                        ) / (1 - beta1 ** t)
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** t)            # un-adapted momentum (bias-corrected)
                    else:
                        step_size = -1                                 # skip the step entirely
                    buffered[2] = step_size

                if rho_t >= 5:
                    # rectified adaptive step: theta -= lr * r_t * m_hat / sqrt(v_hat)
                    if group['weight_decay'] != 0:
                        p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])      # ~ 1/l_t (with v_t, bias corr in step_size)
                    p_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_fp32)
                elif step_size > 0:
                    # rho_t <= 4: variance not well-defined -> plain momentum SGD step
                    if group['weight_decay'] != 0:
                        p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_fp32)
        return loss
```

The causal chain, start to end: Adam needs warmup because its adaptive rate `1/√v̂` is the reciprocal root of a second-moment estimate built from almost no samples, so early on it's a random quantity with enormous — at `t=1`, provably infinite — variance, and a few oversized adaptive steps wreck the gradient distribution within ten updates; warmup helps only because shrinking the step shrinks that variance by `α²`. Modeling the squared gradients as Gaussian makes `ψ²` a scaled-inverse-chi-square with `ρ` effective degrees of freedom, whose `Var[ψ]` is finite only for `ρ>4` and provably decreasing in `ρ`; matching the EMA to a same-center-of-mass SMA gives `ρ_∞ = 2/(1−β₂)−1` and `ρ_t = ρ_∞ − 2tβ₂^t/(1−β₂^t)`; a delta-method approximation `Var[ψ] ≈ ρ/(2(ρ−2)(ρ−4)σ²)` is stable enough to invert into the rectifier `r_t = √((ρ_t−4)(ρ_t−2)ρ_∞ /((ρ_∞−4)(ρ_∞−2)ρ_t))` that pins the rate's variance to its floor; and when `ρ_t ≤ 4` the variance isn't even defined, so the rule drops the adaptive denominator and takes a plain momentum step. The result needs no warmup schedule, adapts itself to `β₂`, and reduces to SGD-with-momentum exactly when `β₂` is too small for the second moment to ever be trustworthy.
