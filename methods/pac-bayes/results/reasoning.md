Let me start from the thing that actually bothers me, which is that I have two correctness stories for learning and they don't talk to each other, and each one is throwing away exactly what the other one has.

On one side is the PAC story. I draw `m` examples i.i.d. from some unknown `D`, I measure a hypothesis's empirical risk `ℓ̂(h, S)`, and I want to bound its true risk `ℓ(h) = E_{z∼D} ℓ(h, z)`. The beautiful thing is that this holds for *every* `D` — I assume nothing about the world. The ugly thing is how complexity enters. For a single fixed `h` I have concentration: `ℓ̂(h)` is an average of `m` i.i.d. `[0,1]` numbers with mean `ℓ(h)`, so Hoeffding gives me `Pr_S(ℓ(h) − ℓ̂(h) ≥ x) ≤ e^{−2 m x²}`. Fine. But the moment I want this to hold for *all* `h` in my class at once, I pay through a union bound or a VC dimension, and for a rich class — certainly for a continuous one — that price is ruinous. The bound goes vacuous.

On the other side is the Bayesian story. I put a prior `P` over hypotheses, encoding what I expect to be good, and if `P` is right I get an optimal, beautifully tuned predictor. The ugly thing: if `P` is wrong, I get *nothing*. No guarantee. And Kearns and Mansour and Ng and Ron showed empirically that exactly this happens — Bayesian and MDL methods overfit when their assumptions fail.

So I want both. I want to *put in* an informative prior the way Bayes does, and I want the guarantee to hold distribution-free the way PAC does, with no assumption that the prior matches the data. And I want it to survive continuous classes. The question that organizes everything: what is the right notion of "complexity" of a hypothesis that is prior-shaped, controllable without trusting the prior, and finite even for continuous classes?

Let me look hard at the one bound I already have that *uses* a prior, because it's almost what I want. Take a countable class with a prior `P`, a weighting summing to one — Linial and Mansour and Rivest used such weights, and Shawe-Taylor and Williamson noticed that any weighting summing to one is just a Bayesian prior. I apply Hoeffding to each `h_i` but with the failure budget split according to the prior: I demand the bound for `h_i` fail with probability at most `P(h_i)·δ`. Then the total probability that *any* `h_i` violates its bound is at most `Σ_i P(h_i)·δ = δ`, since the prior sums to one. Unwinding the per-hypothesis Chernoff with that budget, I get, with probability `≥ 1 − δ`, simultaneously for all `i`,

```
ℓ(h_i) ≤ ℓ̂(h_i, S) + sqrt( ( ln(1/P(h_i)) + ln(1/δ) ) / (2m) ).
```

The complexity of `h_i` is `−ln P(h_i)`, a description length. A hypothesis the prior likes pays little. This is exactly the prior-as-bias I wanted, and it's distribution-free. So why am I not done?

Two things gnaw at me. First, `−ln P(h_i)` is the prior mass of *one* hypothesis. For a continuous class every singleton has mass zero and `−ln P(h_i) = +∞`. The bound is vacuous precisely where I need it most. Second — and this is the one I want to chase — the bound scores a single *selected* hypothesis. It justifies picking the `h` that minimizes the right-hand side, which is a MAP rule. But I know from the Bayesian side that MAP is *not* the right predictor. When the prior is correct the optimal thing is the posterior-weighted vote over all consistent hypotheses — `P(y | x, S) ∝ Σ_{h consistent} P(h)·[h votes y]`. Averaging beats selecting. The selection bound is married to the wrong algorithm.

So let me stop trying to bound a single hypothesis and instead bound a *randomized* predictor. Put a distribution `Q` over hypotheses — call it a posterior, since it can depend on the data — and predict by drawing `h ∼ Q` and using that `h`. The natural quantities are the expected true risk `ℓ(Q) = E_{h∼Q} ℓ(h)` and the expected empirical risk `ℓ̂(Q, S) = E_{h∼Q} ℓ̂(h, S)`. I want `ℓ(Q) ≤ ℓ̂(Q, S) + (something)`. This already matches the Bayesian instinct that averaging is the right object. But it's not obvious it buys me anything in the bound — naively, an expectation over `Q` of a bound that holds for each `h` is just... the same bound averaged, complexity and all.

With the union bound I was insisting that the deviation be controlled *at every single `h` simultaneously* — that's what's expensive, and that's what blows up for continuous classes. But if I only ever report `E_{h∼Q}` of things, I don't need the event "every `h` is good." I need something weaker: I need to take *one* concentration statement that lives on the prior side and *transport* it over to `Q`. The question becomes — what does it cost to move an expectation from `P` to `Q`?

That's a question I actually have a tool for. There's an identity from large deviations, the Donsker–Varadhan / Gibbs variational formula: for any reference measure `P` and any function `g`,

```
log E_{h∼P} e^{g(h)} = sup_Q { E_{h∼Q} g(h) − KL(Q‖P) }.
```

Read the consequence I care about: for *any* `Q` absolutely continuous w.r.t. `P`,

```
E_{h∼Q} g(h) ≤ KL(Q‖P) + log E_{h∼P} e^{g(h)},
```

where `KL(Q‖P) = E_{h∼Q} ln(dQ/dP)`. There it is. The price of transporting an expectation of `g` from the prior `P` to a posterior `Q` is the divergence `KL(Q‖P)`, plus the log of an exponential moment of `g` *under `P`*. And `KL(Q‖P)` is exactly a complexity that is prior-shaped, that reduces to `−ln P(h)` when `Q` collapses to a point mass at `h` (then `KL = ln(1/P(h))`), and — crucially — that is *finite for continuous classes*, because two densities can be close in relative entropy even though no singleton has positive mass. The pathology that killed the Occam bound is gone. And it's the "information gained in specializing from `P` to `Q`" — the bits I'd normally have to extract from the sample to justify moving off the prior.

But I have to be honest about what this identity needs. The thing inside the expectation on the right is `log E_{h∼P} e^{g(h)}` — an *exponential* moment of `g`, under the prior. That tells me what `g` has to be. I want `g(h)` to be a function of the deviation `Δ(h) = |ℓ(h) − ℓ̂(h, S)|`, so that controlling `E_{h∼Q} g` controls the gap I care about. And I need `E_{h∼P} e^{g(h)}` to be small with high probability over the sample. So the whole bound is going to rest on a single inequality: an exponential moment of the deviation, *averaged over the prior*, is controlled. Let me see whether the per-hypothesis Hoeffding I started with gives me that.

Let me try the naive thing first and watch it work or break. Suppose I take `g(h) = β·Δ(h)` for some `β > 0` — linear in the deviation. Then I need `E_{h∼P} e^{β Δ(h)}` controlled, which by Fubini (since `P` doesn't depend on the sample) is `E_{h∼P} E_S e^{β Δ(h)}`, and for each fixed `h` I'd need a bound on `E_S e^{β Δ(h)}`. For a `[0,1]` average, the sub-Gaussian / Hoeffding control gives the one-sided moment `E_S e^{β(ℓ(h) − ℓ̂)} ≤ e^{β²/(8m)}`, and the opposite sign is symmetric in the same bound; controlling the absolute value only adds bookkeeping. It works, but it's giving me a Gaussian-tail shape, and when I push it through I'll get the square-root form with a `β` I have to optimize. Let me instead aim straight for the cleanest version and use a *quadratic* exponent, because I suspect the natural object is `e^{(constant)·Δ²}` — that's what a `e^{−2mx²}` tail integrates against.

So let me ask for `E_S e^{(2m−1) Δ(h)²}` for a single fixed `h`. I know `Pr_S(Δ ≥ x) ≤ 2 e^{−2 m x²}` (two-sided Chernoff). I should not pretend that `2e^{−2mx²}` is itself a probability tail at `x = 0`; it is bigger than one there. The safe way is the tail-integral identity for a nonnegative random variable. If `a = 2m−1`, then

```
E_S e^{(2m−1)Δ²}
  = 1 + ∫_0^∞ 2a x e^{a x²} Pr_S(Δ ≥ x) dx
  ≤ 1 + ∫_0^∞ 2a x e^{a x²} · 2e^{−2mx²} dx
  = 1 + 4a ∫_0^∞ x e^{−(2m−a)x²} dx
  = 1 + 4(2m−1) ∫_0^∞ x e^{−x²} dx
  = 1 + 2(2m−1)
  = 4m − 1
  ≤ 4m.
```

The `(2m−1)` in the exponent is chosen exactly so that `2m − (2m−1) = 1` and the integral collapses to `∫ x e^{−x²} dx = 1/2`. So for every single `h`, `E_S e^{(2m−1) Δ(h)²} ≤ 4m`. Now average over the prior and use Fubini (`P` is data-independent, so the expectations commute):

```
E_S E_{h∼P} e^{(2m−1) Δ(h)²} = E_{h∼P} E_S e^{(2m−1) Δ(h)²} ≤ 4m.
```

This is a statement about a *single* random quantity `Y(S) = E_{h∼P} e^{(2m−1)Δ(h)²}` whose expectation over `S` is at most `4m`. Markov turns that into a high-probability statement: `Pr_S( Y(S) > 4m/δ ) ≤ δ`. So with probability at least `1 − δ` over the draw of the sample,

```
E_{h∼P} e^{(2m−1) Δ(h)²} ≤ 4m/δ.
```

Stare at this for a second, because this is the whole game. This statement is about the *prior*. It is proven and fixed *before I ever choose `Q`*. The sample is drawn, the event "`Y(S) ≤ 4m/δ`" happens with probability `1 − δ`, and that's that — no `Q` appears anywhere. Which means that once this good event holds, *every* `Q`, including a `Q` I cook up after staring at `S`, can be plugged into the change-of-measure identity and inherit a bound. That's the resolution of "holds for all `Q` simultaneously": I never union-bounded over `Q` at all. I concentrated *one* quantity on the prior side and the variational identity hands the conclusion to every `Q` for free.

Now let me actually run the transport. I want to bound `E_{h∼Q} Δ(h)`, but the controlled exponential moment is in `Δ²`. Let me set `g(h) = (2m−1) Δ(h)²` and feed it through. I'll redo the variational step concretely so I trust every constant, by solving the constrained problem directly rather than quoting the identity. Work with a finite class for the algebra; the continuous case follows by a limiting argument. I want to maximize `Σ_i Q_i y_i` over `y`, subject to `Σ_i P_i e^{β y_i} ≤ K`, where I'll put `y_i = Δ(h_i)²`, `β = 2m−1`, `K = 4m/δ`. This is the right thing to maximize: an *upper* bound on `Σ_i Q_i y_i` that holds whenever the constraint holds is exactly what I need, since my good event gives me the constraint.

Lagrange. Maximize `Σ Q_i y_i − μ(Σ P_i e^{β y_i} − K)`. Stationarity in `y_i`: `Q_i − μ P_i β e^{β y_i} = 0`, so `e^{β y_i} = Q_i/(μ β P_i)`, i.e. `y_i = (1/β) ln(Q_i/(μ β P_i))`. At the optimum the constraint is active (pushing `y` up only helps the objective until the constraint bites), so `Σ_i P_i e^{β y_i} = Σ_i Q_i/(μβ) = 1/(μβ) = K`, giving `μ = 1/(βK)`. Substitute back:

```
y_i = (1/β) ln( Q_i / (μ β P_i) ) = (1/β) ln( Q_i · K / P_i ) = (1/β)( ln(Q_i/P_i) + ln K ).
```

Therefore the maximum value of the objective is

```
Σ_i Q_i y_i = (1/β) Σ_i Q_i ( ln(Q_i/P_i) + ln K ) = ( KL(Q‖P) + ln K ) / β.
```

That's it — and notice this *is* the Donsker–Varadhan identity, derived from scratch: `sup_Q [ E_Q y − (1/β) KL(Q‖P) ] = (1/β) log E_P e^{β y}`, with the optimizing `Q` proportional to `P_i e^{β y_i}`, a Gibbs distribution. So for any `Q`, whenever `Σ_i P_i e^{β y_i} ≤ K`,

```
Σ_i Q_i y_i ≤ ( KL(Q‖P) + ln K ) / β.
```

(If some `P_i = 0` where `Q_i > 0` then `KL(Q‖P) = ∞` and the bound is trivially true; if `Q_i = 0` I just drop those terms, which only relaxes the constraint. So no loss of generality in assuming both positive.)

Plug in `y_i = Δ(h_i)²`, `β = 2m−1`, `K = 4m/δ`. On the good event,

```
E_{h∼Q} Δ(h)² ≤ ( KL(Q‖P) + ln(4m/δ) ) / (2m−1).
```

Almost there, but I have `Δ²` and I want the gap `ℓ(Q) − ℓ̂(Q, S)`. Two Jensens. First, the gap itself: `ℓ(Q) − ℓ̂(Q, S) = E_{h∼Q}(ℓ(h) − ℓ̂(h)) ≤ E_{h∼Q} |ℓ(h) − ℓ̂(h)| = E_{h∼Q} Δ(h)`. Second, since `x ↦ x²` is convex, `(E_{h∼Q} Δ(h))² ≤ E_{h∼Q} Δ(h)²`. Chaining,

```
ℓ(Q) − ℓ̂(Q, S) ≤ E_{h∼Q} Δ(h) ≤ sqrt( E_{h∼Q} Δ(h)² ) ≤ sqrt( ( KL(Q‖P) + ln(4m/δ) ) / (2m−1) ).
```

And `ln(4m/δ) = ln(1/δ) + ln m + 2 ln 2 ≤ ln(1/δ) + ln m + 2` since `2 ln 2 ≈ 1.386 < 2`. So, with probability at least `1 − δ` over the sample, simultaneously for all posteriors `Q`,

```
ℓ(Q) ≤ ℓ̂(Q, S) + sqrt( ( KL(Q‖P) + ln(1/δ) + ln m + 2 ) / (2m − 1) ).
```

This is the bound I was after. Let me make sure it really does everything I demanded. It's distribution-free — `D` never entered any assumption; I only used Chernoff for a `[0,1]` mean, which holds for any `D`. It accepts an informative prior `P`. The complexity is `KL(Q‖P)`, not `−ln P(h)`: for continuous classes it can be finite, so the singleton-mass failure that killed the Occam bound is gone. When `Q` is a point mass at one hypothesis `h` in a countable class, `KL(Q‖P) = ln(1/P(h))` and I recover essentially the old single-hypothesis bound — so I've strictly generalized it, not traded it away. And it holds for *all* `Q` at once, even a `Q` chosen after seeing the data, because the only probabilistic step was concentrating that one prior-side quantity. The whole bound is empirical: `ℓ̂(Q, S)` and `KL(Q‖P)` are both computable, so I can report it as a certificate next to whatever predictor I produce — and that certificate is valid even if I only searched the hypothesis space incompletely, because it didn't rely on finding any particular `Q`.

The linear route still nags at me. If I put `g(h) = β Δ(h)` and transport that, I do get a valid bound after optimizing `β`, but I have to carry the one-sided moment bounds and the absolute-value bookkeeping through the whole calculation. Going directly to `y_i = Δ_i²` with the precisely tuned exponent `2m−1` makes the tail integral collapse to `∫ x e^{−x²} dx = 1/2` and lands the square-root certificate in one shot. That's the cleaner route, and it's the one to keep.

Now I'm slightly unhappy with the square-root form, and I want to chase tightness, because there's information I threw away. When I used Hoeffding's `e^{−2mx²}` tail, I treated the deviation as if the loss were always at its hardest — but for the 0–1 loss, `m·ℓ̂(h)` is `Binomial(m, ℓ(h))`, and a binomial concentrates *better* than Hoeffding when its mean is small. The natural distance for a binomial isn't squared deviation, it's the **Bernoulli relative entropy** `kl(q, p) = q ln(q/p) + (1−q) ln((1−q)/(1−p))`. So let me redo the concentration step with `kl` in the exponent and see if it tightens.

Per fixed `h`, with true risk `p = ℓ(h)` and empirical `q = ℓ̂(h)` (so `mq ∼ Binomial(m, p)`), look at the exponential moment of `m·kl(q, p)`:

```
E_S e^{m·kl(ℓ̂(h), p)} = Σ_{j=0}^m C(m, j) p^j (1−p)^{m−j} · e^{m·kl(j/m, p)}.
```

Write out `e^{m·kl(j/m, p)} = (j/m / p)^j · ((1 − j/m)/(1 − p))^{m−j}`, by the definition of `kl`. Multiply by `p^j (1−p)^{m−j}` and the `p`-dependence *cancels completely*: the term becomes `C(m, j) (j/m)^j (1 − j/m)^{m−j}`. So

```
E_S e^{m·kl(ℓ̂, p)} = Σ_{j=0}^m C(m, j) (j/m)^j (1 − j/m)^{m−j},
```

which doesn't depend on `p` at all — pretty. Now bound it. With the endpoint convention `0^0 = 1`, the entropy bound gives `C(m, j) ≤ e^{m H(j/m)}` where `H(t) = −t ln t − (1−t) ln(1−t)`. Since `(j/m)^j(1−j/m)^{m−j} = e^{−mH(j/m)}`, every term satisfies `C(m,j)(j/m)^j(1−j/m)^{m−j} ≤ 1`, and there are `m+1` terms. So

```
E_S e^{m·kl(ℓ̂(h), p)} ≤ m + 1.
```

A bound of `m+1`, cleanly, far better than carrying a loose quadratic tail. And one can do better still. The function `x ↦ e^{m kl(mean(x), μ)}` is convex and permutation-symmetric, so for `[0,1]` losses its expectation is dominated by the Bernoulli case. In that case the exact sum is `(m!/m^m) Σ_{k=1}^{m−1} k^k/k! · (m−k)^{m−k}/(m−k)! + 2`, and Stirling turns it into at most `e^{1/(12m)} sqrt(m/(2π)) Σ_{k=1}^{m−1} 1/sqrt(k(m−k)) + 2`. The sum is a Riemann-sum upper bound for `∫_0^1 dt/sqrt(t(1−t)) = π`, giving `e^{1/(12m)} sqrt(πm/2) + 2 ≤ 2√m` for `m ≥ 8`. So the `m+1` can be sharpened to `2√m`, which replaces `ln(m+1)` by `ln(2√m)`. I'll keep `m+1` for the clean statement and note the refinement.

Now the transport, in the cleanest form. Average over the prior and Markov: with probability `≥ 1 − δ`,

```
E_{h∼P} e^{m·kl(ℓ̂(h), ℓ(h))} ≤ (m+1)/δ.
```

And the change of measure, which I'll do in its Gibbs-distribution form this time because it's the most transparent. Fix a sample on the good event. Define the Gibbs measure `dP_G(h) = e^{m·kl(ℓ̂(h), ℓ(h))} dP(h) / E_{h∼P} e^{m·kl}`. Relative entropy is always nonnegative, so for any `Q`,

```
0 ≤ KL(Q‖P_G)
   = E_{h∼Q} ln( dQ/dP_G )
   = E_{h∼Q} ln( (dQ/dP) · (E_{h∼P} e^{m·kl}) / e^{m·kl(ℓ̂(h),ℓ(h))} )
   = KL(Q‖P) + ln( E_{h∼P} e^{m·kl} ) − E_{h∼Q}[ m·kl(ℓ̂(h), ℓ(h)) ].
```

Rearrange, exactly the variational inequality again:

```
E_{h∼Q}[ m·kl(ℓ̂(h), ℓ(h)) ] ≤ KL(Q‖P) + ln( E_{h∼P} e^{m·kl} ) ≤ KL(Q‖P) + ln( (m+1)/δ ).
```

Divide by `m`:

```
E_{h∼Q}[ kl(ℓ̂(h), ℓ(h)) ] ≤ ( KL(Q‖P) + ln((m+1)/δ) ) / m.
```

One last Jensen, and this is the one that makes the statement about the *averaged* risks. The map `(q, p) ↦ kl(q, p)` is jointly convex, so

```
kl( E_{h∼Q} ℓ̂(h),  E_{h∼Q} ℓ(h) ) ≤ E_{h∼Q} kl( ℓ̂(h), ℓ(h) ).
```

Writing `ℓ̂(Q) = E_{h∼Q} ℓ̂(h)` and `ℓ(Q) = E_{h∼Q} ℓ(h)`, I land on the tight form, holding with probability `≥ 1 − δ` over the sample, simultaneously for all `Q`:

```
kl( ℓ̂(Q, S),  ℓ(Q) ) ≤ ( KL(Q‖P) + ln((m+1)/δ) ) / m.
```

This is sharper than a quadratic deviation certificate in exactly the way I hoped. Because `kl(q, p) ≥ 2(p − q)²` — Pinsker — this kl-bound implies the explicit corollary `ℓ(Q) − ℓ̂(Q) ≤ sqrt((KL(Q‖P) + ln((m+1)/δ))/(2m))`. That is not the same constant as the earlier square-root bound; it is the Pinsker relaxation of the kl statement. The real gain is that, when `ℓ̂(Q)` is small, `kl(q, p)` rises more informatively than the quadratic lower bound, so the same right-hand side can force `ℓ(Q)` much closer to `ℓ̂(Q)`. To read off an explicit upper bound on the true risk, note `p ↦ kl(q, p)` is increasing for `p ≥ q`, so I invert it: `ℓ(Q) ≤ kl^{−1}( ℓ̂(Q, S); (KL(Q‖P) + ln((m+1)/δ))/m )`, the largest `p` whose Bernoulli-divergence from `ℓ̂(Q)` doesn't exceed the right-hand side — solvable numerically by a couple of Newton steps. And with the `2√m` refinement the `ln(m+1)` becomes `ln(2√m)`, halving that term.

I want to make sure I see why the exponential moment was forced on me, and not, say, a plain `E_P Δ`. The change of measure is multiplicative: `E_{h∼Q} g = E_{h∼P} (dQ/dP) g`, and the only way to peel `dQ/dP` off and pay for it with `KL(Q‖P) = E_Q ln(dQ/dP)` is to use the convex conjugacy of `x ln x`, which is the exponential — `sup_Q[ E_Q g − KL ] = log E_P e^g`. A linear moment of the deviation simply cannot be transported with `KL` as the price; the `log E_P e^{(·)}` shape is what `KL` is dual to. So the architecture is forced: I *must* control an exponential moment of the deviation under the prior, and that's exactly what Hoeffding (giving `e^{β Δ²}`) and the binomial (giving `e^{m·kl}`) hand me. Everything else is bookkeeping — Markov to go from "small mean" to "small with high probability," and Jensen to descend from `Δ²` or per-`h` `kl` to the averaged quantity.

The visible price of confidence sits in Markov on the exponential moment. That step turns `E_S Y ≤ (m+1)` into `Pr(Y > (m+1)/δ) ≤ δ`, paying the `ln(1/δ)` and carrying the polynomial prefactor into `ln(m+1)` (or `ln(2√m)`). There is still ordinary slack in the entropy/Stirling upper bound and in Jensen unless the per-hypothesis risks line up, but the `ln m` term enters through this prior-side moment-and-Markov step. If I ever wanted to remove it, that's the step to attack.

There's a payoff I didn't ask for but which falls out, and it tells me the bound is pointing at the right algorithm. The right-hand side `ℓ̂(Q) + sqrt((KL(Q‖P) + K)/γ)`, with `K = ln(1/δ)+ln m+2` and `γ = 2m−1`, is a functional of `Q` that I can *minimize* over `Q`. In a finite class, write `P_i` and `Q_i` for the prior and posterior masses and `ℓ̂_i` for the empirical risks. At an interior optimum the simplex constraint gives one multiplier `λ`, so every active coordinate has the same derivative:

```
λ = ∂/∂Q_i [ Σ_j Q_j ℓ̂_j + sqrt((KL(Q‖P)+K)/γ) ]
  = ℓ̂_i + (1/(2 sqrt(γ(KL(Q‖P)+K)))) (1 + ln(Q_i/P_i)).
```

Rearrange this derivative condition:

```
ln(Q_i/P_i) = 2 sqrt(γ(KL(Q‖P)+K)) (λ − ℓ̂_i) − 1.
```

All the terms not depending on `i` disappear into the normalizing constant. If

```
β = 2 sqrt( γ(KL(Q‖P)+K) ).
```

then the minimizing posterior has the Gibbs form

```
dQ_β(h) ∝ e^{−β ℓ̂(h, S)} dP(h),
```

with the self-consistency condition

```
β = 2 sqrt( (2m−1)(KL(Q_β‖P) + ln(1/δ) + ln m + 2) ).
```

The bound didn't just certify a predictor — it *recommended* one, and the recommendation is the exponentially-weighted average over hypotheses, the same Gibbs/Boltzmann weighting that shows up in weighted-majority and in exponentially-weighted aggregation. Selecting was never the right move; the stochastic predictor that minimizes this certificate is a posterior, and it's Gibbs. As `β → 0` it's the prior (blind to data); as `β → ∞` it collapses onto the empirical risk minimizer. The whole Bayesian-averaging instinct I started from comes back as the optimum of a distribution-free bound.

Let me write the certificate as I'd compute it, to be concrete. For a finite hypothesis class with prior vector `P`, given a sample, I form the empirical risk of each hypothesis, choose any posterior `Q` (or the Gibbs `Q_β`), compute `KL(Q‖P)` and the averaged empirical risk `ℓ̂(Q)`, and return the right-hand side — for the tight form, invert the Bernoulli divergence.

```python
import numpy as np

def kl_div(Q, P):                      # KL(Q || P), the complexity term
    Q = np.asarray(Q, float); P = np.asarray(P, float)
    if np.any((Q > 0) & (P == 0)):
        return float("inf")
    mask = Q > 0
    return float(np.sum(Q[mask] * np.log(Q[mask] / P[mask])))

def emp_risk_Q(Q, emp_risks):          # E_{h~Q} empirical risk
    return float(np.dot(Q, emp_risks))

def kl_ber(q, p):                      # Bernoulli relative entropy kl(q || p)
    q = float(np.clip(q, 0.0, 1.0)); p = float(np.clip(p, 0.0, 1.0))
    if p == 0.0:
        return 0.0 if q == 0.0 else float("inf")
    if p == 1.0:
        return 0.0 if q == 1.0 else float("inf")
    out = 0.0
    if q > 0:      out += q * np.log(q / p)
    if q < 1:      out += (1 - q) * np.log((1 - q) / (1 - p))
    return float(out)

def kl_inverse(q, c):                  # largest p >= q with kl(q || p) <= c  (bisection)
    q = float(np.clip(q, 0.0, 1.0))
    if c < 0:
        raise ValueError("kl radius must be nonnegative")
    if q >= 1.0 or c == 0:
        return q
    lo, hi = q, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kl_ber(q, mid) > c: hi = mid
        else:                  lo = mid
    return lo

def pac_bayes_certificate(Q, P, emp_risks, m, delta, form="kl"):
    """With prob >= 1 - delta over the sample, simultaneously for ALL Q,
    an upper bound on the true risk of the Gibbs predictor drawn from Q.
    Finite for continuous classes (KL replaces -ln P(h)); computable from data."""
    Q = np.asarray(Q, float); P = np.asarray(P, float)
    KL = kl_div(Q, P)
    rhat = emp_risk_Q(Q, emp_risks)
    if form == "sqrt":                 # square-root form
        return rhat + np.sqrt((KL + np.log(1/delta) + np.log(m) + 2) / (2*m - 1))
    c = (KL + np.log((m + 1) / delta)) / m        # kl-form bound
    return kl_inverse(rhat, c)         # invert kl(rhat || .) to bound the true risk

def gibbs_posterior(P, emp_risks, beta):          # the optimal posterior the bound recommends
    P = np.asarray(P, float); emp_risks = np.asarray(emp_risks, float)
    with np.errstate(divide="ignore"):
        log_w = np.log(P) - beta * emp_risks
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()
```

So the causal chain, start to finish: I wanted a guarantee that takes a prior like Bayes but holds distribution-free like PAC and survives continuous classes; the obstruction was that a single-hypothesis Occam bound charges `−ln P(h)` and needs a union bound over the whole class, which is infinite for continuous classes and loose elsewhere; the escape was to score a *randomized* predictor `Q` and only ever transport one statement from the prior to `Q`; the price of that transport is `KL(Q‖P)`, by the Donsker–Varadhan / Gibbs duality that I derived as a constrained optimization; the transport runs on an *exponential* moment of the deviation under the prior, where the square-root route gives `E_S E_P e^{(2m−1)Δ²} ≤ 4m`, while the Bernoulli-entropy route gives `E_S E_P e^{m·kl} ≤ m+1` and the sharper `2√m` constant for `m ≥ 8`; Markov makes the prior-side bound hold with probability `1 − δ` *before* `Q` is chosen, which is why the bound holds for *all* `Q` at once; and Jensen descends to the averaged risk, landing `ℓ(Q) ≤ ℓ̂(Q) + sqrt((KL(Q‖P) + ln(1/δ) + ln m + 2)/(2m−1))` and its tight companion `kl(ℓ̂(Q), ℓ(Q)) ≤ (KL(Q‖P) + ln((m+1)/δ))/m`. The same variational identity that gave the complexity also names the optimal predictor — the Gibbs posterior `dQ_β ∝ e^{−β ℓ̂} dP`. Complexity stopped being the prior mass of one hypothesis and became the information gained in moving from the prior to the posterior.
