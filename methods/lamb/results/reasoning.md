I start from the wall-clock problem: pre-training a deep transformer takes three days on sixteen accelerator chips, and a deep image classifier takes the better part of a day. I have a pod with hundreds of chips sitting there. Why can't I just throw all of them at the problem and finish in an hour?

The answer is that SGD is sequential. Update `t+1` waits on update `t`. The only thing I can parallelize *within* a step is the gradient computation: split a mini-batch of `b` examples across many devices, each computes part of `g = (1/b) Σ ∇ℓ(x, s)`, sum the partials. So more devices buys me a bigger batch, not more steps. Double the chips, double `b`. Fine — so what does doubling `b` actually buy me?

Two things, and they pull in opposite directions. The good thing: `g` is an unbiased estimate of the true gradient `∇f`, and averaging over `b` independent samples cuts its variance by `1/b`. A lower-variance gradient is a more trustworthy direction, so I can afford a bigger step. The bad thing: if I hold the number of *epochs* fixed — which I have to, because that's roughly the amount of data the model needs to see — then the number of optimizer updates is `T = (#examples × epochs)/b`, and that falls *linearly* in `b`. Batch goes 512 → 32k, that's 64×, so I get 64× fewer steps. Each surviving step has to do 64× more work or the model just doesn't finish learning.

So the obvious move is: bigger batch, bigger learning rate. The variance argument says the standard deviation of `g` falls like `1/√b`, so scale `η` by `√b`. People have also found empirically that scaling `η` *linearly* with `b` works better up to a point. Either way, crank `η` up to compensate for the lost steps.

And here's where it falls apart. Past some batch-size ceiling, cranking `η` doesn't just slow down — it destabilizes. Large batches drift toward sharp minima that generalize worse; you can partly recover by training *longer*, which is exactly the symptom of having taken too few steps. And the large `η` is poison specifically in the *early* phase — people patched that with a hand-tuned warmup, ramping `η` from near-zero up over the first few epochs and then switching to the normal decay, which got a batch of 8192 on the image classifier without losing accuracy. But when these recipes get measured across a pile of tasks they just *don't transfer*: the ceiling moves, the right exponent moves, problem to problem. So "scale `η` and warm up" is a fragile, per-problem hand-tune. I want something I can push to 32k on a brand-new task without re-deriving the recipe.

Let me stare at *why* a single global `η` is the thing that keeps breaking. A network is a stack of layers, and the layers are nothing alike. A normalization gain, an embedding table, a deep weight matrix — their parameter norms `‖x^(i)‖` differ by orders of magnitude, and so do the norms of the steps the optimizer wants to take on them, `‖u^(i)‖`. A global `η` slaps the *same* multiplier on every layer. Suppose layer A wants a step whose norm is a tiny fraction of its weight norm, and layer B wants a step that's a large fraction of *its* weight norm. I pick `η` big enough that layer A actually moves — and now layer B is moving a huge fraction of its own norm in a single step. That's the divergence. The instability isn't really about the batch; it's that one number can't be simultaneously right for a layer with ratio 0.001 and a layer with ratio 1. Big batch just lets me crank `η` high enough that the mismatch becomes fatal.

So the fix shouldn't be a cleverer *global* schedule. It should be: give each layer its own effective step, matched to *its* geometry. What's the right per-layer scale? I want each layer to move by an amount commensurate with its own size — say, the step norm should be on the order of the weight norm `‖x^(i)‖`. If the base optimizer hands me a raw direction `u^(i)` for layer `i`, then I should kill its arbitrary magnitude and re-impose a magnitude tied to the layer. Normalize: `u^(i)/‖u^(i)‖` is a pure unit direction. Then scale it by something like `‖x^(i)‖`. The per-layer update becomes

`x_{t+1}^(i) = x_t^(i) − η · ‖x_t^(i)‖ · u_t^(i)/‖u_t^(i)‖`.

Now `η` is just a single global knob multiplying *unit-direction × weight-norm*; the actual distance layer `i` travels is `η‖x^(i)‖`, automatically proportional to how big that layer is. Layer A and layer B each get a step sized to themselves. The global `η` is finally decoupled from per-layer scale. Let me write the two moves cleanly, because I'll reuse them: take any base optimizer `A` that produces a layerwise direction `u_t` (so vanilla would be `x ← x + η u_t`), and (1) normalize each layer's update to unit `ℓ2` norm, `u^(i) → u^(i)/‖u^(i)‖`; (2) scale the per-layer learning rate by `φ(‖x^(i)‖)` for some function `φ : ℝ⁺ → ℝ⁺`.

Why normalize away the magnitude entirely? At small batch I'd be nervous — the gradient magnitude carries real information. But at large batch the gradient *direction* is estimated very well (low variance), while its raw magnitude is exactly the thing that varies wildly across layers and blows things up. So throwing the magnitude away and re-imposing a layer-matched one is cheap in bias and buys robustness: a layer with an exploding gradient and a layer on a plateau both get the same unit-norm treatment. It does introduce a small bias in the update direction, but in the large-batch regime that bias is small precisely because the direction is well-estimated.

Now `φ`. Take `φ(z) = z` first, the identity. Then the per-layer multiplier in front of the unit direction is `‖x^(i)‖/‖u^(i)‖`. Is there a reading of that? If `u^(i)` is roughly the gradient, then `‖x^(i)‖/‖u^(i)‖` is a step size that's large when the weights are large relative to the gradient and small when they're not — and `1/(step size)` looks like a curvature. In fact, if layer `i` is `L_i`-smooth, the textbook safe step for gradient descent is `1/L_i`, and `‖x^(i)‖/‖∇_i f‖` is a cheap running *estimate* of `1/L_i`. So the trust ratio isn't arbitrary; with `φ = id` it's a per-layer estimate of the inverse local smoothness. That's reassuring — it says the scheme is automatically doing per-layer second-order-ish step sizing without any Hessian.

But pure identity is dangerous at the extremes. If a layer's weights are huge, `‖x^(i)‖` is huge and the step explodes; if the update norm `‖u^(i)‖` is near zero, same thing. So the mathematical strategy allows clipping: `φ(z) = min(max(z, γ_l), γ_u)`. Identity in the middle, flat at the ends — a clipped identity. The convergence theorem assumes a positive lower and upper bound `α_l ≤ φ ≤ α_u`; the cleanest core uses the unclipped identity `φ(z)=z` with a zero-norm fallback, while one can also add an upper cap such as `10`. So clipping is a valid implementation choice, but not the canonical core.

Good — this is a *strategy*, not yet an algorithm, because I haven't said what the base optimizer `A` is. Let me instantiate it. The most natural first try: `A` = momentum SGD, `u_t = m_t` with `m_t = β₁ m_{t-1} + (1−β₁)(g_t + λ x_t)`. Plug in:

`x_{t+1}^(i) = x_t^(i) − η_t · (φ(‖x_t^(i)‖)/‖m_t^(i)‖) · m_t^(i)`.

This is exactly a layerwise-trust-ratio momentum scheme, the kind of thing already shown to train a deep image classifier at batch 32k in minutes. So the *framework* is sound. But there are two problems I can't ignore. First, there's no convergence theory — it's a heuristic, and I'd like to know *when* it actually beats SGD. Second, and worse: when I take this momentum-based layerwise scheme to an attention/language model, it trains poorly and at the biggest batches it diverges. That's a sharp clue. Momentum SGD is known to underperform on these models in the first place; what works on transformers is per-*coordinate* adaptivity — Adam, which divides each coordinate by the running root-mean-square of its own gradient. The layerwise trust ratio handles the *across-layer* scale mismatch, but within a layer of a transformer the coordinates themselves are wildly ill-conditioned, and a momentum base does nothing about that.

So the two adaptivities are orthogonal and I want *both*: per-coordinate rescaling inside a layer (Adam) *and* per-layer rescaling across layers (trust ratio). Set the base optimizer `A` = Adam. Adam's raw direction is

`m_t = β₁ m_{t-1} + (1−β₁) g_t`, `v_t = β₂ v_{t-1} + (1−β₂) g_t²`,
`m̂_t = m_t/(1−β₁^t)`, `v̂_t = v_t/(1−β₂^t)`, `r_t = m̂_t/(√v̂_t + ε)`.

So `r_t` is the per-coordinate-normalized Adam update — that's my `u_t`. Now wrap it in the layerwise strategy. Per layer:

`x_{t+1}^(i) = x_t^(i) − η_t · (φ(‖x_t^(i)‖)/‖r_t^(i)‖) · r_t^(i)`.

Two-fold adaptivity, exactly as wanted: `r_t` rescales each *dimension* by `1/√v̂`, and the trust ratio `φ(‖x^(i)‖)/‖r_t^(i)‖` rescales each *layer*. The global `η` is decoupled from both.

One more thing to fold in: weight decay. On transformers I want decoupled decay (the AdamW lesson — don't push `λx` through the adaptive denominator, apply it as its own term), but I also have to be careful *where* it sits relative to the layerwise normalization. If I add `λ x^(i)` to the direction *after* normalizing, the decay isn't seen by the trust ratio and its scale floats free of the layer step. So put it inside: the effective per-layer direction is `r_t^(i) + λ x_t^(i)`, and *that* is what gets normalized and trust-ratio'd:

`x_{t+1}^(i) = x_t^(i) − η_t · (φ(‖x_t^(i)‖)/‖r_t^(i) + λ x_t^(i)‖) · (r_t^(i) + λ x_t^(i))`.

Now the decay rides inside the same unit-norm-and-rescale machinery as the gradient part, so its magnitude stays commensurate with the layer's step. That's the update. Let me note a sanity-check special case: set `β₁ = 0` and `β₂ = 0`. Then `m̂ = g`, `v̂ = g²`, and `r_j = g_j/(|g_j| + ε)`, which is the sign of `g_j` up to the numerical `ε`. The layer update becomes a normalized sign vector — `sign(g)` divided by its norm `√(d_i)` — i.e. signSGD scaled by the square root of the layer dimension.

Now I owe myself the thing the heuristic scheme never had: a convergence guarantee, and a precise statement of *when* this beats SGD. Let me set up the nonconvex stochastic problem `min_x f(x) = E_s[ℓ(x, s)] + (λ/2)‖x‖²`. Assume `ℓ` is `L_i`-smooth in the layer-`i` block: `‖∇_i ℓ(x) − ∇_i ℓ(y)‖ ≤ L_i ‖x^(i) − y^(i)‖`. Collect `L = (L_1, …, L_h)`, write `L_∞ = max_i L_i`, `L_avg = (1/h) Σ_i L_i`, `‖L‖_1 = Σ_i L_i`. Bounded per-layer gradient variance `E‖∇_i ℓ − ∇_i f‖² ≤ σ_i²`, per-dimension version `σ̃`, and a coordinatewise gradient bound `|[∇ℓ]_j| ≤ G`. The benchmark I'm trying to beat is SGD with `b = T`, whose standard nonconvex rate is `E‖∇f‖² ≤ O((f(x_1) − f*)L_∞/T + ‖σ‖²/T)`. Note that `L_∞`. The whole network's rate is hostage to its single worst-conditioned layer. If my layerwise scheme can replace `L_∞` by `L_avg`, that's a real, structural win whenever curvature is uneven across layers.

Let me do the trust-ratio scheme cleanly first with the simplest base — set `β₁ = 0`, `λ = 0`, and let the base direction be the raw normalized gradient (this is the trust-ratio scheme with momentum off; it's the cleaner sibling and the LAMB argument reuses every step). The update is `x_{t+1}^(i) = x_t^(i) − η_t φ(‖x_t^(i)‖) g_t^(i)/‖g_t^(i)‖`.

Start from the per-layer smoothness descent lemma. Smoothness of `f` gives, summing the per-block quadratic bounds,
`f(x_{t+1}) ≤ f(x_t) + Σ_i ⟨∇_i f(x_t), x_{t+1}^(i) − x_t^(i)⟩ + Σ_i (L_i/2)‖x_{t+1}^(i) − x_t^(i)‖²`.
The step on layer `i` is `x_{t+1}^(i) − x_t^(i) = −η_t φ(‖x_t^(i)‖) g_t^(i)/‖g_t^(i)‖`, whose norm is exactly `η_t φ(‖x_t^(i)‖) ≤ η_t α_u` (using `α_l ≤ φ ≤ α_u`). So the curvature term is bounded with no gradient dependence at all:
`Σ_i (L_i/2)‖Δx^(i)‖² ≤ (η_t² α_u²/2) Σ_i L_i = (η_t² α_u²/2)‖L‖_1`.
That's the payoff of normalizing — the second-order term can't blow up with the gradient. Now the linear term. Write it coordinatewise and add-and-subtract the *true*-gradient unit direction:
`⟨∇_i f, Δx^(i)⟩ = −η_t φ(‖x_t^(i)‖) Σ_j [∇_i f]_j ( g_{t,j}^(i)/‖g_t^(i)‖ − [∇_i f]_j/‖∇_i f‖ + [∇_i f]_j/‖∇_i f‖ )`.
The clean piece, summing the last term over `j`, is `−η_t φ(‖x_t^(i)‖) ‖∇_i f‖` — pure descent proportional to the layer's gradient norm. The messy piece is the difference between stepping along the *stochastic* unit direction `g/‖g‖` and the *true* unit direction `∇f/‖∇f‖`. Let `Δ_t^(i) = g_t^(i) − ∇_i f(x_t)`. I need to bound

`φ(‖x^(i)‖) ⟨∇_i f, ∇_i f/‖∇_i f‖ − g^(i)/‖g^(i)‖⟩`.

Pull `∇_i f = (Δ + ∇_i f) − Δ` and use `g = Δ + ∇_i f`. The quantity to control is
`‖∇_i f‖·‖g^(i)‖ − ⟨g^(i), ∇_i f⟩` all over `‖g^(i)‖`. Write `g = Δ + ∇_i f` in the numerator:
`‖∇_i f‖‖g^(i)‖ − ⟨Δ + ∇_i f, ∇_i f⟩ = ‖∇_i f‖‖g^(i)‖ − ‖g^(i)‖² + ⟨Δ, g^(i)⟩`
after adding and subtracting `‖g^(i)‖²` and regrouping `⟨g, ∇_i f⟩ = ‖g‖² − ⟨Δ, g⟩`. Now `‖∇_i f‖ − ‖g^(i)‖ ≤ ‖∇_i f − g^(i)‖ = ‖Δ‖` by the reverse triangle inequality, and `⟨Δ, g⟩/‖g‖ ≤ ‖Δ‖` by Cauchy-Schwarz. So the whole error term per layer is at most `2‖Δ_t^(i)‖`, and the linear term is

`⟨∇_i f, Δx^(i)⟩ ≤ −η_t φ(‖x^(i)‖)‖∇_i f‖ + 2 η_t φ(‖x^(i)‖)‖Δ_t^(i)‖`.

Put the pieces back:
`f(x_{t+1}) ≤ f(x_t) − η_t Σ_i φ(‖x^(i)‖)‖∇_i f‖ + 2η_t Σ_i φ(‖x^(i)‖)‖Δ_t^(i)‖ + (η_t² α_u²/2)‖L‖_1`.
Take expectation. The signal term uses `φ ≥ α_l`. The noise term uses `φ ≤ α_u` and `E‖Δ_t^(i)‖ ≤ √(E‖Δ_t^(i)‖²) ≤ σ_i/√b` (variance of a `b`-sample mean):
`E[f(x_{t+1})] ≤ f(x_t) − η_t α_l Σ_i ‖∇_i f‖ + 2η_t α_u ‖σ‖_1/√b + (η_t² α_u²/2)‖L‖_1`.
Hold `η_t = η`, sum `t = 1..T`, telescope `Σ(f(x_t) − E f(x_{t+1})) = f(x_1) − E f(x_{T+1}) ≤ f(x_1) − f*`, divide by `ηTα_l`:
`(1/T) Σ_t Σ_i E‖∇_i f(x_t)‖ ≤ (f(x_1) − f*)/(Tηα_l) + 2α_u‖σ‖_1/(α_l√b) + ηα_u²‖L‖_1/(2α_l)`.
Set `b = T` and pick `η = √(2(f(x_1)−f*)/(α_u²‖L‖_1 T))` to balance the first and third terms, and divide through by `√h`:
`(E (1/√h) Σ_i ‖∇_i f(x_a)‖)² ≤ O((f(x_1) − f*) L_avg / T + ‖σ‖_1²/(Th))`,
where the `‖L‖_1·(1/√h)²` collapses to `L_avg` and `x_a` is a uniformly-random iterate. There it is — the rate depends on `L_avg`, not `L_∞`. The normalized layerwise step traded the worst-layer constant for the average one.

Now the LAMB version, base = Adam, `β₂ > 0`, again with `β₁ = 0`, `λ = 0` to keep it readable (the general case is the same skeleton, just messier). Update `x_{t+1}^(i) = x_t^(i) − η_t φ(‖x_t^(i)‖) r_t^(i)/‖r_t^(i)‖`. The descent lemma and the curvature bound are identical — the step norm is still `η_t φ ≤ η_t α_u`, so the second-order term is again `(η_t² α_u²/2)‖L‖_1`. What changes is the signal term

`T_1 = −η_t Σ_i Σ_j φ(‖x^(i)‖) [∇_i f]_j r_{t,j}^(i)/‖r_t^(i)‖`.

With `β₁ = 0`, `m̂ = g`, so `r_j = g_j/(√v̂_j + ε)`. Two facts I need about `r`. The single freshest gradient contributes weight `(1−β₂)` to `v̂_j`, so `√v̂_j ≥ √(1−β₂)·|g_j|`, hence each coordinate is bounded `|r_j| ≤ |g_j|/(√(1−β₂)|g_j|) = 1/√(1−β₂)`, and over `d_i` coordinates `‖r_t^(i)‖ ≤ √(d_i/(1−β₂))`. And the running RMS is bounded by the gradient bound, `√v̂_j ≤ G`. Use these to lower-bound the contribution of each coordinate where the stochastic update agrees in sign with the true gradient, and separately bound the coordinates where the signs *disagree*.

Split the sum over `j` into sign-agreement and sign-disagreement. On agreement coordinates, `[∇_i f]_j r_j/‖r^(i)‖ ≥ √((1−β₂)/(G² d_i)) [∇_i f]_j g_j` — I've replaced the unit-direction coordinate by its lower bound using `‖r^(i)‖ ≤ √(d_i/(1−β₂))` and `√v̂ ≤ G`, which turns `r_j/‖r^(i)‖` into `≥ √((1−β₂)/(G²d_i)) g_j` on the matching-sign set. So

`T_1 ≤ −η_t Σ_i Σ_j √((1−β₂)/(G²d_i)) φ(‖x^(i)‖) [∇_i f]_j g_j − η_t Σ_i Σ_j φ(‖x^(i)‖) [∇_i f]_j (r_j/‖r^(i)‖) 𝟙(sign[∇_i f]_j ≠ sign r_j)`.

On the disagreement coordinates I just want an upper bound on the (positive) penalty: `|φ [∇_i f]_j r_j/‖r^(i)‖| ≤ α_u |[∇_i f]_j|` since `|r_j|/‖r^(i)‖ ≤ 1`. Take expectations. The first term is over all coordinates, and unbiasedness gives `E[g_j] = [∇f]_j`, so `E[[∇_i f]_j g_j] = [∇_i f]_j²`. The disagreement term is bounded by `α_u |[∇_i f]_j| · P(sign[∇_i f]_j ≠ sign g_j)`. And here's where the signSGD device earns its keep: by Markov on the bounded-variance gradient, the probability a stochastic coordinate's sign flips relative to the true gradient is `P(sign g_j ≠ sign ∇f_j) ≤ σ_{i,j}/(√b |[∇_i f]_j|)`. The `|[∇_i f]_j|` cancels, leaving `α_u σ_{i,j}/√b` per coordinate. Summing the signal term over coordinates gives the squared full gradient norm:

`E[T_1] ≤ −η_t α_l √(h(1−β₂)/(G²d)) ‖∇f(x_t)‖² + η_t α_u ‖σ̃‖_1/√b`,

using `d_i = d/h` so `√(1/d_i) = √(h/d)` and `φ ≥ α_l` on the signal, `φ ≤ α_u` on the noise. Substitute into the descent lemma:
`E[f(x_{t+1})] ≤ f(x_t) − η_t α_l √(h(1−β₂)/(G²d)) ‖∇f(x_t)‖² + η_t α_u ‖σ̃‖_1/√b + (η_t² α_u²/2)‖L‖_1`.
Telescope over `t`, divide by `ηTα_l`, set `b = T`, `η = √(2(f(x_1)−f*)/(α_u²‖L‖_1 T))`:
`√(h(1−β₂)/(G²d)) · (1/T) Σ_t E‖∇f(x_t)‖² ≤ (f(x_1)−f*)/(Tηα_l) + α_u‖σ̃‖_1/(α_l√b) + ηα_u²‖L‖_1/(2α_l)`,
and solving for `E‖∇f‖²` pulls the `√(G²d/(h(1−β₂)))` to the other side:
`E‖∇f(x_a)‖² ≤ O( √(G²d/(h(1−β₂))) · [ √(2(f(x_1)−f*)‖L‖_1/T) + ‖σ̃‖_1/√T ] )`.
For `β₂ = 0` the coordinate bound sharpens — the agreement term carries `√(1/d_i) φ |[∇_i f]_j|` directly — and the same telescoping yields `(E (1/√d)‖∇f(x_a)‖_1)² ≤ O((f(x_1)−f*)L_avg/T + ‖σ̃‖_1²/(Th))`, the same `L_avg`-not-`L_∞` shape as the trust-ratio scheme.

So both versions provably converge to a stationary point, but only the trust-ratio theorem and the `β₂ = 0` LAMB theorem have the clean `L_avg` dependence. The `β₂ > 0` LAMB theorem is looser: it carries `√(G²d/(h(1−β₂)))` times a term involving `‖L‖_1`. Is `L_avg` actually a *win* over SGD's `L_∞` in the clean cases? Not unconditionally — the convergence *criterion* changed too (I'm bounding `(Σ_i ‖∇_i f‖)²` or `‖∇f‖_1²`, not `‖∇f‖²`), so I have to compare like with like. Borrow the signSGD-style density bookkeeping. Define `(Σ_i ‖∇_i f‖)² = ψ_g · d‖∇f‖²/h`, `‖L‖_1² ≤ ψ_L · d²L_∞²/h²`, `‖σ‖_1² = ψ_σ · d‖σ‖²/h`, where each `ψ` measures how *dense* (spread out) that quantity is across coordinates. Substituting, the layerwise rate rewrites as `O((f(x_1)−f*)L_∞/T · ψ_L/ψ_g² + ‖σ‖²/T · ψ_σ²/ψ_g²)`. So I beat SGD exactly when `ψ_L ≪ ψ_g²` and `ψ_σ ≪ ψ_g²` — when the **gradient is denser than the curvature and the noise**. That's the precise condition. The layerwise trust ratio helps when the signal is spread across many coordinates/layers while the curvature and stochasticity are concentrated; it does *not* help when curvature is as dense as the gradient.

A couple of practical things the derivation suggests. The debiasing `m̂ = m/(1−β₁^t)`, `v̂ = v/(1−β₂^t)` changes the effective learning-rate schedule by the familiar Adam correction factor, so if I am already running an explicit warmup schedule, the two mechanisms overlap and the correction can be removed in practice. For the per-layer norm in the trust ratio I've been writing `ℓ2`; in principle any norm would do, and swapping it barely moves the result, so `ℓ2` is the default. The clipped form `φ(z) = min(max(z, γ_l), γ_u)` is the mathematical guardrail when I want bounded trust-ratio numerators, while the simplest implementation uses the unclipped identity and relies on the zero-norm fallback; an upper clamp can be added as an engineering choice.

Let me also make sure the across-batch-size story needs no per-task tuning. Since the gradient variance falls as `1/b`, the natural learning-rate knob is `√b` scaling, paired with a warmup whose length grows with the number of steps (linear-epoch warmup). With the layerwise trust ratio absorbing the per-layer geometry, I shouldn't have to re-tune as I sweep batch 512 → 32k — set the base `η` once, scale by `√b`, warm up proportionally. That's the whole point: the trust ratio is what makes a single recipe survive the sweep.

Now the code. Per parameter group ("layer"): maintain Adam's two EMAs, form the Adam direction, fold in decoupled decay `r + λx`, compute the layer's weight norm and the direction norm, take their ratio as the trust ratio (defaulting to 1 if either norm is zero so a dead layer just gets the base Adam direction), and step by `−η · trust_ratio · direction`. One variant uses bias correction exactly as derived; another omits bias correction and relies on explicit warmup. I can show that as a toggle without changing the update order.

```python
import torch
from torch.optim import Optimizer

class Lamb(Optimizer):
    # Layerwise-adaptive Adam for the large-batch regime.
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, debias=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.debias = debias             # Adam bias correction; can be omitted with explicit warmup
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)     # m
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # v
                m, v = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Adam moment EMAs: m_t, v_t
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                step_size = group['lr']
                if self.debias:
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    r = (m / bc1) / ((v / bc2).sqrt().add(group['eps']))
                else:
                    r = m / (v.sqrt().add(group['eps']))
                # decoupled weight decay inside the trust-ratio direction: r + lambda*x
                if group['weight_decay'] != 0:
                    r = r.add(p.data, alpha=group['weight_decay'])

                # core uses phi(||x||)=||x||; an upper clamp is an optional variant.
                weight_norm = p.data.pow(2).sum().sqrt()
                r_norm = r.pow(2).sum().sqrt()
                if weight_norm == 0 or r_norm == 0:
                    trust_ratio = 1.0          # dead layer -> plain Adam step
                else:
                    trust_ratio = weight_norm / r_norm

                # per-layer step ~ proportional to weight norm; global lr decoupled
                p.data.add_(r, alpha=-step_size * trust_ratio)
        return loss
```

The causal chain, end to end: I want hardware-parallel speed, which forces large batches; large batches cut the step count, which forces a bigger learning rate; a single global learning rate is wrong because layers have wildly different weight-to-update ratios, so it destabilizes the worst-matched layer; the fix is to give each layer its own effective step — normalize the per-layer update to a unit direction and rescale it by the layer's weight norm via a (possibly clipped) trust ratio `φ(‖x^(i)‖)/‖u^(i)‖`, which decouples the global rate from per-layer geometry and reads as a `1/L_i` estimate; choosing the base optimizer fixes the algorithm — momentum gives the trust-ratio scheme but fails on attention models because it lacks per-coordinate adaptivity, while Adam-as-base supplies that, yielding the two-fold (per-dimension × per-layer) update `x^(i) ← x^(i) − η φ(‖x^(i)‖)/‖r^(i)+λx^(i)‖ (r^(i)+λx^(i))`; and the nonconvex analysis confirms convergence, with the clean `L_avg`-rather-than-`L_∞` comparison in the trust-ratio and `β₂=0` LAMB cases, and a looser but still stationary-point guarantee for `β₂>0`. That is the technical reason the method can scale BERT to very large batches with a single `√b`-plus-warmup recipe.
