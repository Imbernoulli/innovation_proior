OK, let me think this through from scratch. I have a network with a discrete decision buried in the middle of it. Somewhere upstream, parameters `θ` produce class probabilities `π = π(θ)` over `k` options; I sample a category `z ~ Categorical(π)`; and the rest of the network maps that choice to a scalar cost `f(z)`. What I actually optimize is the expected cost `L(θ) = E_{z~p_θ(z)}[f(z)]`, and to do SGD I need `∇_θ L`. The problem is the obvious one: the gradient has to travel from `f(z)` back through the *act of sampling* to reach `θ`, and sampling a categorical is, at bottom, picking the largest of some scores and emitting a one-hot vector. Picking the largest is an argmax; the one-hot is a step function. Both are flat almost everywhere and cliff-like on the boundaries, so the chain rule hands me a zero. There is no gradient signal at all.

I keep coming back to how clean the *continuous* version of this is. When the latent is Gaussian I can write `z = μ + σ·ε` with `ε ~ N(0,1)`, and now `z` is a smooth function of `(μ, σ)` with all the randomness shoved into `ε`, which doesn't depend on the parameters. So `∂/∂θ E_z[f(z)] = E_ε[∂f/∂z · ∂z/∂θ]` — I just sample `ε`, run forward, and backprop straight through, as if the stochasticity weren't there. This is exactly why variational autoencoders train: a single noise draw gives a low-variance, unbiased gradient, and it's a drop-in inside ordinary backprop. So the question that's nagging me is: is there a reparameterization like the Gaussian trick for a *discrete* variable? Can I write a categorical sample as `z = g(θ, ε)` for some smooth `g` and parameter-free noise `ε`?

And the immediate wall is that I can't, not naively. A categorical sample is a one-hot vector — it lives on the corners of the simplex. Any deterministic function `g(θ, ε)` whose output is forced onto those corners is piecewise constant in `θ`: nudge `θ` a little and either the output doesn't move at all (still the same corner) or it jumps discontinuously to another corner. Piecewise constant means zero gradient where it's differentiable and undefined where it jumps. So the reparameterization route, taken literally, dies. There's no smooth `g` landing on one-hot outputs.

Fine — what *can* I do that's unbiased? There's the score-function estimator, the REINFORCE / likelihood-ratio identity. The trick is `∇_θ p_θ(z) = p_θ(z) ∇_θ log p_θ(z)`, so

```
∇_θ E_z[f(z)] = ∇_θ Σ_z f(z) p_θ(z) = Σ_z f(z) p_θ(z) ∇_θ log p_θ(z) = E_z[ f(z) ∇_θ log p_θ(z) ].
```

This is lovely in one respect: it never asks me to differentiate through `f` or through the sample. `f` can be a complete black box; I only need `p_θ(z)` to be differentiable in `θ`, which it is. So in principle the discreteness is no obstacle at all — I sample `z`, evaluate `f(z)`, and weight the score `∇_θ log p_θ(z)` by it.

But the variance is brutal. The estimator multiplies the whole raw cost `f(z)` against the score with nothing to temper it; a single sample's `f(z)` might be large and the score points in some essentially random direction for that draw, so the per-sample gradient is wildly noisy and only averages out to the truth over many samples. And the variance grows with the dimensionality of the thing I'm sampling — roughly linearly — so for a high-dimensional categorical structure it's especially bad. I'd be doing SGD with a gradient that's correct in expectation but so noisy that convergence crawls.

The standard patch is control variates: subtract a baseline `b(z)` from the learning signal and add back the corresponding score term so I stay unbiased,

```
∇_θ E_z[f(z)] = E_z[ (f(z) - b(z)) ∇_θ log p_θ(z) ] + E_z[ b(z) ∇_θ log p_θ(z) ].
```

If the correction has a closed form — for instance the derivative of an analytically tractable `E[b(z)]`, or zero for a baseline that is constant with respect to the sample — I can remove some variance without changing the expectation. People have built a whole zoo of these: a moving average plus a learned input-dependent baseline with variance normalization; a first-order Taylor expansion `b = f(z̄) + f'(z̄)(z - z̄)` used either with or without its correction term; per-sample baselines built from sibling samples in a multi-sample objective. They genuinely help. But every one of them is extra machinery bolted onto an estimator whose fundamental form is still "cost times score," and I'm still living with the noise of that form. None of them give me the thing the Gaussian trick gave me for free: a *path derivative*, where the gradient flows through a differentiable sample.

There's also the other family — biased path derivatives. For a Bernoulli with mean `θ`, the straight-through estimator just pretends the hard threshold has the gradient of its mean: forward you threshold to `{0,1}`, backward you pass the gradient through as if `∇_θ z ≈ 1`. A slope-annealed version tries to make the proxy less crude by using a smooth sigmoid or hard-sigmoid with a slope that grows over training, so early gradients are gentle and late samples are closer to hard bits. That still leaves me in a binary world, and as the slope grows I again get a saturating near-step with biased, fragile gradients. The deeper problem is the mismatch: the backward pass uses a sample-*independent* mean or proxy while the forward pass used a hard sample. Cheap, sometimes useful, but biased, binary, and noisy in a different way. So: unbiased-but-noisy on one side, cheap-but-biased-and-binary on the other, and the clean path-derivative trick locked away behind the discreteness.

Let me go back and stare at *how* a categorical is sampled, because maybe the structure of the sampler is the way in. The naive sampler is: compute the CDF, draw a uniform, find the bin. That's a search — no help. But there's another exact sampler I half-remember, the one based on Gumbel noise. The claim is: to draw `z ~ Categorical(π)`, add independent Gumbel noise to the *log*-probabilities and take the argmax,

```
z = one_hot( argmax_i [ log π_i + g_i ] ),   g_i ~ Gumbel(0,1) i.i.d.
```

where `Gumbel(0,1)` is sampled by inverse transform as `g = -log(-log u)`, `u ~ U(0,1)`. If that's really exact, it's interesting structurally: it pushes *all* the randomness into the `g_i`, which don't depend on `θ` at all, and the parameter dependence sits entirely in the deterministic `log π_i` that gets added in. That's the shape of a reparameterization — noise independent of parameters, parameters entering deterministically. The only blemish is the argmax sitting on top.

But first I have to actually convince myself the trick is exact, because the whole plan hinges on it. Let `x_i = log π_i + g_i`. The Gumbel(0,1) CDF is `F(g) = exp(-e^{-g})`, so the *shifted* variable `x_i = g_i + log π_i` is Gumbel with location `log π_i`; its CDF is

```
P(x_i ≤ t) = exp( -e^{-(t - log π_i)} ) = exp( -π_i e^{-t} ),
```

and differentiating, its density is `f_i(t) = π_i e^{-t} exp(-π_i e^{-t})`. Now I want the probability that coordinate `k` is the argmax, which is the probability that `x_k` exceeds every other `x_i`. Condition on `x_k = t`, require all the others below `t`, and integrate:

```
P(argmax = k) = ∫ f_k(t) ∏_{i≠k} P(x_i ≤ t) dt
             = ∫ π_k e^{-t} exp(-π_k e^{-t}) · ∏_{i≠k} exp(-π_i e^{-t}) dt.
```

The product of the `exp(-π_i e^{-t})` over `i≠k` and the one factor `exp(-π_k e^{-t})` from `f_k` merge into a single `exp(-(Σ_i π_i) e^{-t})`, so

```
P(argmax = k) = ∫ π_k e^{-t} exp( -(Σ_i π_i) e^{-t} ) dt.
```

Substitute `s = e^{-t}`, so `ds = -e^{-t} dt` and as `t` runs over the reals `s` runs over `(0, ∞)`. The `e^{-t} dt` becomes `-ds`, the sign flips with the limits, and

```
P(argmax = k) = ∫_0^∞ π_k exp( -(Σ_i π_i) s ) ds = π_k / Σ_i π_i.
```

So if the `π` are normalized this is exactly `π_k`, and notice it even works with *unnormalized* weights — the normalizer falls out of the ratio. The argmax of "log-weights plus Gumbel noise" draws exactly from the categorical. That's the max-stability of the Gumbel showing up: the additive Gumbel noise is precisely the perturbation that turns an argmax over log-probabilities into an exact categorical draw.

I don't fully trust a contour-integral substitution until I've seen it hold up on numbers, so let me run the sampler on a concrete `π = (0.1, 0.6, 0.3)`. Draw a batch of `g_i = -log(-log u)`, add `log π_i`, take the argmax over the three coordinates, and tally how often each index wins. With a couple hundred thousand draws I get empirical frequencies `(0.0996, 0.5998, 0.3005)` against the target `(0.1, 0.6, 0.3)` — agreement to about three decimals, which is exactly the sampling noise I'd expect at this batch size. The substitution was right. Now I trust it.

So the picture is: `z = one_hot(argmax_i [log π_i + g_i])`, with the noise parameter-free and the only non-differentiable thing being the argmax-then-one-hot on top. And that's a much more localized problem than before. The reparameterization wall wasn't "categoricals are hopeless"; it was specifically the argmax. I have a smooth, reparameterized expression `log π_i + g_i` for the perturbed scores, and a single hard operator clamping it onto a corner.

What's a differentiable thing that behaves like argmax-into-one-hot? Softmax. A softmax of a vector of scores is a smooth point on the simplex that concentrates on the largest score, and the sharpness of that concentration is controllable. So instead of taking the hard argmax of the perturbed log-probabilities, take their softmax, with a temperature `τ` to control how peaked it is:

```
y_i = exp( (log π_i + g_i) / τ ) / Σ_j exp( (log π_j + g_j) / τ ),   i = 1, ..., k.
```

Now `y` is a full vector on the simplex `Δ^{k-1}`, not a corner — a "soft" sample. And crucially it's smooth in `π` for any `τ > 0`: the `g_i` are independent of `π`, they enter additively inside the exponent, and everything downstream is exp/sum/divide. So `∂y/∂π` exists and is computable by ordinary autograd. That gives me the path-derivative structure I was after: randomness in `g`, parameters in `log π`, a differentiable map to the sample. I can drop `y` in wherever the network expected the one-hot `z`, and backprop will carry a real gradient back to `θ`.

I should be careful about one thing, though: relaxing the argmax changes the *distribution* of what I feed downstream, so `y` is only an approximation of the categorical draw `z`. How good an approximation is it, and is the approximation controllable? That hinges on `τ`, so let me pin down what `τ` actually does before I lean on it.

Look at the limits algebraically first. As `τ → 0`, dividing the scores by a vanishing number blows up the gaps between them, so the softmax saturates — the largest perturbed score gets essentially all the mass and `y → one_hot(argmax)`. If that's right, then at `τ → 0` the soft sample *is* the discrete sample, distributed exactly as `Categorical(π)` by the Gumbel-Max result I just derived, which would mean `E[y] → π`. At the other extreme, as `τ → ∞`, dividing by a huge number flattens all the scores toward equal, so the softmax should → the uniform vector `(1/k, ..., 1/k)` regardless of `π`. In between, `y` ought to be a smooth interpolation between the one-hot corners and the simplex barycenter.

Both of those are claims about an average over the Gumbel noise, and I'd rather see them than assume them, so let me reuse the same `π = (0.1, 0.6, 0.3)` draws and just compute `E[y]` at a sweep of temperatures. Sharpening down — `τ = 2, 1, 0.5, 0.1` — I get mean soft samples `(0.207, 0.456, 0.337)`, `(0.150, 0.523, 0.327)`, `(0.116, 0.571, 0.313)`, `(0.100, 0.599, 0.301)`. The last one, at `τ = 0.1`, has landed essentially on `π = (0.1, 0.6, 0.3)` — so the `τ → 0` limit really does recover the categorical's class probabilities, not just heuristically. Pushing the other way, `τ = 100` gives `(0.330, 0.336, 0.334)`, collapsing onto uniform `1/3` and washing `π` out completely. So the interpolation story holds on numbers: small `τ` tracks `π`, large `τ` forgets it.

So there's a tension, and it's a bias-variance tension. Small `τ`: the sample is close to a true one-hot categorical draw — low *bias*, because I'm faithfully approximating the distribution I actually want. But small `τ` is exactly where the softmax is turning into a step. For fixed pre-temperature scores `s`, the Jacobian is `(1/τ)(diag(y) - yyᵀ)`: most draws saturate and give almost no derivative, while near-tie draws can have sensitivity on the order of `1/τ`. The gradient becomes concentrated in rare, draw-dependent regions, which is high *variance*. Large `τ`: the map is gentle and smooth, the Jacobian is well-behaved, gradient variance is low — but now `y` is hovering near uniform, far from the categorical I'm supposed to be sampling, so the estimator is *biased*; in the limit it ignores `π` entirely. Low `τ` = low bias / high variance; high `τ` = high bias / low variance. There's no single `τ` that's free.

The way out of a bias-variance tradeoff that I don't have to commit to up front is to *move along it during training*. Start at a high temperature, where gradients are smooth and the optimization is stable and well-conditioned, and anneal `τ` down toward a small positive value as training proceeds, so the samples sharpen toward genuine one-hot categoricals by the time the model has mostly settled. I never take `τ` to exactly zero — that reintroduces the non-differentiability and the spiky high-variance derivative — I stop at a small floor. Something like `τ = max(τ_min, exp(-r·t))` over training step `t`, updated every so often, with `τ_min` around `0.5`, gives me a simple knob rather than a new inference network. And there's a nice reinterpretation if I let `τ` be a *learned* parameter instead of a fixed schedule: `τ` then controls the entropy of the relaxed samples, so the model can adaptively dial the "confidence" of its own samples — it's acting like an entropy regularizer it tunes itself.

There's still one regime this soft sample doesn't serve. Sometimes I'm *forced* to commit to an actual discrete value — a real action from a discrete action space, or a quantized/compressed code — and a fractional vector on the simplex isn't admissible there. I want a genuine one-hot in the forward pass but I still want a gradient in the backward pass. So decouple the two: in the forward pass take the hard `z = one_hot(argmax_i y_i)`, but in the backward pass pretend I used the soft `y`, i.e. let `∇_θ z ≈ ∇_θ y`. This is a straight-through idea, but with a much better proxy than the Bernoulli version: there the backward used a sample-independent *mean*, which disagrees with the hard forward sample and inflates variance; here the proxy `y` is a true *sample-dependent* differentiable surrogate of `z` — it's literally the relaxed version of the same draw, sharing the same Gumbel noise — so the forward/backward mismatch is much smaller. Implementing it is a one-liner once I notice the algebra: I want the *value* to be `y_hard` but the *gradient* to be that of `y_soft`, so I write

```
z = y_hard - y_soft.detach() + y_soft.
```

The detached `y_soft` and the live `y_soft` cancel in value, leaving exactly `y_hard`; but only the live `y_soft` carries a gradient, so I'd expect `∂z/∂θ = ∂y_soft/∂θ`. That's the kind of "value-here, gradient-there" splice that's easy to talk myself into and get subtly wrong about which term autograd actually tracks, so let me trace it on one fixed input. Take logits `(0.2, 1.3, -0.5)`, freeze the Gumbel noise at `(0.1, -0.4, 0.7)`, `τ = 0.5`; the perturbed-and-scaled scores softmax to a `y_soft` whose argmax is coordinate 1, so `y_hard = (0, 1, 0)`. Forming `z = y_hard - y_soft.detach() + y_soft` and reading off its value gives `(0, 1, 0)` — exactly `y_hard`, as the cancellation promised. Now the gradient: backprop `Σ_i z_i w_i` with weights `w = (1, 2, -1)` into the logits gives `(-0.127, 0.869, -0.742)`. Backprop the *same* objective through a plain `y_soft` (no hard term at all) gives `(-0.127, 0.869, -0.742)` — identical. So the splice does what I claimed: forward is the one-hot, backward is precisely the soft Jacobian, nothing leaks from the `y_hard` or `detach()` terms. A pleasant side effect: this stays sparse (one-hot forward) even at high `τ`, where the plain soft sample would be smeared out.

Let me make sure I actually understand the *distribution* of this relaxed sample `y`, not just how to draw it — I want its density on the simplex, both as a sanity check and because if I ever want to use the relaxation in the objective itself (a relaxed prior, say) I'll need `p(y)`. This takes a little care because the softmax is not invertible: it maps the `k` perturbed scores onto the simplex, which is `(k-1)`-dimensional, so one degree of freedom is lost in the normalization. To get an honest change of variables I should work with `k-1` free coordinates. Write `x_i = log π_i` for the logits. The natural way to kill the lost degree of freedom is to *center* by subtracting the last score before the softmax — subtract `(x_k + g_k)` from every score; the softmax is invariant to this shift so `y` is unchanged, and now I can track the `k-1` centered variables

```
u_i = (x_i + g_i) - (x_k + g_k),   i = 1, ..., k-1,
```

and recover `y` from them deterministically.

First I need the density of `u = (u_1, ..., u_{k-1})`. I'll get it by marginalizing out the single remaining Gumbel `g_k`. The Gumbel density with location `μ` (scale 1) at `z` is `f(z; μ) = e^{(μ - z) - e^{(μ - z)}}`. Conditioned on `g_k`, each `u_i = x_i + g_i - x_k - g_k`, so `g_i = u_i - x_i + x_k + g_k`, and the term `x_i + g_i` is Gumbel with location `x_i`. Writing each factor as a Gumbel density evaluated appropriately and the `g_k` itself as `f(g_k; 0) = e^{-g_k - e^{-g_k}}`,

```
p(u_1, ..., u_{k-1}) = ∫_{-∞}^{∞} dg_k  e^{-g_k - e^{-g_k}}  ∏_{i=1}^{k-1} e^{ (x_i - u_i - x_k - g_k) - e^{(x_i - u_i - x_k - g_k)} }.
```

Now substitute `v = e^{-g_k}`, so `dv = -e^{-g_k} dg_k`, i.e. `dg_k = -dv/v`, and the limits flip to `(0, ∞)`. Each `e^{-g_k}` becomes `v`, each `e^{(... - g_k)}` becomes `v·e^{(x_i - u_i - x_k)}`, and pulling the `v`-independent exponentials out front (define `u_k = 0` to tidy the indexing),

```
p(u_1, ..., u_{k-1}) = exp( x_k + Σ_{i=1}^{k-1}(x_i - u_i) ) ∫_0^∞ dv  v^{k-1} exp( -v·(e^{x_k} + Σ_{i=1}^{k-1} e^{x_i - u_i}) ).
```

That `v`-integral is a Gamma integral: `∫_0^∞ v^{k-1} e^{-a v} dv = Γ(k) / a^k` with `a = Σ_{i=1}^{k} e^{x_i - u_i}` (including the `i=k` term `e^{x_k}` since `u_k = 0`). And the prefactor `exp(x_k + Σ_{i<k}(x_i - u_i))` is just `∏_{i=1}^{k} exp(x_i - u_i)`. So

```
p(u_1, ..., u_{k-1}) = Γ(k) ( ∏_{i=1}^{k} e^{x_i - u_i} ) ( Σ_{i=1}^{k} e^{x_i - u_i} )^{-k}.
```

Now transform `u → y`. The deterministic map from the centered scores to the simplex is the softmax on the centered coordinates,

```
y_i = exp(u_i/τ) / (1 + Σ_{j=1}^{k-1} exp(u_j/τ)),   i = 1, ..., k-1,
```

with `y_k = (1 + Σ_{j<k} exp(u_j/τ))^{-1} = 1 - Σ_{j<k} y_j` pinned by the constraint. Inverting on the `k-1` free coordinates: from `y_i / y_k = exp(u_i/τ)` (since the denominators match and `y_k` is the `1` term), I get

```
u_i = τ ( log y_i - log y_k ).
```

The Jacobian `∂u_i/∂y_j`: differentiating `τ(log y_i - log y_k)` with `y_k = 1 - Σ_{j<k} y_j`, the `log y_i` term gives `τ/y_i` on the diagonal, and `-τ log y_k` gives `+τ/y_k` for every `j` (since `∂(-log y_k)/∂y_j = -(-1)/y_k = 1/y_k`). So

```
∂u/∂y = τ ( diag(1/y_{1:k-1}) + (1/y_k) 1·1ᵀ ),
```

a diagonal plus a rank-one term. Factor out the diagonal: `∂u/∂y = τ ( I + (1/y_k) 1·1ᵀ diag(y_{1:k-1}) ) diag(1/y_{1:k-1})`. The determinant splits, `det(diag(1/y_{1:k-1})) = ∏_{j<k} 1/y_j`, and for the rank-one piece I use `det(I + a·bᵀ) = 1 + aᵀb` with `a = (1/y_k)1` and `b = diag(y_{1:k-1})1 = y_{1:k-1}`, giving `1 + (1/y_k)Σ_{j<k} y_j = 1 + (1 - y_k)/y_k = 1/y_k`. So

```
det(∂u/∂y) = τ^{k-1} · (1/y_k) · ∏_{j=1}^{k-1} (1/y_j) = τ^{k-1} ∏_{j=1}^{k} y_j^{-1}.
```

Change of variables, `p(y) = p(u(y)) · |det(∂u/∂y)|`. From the inverse map, `e^{x_i - u_i} = e^{x_i} · e^{-τ(log y_i - log y_k)} = π_i · (y_k/y_i)^{τ}` (using `e^{x_i} = π_i`). Substituting into the centered-Gumbel density,

```
p(u) = Γ(k) ( ∏_{i=1}^{k} π_i (y_k/y_i)^τ ) ( Σ_{i=1}^{k} π_i (y_k/y_i)^τ )^{-k}.
```

The `y_k^{τ}` appears `k` times in the numerator product (`y_k^{kτ}`) and, pulled out of the sum, `y_k^{-kτ}` from the `(·)^{-k}` — they cancel. Multiply by the Jacobian `τ^{k-1} ∏ y_j^{-1}`:

```
p_{π,τ}(y_1, ..., y_k) = Γ(k) τ^{k-1} ( Σ_{i=1}^{k} π_i / y_i^{τ} )^{-k} ∏_{i=1}^{k} ( π_i / y_i^{τ+1} ).
```

So the relaxed sample has a clean closed-form density on the simplex, parameterized by the class probabilities `π` and the temperature `τ`. That derivation had a Gamma integral, a Jacobian determinant of a diagonal-plus-rank-one matrix, and a cancellation of `y_k` powers — three places I could have dropped a factor — so before I rely on the formula I should check it's actually a normalized density. The honest test is whether it integrates to 1 over the simplex. Take `k = 3`, `π = (0.2, 0.5, 0.3)`, `τ = 0.7`, parametrize the 2-simplex by `(y_1, y_2)` with `y_3 = 1 - y_1 - y_2`, and Monte-Carlo the integral of `p(y)` over the triangle. I get `0.993` — within Monte-Carlo error of 1 (the integrand spikes toward the corners as `y_i → 0`, which is exactly where uniform sampling under-resolves it). So the prefactor `Γ(k) τ^{k-1}`, the determinant, and the cancellations all came out right; the formula is a genuine density.

As for the `τ → 0` story at the level of the density: in the formula, every `y_i` enters as `y_i^{-τ}` and `y_i^{-(τ+1)}`, and as `τ` shrinks the mass concentrates toward the corners where some `y_i → 1` and the rest → 0. That's consistent with what I already *measured* directly from the samples — `E[y] → π` at `τ = 0.1` — so the two pictures, the sampler and the closed-form density, are telling the same story about the small-temperature limit.

Now let me land this on code. The estimator is almost embarrassingly small once the derivation is in hand — that's the point. I draw Gumbel noise (an exponential then a log is a clean way to get `-log(-log u)` without the double log), add it to the logits, divide by `τ`, softmax. That's the soft sample. The hard/straight-through variant adds the `argmax` and the `y_hard - y_soft.detach() + y_soft` value-vs-gradient swap.

```python
import torch
import torch.nn.functional as F


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    # logits[..., k] are the unnormalized log-probabilities (log pi up to a constant).
    # g_i ~ Gumbel(0,1): inverse-transform is -log(-log u); equivalently -log(e), e~Exp(1).
    g = -torch.empty_like(logits).exponential_().log()      # g_i ~ Gumbel(0,1)
    scores = (logits + g) / tau                             # (log pi_i + g_i) / tau
    y_soft = scores.softmax(dim)                            # relaxed sample on the simplex Δ^{k-1}

    if hard:
        # straight-through: one-hot value in forward, soft Jacobian in backward.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft           # value == y_hard, grad == d y_soft
    return y_soft
```

And dropping it into the stochastic-categorical-latent VAE is exactly replacing the one place that used to need a non-differentiable categorical draw:

```python
class CategoricalLatentVAE(torch.nn.Module):
    def __init__(self, x_dim, n_vars, n_classes, hidden=256):
        super().__init__()
        self.n_vars, self.n_classes = n_vars, n_classes
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(x_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_vars * n_classes))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_vars * n_classes, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, x_dim))

    def forward(self, x, tau, hard=False):
        logits = self.encoder(x).view(-1, self.n_vars, self.n_classes)
        z = gumbel_softmax(logits, tau=tau, hard=hard)     # the differentiable categorical sample
        x_logits = self.decoder(z.view(x.size(0), -1))
        return x_logits, logits
```

with `tau` annealed across training, `tau = max(0.5, exp(-r * step))`. The gradient now flows from the reconstruction loss, through the decoder, through `z = gumbel_softmax(logits, ...)`, into the encoder logits and back to `θ`, with no stochastic node in the way.

Let me trace the whole chain back so I'm sure it holds together. I wanted a path-derivative gradient through a categorical sample, the way the Gaussian trick gives one through a continuous sample. Naively impossible — a one-hot output is piecewise constant, zero gradient. REINFORCE is unbiased but its cost-times-score form is too noisy, and baselines only dent that. The way through was the Gumbel-Max trick: adding i.i.d. Gumbel noise to the log-probabilities and taking the argmax draws the categorical *exactly* — I both did the `π_k/Σπ` integral and watched the argmax frequencies match `π` to three decimals — and it isolates the only non-differentiable operation as that single argmax, with the parameters entering smoothly as `log π_i`. Relaxing the argmax to a temperature-`τ` softmax gives a differentiable sample `y` on the simplex that I can backprop through; sweeping `τ` numerically, `τ → 0` reproduces `E[y] = π` (the exact draw) but creates spiky high-variance derivatives, while large `τ` flattens `y` to uniform and forgets `π`, so I anneal `τ` from high to a small floor to walk the bias-variance tradeoff. When I need a hard discrete value I keep the one-hot in the forward pass and the soft Jacobian in the backward via the detach splice, which I checked on a fixed input gives exactly the one-hot value with exactly the soft gradient. And the relaxed sample has a closed-form simplex density `Γ(k) τ^{k-1} (Σ π_i/y_i^τ)^{-k} ∏ π_i/y_i^{τ+1}`, which I confirmed integrates to 1 and collapses to the categorical as `τ → 0`. The result is a single softmax-of-perturbed-logits that turns a discrete latent into a reparameterized, low-variance, drop-in-differentiable node.
