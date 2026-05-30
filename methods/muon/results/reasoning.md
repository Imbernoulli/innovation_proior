OK, let me think this through from scratch. I'm training a large transformer and almost every parameter, by count and by compute, lives in the 2D weight matrices of the attention and MLP blocks. The optimizer I reach for by default is AdamW: per scalar weight it keeps a first moment `m` and a second moment `v` and it steps with `m̂/(√v̂+ε)`, plus a decoupled decay `W ← (1−ηλ)W`. It works, the update magnitude is well-behaved (its RMS sits around 0.2–0.4 no matter the tensor), and the same learning rate transfers across very differently shaped parameters. So why am I unhappy?

Let me strip Adam down to see what it's really doing. Turn off the moving averages — set `β₁=β₂=0`, drop `ε`. Then the update is `g/√(g²) = sign(g)`. That's the bones of it: Adam is sign gradient descent with some smoothing. It normalizes each coordinate of the gradient *in isolation*, to ±1, and the EMAs just soften that. Fine for a vector of unrelated scalars. But my parameter is a `[A,B]` weight matrix, and it is not a bag of `A·B` unrelated scalars — it's a linear operator that maps the layer's input space into its output space. What I care about is how my step changes that operator's *action*, and "normalize each entry to unit size" is completely blind to the matrix's structure: it depends only on the signs of individual entries, not on, say, whether the update is concentrated in one direction of the operator or spread across many. So the question I actually want to answer is: what is the right notion of a *unit-sized step for a matrix parameter*?

Before I go abstract, let me look at the updates I'm actually producing, because there's a concrete fact about them. If I take SGD-momentum or Adam updates for the 2D weights of a transformer and look at their singular spectrum, they're nearly low-rank — a huge condition number. A few singular directions carry almost all the magnitude of the update, and a long tail of small-singular-value directions get almost nothing. But those small directions can still matter for learning; there's no reason the loss only cares about the top few directions of the update. So whatever a "good" matrix update is, I have a hunch it should *not* let a handful of directions eat the whole step. It should equalize the step across directions. Hold that thought.

Now, how do I make "a unit step for a matrix" precise? Here's a framing that turns the choice of optimizer into the choice of one object. Any sensible first-order optimizer can be read as: build a local model of the loss around the current weights and minimize it. To first order the loss changes by `⟨G, ΔW⟩` where `G` is the gradient (or my momentum estimate of it), and I don't trust that linear model far, so I add a penalty that says "don't move too far," measured in some norm:

  minimize over `ΔW` of  `⟨G, ΔW⟩ + (λ/2)·‖ΔW‖²`.

The whole personality of the optimizer is hidden in *which norm* `‖·‖` I pick. Let me actually solve this generic problem once, because it cleanly separates "how far to step" from "in what direction." Split the update into a magnitude and a unit direction: `ΔW = c·T` with `c ≥ 0` and `‖T‖ = 1`. Substituting,

  `⟨G, c T⟩ + (λ/2)c²‖T‖² = c·⟨G,T⟩ + (λ/2)c²`,

since `‖T‖=1`. Now minimize over the direction `T` first: I want to make `⟨G,T⟩` as negative as possible, so `T = −argmax_{‖T‖=1} ⟨G,T⟩`. Call `‖G‖† := max_{‖T‖=1} ⟨G,T⟩` — that's exactly the definition of the **dual norm** of `G`. Plugging back, the magnitude problem is

  `min_{c≥0} [ −c·‖G‖† + (λ/2)c² ]`,

a simple parabola in `c`, minimized at `c = ‖G‖†/λ`. So:

  `ΔW = −(‖G‖†/λ) · argmax_{‖T‖=1} ⟨G,T⟩`.

Clean. The step size is the dual norm of the gradient over the sharpness; the step *direction* is the unit vector (in my chosen norm) most aligned with the gradient. Everything now rides on the norm.

Let me check that Adam itself drops out of this, just to make sure the framing is faithful, and to see *which* norm Adam secretly chose. Take the plain `ℓ∞` norm on the flattened weights, `‖ΔW‖∞ = max|ΔW_ij|`. The direction is `argmax_{‖T‖∞=1} ⟨G,T⟩`: to maximize `Σ G_ij T_ij` subject to every `|T_ij| ≤ 1`, set each `T_ij = sign(G_ij)`. And the dual norm is `max_{‖T‖∞=1}⟨G,T⟩ = Σ|G_ij| = ‖G‖₁`. So steepest descent under `ℓ∞` is `ΔW ∝ −sign(G)` — exactly EMA-off Adam. So Adam is the choice "measure the step by its largest single entry." That's a statement about individual coordinates; it throws away the matrix structure entirely. (And the only reason `ℓ∞` even touches the layer structure is a coincidence — a max of a max is a max, so the `ℓ∞` norm of the flattened weights equals the largest entrywise norm over layers; nothing about that says the *operator* geometry of a weight matrix is being respected.) That confirms the diagnosis and tells me the fix is to *pick a better norm* — one that sees the matrix as an operator.

What norm respects a weight matrix as an operator? It maps vectors in the input space to vectors in the output space, and both of those spaces are (locally) Euclidean — distances there are `ℓ₂`. The natural way to measure an operator between two Euclidean spaces is its **induced `ℓ₂→ℓ₂` operator norm**, i.e. the largest factor by which it can stretch a vector: `‖M‖₂ = max_{x≠0} ‖Mx‖₂/‖x‖₂`, which is the **spectral norm** — the largest singular value. That's the right yardstick: it asks "how much does this update change the operator's action in the worst direction," not "how big is any single entry." So let me redo the steepest-descent solution with `‖·‖` being the spectral norm and see what direction falls out.

I need `argmax_{‖T‖₂=1} ⟨G,T⟩`. Write the SVD `G = Σᵢ σᵢ uᵢ vᵢᵀ = UΣVᵀ`. Then

  `⟨G,T⟩ = trace(GᵀT) = trace(Σᵢ σᵢ vᵢ uᵢᵀ T) = Σᵢ σᵢ (uᵢᵀ T vᵢ)`.

Now `T` has spectral norm 1, which means `‖Tv‖₂ ≤ ‖v‖₂` for any `v`; in particular for unit vectors `uᵢ, vᵢ` we get `uᵢᵀ T vᵢ ≤ ‖uᵢ‖·‖Tvᵢ‖ ≤ 1`. So each term is at most `σᵢ`, and

  `⟨G,T⟩ ≤ Σᵢ σᵢ`.

Can I hit that bound? Take `T = Σᵢ uᵢ vᵢᵀ = UVᵀ`. It has spectral norm 1 (its singular values are all 1). And `uᵢᵀ (Σⱼ uⱼvⱼᵀ) vᵢ = Σⱼ (uᵢᵀuⱼ)(vⱼᵀvᵢ) = 1` because the `u`'s and the `v`'s are each orthonormal. So `⟨G, UVᵀ⟩ = Σᵢ σᵢ`, the bound is attained, and the optimal direction is

  `T* = U Vᵀ`,   with dual norm `‖G‖₂† = Σᵢ σᵢ`, the nuclear norm.

Stare at `T* = U Vᵀ`. That's the SVD of `G` with every singular value **replaced by 1**. I keep the singular *vectors* — the directions the operator acts in — and throw away the singular *values* — the magnitudes. This is the orthogonal **polar factor** of `G`, and it's the matrix version of `sign`: just as scalar `sign` maps every nonzero number to ±1, this maps every singular value to 1. Write it `msign(G) = UVᵀ`.

And this is exactly the equalization I wanted three paragraphs ago. The raw momentum `G = UΣVᵀ` pours most of its magnitude into the top few `σᵢ`; replacing `Σ` by the identity gives every singular direction an equal-size step. The rare low-`σ` directions, which were being starved, now get a full unit of update. So the spectral-norm choice doesn't just *sound* more principled than per-entry sign — it produces precisely the behavior the diagnostic on update spectra was begging for.

Let me sanity-check that `UVᵀ` is a natural object from a second angle, because I want to be sure it's not an artifact of one particular variational problem. Ask instead: what is the closest *semi-orthogonal* matrix to `G`, in Frobenius distance? Among matrices `O` with `OᵀO=I` (or `OOᵀ=I`), minimize `‖O−G‖_F²`. Expand: `‖O−G‖_F² = ‖O‖_F² − 2⟨O,G⟩ + ‖G‖_F²`. For a semi-orthogonal `O`, `‖O‖_F² = trace(OᵀO) = min(A,B)` is fixed, and `‖G‖_F²` is fixed, so minimizing the distance is the same as maximizing `⟨O,G⟩` — the identical argmax I just solved, with the same answer `O = UVᵀ`. So `UVᵀ` is at once the steepest-descent direction under the spectral norm *and* the projection of `G` onto the semi-orthogonal matrices. Two independent characterizations land on the same object; that's reassuring.

There's a third equivalent form worth writing down because it'll matter for computation. `UVᵀ = U Σ⁻¹ Σ Vᵀ`-ish — let me do it properly. `(G Gᵀ)^{-1/2} G = (UΣ²Uᵀ)^{-1/2} UΣVᵀ = UΣ⁻¹Uᵀ · UΣVᵀ = U Vᵀ`. So `msign(G) = (GGᵀ)^{-1/2} G`. Good to know there's an inverse-square-root expression, even if I won't want to compute it that way.

Now, have I seen this update before? Yes — and it's a useful gut-check. Shampoo steps with `W ← W − η L^{-1/4} G R^{-1/4}`, where `L = ΣGGᵀ`, `R = ΣGᵀG` are accumulated left/right preconditioners. If I *disable the accumulation*, setting `L = GGᵀ` and `R = GᵀG` for the current `G`, then `L^{-1/4} G R^{-1/4} = (GGᵀ)^{-1/4} (UΣVᵀ) (GᵀG)^{-1/4} = (UΣ⁻¹ᐟ²Uᵀ)(UΣVᵀ)(VΣ⁻¹ᐟ²Vᵀ) = U Σ^{-1/2} Σ Σ^{-1/2} Vᵀ = U Vᵀ`. Same orthogonalized update. So the thing I derived from the spectral norm is what Shampoo-without-accumulation computes. That's encouraging — but it also flags the problem: Shampoo gets there by forming and inverse-fourth-rooting two preconditioners of size `A×A` and `B×B`, which is `O(A³+B³)` work and `O(A²+B²)` memory per matrix, and the inverse roots want high precision. At billion-parameter scale, every step, that's unaffordable. So I know *what* I want — `UVᵀ` — and I know the obvious routes to it (SVD, or inverse roots) are too slow and too precision-hungry for a GPU inner loop. I need a cheap, low-precision way to compute `UVᵀ`.

What's the cheapest primitive I have on a GPU? Matrix multiplication, in bfloat16. So I want an iteration that uses only matmuls and converges to `UVᵀ`. Here's the lever: `UVᵀ` is `G` with all singular values forced to 1. If I had a matrix function that, applied to `G`, left `U` and `V` alone and pushed every singular value to 1, I'd be done. An *odd* matrix polynomial does exactly the "leaves `U,V` alone" part. Consider `p(X) = X Xᵀ X`. With `X = UΣVᵀ`, `X Xᵀ X = (UΣVᵀ)(VΣUᵀ)(UΣVᵀ) = U Σ³ Vᵀ`. So a polynomial in `X` of this odd form acts on the singular values as a scalar polynomial while leaving the singular vectors fixed. That means if I iterate

  `X_{k+1} = α X_k + β X_k X_kᵀ X_k`

the singular values evolve under the scalar map `φ(σ) = ασ + βσ³`, and `U,V` never move. I just need to choose `α,β` so that `φ` drives every `σ` to 1.

Pin down the cubic. Take `φ(σ) = (3/2)σ − (1/2)σ³`. Its fixed points solve `σ = 1.5σ − 0.5σ³`, i.e. `0.5σ³ = 0.5σ`, i.e. `σ ∈ {0, ±1}`. Check stability: `φ'(σ) = 1.5 − 1.5σ²`, so `φ'(1) = 0` — the fixed point at 1 is *super*attracting. And for `0 < σ < √3`, `φ(σ) > σ` when `σ<1` and `φ(σ)<σ` when `σ>1`, so the iteration squeezes every singular value in `(0,√3)` toward 1. So iterating `X_{k+1} = 1.5 X_k − 0.5 X_k X_kᵀ X_k` sends `X_k → UVᵀ`, using nothing but matmuls. This is a Newton–Schulz iteration — the matrix-function trick from numerical analysis for the sign/polar function — and it never forms an inverse, so unlike the inverse-root route it's well-behaved even when `G` is nearly singular.

But there's a precondition I just used: the map only contracts toward 1 inside the basin `0 < σ < √3`. If any singular value of my starting matrix exceeds `√3`, the cubic blows it up instead. So I must normalize first. The cleanest guarantee: divide by the Frobenius norm, `X₀ = G/‖G‖_F`. Then `σ_max(X₀) = ‖X₀‖₂ ≤ ‖X₀‖_F = 1 < √3`, safely inside the basin, with room to spare. And this rescaling is free of consequences for the *direction*: scaling `G` by a positive constant scales all `σ` equally and doesn't touch `U,V`, so `msign(cG) = msign(G)`. Normalize, then iterate.

Now, how many steps? With the cubic, convergence near `σ=0` is slow: `φ'(0) = 1.5`, so a tiny singular value only grows by a factor ~1.5 per step, and my updates are nearly low-rank — they have *lots* of tiny singular values that I specifically want to lift. The rate at which small `σ` come up is governed by the slope of the polynomial at 0. So I should make that slope as large as I can. That pushes me from a cubic to a **quintic**, which gives me an extra coefficient to play with:

  `X_{k+1} = a X_k + b (X_kX_kᵀ)X_k + c (X_kX_kᵀ)²X_k`,  acting on singular values as `g(σ) = aσ + bσ³ + cσ⁵`.

`g'(0) = a`, so I want `a` large to lift the small singular values fast. If I demanded `g` converge exactly to 1 everywhere on `(0,1]`, that would constrain `a` to be modest. But do I actually need exact orthogonality? No — the point of `UVᵀ` was to *roughly equalize* the singular directions, not to get them all to exactly 1.0. If I let go of exact convergence, I can crank `a` up and accept that `g` no longer has its fixed point pinned at 1 — the singular values end up scattered somewhere around 1 (in practice roughly uniform on `[0.5,1.5]`) rather than all exactly 1. That's a perfectly fine update: directions are equalized to within a factor of ~3, the dominant-direction problem is gone, and empirically the loss doesn't care about the difference between "all `σ=1`" and "all `σ ∈ [0.5,1.5]`." So I tune `(a,b,c)` to maximize the slope at 0 while keeping `g` bounded and roughly flat near 1 over the interval. The values that come out of that tuning are

  `(a, b, c) = (3.4445, −4.7750, 2.0315)`,

with `g'(0) = a ≈ 3.44`, more than double the cubic's slope, so small singular values shoot up in a couple of steps. With these, **5 iterations** are enough — `N=10` gives a more faithful orthogonalization but the loss is no better, so 5 it is, for speed. And because the whole iteration is bounded matmuls (no inverses, no growth beyond the normalization), it runs stably in **bfloat16**; I don't need fp32 the way an inverse-root Newton iteration would. One small implementation note: form `A = X Xᵀ` so the iteration reads `B = bA + cA², X ← aX + BX` — that's the cheap way to assemble the quintic — and if the matrix is tall I transpose it first so `XXᵀ` is the smaller of the two Gram matrices.

So the core algorithm for a 2D weight matrix is now: momentum `M = μ M + G`; orthogonalize `O = NewtonSchulz5(M) ≈ UVᵀ`; step `W ← W − η O`. (In practice I feed the Nesterov look-ahead `μM + G` into the iteration rather than `M`, the usual momentum refinement.)

Which parameters do I actually apply this to? The derivation assumed a weight matrix that's a *hidden operator* on a Euclidean space — that's the attention and MLP weights. The embedding table and the final output head are technically 2D, but they don't act as operators on a hidden space the same way: the embedding is a vocabulary-indexed lookup, the head is a class scorer, and treating their rows as singular directions to equalize isn't the right model — empirically a per-entry method does better there. And the 1D parameters — RMSNorm gains, biases — have no matrix structure at all; "singular directions" is meaningless for a vector. So I'll run this matrix update only on the hidden 2D weights and keep AdamW for the embedding, the head, and all the 1D params. (Convolution weights, if any, I flatten the trailing dims to view as 2D.) A nice side benefit: this matrix update needs only *one* state buffer, the momentum — half the optimizer memory of Adam's two.

I'd like to stop here, but I know from trying to scale this up that two things break, and I have to fix them before it's actually usable on a large model with many tokens.

First problem: weight growth. Vanilla, this update has no weight decay, and when I train a big model for a long time I watch the weight RMS and the layer-output RMS keep climbing, eventually pushing past the range where bfloat16 still has precision. The over-trained regime then suffers — vanilla converges fast early but gets overtaken later. The fix is the same decoupled decay AdamW uses, applied straight to the weights so it doesn't interact with the orthogonalization:

  `W_t = W_{t-1} − η_t (O_t + λ W_{t-1})`,

equivalently `W ← (1−η_tλ)W − η_t O_t`. With this, the weights stop running away and the late-training loss is better than both vanilla and AdamW. (I also have to remember to apply weight decay to the RMSNorm gains — letting those grow blows up the per-layer output RMS — but those are on the AdamW side anyway.)

Second problem, and this one is subtle: the *magnitude* of my orthogonalized update depends on the **shape** of the matrix, which it shouldn't. Let me compute the RMS of `O = UVᵀ` and see. Take orthogonal `U ∈ ℝ^{n×n}`, `V ∈ ℝ^{m×m}` with `n ≥ m ≥ r`, and the update is `O = U_{:, :r} V_{:r, :}`, i.e. `O_{ij} = Σ_{k=1}^r U_{ik} V_{kj}`. Then

  `RMS(O)² = (1/nm) Σ_i Σ_j O_{ij}² = (1/nm) Σ_i Σ_j (Σ_k U_{ik}V_{kj})²`.

Expand the square as `Σ_k Σ_l U_{ik}U_{il} V_{kj}V_{lj}` and sum over `i,j`. The sum over `i` gives `Σ_i U_{ik}U_{il} = δ_{kl}` (columns of `U` orthonormal), which kills the cross terms `k≠l`, leaving

  `RMS(O)² = (1/nm) Σ_k (Σ_i U_{ik}²)(Σ_j V_{kj}²) = (1/nm) Σ_{k=1}^r 1·1 = r/(nm)`.

For the common full-rank case `r = m = min(n,m)`, this is `m/(nm) = 1/n = 1/max(n,m)`. So

  `RMS(O) = √(1/max(A,B))`.

That's the catch: the bigger the matrix's long side, the *smaller* its update RMS. A big dense MLP matrix gets a tiny update — it underfits, wastes capacity. And if I'd carved a parameter into small pieces (say treating each KV head of grouped/latent attention as its own matrix), `max(A,B)` is small, the update RMS is large, and training goes unstable. The orthogonalization equalized directions *within* a matrix but left the *overall* scale shape-dependent across matrices. To cancel exactly the `√(1/max(A,B))` factor, multiply each matrix's update by `√max(A,B)`:

  `O ← √max(A,B) · O`,

and now every matrix, whatever its shape, has update RMS ≈ 1. (Equivalently one can scale by `√(fan_out/fan_in)` — the same correction up to a global constant when matrices share their second dimension; and this is the very `d_in/d_out` factor that shows up if you derive the spectral-norm step from a square-loss bound, where the majorizer is `½·(d_in/d_out)·‖ΔW‖₂²`.)

One more matching step. I'm running AdamW alongside this on the embedding/head/1D params, and I'd love to share a single learning rate and weight decay across both so I don't tune two sets. AdamW's update RMS is empirically about 0.2–0.4. So I'll set my matrix update's RMS to that range too — multiply by `0.2`:

  `W_t = W_{t-1} − η_t ( 0.2 · √max(A,B) · O_t + λ W_{t-1} )`.

A small sweep over target RMS `{0.05, 0.1, 0.2, 0.4, 0.8}` confirms 0.2 and 0.4 are best and roughly tied; I take 0.2. With this, the matrix optimizer and AdamW have comparable update magnitudes, and the learning rate and weight decay tuned for an AdamW run transfer to this optimizer out of the box. (Momentum I set to `μ = 0.95`; tuning it doesn't consistently help.)

Last, the distributed picture, because at scale the optimizer state is sharded. ZeRO-1 splits the optimizer state (master weights, momentum) across the data-parallel group, which is trivial for AdamW because its update is *elementwise* — each shard can do its own. But my update is *not* elementwise: orthogonalizing needs the *whole* matrix to take an SVD-like step. So I can't just run it on a shard. The fix: reduce-scatter the gradient as usual, apply momentum on the local shard, then **gather the sharded momentum across the data-parallel group into the full matrix**, run Newton–Schulz on the full matrix (in bfloat16, so the gather is half the bytes of fp32), and then **keep only the local partition** of the result to apply the update, finally all-gathering the updated params. That adds one bf16 gather over a vanilla ZeRO-1 AdamW, the iteration is only ~5 cheap matmuls, and the whole optimizer is a low single-digit percentage of the forward/backward time — so the extra cost is negligible, and I still pay only one momentum buffer (half AdamW's optimizer memory).

Let me write it down end to end.

```python
import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    # Approximate the polar factor U Vᵀ (matrix sign) of G with matmuls only.
    # Coefficients tuned so g(s)=a s+b s^3+c s^5 has large slope a at 0 (lifts small
    # singular values fast); converges to s≈1 (roughly Uniform(0.5,1.5)), good enough.
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if X.size(0) > X.size(1):                 # transpose so X Xᵀ is the smaller Gram matrix
        X = X.T
    X = X / (X.norm() + eps)                  # Frobenius-normalize: all σ ≤ 1, inside the basin
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A                 # cheap assembly of the quintic
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5):
    momentum.lerp_(grad, 1 - beta)           # SGD-momentum (EMA of the gradient)
    update = grad.lerp_(momentum, beta)      # Nesterov look-ahead: feed μM + G to the iteration
    if update.ndim == 4:                      # conv weight → view as 2D
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)   # O ≈ U Vᵀ
    update *= 0.2 * (max(update.size(-2), update.size(-1)) ** 0.5) # cancel √(1/max(A,B)); RMS≈0.2
    return update


def adam_update(grad, m, v, step, betas, eps):
    m.lerp_(grad, 1 - betas[0])
    v.lerp_(grad.square(), 1 - betas[1])
    m_hat = m / (1 - betas[0] ** step)
    v_hat = v / (1 - betas[1] ** step)
    return m_hat / (v_hat.sqrt() + eps)


class MatrixAwareOptimizer(torch.optim.Optimizer):
    # Matrix rule for 2D hidden weights; AdamW for embeddings, head, and 1D params.
    def __init__(self, param_groups):
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_matrix_rule"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                else:
                    if not state:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = adam_update(p.grad, state["m"], state["v"],
                                         state["step"], group["betas"], group["eps"])
                p.mul_(1 - lr * wd)                       # decoupled weight decay
                p.add_(update.reshape(p.shape), alpha=-lr)
        return loss
```

So the causal chain, start to finish: Adam normalizes each weight entry in isolation and so cannot see that a layer's weight is an operator, and the updates it (and SGD-momentum) make are nearly low-rank, wasting the rare directions. Reading an optimizer as steepest descent under a norm, I pick the spectral norm — the natural operator norm of a weight matrix — and the steepest-descent direction is forced to be `UVᵀ`, the gradient's singular vectors with all singular values flattened to 1, which is exactly the equalization the spectra were asking for. Computing `UVᵀ` exactly (SVD, or Shampoo's inverse roots) is too slow and precision-hungry, so I drive every singular value to 1 with a tuned quintic Newton–Schulz iteration in bfloat16 — Frobenius-normalize into the basin, then five matmul steps with a large slope at zero so the small singular values come up fast. I run this only on the hidden 2D matrices and leave AdamW for the embedding, head, and 1D params. Then to make it scale: decoupled weight decay so the weights don't outgrow bfloat16; a `√max(A,B)` rescale because the orthogonalized update's RMS is otherwise `√(1/max(A,B))` and shape-dependent; and a `0.2` factor so the update RMS matches AdamW's, letting one learning rate and weight decay serve both. Distributed, I gather the sharded momentum into the full matrix, orthogonalize, and keep the local slice — one extra bf16 gather, one momentum buffer, negligible overhead.
