Let me start from what actually hurts. I am training a model across many workers, data-parallel, and every single step each worker computes a stochastic gradient `g_t` on its local minibatch and then has to all-reduce it — sum everyone's gradient into one — before the optimizer can move. That gradient is the whole model laid flat: millions of 32-bit floats, shipped over the network every iteration. And when I profile it, the arithmetic is not what's slow; the synchronization is. The GPUs sit idle waiting on the wire. So the thing I want is brutally simple to state: send far fewer bits per step — ideally by a factor of a hundred or a thousand — and still land on the same model I'd have gotten from the full gradient. The catch is equally simple. The moment I send a lossy version of `g_t`, I've perturbed the descent direction, and a perturbed direction can slow me down, bias where I end up, or stop me converging at all. So I need two things at once: a compressor aggressive enough to matter, and a reason to believe the optimizer still arrives where SGD would.

What do I have to work with? SGD itself, `x_{t+1} = x_t − γ g_t`, with `E[g_t] = ∇f(x_t)`, `f` is `L`-smooth, and the gradient's second moment is bounded, `E‖g_t‖² ≤ σ²`. Under exactly those assumptions plain SGD with `γ ~ 1/√T` gets me `min_t E‖∇f(x_t)‖² = O(1/√T)` on a non-convex objective. That rate is my yardstick. If a compressed method matches that leading term, compression cost me nothing asymptotically — that's the bar I'm chasing.

Now, why should I even believe a tiny slice of the gradient can stand in for the whole thing? Because of a structural fact about these gradients: they're strongly positively skewed. Most coordinates are nearly zero, a few are large. Look at an embedding matrix in a translation model — a minibatch only touches a handful of vocabulary rows, so only those rows get a real gradient and everything else is noise floor. Concretely that means most of the gradient's *energy*, its `‖g‖²`, lives in a small number of coordinates. So if I keep the `k` largest-magnitude coordinates and zero the rest, I keep almost all of `‖g‖²` while sending only `k` numbers (plus their `k` indices). That's the natural compressor: `top_k(g)`. Magnitude is the importance signal, and the skew is what makes magnitude a *good* importance signal. One wrinkle I should respect from the start: different parameter blocks live on wildly different scales — a conv layer's gradient and a big embedding's gradient aren't comparable — so "largest magnitude" only means something *within* a block. If I pooled all parameters and took a single global top-k, the large-scale blocks would eat the whole budget and the small-scale ones would never send anything. So I'll apply the compressor per tensor, layer-wise.

Let me try the obvious thing: replace `g_t` with `top_k(g_t)` and run SGD. And let me immediately ask whether the analysis survives, because the whole field hinges on one distinction — is my compressor unbiased? If `E[C(g)] = g`, then `C(g)` is still a legitimate stochastic gradient of `f`, the entire SGD proof goes through untouched, and the only damage is inflated variance. That's the QSGD world — randomized rounding to discrete levels, built so it's unbiased; you pay a variance factor `κ` and converge `κ` times slower, and there's a bits floor because even at the coarsest level you're shipping order `√d` coordinates. Clean, but limited.

Is `top_k` unbiased? No. It deterministically keeps the largest coordinates and zeros the rest, so `E[top_k(g)] ≠ ∇f(x)`. It is a *biased* compressor. And the second I'm biased, none of the SGD guarantees apply, because `top_k(g)` is not a stochastic gradient of `f` at all — it's a stochastic gradient of `f` with `(d−k)` coordinates amputated, and the amputation isn't mean-zero.

How bad can biased actually be? I shouldn't wave this off, because the cautionary tale is right next door: the sign compressor, `sign(g)`, also biased, one bit per coordinate. Let me see it fail, because if I understand *how* a biased compressor breaks I'll understand what my fix has to protect. Take the one-dimensional problem with a bimodal stochastic gradient: `g = +4` with probability `1/4`, `g = −1` with probability `3/4`. Its mean is `(1/4)(4) + (3/4)(−1) = 1/4 > 0`, so true descent moves left. But `E[sign(g)] = (1/4)(+1) + (3/4)(−1) = −1/2 < 0` — the *opposite sign* of the true gradient. So `x − γ·sign(g)` moves *right* in expectation; the objective increases for any `γ > 0`. The sign threw away the magnitude information — that one rare `+4` should dominate the average, and stripping it to `+1` lets the three `−1`'s win. That's failure mode one: a biased compressor can forget magnitude and reverse the mean direction.

And there's a second, sharper failure. Two dimensions, `f(x) = ε|x₁+x₂| + |x₁−x₂|`, optimum at the origin, `0 < ε < 1`. The subgradient is `sign(x₁+x₂)·ε(1,1) + sign(x₁−x₂)·(1,−1)`. Start at `(1,1)`. As long as `x₁+x₂ > 0`, the sign of the gradient is `±(1,−1)`, so signSGD only ever moves along `(1,−1)` and the quantity `x₁+x₂` is *frozen* at `2`. The iterates are stuck on the line `x₁+x₂ = 2` forever, `f(x_t) ≥ f(x_0)`, for *any* step-size schedule, even with the exact full subgradient. The component the sign keeps discarding is the `ε(1,1)` direction — and that's precisely the direction I need to travel to reach the origin. It's discarded at every step and never makes it into the update. That's failure mode two: a biased compressor can systematically throw away a fixed direction, and if that direction is where the optimum lives, you never get there.

Now I look back at naive `top_k` and I see the *same* disease wearing different clothes. A coordinate whose magnitude is persistently small never enters the top-k, so its gradient signal is never transmitted — that direction is starved exactly the way `ε(1,1)` was starved for the sign. Whatever consistent signal lives in the small coordinates is dropped at every step and never accumulates into a move. Practitioners saw this directly: zeroing out the dropped coordinates damages convergence. So `top_k`, used naively, is not safe, and I now know *why* — it's a biased compressor with a starvation failure mode. Wall.

So how do I get the energy-concentration win of top-k without the starvation? Stare at what starvation actually is. The problem isn't that I send only `k` coordinates this step — that, by itself, would be fine if over time *every* coordinate eventually got sent. The problem is that the `(d−k)` coordinates I don't send are *thrown away* and never come back. A coordinate that's small now is small again next step, and again, so it's dropped forever even though, summed over a hundred steps, its contribution might be large and consistent. The information isn't worthless; it's just below threshold *at each instant*. So don't throw it away — *keep* it. Maintain a local residual vector `e_t`, the running total of everything I've suppressed so far. Each step, before I compress, I add the residual back into the gradient: I compress `p_t = γ g_t + e_t` instead of `γ g_t`. I send `C(p_t)`, I step with it, and I stash the leftover — the part that didn't make the cut — back into the residual: `e_{t+1} = p_t − C(p_t)`. Now a coordinate that's persistently small keeps *accumulating* in `e_t`, and `e_t` grows until that coordinate finally crosses into the top-k and gets sent in one shot. Nothing is forgotten; it's only *delayed*. The residual is the memory of the part of the gradient that's still owed.

Let me write the loop cleanly. Initialize `e_0 = 0`. At each step: form `p_t = γ g_t + e_t` (error correction); compress, `Δ_t = C(p_t)`; update the iterate, `x_{t+1} = x_t − Δ_t`; update the residual, `e_{t+1} = p_t − Δ_t`. Notice I folded the step size `γ` into `p_t` *before* compressing — the gradient enters the memory already scaled by `γ`, and what's communicated is `C(γ g_t + e_t)`. This is error feedback: I feed the compression error back into the next step.

Does this actually fix convergence, or does it just feel right? I need to prove it, and I want the cleanest possible handle. The residual is an update that has been owed but not applied, and updates are subtracted from the iterate, so I should subtract that owed update from the real point. Let `x̃_t = x_t − e_t`. The real iterate `x_t` is lagging behind by the suppressed update mass; `x̃_t` is the point where that delayed update has already been paid. Watch what recursion `x̃_t` satisfies:

  `x̃_{t+1} = x_{t+1} − e_{t+1} = (x_t − Δ_t) − (p_t − Δ_t) = x_t − p_t = x_t − (γ g_t + e_t) = (x_t − e_t) − γ g_t = x̃_t − γ g_t.`

The virtual iterate `x̃_t` does *exact, plain SGD*: `x̃_{t+1} = x̃_t − γ g_t`, no compression in sight. The `Δ_t` and the `C(·)` canceled perfectly. So error feedback isn't approximating SGD heuristically — it's running honest SGD on a shadow sequence `x̃`, and the only thing standing between `x̃_t` and the iterate I actually have, `x_t`, is the residual `e_t`. If I can show `e_t` stays *bounded*, then `x_t ≈ x̃_t`, and because `f` is smooth its gradient doesn't change fast, so `∇f(x_t) ≈ ∇f(x̃_t)`, and the SGD descent on `x̃` carries over to `x`. Error feedback is a *delayed* gradient method. Each step a fraction of the gradient information goes out, and the rest waits in `e_t` instead of being silently lost. On a smooth function a small delay barely matters.

So the linchpin is: keep `e_t` bounded. For that I need to say something quantitative about the compressor, and the right abstraction is to stop talking about top-k specifically and talk about what property of it I'm actually using. What I need is that compressing `p` doesn't lose *too much* of `p`. Say `C` is a `δ`-approximate compressor, for some `δ ∈ (0,1]`, if

  `‖C(x) − x‖² ≤ (1 − δ)‖x‖²`  for all `x`.

`δ = 1` means lossless (`C(x) = x`); smaller `δ` means more is dropped. This is exactly the contraction I want — it bounds the residual energy as a fraction of the input energy. Now I have to check top-k actually satisfies it and pin down its `δ`. Compare top-k to random-k, the operator that keeps a uniformly random size-`k` subset. By definition top-k keeps the `k` *largest* coordinates, so the energy it drops is the smallest possible for any size-`k` keep set; in particular it drops no more than a random size-`k` set: `‖x − top_k(x)‖² ≤ ‖x − rand_k(x)‖²` pointwise. And rand-k I can take the expectation of directly:

  `E_ω ‖x − rand_k(x)‖² = Σ_{i=1}^d x_i² · Pr[i not kept] = Σ_i x_i² · (1 − k/d) = (1 − k/d)‖x‖²,`

because each coordinate is kept with probability `k/d`. So `‖x − top_k(x)‖² ≤ (1 − k/d)‖x‖²`, which is exactly the `δ`-compressor property with `δ = k/d`. Good: top-k is a `(k/d)`-approximate compressor, and keeping a fraction `k/d` of coordinates buys me `δ = k/d`. (For the special case `k = 1`, `δ = 1/d` — the most aggressive sparsifier, one coordinate per step.)

Now bound the residual. From the update, `e_{t+1} = p_t − C(p_t)`, so `‖e_{t+1}‖² = ‖C(p_t) − p_t‖² ≤ (1 − δ)‖p_t‖² = (1 − δ)‖e_t + γ g_t‖²`. There's the recurrence — the residual energy is a contraction of itself plus the freshly injected gradient. I expand `‖e_t + γ g_t‖²` but I can't just expand it as equality because the cross term couples `e_t` and `g_t`; I'll use Young's inequality, `‖a + b‖² ≤ (1 + η)‖a‖² + (1 + 1/η)‖b‖²` for any `η > 0`, which lets me split them at the cost of two constants I get to choose:

  `E‖e_{t+1}‖² ≤ (1 − δ)(1 + η) E‖e_t‖² + (1 − δ)(1 + 1/η) γ² E‖g_t‖² ≤ (1 − δ)(1 + η) E‖e_t‖² + (1 − δ)(1 + 1/η) γ² σ².`

This is a linear recursion `a_{t+1} ≤ r·a_t + c` with `r = (1 − δ)(1 + η)` and `c = (1 − δ)(1 + 1/η)γ²σ²`. If `r < 1`, it's a contraction and unrolling from `e_0 = 0` gives the geometric series `a_t ≤ c/(1 − r)`. I want `r < 1`, so I need `(1 − δ)(1 + η) < 1`, i.e. `η < δ/(1 − δ)`. The natural choice is to take `η` a bit below that ceiling; set `η = δ/(2(1 − δ))`. Then `1 + η = 1 + δ/(2(1−δ)) = (2 − δ)/(2(1 − δ))`, so

  `r = (1 − δ)·(2 − δ)/(2(1 − δ)) = (2 − δ)/2 = 1 − δ/2,`  hence  `1 − r = δ/2.`

And `1 + 1/η = 1 + 2(1 − δ)/δ = (2 − δ)/δ ≤ 2/δ`. So

  `c = (1 − δ)(1 + 1/η) γ² σ² ≤ (1 − δ)(2/δ) γ² σ²,`  and  `E‖e_t‖² ≤ c/(1 − r) ≤ (1 − δ)(2/δ)γ²σ² / (δ/2) = 4(1 − δ)γ²σ²/δ².`

So the residual is bounded for all `t`:

  `E‖e_t‖² ≤ 4(1 − δ)γ²σ² / δ².`

It's `O(γ²)`, it's finite, and at `δ = 1` (lossless) it's zero, exactly as it should be. The memory never blows up; it holds at most an `O(γ²σ²(1−δ)/δ²)` amount of suppressed mass. The `1/δ²` warns me that very aggressive compression (`δ → 0`) lets the residual grow large — but it's still bounded for any fixed `δ`, and crucially it's controlled by `γ²`, which I'll shrink.

Now cash it in. Work on the virtual iterate, which does plain SGD, `x̃_{t+1} = x̃_t − γ g_t`. By `L`-smoothness,

  `E_t[f(x̃_{t+1})] ≤ f(x̃_t) + ⟨∇f(x̃_t), E_t[x̃_{t+1} − x̃_t]⟩ + (L/2) E_t‖x̃_{t+1} − x̃_t‖² = f(x̃_t) − γ⟨∇f(x̃_t), ∇f(x_t)⟩ + (Lγ²/2) E_t‖g_t‖²,`

using `E_t[g_t] = ∇f(x_t)` and `E_t‖g_t‖² ≤ σ²`. The annoying term is `∇f(x̃_t)` — I never see `x̃_t` in the algorithm, only `x_t`. So I trade it for `∇f(x_t)` and pay a smoothness penalty. Write `⟨∇f(x̃_t), ∇f(x_t)⟩ = ‖∇f(x_t)‖² − ⟨∇f(x_t) − ∇f(x̃_t), ∇f(x_t)⟩`, so

  `−γ⟨∇f(x̃_t), ∇f(x_t)⟩ = −γ‖∇f(x_t)‖² + γ⟨∇f(x_t) − ∇f(x̃_t), ∇f(x_t)⟩.`

Use Young's inequality again on the cross term: `⟨a,b⟩ ≤ (ρ/2)‖b‖² + (1/(2ρ))‖a‖²` with `a = ∇f(x_t) − ∇f(x̃_t)`, `b = ∇f(x_t)`, for any `ρ > 0`:

  `γ⟨∇f(x_t) − ∇f(x̃_t), ∇f(x_t)⟩ ≤ (γρ/2)‖∇f(x_t)‖² + (γ/(2ρ))‖∇f(x_t) − ∇f(x̃_t)‖².`

And `L`-smoothness in its Lipschitz-gradient form gives `‖∇f(x_t) − ∇f(x̃_t)‖² ≤ L²‖x_t − x̃_t‖² = L²‖e_t‖²`, since `x_t − x̃_t = e_t` by definition. Putting it together,

  `E_t[f(x̃_{t+1})] ≤ f(x̃_t) − γ(1 − ρ/2)‖∇f(x_t)‖² + (Lγ²σ²/2) + (γL²/(2ρ)) E‖e_t‖².`

Substitute the residual bound `E‖e_t‖² ≤ 4(1−δ)γ²σ²/δ²`:

  `E_t[f(x̃_{t+1})] ≤ f(x̃_t) − γ(1 − ρ/2)‖∇f(x_t)‖² + (Lγ²σ²/2) + (γ³L²σ²/(2ρ))·4(1−δ)/δ².`

Now I have to keep `ρ` all the way through the telescope; otherwise the constants stop checking out. Rearrange, take total expectation, sum `t = 0…T`, telescope `f(x̃_0) − f(x̃_{T+1})` (with `x̃_0 = x_0` since `e_0 = 0`, and `f(x̃_{T+1}) ≥ f^⋆`), and divide by `γ(1−ρ/2)`. For any `0 < ρ < 2`,

  `(1/(T+1)) Σ_{t=0}^T E‖∇f(x_t)‖² ≤ f₀/(γ(1−ρ/2)(T+1)) + Lγσ²/(2−ρ) + 4γ²L²σ²(1−δ)/(ρ(2−ρ)δ²),`

where `f₀ = f(x_0) − f^⋆`. The left side upper-bounds `min_t E‖∇f(x_t)‖²`. If I take the simple fixed choice `ρ = 1`, I get the coarse but clean bound

  `min_t E‖∇f(x_t)‖² ≤ 2f₀/(γ(T+1)) + Lγσ² + 4γ²L²σ²(1−δ)/δ².`

Balance with `γ = 1/√(T+1)`:

  `min_t E‖∇f(x_t)‖² ≤ (2f₀ + Lσ²)/√(T+1) + 4L²σ²(1−δ)/(δ²(T+1)).`

For any fixed `ρ`, the leading term is still `O(1/√(T+1))`, and the compression quality `δ` appears only in the higher-order `O(1/T)` term. If I want the leading constants to approach the plain-SGD constants as well, I can let `ρ` decrease slowly with `T`; for example, `ρ = (T+1)^{-1/4}` keeps the residual penalty higher order while making the first two constants tend to the SGD proof constants. That is the precise "compression for free" statement I can defend: no compression-dependent leading `O(1/√T)` term, with the `δ` penalty delayed into a smaller-order term. The starvation that killed naive top-k is gone — persistent suppressed signal is carried forward rather than discarded, and on a smooth function that delay is cheap.

I want to double check this isn't a fluke of smoothness, because deep losses are non-convex but also locally rough. What if `f` is convex but *non-smooth*? Then I can't say `∇f(x_t) ≈ ∇f(x̃_t)`, so I expect `δ` to bite the leading term. Run the convex argument on `x̃` with a subgradient `∂f(x_t)` and optimum `x^⋆`:

  `E_t‖x̃_{t+1} − x^⋆‖² = ‖x̃_t − x^⋆‖² + γ²E‖g_t‖² − 2γ⟨∂f(x_t), x̃_t − x^⋆⟩.`

Replace `x̃_t` by `x_t` inside the inner product, picking up `2γ⟨∂f(x_t), x_t − x̃_t⟩ = 2γ⟨∂f(x_t), e_t⟩`. After taking expectation, Cauchy-Schwarz gives `E⟨∂f(x_t), e_t⟩ ≤ (E‖∂f(x_t)‖²)^{1/2}(E‖e_t‖²)^{1/2}`. With `‖∂f(x_t)‖ ≤ σ` and `E‖e_t‖² ≤ 4(1−δ)γ²σ²/δ²`, that cross term is at most `4γ²σ²√(1−δ)/δ`. So

  `E_t‖x̃_{t+1} − x^⋆‖² ≤ ‖x̃_t − x^⋆‖² + γ²σ² − 2γ⟨∂f(x_t), x_t − x^⋆⟩ + 4γ²σ²√(1−δ)/δ.`

Telescope, use convexity `⟨∂f(x_t), x_t − x^⋆⟩ ≥ f(x_t) − f^⋆` then Jensen on the average iterate `x̄_T`:

  `E[f(x̄_T)] − f^⋆ ≤ ‖x_0 − x^⋆‖²/(2γ(T+1)) + γσ²(1/2 + 2√(1−δ)/δ).`

Optimizing `γ` gives `σ‖x_0 − x^⋆‖√(1 + 4√(1−δ)/δ)/√(T+1)`. Here `δ` *does* sit in the leading constant — exactly as I anticipated, because without smoothness the gradient at `x̃_t` and `x_t` can differ even when the points are close, so the delay isn't free anymore. That's the honest picture: smooth → compression free in the leading term; non-smooth → compression shows up in the constant but the method still converges at the right `1/√T` order, which the naive biased compressor could not even guarantee. And for `k = 1`, top-1, this is a convergent greedy-coordinate method on non-smooth functions — the first such guarantee I'm aware of.

There's one more thing worth seeing, on generalization, because in over-parameterized least squares the *which* zero-loss solution you reach matters. SGD's iterates always lie in the span of the gradients, and the min-norm point in the solution set — the one SGD's span constraint drives you toward — is the max-margin solution, which is the well-generalizing one. A biased compressor like top-k or sign breaks the span property: the iterate drifts off the gradient span and can land on a worse solution. But error feedback keeps me close. From the virtual-iterate identity, `x_t − e_t = x_0 − Σ_{i=0}^{t-1} γ g_i`, so when `x_0 = 0`, `x_t − e_t` lies *exactly* in the gradient span, and therefore the distance from `x_t` to the span is at most `‖e_t‖`. With the residual bounded by `O(γ√(1−δ)/δ)·max_i‖g_i‖` and `γ` decaying, that distance shrinks as I converge. So error feedback restores not just convergence but the implicit-regularization property — the iterate is always within `‖e_t‖` of where unbiased SGD would be.

Now make it real. The compressor is a drop-in between backprop and the all-reduce, with `compress`, `decompress`, and an `__init__` that fixes the compression ratio and allocates the local memory. The memory is per-parameter, keyed by name, because each tensor has its own residual and its own scale (recall: layer-wise, not global). What goes on the wire is the top-k payload: the `k` retained values and their `k` indices into the flattened tensor — that's it, `2k` numbers, the `100×` saving. What stays local: the residual, and a small context `(numel, shape)` I need to rebuild the full tensor on the receiving end (`shape` to `view` back, `numel` to size the zero buffer). The context is *not* communicated; it's recomputed/known locally.

`compress(g, name)`: first the error correction — if I have a residual for this name, add it, `g ← g + e[name]`; this is the `p_t = g_t + e_t` step (the step size `γ` lives in the optimizer here, so I accumulate the raw gradient + residual and let the optimizer apply the learning rate after decompression — the per-tensor residual is in gradient units, which is the natural fit for this `compress/decompress` interface). Flatten to 1-D. Set `k = max(1, int(numel · ratio))` — the `max(1, ·)` guarantees even a tiny tensor sends at least one coordinate, so nothing is permanently silent. Find the top-k by magnitude: `torch.topk(|flat|, k)` gives me the indices; gather the values at those indices. Now update the residual: it's everything that did *not* get sent, `e ← p − decompress(top_k(p))`. Concretely, decompress the payload by scattering the kept values into a zero buffer, subtract that decompressed flat vector from the corrected flat tensor, reshape to the tensor's original shape, and store under `name`. Return the payload `[values, indices]` and the context `(numel, shape)`.

`decompress([values, indices], (numel, shape))`: allocate a zero vector of length `numel`, scatter the values into it at their indices, and `view` it back to `shape`. The zeros are the dropped coordinates — but they're not lost, they're sitting in the residual waiting for next step.

Let me write it as the code I'd actually ship, filling the three method bodies in the compressor scaffold:

```python
import torch


class Compressor:
    """Top-K sparsification with error feedback (EF-TopK).

    Keeps the k = max(1, int(d * compress_ratio)) largest-magnitude coordinates
    of each gradient tensor; the rest are zeroed but NOT discarded. The suppressed
    part is accumulated in a per-tensor residual and added back before the next
    compression, so persistent suppressed signal is delayed rather than erased.
    This is what turns a biased compressor into one that matches SGD's
    convergence rate."""

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}                       # e[name]: local memory, NOT communicated

    def compress(self, tensor, name):
        # error correction: p_t = g_t + e_t  (feed the suppressed mass back in)
        if name in self.residuals:
            tensor = tensor + self.residuals[name]

        shape = tensor.size()
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        k = max(1, int(numel * self.compress_ratio))   # >=1 so nothing is silenced forever

        # top-k by magnitude: keep the coordinates carrying the most energy
        _, indices = torch.topk(tensor_flat.abs(), k, sorted=False)
        values = torch.gather(tensor_flat, 0, indices)

        # residual = what was NOT sent: e_{t+1} = p_t - C(p_t)
        decompressed_flat = self.decompress([values, indices], (numel, shape)).flatten()
        self.residuals[name] = (tensor_flat - decompressed_flat).view(shape)

        return [values, indices], (numel, shape)       # payload to communicate; local ctx

    def decompress(self, compressed_tensors, ctx):
        values, indices = compressed_tensors
        numel, shape = ctx
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)   # dropped coords are 0 (held in e)
        return tensor_decompressed.view(shape)
```

So I end with a compressor that is as simple as the original greedy sparsifier but no longer forgets its mistakes. The proof object is not the sparse update by itself; it is the pair of the sparse update and the residual. Top-k gives me the contraction parameter `δ = k/d`, the residual recursion keeps the delayed mass bounded, and the virtual iterate identity tells me exactly why the real iterate shadows SGD. In the smooth case, the constant-correct bound is the `ρ`-parameterized one, with the `δ` term strictly higher order for fixed `ρ`; in the non-smooth convex case, `δ` must enter the leading constant because there is no Lipschitz-gradient bridge between `x_t` and `x̃_t`. The code is just that mechanism made local to each tensor: add residual, take absolute top-k, scatter-decompress, and store what was not sent.
