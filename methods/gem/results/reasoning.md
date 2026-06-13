Let me start from what actually goes wrong, which is the single most consequential line in this whole descriptor. I have a convolutional backbone that hands me a tensor of activations: `K` feature maps, each a `W × H` grid, all non-negative because the last thing the backbone does is a ReLU. Everything downstream — the classifier head, or an inner-product similarity if I'm doing retrieval — only ever sees a fixed-length vector. So somewhere I have to collapse each `W × H` map into one number, turning the `W × H × K` tensor into a `K`-vector. That collapse is the entire image representation as far as the rest of the network is concerned. Whatever it discards is gone forever; whatever it over-weights, the classifier inherits. So the question is narrow and brutal: given one feature map, a bag of `N = W·H` non-negative activations, what single scalar should I emit?

What do people do today? Two rules, and they sit at opposite ends of a spectrum. One is max pooling — Razavian et al. read off the strongest activation per map, `f_k = max_{x ∈ X_k} x`. The appeal is sharp: that one activation corresponds to one receptive-field patch, the most distinctive thing the channel found, so the descriptor inner product ends up implicitly matching the most-distinctive patches across two images. Tolias, Sicre and Jégou push this with integral max-pooling and R-MAC, maxing over a grid of regions and then summing the region vectors to get some locality back. The other rule is the polar opposite: sum or average pooling — Babenko and Lempitsky's SPoC, `f_k = (1/N) Σ_{x ∈ X_k} x`, which uses every location, is smooth, stable, and pairs nicely with whitening afterward.

So let me actually feel where each one hurts, because the gap between them is the whole game. Picture a real feature map: a channel that detects, say, a window pattern. On an image of a building it fires hard on a handful of spatial locations — the windows — and is near zero across the sky, the road, the blur. That's the typical structure: sparse strong responses against a large mostly-dead field. Now run max on it. I keep exactly one number, the single strongest window. Every other window that also fired — corroborating evidence, the bulk of the actual signal — I throw on the floor, and worse, a single noisy peak somewhere now dictates the entire component; one outlier activation and the descriptor lurches. R-MAC mitigates this by maxing over many regions, but now I buy a rigid grid: how many scales, what sizes, what strides, what overlap — hyperparameters I have to hand-set, and the locality is still carved by a fixed geometry, not by the image. Now run average on the same map. The opposite failure: the five strong windows get averaged against the thousand dead locations, so each contributes a per-pixel weight of `1/N` and the discriminative signal gets diluted into the background. The mean is stable precisely because it ignores that the field is sparse — it weighs the informative location identically to a dead one. Kalantidis et al.'s CroW tries to fix exactly this by putting weights on locations and channels before summing, up-weighting the salient regions, but those weights are hand-designed statistics of the activations, not something the network learns, and the thing is still at bottom a re-weighted arithmetic mean — it inherits the mean's character.

So I have two fixed rules at two extremes — one location versus all locations equally — and neither matches the sparsity of the field. The obvious first patch is to blend them. Mixed pooling, `a·max + (1−a)·avg`, a convex combination with a mixing scalar `a`. And maybe make `a` learnable, so the network slides between max and mean. Let me chase that for a second because it's the natural move. With `a` learned I can land anywhere on the segment between the mean and the max of the pool. But stare at what that segment actually is. The output is forced to be a straight-line interpolation of two *summary numbers* — the arithmetic mean of the bag and the max of the bag. It can sit `30%` of the way from the mean to the max, fine. What it cannot do is change *how the locations inside the pool are weighted*. The mean weights every location `1/N`; the max weights one location `1` and the rest `0`; a blend of the two outputs is still, internally, "mostly uniform plus a little bit of the single top guy." It can't express "use all the locations, but weight the strong ones much more heavily than the weak ones in a smooth, graded way." That graded re-weighting is exactly what the sparse field calls for, and an additive mixture of two fixed rules structurally can't represent it. So mixing the *outputs* is the wrong axis. I don't want to interpolate between two fixed summaries — I want to interpolate the *operation*, deforming smoothly from "average everything" to "take the strongest" by reshaping the weights, not by blending two endpoints. Wall on the linear blend.

OK, so what does it mean to reshape the weights smoothly between uniform and winner-take-all? Let me think about what makes the max special and the mean special as members of one family. The mean is `(1/N) Σ x_i`. The max is... harder to write as a sum. But there's an old trick for turning a max into a smooth sum-based quantity: emphasize the large values before you average. If I raise every activation to a power `p > 1` before averaging, the big values get amplified far more than the small ones — `x^p` for `p=3` makes a value of `0.9` into `0.73` but a value of `0.1` into `0.001`, so the strong locations come to dominate the sum. Then to put the result back on the scale of an activation, I undo the power at the end: take the `p`-th root. That gives me

  f_k = ( (1/N) Σ_{x ∈ X_k} x^p )^{1/p}.

I've seen this object before — it's the generalized mean, the power mean. Dollár, Tu, Perona and Belongie, building integral-channel features, noted exactly this: raise each element to the `p`-th power and you compute `(1/n Σ x_i^p)^{1/p}`, and a large `p` approximates the maximum, which is handy for accumulating a region of values into one number. They used it for hand-crafted channels; nobody's put it where my global pooling sits. Let me check whether it actually does what I want, because if it does, `p` is a single dial that reshapes the weighting continuously.

First, does `p = 1` recover the mean? Trivially: `((1/N) Σ x_i^1)^{1/1} = (1/N) Σ x_i`. Yes, that endpoint is exactly SPoC.

Now the other endpoint: does `p → ∞` recover the max? This I should actually prove, not assert, because it's the load-bearing claim. Let `m = max_i x_i`, and assume `m > 0` (after ReLU the activations are non-negative, and a feature map that's identically zero pools to zero either way, so I can take `m > 0`). Factor `m` out of every term:

  ( (1/N) Σ x_i^p )^{1/p} = m · ( (1/N) Σ (x_i/m)^p )^{1/p}.

Every ratio `x_i/m` lies in `[0, 1]`. Raise it to the `p`-th power: for any location strictly below the max, `x_i/m < 1`, so `(x_i/m)^p → 0` as `p → ∞`; for the maximizer(s), `(x_i/m)^p = 1`. So the bracketed mean `(1/N) Σ (x_i/m)^p` converges to `(number of maximizers)/N`, call it `c`, a constant in `[1/N, 1]`. And `c^{1/p} → 1` as `p → ∞` for any fixed positive `c` — because `ln(c^{1/p}) = (ln c)/p → 0`. Therefore the whole thing `→ m · 1 = m`. There it is: `p → ∞` gives the max, exactly. So one expression, `((1/N) Σ x^p)^{1/p}`, literally has average pooling (`p=1`) and max pooling (`p → ∞`) as its two endpoints, and a continuum in between.

But "a continuum in between" is only useful if `p` actually controls the weighting the way I argue, monotonically — bigger `p` really should mean more emphasis on the strong locations. Let me pin this down with the power-mean inequality. For positive reals, if `p < q` then `M_p ≤ M_q`, with equality iff all the `x_i` are equal. So `M_p` is increasing in `p`: for `p ≥ 1` the generalized mean is at least the arithmetic mean and climbs monotonically toward the max as `p` grows. That's the precise statement of "larger `p` weights the large values more." And `M_p` is continuous in `p`, so I have a smooth, monotone dial from the mean (`p=1`) up to the max (`p=∞`), passing through every intermediate degree of selectivity — the geometric mean sits at `p=0` and the RMS at `p=2` if I cared, but I'm living in `p ≥ 1`, the avg-to-max regime. This is exactly the graded re-weighting the linear blend can't give me: it uses every location (nothing is zeroed unless `p` is literally infinite), but it weights the strong ones progressively harder as `p` rises. The right axis is reshaping the operation, and the power mean is the family that does it.

Let me sanity-check the intuition against what raising to `p` does to the spatial field itself, because I want to be sure I understand the mechanism and not just the endpoints. Take an off-the-shelf map and look at `x^p` across the grid for `p = 1, 3, 10`. At `p=1` it's the raw, diffuse response. As `p` grows, the field visibly contracts onto the few maximal sites — the response localizes, the weak background gets crushed toward zero, the strong peaks stand out in relief. So `p` is a contrast / selectivity knob on the pre-aggregation field: turn it up, the pool increasingly listens to the salient locations and tunes out the field; turn it to `1`, it listens to everyone equally. That's the qualitative picture that matches the math.

And the fixed-rule baselines can't do this part at all: this thing is differentiable in `p`, so I don't have to *guess* the right selectivity — I can make `p` a parameter and let the loss find it. But before I commit to learning it, let me work out both gradients, because (a) I need them to backprop at all, and (b) the gradient with respect to the *activations* is going to tell me something about why this trains better than a fixed mean, and the gradient with respect to `p` is what makes it learnable.

Write `S = (1/N) Σ_x x^p` so that `f = S^{1/p}` (dropping the `k` subscript). Gradient with respect to an input activation `x_i`:

  ∂f/∂x_i = (1/p) · S^{1/p − 1} · ∂S/∂x_i,  and  ∂S/∂x_i = (1/N) · p · x_i^{p−1},

so the `p`'s cancel:

  ∂f/∂x_i = S^{1/p − 1} · (1/N) · x_i^{p−1}.

Now simplify `S^{1/p − 1}`. Since `f = S^{1/p}`, we have `S = f^p`, so `S^{1/p − 1} = (f^p)^{(1−p)/p} = f^{1−p}`. Therefore

  ∂f/∂x_i = (1/N) · f^{1−p} · x_i^{p−1}.

Stare at this — it's more than a backprop formula. The upstream gradient that arrives at `f` gets distributed back to the input locations with weight proportional to `x_i^{p−1}` (the `f^{1−p}/N` is a per-map normalizer, the same for every location in the map). So at `p = 1` the weight is `x_i^0 = 1` for all `i` — gradient spread uniformly across the grid, exactly the average-pool backward. As `p → ∞` the weight `x_i^{p−1}` is overwhelmingly largest at the maximizer and negligible elsewhere — gradient routed almost entirely to the strongest location, the max-pool backward. In between, `p` makes this a *soft argmax router*: the gradient flows preferentially to the salient locations, graded by how strong they are, without the all-or-nothing brittleness of a hard max (a hard max sends gradient to exactly one location and starves every other window that fired). That's a concrete reason to expect this layer to train better than either endpoint: forward it builds a contrast-enhanced descriptor that uses all the locations, and backward it focuses learning signal on the discriminative ones while still crediting the rest in proportion to their strength.

Now the gradient with respect to `p`, the one that lets the network learn the selectivity. Easiest through the log: `ln f = (1/p) ln S`, so

  ∂(ln f)/∂p = −(1/p²) ln S + (1/p) · (∂S/∂p)/S.

And `∂S/∂p = (1/N) Σ_x x^p ln x`, so `(∂S/∂p)/S = Σ_x x^p ln x / Σ_x x^p` (the `1/N`'s cancel). Multiply by `f` to get `∂f/∂p = f · ∂(ln f)/∂p`:

  ∂f/∂p = f · [ −(1/p²) ln S + (1/p) · Σ x^p ln x / Σ x^p ].

Pull out `f/p²`:

  ∂f/∂p = (f/p²) · [ −ln S + p · Σ x^p ln x / Σ x^p ].

And `−ln S = −ln((1/N) Σ x^p) = ln( N / Σ x^p )`, so

  ∂f/∂p = (f/p²) · ( ln( N / Σ_x x^p ) + p · Σ_x x^p ln x / Σ_x x^p ).

Good — that's a clean, finite expression in the same quantities I already compute in the forward pass (`Σ x^p`, and `Σ x^p ln x`), so learning `p` costs almost nothing on top of the forward. Autograd will assemble exactly this for me; I work it out to confirm there's no singularity lurking and that it's cheap.

So `p` can be learned. Should it be one `p` shared across all `K` feature maps, or one `p_k` per map? Per-map is strictly more expressive — each channel could pick its own selectivity. But more parameters in the pooling layer means a more complex loss surface and more room to overfit, and the per-channel `p_k`'s are coupled to the channels they sit on in a way that's hard to optimize jointly. The pragmatic read is that a single shared scalar `p`, learned, gives most of the benefit while keeping the optimization clean — one extra parameter, one global selectivity dial for the whole descriptor. I'll go with a single shared learned `p`, with per-channel as the available generalization if I ever want it.

Now I have to actually implement this, and the moment I write `x^p` with a fractional or learned `p` I'm going to hit numerical trouble, so let me think it through before it bites me. The activations are non-negative after ReLU — but non-negative includes *zero*, and lots of locations are exactly zero. The forward value `0^p` is fine for `p > 0`, but the derivative with respect to `p` contains `log x`, and the activation-gradient expression contains `x_i^{p−1}`. I want the tensor entering the power to be strictly positive, both for the calculus above and for the numeric path that autograd follows. The fix is a floor: clamp every activation up to a tiny `ε` before raising to the power, `x ← max(x, ε)` with `ε = 1e−6`. That keeps `x^p`, `log x`, and `x^{p−1}` finite for the floored values. `ε = 1e−6` is far below any healthy activation, so it floors the dead locations without perturbing the real responses.

And `p` itself, once it's learnable and being pushed around by SGD, can wander. If it drifts below `1` I leave the regime I want: `p ≥ 1` is exactly the range where the power mean is at least the arithmetic mean and emphasizes large values monotonically (the power-mean inequality), which is the whole point — "weight the salient locations more." For `p < 1` the family bends the other way, toward the geometric mean (`p → 0`) and eventually the *min*, the wrong direction for highlighting strong activations, and `p → 0` is a genuine singularity of the `(·)^{1/p}` form. So I clamp the *parameter* to `p ≥ 1` in the forward pass — keeps the operation in the avg-to-max regime and guards the learned value from sliding into the degenerate side.

What should `p` start at? `p = 1` is just the average — no head start over SPoC, and the gradient that distinguishes locations is flat there. A very large `p` starts essentially at the brittle max. I want to initialize in the contrast-enhanced-but-still-using-all-locations regime, an intermediate selectivity, and `p = 3` is a good such value — strong enough that `x^3` meaningfully emphasizes the peaks (recall how the field localizes by `p=3`), mild enough that it's nowhere near the max collapse. So `p` initialized to `3.0`, learnable, clamped to `≥ 1`.

Now the actual computation, and here I want to reuse what already exists rather than write a custom kernel. The operation is: raise elementwise to `p`, average over the `H × W` grid of each channel (`(1/N) Σ`), then raise to `1/p`. The middle step is exactly a global average pool — the `1/N · Σ` over spatial positions — which the framework already has as a fused, GPU-friendly primitive, and which by construction handles *any* spatial size: `8×8` on CIFAR or `1×1` after a VGG max-pool stack come out the same way. So the implementation is `avg_pool2d(x.clamp(min=ε).pow(p), (x.size(-2), x.size(-1))).pow(1/p)`, then reshape to `[B, C]`. The channel dimension is untouched — input `C` channels go to a `C`-vector, exactly as the descriptor contract demands.

Let me write it as the layer that drops into the pooling slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPool(nn.Module):
    """Generalized-mean (GeM) pooling.

    f_k = ( (1/N) * sum_{x in X_k} x^p )^{1/p}   over the H x W grid of each channel.
    p = 1  -> average pooling (SPoC);  p -> inf -> max pooling (MAC).
    A single shared, learnable p (init 3.0) is one dial from "average everything"
    to "take the strongest", reshaping how locations are weighted.
    """

    def __init__(self):
        super().__init__()
        # one shared selectivity dial, learned by backprop; init in the
        # contrast-enhanced regime, not at the mean (p=1) or the max (p->inf).
        self.p = nn.Parameter(torch.ones(1) * 3.0)
        self.eps = 1e-6                       # positivity floor for x^p and the x^{p-1} backward

    def forward(self, x):                     # x: [B, C, H, W], x >= 0 after ReLU
        p = self.p.clamp(min=1.0)            # stay in the avg..max regime (power-mean increasing for p>=1)
        x = x.clamp(min=self.eps)            # keep the base strictly positive
        # raise to p, average over the whole H x W grid (= (1/N) sum), raise to 1/p
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).view(x.size(0), -1)
```

So the whole thing, end to end: the descriptor is bottlenecked by one spatial collapse, and the two standard collapses — max and average — sit at opposite extremes, one keeping a single location and one weighting all locations equally, with neither matching the sparse-strong-against-dead-field structure of a real feature map; a linear blend of the two can only slide along the segment between two fixed summaries and can't reshape the within-pool weighting; raising activations to a power `p` before averaging and taking the `p`-th root gives the generalized mean, which I show contains average pooling at `p=1` and max pooling at `p → ∞` and is monotone-increasing in `p` by the power-mean inequality, so `p` is a single continuous dial over selectivity; its gradient with respect to the activations, proportional to `x_i^{p−1}`, makes it a soft-argmax router that focuses learning on the salient locations without the hard max's brittleness, and its gradient with respect to `p` is finite and cheap, so I make `p` a shared learnable parameter initialized at `3`, clamp it to `≥ 1` to stay in the avg-to-max regime, floor the activations at `ε = 1e−6` for positivity, and compute it with the canonical global-average-pool primitive so it costs almost nothing and handles any spatial size.
