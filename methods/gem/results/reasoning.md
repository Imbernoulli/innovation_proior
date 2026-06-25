Let me start from what actually goes wrong, which is the single most consequential line in this whole descriptor. I have a convolutional backbone that hands me a tensor of activations: `K` feature maps, each a `W × H` grid, all non-negative because the last thing the backbone does is a ReLU. Everything downstream — the classifier head, or an inner-product similarity if I'm doing retrieval — only ever sees a fixed-length vector. So somewhere I have to collapse each `W × H` map into one number, turning the `W × H × K` tensor into a `K`-vector. That collapse is the entire image representation as far as the rest of the network is concerned. Whatever it discards is gone forever; whatever it over-weights, the classifier inherits. So the question is narrow and brutal: given one feature map, a bag of `N = W·H` non-negative activations, what single scalar should I emit?

What do people do today? Two rules, and they sit at opposite ends of a spectrum. One is max pooling — Razavian et al. read off the strongest activation per map, `f_k = max_{x ∈ X_k} x`. The appeal is sharp: that one activation corresponds to one receptive-field patch, the most distinctive thing the channel found, so the descriptor inner product ends up implicitly matching the most-distinctive patches across two images. Tolias, Sicre and Jégou push this with integral max-pooling and R-MAC, maxing over a grid of regions and then summing the region vectors to get some locality back. The other rule is the polar opposite: sum or average pooling — Babenko and Lempitsky's SPoC, `f_k = (1/N) Σ_{x ∈ X_k} x`, which uses every location, is smooth, stable, and pairs nicely with whitening afterward.

So let me actually feel where each one hurts, because the gap between them is the whole game. Picture a real feature map: a channel that detects, say, a window pattern. On an image of a building it fires hard on a handful of spatial locations — the windows — and is near zero across the sky, the road, the blur. That's the typical structure: sparse strong responses against a large mostly-dead field. Now run max on it. I keep exactly one number, the single strongest window. Every other window that also fired — corroborating evidence, the bulk of the actual signal — I throw on the floor, and worse, a single noisy peak somewhere now dictates the entire component; one outlier activation and the descriptor lurches. R-MAC mitigates this by maxing over many regions, but now I buy a rigid grid: how many scales, what sizes, what strides, what overlap — hyperparameters I have to hand-set, and the locality is still carved by a fixed geometry, not by the image. Now run average on the same map. The opposite failure: the five strong windows get averaged against the thousand dead locations, so each contributes a per-pixel weight of `1/N` and the discriminative signal gets diluted into the background. The mean is stable precisely because it ignores that the field is sparse — it weighs the informative location identically to a dead one. Kalantidis et al.'s CroW tries to fix exactly this by putting weights on locations and channels before summing, up-weighting the salient regions, but those weights are hand-designed statistics of the activations, not something the network learns, and the thing is still at bottom a re-weighted arithmetic mean — it inherits the mean's character.

So I have two fixed rules at two extremes — one location versus all locations equally — and neither matches the sparsity of the field. The obvious first patch is to blend them. Mixed pooling, `a·max + (1−a)·avg`, a convex combination with a mixing scalar `a`. And maybe make `a` learnable, so the network slides between max and mean. Let me chase that for a second because it's the natural move. With `a` learned I can land anywhere on the segment between the mean and the max of the pool. But stare at what that segment actually is. The output is forced to be a straight-line interpolation of two *summary numbers* — the arithmetic mean of the bag and the max of the bag. It can sit `30%` of the way from the mean to the max, fine. What it cannot do is change *how the locations inside the pool are weighted*.

Let me make sure that's a real limitation and not just a way of talking, because it's the thing the whole design hinges on. Suppose I'm worried it can match anything — take a bag and tune `a` so the blend matches some target value. Then perturb a *non-maximal* strong location and ask whether the blend tracks the change the way a properly graded rule would. Concretely: five strong windows `{0.90, 0.85, 0.80, 0.78, 0.75}` against eleven dead `0.05`'s; the mean is `0.289`, the max `0.90`. Bump the second window from `0.85` to `0.88`. The max doesn't move (it's still `0.90`), and the mean moves only by `0.03/16`. So a blend — *any* `a` — responds to that bump only through the mean's `1/N` channel: with `a = 0.437` (the value that, say, matches a target of `0.556`), the blend output shifts by `0.00106`. The blend literally cannot tell that the location that moved was already a strong, informative one; to it, every non-max location is just another `1/N` contributor to the mean. A rule that weighted the strong locations more heavily would respond *more* to that same bump. That's the concrete sense in which the blend can't reshape within-pool weighting: it sees the pool only through two fixed summaries, the mean (everyone `1/N`) and the max (the single top location). So mixing the *outputs* is the wrong axis. I don't want to interpolate between two fixed summaries — I want to interpolate the *operation*, deforming smoothly from "average everything" to "take the strongest" by reshaping the weights, not by blending two endpoints. Wall on the linear blend.

OK, so what does it mean to reshape the weights smoothly between uniform and winner-take-all? Let me think about what makes the max special and the mean special as members of one family. The mean is `(1/N) Σ x_i`. The max is... harder to write as a sum. But there's an old trick for turning a max into a smooth sum-based quantity: emphasize the large values before you average. If I raise every activation to a power `p > 1` before averaging, the big values get amplified far more than the small ones — `x^p` for `p=3` makes a value of `0.9` into `0.73` but a value of `0.1` into `0.001`, so the strong locations come to dominate the sum. Then to put the result back on the scale of an activation, I undo the power at the end: take the `p`-th root. That gives me

  f_k = ( (1/N) Σ_{x ∈ X_k} x^p )^{1/p}.

I've seen this object before — it's the generalized mean, the power mean. Dollár, Tu, Perona and Belongie, building integral-channel features, noted exactly this: raise each element to the `p`-th power and you compute `(1/n Σ x_i^p)^{1/p}`, and a large `p` approximates the maximum, which is handy for accumulating a region of values into one number. They used it for hand-crafted channels; nobody's put it where my global pooling sits. The question is whether `p` actually interpolates the *operation* the way I'm hoping, and I don't want to take that on faith — let me work it through on a concrete map first, then see what's provable in general.

Back to the same bag I just used against the blend — `N = 16`, five strong window responses `{0.90, 0.85, 0.80, 0.78, 0.75}` and eleven near-dead `0.05`'s, mean `0.2894`, max `0.90`. Sweep `p` through the expression:

```
p =   1   GeM = 0.28937      (= the arithmetic mean, to the digit)
p =   2   GeM = 0.45900
p =   3   GeM = 0.55619
p =   5   GeM = 0.65219
p =  10   GeM = 0.74045
p =  50   GeM = 0.85246
p = 200   GeM = 0.88761
```

So on this bag the value rises monotonically from the mean and crawls toward `0.90`, never overshooting — exactly one dial sliding from average toward max. Two things to confirm against that: that `p=1` is the mean and `p → ∞` is the max are real identities and not an artifact of this bag, and that the monotonicity holds in general.

First, `p = 1` recovers the mean trivially: `((1/N) Σ x_i^1)^{1/1} = (1/N) Σ x_i`. That endpoint is exactly SPoC, and the numeric `0.28937` matching the mean to five digits is just that identity showing up.

Now `p → ∞`. This I should actually prove, because it's the load-bearing claim. Let `m = max_i x_i`, and assume `m > 0` (after ReLU the activations are non-negative, and a feature map that's identically zero pools to zero either way, so I can take `m > 0`). Factor `m` out of every term:

  ( (1/N) Σ x_i^p )^{1/p} = m · ( (1/N) Σ (x_i/m)^p )^{1/p}.

Every ratio `x_i/m` lies in `[0, 1]`. Raise it to the `p`-th power: for any location strictly below the max, `x_i/m < 1`, so `(x_i/m)^p → 0` as `p → ∞`; for the maximizer(s), `(x_i/m)^p = 1`. So the bracketed mean `(1/N) Σ (x_i/m)^p` converges to `(number of maximizers)/N`, call it `c`, a constant in `[1/N, 1]`. And `c^{1/p} → 1` as `p → ∞` for any fixed positive `c` — because `ln(c^{1/p}) = (ln c)/p → 0`. Therefore the whole thing `→ m · 1 = m`.

Let me watch that limit actually happen on the bag, because the convergence looked slow above (`p=200` was still only `0.888`, not `0.900`) and I want to be sure it's converging to `m` and not stalling short. Here there's a single maximizer, so `c = 1/16 = 0.0625`. Tracking the factored form:

```
p =   10   bracket = 0.14207   bracket^(1/p) = 0.82272   m·that = 0.74045
p =   50   bracket = 0.06632   bracket^(1/p) = 0.94718   m·that = 0.85246
p =  200   bracket = 0.06250   bracket^(1/p) = 0.98623   m·that = 0.88761
p = 1000   bracket = 0.06250   bracket^(1/p) = 0.99723   m·that = 0.89751
```

The bracket collapses onto `c = 0.0625` exactly as the proof says, and `c^{1/p}` then climbs toward `1` only as fast as `(ln c)/p` decays — which is why `p=200` is still visibly short of the max. The convergence is genuine but logarithmically slow, and that is itself reassuring for what I'll do next: I never want `p` enormous, so I'll be living in the regime where the value is a meaningful blend, not pinned at the max.

So one expression, `((1/N) Σ x^p)^{1/p}`, has average pooling (`p=1`) and max pooling (`p → ∞`) as its two endpoints. The monotonicity I saw on the bag is also general: by the power-mean inequality, for positive reals if `p < q` then `M_p ≤ M_q`, equality iff all the `x_i` are equal. So `M_p` is increasing in `p`, continuous in `p`, and for `p ≥ 1` it climbs from the arithmetic mean up toward the max — every intermediate degree of selectivity in between (the geometric mean sits at `p=0`, the RMS at `p=2`, but I'm living in `p ≥ 1`, the avg-to-max regime).

And this *is* the within-pool re-weighting the blend couldn't do — let me confirm it by running the exact test that defeated the blend. Back on the bag, bump the second window `0.85 → 0.88` again. The blend moved by `0.00106`. GeM at `p=3` moves from `0.55619` to `0.56069`, a shift of `0.00450` — four times as much, because in the power mean that window doesn't enter with a flat `1/N` weight, it enters with the amplified weight `x^{p−1}` that its strength earns it. So GeM responds to a change in a strong location far more than the blend does, which is precisely the graded sensitivity the sparse field calls for and the linear blend structurally lacks. It uses every location (nothing is zeroed unless `p` is literally infinite), but weights the strong ones progressively harder as `p` rises. The right axis is reshaping the operation, and the power mean is the family that does it.

I still want to see the *mechanism*, not just the endpoints — what raising to `p` does to the field before averaging. On the same bag, look at how much of the post-power mass sits on the five strong locations versus the eleven dead ones, by normalizing `x^p` so it sums to one. At `p=1` the five strong account for `0.31` of the mass (just their `5/16` share — uniform), at `p=3` for `0.99`, at `p=10` for essentially all of it. So raising to `p` really does contract the field onto the salient sites: turn `p` up and the pool increasingly listens to the strong locations and tunes out the dead background; turn it to `1` and it listens to everyone equally. The mechanism matches the math.

And the fixed-rule baselines can't do this part at all: this thing is differentiable in `p`, so I don't have to *guess* the right selectivity — I can make `p` a parameter and let the loss find it. But before I commit to learning it, let me work out both gradients, because (a) I need them to backprop at all, and (b) the gradient with respect to the *activations* is going to tell me something about why this trains better than a fixed mean, and the gradient with respect to `p` is what makes it learnable.

Write `S = (1/N) Σ_x x^p` so that `f = S^{1/p}` (dropping the `k` subscript). Gradient with respect to an input activation `x_i`:

  ∂f/∂x_i = (1/p) · S^{1/p − 1} · ∂S/∂x_i,  and  ∂S/∂x_i = (1/N) · p · x_i^{p−1},

so the `p`'s cancel:

  ∂f/∂x_i = S^{1/p − 1} · (1/N) · x_i^{p−1}.

Now simplify `S^{1/p − 1}`. Since `f = S^{1/p}`, we have `S = f^p`, so `S^{1/p − 1} = (f^p)^{(1−p)/p} = f^{1−p}`. Therefore

  ∂f/∂x_i = (1/N) · f^{1−p} · x_i^{p−1}.

Stare at this — it's more than a backprop formula. The upstream gradient that arrives at `f` gets distributed back to the input locations with weight proportional to `x_i^{p−1}` (the `f^{1−p}/N` is a per-map normalizer, the same for every location in the map). So at `p = 1` the weight is `x_i^0 = 1` for all `i` — gradient spread uniformly across the grid, exactly the average-pool backward. As `p → ∞` the weight `x_i^{p−1}` is overwhelmingly largest at the maximizer and negligible elsewhere — gradient routed almost entirely to the strongest location, the max-pool backward.

Let me read off the in-between behavior on the same bag, since the backward weight is just `x_i^{p−1}` normalized to sum to one. Splitting it between the five strong locations and the eleven dead ones, and pulling out the single strongest location's share:

```
p =  1   strong-5 share = 0.3125   dead-11 = 0.6875   top-1 = 0.0625
p =  3   strong-5 share = 0.9918   dead-11 = 0.0082   top-1 = 0.2403
p = 10   strong-5 share = 1.0000   dead-11 = 0.0000   top-1 = 0.4143
```

At `p=1` the gradient is split by count — each of the 16 locations gets `1/16 = 0.0625`, the eleven dead ones soaking up `0.69` of the learning signal between them, which is the average pool's flaw on the backward side too. By `p=3` essentially all the gradient (`0.99`) goes to the five locations that actually fired, but it's still *spread across all five* — the top location only takes `0.24`, the other four windows share the rest. By `p=10` it's collapsing onto the single strongest (`0.41` on one location). So `p` makes this a *soft argmax router*: the gradient flows preferentially to the salient locations, graded by how strong they are, and at moderate `p` it credits every window that fired rather than starving all but one the way a hard max does. That's a concrete reason to expect this to train better than either endpoint, and it's also an argument for keeping `p` moderate — at `p=3` the five corroborating windows all get learning signal, which is the behavior I want, whereas `p=10` is already discarding four of them on the backward pass.

Now the gradient with respect to `p`, the one that lets the network learn the selectivity. Easiest through the log: `ln f = (1/p) ln S`, so

  ∂(ln f)/∂p = −(1/p²) ln S + (1/p) · (∂S/∂p)/S.

And `∂S/∂p = (1/N) Σ_x x^p ln x`, so `(∂S/∂p)/S = Σ_x x^p ln x / Σ_x x^p` (the `1/N`'s cancel). Multiply by `f` to get `∂f/∂p = f · ∂(ln f)/∂p`:

  ∂f/∂p = f · [ −(1/p²) ln S + (1/p) · Σ x^p ln x / Σ x^p ].

Pull out `f/p²`:

  ∂f/∂p = (f/p²) · [ −ln S + p · Σ x^p ln x / Σ x^p ].

And `−ln S = −ln((1/N) Σ x^p) = ln( N / Σ x^p )`, so

  ∂f/∂p = (f/p²) · ( ln( N / Σ_x x^p ) + p · Σ_x x^p ln x / Σ_x x^p ).

That's a clean expression in the same quantities I already compute in the forward pass (`Σ x^p`, and `Σ x^p ln x`), so if it's right, learning `p` costs almost nothing on top of the forward. But I derived it by hand through a log, which is exactly where a stray factor or sign slips in, so let me check it against a finite difference of `f` on the bag before I trust it. Comparing the formula to `(f(p+h) − f(p−h))/2h` with `h = 1e−6`:

```
p =  1   finite-diff = 0.204541   formula = 0.204541   |diff| = 1.3e-11
p =  3   finite-diff = 0.072782   formula = 0.072782   |diff| = 9.5e-12
p = 10   finite-diff = 0.010143   formula = 0.010143   |diff| = 7.4e-11
```

They agree to eleven digits at all three values, so the derivation is right, the gradient is positive (raising `p` raises `f`, consistent with the monotone sweep above), and it's finite at `p=1` with no singularity lurking — the only singular point of the `(·)^{1/p}` form is `p → 0`, which I'm nowhere near. Autograd will assemble this same quantity for me; the hand-derivation was just to confirm there's nothing pathological and that the cost is the two sums I already have.

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
