Let me start from the thing that actually keeps biting me. I build a deep convolutional network, and after every convolution I put a normalization layer, and the first decision I am forced to make is which normalization to use — and I have to make it once, globally, for the whole network, by hand. If it is a classifier I reach for batch statistics, because that is what the field found works best for recognition. If it is a stylization network I reach for per-image statistics, because that is what works for style transfer. If it is a recurrent cell I reach for per-sample-over-all-channels statistics. So I already know, just from how I behave, that the right normalizer is not a universal constant — it depends on the task, on the architecture, and on something I keep getting burned by, the batch size. And yet the practice is: pick one, stamp it into every layer, hope it is close enough. That bothers me on two levels. The shallow one is that it is annoying — every new task is a fresh round of "which normalizer, re-tune everything." The deep one is that it is almost certainly leaving accuracy on the table, because there is no reason a single fixed choice should be simultaneously optimal for the early conv layers, the deep conv layers, the detection head, and the recurrent part of a network. One global decision, made before I have seen any data, applied uniformly. That is the thing to attack.

So before I try to fix it, let me get precise about what the choice even *is*, because if I can write all the existing normalizers in one common form, the degrees of freedom I have to play with will show themselves. Take the input to a normalization layer as a 4D tensor `h` of shape `(N,C,H,W)` — `N` samples in the batch, `C` channels, `H` by `W` spatial. Every normalizer I know does the identical thing to a single pixel: subtract a mean, divide by a standard deviation, then re-scale and re-shift by a learnable per-channel affine,

  ĥ_{ncij} = γ · (h_{ncij} − μ) / sqrt(σ² + ε) + β.

That `γ`, `β` and the `ε` floor are common to all of them; nobody disagrees about that part. Where they differ — and this is the *only* place they differ — is in which set of pixels `I` they average over to compute `μ` and `σ²`:

  μ = (1/|I|) Σ_{(n,c,i,j)∈I} h_{ncij},   σ² = (1/|I|) Σ_{(n,c,i,j)∈I} (h_{ncij} − μ)².

That is the whole design space, and it is just "pick an index set." Batch normalization pools over the batch and the spatial axes per channel, `I_bn = {(n,i,j)}`, so it has one mean and one variance per channel — `2C` numbers. Instance normalization pools over only the spatial axes, separately per sample and per channel, `I_in = {(i,j)}`, so it has a mean and variance for each of the `N×C` channel-instances — `2NC` numbers. Layer normalization pools over all channels and spatial positions within one sample, `I_ln = {(c,i,j)}`, so it has one mean and variance per sample — `2N` numbers. Same arithmetic, three different index sets, three wildly different statistics. So the "choice of normalizer" is literally the choice of which axes I average over. Stated that way, the global-hand-choice problem looks almost silly: I am picking one corner of a small discrete set and forbidding myself from ever using the others, or any blend.

Why not just blend them, then? That is the obvious first thought, so let me chase it and see where it breaks. Instead of choosing one set of statistics, compute all three — `μ_in,σ²_in`, `μ_ln,σ²_ln`, `μ_bn,σ²_bn` — and use a combination. The cleanest combination is a weighted average of the means and a weighted average of the variances,

  ĥ_{ncij} = γ · (h_{ncij} − Σ_k w_k μ_k) / sqrt(Σ_k w'_k σ²_k + ε) + β,   k ∈ {in, ln, bn},

with one set of weights `w_k` for the means and a separate set `w'_k` for the variances. Now the central question is what those weights should be and how I set them. I could fix them by hand — say a third each, or half batch and half instance — but that just trades one hand-chosen normalizer for one hand-chosen *blend*, and it is still global, still the same compromise in every layer. A fixed blend has to over-mix somewhere and under-mix elsewhere, because the right mixture is exactly the thing that varies across layers and tasks, which is what started this whole complaint. So a static weight is not the answer. The thing I actually want is for the network to *discover* the weights from data, per layer, by the same gradient descent that trains everything else. The weights should be learnable parameters.

The moment I say "learnable weights" I have to be careful, because not every parameterization of a weighted average is safe to drop into gradient descent. I want three properties at once. The weights for the means should sum to one (and likewise for the variances), because Σw_k μ_k is supposed to *be* a mean — if the weights summed to two, I would be subtracting twice the typical activation level and the layer would normalize to the wrong center; and Σw'_k σ²_k is supposed to be a variance, so I want it to stay a genuine convex combination of variances, which is automatically nonnegative and bounded between the smallest and largest of the three — a convex combination of variances can't blow up the denominator or make it negative, which is exactly the stability I need under the square root. So I want each weight in [0,1] and each set to sum to one. And I want the whole thing differentiable, so the weights train by backprop with no projection step interrupting the gradient. If I parameterize the weights directly as free numbers I have to keep re-projecting them onto the simplex after every step, which is a nuisance and not smooth. 

The function that takes unconstrained real numbers and hands back exactly a point on the simplex, smoothly, is the softmax. Introduce three free control parameters `λ_in, λ_ln, λ_bn` and set

  w_k = e^{λ_k} / Σ_{z∈{in,ln,bn}} e^{λ_z},

and the `w_k` are automatically positive, automatically sum to one, and `λ` is unconstrained so it just rides along in backprop with everything else. Do the same with a second triple `λ'_in, λ'_ln, λ'_bn` for the variance weights `w'_k`. That is the parameterization: six scalar control parameters per normalization layer, six importance weights coming out of two softmaxes. Six numbers per layer is nothing next to the convolution filters, and they are shared across all channels of the layer, so the parameter budget is `2C` for the affine plus `6` — negligible. Each layer now carries its own little softmax that decides, from the training signal, how much to lean on batch versus instance versus layer statistics.

Let me pause on the decision to keep the mean weights and the variance weights *separate* rather than tying them, because it would be one fewer triple to tie them and I should know why I am paying for the extra three parameters. The mean and the variance are two different statistics with two different noise levels. A batch-estimated mean is an average of `NHW` numbers; a batch-estimated variance is an average of `NHW` squared deviations, and a variance estimate is intrinsically noisier than a mean estimate — squaring amplifies the spread, and the variance is what sits under the square root in the denominator, so its noise feeds straight into the scale of every normalized activation. There is no reason the network's best trade-off for "which mean do I trust" should equal its best trade-off for "which variance do I trust." If batch variance is too noisy to trust in some layer but the batch mean is fine, I want the freedom to pull down the batch *variance* weight while leaving the batch *mean* weight up. Tying them would forbid that. So the two separate triples are not redundancy; they are the degrees of freedom that let the layer down-weight a noisy variance without throwing away a usable mean.

Now I have to confront the cost, because naively this blend computes three full sets of statistics where each existing method computes one, and if that triples the compute then nobody will use it regardless of accuracy. So I need the combined cost to stay at the `O(NCHW)` of a single normalizer. The redemption is that the three statistics are not independent — they are nested averages of the *same* per-instance quantities. Layer statistics are an average over channels of instance statistics; batch statistics are an average over samples of instance statistics. So compute the instance statistics once, then derive the other two from them by averaging, and never re-touch the raw `h`. The means are immediate:

  μ_in = (1/HW) Σ_{ij} h_{ncij},   μ_ln = (1/C) Σ_c μ_in,   μ_bn = (1/N) Σ_n μ_in.

The variances need a little care, because variance is not linear, so I can't just average the instance variances. Let me actually derive the layer variance from the instance statistics rather than recompute it from `h`. The layer variance pools over all channels and spatial positions of one sample. Group the pixels by channel; within channel `c` they have mean `μ_in` (the instance mean) and variance `σ²_in` (the instance variance). The total variance over the whole pooled set is the average within-group variance plus the variance of the group means — the law of total variance. The average within-group variance is (1/C)Σ_c σ²_in. The variance of the group means is the average of μ_in² minus the square of the overall mean, (1/C)Σ_c μ_in² − μ_ln². Add them:

  σ²_ln = (1/C) Σ_c σ²_in + ( (1/C) Σ_c μ_in² − μ_ln² ) = (1/C) Σ_c (σ²_in + μ_in²) − μ_ln².

Let me sanity-check that this really is the variance and not something I fooled myself into. The quantity (σ²_in + μ_in²) is the per-channel *second raw moment*, the average of h² over (H,W) — because second moment equals variance plus mean squared. Averaging the per-channel second moments over channels gives the second raw moment over the whole (C,H,W) set, and subtracting the square of the overall mean μ_ln gives its variance. Yes — that is exactly the variance, computed from the per-instance second moments without ever revisiting individual pixels. The identical argument over the sample axis gives the batch variance,

  σ²_bn = (1/N) Σ_n (σ²_in + μ_in²) − μ_bn².

So I compute `μ_in, σ²_in` in one pass over `h` — that is the `O(NCHW)` cost — and everything else is cheap reductions of `N×C`-sized arrays. The blend costs essentially the same as a single normalizer. That removes the only practical objection to computing all three.

Let me think about what this learning-to-blend actually does in the cases I cared about, because I want to convince myself it solves the batch-size problem and is not just a fancier knob. The whole reason batch statistics degrade at small batch is that the batch mean and variance are estimated from few samples, so they are noisy, and that noise gets injected into every normalized activation. Instance and layer statistics, by contrast, are computed per sample and don't depend on the batch at all — they have no small-batch noise problem. So at large batch, where batch statistics are clean and carry useful cross-sample information, I would expect the network to *want* a large batch weight; at small batch, where batch statistics are noise, I would expect it to pull the batch weight down and lean on the per-sample statistics instead. And the beautiful thing is I do not have to *tell* it to do that — the weights are trained to minimize the loss, and a noisy batch statistic *raises* the loss, so gradient descent will naturally suppress whichever statistic is hurting. The layer self-tunes to its batch regime. At the extreme of one sample per batch, the batch mean and variance over `(N,H,W)` with `N=1` degenerate to the instance statistics over `(H,W)` — batch and instance coincide — so there is nothing left for batch to add, and the blend degrades gracefully to instance-plus-layer instead of failing the way pure batch normalization does when `N=1`. That is the robustness I was after, and it falls out of the construction rather than being bolted on.

I want to understand this more deeply than "it adapts," though, because right now my justification is operational. Let me try to see *what each normalizer is doing to the network's weights*, in a single common language, so I can say why blending them is principled and not just convenient. There is a way to talk about normalization purely in weight space rather than activation space: reparameterize a filter as its direction times a length, normalize the direction to unit norm and let a scalar carry the length. Concretely, for a filter `w` acting on a patch `x`, write the response as `v · (wᵀx)/‖w‖₂` — the filter direction `w/‖w‖₂` applied to `x`, rescaled by a learnable length `v`. This decouples the length of the weight from its direction. Now let me ask: if I take `x` to be a patch with zero mean and unit variance, what does each of my three normalizers look like in this language?

Take instance normalization first. It standardizes the response of channel `i` by that channel's own mean and standard deviation. With `x` already zero-mean unit-variance, the response `h = wᵢᵀx` has mean zero and standard deviation `‖wᵢ‖₂` (the standard deviation of a linear projection of a unit-variance, zero-mean vector along `wᵢ` is the norm of `wᵢ`). So instance normalization gives

  ĥ_in = γ · (wᵢᵀx) / ‖wᵢ‖₂ + β,

which is exactly the weight-normalized form with length `γ`. The filter direction is normalized to unit norm and rescaled by `γ`, and `γ` is free to be whatever it learns. So in this geometry, instance normalization is the "free length" case: the filter can have any length it wants.

Layer normalization is different because it standardizes using statistics pooled over *all* channels of the sample, not just channel `i`. The variance over the full set of channels involves every filter's contribution. Working it through, the response of channel `i` gets divided not by `‖wᵢ‖₂` alone but by a quantity that mixes in the norms of the other channels' filters,

  ĥ_ln = γ · (wᵢᵀx) / ( ‖wᵢ‖₂ + Σ_{j≠i} ‖wⱼ‖₂ ) + β.

The denominator is larger than instance normalization's, and it couples the channels together. The consequence is that the effective length constraint on `wᵢ` is *looser* — the per-channel filter norm is less tightly controlled, the cross-channel terms absorb some of the normalization, so the layer can effectively push `γ` larger than the WN length `v` if that helps. So layer normalization sits at the high-learning-ability end: the filter is the least constrained.

Batch normalization is the subtle one, because its statistics are random variables — they depend on which samples happened to be in the batch. If I treat the batch mean and variance as random and look at the *expected* loss, integrating over those random statistics, batch normalization decomposes into a part that uses the population statistics — population normalization, which with normalized input is again the weight-normalized form `γ·(wᵢᵀx)/‖wᵢ‖₂ + β` — plus an extra term that scales like `γ²` with a data-dependent coefficient. That extra `γ²` term is exactly an adaptive regularization on `γ`: the randomness of the batch statistics acts like a penalty that grows with `γ²`, pushing `γ` to stay small. In the weight-space picture this means batch normalization is weight normalization *with a regularizer on the length*: it imposes `γ ≤ v`, making the filter shorter than the free instance-normalization case. Shorter filters with a length penalty are exactly the kind of regularization that improves generalization — it keeps filters from coadapting, increases the angle between them. So batch normalization buys generalization at the price of constraining the filter length, and the price is the random-batch noise — which is fine when the batch is large and that noise is small, and harmful when the batch is small and that noise is large.

Now the blend reads beautifully. Combining the three weight-space forms, the switched version comes out as

  ĥ_sn = γ · (wᵢᵀx) / ( ‖wᵢ‖₂ + w_ln Σ_{j≠i} ‖wⱼ‖₂ ) + β,   subject to   w_bn · γ ≤ v.

So `w_ln` controls how much of the loosening cross-channel coupling enters the denominator — how far toward layer-normalization's high learning ability the layer slides — and `w_bn` controls the strength of the length regularization, how far toward batch-normalization's generalization the layer slides. The blend is literally a dial between *learning ability* and *generalization*, and the network turns the dial per layer. And this re-explains the small-batch behavior at a deeper level than "the noise is bad": when the batch is small, batch normalization's `γ²` regularization is being driven by random noise rather than by a clean signal, so the regularization is too strong and too erratic; the network responds by decreasing `w_bn` — weakening that noisy regularization — and increasing `w_ln` — recovering learning ability through the cross-channel coupling. That is not a heuristic; it is the gradient of the loss moving the dial away from a regularizer that has gone bad. I find this genuinely convincing now: the blend is not three arbitrary options stapled together, it is a continuous interpolation along the single axis that actually matters, learning versus generalization.

I should also pin down why I am letting the network optimize these control parameters jointly with the filters on the *same* training data, in one backprop, because that is exactly the thing that goes wrong in architecture search and I do not want to walk into the same trap. In architecture search the control parameters choose among modules of different *capacity*, and if you optimize the architecture parameters and the weights on the same data, the architecture parameters will simply pick the highest-capacity module and overfit — which is why that line of work splits the data into a training set for the weights and a validation set for the architecture, and alternates. Two stages, two data sets. Do I need that here? Let me check what my control parameters actually choose among. They choose among normalizers, and all three normalizers have the *same* affine, the *same* parameter count, the *same* statistics-estimation cost — switching the weights does not give the layer more capacity to memorize with. Worse for overfitting, two of the three normalizers (batch and, through cross-channel coupling, layer) *add* regularization, as I just derived. So there is no high-capacity module for the control parameters to run toward; minimizing training loss by choosing normalizers tends to improve generalization, not destroy it. That means I can safely optimize `λ` and the filters together, on one training set, in a single backprop stage — no train/validation split, no alternating optimization. The whole thing trains like an ordinary network with six extra scalars per layer.

Let me make sure I can actually backprop through it, because if I had to hand-derive gradients this would be fragile, but I also want to understand the gradient so I trust it. Most of it is the standard normalization backward pass. With ĥ = γ h̃ + β and h̃ = (h − μ)/sqrt(σ²+ε), the gradient into the standardized value is ∂L/∂h̃ = ∂L/∂ĥ · γ. The gradient into the shared variance is ∂L/∂σ² = −(1/(2(σ²+ε))) Σ_{ij} (∂L/∂h̃)·h̃, and into the shared mean ∂L/∂μ = −(1/sqrt(σ²+ε)) Σ_{ij} ∂L/∂h̃ — the usual chain rule through standardization. The gradient back to the input `h` then has the direct term ∂L/∂h̃ / sqrt(σ²+ε) plus, for each of the three statistics, a contribution routed through how that statistic depends on `h`, weighted by `w_k` and by the size of the pooling set: the instance pieces carry `w_in/(HW)`, the layer pieces `w_ln/(CHW)` summed over channels, the batch pieces `w_bn/(NHW)` summed over samples — each statistic spreading its gradient back over exactly the pixels it pooled. The affine gradients are the usual ∂L/∂γ = Σ (∂L/∂ĥ)·h̃ and ∂L/∂β = Σ ∂L/∂ĥ.

The only genuinely new gradient is into the control parameters `λ`, and that is where the softmax bites back, so let me do it carefully. The loss sees `λ_in` only through the weights `w_in, w_ln, w_bn`, and through them the combined `μ` and `σ²`. The softmax Jacobian is the standard one: ∂w_k/∂λ_z = w_k(δ_{kz} − w_z), i.e. `w_k(1−w_k)` on the diagonal and `−w_k w_z` off-diagonal. And `μ = Σ_k w_k μ_k`, `σ² = Σ_k w'_k σ²_k`, so ∂μ/∂w_k = μ_k and ∂σ²/∂w'_k = σ²_k. Chaining, the gradient into `λ_in` is the diagonal term `w_in(1−w_in)` times the sensitivity of the loss to the *instance* statistics, minus the off-diagonal terms `w_in w_ln` and `w_in w_bn` times the sensitivities to the layer and batch statistics:

  ∂L/∂λ_in = w_in(1−w_in) Σ_{nc} [ (∂L/∂μ)_{nc} μ_in + (∂L/∂σ²)_{nc} σ²_in ]
           − w_in w_ln Σ_{nc} [ (∂L/∂μ)_{nc} μ_ln + (∂L/∂σ²)_{nc} σ²_ln ]
           − w_in w_bn Σ_{nc} [ (∂L/∂μ)_{nc} μ_bn + (∂L/∂σ²)_{nc} σ²_bn ],

and cyclically for `λ_ln` and `λ_bn`. The structure is exactly the softmax Jacobian: each control parameter is pushed up in proportion to how much *its own* statistic reduces the loss and pulled down in proportion to how much the *competing* statistics do — that is precisely the `w_k(δ−w)` shape made concrete. I can write all of this by hand if a framework lacks autodiff, but in a framework with automatic differentiation I just write the forward pass and get every one of these gradients for free, including the softmax competition between normalizers. Good — the method is trainable end to end with no special machinery.

There is one place where I cannot just "use the statistics," and that is test time, specifically the batch part. Instance and layer statistics are per-sample, so at test I compute them exactly as in training — no issue. But the batch statistic depends on having a batch of other samples, which at test I may not have, and even if I do, I do not want the prediction for one image to depend on which other images happen to share its batch. So I need a fixed population estimate of the batch mean and variance to use at test, frozen after training. The reflex from batch normalization is to maintain a running exponential moving average of the batch statistics during training and use that at test. I can do that. But let me question it, because the moving average has a known awkwardness: early in training the batch statistics are garbage (the network is changing fast), and the EMA carries that stale early history forward, so it converges slowly and somewhat unstably to the right population value. There is a cleaner option for a frozen test statistic: after training is done, freeze the network and all the control parameters, push a number of training minibatches through, and simply *average* the per-minibatch batch means and variances that come out — a plain average over fresh, converged-network statistics rather than an EMA accumulated over a moving target. This "batch average" uses the final, settled network for every sample it averages, so it is a less biased estimate of the population statistics, and it converges faster and more stably; a modest number of samples suffices. So I will support both — keep a running average during training for convenience, but prefer post-hoc batch averaging for the cleanest test statistic. Either way, at eval the batch statistic is read from the frozen buffer instead of recomputed from the test batch.

Let me also note a couple of variants this construction naturally suggests, without chasing them. Once a network is trained, I could apply an argmax to each layer's control parameters and keep only the single highest-weighted normalizer per layer — a sparsified version where each layer commits to one operation; finetuning from the soft weights, this is essentially free in accuracy and turns the layer back into a single normalizer at inference. And the weights are currently shared across all channels of a layer; I could instead split the channels into groups and let each group carry its own softmax, increasing the representational power of the switch at the cost of more control parameters. I will keep the main method as the channel-shared soft blend and leave these as extensions.

Now let me write the layer as I would actually ship it, filling the one empty slot — the rule that produces the mean and variance the layer normalizes with. The skeleton already gives me the per-channel affine `weight` (`γ`) and `bias` (`β`), the running buffers for the frozen batch statistic, and the standardize-and-affine tail. Into the slot go: the two triples of control parameters as `mean_weight` and `var_weight`, the single pass that computes the instance statistics, the law-of-total-variance reductions that derive the layer and batch statistics from them, the two softmaxes, and the weighted combination — with the batch branch reading the running buffer at eval and updating it (or being batch-averaged) at train.

```python
import torch
import torch.nn as nn


class SwitchNorm2d(nn.Module):
    """Switchable Normalization for [N, C, H, W] feature maps. A drop-in replacement
    for batch normalization: each layer learns, via softmax importance weights, how much
    to lean on instance- (per-sample, per-channel), layer- (per-sample, all-channel),
    and batch- (per-channel, across the batch) statistics. weight=gamma, bias=beta."""

    def __init__(self, num_features, eps=1e-5, momentum=0.9,
                 using_moving_average=True, using_bn=True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn                                   # at N=1, batch == instance
        # learnable per-channel affine
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))   # gamma
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))    # beta
        # control parameters: softmax over these gives the importance weights.
        # 3 statistics (in, ln, bn) with batch, 2 (in, ln) without.
        n = 3 if using_bn else 2
        self.mean_weight = nn.Parameter(torch.ones(n))    # lambda  for the means
        self.var_weight = nn.Parameter(torch.ones(n))     # lambda' for the variances
        if using_bn:
            # frozen population batch statistic used at eval
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)                              # [N, C, H*W]

        # one pass over the data: the instance statistics, per sample and channel
        mean_in = x.mean(-1, keepdim=True)               # mu_in   over (H, W)
        var_in = x.var(-1, keepdim=True)                 # sigma^2_in over (H, W)

        # layer statistics derived from instance stats (law of total variance)
        mean_ln = mean_in.mean(1, keepdim=True)          # mu_ln = mean over channels of mu_in
        temp = var_in + mean_in ** 2                     # per-channel second raw moment
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2   # sigma^2_ln

        if self.using_bn:
            if self.training:
                # batch statistics, also derived from the instance stats
                mean_bn = mean_in.mean(0, keepdim=True)          # mu_bn over the batch
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2   # sigma^2_bn
                # accumulate the population estimate for eval
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum).add_((1 - self.momentum) * var_bn.data)
                else:                                            # batch-average accumulation
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = self.running_mean                       # frozen population stat
                var_bn = self.running_var

        # importance weights: softmax of the control parameters -> simplex, differentiable
        mean_w = torch.softmax(self.mean_weight, 0)
        var_w = torch.softmax(self.var_weight, 0)

        # weighted combination of the chosen statistics
        if self.using_bn:
            mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
            var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        else:
            mean = mean_w[0] * mean_in + mean_w[1] * mean_ln
            var = var_w[0] * var_in + var_w[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()         # standardize with the blended stats
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias               # per-channel affine
```

Let me retrace the causal chain to be sure it holds. I started stuck because the normalizer is one global, hand-made choice, when in fact the best normalizer depends on task, architecture, and batch size, and even varies layer to layer. Writing every normalizer in one common form showed they differ only in which axes they pool over, so the choice is just picking an index set — which made a *blend* the obvious move. A fixed blend is still a global compromise, so I made the blend weights learnable, parameterized through softmax so they stay a differentiable convex combination — keeping the mean and variance weights separate because the two moments have different noise. The naive blend tripled the compute, but the law of total variance lets me compute the instance statistics once and derive the layer and batch statistics from them, restoring `O(NCHW)`. Reading the three normalizers in weight space showed the blend is a continuous dial between learning ability (instance/layer) and generalization (the length regularization batch normalization carries), which also explains why the network lowers the batch weight when the batch is small and its statistics turn noisy. Because the switched options do not differ in capacity, I can train the control parameters and the filters jointly on one training set without the overfitting that forces architecture search into a two-stage split. The backward pass is the standard normalization gradient plus a softmax-Jacobian gradient into the control parameters, all available from autodiff. At test the per-sample statistics are recomputed but the batch statistic is read from a frozen population estimate — preferably a post-hoc batch average rather than a moving average, for a faster, more stable, less biased estimate. The result drops into the existing harness as a normalization layer with six extra scalars, the same compute as a single normalizer, and a per-layer switch the network learns for itself.
