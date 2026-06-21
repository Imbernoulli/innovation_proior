The thing that keeps biting me is the very first decision a deep convolutional network forces on me: after every convolution I put a normalization layer, and I have to commit, once and globally, to *which* normalizer to use across the whole network, by hand. If it is a classifier I reach for batch statistics; if it is a stylization network I reach for per-image statistics; if it is a recurrent cell I reach for per-sample-over-all-channels statistics. So just from how I behave I already know the right normalizer is not a universal constant — it depends on the task, on the architecture, and on the batch size, the place I keep getting burned. The practice is nonetheless to pick one operation, stamp it into every layer, and hope it is close enough. That is wrong on two levels. The shallow one is that it is annoying: every new task or smaller batch is a fresh round of swap-the-normalizer-and-retune. The deep one is that a single fixed choice, made before any data is seen and applied uniformly, is almost certainly leaving accuracy on the table — there is no reason the early conv layers, the deep conv layers, a detection head, and a recurrent block should all want the same normalizer. And the batch-dependent option fails outright in the regime detection and segmentation force on me: batch statistics are estimated from the samples in the batch, so as the per-GPU sample count shrinks the mean and variance become noisy and that noise is injected into every forward pass; at one sample per GPU the method does not even converge.

To attack this I first need to be precise about what the choice even is, because if I write every existing normalizer in one common form the degrees of freedom reveal themselves. Take a layer input as a 4D tensor $h$ of shape $(N,C,H,W)$. Every normalizer does the identical thing to a single pixel — subtract a mean, divide by a standard deviation, then apply a learnable per-channel affine, $\hat h_{ncij} = \gamma\,(h_{ncij}-\mu)/\sqrt{\sigma^2+\epsilon} + \beta$ — and they differ in *only one place*: which set of pixels $I$ they average over to form $\mu$ and $\sigma^2$. Instance normalization pools over $(H,W)$ per sample and channel, $\mu_{in},\sigma^2_{in}\in\mathbb{R}^{N\times C}$; layer normalization pools over $(C,H,W)$ per sample, $\mu_{ln},\sigma^2_{ln}\in\mathbb{R}^{N}$; batch normalization pools over $(N,H,W)$ per channel, $\mu_{bn},\sigma^2_{bn}\in\mathbb{R}^{C}$. Same arithmetic, three index sets, three wildly different statistics. So the "choice of normalizer" is literally the choice of which axes to average over, and the global-hand-choice problem looks almost silly: I am picking one corner of a small discrete set and forbidding myself the others, or any blend, when the right corner is exactly what varies.

I propose Switchable Normalization (SN): instead of committing to one normalizer, compute all three statistics and normalize with a *learned convex combination* of them, with the combination weights trained end-to-end by the same backprop that trains the filters, separately in every layer. The normalized pixel is
$$\hat h_{ncij} = \gamma\,\frac{h_{ncij} - \sum_k w_k\,\mu_k}{\sqrt{\sum_k w'_k\,\sigma^2_k + \epsilon}} + \beta, \qquad k\in\{\text{in},\text{ln},\text{bn}\},$$
with one set of weights $w_k$ for the means and a separate set $w'_k$ for the variances. The central question is how to set those weights. Fixing them by hand — a third each, or half batch half instance — just trades one hand-chosen normalizer for one hand-chosen *blend*, still global, still a compromise that over-mixes somewhere and under-mixes elsewhere. So the weights must be learnable, discovered per layer from the loss. But not every parameterization of a weighted average is safe to drop into gradient descent. I need three things at once: the mean weights to sum to one (so $\sum_k w_k\mu_k$ is genuinely a mean and not, say, twice the activation level), the variance weights to stay a convex combination (so $\sum_k w'_k\sigma^2_k$ is automatically nonnegative and bounded between the smallest and largest of the three variances — the denominator under the square root can never blow up or go negative), and the whole map differentiable so the weights train with no projection step interrupting the gradient. The function that takes unconstrained reals and returns exactly a point on the simplex, smoothly, is the softmax. So I introduce six free control parameters per layer and set
$$w_k = \frac{e^{\lambda_k}}{\sum_{z}e^{\lambda_z}}, \qquad w'_k = \frac{e^{\lambda'_k}}{\sum_{z}e^{\lambda'_z}}, \qquad z\in\{\text{in},\text{ln},\text{bn}\}.$$
The $w_k$ are automatically positive and sum to one, and $\lambda$ is unconstrained so it just rides along in backprop. Six scalars per layer, shared across channels, is negligible next to the convolution filters — $2C$ for the affine plus $6$.

I keep the mean weights and the variance weights *separate* rather than tying them, and that extra triple is paid for deliberately. A batch-estimated mean is an average of $NHW$ numbers; a batch-estimated variance is an average of $NHW$ squared deviations, and a variance estimate is intrinsically noisier — squaring amplifies the spread, and the variance is what sits under the square root, so its noise feeds straight into the scale of every normalized activation. There is no reason the best trade-off for "which mean do I trust" should equal the best trade-off for "which variance do I trust." If a layer's batch variance is too noisy but its batch mean is fine, I want to pull down the batch *variance* weight while leaving the batch *mean* weight up; tying them would forbid exactly that.

The obvious objection is cost: naively this computes three full sets of statistics where each existing method computes one, and tripling the compute would kill it regardless of accuracy. The redemption is that the three statistics are nested averages of the *same* per-instance quantities, so I compute the instance statistics once and derive the rest by averaging, never re-touching the raw $h$. The means are immediate, $\mu_{ln}=\frac1C\sum_c\mu_{in}$ and $\mu_{bn}=\frac1N\sum_n\mu_{in}$. The variances need the law of total variance, because variance is not linear. Grouping the pooled pixels by channel, each channel has mean $\mu_{in}$ and variance $\sigma^2_{in}$; the total variance is the average within-group variance plus the variance of the group means, which gives
$$\sigma^2_{ln} = \frac1C\sum_c\bigl(\sigma^2_{in}+\mu_{in}^2\bigr) - \mu_{ln}^2, \qquad \sigma^2_{bn} = \frac1N\sum_n\bigl(\sigma^2_{in}+\mu_{in}^2\bigr) - \mu_{bn}^2.$$
The quantity $\sigma^2_{in}+\mu_{in}^2$ is the per-channel second raw moment (the average of $h^2$ over $(H,W)$); averaging it over the larger pooling set and subtracting the square of that set's mean recovers its variance exactly, without ever revisiting individual pixels. So one pass over $h$ for $\mu_{in},\sigma^2_{in}$ — the $O(NCHW)$ cost — and everything else is cheap reductions of $N\times C$-sized arrays. The blend costs essentially the same as a single normalizer.

What makes this principled rather than merely convenient is what the blend does in weight space. Reparameterize a filter as direction times length, $v\cdot(w_i^\top x)/\lVert w_i\rVert_2$, and feed a zero-mean unit-variance patch $x$. Then instance normalization becomes $\gamma(w_i^\top x)/\lVert w_i\rVert_2+\beta$ — the free-length case, the filter can be any length. Batch normalization, once its random batch statistics are integrated out of the expected loss, decomposes into that same population-normalized form *plus* an adaptive $\zeta(h)\gamma^2$ regularizer, i.e. weight normalization with a length penalty $\gamma\le v$ — a shorter filter, generalization bought at the price of batch noise. Layer normalization mixes the other channels' norms into the denominator, $\gamma(w_i^\top x)/(\lVert w_i\rVert_2+\sum_{j\ne i}\lVert w_j\rVert_2)+\beta$, loosening the per-channel constraint so it can push $\gamma>v$ — the high-learning-ability end. Combining the three in this picture,
$$\hat h_{sn} = \gamma\,\frac{w_i^\top x}{\lVert w_i\rVert_2 + w_{ln}\sum_{j\ne i}\lVert w_j\rVert_2} + \beta, \qquad \text{subject to } w_{bn}\,\gamma\le v.$$
So $w_{ln}$ slides the layer toward learning ability and $w_{bn}$ sets the strength of the length regularization: SN is a continuous dial between learning ability and generalization, turned per layer by the loss. This re-explains the small-batch behavior at a deeper level than "noise is bad": when the batch is small, batch normalization's $\gamma^2$ regularization is driven by random noise rather than a clean signal, so gradient descent lowers $w_{bn}$ to weaken that erratic regularizer and raises $w_{ln}$ to recover learning ability. At one sample per batch, the batch training statistic over $(N,H,W)$ with $N=1$ coincides with the instance statistic over $(H,W)$, so the batch branch carries nothing distinct and can simply be dropped, leaving the instance/layer switch.

One more thing has to be checked before I trust training $\lambda$ and the filters jointly on the *same* data in one backprop, because that is precisely what goes wrong in architecture search. There the control parameters choose among modules of different *capacity*, so optimizing them on the training data makes them pick the highest-capacity module and overfit, which forces a train/validation split and alternating optimization. Here the control parameters choose among normalizers that all have the same affine, the same parameter count, the same statistics cost — switching gives no extra capacity to memorize with — and two of the three options *add* regularization, as the weight-space reading shows. So minimizing training loss by choosing normalizers tends to improve generalization, not destroy it, and I can optimize everything together in a single stage with no split. The backward pass is the standard normalization gradient plus a softmax-Jacobian term into the control parameters, $\partial w_k/\partial\lambda_z = w_k(\delta_{kz}-w_z)$ chained through $\mu=\sum_k w_k\mu_k$ and $\sigma^2=\sum_k w'_k\sigma^2_k$, all of which autodiff supplies for free. The only test-time subtlety is the batch branch: instance and layer statistics are per-sample and recomputed at test exactly as in training, but the batch statistic needs a frozen population estimate so a prediction never depends on which other images share its batch. I support a running moving average during training, but prefer a post-hoc *batch average* — freeze the network and SN weights, push some training minibatches, and average the per-minibatch batch means and variances; using the final settled network for every sample makes this less biased and faster, more stably converged, than an EMA that drags stale early-training history forward. A couple of variants fall out naturally: Sparse SN, applying an argmax to each layer's control parameters after training so each layer commits to one normalizer; and Group SN, splitting channels into groups with a softmax per group for more switch capacity at the cost of more control parameters. I keep the main method as the channel-shared soft blend.

```python
import torch
import torch.nn as nn


class SwitchNorm2d(nn.Module):
    """Switchable Normalization for [N, C, H, W] feature maps. Drop-in replacement for
    BatchNorm2d. Each layer learns, via softmax importance weights, how much to lean on
    instance (per-sample, per-channel), layer (per-sample, all-channel), and batch
    (per-channel, across-batch) statistics. weight=gamma, bias=beta."""

    def __init__(self, num_features, eps=1e-5, momentum=0.9,
                 using_moving_average=True, using_bn=True, last_gamma=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))   # gamma
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))    # beta
        n = 3 if using_bn else 2
        self.mean_weight = nn.Parameter(torch.ones(n))    # control params for the means
        self.var_weight = nn.Parameter(torch.ones(n))     # control params for the variances
        if using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        self.weight.data.fill_(0 if self.last_gamma else 1)
        self.bias.data.zero_()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        # instance statistics (one pass), then derive layer/batch by law of total variance
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2                       # per-channel 2nd raw moment
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum).add_((1 - self.momentum) * var_bn.data)
                else:                                       # batch-average accumulation
                    self.running_mean.add_(mean_bn.data)
                    # Divide accumulated buffers by the number of minibatches before eval.
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = self.running_mean
                var_bn = self.running_var

        mean_w = torch.softmax(self.mean_weight, 0)         # -> simplex, differentiable
        var_w = torch.softmax(self.var_weight, 0)

        if self.using_bn:
            mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
            var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        else:
            mean = mean_w[0] * mean_in + mean_w[1] * mean_ln
            var = var_w[0] * var_in + var_w[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
```

A minimal drop-in form spells the three statistics out directly with no running buffers, recomputing all of them from `x` each forward and matching the statistics to the `[B,C,H,W]` axes (hence `unbiased=False` for the population-variance denominator $1/|I|$):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNorm(nn.Module):
    """Switchable Normalization, drop-in for BatchNorm2d. CustomNorm(num_features)."""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_features))    # gamma
        self.bias = nn.Parameter(torch.zeros(num_features))     # beta
        self.mean_weight = nn.Parameter(torch.ones(3))          # lambda  (in, ln, bn)
        self.var_weight = nn.Parameter(torch.ones(3))           # lambda' (in, ln, bn)

    def forward(self, x):                                       # x: [B, C, H, W]
        mean_w = F.softmax(self.mean_weight, dim=0)
        var_w = F.softmax(self.var_weight, dim=0)
        mean_in = x.mean(dim=(2, 3), keepdim=True)              # over (H, W)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)           # over (C, H, W)
        var_ln = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)           # over (B, H, W)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
        var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
```
