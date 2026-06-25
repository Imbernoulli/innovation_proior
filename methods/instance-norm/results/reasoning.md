Let me start from the failure, because it is not the usual "needs more data" failure. I have a feed-forward stylizer trained to imitate Gatys' slow per-image optimization. It takes a content image, maybe a noise seed, and produces the image that should keep the content but wear one fixed style. The loss is perceptual: content features from deeper layers, style Gram matrices from shallower layers. The network is fast, but the results lag the optimization target, and the failures have a pattern. A model trained on a small set with early stopping can look better than one trained on thousands of content images, and the most obvious artifacts collect near borders where padding keeps entering the convolutional stack.

That combination is strange. More data making the qualitative result worse is the opposite of the usual story, where a feed-forward net trained on more examples gets closer to its target. So I do not think this is a sample-size problem. It looks more like the model is being asked to learn a function its building blocks express poorly, and extra examples just let it overfit the indirect approximation it has cobbled together. If that is right, the fix is not more data; it is finding the operation the conv/ReLU/pool stack is struggling to approximate and handing it over directly.

So: what operation does the style-transfer objective force, that a stack of local convolutions would have trouble with? Here is the asymmetry I keep coming back to. The output contrast mostly comes from the style image, not the content image. Take a content photograph and a low-contrast copy of the same photograph. The style loss asks both outputs to match the same style image's Gram statistics, so both stylized outputs should land at roughly the same, style-controlled contrast. Which means the generator has to discard the content image's instance-specific contrast — it must not let "this particular input was bright and high-contrast" survive to the output, because that brightness is not part of the desired style.

Discarding a per-image global contrast is a whole-image operation: you need a statistic pooled over all spatial positions of the image, then a division broadcast back to every position. The most primitive version is global contrast normalization,

```text
y_tijk = x_tijk / sum_l sum_m x_tilm,
```

a global reduction followed by a broadcast division. A stack of local convolutions and pointwise nonlinearities can approximate many things, but a clean image-wide reduction is awkward for it — that is plausibly the nuisance transformation the network is spending capacity on and overfitting around, and the reason borders (where padding distorts those whole-image sums) are where the seams show.

Before I invent anything, there is already a normalization primitive in these generators: batch normalization. Maybe it already does this. Let me write it carefully and check. For `x` with shape `T x C x W x H`, batch normalization computes, for each channel `i`,

```text
mu_i       = (1/(HWT)) sum_t sum_l sum_m x_tilm
sigma_i^2 = (1/(HWT)) sum_t sum_l sum_m (x_tilm - mu_i)^2
y_tijk    = (x_tijk - mu_i) / sqrt(sigma_i^2 + epsilon).
```

The signs and constants matter: subtract the channel mean, divide by the square root of the channel variance plus epsilon, denominator averaging over `H * W * T` values. The thing to notice is the `T`: the sum runs over the whole minibatch. So the statistic that gets subtracted and divided out of image `t` is not image `t`'s contrast — it is a contrast blended across every image in the batch. Image `t`'s own contrast is only partly removed, in proportion to how much it happens to resemble its batchmates.

Does that "partly" actually matter, or am I splitting hairs? Let me make it concrete with a tiny numeric case I can compute by hand and on the machine. Take one channel, a 2x2 spatial grid, and two images that are the same content at different contrast: image 0 is `[[1,2],[3,4]]`, image 1 is `10*image0 + 5 = [[15,25],[35,45]]`. If contrast removal works, both should normalize to the same pattern. Batch norm pools both images: the pooled mean and variance sit somewhere between the two, so when I subtract and divide I get, for image 0, `[[-0.96,-0.90],[-0.83,-0.77]]`, and for image 1, `[[-0.08,0.55],[1.18,1.81]]`. Those are not the same picture at all — the largest disagreement between the two outputs is about 2.58. The low-contrast image stays squashed near the bottom of the range and the high-contrast image spreads out. Batch norm leaks the content's contrast straight through to the output. That is exactly the thing the style objective said should not survive, so batch norm is not the operation I want.

It also has a second problem that is independent of the pooling. Batch norm runs two different computations: at training time it uses the minibatch moments, and at inference it substitutes fixed population estimates accumulated during training, because minibatch statistics are noisy and batch-dependent. But at test time I stylize one actual image, and the whole point is to remove *that image's* contrast. Normalizing it by a frozen training-set population does the wrong thing, and it makes train and test behave differently on the one input I care about.

The fix suggested by the hand calculation is mechanical: keep the normalization form, but drop the sum over `t`. For each image `t` and channel `i`, compute the moments over only that image-channel's spatial grid,

```text
mu_ti       = (1/(HW)) sum_l sum_m x_tilm
sigma_ti^2 = (1/(HW)) sum_l sum_m (x_tilm - mu_ti)^2
y_tijk     = (x_tijk - mu_ti) / sqrt(sigma_ti^2 + epsilon).
```

Denominator `HW`, not `HWT`; epsilon still inside the square root; the mean subtracted inside the variance is the same `mu_ti`. Now go back to the two-image test. Image 0 has mean `2.5` and variance `1.25`, so it normalizes to `[[-1.34,-0.45],[0.45,1.34]]`. Image 1 is image 0 under `x -> 10x+5`; centering removes the `+5` and dividing by the standard deviation cancels the `10`, so it should give the same pattern. Computing it: image 1 also comes out `[[-1.34,-0.45],[0.45,1.34]]`, matching image 0 to within floating point (largest disagreement ~5e-6). So this version genuinely makes the output invariant to a global affine rescale of the input — it removes the per-image contrast that batch norm left behind. That is the operation the objective was asking for, and it differs from batch norm by a single axis in the moment computation.

Dropping `t` also disposes of the train/test split for free. Because the moments are computed within a single image, a one-image forward pass already has everything it needs; there is nothing to pool and therefore no reason to keep a separate running-statistics path. The same arithmetic should run at training and at inference.

Now the implementation. I want the per-instance moments, and I would rather reuse the existing, well-tested spatial batch-norm kernel than write a new reduction. There is a trick: fold the batch axis into the channel axis. Reshape `(N, C, H, W)` to `(1, N*C, H, W)` and call spatial batch normalization on that. With a folded batch size of one, the kernel's "batch + spatial" reduction has no batch to sum over, so each of the `N*C` folded channels is normalized over only its own `H x W` positions — which is precisely per-image, per-channel spatial moments. I should not just assert this; let me confirm the fold actually equals the per-instance formula. On a random `(2, 3, 4, 4)` tensor, folding to `(1, 6, 4, 4)`, running batch norm with `training=True` and no running stats, then reshaping back, matches a manual per-instance normalize (mean and variance over dims `H, W` per image-channel) to within `1e-5`. Good — the fold is the formula, not an approximation of it.

The affine has one subtlety in this folded view. The learned scale and shift are length `C`, one per channel, and should apply to the same channel in every image. After folding, there are `N*C` channels, so I repeat the length-`C` weight and bias `N` times before handing them to the kernel. To check I have the repeat right rather than, say, tiling the wrong way: with per-channel `weight=[2,0.5,1]`, `bias=[1,-1,0]`, the folded path with `weight.repeat(N)`, `bias.repeat(N)` matches "normalize per instance, then apply that per-channel affine" exactly. So `repeat(N)` is the correct broadcast.

I want to be sure the module really cannot drift into a running-statistics path, since that drift is the failure mode I was trying to escape. I call the batch-norm functional with `running_mean=None`, `running_var=None`, and `training=True`, always. As a sanity check on why that is the only sensible choice here: calling the same functional with `training=False` and no running stats raises a `RuntimeError` — there is literally no population branch to fall back on. That is the right property. The Torch module makes its `train()` and `eval()` methods no-ops for exactly this reason, so switching the surrounding network to eval mode cannot silently change what this layer computes. Defaults to match the reference: `eps=1e-5`, `momentum=0.0`, affine enabled, affine weight initialized uniformly, bias initialized to zero.

Here is the layer in PyTorch, using the fold trick and always asking the functional for batch statistics:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNormalization(nn.Module):
    """Reference-faithful per-instance, per-channel spatial normalization."""

    def __init__(self, num_features, eps=1e-5, momentum=0.0, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            nn.init.uniform_(self.weight)
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        n, c, h, w = x.shape
        if c != self.num_features:
            raise ValueError(f"expected {self.num_features} channels, got {c}")

        folded = x.contiguous().view(1, n * c, h, w)
        if self.affine:
            weight = self.weight.repeat(n)
            bias = self.bias.repeat(n)
        else:
            weight = None
            bias = None

        y = F.batch_norm(
            folded,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=bias,
            training=True,
            momentum=self.momentum,
            eps=self.eps,
        )
        return y.view_as(x)
```

The architectural move is then to put this layer wherever the generator previously used batch normalization, in every block. The two-image check is the whole argument in miniature: the style objective wants content structure carried through but content contrast thrown away; batch norm carried the contrast through (disagreement 2.58 between the rescaled copies) and changed behavior at test time, while per-instance spatial moments collapsed the rescaled copies to the same output (disagreement ~5e-6) using the identical computation at train and test. The expensive nuisance part of the objective — the whole-image contrast reduction the conv stack was approximating badly and overfitting around — becomes a single built-in layer, reached by one axis change in the normalization statistics.
