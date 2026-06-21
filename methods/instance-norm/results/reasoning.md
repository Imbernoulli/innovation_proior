Let me start from the failure, because it is not the usual "needs more data" failure. I have a feed-forward stylizer trained to imitate Gatys' slow per-image optimization. It takes a content image, maybe a noise seed, and produces the image that should keep the content but wear one fixed style. The loss is perceptual: content features from deeper layers, style Gram matrices from shallower layers. The network is fast, but the results lag the optimization target, and the failures have a pattern. A model trained on a small set with early stopping can look better than one trained on thousands of content images, and the most obvious artifacts collect near borders where padding keeps entering the convolutional stack.

That suggests the model is being asked to learn a function its building blocks express poorly. If more examples make the qualitative result worse, maybe the network is not short on examples; maybe the target function contains an operation that a conv/ReLU/pool stack has to approximate indirectly, so training spends capacity on a nuisance transformation and overfits around it.

What operation does style transfer force? The output contrast mostly comes from the style image, not the content image. If I take a content photograph and a low-contrast copy of the same photograph, the stylized outputs should have similar contrast because the style loss asks for the style image's statistics. So the generator must throw away the content image's instance-specific contrast. It cannot let "this content image was bright and high contrast" pass through unchanged, because that is not part of the desired style.

The most primitive version of this is global contrast normalization. For each image `t` and channel `i`, I could divide every spatial position by a spatial sum:

```text
y_tijk = x_tijk / sum_l sum_m x_tilm.
```

That is a global reduction followed by division and broadcast. A stack of local convolutions and pointwise nonlinearities can approximate many things, but this is exactly the sort of whole-image arithmetic I should not make it rediscover from examples if I can provide it directly.

There is already a normalization primitive in these generators: batch normalization. Let me write it carefully. For `x` with shape `T x C x W x H`, batch normalization computes, for each channel `i`,

```text
mu_i       = (1/(HWT)) sum_t sum_l sum_m x_tilm
sigma_i^2 = (1/(HWT)) sum_t sum_l sum_m (x_tilm - mu_i)^2
y_tijk    = (x_tijk - mu_i) / sqrt(sigma_i^2 + epsilon).
```

The signs and constants matter: subtract the channel mean, divide by the square root of the channel variance plus epsilon, and the denominator averages over `H * W * T` values. For convolutional batch norm, the batch and spatial positions all contribute to the same channel statistic.

This is close to what I want, but the axes are wrong. I need to remove contrast from this image. Batch norm removes the contrast of a channel relative to a statistic mixed across all images in the minibatch. The individual image's contrast is only partly removed because its statistic has been blended with the others. At inference, it is even less aligned with the task: the layer uses fixed running statistics from training, so a new image is normalized by the training population, not by its own contrast.

So I should keep the normalization form but change the axes. Do not sum over `t`. For each image `t` and channel `i`, compute the mean and variance over only that channel's spatial grid:

```text
mu_ti       = (1/(HW)) sum_l sum_m x_tilm
sigma_ti^2 = (1/(HW)) sum_l sum_m (x_tilm - mu_ti)^2
y_tijk     = (x_tijk - mu_ti) / sqrt(sigma_ti^2 + epsilon).
```

Now every image-channel slice is centered and scaled by its own spatial statistics. The denominator is `HW`, not `HWT`; epsilon stays inside the square root; the mean subtracted inside the variance is the same `mu_ti`. This is the contrast-removing operation I needed, expressed as a normalization layer rather than as something the generator has to approximate through many convolutions.

This also settles the train/test case. Batch norm has a training-time path and an inference-time path because minibatch statistics are noisy and batch-dependent, so inference substitutes population estimates. Here that distinction is not just unnecessary; it would break the point. At test time I stylize one actual image, and I want the statistics of that image. Since the moments are per instance, the forward pass is deterministic for a single image and does not need any running population estimate. The same computation should run during training and inference.

The architectural move is therefore simple: replace the generator's batch-normalization layers with this per-image, per-channel spatial normalization everywhere. The core equation is just the normalization. A practical module can also support a learned per-channel affine afterward, as batch normalization modules usually do, but that affine is an implementation detail on top of the moment calculation, not a different set of statistics.

I should make the code match the reference rather than accidentally write a generic modern layer. The canonical Torch module does not implement a new kernel. It folds `(N, C, H, W)` into `(1, N*C, H, W)`, applies `SpatialBatchNormalization` to those `N*C` folded channels, and reshapes back. With a batch size of one in the folded tensor, each folded channel is normalized over only its `H x W` spatial positions. If affine parameters are enabled, the length-`C` parameters are repeated `N` times so the same channel affine applies to each image. The defaults are `eps=1e-5`, `momentum=0.0`, affine enabled, affine weight initialized uniformly, and bias initialized to zero. The module's train/eval methods are no-ops, which prevents an inference-time running-statistics branch from changing the computation.

Here is the faithful layer in PyTorch shape, using the same reshape trick and always asking the batch-norm functional for batch statistics:

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

The reason this is the missing operation is now precise. Style transfer wants content structure but not content contrast. The earlier generator has to learn that global contrast removal indirectly. Batch normalization supplies a related operation but pools across examples and changes behavior at inference, so it does not normalize the actual image whose style is being changed. Per-instance spatial moments remove each image's own mean and scale before the generator paints in the fixed style, and the same calculation runs at test time. That reduces the hard nuisance part of the objective to a built-in layer: one axis change in the normalization statistics, applied everywhere in the generator.
