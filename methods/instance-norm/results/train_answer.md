I have a feed-forward generator trained to imitate Gatys' slow per-image style transfer: it takes a content image (and optionally a noise seed) and produces an image that keeps the content layout but wears one fixed style, optimized with a perceptual loss that reads content from deeper VGG features and style from shallower Gram matrices. The network is fast at test time, but its output consistently lags the per-image optimization it is meant to amortize, and the way it fails is diagnostic rather than generic. A model trained on a small set with early stopping can look better than one trained on thousands of content images, and the worst artifacts collect at image borders where repeated zero padding keeps entering the convolutional stack. This is not the "needs more data" failure: if more examples make the qualitative result worse, the generator is not short on examples, it is being asked to learn a function its building blocks express poorly, spending capacity to approximate some nuisance transformation indirectly and overfitting around it.

Which transformation? The style-transfer objective fixes output contrast through the style image, not the content image. Take a content photograph and a low-contrast copy of the same photograph; both should stylize to outputs with nearly the same style-controlled contrast, because the Gram-matrix style loss asks for the style image's statistics. So a successful generator must discard each input's instance-specific contrast before imposing the fixed style — it cannot let "this content image was bright and high contrast" pass through unchanged. The most primitive form of this is a global contrast normalization, $y_{tijk} = x_{tijk} / \sum_l \sum_m x_{tilm}$, a whole-image spatial sum followed by a division broadcast back to every location. A stack of local convolutions and pointwise nonlinearities can approximate many things, but this kind of global reduction-and-broadcast is exactly what one should not force the network to rediscover from examples. The generators already carry a normalization primitive, batch normalization, which for $x$ of shape $T \times C \times W \times H$ computes per channel $i$ the moments $\mu_i = \frac{1}{HWT}\sum_t\sum_l\sum_m x_{tilm}$ and $\sigma_i^2 = \frac{1}{HWT}\sum_t\sum_l\sum_m (x_{tilm}-\mu_i)^2$, then $y_{tijk} = (x_{tijk}-\mu_i)/\sqrt{\sigma_i^2+\epsilon}$. This is close to the operation I want, but its axes are wrong: it removes the contrast of a channel relative to a statistic mixed across all images in the minibatch, so any one image's contrast is only partly removed because its statistic has been blended with the others. Worse, at inference batch normalization switches to fixed running population statistics, so a new image is normalized by the training population rather than by its own contrast — precisely backwards for a task where the test image is the thing whose contrast must go.

The fix is to keep the normalization form and change only the axis of summation, which I call Instance Normalization. Do not sum over the batch index $t$. For each image $t$ and channel $i$, take the mean and variance over only that channel's spatial grid,
$$\mu_{ti} = \frac{1}{HW}\sum_l\sum_m x_{tilm}, \qquad \sigma_{ti}^2 = \frac{1}{HW}\sum_l\sum_m (x_{tilm}-\mu_{ti})^2,$$
and normalize
$$y_{tijk} = \frac{x_{tijk}-\mu_{ti}}{\sqrt{\sigma_{ti}^2+\epsilon}}.$$
Every detail here is load-bearing: the denominator is $HW$ rather than $HWT$ because the batch is dropped from the average; the mean subtracted inside the variance is the same per-instance $\mu_{ti}$; and $\epsilon$ sits inside the square root to keep the scaling finite when a slice is nearly flat. Now each image-channel slice is centered and scaled by its own spatial statistics, so the generator receives the contrast-removal operation directly instead of having to approximate that global arithmetic through many convolutions. The whole method is this single axis change, applied everywhere the generator previously used batch normalization, in blocks of the form convolution then Instance Normalization then nonlinearity.

This axis change also settles the train/test question, and that is the second reason it works rather than an afterthought. Batch normalization needs two paths — noisy minibatch statistics during training, fixed population estimates at inference — because its statistics depend on the batch. With per-instance moments there is no batch to depend on: the forward pass is a deterministic function of a single image, so the same computation runs at training and at test time, and there is no running-statistics branch to swap in and silently change behavior on the one image whose style I am changing. A learned per-channel affine after normalization, as batch-norm modules usually carry, is fine to keep, but it is an implementation detail layered on top of the moment calculation, not a different choice of statistics.

To avoid accidentally writing a generic modern layer that diverges from the reference, I match the canonical Torch module: it implements no new kernel but folds $(N, C, H, W)$ into $(1, N\cdot C, H, W)$ and applies spatial batch normalization to those $N\cdot C$ folded channels. With a folded batch size of one, each folded channel is normalized over only its $H\times W$ positions — exactly the per-instance, per-channel statistics above. When affine parameters are enabled the length-$C$ weight and bias are repeated $N$ times so the same channel affine applies to each image. The defaults are $\epsilon = 10^{-5}$, momentum $0$, affine enabled with the weight initialized uniformly and the bias initialized to zero, and the module always asks the batch-norm functional for batch statistics (`training=True`) so that inference and training share one code path.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNormalization(nn.Module):
    """Mirrors DmitryUlyanov/texture_nets InstanceNormalization.lua."""

    def __init__(self, num_features, eps=1e-5, momentum=0.0, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            nn.init.uniform_(self.weight)  # canonical Lua code uses Tensor:uniform()
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        n, c, h, w = x.shape
        if c != self.num_features:
            raise ValueError(f"expected {self.num_features} channels, got {c}")

        folded = x.contiguous().view(1, n * c, h, w)
        weight = self.weight.repeat(n) if self.affine else None
        bias = self.bias.repeat(n) if self.affine else None

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
