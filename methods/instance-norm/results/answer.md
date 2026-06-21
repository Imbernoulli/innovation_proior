# Instance Normalization

Replace each generator batch-normalization layer with a normalization that computes statistics separately for every image and every channel, over spatial positions only, and use that same computation at inference.

For `x` with shape `T x C x W x H`:

```text
Batch norm:
mu_i       = (1/(HWT)) sum_t sum_l sum_m x_tilm
sigma_i^2 = (1/(HWT)) sum_t sum_l sum_m (x_tilm - mu_i)^2

Instance norm:
mu_ti       = (1/(HW)) sum_l sum_m x_tilm
sigma_ti^2 = (1/(HW)) sum_l sum_m (x_tilm - mu_ti)^2

y_tijk = (x_tijk - mu_ti) / sqrt(sigma_ti^2 + epsilon)
```

The method is the axis change: drop the batch sum from the moments. The reference code also supports an optional learned per-channel affine after normalization.

## Why It Works

- Stylized contrast is mostly determined by the style image, so the generator should not preserve the content image's instance-specific contrast.
- Batch normalization pools statistics across the minibatch and then switches to fixed population statistics at inference, so it does not normalize a test image by that image's own statistics.
- Instance normalization removes each image-channel slice's own spatial mean and scale, giving the generator the contrast-removal operation directly instead of forcing a conv/ReLU stack to approximate it.
- Because the statistics are per instance, the train-time and test-time computations are identical.

## Reference-Faithful Code

The canonical Torch implementation reuses spatial batch normalization by folding batch into channels:

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

In a modern PyTorch model, `nn.InstanceNorm2d(C, eps=1e-5, affine=True, track_running_stats=False)` gives the same normalization axes and train/test behavior, though its affine initialization differs unless reset.

Use this layer everywhere the stylization generator previously used batch normalization:

```text
convolution -> InstanceNormalization(C) -> nonlinearity -> ...
```
