## Research question

A feed-forward generator is trained to apply one fixed artistic style to arbitrary content images in a single pass. It is optimized with the same perceptual objective used by slow neural style transfer: preserve content features from the input image while matching style statistics from the style image. These generators are fast, but their images remain worse than the per-image optimization they are meant to amortize. Two failures are especially diagnostic: training on many content images can look worse than training on a small set with early stopping, and severe artifacts appear at image borders where repeated zero padding enters the convolutional stack. The practical question is whether this is a data or schedule issue, or whether the generator is missing a simple architectural operation that the stylization objective keeps asking it to learn.

## Background

**Optimization-based style transfer.** Gatys et al. synthesize a stylized image by directly optimizing pixels. A pretrained classification CNN supplies two kinds of measurements: deeper feature activations preserve content layout, while shallower spatially averaged feature correlations, expressed as Gram matrices, capture texture or style. The result can be high quality, but each image requires an iterative optimization process.

**Feed-forward stylization.** Texture Networks and the Johnson residual generator move the cost to training time. For a fixed style image `x_0`, a generator `g(x_t, z_t)` is learned from content images `x_t` and optional noise `z_t` by minimizing a perceptual loss,

```text
min_g (1/n) sum_t L(x_0, x_t, g(x_t, z_t)).
```

At test time the trained generator stylizes a new content image with one forward pass. The architectures are built from convolution, pooling or strided convolution, upsampling, nonlinearities, and channel-wise normalization layers.

**A contrast invariance forced by the objective.** In the style-transfer objective, output contrast is largely determined by the style image rather than the content image. A high-contrast and a low-contrast version of the same content image should stylize to outputs with nearly the same style-controlled contrast. Therefore a successful generator must somehow discard instance-specific content contrast before imposing the fixed style. A simple global contrast normalization already shows the kind of operation involved:

```text
y_tijk = x_tijk / sum_l sum_m x_tilm.
```

This depends on a whole-image spatial sum and a division broadcast back to every location. It is not an obviously easy function to synthesize from local convolutions and pointwise nonlinearities.

## Baselines

**Gatys optimization.** The per-image optimization is the visual reference point. Its strength is quality; its weakness is speed, because every stylized image is produced by a fresh iterative solve.

**Texture Networks.** The earlier feed-forward generator amortizes the Gatys objective into a compact convolutional network trained per style. It is fast, but qualitative quality can degrade when trained on thousands of examples rather than a small set, and border artifacts persist even with more complex padding.

**Johnson residual generator.** This residual feed-forward generator is trained with perceptual losses and achieves real-time stylization. It is a stronger architecture, but it still serves the same role: a convolutional generator attempting to approximate the style-transfer optimization in one pass.

**Batch normalization.** The existing normalization layer computes one mean and variance per feature channel using all images in the minibatch and all spatial positions:

```text
mu_i      = (1/(HWT)) sum_t sum_l sum_m x_tilm
sigma_i^2 = (1/(HWT)) sum_t sum_l sum_m (x_tilm - mu_i)^2
y_tijk    = (x_tijk - mu_i) / sqrt(sigma_i^2 + epsilon).
```

For convolutional feature maps, the statistics are shared across spatial positions. During inference, batch normalization uses fixed population estimates accumulated during training, so the test-time transform no longer depends on the particular input image's own statistics.

## Evaluation settings

The relevant setting is fixed-style feed-forward artistic style transfer. Each model is trained with a pretrained-CNN perceptual loss combining Gram-matrix style loss and content feature reconstruction loss. The primary comparison is qualitative: generated images are compared against the slow Gatys optimization and against the same generator architecture with the existing normalization layer. Both the Texture Networks architecture and a reproduced Johnson-style residual architecture matter, with attention to border artifacts, training-data sensitivity, and whether single-image inference behaves like training.

## Code framework

The open slot is the normalization operation placed inside the generator blocks. It receives a 4D activation tensor `(N, C, H, W)` and should return the same shape.

```python
import torch
import torch.nn as nn


class Normalization(nn.Module):
    """Per-channel normalization layer used between generator convolutions."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return normalize_with_chosen_statistics(
            x, eps=self.eps, affine=self.affine,
            weight=getattr(self, "weight", None),
            bias=getattr(self, "bias", None),
        )

```

Generator blocks have the form `convolution -> Normalization(C) -> nonlinearity -> ...`. The model is trained per fixed style image with a VGG perceptual loss: Gram-matrix style loss plus deep-feature content loss.
