The problem is the final spatial collapse of a convolutional image descriptor. After a standard CNN backbone, we have a tensor of activations with shape [B, C, H, W]. Every value is non-negative because the last layer is a ReLU. The classifier or retrieval head downstream needs a fixed-length vector, so each H by W feature map must be reduced to a single scalar, producing a [B, C] descriptor. Whatever information this collapse discards is lost forever, and whatever it overemphasizes the rest of the network must live with. The question is therefore how to aggregate each bag of N = W·H non-negative activations into one number that preserves as much discriminative signal as possible.

The standard rules are max pooling and average pooling, and both have well-known weaknesses. Max pooling, used by MAC and its regional variants, keeps only the single strongest activation per channel. It gives a sharp descriptor tied to one distinctive receptive-field patch, but it throws away every other location that fired, including corroborating evidence from other parts of the object, and a single noisy outlier can dominate the entire component. Regional max pooling mitigates this with a hand-designed grid of regions, but it introduces scales, strides, and overlap parameters that must be set externally and still partitions the image by fixed geometry rather than by the actual activation field. Average pooling, used by SPoC, treats every spatial location equally. It is smooth and stable and works well with whitening, but it dilutes a sparse strong response by averaging it against the large dead background, so the rare informative location receives the same per-pixel weight as a zero. CroW tries to recover with hand-designed saliency and channel weights, but it is still fundamentally a reweighted arithmetic mean and inherits the mean's character. Mixed pooling blends a max and an average with a learned scalar, but the output is constrained to a straight-line interpolation between two fixed summaries; it can sit partway between mean and max, but it cannot reshape how the individual locations inside the pool are weighted. None of these fixed or semi-fixed rules matches the sparse-strong-against-dead-field structure of real feature maps.

The better approach is GeM, Generalized-Mean pooling. Instead of choosing between keeping one location and weighting all equally, GeM raises each activation to a power p before averaging, then takes the p-th root. For one feature map X_k with N activations, it outputs f_k = ((1/N) · Σ_{x∈X_k} x^p)^{1/p}. When p = 1 this reduces to the arithmetic mean, so it recovers average pooling exactly. When p grows toward infinity it approaches max pooling. To see why, let m be the maximum activation and factor it out: the generalized mean equals m · ((1/N) Σ (x_i/m)^p)^{1/p}. Every ratio below one is suppressed to zero as p grows, while the maximizers stay at one, so the bracket tends to a positive constant and its p-th root tends to one, leaving m. The power-mean inequality guarantees the generalized mean increases monotonically and continuously with p, so p is a single smooth dial over selectivity from average toward max. This is structurally different from a convex combination of max and average outputs: GeM reweights the locations inside the pool itself, which a linear mixture cannot express.

Making p learnable is natural because the operation is differentiable. The backward pass distributes gradient to each spatial location with weight proportional to x_i^{p−1}, so at p = 1 the gradient is uniform like average pooling, and as p grows it routes gradient toward the strongest locations like a soft argmax. This means the forward descriptor uses all locations while the backward pass focuses learning signal on the salient ones. The gradient with respect to p is also cheap and finite, built from the same sums already computed in the forward pass. In practice p is initialized to 3.0, which sits in a contrast-enhanced regime without collapsing to a hard max, and is clamped to at least 1 so the layer stays in the average-to-max regime where larger p emphasizes larger values. Activations are clamped to a small epsilon before the power to keep zero-valued dead locations from producing infinities in log x or negative powers. The layer is implemented with the standard global-average-pool primitive, so it handles any spatial size including 1×1 and leaves the channel dimension unchanged.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized-mean pooling: f_k = ((1/N) sum_x x^p)^{1/p} per channel.
    p=1 -> average pooling; p->inf -> max pooling. p is a shared learnable dial."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):                          # x: [B, C, H, W], x >= 0 after ReLU
        p = self.p.clamp(min=1.0)                  # stay in the avg..max regime
        x = x.clamp(min=self.eps)                  # strictly positive base
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).view(x.size(0), -1)
```
