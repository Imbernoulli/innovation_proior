**Problem (from rung two).** Group normalization recovered the deep CIFAR-100 ResNets (67.90 / 70.43) but
slipped on MobileNetV2 / FashionMNIST (93.16), because its hand-written adaptive grouping collapses narrow
layers to the instance-equivalent boundary and, being strictly batch-free, can never use the comfortable
batch-128 statistic. The remaining headroom is the rigidity: one fixed grouping and pooling rule at every
layer, chosen by heuristic rather than by the task.

**Key idea.** Compute all three corners of the family — instance (over `(2,3)`), layer (over `(1,2,3)`),
batch (over `(0,2,3)`) — and let each layer **learn** a convex combination over them, end to end. Two
softmaxed weight 3-vectors, one for the mean and one for the variance, give `mean = Σ w_k mean_k`,
`var = Σ w'_k var_k`; standardize by the mixed moments, then a per-channel affine. Collapsing the weights
recovers any single normalization; any interior point is a new operating point no fixed rule could reach.

**Why.** This strictly generalizes the earlier rungs: it can reproduce a layer/instance-leaning blend on
the deep ResNets (matching group normalization) *and* dial in the **batch** corner on the narrow
MobileNetV2 layers that group normalization was denied. Separate mean/variance weights let centering and
scaling pick different corners. The softmax forces a genuine convex interpolation, not an unstable arbitrary
linear combination.

**Scaffold edit / hyperparameters.** Per-layer weights shared across channels — `nn.Parameter(torch.ones(3))`
for mean and for variance (just 6 numbers per layer), softmaxed; initialized all-ones so every layer starts
as a uniform blend. Batch stats computed **on the fly** in `forward` (no running buffers) — stable at batch
128, and the mix can learn its way off the batch corner if the eval-batch coupling hurts. Biased variance,
`ε = 1e-5`, affine `γ/β` initialized to 1/0.

**What to watch.** Clearest expected gain on MobileNetV2 / FashionMNIST (reclaiming the batch statistic the
narrow layers were denied); match-or-modest-gain on the two CIFAR-100 ResNets. The one risk is the
on-the-fly batch corner destabilizing a deep stack if over-weighted — if so, ResNet-110 could land at or
below group normalization.

```python
class CustomNorm(nn.Module):
    """Switchable Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Learns to combine BatchNorm, InstanceNorm, and LayerNorm statistics via
    softmax-weighted importance weights. Adapts normalization strategy per
    channel during training.

    Reference: Luo et al., "Differentiable Learning-to-Normalize via
    Switchable Normalization" (ICLR 2019)
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Importance weights for mean (3 norms) and var (3 norms)
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Softmax over importance weights
        mean_w = F.softmax(self.mean_weight, dim=0)
        var_w = F.softmax(self.var_weight, dim=0)
        # Instance stats: per (B, C) over (H, W)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        # Layer stats: per B over (C, H, W)
        mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)
        var_ln = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        # Batch stats: per C over (B, H, W)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        # Weighted combination
        mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
        var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
```
