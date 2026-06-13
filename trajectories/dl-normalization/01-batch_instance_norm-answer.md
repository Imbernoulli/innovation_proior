**Problem.** The two batch-free members of the normalization family fail in opposite directions on conv
features: instance normalization removes per-image style (good for style transfer) but also removes the
between-image scale that carries class signal, while the batch layer keeps that scale but also keeps
nuisance style. A single global choice cannot serve channels that want one behavior alongside channels
that want the other.

**Key idea.** At each layer, compute both the batch-normalized and the instance-normalized feature and
combine them with a **per-channel learnable gate** `g_c = σ(ρ_c) ∈ [0, 1]`:
`x̂ = g · x̂_BN + (1 − g) · x̂_IN`, then one per-channel affine `γ x̂ + β`. The network learns, channel by
channel and from the task loss alone, how much per-image style to wash out versus how much batch-level
scale to keep. `g = 1` is pure BatchNorm; `g = 0` is pure InstanceNorm.

**Why.** The blend is strictly more flexible than either endpoint — at `g ≡ 1` it reproduces the batch
layer it must at least match — and lets style-like channels slide toward instance while shape-like
channels stay batch-leaning. One affine on the *combined* feature (not per branch) keeps the new degrees
of freedom to just the `C` gates, so any accuracy change is attributable to the blend.

**Scaffold edit / hyperparameters.** Both branches compute statistics **directly in `forward`** from the
current batch (no running buffers, no momentum) — the eval batch is a full 128, so on-the-fly batch
moments are stable and the layer is one consistent function in train and eval. Batch stats over `(0,2,3)`,
instance stats over `(2,3)`, biased variance, `ε = 1e-5`. The gate is stored pre-sigmoid as `ρ`,
initialized at `1.0` so `σ(1) ≈ 0.73` — every channel starts batch-leaning (near the known-good batch
layer) and gradient descent slides it.

**What to watch.** On the two deep CIFAR-100 ResNets, where between-image scale is most class-relevant, the
instance branch has the most to lose and I expect this to land at or modestly below a clean batch layer; on
near-ceiling MobileNetV2 / FashionMNIST, where per-image contrast is more nuisance, the blend could help.
A thin or negative result on the ResNets says "blending toward instance is not the lever here" and points
the next rung at bringing more normalizers (the layer statistic) into the mix.

```python
class CustomNorm(nn.Module):
    """Batch-Instance Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Learns a per-channel gate rho in [0, 1] (via sigmoid) that interpolates
    between BatchNorm statistics and InstanceNorm statistics.

    Reference: Nam & Kim, "Batch-Instance Normalization for Adaptively
    Style-Invariant Neural Networks" (NeurIPS 2018)
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Gate parameter (before sigmoid); init at 1.0 -> sigmoid ~ 0.73 -> mostly BN
        self.rho = nn.Parameter(torch.ones(num_features) * 1.0)

    def forward(self, x):
        # x: [B, C, H, W]
        gate = torch.sigmoid(self.rho).view(1, -1, 1, 1)
        # Batch stats: per C over (B, H, W)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        # Instance stats: per (B, C) over (H, W)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        # Interpolate
        x_bn = (x - mean_bn) / (var_bn + self.eps).sqrt()
        x_in = (x - mean_in) / (var_in + self.eps).sqrt()
        x_norm = gate * x_bn + (1 - gate) * x_in
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
```
