**Problem (from rung one).** The batch-instance blend slid along the *batch-versus-instance* axis and lost
accuracy on the deep CIFAR-100 ResNets (66.06 / 68.65), because the instance branch removes the
between-image scale those tasks rely on and the on-the-fly batch branch couples to batch composition. The
productive question is not "how much per-image style to remove" but "which feature positions should share
one statistic without crossing the batch axis."

**Key idea.** All of batch / layer / instance normalization are one formula with different choices of the
set `S_i` that shares a `(mean, var)`. Instance shares with one channel (under-sharing, noisy, blind to
neighbors); layer shares across all channels (over-sharing, conflates unlike filters). Split the channels
into `G` contiguous **groups** and pool within a group inside one image:
`S_i = {k : k_N = i_N, ⌊k_C/(C/G)⌋ = ⌊i_C/(C/G)⌋}`, then a per-channel affine. `G = 1` recovers layer,
`G = C` recovers instance; an interior `G` is the new intermediate.

**Why.** The set fixes `k_N = i_N`, so it is batch-free — identical in train and eval, no running buffer,
no batch-composition coupling (fixing rung one's batch branch). The group pools `C/G` channels over the
full spatial grid, a large stable sample that uses cross-channel structure (fixing rung one's noisy
isolated-channel instance branch) without ever over-removing per-image scale.

**Scaffold edit / hyperparameters.** Delegate the moment computation to `nn.GroupNorm`. Choose the group
count *adaptively* because the layer widths here are small and uneven: start from the paper default 32, cap
at the channel count `min(32, num_features)`, then decrement until it divides `num_features`. On 64 ch →
32 groups; on 32 or 16 ch → 1 channel/group (instance-equivalent — the graceful fallback at the `G = C`
boundary); on 96/160 ch → 32 groups. `ε = 1e-5` and the per-channel affine are `nn.GroupNorm`'s defaults.

**What to watch.** I expect clear recovery past the blend on the two CIFAR-100 ResNets (clearest on the
deeper ResNet-110), and roughly even on near-ceiling MobileNetV2 / FashionMNIST, whose many small/awkward
channel counts collapse the group statistic toward instance-equivalent. A confirmed split says the lever is
*grouping inside one image*, and points the next rung at *learning* the mix of statistics per layer instead
of fixing the grouping by hand.

```python
class CustomNorm(nn.Module):
    """Group Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Divides channels into groups and normalizes within each group independently.
    Works well with small batch sizes where BatchNorm statistics are noisy.

    Reference: Wu & He, "Group Normalization" (ECCV 2018)
    """

    def __init__(self, num_features):
        super().__init__()
        num_groups = min(32, num_features)
        # Ensure num_features is divisible by num_groups
        while num_features % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, num_features)

    def forward(self, x):
        return self.norm(x)
```
