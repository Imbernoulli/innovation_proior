**Problem (from step 1).** Global Max trailed where the pool actually pools — ResNet-56 69.96,
MobileNetV2 94.27 — because winner-take-all credit starves the 63 sub-maximal positions and the max is
a noisier, scale-shifted input to the frozen head. Pure mean (GAP) and pure max are each the disease
the other cures: GAP dilutes a localized peak, max discards everything but it.

**Key idea (Average + Max).** Stop choosing an extreme; report *both* statistics fused back into one
`C`-vector. Compute the per-channel mean and the per-channel max, return their element-wise average
`(avg + mx)/2 → [B, C]`. Concatenation is impossible here (the frozen head expects `C`, not `2C`), so
the fusion must preserve the channel dimension — the element-wise mean does.

**Why it works.** Per channel it keeps the better of the two extremes: most of the peak where evidence
is localized, the mean-field statistic where evidence is spread. The mean branch routes gradient to
*every* position (the credit-assignment fix for 69.96) while the max branch still lets a channel
sharpen a real peak; the output is bounded between mean and max, so it is strictly less scale-shifted
than the pure max the head already tolerated. Parameter-free, so the comparison to the floor stays
clean.

**Hyperparameters.** None. Fixed 50/50 blend; `__init__` is `super().__init__()`. Valid down to
`1×1`, where `avg = max` makes it identical to GAP (so VGG-16-BN, whose map is `1×1` entering the pool,
is again uninformative).

**What to watch.** Expect a rise over Global Max on the two real-map backbones (ResNet-56 into the
low-71s, MobileNetV2 mid-94s); VGG near the floor's 74.43, any dip being upstream noise on an invisible
operator. The remaining weakness: the blend is *fixed* 50/50 for every channel and backbone — the next
rung learns where to sit on the mean↔max axis.

```python
# EDITABLE region of pytorch-vision/custom_pool.py (lines 31-48) — step 2: Average + Max Pooling
class CustomPool(nn.Module):
    """Average + Max Pooling.

    Element-wise mean of global average pooling and global max pooling.
    Combines mean-field statistics with peak activations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return ((avg + mx) / 2).view(x.size(0), -1)
```
