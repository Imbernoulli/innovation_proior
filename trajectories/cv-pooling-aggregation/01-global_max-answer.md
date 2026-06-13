**Problem.** The scaffold's default global pool is Global Average Pooling, which divides each channel's
spatial map by `HW` and so dilutes a localized, discriminative peak into the surrounding silence. The
first rung tests the opposite hypothesis on the mean↔max axis: that the per-channel *peak* carries the
signal GAP averages away.

**Key idea (Global Max Pooling).** Replace the spatial mean with the per-channel spatial maximum:
`F.adaptive_max_pool2d(x, 1)` then flatten to `[B, C]`. Parameter-free, like GAP, so any accuracy
difference is attributable to the aggregation rule alone. It is the extremal opposite of the default
and the canonical "report the strongest evidence anywhere" rule (cf. Oquab et al., CVPR 2015).

**Why it is the floor.** Max gives winner-take-all credit: the whole upstream gradient for a channel
flows to one argmax position and zero elsewhere, so sub-maximal evidence never learns and the argmax
flickers early in training; and the max of many post-ReLU activations is larger and noisier than their
mean, feeding the frozen linear head a scale-shifted, higher-variance input. On the two CIFAR-100
backbones (ResNet-56 `8×8`, MobileNetV2) where the map genuinely pools, this should trail averaging.
On VGG-16-BN the map is already `1×1` entering the pool, so max = avg = a no-op there — VGG is an
uninformative tie, not evidence that max is good.

**Hyperparameters.** None. `__init__` is `super().__init__()`; the contract (`[B,C,H,W]→[B,C]`,
`C` preserved, valid down to `1×1`) is met by `adaptive_max_pool2d(x, 1).view(x.size(0), -1)`.

**What to watch.** Expect GMP weakest of the parameter-free rungs on ResNet-56 and MobileNetV2; if it
trails pure mean on the real maps, the fix is to *report both* mean and peak — rung 2.

```python
# EDITABLE region of pytorch-vision/custom_pool.py (lines 31-48) — step 1: Global Max Pooling
class CustomPool(nn.Module):
    """Global Max Pooling.

    Selects the maximum activation per channel across spatial dimensions.
    Captures the most salient features rather than averaging over all positions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
```
