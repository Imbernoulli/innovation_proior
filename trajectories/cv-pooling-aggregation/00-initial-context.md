## Research question

The final spatial aggregation step of a modern image-classification CNN reduces a feature map
`[B, C, H, W]` to a vector `[B, C]` that the linear classifier head reads. The single thing being
designed is **that pooling rule** — how the `H×W` spatial activations of each channel are collapsed
into one number — under a hard interface constraint: the output channel dimension must equal the input
channel dimension exactly, the rule must work for spatial sizes from `8×8` (ResNet on CIFAR) down to
`1×1` (after VGG's max-pools), and it sees no labels and no training data. Everything else — the
backbone, the classifier head, the optimizer, the schedule, the augmentation — is frozen. The question
is whether a better pooling rule than the default Global Average Pooling raises test accuracy across
architectures that differ in channel count and feature-map statistics.

## Prior art before the first rung (the pooling lineage)

The default the first rung reacts to is Global Average Pooling, and GAP is itself the resolution of a
short lineage of aggregation choices. These precede the ladder; each is named with the gap that the
next pooling rule reacts to.

- **Flatten + fully-connected head (Krizhevsky et al. 2012; Simonyan & Zisserman 2015).** The original
  classifier reshaped the whole `C×H×W` feature map into a long vector and fed it to large FC layers.
  This keeps every spatial position but ties the head to a fixed input resolution, carries most of the
  network's parameters, and overfits readily. Gap: parameter-heavy, resolution-bound, no spatial
  invariance.
- **Global Average Pooling (Lin et al., "Network in Network," 2014).** Replace the FC head with a
  per-channel spatial mean: `[B,C,H,W] → [B,C]`, then one linear layer. Parameter-free, resolution-
  agnostic, a strong regularizer, and it makes each channel a "category confidence map." It is the
  default in the scaffold. Gap: it treats every spatial location identically and **averages away the
  distribution** of activations — a single strong, localized response is diluted by the many weak
  positions around it.
- **Global Max Pooling (standard in fine-grained recognition / retrieval; cf. Oquab et al., CVPR
  2015, who pool the per-class score map with a max to localize objects from image-level labels).**
  Take the per-channel spatial maximum instead of the mean: keep the single strongest activation,
  discard the rest. This is the opposite extreme of GAP — it answers GAP's dilution complaint by
  reporting only the peak. Gap: it discards *all* spatial context outside the peak and passes gradient
  to only one location per channel.
- **Average + Max, and learnable power-means.** Between the two extremes sit rules that report *both*
  the mean-field and the peak (their element-wise sum/average, no parameters), or that **learn** where
  to sit on the mean↔max axis with a single power parameter. These are exactly the rungs of this
  ladder.

## The fixed substrate

The training stack is frozen and must not be touched. Three architectures are evaluated, each built
with the *same* `CustomPool` slotted in as the final aggregator before the classifier:

- **ResNet-56** (CIFAR-adapted: 3×3 stem, no max-pool, BasicBlocks `[9,9,9]`), `C=64`, feature map
  `8×8` entering the pool, on CIFAR-100.
- **VGG-16-BN** (CIFAR-adapted, five `2×2` max-pools), `C=512`, feature map `1×1` entering the pool,
  on CIFAR-100.
- **MobileNetV2** (width 1.0, stride-1 stem for 32×32 input, `conv_last` to 1280), `C=1280`, on
  FashionMNIST.

The optimizer is SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`), cosine-annealed over `200`
epochs, batch size `128`, Kaiming init, cross-entropy loss. Augmentation is `RandomCrop(32, pad=4)` +
`RandomHorizontalFlip`; FashionMNIST is resized to 32 and channel-replicated to 3. The training and
evaluation loops, the data pipeline, and `build_model` are all fixed. The only object the model
constructs differently is `self.pool = CustomPool()`, called once per forward as
`x = self.pool(x)` on the `[B,C,H,W]` feature map.

## The editable interface

Exactly one region is editable — the `CustomPool` class in `pytorch-vision/custom_pool.py` (lines
31–48). The contract is narrow and the same for every rung: `forward(x)` receives a `[B, C, H, W]`
tensor and must return a `[B, C]` tensor, with `C` preserved exactly (no concatenation that widens the
channel dimension, since `self.fc`/`self.classifier` expects `C` inputs), valid for any `H, W` down to
`1×1`, and using no labels or data. A rung may add learnable parameters inside the class, choose the
aggregation function (mean, max, weighted, power-mean, attention), and decide how spatial information
is summarized — as long as the shape contract holds.

The starting point is the scaffold default: **Global Average Pooling**. Each rung replaces exactly
this class body and nothing else.

```python
# EDITABLE region of pytorch-vision/custom_pool.py (lines 31-48) — default fill (Global Average Pooling)
class CustomPool(nn.Module):
    """Custom global pooling layer.

    Reduces spatial feature maps [B, C, H, W] to feature vectors [B, C].
    Used as the final spatial aggregation before the classifier head.

    Design considerations:
        - How to aggregate spatial information (mean, max, learned, mixed)
        - Whether to use learnable parameters for adaptive aggregation
        - Robustness across different spatial resolutions and channel counts
        - Interaction with downstream classifier and upstream features
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
```

## Evaluation settings

Three (architecture, dataset) pairs, each over seed `42`: **ResNet-56 on CIFAR-100**, **VGG-16-BN on
CIFAR-100**, and **MobileNetV2 on FashionMNIST** (the last is held out / hidden). The metric on every
pair is **best test accuracy (%) achieved during training** — higher is better — recorded as the max
over the 200-epoch run. The pooling module must accept the convolutional feature maps, return the
expected channel vector, handle the variable spatial sizes, and must not change datasets, classifier
targets, optimizer behavior, or test-time evaluation.
