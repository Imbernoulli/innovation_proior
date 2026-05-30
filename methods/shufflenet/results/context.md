# Context

The aim is to design a *basic* convolutional architecture that achieves the best possible accuracy under an extremely tight computational budget — tens to hundreds of MFLOPs (roughly 10–150 MFLOPs) — for mobile platforms such as drones, robots, and smartphones, using the efficient-convolution primitives (group convolutions, depthwise separable convolutions) and residual-network building blocks available in mid-2017. The target is ImageNet classification (and MS COCO detection) at these tiny budgets.

## Research question

The dominant trend is bigger: the most accurate CNNs stack hundreds of layers and thousands of channels and cost billions of FLOPs. The opposite extreme — the best accuracy achievable in a budget of only tens to hundreds of MFLOPs — is comparatively unexplored as an *architecture* problem. Much prior efficiency work takes a "basic" network and then prunes it, compresses it, or quantizes it to low-bit; the question here is different: what is the most efficient *basic architecture* one can design directly for these very small budgets?

The sharp problem hides in where the FLOPs actually go. The state-of-the-art efficient building blocks (depthwise separable convolutions, grouped 3×3 convolutions) make the *spatial* convolution cheap — but they leave the **1×1 (pointwise) convolutions dense**, and in a small network the pointwise convolutions then dominate the cost. So the precise question: in the tiny-budget regime, how do you cut the cost of the pointwise convolutions too, without crippling the network's ability to mix information across channels — and thereby, for a fixed FLOP budget, afford *more channels* (which a tiny network badly needs, since too few channels cannot carry enough information)?

## Background

By mid-2017 several efficient-convolution ideas are established, and the relevant facts concern how a building block spends its FLOPs.

- **The residual bottleneck unit.** A residual block `y = x + F(x)` with `F` a bottleneck `1×1 reduce → 3×3 → 1×1 expand`, running the spatial 3×3 at a reduced channel count. For input `c×h×w` and bottleneck width `m`, a dense residual unit costs `hw(2cm + 9m²)` FLOPs: the two pointwise convolutions contribute `2cm`, the dense 3×3 contributes `9m²`.
- **Grouped convolution.** Partition the input and output channels into `g` groups and convolve only within each group, cutting the cost of that layer by roughly `g`. Introduced in the 2012 two-GPU net to split a model across GPUs, and later shown to *improve* accuracy when used on the 3×3 layers of a residual block (cardinality). With grouped 3×3, a residual unit costs `hw(2cm + 9m²/g)` — the spatial term shrinks by `g`, but **the `2cm` pointwise term does not**.
- **Depthwise separable convolution.** Factor a standard convolution into a *depthwise* convolution (one filter per input channel, no cross-channel mixing) followed by a *pointwise* `1×1` convolution (which does the channel mixing). The depthwise step is extremely cheap in theory. Used to build state-of-the-art lightweight models. A practical caveat: depthwise convolution has a poor compute-to-memory-access ratio, so on low-power mobile hardware it is hard to implement efficiently and is best applied sparingly.
- **No nonlinearity after a depthwise convolution.** Established practice (Xception): putting a ReLU right after a depthwise convolution hurts, so the rectifier is omitted there.
- **Batch Normalization** after convolutions, and ReLU as the nonlinearity, are standard in residual/grouped blocks.
- **A latent primitive.** A "random sparse convolution" — a random channel permutation followed by a grouped convolution — exists in an early CNN library, but was used for a different purpose and seldom exploited.

The diagnostic observation that motivates the whole design: in a grouped-3×3 residual network at small scale, the *pointwise* (1×1) convolutions occupy the overwhelming majority of the multiply-adds — about 93% of a residual unit's compute when only the 3×3 layers are grouped (cardinality 32). So in a tiny network the dense pointwise convolutions are what force the channel count to stay small under the budget, and a too-small channel count is exactly what limits accuracy in tiny networks (thin feature maps cannot carry enough information). A second diagnostic, about regularization: very small networks tend to *underfit* rather than overfit, so they want lighter weight decay and gentler data augmentation than large networks.

## Baselines

**ResNet bottleneck (He et al. 2016).** Residual unit with a dense `1×1 → 3×3 → 1×1` bottleneck. *Core math:* `hw(2cm + 9m²)` FLOPs per unit. *Gap:* both the spatial 3×3 (`9m²`) and the two pointwise convolutions (`2cm`) are dense; at a tiny budget this forces a very small channel count.

**ResNeXt (Xie et al. 2016).** Replaces the dense 3×3 with a *grouped* 3×3 (cardinality). *Core math:* `hw(2cm + 9m²/g)` FLOPs — the spatial term drops by `g`. *Gap:* the pointwise `2cm` term is untouched and now *dominates* (≈93% of a unit's FLOPs at small scale), so ResNeXt becomes inefficient exactly in the tiny-budget regime; the savings were spent on the wrong term.

**Xception / depthwise separable, and MobileNet (Chollet 2016; Howard et al. 2017).** Build blocks from a depthwise 3×3 plus a pointwise 1×1; MobileNet uses these for state-of-the-art lightweight models. *Core idea:* make the spatial convolution nearly free by separating it from channel mixing. *Gap:* the pointwise `1×1` convolutions (which do all the channel mixing) remain dense and dominate the FLOPs in small networks; and the depthwise step, while cheap in theory, is hard to run efficiently on mobile hardware.

**Pruning / quantization / factorization of a fixed network.** Reduce a pre-trained model's redundancy after the fact. *Gap / contrast:* these accelerate an *existing* architecture rather than offering a basic architecture designed from the start for the tiny-budget regime.

## Evaluation settings

- **ImageNet-2012 classification, 1000 classes.** Metric: single-crop top-1 error — center 224×224 crop from a 256× resized image — on the validation set. Models are compared at *matched FLOP budgets* (≈140 MFLOPs, ≈40 MFLOPs, ≈13 MFLOPs) so accuracy differences reflect the architecture, not the budget.
- **MS COCO object detection** as a transfer test of the learned features.
- **Cost metric.** FLOPs (multiply-adds), held fixed across compared models. Crucially, *actual* inference latency on real mobile hardware (an off-the-shelf ARM core) is also a first-class metric, because theoretical FLOPs and wall-clock speed can diverge (e.g. depthwise convolution's poor memory-access behavior).
- **Controlled comparisons that exist at the time.** Vary the number of groups `g` at fixed complexity (adapting channel widths to hold FLOPs constant); toggle a candidate cross-channel-mixing operation on and off; swap the whole building block for the ResNet / ResNeXt / Xception-style / VGG-like alternatives at the same budget.
- **Training protocol (inherited from the grouped-residual recipe, with small-net adjustments).** SGD, batch 1024 across 4 GPUs, ~3×10⁵ iterations; lighter weight decay (4e-5) and a linearly decayed learning rate (0.5 → 0), with less aggressive scale augmentation — because tiny networks underfit.

## Code framework

The available code is a residual-network image-classification harness whose convolution primitive supports a `groups` argument (grouped and depthwise convolution). The libraries supply grouped/depthwise convolution, batch normalization, ReLU, pooling, an SGD optimizer with a linear LR schedule, and a cross-entropy loss. The scaffold is the harness with two empty slots: the cross-channel-mixing operation, and the efficient building unit that the stages repeat.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def grouped_conv1x1(in_ch, out_ch, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, groups=groups, bias=False)


def depthwise_conv3x3(channels, stride=1):
    # one filter per channel: groups == channels (no cross-channel mixing)
    return nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                     padding=1, groups=channels, bias=False)


def mix_channels(x, groups):
    # TODO: the cross-channel mixing operation. After a grouped convolution,
    # each output channel depends only on its own group's inputs. Something is
    # needed here to let information move between groups. Open.
    raise NotImplementedError


class Unit(nn.Module):
    # TODO: the efficient building unit we'll design. Known going in: a residual
    # block of bottleneck shape (reduce -> spatial -> expand) added onto a
    # shortcut. Open: which of its convolutions are grouped vs dense vs depthwise,
    # where the cross-channel mixing goes, and how the stride-2 (downsampling)
    # variant matches and combines the shortcut.
    def __init__(self, in_ch, out_ch, groups=3, stride=1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Net(nn.Module):
    # Fixed by the residual template: a 3x3/stride-2 stem + maxpool, then three
    # stages that repeat `Unit` (first unit per stage downsamples; channels
    # double per stage; bottleneck = 1/4 of output channels), global average
    # pool, and a classification FC. The unit and the mixing op are the slots.
    def __init__(self, groups=3, num_classes=1000, scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stages built from Unit; widths set per `groups` and `scale`
        self.stage2 = nn.Sequential()  # TODO
        self.stage3 = nn.Sequential()  # TODO
        self.stage4 = nn.Sequential()  # TODO
        self.fc = nn.Linear(0, num_classes)  # in_features set from final stage width

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# --- training harness ---
model = Net(groups=3, num_classes=1000, scale=1.0)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=4e-5)
# linear LR decay 0.5 -> 0 over ~3e5 iterations; batch 1024 on 4 GPUs;
# less aggressive scale augmentation (tiny nets underfit).
```
