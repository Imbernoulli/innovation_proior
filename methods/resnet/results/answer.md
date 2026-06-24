# ResNet, distilled

ResNet trains image-recognition networks far deeper than was previously feasible by re-parameterizing each block to learn a **residual** added onto a parameter-free **identity shortcut**. The whole method reduces to one line per block: `out = F(x) + x`.

## The problem

Depth helps image recognition, but you cannot get it by naively stacking layers. With modern initialization (He) and Batch Normalization, very deep *plain* nets do converge — yet they suffer the **degradation problem**: beyond ~20–30 layers, adding layers *increases training error*. This is:

- **not overfitting** — training error itself rises, not just test error;
- **not vanishing gradients** — BN keeps forward signals at healthy variance and backward gradients at healthy norms (you can measure them).

It is an **optimization / conditioning** failure. A good solution provably exists: a deep net can copy a shallow net into its bottom layers and set every extra layer to the identity, so it should never train worse than the shallow net. SGD cannot find that solution — because learning an identity-like mapping through a stack of conv–BN–ReLU is hard. The identity sits at a non-trivial point in weight space that small-init, weight-decayed SGD has no pull toward. The capacity is there; the optimizer can't reach it.

## The key idea

**Reparameterize** each group of layers to learn a **residual** instead of the full mapping. If the desired mapping is `H(x)`, let the layers fit `F(x) = H(x) − x` and recover the output by adding the input back through a **parameter-free identity shortcut**:

```
y = F(x, {W_i}) + x
```

Why this is well-conditioned:
- **Identity becomes free.** If the optimal mapping for a block is the identity, the solver only has to push `F → 0` (zero conv weights, with zero BN shift, make the residual branch output zero) — the easiest target there is, and exactly where weight decay already pushes the learned weights. Previously, fitting an identity *from scratch* through nonlinear layers was the hard case; now it is the easy one.
- **Near-identity is preconditioned.** If the optimal mapping is close to identity, the block learns a *small perturbation referenced to x* ("here is x, nudge it"), not the whole function from scratch.
- **Controlled comparison.** Matched-dimension identity shortcuts add **no parameters and no meaningful compute**, so the pure identity/zero-pad comparison can match a plain net at identical depth / width / params / FLOPs. Projection shortcuts in option B do add a small 1×1+BN path at stage transitions; they are isolated shape matchers, not the source of the residual mechanism.
- **Gradient path (a bonus, not the explanation).** For the pre-activation sum `s = F(x) + x`, the exact Jacobian is `∂s/∂x = J_F(x) + I`; the post-add ReLU then left-multiplies by its activation mask. The identity term gives a direct local path when units are active, but the load-bearing claim is *conditioning*, not gradient rescue — BN already keeps the plain net's gradients healthy.

## Design decisions and their reasons

| Decision | Why this and not the alternative |
|---|---|
| Learn `F = H − x`, recover `H = F + x` | Degradation is an optimization problem; preconditions the solver toward the identity (push weights → 0) instead of forcing it to build identity from scratch. |
| **Identity** shortcut, parameter-free | Vs. a Highway-style gate `y = H·T + x·C`: a gate has parameters and can *close* (carry → 0), losing the skip exactly when depth needs it; identity never closes, always passes all info, costs nothing, and keeps the comparison clean. More constrained on purpose. |
| Final ReLU **after** the add, `σ(F+x)` | A ReLU on the residual branch's output would clamp `F ≥ 0`, preventing negative corrections; rectify only the sum. |
| Dimension match: **option B** (1×1 projection only when channels/stride change; identity elsewhere) | A (zero-pad identity) leaves the padded channels with no residual learning at transitions; C (project every skip) adds parameters on dozens of skips and muddies the controlled comparison for a marginal gain. B keeps almost every skip free and projects only where needed. |
| **Bottleneck** `1×1 reduce → 3×3 → 1×1 restore` (expansion 4) for deep nets | The 3×3's cost is quadratic in channels; running it at the *reduced* width c makes one 3-layer block cost `2·C·c + 9c² = 17c²` (with ends `C = 4c`) vs the basic block's `18c²` — ≈0.94×, i.e. "same FLOPs," so the budget buys 3 weight layers instead of 2. |
| Keep the bottleneck skip an **identity** | A 256→256 projection across the bottleneck's two high-dim ends costs `C² = 16c²` in parallel with the `17c²` branch → block cost `33c²` vs `17c²`, ratio **33/17 ≈ 1.94** (a near-doubling) in FLOPs and params; identity makes it free — essential for affordable very-deep nets. |
| Stride on the first **1×1 reduce** in bottleneck transition blocks | Downsample at the block entrance, reduce channels immediately, then run the expensive 3×3 on the smaller reduced-width map; the shortcut projection uses the same stride so the two branches align for addition. |
| BN after every conv, before activation | Makes plain deep nets converge *and* serves as the evidence that degradation is not a vanishing-gradient problem (healthy signal/gradient norms). |

## The architecture

- **Stem:** 7×7, 64, stride 2 → 3×3 max-pool stride 2.
- **Four stages** of residual blocks (VGG rule: same map size → same #filters; halve map → double filters): channels 64→128→256→512; downsampling by stride-2 convs at `conv3_1 / conv4_1 / conv5_1`.
- **Head:** global average pooling → 1000-way FC → softmax (no fat FC head).
- **BasicBlock** (18/34): two 3×3 convs, `F = W₂σ(W₁x)`; expansion 1.
- **Bottleneck** (50/101/152): 1×1 reduce → 3×3 → 1×1 restore; expansion 4. The 152-layer net (≈11.3 GFLOPs) is lighter than VGG-19 (19.6 GFLOPs) despite being 8× deeper.
- Per-stage block counts: 18 = [2,2,2,2], 34 = [3,4,6,3], 50 = [3,4,6,3], 101 = [3,4,23,3], 152 = [3,8,36,3].

## Training recipe

BN after every conv and before activation; He (kaiming, fan_out) init; train from scratch. SGD, mini-batch 256, momentum 0.9, weight decay 1e-4, lr 0.1 divided by 10 on plateau (up to 60×10⁴ iters). VGG/AlexNet augmentation (scale jitter shorter-side ∈ [256,480], 224×224 random crop + flip, per-pixel mean subtraction, color augmentation). No dropout (BN regularizes). Testing: 10-crop, or fully-convolutional multi-scale. (Very deep CIFAR nets may need a brief 0.01 warm-up before 0.1.)

## Transfer to detection / localization

The deep features transfer by reusing the conv stack itself as both shared extractor and per-region head (no fresh FC head needed):

- **Where to cut (NoC).** Match VGG-16's /16 shared-feature stride: the stride-≤16 layers — conv1 + conv2_x + conv3_x + conv4_x = **91 conv layers** in ResNet-101 — are the shared full-image extractor; RoI-pool before conv5_1; **conv5_x** (the last stage) is the per-region head, ending in sibling cls/reg layers. (91 + conv5_x's 9 = 100 conv + final FC = the "101".)
- **Frozen BN.** With tiny detection batches, re-estimating BN stats is unstable, so freeze BN to its ImageNet population stats. A frozen BN is then exactly a per-channel affine `ŷ = a·x + b`, `a = γ/√(σ²+ε)`, `b = β − γμ/√(σ²+ε)` — foldable into the preceding conv, saving fine-tune memory.
- **Localization.** Per-class RPN: sibling 1×1 heads, `cls` 1000-d (per-class binary logistic), `reg` 1000×4-d (per-class box regressors over anchors). Fast R-CNN stalls here — single-object images give heavily-overlapping proposals whose RoI features are near-identical, so image-centric sampling yields low-variance batches; switch to RoI-centric **R-CNN** (crop→warp 224→independent forward) to restore sample variation.
- **Diagnostic.** Swapping VGG-16→ResNet-101 raises strict mAP@[.5,.95] (≈+6.0) almost as much as loose mAP@.5 (≈+6.9) → depth improves **localization**, not just recognition. Box refinement (re-pool the regressed box), global context (whole-image SPP feature concatenated), and multi-scale testing each add further gains, amplified by the deeper features.

## Working code

```python
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity                 # y = F(x) + x
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)            # reduce; stride at the block entrance
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)                      # spatial at reduced width
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)     # restore
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity                 # F(x) + x
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18():  return ResNet(BasicBlock, [2, 2, 2, 2])
def resnet34():  return ResNet(BasicBlock, [3, 4, 6, 3])
def resnet50():  return ResNet(Bottleneck, [3, 4, 6, 3])
def resnet101(): return ResNet(Bottleneck, [3, 4, 23, 3])
def resnet152(): return ResNet(Bottleneck, [3, 8, 36, 3])
```

The structure keeps the compact PyTorch implementation style while using the original paper/author-code bottleneck downsampling convention above. The entire method reduces to one line in each block: `out += identity`.
