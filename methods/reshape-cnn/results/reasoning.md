Let me start from the thing that doesn't fit. Depth has been the lever for every recent jump in image recognition — eight layers, then sixteen to nineteen, then twenty-two, each deeper net beating the last — and the natural conclusion is just to keep stacking. For a while the only thing standing in the way was that very deep nets wouldn't even start to learn: the signal either blew up or shrank to nothing as it propagated through dozens of layers, and the gradients did the same on the way back. But that obstacle is handled now. Variance-preserving initialization keeps the activation variance roughly constant across layers at the start, and Batch Normalization, sitting after each convolution, renormalizes every layer's pre-activations during training — subtract the mini-batch mean, divide by the mini-batch standard deviation, then a learned scale and shift restore the freedom I just removed. With those in place a deep plain stack of conv-BN-ReLU does converge. So I should be able to just keep adding 3×3 conv layers and watch the accuracy climb.

It doesn't climb. Past a couple of dozen layers it saturates, and then it gets *worse*. A 56-layer plain net, built exactly like a 20-layer one but with more of the same blocks, ends up with higher error. And here's the part that stops me: it's not a test-set story, not overfitting. The *training* error of the 56-layer net is higher than the 20-layer net's. The deeper model is fitting the training data *less* well than the shallower one. Same on the bigger benchmark — go from 18 plain layers to 34 and the training error goes up, throughout the whole run, not just at the end.

Let me make sure I'm not fooling myself about the cause. The reflex is "vanishing gradients" — that's the classic deep-net disease. But I have BN in this net, so I can inspect exactly the quantities that would expose that failure. The forward responses keep non-zero variance layer after layer, and the backward gradient norms are healthy too, not collapsing toward zero down the stack. So neither the forward signal nor the backward gradient is vanishing. And the deep plain net isn't *broken* — left to run, it still reaches a competitive accuracy, just slowly and to a worse place. So the solver works; it's just being defeated by something. This isn't the old gradient problem wearing a new hat. It's something else, and I need to name it precisely before I can fix it.

Take my trained 20-layer net. Now build a 56-layer net by copying those 20 layers into the bottom and making every one of the 36 extra layers compute the identity — pass its input straight through, untouched. That deeper net computes *exactly the same function* as the 20-layer net. Its training error is identical, by construction. So a 56-layer solution that is at least as good as the 20-layer net's provably exists inside the 56-layer net's parameter space. The deeper model contains the shallower one as a special case; its hypothesis space is a superset. There is no representational excuse for it to train worse. And yet, started from scratch and handed to SGD, the 56-layer net lands somewhere worse than that constructed solution. The capacity is sitting right there and the optimizer can't find it — or can't find it in any feasible amount of time.

So the problem is not capacity and not gradients. It's that the optimizer, faced with the deeper architecture, can't navigate to a solution it could in principle represent. That reframes the whole task. I don't need a more expressive net; the plain net is already expressive enough to copy the shallow one. I need to make that good solution *easier to reach*.

Now, what specifically is hard to reach? The constructed solution required those 36 extra layers to each be the identity. So the question sharpens: how hard is it for a stack of conv-BN-ReLU layers to learn to be the identity map? And the answer, apparently, is: surprisingly hard. The identity isn't a natural resting point for a stack of nonlinear layers — to make `W₂σ(W₁x)` reproduce `x` for all `x`, the weights have to land on a very particular nontrivial configuration, and there's nothing pulling SGD toward it. Small-init, weight-decayed SGD pulls weights toward *zero*, and a block of weights at zero computes the *zero* map, not the identity — it annihilates its input. So precisely the mapping I'd most like a redundant deep layer to fall into — "do nothing, pass it along" — is one of the harder things for these layers to express, and it's working against the regularizer rather than with it. If degradation is the solver failing to make extra layers harmless, and "harmless" means "identity," and identity is hard to learn, then the disease and its mechanism line up.

That points away from a better optimizer or a better init and toward changing *what the layers are asked to learn*, so that "do nothing" becomes the easy default instead of the hard target. I keep thinking about how some hard problems become easy under the right re-expression. Encoding a vector by its residual against a dictionary works better than encoding the vector itself; solving a PDE on the residual between a coarse and a fine grid converges far faster than solving it directly; preconditioning doesn't change what's representable, it changes the geometry the solver walks. The common thread: reference the unknown to something you already have, and solve for the small correction.

What do I already have, at the input to a block? I have `x` itself. So let me reference the block's output to its own input. Say the few stacked layers are *supposed* to realize some mapping `H(x)` — I don't know it, and I'm asking the layers to produce it from scratch. Instead, let me hand them `x` for free and ask them only for the difference, `F(x) := H(x) − x`. Then I recover the mapping I wanted by adding the input back:

  y = F(x) + x.

The layers fit `F`; the `+ x` is just a wire from the block's input to its output, an element-wise addition, no parameters. Both formulations can represent the same set of functions — if a stack of nonlinear layers can approximate `H`, it can approximate `H − x` just as well, since the two differ by a fixed term. So I'm not changing capacity. I'm changing the reference point, and with it, what "do nothing" looks like to the solver.

And now watch what happens to the degradation case. If the optimal thing for this block to do is the identity, `H(x) = x`, then the residual it needs is `F(x) = H(x) − x = 0`. To make the block compute the identity, the layers just have to drive `F` to zero — push their weights toward zero. That is the *easiest* target there is, and it's exactly where weight decay was already pushing. The mapping that was the hardest to learn — identity through nonlinear layers — has become the easiest, because in residual coordinates identity *is* the zero map of the trainable part, and zero is the solver's natural rest state. The redundant deep layers can now make themselves harmless almost for free.

It's better than just the identity case, too. In a real net it's unlikely any block needs an exact identity, but many of them probably want something *close* to identity — a small refinement of their input rather than a wholesale new function. In residual form, "close to identity" means "small `F`," and small is easy: the solver is finding a small perturbation referenced to `x`, not building the whole function from nothing. If I were to actually look at the magnitudes of the learned residual responses in such a net, I'd expect them to be generally small — and that smallness would be the evidence that identity is indeed good preconditioning, that the blocks are mostly making gentle corrections. The reformulation conditions the problem toward the regime where the answers live.

So the building block becomes: take `x`, run it through a couple of conv-BN-ReLU layers to compute `F(x)`, and add `x` back. Let me get the wiring exactly right, because the details fight back. First: where does the activation go? If I rectify `F` *before* adding `x` — that is, the last thing in the residual branch is a ReLU — then `F(x) ≥ 0` everywhere, and a non-negative `F` can only ever *increase* `x` component-wise. The block could never make a negative correction, never pull a feature down. That's a crippling restriction; identity-plus-a-one-sided-bump is not the family I want. So the residual branch must be able to output negative values: its last layer is conv-then-BN, *no* final ReLU on the branch. I put the second nonlinearity *after* the addition instead, rectifying the sum `σ(F(x) + x)`. That keeps the usual nonlinearity between blocks while letting `F` be signed. So a two-layer block is `F = W₂ σ(W₁ x)` — conv, BN, ReLU, conv, BN — and then `out = σ(F + x)`.

Second: should the shortcut really be a bare identity wire, or should it carry parameters? There's a tempting alternative on the table. I could make the skip a *gate*: learn, per unit and per input, how much to carry through versus transform — `y = H(x)·T(x) + x·(1 − T(x))`, with `T = σ(W_T x + b_T)` a learned data-dependent gate. That's the gating route, and it does let very deep nets train. But think about what the gate can do that I don't want. The gate has its own weights, so it adds parameters and compute to every block. Worse, because the carry term is `x·(1 − T)` and `T` is learned, the carry path can *close*: if `T → 1` the input contribution `(1 − T) → 0` and the skip is gone — the layer reverts to an ordinary non-residual transform precisely when I might most need the input to flow through untouched. The whole point of my reformulation is that the reference to `x` is *always there*, so that "do nothing" is always one step away. A gate that can shut the reference off undermines that guarantee. So I want the opposite of a gate: a shortcut that is never closed, that always passes all of `x` through, and that carries no parameters at all. A plain identity addition does exactly that. It can never close, it costs nothing to learn, and the residual branch is *always* learning a correction on top of the full input rather than competing with a gate for control of the carry.

The parameter-free part has a second payoff I didn't fully appreciate until now, and it's about *honesty of comparison*. An identity shortcut adds no weights and essentially no computation — just an element-wise add. That means I can take a plain net and its residual counterpart with *identical* depth, width, parameter count, and FLOPs, differing only by these free additions, and compare them head to head. If the residual one trains dramatically better at the same depth and the same budget, the improvement can't be attributed to extra capacity or extra compute — it's purely the reformulation. A gated shortcut, with its parameters, would muddy exactly that comparison. So "parameter-free identity" isn't just convenient; it's what makes the experiment clean and the claim airtight: same model, reparameterized, easier to optimize.

Now a wrinkle the addition forces on me: `F(x) + x` only makes sense if `F(x)` and `x` have the same shape. Inside a stage, where channel count and spatial size are constant, they do, and the shortcut is a literal identity. But between stages I downsample (stride 2) and double the channel count — the VGG complexity rule, halve the map and double the filters so per-layer cost stays put. There, `x` is, say, 56×56×64 and `F(x)` is 28×28×128; I can't add them. I need to bring `x` to `F(x)`'s shape on the skip path. A few options. (A) Keep the shortcut parameter-free: subsample `x` spatially and zero-pad the missing channels — an identity-with-padding. (B) Put a 1×1 convolution with stride 2 on the skip just at these transition points to project `x` into the new shape, identity everywhere else. I can also stress-test the opposite extreme, (C), where every shortcut gets a 1×1 projection, even when shapes already match. But (C) means every single skip now has parameters — thirteen projections in the 34-layer comparison — which adds cost, abandons the parameter-free carry path almost everywhere, and clouds the controlled comparison; the small improvement it might buy is not worth making the skip itself a learned transform. (A) is appealingly free, but the zero-padded channels get *no* residual learning at the transition — they're literally padded with zeros, so those new dimensions start with no input reference at all. (B) is the sweet spot: keep the shortcut a free identity for the overwhelming majority of blocks, and pay for a 1×1 projection `W_s` only where the dimensions actually change, where some linear map is unavoidable anyway:

  y = F(x, {W_i}) + W_s x   (only when dimensions change),  y = F(x) + x   (otherwise).

The projection is an isolated shape-matcher used at a handful of stage boundaries, not the mechanism — the mechanism is still the identity skip carrying the residual everywhere else.

Let me sanity-check the residual function's form before I scale up. Could `F` be a single layer? Then `y = W₁x + x = (W₁ + I)x`, which is just a linear (well, affine-after-BN) layer with a shifted weight — there's nothing for the reformulation to buy, no nonlinearity for the residual to sculpt, and indeed a single-layer residual shows no advantage. So `F` needs at least two layers with a nonlinearity between them, which is what I already have. Good — two 3×3 convs per block it is.

Now the body. I'll follow the VGG skeleton because it's the clean depth-scaling design: a stem that takes the image down to a manageable feature map, then stages of these residual blocks, doubling channels and halving spatial size between stages so each stage's per-layer cost matches, then a head. For the stem on the big benchmark: a single 7×7 conv with 64 filters at stride 2, then a 3×3 max-pool at stride 2 — get the spatial resolution down quickly before spending depth. Then four stages at 64, 128, 256, 512 channels, downsampling at the first block of stages two, three, and four. For the head, I won't carry VGG's fat three-FC stack — that's most of the parameters and it overfits. Global average pooling instead: take the spatial mean of each final feature map, getting one number per channel, and feed that straight to a single linear classifier and softmax. No parameters in the pooling, a built-in structural regularizer, and it sums out position so the prediction is translation-robust. With this I can build an 18-layer and a 34-layer residual net that are, block for block, the residual versions of plain nets at the same depth — the controlled pair I wanted.

There's a gradient-flow story I should acknowledge, because people will reach for it, but I want to be careful not to lean on it as *the* explanation. At the addition itself, before the following ReLU gates anything, `y = F(x) + x` gives the local Jacobian `∂y/∂x = F'(x) + I` — the scalar shorthand is `F' + 1`. So the backward signal has a direct additive term through the skip at each merge, even while the residual branch contributes its own derivative and later nonlinearities still shape the composed Jacobian. That's a real and welcome property. But it is *not* what I'm fixing here, because BN already kept the plain net's gradients healthy — I measured that, the gradients weren't vanishing. The thing I'm actually fixing is the *conditioning*: making the identity/near-identity solution the easy default for the solver to find. The gradient path is a bonus that comes along for free; the load-bearing claim is the reparameterization.

Now the practical wall: I want this to go *very* deep, a hundred-plus layers, and the cost of stacking two-3×3 blocks at 256 or 512 channels is brutal, because a 3×3 conv's cost scales with the square of the channel count — `C_in × C_out × 9 × H × W`. At 512 channels that's enormous, and I can't afford enough of those blocks to reach the depths I want. So I redesign the block for the deep nets to spend its compute more wisely. Instead of two 3×3 convs both running at the full width, use three layers: a 1×1 conv that *reduces* the channel count (say 256 down to 64), then a 3×3 conv that does the spatial work at the *reduced* 64-channel width, then a 1×1 conv that *restores* the width back up (64 to 256). The expensive 3×3 only ever sees the narrow 64-channel tensor; the two 1×1s are cheap reshapes of the channel axis. The whole three-layer "bottleneck" then costs about the same as one two-3×3 basic block, but it's a deeper, more nonlinear function — so the same FLOP budget buys far more depth. The 3×3 is squeezed between two 1×1s like a bottleneck, hence the shape.

And here the identity shortcut earns its keep a second time, decisively. The bottleneck's two ends are both *high*-dimensional (256 in, 256 out); only its middle is narrow. If I'd made the shortcut a learned projection, that projection would be a 256→256 map sitting across those two fat ends — and that single projection would roughly *double* the block's parameters and compute, because it's as expensive as the rest of the block combined. With a parameter-free identity skip across the bottleneck, the shortcut costs nothing, and the block stays as cheap as I designed it to be. So for the deep nets, identity shortcuts aren't just cleaner — they're what makes the deep nets affordable at all. I keep projections only at the stage transitions, exactly as before.

Let me lay out the depths concretely. The basic two-3×3 block, expansion 1, gives the shallower nets: stage block counts [2,2,2,2] for 18 layers, [3,4,6,3] for 34. The bottleneck block, expansion 4 (output width four times the bottleneck width), gives the deep ones: [3,4,6,3] for 50, [3,4,23,3] for 101, [3,8,36,3] for 152. The 152-layer net is eight times deeper than the 19-layer VGG yet, because of the bottleneck economy and the lean GAP head, has *lower* total complexity — around 11 GFLOPs against VGG-19's ~20. Depth went up by 8× and compute went *down*. That's the bottleneck and the parameter-free skips paying off together.

A couple of training details that follow from the design. BN goes right after each convolution and before the activation — that's what let the plain net converge in the first place and what keeps the residual net's signals well-scaled. Weights initialized with the rectifier-aware scheme, trained from scratch, no dropout (BN already regularizes, and I'm relying on the lean architecture and GAP head for the rest). SGD with momentum 0.9, weight decay 1e-4 — note the weight decay is now *helping* the mechanism, since pushing residual weights toward zero is pushing blocks toward the identity. Learning rate 0.1, divided by 10 when the error plateaus. One subtlety I'd anticipate at extreme depth: a 110-plus-layer net started cold at lr 0.1 can be slightly too hot to begin converging, so warm it up at 0.01 until the training error drops below ~80%, then switch to 0.1 and proceed normally — a brief warm-up, not a new schedule.

Let me write the block, and let me make the addition the literal line it is. The basic block: two 3×3 convs each followed by BN, a ReLU after the first conv-BN, then add the (possibly projected) input and ReLU the sum.

```python
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    # 3x3, padded, no bias because normalization follows.
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    # 1x1: a per-pixel linear map of the channel axis.
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
```

And the full network, filling the body of the harness with stages of these blocks, projecting the skip only when stride or channel count changes:

```python
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element list")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18():  return ResNet(BasicBlock, [2, 2, 2, 2])
def resnet34():  return ResNet(BasicBlock, [3, 4, 6, 3])
def resnet50():  return ResNet(Bottleneck, [3, 4, 6, 3])
def resnet101(): return ResNet(Bottleneck, [3, 4, 23, 3])
def resnet152(): return ResNet(Bottleneck, [3, 8, 36, 3])
```

Let me trace the causal chain back so I'm sure it holds together. Depth was the lever for accuracy, so I tried to stack more layers. With init and BN, the deep plain net converges — but it *degrades*: more layers, higher *training* error, which can't be overfitting and isn't vanishing gradients, because BN keeps the signals and gradients healthy and I can measure that. The by-construction argument pinned it down: a deeper net could just copy the shallow net and set the extra layers to identity, so an at-least-as-good solution provably exists, yet the solver can't reach it — an optimization-conditioning failure, traced to the fact that learning an identity (or near-identity) through nonlinear layers is hard while the solver and weight decay both pull toward the zero map. Re-expressing each block to learn the residual `F(x) = H(x) − x` and recover `y = F(x) + x` flips that: identity becomes `F = 0`, the easiest target and the one regularization already favors, and near-identity becomes a small correction referenced to `x`. The shortcut must be a parameter-free identity, not a gate, so the reference is never closed and the plain-vs-residual comparison stays controlled at identical depth/width/params/FLOPs; a 1×1 projection is used only where stride or channels change. The second nonlinearity goes after the add so `F` can be signed. For affordable extreme depth, the bottleneck block runs the costly 3×3 at a reduced width between two cheap 1×1 reductions, and the identity skip across its two high-dimensional ends keeps it free — letting a 152-layer net stay lighter than VGG-19 while being eight times deeper. The whole thing reduces, in each block, to one line: `out += identity`.
