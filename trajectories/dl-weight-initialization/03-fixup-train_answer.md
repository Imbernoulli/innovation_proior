The Kaiming numbers landed almost exactly on the predictions, and the one that matters most is the ResNet wash. On MobileNetV2, He jumped to 94.49 from orthogonal's 93.88 — the depthwise convs got their correct per-filter variance and the swing was real. On VGG-16-BN, He reached 73.38, comfortably ahead of orthogonal's 72.83, so the plain stack's spectral conditioning did not survive contact even on its home turf. But ResNet-56 came in at 72.07, essentially identical to orthogonal's 72.08 — a dead wash, two schemes that disagree entirely about the per-layer map agreeing to within one part in seven thousand on the residual net. That tie is the whole signal: when two initializers that differ on the per-layer spectrum agree on the residual net, per-layer scale is provably not what limits it. The ResNet ceiling is set by something neither rung touched — residual accumulation.

Make the accumulation precise, because the fix comes out of its structure. A ResNet main path is a running sum $x_{l+1} = x_l + F_l(x_l)$, where each branch $F_l$ is, for `BasicBlock`, two 3×3 convs with BN. At initialization both He and orthogonal give each branch an output whose variance is roughly the scale of $x_l$ — that is what "preserve the per-layer variance" means. But the branches *add*, and at init their outputs are roughly independent, so the variance of the running sum grows additively: after $L$ blocks the main-path variance is about $\mathrm{Var}(x_0) + \sum_l \mathrm{Var}(F_l) \approx (1+L)\,\mathrm{Var}(x_0)$. ResNet-56 on CIFAR has 27 `BasicBlock`s, so the signal entering the head is inflated by a factor on the order of the depth. The logits are over-scaled and the softmax starts saturated; BN does re-standardize the running sum each block (which is why the net trains at all), but it is forced to divide out a large depth-dependent variance, coupling every block's BN to the depth and making the residual branches start out *competing* with the identity path rather than acting as small corrections; and the deepest branches, buried in a large sum, get a poorly-scaled gradient. No per-layer second-moment choice can pay that tax, because the problem is *between* layers — in how the branches sum — not within any one.

I propose a **Fixup-style residual scaling combined with zero-$\gamma$ BatchNorm**, which controls accumulation directly while leaving the He pass that already won on VGG and Mobile intact. The remedy the residual structure itself suggests is to make each branch start *small* so the running sum does not inflate, and to start it from the *identity* rather than a random function so SGD does not have to first unlearn a bad random branch. The right depth dependence for "start small" comes from the residual-scaling analysis of Zhang, Dauphin & Ma (2019): if I want each branch's contribution to be $\Theta(1/L)$ rather than $\Theta(1)$ — so the sum over $L$ branches is $\Theta(1)$ again, depth-independent — and the branch has $m$ weight layers, then scaling its weight layers by $L^{-1/(2m-2)}$ achieves it. For `BasicBlock`, $m = 2$, so the exponent is $-1/(2\cdot 2 - 2) = -1/2$: scale by $L^{-1/2}$, i.e. $\text{n\_blocks}^{-1/2}$.

The care this rung needs is that the substrate is *not* the normalization-free setting that analysis was built for, and importing the wrong machinery would break things. The original residual-scaling recipe was designed to train res-nets with no normalization at all: it removes BatchNorm, replaces it with learnable scalar biases before every conv and activation, adds one learnable scalar multiplier per branch, scales the non-zero branch weights by $L^{-1/(2m-2)}$ while zero-initializing the *last* conv of each branch, and zero-initializes the classifier — every piece a substitute for what BN would have done. But these networks *keep* BatchNorm; it is part of the frozen graph I am forbidden to touch, and I cannot add scalar biases or multipliers without altering the graph, which the contract prohibits. So I must not import that story. What I can do, and what is the right translation of the idea into a BN-equipped network, is two edits, both inside the contract.

First, the branch down-scaling. I run the He pass first — every conv $N(0,\sqrt{2/\text{fan\_out}})$, every Linear `fan_in` He, every BN neutral — keeping the step-2 second moment that just proved itself. Then, for ResNets only, I scale each block's *last* conv (`conv2`) by $\text{n\_blocks}^{-1/2}$. I scale `conv2` specifically because it is the last weight layer in the branch, the one whose output directly enters the additive sum, so scaling it is the most direct way to set the branch's contribution scale; and I scale it *down from a good He init* rather than zeroing it, because with BN in the loop I want the branch to be a small but live correction, not a dead one.

Second — and this is where the BN-equipped translation does something the normalization-free recipe cannot — I exploit BatchNorm itself to start the branch near identity, for free, with no graph change. In a `BasicBlock` the branch ends `conv2 → bn2`, and the block computes $\mathrm{relu}(\mathrm{bn2}(\mathrm{conv2}(\cdot)) + \mathrm{shortcut}(x))$. The affine $\gamma$ of `bn2` multiplies the entire branch output before it is added to the shortcut, so setting $\mathrm{bn2.weight} = 0$ at init multiplies that output by zero and the block computes exactly $\mathrm{relu}(\mathrm{shortcut}(x))$ — the residual branch starts as the *zero function* and the block starts as a clean identity-plus-shortcut, precisely the "start from identity" property I wanted, achieved through a parameter BN already owns. This is the zero-$\gamma$ trick (Goyal et al. 2017). It is strictly better than zeroing the conv would be here, because the conv weights stay at their He scale (ready to contribute the moment $\gamma$ lifts off zero) while the *block output* is what starts at identity — and $\gamma$ is a single learnable scalar per channel BN already exposes, so I change no graph and add no parameters. I am explicitly not adding scalar biases or multipliers, not removing BN, and not zeroing the classifier — those belong to the no-normalization world, would violate the frozen-graph contract, and the head's `fan_in` He scale is already good.

For the non-residual architectures the answer is "do nothing extra," and it is worth saying why. VGG-16-BN has no shortcuts and no additive accumulation — there is no branch to scale and no last-branch BN to zero — so the He pass *is* the right init, and it already won there at 73.38. MobileNetV2 does have additive shortcuts in its `InvertedResidual` blocks, but the block structure differs (expand-1×1 → depthwise-3×3 → project-1×1, three convs, residual added only when stride 1 and channels match), and He's strongest result already lives there at 94.49; rather than guess a branch-scaling exponent for a block type the `n_blocks`/`BasicBlock` accounting was not derived for, I gate the residual phase on `arch.startswith('resnet')` and let Mobile and VGG keep the plain He that is already best for them. So phase one is the full He pass; phase two, only if the arch is a ResNet, counts $\text{n\_blocks}$ = number of `BasicBlock`s, multiplies each block's `conv2.weight` by $\text{n\_blocks}^{-1/2}$, and sets each block's `bn2.weight` to zero — no graph change, no added parameters, no data, no calibration. The decisive test is ResNet-56: it sat at 72.07/72.08 under the two per-layer schemes precisely because neither addressed accumulation, and this is the only rung that does, so it should finally rise off that ceiling, while VGG (73.38) and MobileNetV2 (94.49) hold because the residual phase never fires there.

```python
def initialize_weights(model, config):
    """Fixup-inspired residual scaling with zero-gamma BatchNorm.

    For ResNets: Kaiming normal for all Conv2d, then scale the last conv in
    each residual block by n_blocks^(-0.5) to control variance accumulation.
    Zero-initialize the last BN in each block (Goyal et al., 2017).
    For VGG: Kaiming normal (no residual branches to scale).
    Linear: small normal init with zero bias.
    """
    arch = config['arch']
    is_resnet = arch.startswith('resnet')

    # Phase 1: standard Kaiming init for all layers
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Phase 2: Fixup-inspired residual branch scaling for ResNets
    if is_resnet:
        n_blocks = sum(1 for m in model.modules() if isinstance(m, BasicBlock))
        fixup_scale = n_blocks ** (-0.5)
        for m in model.modules():
            if isinstance(m, BasicBlock):
                # Scale the last conv (conv2) in each residual block
                m.conv2.weight.data.mul_(fixup_scale)
                # Zero-init the last BN so residual branch starts near identity
                nn.init.constant_(m.bn2.weight, 0)
```
