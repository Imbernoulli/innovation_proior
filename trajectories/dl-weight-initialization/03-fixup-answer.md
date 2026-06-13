**Problem (from step 2).** Kaiming beat orthogonal on MobileNetV2 (94.49 vs 93.88) and VGG (73.38 vs 72.83),
but on ResNet-56 the two tied dead at 72.07/72.08. Two schemes that disagree on the per-layer spectrum tying
on the residual net proves per-layer scale is *not* the residual net's binding constraint — **residual
accumulation** is: branch outputs `x_{l+1}=x_l+F_l(x_l)` add, so the main-path variance grows `~(1+L)·Var(x_0)`
over the 27 blocks, which no per-layer second moment can fix.

**Key idea.** Control accumulation directly. Keep the He pass (the second moment that won on VGG/Mobile), then
for ResNets scale each block's *last* conv (`conv2`) by `n_blocks^{-1/2}` so each branch contributes `Θ(1/L)`
and the depth-`L` sum stays `Θ(1)` (the `L^{-1/(2m-2)}` rule, `m=2`); and zero the last BN's scale
(`bn2.weight=0`) so each residual branch starts as the *zero function* and the block starts as a clean
identity — SGD grows the branch from zero instead of unlearning a random one (zero-γ, Goyal et al. 2017).

**Why it works / what is omitted vs the normalization-free recipe.** This is a BN-equipped hybrid, *not* the
normalization-free scheme: BN is kept (frozen graph), and the "start near identity" is bought through BN's own
γ rather than added scalar biases/multipliers — so **no graph change, no added parameters**. Deliberately
omitted: removing BN, inserting learnable scalar biases/multipliers, zero-initializing the classifier — all
would violate the frozen-graph / contract constraints, and the head's `fan_in` He scale is already good.

**Hyperparameters / scaffold edit.** Phase 1: He (`conv fan_out`, `Linear fan_in`, BN `(1,0)`, zero bias).
Phase 2, gated on `arch.startswith('resnet')`: `n_blocks` = count of `BasicBlock`s; `conv2.weight *=
n_blocks**-0.5`; `bn2.weight = 0`. VGG and MobileNetV2 keep plain He (residual phase off).

**What to watch.** The decisive test is ResNet-56: it should finally rise off the 72.07 wash, since this is the
only rung that pays the accumulation tax. VGG (73.38) and MobileNetV2 (94.49) should *hold* — the residual
phase never fires there, so holding the best per-layer result is the success condition.

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
