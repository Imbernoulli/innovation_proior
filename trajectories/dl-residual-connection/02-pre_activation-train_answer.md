The gated block landed exactly where its low bandwidth said it would. On ResNet-20/CIFAR-10 it reached $92.96\%$ — a strong, well-tuned vanilla ResNet-20 number, the tell that the scalars settled near 1 and there was little structural slack at depth 20 to exploit. But the deep CIFAR-100 settings are where the gate ran out of road: $71.98\%$ on ResNet-56 and $73.46\%$ on ResNet-110. Read together, those two are the diagnosis. The *very deep* net edges the merely-deep one by barely $1.5$ points, and $73.46\%$ at 110 layers is not a number that says a net is using its depth. A single per-block scalar rescales how *loud* each block is, but it cannot change how cleanly signal and gradient pass *through* the stack — and at 110 layers, "through the stack" is the binding constraint. The lever I deliberately held fixed last rung was the block's internal *wiring*, and that is what I pull now.

Look hard at the ordering the default and the gated block both keep. They are *post-activation*: $\mathrm{out} = \mathrm{ReLU}(\mathrm{BN}(\mathrm{conv1}(x)))$, then $\mathrm{out} = \mathrm{BN}(\mathrm{conv2}(\mathrm{out}))$, then the addition, then a final ReLU on the sum. Two things on the identity path become alarming when I imagine stacking this 110 deep. The final ReLU sits *after* the addition, so every block's output is forced non-negative before the next block and the next shortcut ever see it. And within a stage the shortcut is a bare identity, so the highway from the input of block $\ell$ to its output is $x \mapsto \mathrm{ReLU}(x + F(x))$; chain that across the stage and the clean additive identity I want, $x_{\mathrm{out}} = x_{\mathrm{in}} + \sum_\ell F_\ell$, is interrupted at *every* block by a rectifier. The identity path is not actually clean — it is a sequence of additions each clamped through a ReLU. At depth 20 that barely matters; at depth 110 those repeated clamps are precisely what throttles how a perturbation at the top reaches the bottom and how the input's information survives forward to the top.

I propose the **pre-activation block**: reorder so each conv is preceded by its own BN and ReLU (BN-ReLU-Conv, twice) rather than followed by them, and delete the activation after the addition. The branch becomes $F(x) = \mathrm{conv2}\big(\mathrm{ReLU}(\mathrm{BN}(\mathrm{conv1}(\mathrm{ReLU}(\mathrm{BN}(x)))))\big)$ and the block output is simply
$$H = \mathrm{shortcut}(x) + \alpha \cdot F(x),$$
with nothing rectifying the sum. Trace the through-line again: the branch consumes a *pre-activated copy* of $x$ — it applies BN then ReLU at its own entrance — does its work, and the result is added to the shortcut; the next block applies *its own* entrance BN-ReLU to whatever it receives. So the signal traveling block-to-block along the shortcut is the raw additive stream, never clamped; each block's nonlinearity acts only on the copy it pulls into its branch, not on the highway. Across a stage the identity path is now the clean $x_{\mathrm{out}} = x_{\mathrm{in}} + \sum_\ell F_\ell$ — the rectifiers have all moved off the through-line and into the branches where they belong.

This is not a cosmetic shuffle, and it matters for two concrete reasons. Backward: with no post-addition ReLU, the stage Jacobian is exactly $I + \sum_\ell \partial F_\ell / \partial x$, an identity plus corrections, so the gradient at the top reaches the bottom undiminished along the $I$ term no matter how small the $\partial F_\ell$ factors get. In the post-activation block that $I$ is gated by a ReLU's 0/1 mask at every block, repeatedly multiplied by an indicator that can zero it. Forward: putting BN at each branch entrance means the input to every convolution is normalized — strictly better conditioned than a conv consuming a raw, possibly large additive sum. Both effects grow with depth, which is exactly why I expect this to help ResNet-110 most, ResNet-56 next, and ResNet-20 barely.

Then the substrate-specific care, because the task's fill is not the textbook block and I have to derive the *actual* one. Two deviations, both deliberate. First, the dimension-changing shortcut. In a fully pure identity-mapping block the shortcut would also be a clean parameter-free path, but the stride-2, channel-doubling transitions — the first block of stages 2 and 3 — cannot be a bare identity; the shapes do not match. I resolve that with a *projection*: a 1×1 stride-2 conv followed by BN, applied to the **raw input $x$**, not to the pre-activated copy. So the through-line is the clean unobstructed identity only on the within-stage blocks; at the two transition blocks per net the shortcut is a small Conv-BN projection. That is the right choice here — BN on the projection keeps those two transition points well-scaled, and the great majority of blocks still enjoy the clean additive highway. The branch's entrance pre-activation is computed once and shared: the branch reads the pre-activated $x$, the projection reads the raw $x$.

Second, the deviation the substrate's depth demands: a fixed-small **residual scale**. Even with the clean highway, a freshly initialized 110-layer net sums 54 branch outputs onto the identity stream, and at init each $F_\ell$ is a random small-but-nonzero perturbation; 54 of them summed can push the stream's variance up enough to make the first epochs jittery before the convs settle. The fix that respects the frozen recipe is to multiply each branch by a small learnable scalar $\alpha$ initialized to **0.1**, so each branch starts at one-tenth strength: a deep net begins as roughly the identity plus a gentle residual, the early epochs are dominated by the clean path, and SGD is free to grow $\alpha$ where each block wants it. Having learned the lesson from the gated rung, I do *not* init this at zero — that would waste the early budget on an effectively-shallow net against a fixed 200-epoch schedule. Note the two scalars look alike and mean opposite things: the gated $\alpha = 1$ says "behave like the baseline, adjust"; the pre-activation $\alpha = 0.1$ says "start near identity, grow." Here the scale's job is to *stabilize* the much deeper pre-activation stack at init, not to re-weight around a baseline.

So the edit relative to the gated block is genuinely structural, not a rescaling: move both BNs and ReLUs to precede their convs, delete the post-addition ReLU so the through-line is a pure sum, route the dimension-change shortcut through a Conv-BN projection of the raw input, and scale the branch by a 0.1-initialized learnable $\alpha$. The two convs, kernel sizes, dimension-matching logic, and parameter budget are otherwise the same; what changed is the *order*, which is the thing the gated numbers said was limiting the deep nets.

```python
# EDITABLE region of custom_residual.py (lines 31-61) -- step 2: pre-activation (BN-ReLU-Conv) + 0.1 scale
class CustomBlock(nn.Module):
    """Pre-activation residual block (He et al., 2016 v2).

    Uses BN-ReLU-Conv order for cleaner gradient flow.
    Both main branch and shortcut share the same pre-activation.
    Residual scaling (alpha, init=0.1) stabilizes very deep networks.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out)))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return shortcut + self.alpha * out
```
