**Problem (from step 1).** The learnable gate matched a strong vanilla ResNet on ResNet-20/CIFAR-10
(92.96) but stalled on the deep CIFAR-100 nets (71.98 at 56, 73.46 at 110) — the very deep net barely
edged the merely-deep one. A per-block scalar rescales how *loud* a block is but cannot change how cleanly
signal and gradient pass *through* the stack, and at 110 layers "through the stack" is the binding
constraint. The lever left unpulled is the block's internal *wiring*: a post-activation block puts a ReLU
*after* the addition, so the identity highway is clamped by a rectifier at every block.

**Key idea.** Reorder to **pre-activation**: each conv is preceded by its own BN and ReLU (BN-ReLU-Conv,
twice), and there is **no activation after the addition**. The branch reads a pre-activated copy of `x`; the
block output is `shortcut(x) + alpha · F(x)`. Across a stage the through-line becomes a pure sum
`x_out = x_in + Σ F_ℓ` — the rectifiers move off the highway and into the branches.

**Why it works.** Backward, with no post-add ReLU the stage Jacobian is `I + Σ ∂F_ℓ`, so the gradient
reaches the bottom undiminished along `I` instead of being gated by a 0/1 ReLU mask at every block.
Forward, BN at each branch entrance normalizes every conv's input. Both effects grow with depth, so the
payoff should be largest at ResNet-110, smaller at 56, negligible at 20.

**Substrate-specific deviations (the actual fill, not the textbook block).** (1) The dimension-changing
transition blocks cannot be a bare identity; the shortcut there is a 1×1 stride-2 **Conv-BN projection of
the raw input `x`** (only two such blocks per net, so most blocks keep the clean additive highway).
(2) A fixed-small **residual scale** `alpha = nn.Parameter(0.1)` multiplies the branch: a freshly
initialized 110-layer net sums 54 random branches onto the identity, so starting each branch at one-tenth
strength makes the net begin near the identity and grow the residual in. Unlike the zero-init lineage, 0.1
(not 0) avoids wasting the early budget as an effectively-shallow net while still soft-starting the deep
stack.

**Hyperparameters / scaffold edit.** `expansion = 1`; BN-ReLU-Conv ordering, shared pre-activation `pre =
ReLU(bn1(x))` feeding `conv1`; branch `conv2(ReLU(bn2(conv1(pre))))`; `downsample` = Conv(1×1, stride)-BN on
raw `x` when `stride != 1` or channels change; `alpha` init 0.1, learnable; output `shortcut + alpha · out`,
no final ReLU.

**What to watch.** Expect ResNet-20/CIFAR-10 around the gated 92.96 (possibly a hair under, from the 0.1
warm-up); ResNet-110 should clear 73.46 by the largest margin and ideally pull decisively above the
ResNet-56 number; ResNet-56 should improve by less. If instead all three move together, the limit was
magnitude not wiring and the reordering story is wrong. Residual flow is fixed, but depth is still not
*regularized* — pointing the next rung at implicit regularization across depth.

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
