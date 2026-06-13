**Problem.** The plain post-activation block adds its residual branch at a fixed weight of 1, identical in
every block and at every depth. But the right contribution of a block is not constant — early blocks may
want to write strongly, deep blocks only a small correction — and with the weight pinned the network can
only dial a block down by fighting all of `F`'s conv weights toward zero. The minimal question: give each
block one learnable scalar on its residual branch and let SGD choose how loud each block is.

**Key idea.** A single learnable scalar `alpha` multiplies the residual branch before the addition,
`H = ReLU(shortcut(x) + alpha · F(x))`; the shortcut stays un-gated so the identity highway is always open.
One parameter per block, broadcast over channels and space.

**Why (and the decision that defines this rung).** Initialize `alpha = 1`, **not** 0. The zero-init
lineage (start every block as a pure identity, gates lift off during training) cures un-normalized
extreme-depth trainability — a disease this substrate does not have: BN after every conv already keeps
depth-110 signals healthy, and there is no warm-up to remove. Worse, this block is *post-activation* (a
ReLU after the add, BN inside the branch), so `alpha = 0` is not even a clean identity, and against a fixed
200-epoch budget a zero start just wastes the early epochs as an effectively-shallow net. Initializing at 1
makes the block bit-for-bit the proven baseline at step zero, so the floor is the baseline and the gate is
a pure correction: pull a block below 1 if it over-writes, push above 1 if it wants to write louder.

**Hyperparameters / scaffold edit.** Relative to the default: add `self.alpha = nn.Parameter(torch.ones(1))`
and replace `out += self.shortcut(x)` with `out = self.shortcut(x) + self.alpha * out`. `expansion = 1`;
the two 3×3 conv–BN layers, the inter-conv ReLU, the dimension-matching 1×1 shortcut, and the final ReLU
are the scaffold default, untouched. The gate is trained by SGD (weight decay touches it, a benign small
pull toward quieting unused blocks).

**What to watch.** Expect close to a strong vanilla ResNet on all three settings, best on ResNet-20 /
CIFAR-10 where "match the baseline" is already high; the deep CIFAR-100 nets should show the most
unexploited headroom, since a per-block scalar rescales a block but cannot change how cleanly gradients and
features pass *through* the identity path — which points the next rung at the block's BN/ReLU ordering.

```python
# EDITABLE region of custom_residual.py (lines 31-61) -- step 1: learnable residual gate, alpha init=1.0
class CustomBlock(nn.Module):
    """Residual block with learnable residual gate (scalar scaling).

    A learnable parameter alpha scales the residual branch output before
    adding to the shortcut: out = shortcut(x) + alpha * F(x).
    Initialized at alpha=1.0 (standard residual behavior).
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        # Learnable residual gate initialized at 1.0
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + self.alpha * out
        return F.relu(out)
```
