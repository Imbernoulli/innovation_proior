**Problem (from step 2).** Pre-activation fixed gradient *flow* — ResNet-110/CIFAR-100 took the largest
lift (73.46 → 74.08) and finally beat the 56-layer net — but ResNet-56/CIFAR-100 dipped (71.98 → 71.78)
and ResNet-20/CIFAR-10 slipped (92.96 → 92.62). Flow is no longer the limit; the deep nets still do not
*use or regularize* their depth (74.08 at 110 layers on 100 classes is low). A reordering cannot supply
that. The new tension: I want the *expressiveness* of a deep net at test but the *optimization and
regularization* of a shorter net during training — two opposite things from one network.

**Key idea.** Make blocks effectively *short while training, deep at test*. Gate each branch with a
per-mini-batch Bernoulli `b`: `H = ReLU(b · F(x) + shortcut(x))`. With `b = 0` a within-stage block is an
*exact identity* (`ReLU(x) = x`, the input being non-negative) — free, clean, removed for that step. Each
of `L` blocks independently on/off makes one weight set define `2^L` sub-networks of varying depth; each
step samples one, and test-time combination is an implicit ensemble over depth.

**Survival schedule.** Early blocks feed every later block, so survival decreases with depth (linear):
`p_ℓ = 1 − (ℓ/L)(1 − p_L)`, `p_L = 0.5`. Then `E(L̃) = Σ p_ℓ = (3L − 1)/4 ≈ 3L/4` — ResNet-110's 54 blocks
train ~40 deep, deploy 54, ~25% compute saved, largest effect at the largest `L`.

**Test rule.** All branches active, each scaled by its survival probability so the expected contribution
matches training: `H = ReLU(p_ℓ · F(x) + shortcut(x))`.

**Substrate-specific deviations (the actual fill).** (1) Block-dropping is applied to the **post-activation**
block (reverting step 2's pre-activation + 0.1 scale), because `b = 0` gives a clean exact identity only
there; this rung trades the flow fix for the ensemble fix. (2) The constructor only sees
`(in_planes, planes, stride)`, so the block self-counts via a class-level `_block_counter`, reset at the
first stage-1 block (`in_planes == 16 and planes == 16 and stride == 1`); `L` is read from the class counter
at forward (true total). (3) A dropped *transition* block returns `ReLU(shortcut(x))` with `shortcut` the
Conv-BN projection — not a literal identity, unavoidable on the two dimension-changing blocks per net; the
clean-identity argument holds for all other blocks.

**Hyperparameters / scaffold edit.** `expansion = 1`; `_p_last = 0.5`; class counter for `block_idx`/`L`;
training drops the branch with prob `1 − p` per minibatch (`torch.rand(1) < p`); eval scales the branch by
`p`. No new learnable parameters.

**What to watch.** The benefit grows with `L`, so expect the opposite depth-sort to step 2: ResNet-110 /
CIFAR-100 should clear 74.08 by the largest margin and ResNet-56 should recover past 71.78, while
ResNet-20 / CIFAR-10 is the honest cost — few blocks to drop, near its CIFAR-10 ceiling — and may fall below
both prior rungs. If stochastic depth does *not* sort by depth, the remaining gap was not a
depth-regularization problem.

```python
# EDITABLE region of custom_residual.py (lines 31-61) -- step 3: stochastic depth, linear decay, p_L=0.5
class CustomBlock(nn.Module):
    """Residual block with stochastic depth (Huang et al., 2016).

    During training, each block's residual branch is randomly dropped with
    probability (1 - survival_prob). The survival probability linearly decays
    from 1.0 (first block) to p_L (last block). At test time, the residual
    output is deterministically scaled by the survival probability.
    """
    expansion = 1
    _block_counter = 0
    _p_last = 0.5  # survival prob of the deepest block

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
        # Reset counter when first block of a new model is created
        # (CIFAR ResNets always start layer1 with in_planes=16, planes=16, stride=1)
        if in_planes == 16 and planes == 16 and stride == 1:
            CustomBlock._block_counter = 0
        CustomBlock._block_counter += 1
        self.block_idx = CustomBlock._block_counter

    def forward(self, x):
        shortcut = self.shortcut(x)
        L = CustomBlock._block_counter
        p = 1.0 - (self.block_idx / L) * (1.0 - CustomBlock._p_last)
        if self.training:
            if torch.rand(1).item() < p:
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + shortcut)
            else:
                return F.relu(shortcut)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(p * out + shortcut)
```
