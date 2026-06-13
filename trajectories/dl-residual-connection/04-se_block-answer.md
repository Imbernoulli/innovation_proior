**Problem (from step 3).** Stochastic depth is the strongest rung — ResNet-110/CIFAR-100 75.70,
ResNet-56/CIFAR-100 74.56 — but it cost the shallow case: ResNet-20/CIFAR-10 fell to 91.37, the lowest on
the ladder, because dropping branches removes capacity from a net already near its CIFAR-10 ceiling. Every
rung so far pulls the *same* lever — the residual flow/depth axis (scale the branch, reorder the path, drop
the branch). None asks *which features* the branch should emphasize for a given image. A convolution's
channel mixing is baked in at training time, identical for every input, and locally decided. That axis is
untouched and is orthogonal to depth — so it can help the deep nets without the shallow tax dropping pays.

**Key idea (Squeeze-and-Excitation, Hu et al. 2017, arXiv:1709.01507).** A lightweight, input-dependent,
global-context channel gate on the branch output before the addition. *Squeeze:* global average pool the
`C×H×W` branch map to a length-`C` descriptor (full-image receptive field). *Excite:* a bottleneck MLP
`C → C/r → C` with ReLU then **sigmoid** (gates in `(0,1)`, non-exclusive — several channels may fire at
once). *Scale:* multiply each branch channel by its gate. `H = ReLU(shortcut(x) + SE(F(x)))`.

**Why it works.** The gate is per-image and uses global context the local conv lacks, so it can amplify the
discriminative channels and attenuate the irrelevant ones for *this* input — extra capacity orthogonal to
depth, exactly where a 100-class fit benefits. It acts only on the branch, never the shortcut, so the
identity highway stays open (the principle every rung kept). Unlike stochastic depth it adds capacity
without removing any, so the shallow case should recover.

**Substrate-specific care (faithful fill).** Post-activation block kept at full depth (no dropping, no
reordering, no branch scalar). SE inserted on `out` between the second BN and the addition. Reduction `r =
16`, but the bottleneck width is **floored**: `mid = max(planes // 16, 4)` — the CIFAR stages are 16/32/64
wide, and a bare `16/16 = 1` squeeze in stage 1 would be degenerate, so at least 4 hidden units are kept.
FC layers carry biases; ReLU hidden, sigmoid output; gate reshaped to `(B,C,1,1)` and broadcast. Cost is
`2·planes²/r` params on length-`C` vectors — negligible.

**Hyperparameters / scaffold edit.** `expansion = 1`; SE module `AdaptiveAvgPool2d(1) → Flatten →
Linear(planes, mid) → ReLU → Linear(mid, planes) → Sigmoid`, `mid = max(planes//16, 4)`; recalibrate `out`
before adding the dimension-matching shortcut, final ReLU after the add.

**The bar to clear (vs the strongest baseline).** Stochastic depth: 91.37 / 74.56 / 75.70. For SE to be a
real improvement and not a sideways trade: (1) ResNet-20/CIFAR-10 must *recover* well above 91.37 (toward
the gated 92.96) — if it does not, the channel gate buys no real capacity and the orthogonal-axis premise is
wrong; (2) ResNet-56 and ResNet-110 on CIFAR-100 must at least *hold* 74.56 and 75.70 and ideally edge them.
If SE underperforms on the deep nets, the CIFAR-100 limit was depth-regularization (which SE lacks) and the
endpoint should *combine* SE with dropping rather than substitute it. The endpoint stands on clearing the
shallow case *and* holding the deep cases at once — the net-positive-across-the-sweep no single flow-axis
rung managed.

```python
# EDITABLE region of custom_residual.py (lines 31-61) -- finale: Squeeze-and-Excitation channel attention
class CustomBlock(nn.Module):
    """Basic residual block with Squeeze-and-Excitation attention."""
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
        # Squeeze-and-Excitation
        reduction = 16
        mid = max(planes // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(planes, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SE channel attention
        w = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out = out * w
        out += self.shortcut(x)
        return F.relu(out)
```
