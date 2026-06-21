The plain post-activation block that the backbone repeats — two 3×3 conv–BN layers with a ReLU between them, the shortcut added, a final ReLU on the sum, $H = \mathrm{ReLU}(F(x) + \mathrm{shortcut}(x))$ — commits to one fixed mixing rule: the branch output $F(x)$ is added at full weight, $1\cdot F(x)$, in the first block and the last, at depth 20 and at depth 110 alike. But the right contribution of a block is surely not constant. Early blocks, building the low-level features every later block depends on, may want to write strongly; deep blocks, refining an already-rich representation, may want only a small correction. With the weight pinned at 1, the only way the network can dial a block down is to drive all of $F$'s conv weights toward zero — fighting the very conditioning the residual reformulation was meant to relieve, and doing it with the full convolutional machinery rather than one knob. Before reaching for anything structural, I want the *minimal* deviation from the default that sits exactly on the axis I suspect is wrong, because it calibrates everything after it.

I propose a **gated residual**: a single learnable scalar $\alpha$ on the residual branch, multiplying $F(x)$ after its final BN and before the addition,
$$H = \mathrm{ReLU}\big(\mathrm{shortcut}(x) + \alpha \cdot F(x)\big),$$
with one parameter per block, broadcast across channels and space, trained by SGD alongside the convs. The shortcut stays *un-gated*: the entire value of the identity path is that it is always open, carrying signal and gradient untouched, and gating it would reintroduce the Highway failure where a drifting gate closes the highway exactly where a deep net needs it. The scalar gives each block one extra degree of freedom — how loud to be — at negligible cost, letting the optimizer re-weight a block's contribution instead of forcing it to reshape $F$ itself.

The decision that actually defines this rung is the *initialization* of $\alpha$, and it is deliberately not the one the obvious reference would push toward. The natural place to look for "scale the residual by a learnable scalar" is the lineage that initializes that scalar to **zero** — start every block as a pure identity, let the gates open during training, so a freshly-initialized deep net begins life perfectly signal-preserving and the input–output Jacobian is exactly the identity. That story is about *trainability at extreme depth* in un-normalized networks, and it is real. But it does not fit this substrate, and the place it breaks is load-bearing. First, this block is *post-activation*: there is a ReLU after the addition and BN inside the branch, so at $\alpha = 0$ the block is not the identity at all — it is $\mathrm{ReLU}(\mathrm{shortcut}(x))$, which on the bare-identity shortcuts is $\mathrm{ReLU}(x)$, the identity only because the input happens to be non-negative. Second, and decisively, the substrate already *has* trainability solved: BN after every conv keeps forward and backward signals healthy at depth 110, and the schedule is a fixed cosine over 200 epochs with no warm-up to remove. The disease zero-init cures is simply not present here. If I zero-init $\alpha$, I instead throw away the first stretch of training — every block contributes nothing while the gates crawl off zero, the net is effectively shallow for as long as that takes, and against a fixed budget that is wasted depth, not bought depth.

So I init the gate at **one**. At $\alpha = 1$ the block is bit-for-bit the default scaffold block, $\mathrm{ReLU}(\mathrm{shortcut}(x) + 1\cdot F(x)) = \mathrm{ReLU}(\mathrm{shortcut}(x) + F(x))$. The network begins training as the proven post-activation ResNet, full depth from step zero, and the gate is a pure *correction* the optimizer may apply: pull a block below 1 if it over-writes, push it above 1 if it wants to write louder than the default allows. This makes the floor of the rung the baseline itself — the worst case is "the gates stay near 1 and I recover the baseline," and the upside is "some blocks find they should be quieter or louder." Initializing at 0 would make the floor strictly worse and bet the rung on the gates lifting fast enough; given an already-trainable substrate and a fixed budget, 1 is the only defensible choice, and it is the single decision that separates this rung from the zero-init lineage it superficially resembles.

The implementation is one `nn.Parameter` of shape $(1,)$ initialized to ones, registered so SGD trains it and weight decay touches it — a mild, benign pull toward zero that just gives the network a standing incentive to quiet blocks it isn't using, the same incentive already acting on the conv weights. It is a single per-block scalar, not a per-channel vector: the hypothesis is the coarse one, "how loud is this *block*," and a per-channel gate would start to overlap with channel-recalibration ideas I want to isolate to a later rung. Everything else — the two convs, the inter-conv ReLU, the dimension-matching 1×1 shortcut, the final ReLU — is the scaffold default, untouched. The edit is exactly: declare `self.alpha = nn.Parameter(torch.ones(1))`, and change `out += self.shortcut(x)` to `out = self.shortcut(x) + self.alpha * out`. It is the smallest meaningful change to the default, and that is the point: it sits on the residual-magnitude axis and on nothing else, so whatever it does is a clean signal about whether the fixed block was getting the *amount* of residual wrong.

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
