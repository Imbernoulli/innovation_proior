The pre-activation reordering did almost exactly what I predicted on the deep end: ResNet-110/CIFAR-100 went from the gated $73.46\%$ to $74.08\%$ — the largest absolute lift of the three settings — and the very deep net finally pulled clearly above the merely-deep one, confirming that the binding constraint at 110 layers really was the obstructed identity path. But the story is not clean. ResNet-20/CIFAR-10 slipped to $92.62\%$ (the small shallow cost of the 0.1 warm-up), and ResNet-56/CIFAR-100 actually went *down*, $71.98 \to 71.78\%$. That ResNet-56 dip is the signal that matters. Flow is no longer the limit, and $74.08\%$ at 110 layers on a 100-class problem is still not a number that says the net is using its depth. What is left is that the deep net does not *use or regularize* its depth, and a reordering cannot supply that.

State the new tension precisely, because the contradiction is the whole idea. Depth helps — pre-activation just proved 110 beats 56 once the gradients flow. But depth also hurts on CIFAR-100: more blocks means more parameters fitting a 50k-image, 100-class set, longer chains even with a clean highway, and more opportunity for late blocks to overfit or sit idle passing their input along — the identity path makes "learn nothing" a comfortable equilibrium. So I want two opposite things from one network: the *expressiveness* of a deep net at test time, and the *optimization and regularization* of a shorter net during training. But those properties are needed in different *phases* — capacity at test, easy optimization during training. What if the network could be effectively short while I train it and deep when I deploy it?

I propose **stochastic depth**: drop whole residual branches at random during training. A block computes $H = \mathrm{ReLU}(F(x) + \mathrm{shortcut}(x))$, and the shortcut already carries the input across, so if I gate the branch with a per-mini-batch Bernoulli $b \in \{0,1\}$,
$$H = \mathrm{ReLU}\big(b \cdot F(x) + \mathrm{shortcut}(x)\big),$$
then $b = 1$ is exactly the block I have and $b = 0$ makes it $\mathrm{ReLU}(\mathrm{shortcut}(x))$. On the within-stage blocks the shortcut is a bare identity and the input is non-negative (it is the output of a prior ReLU), so $\mathrm{ReLU}(x) = x$ and a dropped block is an *exact* identity — signal and gradient flow through it untouched, with no forward or backward compute for the dropped branch. A dropped block is free and clean. This delivers the two missing things at once. The *effective* training depth shrinks, so the gradient and forward chains are shorter during training even though the deployed net is full depth. And because each of the $L$ blocks is independently on or off, one weight set now defines $2^L$ sub-networks of *varying depth*; each minibatch samples and updates one, and at test time their combination is an implicit ensemble over depth-diverse members — regularization that gets *stronger* with depth, which is the right direction given the deep nets are where I am stuck.

The first design decision is a deliberate reversal I want to be honest about. The instinct is to stack this on the pre-activation block, but the dropping argument depends on a clean *exact identity* when the branch is off, and that is cleanest on the plain post-activation block: there $b = 0$ gives $\mathrm{ReLU}(\mathrm{shortcut}(x))$, which on the bare-identity shortcuts is exactly $x$. So for this rung I drop the pre-activation reordering and the 0.1 scale and go back to the proven post-activation block as the thing being gated. This is not throwing away the last win — stochastic depth attacks a *different axis*: pre-activation fixed how gradients flow through a full-depth net (mattering most at 110), while stochastic depth makes the net train shallow and regularize like an ensemble, a property of the dropping schedule, not the activation order. The two do not compose for free (the residual scale would interact with the survival scaling at test), so I take the cleanest, most-studied form — block-dropping on the vanilla block — and let the numbers say which axis the deep nets cared about more.

The survival schedule is the second decision, and uniform is wrong. Early blocks extract the low-level features every later block builds on; drop one and I corrupt the foundation the rest of the net depends on for that step. A late block's transformation is specialized and less universally relied on. So survival should *decrease* with depth, and the gentlest such schedule is a straight line, anchored so the first block essentially always survives and the last survives with probability $p_L$:
$$p_\ell = 1 - \frac{\ell}{L}\,(1 - p_L).$$
One knob, $p_L$, to which training is insensitive, so I fix $p_L = 0.5$ — the deepest block survives half the time, the earliest nearly always. The effect on effective depth follows directly: the number of surviving blocks $\tilde L$ is a sum of independent Bernoullis, so $\mathbb{E}(\tilde L) = \sum_\ell p_\ell = \sum_\ell [1 - \ell/(2L)] = L - (L+1)/4 = (3L-1)/4 \approx 3L/4$. For ResNet-110's 54 residual blocks that is about 40 — I train a net on average $\sim$40 blocks deep, deploy all 54, and save roughly a quarter of the forward/backward compute. That is the "short while training, deep at test" wish made concrete, and it is strongest precisely at the largest $L$.

The test-time rule needs care. At test I want the full net, every branch active. But during training block $\ell$'s branch was present only a fraction $p_\ell$ of the time, and everything downstream calibrated to that intermittent presence; turn it on for every test example and its contribution to the sum is on average $1/p_\ell$ larger than the downstream weights expect. This is the Dropout situation, and the fix is the same — scale the branch by its survival probability at test, $H = \mathrm{ReLU}(p_\ell \cdot F(x) + \mathrm{shortcut}(x))$, so the expected contribution matches training. The identity passes at full strength; only the recalibrated branch is weighted.

Then the substrate-specific care, because the harness implements the counting in a way I have to derive rather than assume. There is no global block index handed to a `CustomBlock` — the constructor sees only `(in_planes, planes, stride)`. So the block counts itself with a class-level counter: each block increments a shared counter on construction and records its own index, and the counter is *reset* at the first block of stage 1, detected by the signature `in_planes == 16 and planes == 16 and stride == 1`, which is unique to the first block of a CIFAR ResNet's first stage, so building a fresh model restarts the indexing. The total $L$ is read from the same class counter at forward time — by then every block has been constructed, so $L$ is the true total and `block_idx` runs $1..L$. The training forward draws a fresh `torch.rand(1)` per block per step and keeps the branch iff it is below $p$; the eval forward uses the $p$-scaled branch. One harness detail I note rather than fight: when a *transition* block is dropped the returned value is $\mathrm{ReLU}(\mathrm{shortcut}(x))$ with `shortcut` the Conv-BN projection, so a dropped transition is not a literal identity — it is the projected, rectified input. That is unavoidable on the two dimension-changing blocks per net (there is no identity to fall back to when the shapes change), and the clean-identity argument holds for the overwhelming majority of blocks.

So the edit relative to the pre-activation rung is: revert to the post-activation Conv-BN-ReLU block with the final ReLU after the add, give the class a self-counter and `_p_last = 0.5`, compute the linear-decay survival $p$ per block, drop the branch with probability $1 - p$ per minibatch in training (returning the rectified shortcut alone when dropped), and scale the branch by $p$ at test. No new parameters and no learned gate — the regularization comes entirely from the sampling.

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
