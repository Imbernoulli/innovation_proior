Residual learning already won the depth argument once. Plain deep convolutional nets get *worse training error* as you stack more layers — and worse on the training set, not just the test set, so it cannot be a capacity problem (a deeper net contains a shallower one by construction, just set the extra layers to identity), it has to be that SGD cannot find the identity-preserving solution. The additive shortcut fixed that: fit $F(x) = H(x) - x$ instead of $H(x)$, output $F(x) + x$, and if the right answer is near identity the solver just drives $F$ toward zero. That preconditioning is what let 100+-layer nets train at all. And yet the depth lever is still not clean. One very large residual net can train to tiny training error and still generalize worse than a much smaller one; another extremely deep original-unit net reduces its training loss painfully slowly at the start. So residual learning solved the first optimization wall but the path through a residual block may still carry an avoidable obstruction. Highway networks tried a richer, data-dependent gate on the shortcut, but the carry path is multiplied by a factor in $(0,1)$ that compounds over depth and can close, and they never demonstrated gains at extreme depth. I want to find exactly what is still in the way before touching anything.

Write a unit in the most general form: $y_l = h(x_l) + F(x_l, W_l)$ and $x_{l+1} = f(y_l)$, where $h$ sits on the shortcut, $F$ is the residual branch, and $f$ is whatever happens *after* the element-wise addition. In the unit I have been running, $h$ is identity and $f$ is a ReLU — the two choices I never questioned. Suppose both $h$ and $f$ are exactly identity. Then $x_{l+1} = x_l + F(x_l, W_l)$, and unrolling along a same-shape stretch where the identity shortcuts really apply,

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i).$$

The deep feature equals the shallow feature *plus a sum of residuals* — additive, with $x_l$ sitting cleanly out front, in contrast to a plain net where $x_L \approx \left(\prod_i W_i\right) x_0$ is a product of matrices that explodes or collapses with depth. The backward pass is where depth really dies, and it inherits the same clean structure:

$$\frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i, W_i)\right).$$

That $1$ is a gradient highway: $\partial E/\partial x_L$ is carried straight back to the shallow layer with *no* weight-layer Jacobian multiplying it, and it is hard to cancel — the residual term would have to equal $-1$ coordinate-wise for every sample, which has no systematic reason to happen. This is what residual learning is really providing, and it holds *exactly* only when both $h$ and $f$ are identity.

So I check both. On the shortcut, replace $h(x_l)=x_l$ with a scalar scale $h(x_l)=\lambda_l x_l$. Unrolling gives a leading term $\left(\prod_{i=l}^{L-1}\lambda_i\right)x_l$ and a direct backward term $\left(\prod_i \lambda_i\right)\partial E/\partial x_L$. For a deep net that product blows up if every $\lambda_i>1$ and vanishes if every $\lambda_i<1$ — exactly the multiplicative trap the additive shortcut escaped. The same product-of-Jacobians argument generalizes: any gate, all-layer $1\times1$ shortcut, or dropout on the shortcut introduces $\prod_i h'_i$ on the one path that must stay clean. These add representational power — a gated or $1\times1$ shortcut *contains* the identity in its solution space — but the issue is optimization, not representation: in the studied variants the training error goes *up*. So identity $h$ is load-bearing wherever the dimensions match; projection is reserved for the few shape-changing units that cannot add raw tensors. The remaining culprit is $f$, the ReLU after the add. With $f=\text{ReLU}$, the backward $1$ is gated by $\text{ReLU}'(y_l)$, which is zero wherever the pre-add sum $y_l$ is negative; there the gradient is forced back through the weights. At one or two units that is a minor leak, but it compounds — the direct route from layer $L$ to layer $l$ is clean only where *all* the intervening after-add ReLUs are open. At a thousand layers even an occasional truncation per unit makes the additive approximation fray, which is exactly the slow early optimization I started with. I need $f$ to be the identity.

I propose the **pre-activation residual unit**. I cannot simply delete the ReLU — the network needs its nonlinearities or the conv stack collapses to a linear map. The trick is that the after-add activation is *symmetric*: written across two stacked units, $y_{l+1} = f(y_l) + F(f(y_l), W_{l+1})$, the same $f(y_l)$ feeds both the shortcut input and the residual branch. Make it *asymmetric* — let the activation $\hat f$ touch only the residual branch and leave the shortcut bare:

$$x_{l+1} = x_l + F(\hat f(x_l), W_l).$$

Now the after-addition operation is a bare add — the identity I wanted on $f$ — and the activation has not vanished, it has moved *inside* the residual branch in front of its weight layers. That is precisely the pre-activation of the next unit's weights, so this only matters because of the branch-and-merge structure; in a plain chain calling an activation "pre" or "post" is meaningless. With $f$ identity, the exact additive forward recursion and the direct backward $1$ both hold again.

That still leaves *which* of BN and ReLU move in front of the conv, and in what order; I isolate the ordering by elimination, letting each failure name the next fix. *BN after the addition* puts BN — a non-identity scale-and-shift by $\gamma/\sqrt{\text{var}}$ — directly on the clean shortcut, breaking the very path I am protecting; rejected. *ReLU before the addition* makes $f$ identity but ends the residual branch in a ReLU, so $F \geq 0$ always, and then within a same-shape additive chain $x_L = x_l + \sum F$ adds only non-negative terms and the feature is monotonically non-decreasing with depth — but a residual must range over $(-\infty,\infty)$, since subtracting is half the point of fitting $H(x)-x$; rejected. *ReLU-only pre-activation* moves the ReLU to the front but leaves BN after the conv, so the leading ReLU acts on un-normalized activations and gets no BN benefit; this only matches baseline. *Full pre-activation*, **BN → ReLU → Conv** for each weight layer, satisfies every constraint at once: the after-add operation is a bare add so $f$ is identity and the highway is exact; the branch ends in a conv so $F$ can be negative; no BN sits on the shortcut; and the first operation each weight layer sees is a BN, so *every* weight layer in the network receives a normalized input.

Two distinct payoffs follow from two different fixes, and they have different fingerprints. The first is ease of optimization, from $f=\text{identity}$: the forward $x_L = x_l + \sum F$ and backward $\partial E/\partial x_l = \partial E/\partial x_L (1 + \cdots)$ are now exact through every same-shape stretch rather than ReLU-gated approximations, so the extreme-depth net stops crawling at the start of training. This explains why the original after-add ReLU was only *mildly* bad at moderate depth — there $x_l$ is the output of the previous unit's ReLU, so $x_l \geq 0$ and $y_l = x_l + F(x_l)$ is negative only when $F$ is sufficiently negative, which is rare at ~100 layers but compounds at ~1000, so the fix's benefit *grows* with depth. The second is regularization, from BN moving to the front: in the *original* unit the branch ends in conv → BN, normalizing $F$, but that normalized output is immediately added to the un-normalized shortcut $x_l$, so $x_l + \text{BN}(F)$ is un-normalized and that is what feeds the next weight layer — BN's normalization is undone by the addition. In full pre-activation BN is the first thing each weight layer sees, so every weight layer's input is genuinely normalized and BN's mini-batch noise regularizes it; the signature is slightly *higher* training loss but *lower* test error, separate from the optimization fingerprint of faster, lower training loss.

Made concrete as a block: the basic branch is two $3\times3$ convs in pre-activation order, $\text{BN}\to\text{ReLU}\to\text{conv}\to\text{BN}\to\text{ReLU}\to\text{conv}$, then a clean add with nothing after it. When the stage changes — a stride for downsampling or a jump in channels — the identity cannot add to a differently-shaped tensor, so a $1\times1$ projection matches dimensions, and it too is pre-activated, fed from the shared $\text{pre} = \text{ReLU}(\text{BN}(x_l))$ computed once before the split. For equal-shape units the shortcut stays raw $x_l$; there is no scaling and no gate on either path, since a multiplicative factor on the clean path is exactly the mistake the whole analysis rules out. Two boundary cases are forced by the asymmetric shift rather than bolted on: the very first unit's first $\text{BN}\to\text{ReLU}$ belongs right after the stem conv and before the first split, and because the last unit's after-add activation got pushed forward to a next unit that does not exist, one extra $\text{BN}\to\text{ReLU}$ belongs after the last addition, before global average pooling and the linear classifier.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Interior full pre-activation basic residual block (BN -> ReLU -> Conv, twice)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            # Shape-matching projection for dimension changes; fed from pre-activation.
            self.shortcut = nn.Conv2d(
                in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x))                  # first pre-activation
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out)))    # branch ends in a conv => F can be negative
        shortcut = self.shortcut(pre) if self.shortcut is not None else x
        return shortcut + out                      # clean add; no activation after it
```

The pre-activation bottleneck unit, matching the official `resnet-1k-layers` CIFAR implementation, applies the same $\text{BN}\to\text{ReLU}\to\text{Conv}$ ordering to a $1\times1\to3\times3\to1\times1$ branch. Equal-shape units use a raw identity shortcut; dimension-changing units share a common first BN-ReLU before both the residual branch and the projection shortcut, with the downsampling stride on the first $1\times1$ conv and on the $1\times1$ projection:

```python
class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(
                in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        shortcut = self.shortcut(pre) if self.shortcut is not None else x
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return shortcut + out
```
