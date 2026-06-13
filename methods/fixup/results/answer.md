# Fixup, distilled

Fixup (fixed-update initialization) trains deep residual networks *without any normalization
layer* by rescaling a standard initialization so that one SGD step changes the network function
by a depth-independent amount. It diagnoses why plain residual nets need normalization — the
additive skip makes activation variance explode with depth, and positive homogeneity turns that
logit blowup into a lower bound on the gradient norm — and fixes it at the source: scale each
residual branch's weight layers by `L^{-1/(2m-2)}`, zero-initialize the last layer of each branch
and the classifier, and add scalar biases plus a per-branch scalar multiplier to recover what
normalization's shift and scale were doing.

## Problem it solves

Train very deep (hundreds to thousands of layers) residual networks with no batch/layer
normalization, at the *same* maximal learning rate, *same* convergence speed, and *at least as
good* generalization as a normalized net — removing normalization's batch dependence, train/test
discrepancy, running statistics, and its entanglement with weight decay and the LR schedule.

## Why plain residual nets fail without normalization

For a plain residual net `x_l = x_0 + sum_{i<l} F_i(x_i)`, the law of total variance gives
`Var[x_{l+1}] = E[Var[F_l(x_l)|x_l]] + Var[x_l]`, so the skip *forces* the variance to grow;
with variance-preserving (He) init `Var[x_{l+1}] ~ 2 Var[x_l]`, i.e. `~ 2^l`, exploding
exponentially in depth. A normalization-free net of bias-free convs/linears + ReLU is positively
homogeneous of degree one (`f(alpha x) = alpha f(x)`, `alpha > 0`), and that yields two gradient
lower bounds at initialization:

```
|| dl/dx_{i-1} ||       >= (l(z,y) - H(p)) / ||x_{i-1}||
|| dl_avg/dtheta_ph ||  >= (1/(M ||theta_ph||)) sum_m (l(z^m,y^m) - H(p^m)) =: G(theta_ph)
E[G(theta_ph)]          >= (E[max_i z_i] - log c) / ||theta_ph||
```

derived by differentiating the cross-entropy `l(z,y) = -y^T(z - logsumexp(z))` along the
homogeneous scaling direction: `d/d_eps l(f((1+eps)x), y)|_0 = (p - y)^T z = l(z,y) - H(p)`, then
bounding the gradient norm by this directional derivative. So if the logits `z` blow up (which
the variance explosion guarantees), the gradients of the network's positively-homogeneous weight
sets are lower-bounded by something that grows with depth. Normalization works precisely because
it holds the logits at `O(1)`, keeping these bounds small.

## Key idea

Don't control per-layer activation *scale* (that misses the real failure); control the *function
update* per SGD step. One step changes the function by

```
Delta f(x_0) = -eta ( sum_l sum_i J_l^i ) (dl/dz) + O(eta^2),
J_l^i = ||F_l^{(i-)}||^2 (df/dx_l)^T F_l^{(i+)} (F_l^{(i+)})^T (df/dx_l)  (c x c, symmetric PSD).
```

Each `J_l^i` is PSD, so `trace(sum J_l^i)` is additive and scales with the number of branches
`L`: **residual branches update the network in sync** (their contributions align and add, they
do not cancel). To keep `||Delta f|| = Theta(eta)` independent of depth, each branch must
contribute `Theta(eta/L)`.

For a scalar model of an `m`-layer branch `F(x) = (prod a_i) x`:

```
dl/da_i = (dl/dF) F(x)/a_i,
Delta F(x) = -eta (dl/dF) F(x)^2 sum_i 1/a_i^2  = Theta( eta F(x)^2 / A^2 ),  A = min_k a_k,
```

since `M = sum 1/a_i^2` is dominated by the smallest `a_i`. Setting `Delta F = Theta(eta/L)`
gives the constraint (**iff**)

```
( prod_{k != j} a_k ) x = Theta(1/sqrt(L)),   j = argmin_k a_k.
```

The smallest-scaled layer pins the update. Solving it with all branch layers at one scale gives
`a^{m-1} = L^{-1/2}`, i.e. `a = L^{-1/(2m-2)}`.

## The method (three rules)

1. **Zero-init** the classification (final) layer and the *last* layer of each residual branch.
   (Each branch starts as the zero function, so the residual stack starts on the identity path;
   logits start at 0, making `l(z,y) - H(p) = 0` in the activation-side bound; nothing random in
   the branches has to be "unlearned".)
2. **Scale** every other layer with a standard init (He), and multiply the weight layers
   *inside each residual branch* by `L^{-1/(2m-2)}` (`L` = number of residual branches, `m` =
   layers per branch). For `m=2`: `L^{-1/2}`; for `m=3`: `L^{-1/4}`. This is the essential rule in
   the branch-update calculation. (Naively zeroing only the last layer *without* scaling the
   others fails: after one step the zeroed layer gets an `O(1)` gradient and the branch
   contributes a `(1+O(1))` factor, re-exploding the output for large `L`. The scaling makes the
   post-update branch contribution `O(1/L)`.)
3. **Add** a scalar multiplier (init 1) per branch and scalar biases (init 0) before each conv,
   linear, and element-wise activation layer; keep the convolutional weights bias-free. The scalar
   biases restore normalization's shift (`beta`); the scalar multiplier restores its weight-norm /
   effective-learning-rate dynamics: a high-dim weight is ~orthogonal to its gradient, so under
   weight decay `||w||` shrinks and the multiplier absorbs the scale, raising the effective LR
   `eta/||w||^2` the way scale-invariant normalized layers do — so the existing LR schedule
   transfers. This costs only `O(K)` extra parameters (vs `O(K*C)` for per-channel norm params).

## Working code

Filling the residual harness's `initialize`/per-block slots (faithful to the CIFAR
implementation: bias-free convs; scalar params as `nn.Parameter(torch.zeros/ones(1))`):

```python
import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):          # m = 2
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))     # branch multiplier (init 1)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)
        out += identity
        out = self.relu(out)
        return out


class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.num_layers = sum(layers)                # L = number of residual branches
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                # Rule 2: He init on conv1, scaled by L^{-1/(2m-2)} == L^{-0.5} for m=2
                nn.init.normal_(
                    m.conv1.weight, mean=0,
                    std=np.sqrt(2 / (m.conv1.weight.shape[0] *
                                     np.prod(m.conv1.weight.shape[2:])))
                        * self.num_layers ** (-0.5))
                # Rule 1: zero the last conv of the branch
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                # Rule 1: zero the classification layer
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)   # parameter-free shortcut
        seq = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            seq.append(block(planes, planes))
        return nn.Sequential(*seq)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x + self.bias2)
        return x


def fixup_resnet56(**kwargs):
    return FixupResNet(FixupBasicBlock, [9, 9, 9], **kwargs)
```

Bottleneck (`m = 3`): scale `conv1` and `conv2` by `num_layers ** (-1/(2*3-2)) ==
num_layers ** (-0.25)`, zero `conv3`, and keep the same scalar-parameter pattern extended to
`bias1a/1b`, `bias2a/2b`, `bias3a/3b`, plus one multiplier. The ImageNet downsample shortcut is
a `1x1` conv He-initialized at standard scale (unscaled), and the stem uses a `7x7` conv with
maxpool.

## Relation to prior methods

- **He / Xavier init**: variance-preserving for plain nets; blind to the additive skip, so they
  cause the `2^l` variance explosion in residual nets. Fixup *starts from* He init and rescales
  the branch by `L^{-1/(2m-2)}`.
- **Batch / layer normalization**: hold logits `O(1)` by dividing out activation statistics, and
  give an automatic effective-LR schedule via scale invariance + weight decay. Fixup is *not* a
  normalization method -- it computes no statistics at init or during training; it reproduces the
  shift with scalar biases and the effective-LR dynamics with a scalar multiplier.
- **sqrt(1/2)-scaled recurrence**: keeps variance `O(1)` but geometrically suppresses deep
  branches. Fixup controls the *function* update without adding a geometric decay along the skip
  path.
- **LSUV**: data-dependent per-layer variance calibration. Fixup is data-independent and controls
  the synchronized per-step function update across all `L` branches, which per-layer scale misses.
- **Zero/near-zero residual init** (zero-gamma BN, highway networks, linearized residual-net
  analyses): start residual
  blocks close to identity. Fixup makes this precise: zeroing alone is insufficient; the
  `L^{-1/(2m-2)}` scaling on the remaining branch layers is what makes the post-update branch
  contribution `O(1/L)`.
