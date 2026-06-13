Let me start from what actually goes wrong. I want to train a deep residual network — hundreds, ideally thousands of layers — and I want to do it *without* a normalization layer in each block. Everyone reaches for batch normalization because, the story goes, it stabilizes training, lets you crank the learning rate, speeds convergence, and helps generalization. But that bundle of four claims hides which benefit is really intrinsic to normalization and which is incidental, and normalization itself drags in a batch dependence in the forward pass, a train/test discrepancy, running statistics, and a tangle with weight decay. So I want to know: can I just *initialize* a normalization-free residual net so well that it trains at the same maximal learning rate, converges as fast, and generalizes as well? Before I can hope to do that I have to understand, precisely, why the plain version fails. Not "it's unstable" — the actual mechanism.

So take a plain residual network, the blocks `F_1, ..., F_L` stacked with skip connections, `x_l = x_0 + sum_{i<l} F_i(x_i)`. View initialization only: `x_0` fixed, the weights random. What does the activation variance do as I go up? At init the branch is zero-mean given its input, `E[F_l(x_l) | x_l] = 0`, so by the law of total variance, `Var[x_{l+1}] = E[Var[F_l(x_l)|x_l]] + Var[x_l]`. The first term is nonnegative, so the variance can *only grow* with depth — the skip connection guarantees it. And if I initialize the branch the standard way, He-style, designed so a layer maps unit-variance input to unit-variance output, then `Var[F_l(x_l)|x_l] ~ Var[x_l]`, and `Var[x_{l+1}] ~ 2 Var[x_l]`. The variance roughly *doubles every block*. By the top of an `L`-block net it's blown up like `2^L`. The logits at the output are enormous and depend wildly on the random weights. That's the disease. He init was derived for a plain feed-forward stack where preserving variance layer-to-layer is exactly right; it has no notion of the additive skip that re-adds the running sum at every block, so it over-provisions each branch and the sum explodes.

Now I have to connect "logits explode at init" to "can't train." Big logits aren't automatically fatal — what's fatal is what they do to the gradients. Let me see if I can lower-bound the gradient in terms of the loss, because if exploding loss forces exploding gradient, that's the link. The lever is positive homogeneity. Strip the normalization out of the net and zero the biases, and what's left — bias-free convs and linear layers, ReLU, pooling, addition, concatenation, dropout — every one of those is positively homogeneous of degree one: scale the input by `alpha > 0` and the output scales by `alpha`. ReLU obviously: `relu(alpha x) = alpha relu(x)` for `alpha > 0`. A bias-free linear map trivially. And a composition of degree-one homogeneous maps is again degree-one homogeneous — chain the scalings through. So the whole network, as a function of its input, is p.h., and so is the part of the network *above* any block: if `z = f_{i->L}(x_{i-1})` is the map from the input of block `i` up to the logits, then `f_{i->L}((1+epsilon) x_{i-1}) = (1+epsilon) f_{i->L}(x_{i-1})`.

That homogeneity lets me compute a directional derivative for free. Perturb the input to block `i` along its own direction, `x_{i-1} -> (1+epsilon) x_{i-1}`, which by homogeneity sends `z -> (1+epsilon) z`. Differentiate the cross-entropy loss `l(z,y) = -y^T(z - logsumexp(z))` with respect to `epsilon` at `epsilon = 0`. By the chain rule it's `(∂l/∂z) · (∂z/∂epsilon)`, and `∂z/∂epsilon = z`. Now `∂l/∂z` is the standard softmax-minus-one-hot, `p - y`, where `p = softmax(z)`. So the derivative is `(p - y)^T z = p^T z - y^T z`. Let me check this equals something interpretable in terms of the loss. The loss is `l = -y^T z + logsumexp(z)` (since `y` is one-hot, `y^T 1 = 1`). The entropy of `p`: `log p_i = z_i - logsumexp(z)`, so `H(p) = -sum_i p_i log p_i = -sum_i p_i z_i + logsumexp(z) = -p^T z + logsumexp(z)`. Then `l - H(p) = (-y^T z + logsumexp(z)) - (-p^T z + logsumexp(z)) = -y^T z + p^T z = (p - y)^T z`. So the directional derivative is exactly `l(z,y) - H(p)`. Clean.

A directional derivative can't exceed the gradient norm times the unit direction, so taking the perturbation direction `x_{i-1}/||x_{i-1}||` and rescaling `epsilon = t/||x_{i-1}||`,

  `|| ∂l/∂x_{i-1} || >= (l(z,y) - H(p)) / ||x_{i-1}||`.

There's the link. The entropy `H(p)` is at most `log c` for `c` classes, a small constant; `||x_{i-1}||` is small in the lower blocks. So if the loss `l` is large — which it is when the logits have blown up to nonsense values that disagree with the labels — then the gradient with respect to the input of a low block is large. Exploding logits force exploding gradients. That's why normalization is "essential" with a standard init: by holding the logits at `O(1)`, it keeps this lower bound small.

But I care about *weights*, not just activations, because it's the weights that get updated and can diverge. Can I get the same kind of bound for a set of weights? Yes, if that set is p.h. in the right sense. Call `theta_ph` a *positively homogeneous set* if scaling all of those parameters together by `alpha > 0` scales the network output by `alpha`. Then `f(x; rest, alpha * theta_ph) = alpha f(x; rest, theta_ph)`, and the exact same directional-derivative argument runs, now perturbing along `theta_ph` instead of along the activation. Averaging over a minibatch `D_M`,

  `|| ∂l_avg/∂theta_ph || >= (1/(M ||theta_ph||)) sum_m (l(z^m,y^m) - H(p^m)) =: G(theta_ph)`.

I want to know how big `G` is at initialization in expectation, so I can see whether it grows with depth. The output-layer weights are i.i.d. from a symmetric zero-mean distribution, so the logits `z` have a symmetric density with mean zero. Then `E[l(z,y)] = E[-y^T(z - logsumexp(z))] >= E[y^T(max_i z_i - z)]`, using `logsumexp(z) >= max_i z_i`. Expand: `E[y^T max_i z_i] - E[y^T z] = E[max_i z_i] - 0`, because at init `y` and `z` are independent and `E[z] = 0` kills the second term, and `y^T 1 = 1` turns the first into `E[max_i z_i]`. With `E[H(p)] <= log c`,

  `E[G(theta_ph)] >= (E[max_i z_i] - log c) / ||theta_ph||`.

So the gradient norm of a p.h. weight set is `Omega(E[max_i z_i])` at init. And which weight sets in a residual net are p.h. sets? The first convolution before the pooling; the fully-connected layer before softmax; and the union of a spatial downsampling layer in the backbone together with the convolution in its corresponding residual branch — scale those together and the output scales. For each of these, if the logits blow up — and the variance explosion guarantees `E[max_i z_i]` grows with depth — the gradient norm is lower-bounded by something that grows with depth. Exploding gradients on the weights. There it is, fully: standard init makes the logits blow up exponentially, the loss is huge, and that *lower-bounds* the gradient norm of these weight sets, so the very first updates are catastrophic at any learning rate normalization would tolerate. This is the failure mode in equations, and it tells me exactly what an initialization has to prevent: it must keep the output scale from exploding with depth.

The naive cure suggests itself: keep the variance from doubling by scaling down each branch. The `sqrt(1/2)` recurrence, `x_l = sqrt(1/2)(x_{l-1} + F_l(x_{l-1}))`, does keep the activation variance `O(1)`. But unroll it: `x_l` is a weighted sum where the contribution of branch `F_i` is multiplied by `(sqrt(1/2))^{l-i}`, which decays geometrically as you go down. So the deep branches near the output dominate and the early branches are suppressed exponentially — I've kept the variance under control by quietly switching off most of my residual branches, defeating the point of being deep. And LSUV — orthogonal-init then rescale each layer to unit output variance using a calibration minibatch — also just controls the *scale* of activations and gradients layer by layer, and it needs to peek at data. Both of these control *scale*. But I have a nagging feeling that scale isn't the right object. Stabilizing the activation/gradient *magnitude* at init is necessary, sure, but it's a property of one forward/backward pass. What I actually want is for the *network function* to be well-behaved *as I train it* — for one SGD step to move the function by a sane, depth-independent amount. That's a statement about the update to `f`, not about the variance of `x_l`. Let me make that the goal and see if it's stronger.

Set the goal precisely. Let `eta` be the learning rate. I want: after initialization, one SGD step changes the network function by `Theta(eta)`, independent of depth, as `eta -> 0`. That is, `||Delta f(x)|| = Theta(eta)` where `Delta f(x) = f(x; theta - eta ∂l/∂theta) - f(x; theta)`. Not "the variance is `O(1)`" — the *function's update* is the right scale. If I can engineer that, then the same learning rate that works for a shallow net works for the deep one, because each step moves the function the same amount regardless of `L`.

Now, how does one SGD step move `f`? I have `L` residual branches, each getting updated. Does the total move add up to something `Theta(eta)`, or do the branches' contributions cancel? If they cancel, I'd be fine even with large per-branch updates; if they align, then `L` branches each moving the output a little could move it a lot. Let me actually compute `Delta f(x_0)` to first order in `eta`. Two tricks make this tractable. First, condition on the input `x_0`: then each ReLU is just a fixed diagonal 0/1 mask `D`, so the forward and backward passes are *linear* in the weights — and this stays valid even one step later, as long as `eta` is small enough that the signs of the preactivations don't flip. Second, for a degree-one p.h. block with no bias, Euler's homogeneous-function identity gives `f_l(x_{l-1}) = (∂f_l/∂x_{l-1}) x_{l-1}` — the block equals its own Jacobian applied to its input.

With the ReLUs frozen as masks, write the gradient of the loss with respect to the `i`-th weight layer in the `l`-th block. Let `F_l^{(i-)}` be the forward map from `x_{l-1}` up to just before layer `i` (the masked-linear product of the lower layers and `x_{l-1}`), and `F_l^{(i+)}` the masked-linear map from after layer `i` to the block output. Then by the chain rule and the Kronecker structure of a weight-layer derivative,

  `∂l/∂Vec(W_l^{(i)}) = (F_l^{(i-)} ⊗ I)(F_l^{(i+)})^T (∂f/∂x_l)(∂l/∂z)`.

Plug `Delta theta = -eta ∂l/∂theta` back into the forward pass, expand to first order in `eta`, and use the homogeneity identity to collapse the algebra. What drops out is

  `Delta f(x_0) = -eta sum_l sum_i [ ||F_l^{(i-)}||^2 (∂f/∂x_l)^T F_l^{(i+)}(F_l^{(i+)})^T (∂f/∂x_l) ] (∂l/∂z) + O(eta^2)`,

and the bracketed thing — call it `J_l^i` — is a `c x c` matrix, real, symmetric, positive semidefinite (it's of the form `A A^T` sandwiched with a nonnegative scalar `||F_l^{(i-)}||^2`). Write `J = sum_l sum_i J_l^i`, so `Delta f(x_0) = -eta J (∂l/∂z) + O(eta^2)`.

Now `J` is a sum of `L`-many (times the per-block layer count) PSD matrices. The trace norm of a PSD matrix is its trace, and trace is additive, so `||J||_* = trace(J) = sum_l sum_i trace(J_l^i)`, which scales *linearly* with the number of residual branches `L` as long as the average branch trace stays at ordinary scale. The branches don't fight each other — their `J_l^i` are all PSD, they all push in trace-positive directions, they *add*. And at initialization the logits `z` have no correlation with the label `y`, so `∂l/∂z = p - y` is a vector in a random direction; the expected size of `Delta f = -eta J (∂l/∂z)` is governed by the trace norm of `J`, which is proportional to `L`. So the residual branches update the network *in sync*: each contributes a coherent, additive shove, and `L` of them make a shove proportional to `L`. That is exactly what LSUV and `sqrt(1/2)`-scaling miss — they bound each branch's *scale* but never account for the fact that `L` correlated updates pile up.

And that hands me the design rule for free. If `L` branches each move the function by some amount and those moves add up, then to make the *total* `Delta f = Theta(eta)` I need each branch to move the function by `Theta(eta/L)` on average. Not `Theta(eta)` per branch — that would give `Theta(eta L)` total, blowing up with depth. Each residual branch should change the network output by `Theta(eta/L)`. That's the quantitative target, and it came out of the in-sync computation, not out of a variance argument.

So now the question is purely: how do I initialize a single residual branch with `m` layers so that one SGD step changes its output by `Theta(eta/L)`? Since I only care about *scale*, I can study the scalar caricature of a branch: `F(x) = (prod_{i=1}^m a_i) x`, with `a_i, x >= 0`. The standard variance-preserving init corresponds to every `a_i = 1` (each layer preserves scale); setting `a_i` to something other than 1 means rescaling layer `i` by `a_i`. I want to find the constraint on the `a_i` that gives `Delta F(x) = Theta(eta/L)` under one gradient step, and then read off how to rescale.

Compute it. The gradient of the loss with respect to `a_i` is `∂l/∂a_i = (∂l/∂F)(prod_{k != i} a_k) x = (∂l/∂F) F(x)/a_i`. One gradient step sets `Delta a_i = -eta ∂l/∂a_i`. To first order the change in the branch output is `Delta F = sum_i (∂F/∂a_i) Delta a_i`, and `∂F/∂a_i = (prod_{k != i} a_k) x = F(x)/a_i` as well. So

  `Delta F(x) = -eta sum_i (F/a_i)(∂l/∂F)(F/a_i) = -eta (∂l/∂F) F(x)^2 sum_i 1/a_i^2`.

(When some `a_j = 0`, read `F/a_j` as the product over the other factors times `x` — the limiting value, which is finite.) Let `M = sum_i 1/a_i^2` and `A = min_k a_k`. The term `1/a_i^2` is largest for the *smallest* `a_i`, so `M` is dominated by `A`: `1/A^2 <= M <= m/A^2`. Therefore `F^2/A^2 <= F^2 M <= F^2 m/A^2`, and with `∂l/∂F = Theta(1)`, `x = Theta(1)`,

  `Delta F(x) = Theta(eta F(x)^2 / A^2)`.

Set this to `Theta(eta/L)`: I need `F(x)^2/A^2 = Theta(1/L)`, i.e. `F(x)/A = Theta(1/sqrt(L))`. Unpacking, `F(x)/A = (prod_{k} a_k) x / min_k a_k = (prod_{k != j} a_k) x` where `j = argmin_k a_k`. So the constraint is

  `(prod_{k in [m]\{j}} a_k) x = Theta(1/sqrt(L)),  j = argmin_k a_k`.

This is the condition for a branch to make `Theta(eta/L)` updates — both directions: rearranging the `Delta F` computation shows it's *if and only if*. The interesting content is that `min`: the update scale of the whole branch is pinned by the *smallest-scaled layer*, because that's the layer whose tiny `a_j` gives the biggest `1/a_j^2` and dominates `M`.

Now solve the constraint into an actual rescaling. The cleanest symmetric solution: scale *every* layer in the branch by the same factor `a_i = a`. Then `j` is any layer, `prod_{k != j} a_k x = a^{m-1} x`, and I need `a^{m-1} x = Theta(1/sqrt(L))`, so with `x = Theta(1)`, `a^{m-1} = L^{-1/2}`, giving `a = L^{-1/(2m-2)}`. So: take a standard initialization, and *scale the weights of every layer inside each residual branch by* `L^{-1/(2m-2)}`. For a two-conv branch, `m = 2`, that's `L^{-1/2}`; for a three-conv bottleneck, `m = 3`, that's `L^{-1/4}`. The exponent looks odd until you see where it comes from: I have `m-1` nonzero factors multiplying `x`, each raised to the same power, and their product has to hit `L^{-1/2}`, so each carries the `(m-1)`-th root of `L^{-1/2}`, which is `L^{-1/(2m-2)}`. The reason I want all layers at the *same* scale rather than dumping the whole reduction onto one of them: if the scales are wildly imbalanced, so are the gradients and the per-layer contributions to `Delta F`, and the smallest layer dominates everything (that's the `min` again). Balanced scales make the layers contribute equally to the branch's update.

A separate worry now nags at me. Even if I scale the branch correctly, at init each branch is still a small random function, and the first thing SGD has to do is *unlearn* whatever junk that random branch computes before it can learn anything useful. Can I avoid paying that? What if I start the branch as the exact zero function? Set the *last* layer of the branch to zero, `a_m = 0`, and scale the other `m-1` layers by `L^{-1/(2m-2)}` as before. At init `F(x) = 0` for every branch, so the whole residual network is just the identity skip path — `x_l = x_0` all the way up. The variance no longer explodes (no branch contributes anything), and there's no random junk to unlearn. Does the update still come out right? With `a_m = 0`, the argmin is `j = m`, and the constraint quantity is `prod_{k != m} a_k x = (L^{-1/(2m-2)})^{m-1} x = L^{-1/2} = Theta(1/sqrt(L))`. Still satisfied. So zeroing the last layer and scaling the rest is *also* a valid solution to the same constraint, and it has the extra benefits of starting at identity and not needing to unlearn. I'll prefer this one.

I have to be careful, though, because there's a naive version of "zero the last layer" that *doesn't* work, and the constraint is precisely what tells them apart. Suppose I just zero the last conv of each branch and leave the other layers at standard scale (`a_i = 1` for `i < m`). At init the branch is zero, fine. But after one SGD step the zeroed layer picks up an `O(1)` gradient, and now the branch contributes a multiplicative factor like `(1 + O(1))` to the running sum — and with the other layers at `a_i = 1`, `prod_{k != m} a_k x = x = Theta(1)`, not `Theta(1/sqrt(L))`. The constraint is violated. So `L` branches each contribute an `O(1)` factor and the output blows up again the moment training starts, even though it was zero at init. The fix is exactly the `L^{-1/(2m-2)}` scaling on the *other* layers: it shrinks the post-update branch contribution to `O(1/L)`, so that `L` of them sum to `O(1)`. Zeroing without scaling fails; scaling is the load-bearing part in the update calculation, while zeroing the last layer is the refinement that removes the initial random branch function.

And while I'm zeroing things: I should zero the *classification* layer too. The whole gradient-explosion story bottomed out at "if the logits blow up at init, the gradient is lower-bounded by something huge." If the final FC layer starts at zero, the logits start at exactly zero. Then `p` is uniform, `l(z,y) = log c`, `H(p) = log c`, and the activation-side directional derivative `l(z,y) - H(p)` is exactly zero at step zero. I should not reuse the p.h.-weight lower bound for the zero classifier itself, because its norm is zero and that denominator is not defined; the point is simpler and cleaner: zero logits remove the positive logit-scale term that made the lower bound grow. There's a standard objection that zeroing weights is a bad init because it fails to break symmetry between units — but that objection is about zeroing *everything*; here only the branch's last layer and the classifier are zero, while every other layer keeps its symmetry-breaking random init, so units are still distinguishable and gradients still flow to the non-zero layers.

So the scaling rule plus the two zero-inits gives the network the right first-step update scale at the learning rate I actually want to use. But "right update scale" isn't "as good as a normalized net." A normalized net does two more things that I've thrown away, and if I want to *match* its training dynamics rather than merely avoid the initial explosion, I need to recover them: the shift and the scale that normalization layers apply.

The shift first. Batch norm's `beta` adds a learnable per-channel bias after normalizing. Why does that matter? Because the preferred input mean of a weight layer is generally not the preferred output mean of the activation feeding it; a ReLU outputs only nonnegatives, so the mean going into the next conv is shifted positive, and the network may want to recenter it. Normalization gets to do this implicitly. Without it, I should put explicit bias terms back. The cheap way: instead of full per-channel biases (which cost `O(channels)` per layer), insert a *single scalar* bias before each convolution and linear layer, and also before each ReLU, while keeping the convolutional layers themselves bias-free. One learnable scalar per insertion point, `O(1)` per layer rather than `O(channels)`, gives the branch a global recentering knob while keeping the parameterization minimal. Initialize these scalar biases at zero; for the final linear classifier, keep the ordinary bias if the training harness already has it and zero it with the weights.

Now the scale. Batch norm has a `gamma` that scales each channel, so the obvious move is to add one learnable scalar multiplier per residual branch, initialized at 1 — but I want to know *why* that actually matters for matching the normalized net, since otherwise it's just cargo-culting `gamma`. The answer is in the interaction between scale invariance, weight decay, and the learning rate. A normalized layer's output is invariant to rescaling the weights that feed it — divide the activations by their standard deviation and any scalar on the weights cancels. So under normalization, an `L2` weight penalty doesn't regularize the function at all; it only shrinks `||w||`. And because the output is invariant to the weight's scale, the gradient is (to leading order) orthogonal to the weight vector. Then the *effective* learning rate, measured on the unit-normalized weight, is `eta/||w||^2`: as you decay `eta` and weight decay keeps shrinking `||w||`, the effective rate `eta/||w||^2` actually *rises*. Normalization quietly gives you an automatic effective-learning-rate schedule. If I drop normalization but keep the same hand-tuned `eta` schedule, I'll lose that, and I'd have to re-tune the schedule from scratch. But notice: even without any normalization, a weight *tensor* in high dimensions is still approximately orthogonal to its gradient, just by dimensionality — two generic high-dimensional vectors are nearly orthogonal. So if I attach one scalar multiplier to the branch, the branch output becomes invariant to rescaling the weights relative to that multiplier, and the same mechanism kicks in: weight decay shrinks `||w||`, the multiplier absorbs the lost scale and grows, the effective learning rate of the weight layers rises the same way it would under normalization. One scalar multiplier per branch, initialized at 1, reproduces the weight-norm dynamics of a normalized branch and spares me from searching for a new learning-rate schedule.

Let me assemble the recipe. Initialize the classification layer and the last layer of each residual branch to zero. Initialize every other layer with a standard method (He), and scale only the weight layers *inside* residual branches by `L^{-1/(2m-2)}`, where `L` is the number of residual branches and `m` the number of layers in the branch. Add a scalar multiplier (init 1) in every branch and a scalar bias (init 0) before each convolution, linear, and element-wise activation layer, with the convolutional weights bias-free. The scaling is the essential rule that the constraint forced; the zero-inits and the bias/multiplier are what close the gap to a normalized network.

Now to code, filling the slot in the residual harness. The block stops being "two convs and a skip" and becomes "two convs, with a scalar bias before each conv and before each ReLU, a scalar multiplier on the branch output before the add, and a scalar bias after it," and the network's init loop does the scaling and the zeroing. I'll mirror the standard residual-CIFAR structure, where `L = num_layers` is the total count of residual blocks (so for a basic block `m = 2`, the scaling is `num_layers^{-1/2}`).

```python
import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    # bias is OFF; the branch carries explicit scalar biases instead
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    """A residual branch of m=2 convs, no normalization. It carries the scalar
    biases bias1a, bias1b, bias2a, bias2b and one branch scale multiplier."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))   # scalar shift before conv1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))   # scalar shift before relu
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))   # scalar shift before conv2
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))     # branch multiplier (init 1)
        self.bias2b = nn.Parameter(torch.zeros(1))   # scalar shift after branch
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b          # multiplier then shift
        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)
        out += identity                               # branches add IN SYNC
        out = self.relu(out)
        return out


class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.num_layers = sum(layers)                 # L = number of residual branches
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))     # scalar shift before stem relu
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))     # scalar shift before classifier
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                # standard (He) init on the FIRST conv, then scale by L^{-1/(2m-2)};
                # for m=2 that exponent is -1/2 == num_layers ** (-0.5)
                fan_out = m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:])
                nn.init.normal_(
                    m.conv1.weight, mean=0,
                    std=np.sqrt(2.0 / fan_out) * self.num_layers ** (-0.5))
                # zero the LAST conv of the branch -> residual branch starts as zero
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                # zero the classification layer -> logits start at 0
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)  # parameter-free shortcut
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
```

For a bottleneck branch with `m = 3` the exponent and the zeroed layer change: scale the first two convs by `num_layers ** (-1/(2*3-2)) = num_layers ** (-0.25)` and zero the third conv; the scalar-bias pattern extends to `bias1a/bias1b`, `bias2a/bias2b`, and `bias3a/bias3b`, with one multiplier and the zeroed classifier unchanged.

Let me trace the whole causal chain once. I wanted to train a deep residual net with no normalization at the maximal learning rate. The plain version fails because the additive skip forces the activation variance to grow — doubling per block under He init, so exponentially in depth — and positive homogeneity turns that logit blowup into a *lower bound* on the gradient norm of the network's weight sets, so the first step explodes. Controlling per-layer activation scale (LSUV, `sqrt(1/2)`-scaling) isn't enough, because the real object is the change in the network *function* per step, and computing that change to first order showed the residual branches update the function in sync — their contributions are PSD and add up, so the total move scales with the number of branches `L`. To keep the function's per-step move at `Theta(eta)` independent of depth, each branch must move it by `Theta(eta/L)`; solving that constraint on a scalar branch forced the rescaling `L^{-1/(2m-2)}` on the branch's weight layers, with the smallest-scaled layer pinning the update. Zeroing the last layer of each branch makes the residual stack start on the identity path, and zeroing the classifier makes the initial logits exactly zero — no variance explosion, nothing random in the branches to unlearn, and no positive logit-scale term in the activation-side lower bound — while the scaling on the other layers keeps the post-update branch contribution at `O(1/L)`, so zeroing alone (which the naive version got wrong) becomes safe. Finally, to match a normalized net rather than merely survive without one, scalar biases at the insertion points restore the shift that normalization's `beta` provided, and one scalar multiplier per branch restores the weight-norm-and-effective-learning-rate dynamics that scale invariance plus weight decay gave a normalized net for free. The whole thing is a pure initialization plus a small number of scalar parameters per branch — four biases and one multiplier in the two-conv block — no batch dependence, no running statistics, no train/test gap.
