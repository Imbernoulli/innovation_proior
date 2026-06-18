Let me lay out the tension I'm actually stuck on, because the whole thing hinges on it. Depth buys expressiveness — every jump in the state of the art, from a handful of conv layers to a hundred and fifty, has come with more depth. But depth fights me three ways at once. Backward, the gradient gets multiplied through layer after layer and shrinks to nothing in the early layers, so they barely train. Forward, the input features and whatever the early layers computed get repeatedly multiplied by weight matrices and washed out, so the late layers have little clean signal to work with. And practically, every forward and backward pass costs time linear in depth, so a 152-layer net takes weeks.

So I'm caught: a short network trains fast and lets information flow cleanly both directions, but it's not expressive enough; a deep network is expressive but slow and hard to optimize. Stated like that it sounds like I just have to pick. But look at *when* each property matters. I need expressiveness at *test* time — that's when the model has to represent complex concepts. I need easy optimization and fast iteration at *training* time. Those are different phases. What if the network could be short while I train it and deep when I deploy it?

That seems impossible for a fixed architecture — the depth is the depth. Unless I can make some layers *not count* during a training step. And here ResNet hands me the tool, because of the shape of its block. A residual block computes H_ℓ = ReLU(f_ℓ(H_{ℓ-1}) + s_ℓ(H_{ℓ-1})): a transformation f_ℓ added to a shortcut s_ℓ. Usually s_ℓ is just the identity; at a stage transition it is the shape-matching shortcut, such as average-pooling plus zero-padding on CIFAR/SVHN or a projection in an ImageNet bottleneck. The bypass path is already there. So if, on a given training step, I could just *delete* f_ℓ and keep only the shortcut, that block would become the cheap path alone — the network would behave, for that step, as if the residual transformation weren't there. The depth I actually backpropagate through would shrink.

Make that concrete. For each block introduce a Bernoulli gate b_ℓ ∈ {0,1}, drawn fresh each mini-batch, and multiply the transformation by it:

  H_ℓ = ReLU( b_ℓ · f_ℓ(H_{ℓ-1}) + s_ℓ(H_{ℓ-1}) ).

When b_ℓ = 1 this is exactly the original ResNet block — nothing changes. When b_ℓ = 0 the f_ℓ term vanishes and I'm left with H_ℓ = ReLU(s_ℓ(H_{ℓ-1})). Now, is that a clean bypass? For the normal same-shape blocks, yes: s_ℓ is identity, the input H_{ℓ-1} is non-negative because it is the output of the previous ReLU (or the initial Conv-BN-ReLU stem), and ReLU on a non-negative tensor is the identity. So in those blocks H_ℓ = H_{ℓ-1} exactly. For transition blocks, I should not lie to myself: the shortcut may downsample and zero-pad channels, or use a projection in the ImageNet variant, so it is not literally H_{ℓ-1} in the same tensor space. But the residual transformation is still gone; the block reduces to the shortcut branch alone, with no compute through f_ℓ and no parameter update for its weights. That's the operational shortening I need.

That immediately delivers the training-time speedup and the shortening. But it also raises the question of how to set the drop probabilities. Call p_ℓ = Pr(b_ℓ = 1) the survival probability of block ℓ. These are new hyperparameters. The simplest choice is uniform: every block survives with the same p_L. But I don't think every block should be treated equally. The early layers extract low-level features — edges, textures — that *every* later layer builds on. If I drop an early block, I'm corrupting the foundation that the whole rest of the network depends on for that step. A late block's transformation is more specialized and less universally relied upon. So early blocks should be present more reliably than late ones: survival should *decrease* with depth.

The gentlest schedule with that property is a straight line. Anchor it: treat the input to the first block as always present (p_0 = 1), and let survival decay linearly to some p_L at the last block:

  p_ℓ = 1 − (ℓ/L)(1 − p_L).

At ℓ = 0 this is 1; at ℓ = L it is p_L; in between it slides down linearly. One free knob, p_L. And training turns out to be remarkably insensitive to it, so I'll fix p_L = 0.5 — the last block survives half the time, the first essentially always.

Let me work out what this does to the effective depth, because that's the quantity that determines both the speedup and the optimization benefit. The number of surviving blocks in a step, L̃, is a random variable, a sum of independent Bernoullis, so its expectation is just the sum of the survival probabilities:

  E(L̃) = Σ_{ℓ=1}^L p_ℓ.

Plug in the linear decay with p_L = 0.5, so 1 − p_L = 1/2 and p_ℓ = 1 − ℓ/(2L):

  E(L̃) = Σ_{ℓ=1}^L [1 − ℓ/(2L)] = L − (1/(2L)) Σ_{ℓ=1}^L ℓ = L − (1/(2L)) · L(L+1)/2 = L − (L+1)/4 = (4L − L − 1)/4 = (3L − 1)/4.

So E(L̃) = (3L − 1)/4 ≈ 3L/4 for large L. For the 110-layer CIFAR ResNet, which is L = 54 residual blocks, that's E(L̃) ≈ (162 − 1)/4 ≈ 40. I train a network that is, on average, 40 blocks deep, and at test time I get to use all 54. And since the dropped blocks need no forward/backward, the saved fraction is about 1 − 3/4 = 25% of training time. If I want more, I lower p_L: at p_L = 0.2 the same CIFAR-10 test error comes out as constant depth but with roughly 40% speedup.

Now the test-time question, which is where I have to be careful again. At test time I want the full model — all f_ℓ active, all the capacity. But during training, block ℓ's transformation was only present a fraction p_ℓ of the time, and everything downstream of it adapted its weights to that intermittent presence. If I suddenly turn f_ℓ on for *every* test example, its contribution to the sum is, on average, larger than what the downstream weights were calibrated against — by a factor of 1/p_ℓ. This is exactly the situation Dropout faces, and the fix is the same: scale the transformation at test time by the probability it was present, so its expected contribution matches training. The test forward rule becomes

  H_ℓ^Test = ReLU( p_ℓ · f_ℓ(H_{ℓ-1}^Test) + s_ℓ(H_{ℓ-1}^Test) ).

So at test the residual branch is weighted by its survival probability, the shortcut passes through full, and the statistics line up with what training produced.

There's a second payoff I didn't design for but that falls out of the structure, and it's worth seeing because it explains why test error *drops*, not just training time. Each of the L blocks is independently on or off, so a single set of shared weights actually defines 2^L different sub-networks — every subset of blocks present gives a different, shorter network. Each mini-batch samples one of these 2^L networks and updates it; over training I'm jointly training an enormous family of networks of *varying depth* that share weights. At test time, the survival-weighted forward rule combines them into one network where each block contributes in proportion to how often it was present — an implicit average over the ensemble. And because the members differ in depth, not just in which units are thinned, the diversity is higher than a same-depth ensemble. That's the ensemble interpretation; on top of it, the shorter expected training depth means shorter gradient chains, so the gradients reaching the early layers are stronger, which directly attacks the vanishing-gradient problem I started from.

I should sanity-check this against the obvious alternative, Dropout, since both are Bernoulli-masking schemes. Dropout multiplies individual hidden activations by Bernoulli variables — it makes the network *thinner*. That regularizes by breaking co-adaptation, but it does nothing about the *length* of the forward and backward chains, which is the actual disease in a very deep net. And Dropout's benefit is known to fade once Batch Normalization is in the picture — on a 110-layer BN ResNet, sweeping Dropout rates buys essentially nothing. What I'm doing is orthogonal: I drop whole *blocks*, making the network *shorter* rather than thinner, which is why it keeps working with Batch Normalization and why it directly relieves vanishing gradients and diminishing feature reuse. So this is complementary to Dropout, not a competitor — and it's the right tool for the depth problem specifically.

Time to make it concrete in code. A residual block holds its transformation branch f (Conv-BN-ReLU-Conv-BN) and a shortcut (identity for same-shape blocks; average-pool plus zero-padded extra channels when the CIFAR/SVHN stage widens). The canonical Torch code stores the complement, `deathRate = 1 - p_ℓ`, and a Boolean gate. In training, it draws the gate once per mini-batch; if open, it adds f's output to the shortcut; if closed, it returns the shortcut alone and never computes f. In testing, it always adds f's output but scales it by `1 - deathRate = p_ℓ`. The network sets each block's death rate by a linear ramp. I'll write that behavior in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDropBlock(nn.Module):
    # H = ReLU(skip(x) + gate * f(x)); one gate is sampled for the whole mini-batch.
    def __init__(self, in_ch, out_ch, stride=1, death_rate=0.0):
        super().__init__()
        self.death_rate = death_rate
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.stride = stride
        self.in_ch, self.out_ch = in_ch, out_ch
        self.relu = nn.ReLU(inplace=True)

    def skip(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, self.stride, self.stride)
        if self.out_ch > self.in_ch:
            zeros = x.new_zeros(x.size(0), self.out_ch - self.in_ch, x.size(2), x.size(3))
            x = torch.cat([x, zeros], dim=1)
        return x

    def forward(self, x):
        skip = self.skip(x)
        if self.training:
            if torch.rand((), device=x.device).item() >= self.death_rate:
                return self.relu(self.f(x) + skip)
            return self.relu(skip)
        else:
            return self.relu((1.0 - self.death_rate) * self.f(x) + skip)

class StochasticDepthResNet(nn.Module):
    def __init__(self, block_chs, num_classes=10, p_L=0.5):
        super().__init__()
        self.stem = nn.Sequential(                            # input always non-negative after this
            nn.Conv2d(3, block_chs[0][0], 3, padding=1, bias=False),
            nn.BatchNorm2d(block_chs[0][0]), nn.ReLU(inplace=True),
        )
        L = len(block_chs)
        blocks = []
        for ell, (in_ch, out_ch, stride) in enumerate(block_chs, start=1):
            death_rate = (ell / L) * (1.0 - p_L)              # deathRate = 1 - p_ell
            blocks.append(ResidualDropBlock(in_ch, out_ch, stride, death_rate=death_rate))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_chs[-1][1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

The causal chain: depth is needed for expressiveness but breaks training through vanishing gradients, diminishing feature reuse, and linear-in-depth cost — and what I really want is a short net during training and a deep one at test. ResNet's shortcut path makes that achievable: gate each block's transformation with a per-mini-batch Bernoulli so a dropped same-shape block becomes an exact identity, while a dropped transition block becomes only its shape-matching shortcut. Survival should fall with depth, since early low-level features are universally reused, so I use a linear decay p_ℓ = 1 − (ℓ/L)(1−p_L) with p_L = 0.5, giving expected training depth (3L−1)/4 ≈ 3L/4 (≈40 of 54 blocks) and ~25% time saved. At test, all blocks run with f scaled by p_ℓ to match training statistics. The scheme is simultaneously a depth-shortener (stronger gradients, faster training) and an implicit ensemble over 2^L depth-varying sub-networks — which is why test error falls too, and why, unlike Dropout, it keeps working with Batch Normalization.
