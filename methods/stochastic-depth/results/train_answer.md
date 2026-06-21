Depth is the strongest lever I have on the expressiveness of a convolutional network — every leap in the state of the art, from a handful of convolutional layers up to a hundred and fifty, has come by going deeper. But depth fights me on three fronts at once. Backward, the gradient is multiplied through layer after layer and shrinks toward nothing by the time it reaches the early layers, so they barely learn. Forward, the input features and whatever the early layers computed get repeatedly multiplied by weight matrices and washed out, so the late layers have little clean signal to build on — diminishing feature reuse. And practically, each forward and backward pass costs time linear in depth, so a $152$-layer network takes weeks to converge. The existing remedies each address only a slice of this. Careful initialization and Batch Normalization tame the vanishing gradient somewhat but leave the chain length and the cost untouched. ResNet's identity shortcuts make hundred-plus-layer training feasible by giving signal a clear path between layers, but every block is still paid for in full on every iteration, so the optimization chain and the wall-clock cost stay locked to the deployed depth. Dropout regularizes by masking individual activations, yet it makes the network thinner, not shorter, so it does nothing about the chain length that is the real disease — and its benefit is known to fade once Batch Normalization is present, buying essentially nothing on a $110$-layer BN-ResNet. Highway Networks can route around layers, but their gates are learned and parameterized, and ResNet already showed the simpler parameter-free skip trains better.

The tension looks like a forced choice — short nets train fast and flow information cleanly but lack capacity; deep nets have capacity but are slow and hard to optimize — until I notice that the two properties are needed in different phases. I want expressiveness at *test* time, when the model must represent complex concepts, and I want easy optimization and fast iteration at *training* time. So the real question is whether a network can be short while I train it and deep when I deploy it, under a single set of shared weights. I propose Stochastic Depth, and what makes it possible is the shape of the ResNet block. A residual block computes $H_\ell = \mathrm{ReLU}\!\left(f_\ell(H_{\ell-1}) + s_\ell(H_{\ell-1})\right)$, a transformation $f_\ell$ (a small stack, Conv-BN-ReLU-Conv-BN) added to a shortcut $s_\ell$ that is the identity for same-shape blocks, average-pool plus zero-padded channels at CIFAR/SVHN transitions, or a projection in ImageNet-style transitions. The bypass path is already there; I only have to be able to delete the transformation on a given step. So I gate the transformation — and only the transformation — with a Bernoulli variable $b_\ell \in \{0,1\}$ drawn fresh each mini-batch, with survival probability $p_\ell = \Pr(b_\ell = 1)$, and train with

$$H_\ell = \mathrm{ReLU}\!\left(b_\ell \cdot f_\ell(H_{\ell-1}) + s_\ell(H_{\ell-1})\right).$$

When $b_\ell = 1$ this is exactly the original block, nothing changes. When $b_\ell = 0$ the transformation term vanishes and the block collapses to $H_\ell = \mathrm{ReLU}(s_\ell(H_{\ell-1}))$. For the ordinary same-shape blocks this is a clean bypass: $s_\ell$ is the identity, the input $H_{\ell-1}$ is non-negative because it is the output of the previous ReLU (or of the initial Conv-BN-ReLU stem), and ReLU on a non-negative tensor is the identity, so $H_\ell = H_{\ell-1}$ exactly. For transition blocks I will not lie to myself — the shortcut downsamples and zero-pads channels, or projects, so it is not literally the same tensor — but the expensive $f_\ell$ branch is still gone, with no forward or backward compute through it and no parameter update for its weights. That is the operational shortening I was after.

The remaining design choice is how to set the survival probabilities, and I do not think every block deserves the same one. The early layers extract low-level features — edges, textures — that every later layer builds on; dropping an early block corrupts the foundation the whole rest of the network depends on for that step, whereas a late block's transformation is more specialized and less universally relied upon. So survival should decrease with depth, and the gentlest schedule with that property is a straight line. I anchor the input to the first block as always present, $p_0 = 1$, and let survival decay linearly to $p_L$ at the last block:

$$p_\ell = 1 - \frac{\ell}{L}\,(1 - p_L).$$

This leaves a single free knob, $p_L$, and training is remarkably insensitive to it, so I fix $p_L = 0.5$: the last block survives half the time, the first essentially always. The quantity that governs both the speedup and the optimization benefit is the effective depth $\tilde L$, the number of surviving blocks in a step — a sum of independent Bernoullis, so its expectation is the sum of the survival probabilities, $\mathbb{E}(\tilde L) = \sum_{\ell=1}^{L} p_\ell$. Substituting the linear decay with $p_L = 0.5$, so that $p_\ell = 1 - \ell/(2L)$,

$$\mathbb{E}(\tilde L) = \sum_{\ell=1}^{L}\left(1 - \frac{\ell}{2L}\right) = L - \frac{1}{2L}\cdot\frac{L(L+1)}{2} = L - \frac{L+1}{4} = \frac{3L - 1}{4} \approx \frac{3L}{4}.$$

For the $110$-layer CIFAR ResNet, which is $L = 54$ residual blocks, that is $\mathbb{E}(\tilde L) \approx (162 - 1)/4 \approx 40$: I train a network that is on average $40$ blocks deep and at test time deploy all $54$. Since the dropped blocks need no forward or backward pass, the saved fraction is about $1 - 3/4 \approx 25\%$ of training time; pushing $p_L$ down to $0.2$ buys roughly $40\%$ on CIFAR-10 at the same test error.

At test time I want the full model — all $f_\ell$ active, all the capacity — but here I must be careful. During training, block $\ell$'s transformation was present only a fraction $p_\ell$ of the time, and everything downstream adapted its weights to that intermittent presence. Turning $f_\ell$ on for every test example would make its contribution to the sum larger, on average, than what the downstream weights were calibrated against, by a factor of $1/p_\ell$. This is exactly Dropout's situation, and the fix is the same: scale the transformation by the probability it was present, so its expected contribution matches training. The test forward rule is

$$H_\ell^{\text{Test}} = \mathrm{ReLU}\!\left(p_\ell \cdot f_\ell(H_{\ell-1}^{\text{Test}}) + s_\ell(H_{\ell-1}^{\text{Test}})\right).$$

The residual branch is weighted by its survival probability, the shortcut passes through full, and the statistics line up with what training produced. There is a second payoff I did not design for but that falls out of the structure, and it explains why test error drops rather than merely training time. Each of the $L$ blocks is independently on or off, so a single set of shared weights defines $2^L$ different sub-networks — every subset of present blocks is a different, shorter network. Each mini-batch samples one of these $2^L$ networks and updates it, so over training I am jointly training an enormous family of networks of *varying depth* that share weights, and the survival-weighted test rule combines them into one network where each block contributes in proportion to how often it was present — an implicit average over the ensemble. Because the members differ in depth, not just in which units are thinned, the diversity is higher than a same-depth ensemble. On top of that, the shorter expected training depth means shorter gradient chains, so the gradients reaching the early layers are stronger, directly attacking the vanishing-gradient problem I started from. This is why Stochastic Depth is complementary to Dropout rather than a competitor: I drop whole blocks to make the network shorter, not thinner, which is the right tool for the depth problem specifically and why it keeps working with Batch Normalization where Dropout does not. In implementation it is convenient to store the complement, the death rate $1 - p_\ell$, draw one Boolean gate per mini-batch, add $f$'s output to the shortcut when the gate is open and return the shortcut alone (never computing $f$) when it is closed, and at test always add $f$'s output but scaled by $1 - \text{deathRate} = p_\ell$, with the network assigning each block its death rate by a linear ramp.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDropBlock(nn.Module):
    """H = ReLU(skip(x) + gate * f(x)); one gate is sampled per mini-batch."""
    def __init__(self, in_ch, out_ch, stride=1, death_rate=0.0):
        super().__init__()
        self.death_rate = death_rate
        self.stride, self.in_ch, self.out_ch = stride, in_ch, out_ch
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def skip(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, self.stride, self.stride)
        if self.out_ch > self.in_ch:
            zeros = x.new_zeros(x.size(0), self.out_ch - self.in_ch, x.size(2), x.size(3))
            x = torch.cat([x, zeros], dim=1)           # zero-pad channels (option A)
        return x

    def forward(self, x):
        skip = self.skip(x)
        if self.training:
            if torch.rand((), device=x.device).item() >= self.death_rate:
                return self.relu(self.f(x) + skip)
            return self.relu(skip)                      # dropped block: shortcut only, no f compute
        return self.relu((1.0 - self.death_rate) * self.f(x) + skip)

class StochasticDepthResNet(nn.Module):
    def __init__(self, block_chs, num_classes=10, p_L=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, block_chs[0][0], 3, padding=1, bias=False),
            nn.BatchNorm2d(block_chs[0][0]), nn.ReLU(inplace=True),
        )
        L = len(block_chs)
        blocks = []
        for ell, (in_ch, out_ch, stride) in enumerate(block_chs, start=1):
            death_rate = (ell / L) * (1.0 - p_L)        # deathRate = 1 - p_ell
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
