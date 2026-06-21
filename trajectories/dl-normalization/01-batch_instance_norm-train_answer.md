The starting point is a plain `nn.BatchNorm2d` after every convolution, and the whole task reduces to a single choice — which feature positions share a mean and variance — with everything else (backbones, optimizer, schedule, data) frozen. The known-good batch layer trains these nets, but the lineage names a tension I can attack at the very first rung without opening any new design space: the batch statistic and the instance statistic fail in *opposite* directions on conv features. The batch layer pools per channel over $(N, H, W)$, so it keeps the between-image scale that carries class signal — but it also keeps whatever per-image contrast or style lives in that channel, because it never normalizes *within* an image. Instance normalization is the mirror image: it standardizes each channel of each image by that image's own spatial moments over $(H, W)$, which wipes out nuisance per-image style (exactly why it dominates style transfer) but also destroys the global magnitude that distinguishes one image's channel response from another's. One estimator keeps discriminative scale and nuisance style together; the other kills both. A single global choice cannot serve channels that want one behavior alongside channels that want the other.

I propose **Batch-Instance Normalization**: at each layer compute both standardizations and combine them with a *per-channel learnable gate*. Let $g_c \in [0,1]$ be the gate for channel $c$. Rather than constrain it in the optimizer, I store a raw real parameter $\rho_c$ and pass it through a sigmoid, $g_c = \sigma(\rho_c)$, so it stays in the unit interval for free. The mixed feature, before the affine, is

$$\hat{x} = g \cdot \hat{x}_{\mathrm{BN}} + (1 - g)\cdot \hat{x}_{\mathrm{IN}},$$

broadcasting $g$ over $(B, H, W)$ so each channel uses its own gate. At $g_c = 1$ the channel is pure batch normalization; at $g_c = 0$ it is pure instance normalization; in between it is a true blend of the two standardizations. The network learns, channel by channel and from the classification loss alone, how much of each image's own style to wash out versus how much of the batch-level scale to keep — style-like channels (low-level texture, color blobs) can slide toward instance while shape-carrying channels stay batch-leaning.

Several design choices are load-bearing. After the mixture I apply *one* per-channel affine $\gamma_c \hat{x} + \beta_c$ to the *combined* feature, not a separate scale-and-shift folded into each branch: the branches differ only in their standardization statistics, so once blended a single $(\gamma, \beta)$ suffices to set each channel's output distribution, and keeping the affine single holds the parameter count and the optimization surface as close to the plain batch layer as possible — so any accuracy difference is attributable to the *blend*, not to extra affine capacity. The only new degrees of freedom over the batch layer are the $C$ gate parameters, one per channel.

The batch branch deserves equal care, because the obvious heavier machinery is not needed here. The textbook batch layer keeps a running mean and variance, accumulated by an exponential average in training and frozen for inference, precisely so test time — where there may be no batch — still has a population statistic; that running buffer is also the source of the train/test discrepancy the lineage warns about. But here the batch is a fixed 128 in *both* train and eval, so I do not need a frozen buffer to have a stable statistic. I compute the batch mean and variance directly from the current batch in `forward`, identically in train and eval: $\mathrm{mean}_{\mathrm{BN}} = x.\mathrm{mean}$ over $(0,2,3)$, $\mathrm{var}_{\mathrm{BN}} = x.\mathrm{var}$ over $(0,2,3)$ with the biased (divide-by-$m$) estimator, and $\hat{x}_{\mathrm{BN}} = (x - \mathrm{mean}_{\mathrm{BN}}) / \sqrt{\mathrm{var}_{\mathrm{BN}} + \epsilon}$. The instance branch is the same arithmetic with the batch axis removed — moments over $(2,3)$ — giving a separate $(\mathrm{mean}, \mathrm{var})$ per $(\mathrm{image}, \mathrm{channel})$. The biased variance is the right one because it is the moment I am *standardizing by*, not an unbiased estimate I am reporting; consistency between the subtraction and the division is what matters. I take $\epsilon = 10^{-5}$ for the numerical floor. This keeps the layer a single consistent function of the input, with no hidden state and no momentum hyperparameter to tune.

The gate initialization sets the network's prior over the blend. I want the rung to begin life behaving like the standard batch layer, so I initialize $\rho_c = 1.0$, giving $g_c = \sigma(1) \approx 0.73$ — every channel starts about three-quarters batch, one-quarter instance, near the known-good behavior — and gradient descent then slides each channel's gate toward whichever end helps. Because $g \equiv 1$ reproduces the batch layer exactly, the blend is strictly more flexible than either endpoint and can in principle do no worse than batch normalization if the gates simply stay near 1; the bet is that letting some channels slide toward instance buys accuracy on top of that. Where I expect that bet to come back thin or negative is on the two deep CIFAR-100 ResNets, where between-image scale is most class-relevant: there the instance branch has the most to lose, and any gate the optimizer drops below 1 on a shape-carrying channel actively deletes signal, while the on-the-fly batch branch couples the standardization to batch composition. On near-ceiling MobileNetV2 / FashionMNIST, where per-image contrast is more nuisance than signal, the instance behavior on some channels could plausibly help. A thin or negative result on the ResNets would say "blending toward instance is not the lever here" and point the next rung at bringing more of the family — the layer statistic — into the mix.

```python
class CustomNorm(nn.Module):
    """Batch-Instance Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Learns a per-channel gate rho in [0, 1] (via sigmoid) that interpolates
    between BatchNorm statistics and InstanceNorm statistics.

    Reference: Nam & Kim, "Batch-Instance Normalization for Adaptively
    Style-Invariant Neural Networks" (NeurIPS 2018)
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Gate parameter (before sigmoid); init at 1.0 -> sigmoid ~ 0.73 -> mostly BN
        self.rho = nn.Parameter(torch.ones(num_features) * 1.0)

    def forward(self, x):
        # x: [B, C, H, W]
        gate = torch.sigmoid(self.rho).view(1, -1, 1, 1)
        # Batch stats: per C over (B, H, W)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        # Instance stats: per (B, C) over (H, W)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        # Interpolate
        x_bn = (x - mean_bn) / (var_bn + self.eps).sqrt()
        x_in = (x - mean_in) / (var_in + self.eps).sqrt()
        x_norm = gate * x_bn + (1 - gate) * x_in
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
```
