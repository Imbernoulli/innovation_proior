Group normalization did exactly what I bet it would, and the numbers moved where I most wanted to see them: ResNet-56 climbed from the blend's 66.06 to 67.90, ResNet-110 from 68.65 to 70.43 — clear recovery on the two deep CIFAR-100 stacks, with the larger absolute gain on the deeper net, just as a stable per-image group statistic should give across many blocks. The one place it did *not* gain was MobileNetV2 / FashionMNIST, which slipped from 93.64 to 93.16 — and that slip is the tell. Its cause is built into the adaptive grouping rule: to stay legal on the small, awkwardly factored channel counts in MobileNetV2's inverted residuals, the rule caps groups at the channel count and decrements until divisibility, so on a 16- or 32-channel layer it collapses to one channel per group, which *is* instance normalization. On exactly the network with many narrow layers, group normalization quietly reverts to the instance-equivalent boundary I was trying to escape. More structurally, the rung made one commitment: a single fixed grouping and a single fixed pooling rule, identical at every layer, chosen by a hand-written heuristic rather than by the task. The deep ResNets rewarded that rigidity; the narrow MobileNetV2 layers were penalized by it. The headroom is the rigidity itself — different layers may want different statistics, and group normalization gives them no way to say so.

I propose **Switchable Normalization**: instead of *fixing* which statistic each layer uses, let the layer *learn* it. The family is already three points in one space — instance, layer, batch, each a different membership set. Rather than pick a fixed interior point on the channel-sharing dial, compute *all three* standardizations and let the layer learn a weighting over them, end to end from the classification loss. This strictly generalizes everything so far: if the learned weights collapse onto batch, the layer is batch normalization; onto instance or layer, those; any soft combination is a new operating point no single fixed rule could reach. The three statistics are not independent quantities to recompute from scratch but nested reductions of the same map. For $x$ of shape $[B, C, H, W]$: the instance statistic is per $(\mathrm{image}, \mathrm{channel})$, $\mathrm{mean}_{\mathrm{IN}}, \mathrm{var}_{\mathrm{IN}}$ over $(2,3)$; the layer statistic is per image, over $(1,2,3)$; the batch statistic is per channel, over $(0,2,3)$ — all with the biased variance, consistent with the standardization.

The mix is the heart of it. I keep two small sets of learnable weights — one for the mean, one for the variance — each a 3-vector over $\{\mathrm{IN}, \mathrm{LN}, \mathrm{BN}\}$, passed through a softmax so they form a convex combination summing to one: $\mathrm{mean\_w} = \mathrm{softmax}(\mathrm{mean\_weight})$, $\mathrm{var\_w} = \mathrm{softmax}(\mathrm{var\_weight})$, and

$$\mathrm{mean} = \sum_k \mathrm{mean\_w}_k\, \mathrm{mean}_k, \qquad \mathrm{var} = \sum_k \mathrm{var\_w}_k\, \mathrm{var}_k,$$

then standardize by the *combined* moments, $\hat{x} = (x - \mathrm{mean}) / \sqrt{\mathrm{var} + \epsilon}$, then one per-channel affine $\gamma \hat{x} + \beta$. Two design points are deliberate. First, I learn *separate* weights for the mean and the variance rather than one shared weighting, because the right centering and the right scaling need not come from the same statistic: a channel might want batch-level centering (to preserve between-image offset) while wanting instance-level scaling (to wash out per-image contrast magnitude), and two weight vectors let the layer express that decomposition. Second — and this pins the rung to the task's edit surface — the weights are *per-layer, shared across all channels*: a single 3-vector each, `nn.Parameter(torch.ones(3))` for the mean and the same for the variance, six numbers per layer, softmaxed and applied identically to every channel. That is a much lighter footprint than the first rung's per-channel gate (which carried $C$ parameters per layer), and it makes the comparison clean — the flexibility added over group normalization is *which corner of the family this layer leans toward*, decided globally for the layer. The softmax is what makes the weights a true selection: it forces them non-negative and summing to one, so the combination is a genuine convex interpolation among the three normalizations rather than an arbitrary linear combination that could amplify or cancel statistics and destabilize training.

I handle the batch statistic the same honest way as the earlier rungs, and it carries the one residual risk here. The textbook batch layer keeps a running mean/variance frozen for inference; I compute the batch statistic directly from the current batch in `forward`, in both train and eval, with no running buffer — stable at batch 128 in both phases and keeping the layer a single consistent function with no hidden state. The residual risk is that the BN corner inherits this on-the-fly dependence: to whatever extent the learned $\mathrm{mean\_w}_{\mathrm{BN}}$ / $\mathrm{var\_w}_{\mathrm{BN}}$ lean on batch, the eval behavior depends on the eval batch's composition. But the softmax mix can learn its way out — if leaning on batch hurts, the optimizer shifts weight onto the batch-free layer or instance corners — and that adaptivity is the point of the rung. The two weight 3-vectors are initialized all-ones so the softmax starts uniform: every layer begins as an equal blend of the three corners, with $\epsilon = 10^{-5}$ and the affine initialized to 1 and 0, and gradient descent slides each layer toward whatever corner, or interior point, the loss prefers.

Why this should beat group normalization, and where, is a generalization argument. Switchable normalization *contains* the previous rungs as special cases of its learned weights, so it can recover group normalization's good behavior on the deep ResNets by leaning toward the layer/instance corners it interpolated, while also doing something group normalization could not on the narrow MobileNetV2 layers — instead of being forced by the decrement heuristic onto the instance-equivalent boundary, it can lean on the *batch* corner there, which group normalization never had access to (group normalization is strictly batch-free). On MobileNetV2 / FashionMNIST, where the comfortable batch of 128 makes the batch statistic genuinely useful and the narrow-layer collapse hurt group normalization, recovering the batch corner per layer is exactly the missing degree of freedom, so this is where I expect the clearest gain over the prior rung — recovering and exceeding 93.16. On the two CIFAR-100 ResNets — 67.90 and 70.43 — I expect to match or modestly exceed, since the learned mix can reproduce a layer/instance-leaning blend close to what group normalization already found and add only the marginal benefit of letting centering and scaling pick different corners. The risk that would falsify the bet is the on-the-fly batch corner destabilizing a deep stack: if the optimizer over-weights batch on ResNet-110 and the eval-batch coupling bites, it could land at or below group normalization. If switchable normalization clears group normalization across the board — and especially if it reclaims the MobileNetV2 setting — it confirms that the productive move at the top of this ladder is to stop fixing the statistic per layer and *learn* the convex combination of the family's corners.

```python
class CustomNorm(nn.Module):
    """Switchable Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Learns to combine BatchNorm, InstanceNorm, and LayerNorm statistics via
    softmax-weighted importance weights. Adapts normalization strategy per
    channel during training.

    Reference: Luo et al., "Differentiable Learning-to-Normalize via
    Switchable Normalization" (ICLR 2019)
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Importance weights for mean (3 norms) and var (3 norms)
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Softmax over importance weights
        mean_w = F.softmax(self.mean_weight, dim=0)
        var_w = F.softmax(self.var_weight, dim=0)
        # Instance stats: per (B, C) over (H, W)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        # Layer stats: per B over (C, H, W)
        mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)
        var_ln = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        # Batch stats: per C over (B, H, W)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        # Weighted combination
        mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
        var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
```
