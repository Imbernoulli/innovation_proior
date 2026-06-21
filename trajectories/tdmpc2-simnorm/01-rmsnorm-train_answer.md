The latent in this world model is defined by nothing external. The encoder sends an observation to a 128-dim latent, the dynamics net predicts the next latent from latent and action, and reward and value heads read off that latent — and the losses that pin it are all self-referential: a self-predictive consistency term $\lVert d(z,a) - \mathrm{sg}(h(s'))\rVert^2$ with a stop-gradient on the target, a reward regression, and a bootstrapped value loss whose target $q = r + \gamma\,\bar Q(z', p(z'))$ is itself a function of the very latents the network produces. That self-reference is where instability lives: a squared-error-in-feature-space consistency loss has a cheap, useless way to go down that has nothing to do with dynamics — change the overall *scale* of the representation. If the last linear map can inflate or deflate the latent freely, the losses get a free parameter, and because the value target reads the network's own latents this closes a feedback loop with a runaway direction. The exploding gradients people see on harder tasks are that magnitude blowing up. The intermediate LayerNorms do not save me: they normalize the pre-activations flowing *through* each layer, but the latent I actually consume is the output of a last linear map *after* the final LayerNorm, and that map can have arbitrarily large weights. LayerNorm controls the river, not the mouth. So the final activation's first job is structural — remove the magnitude degree of freedom and bound the latent.

I propose **group-wise RMSNorm** as that final activation. The reasoning starts from a dissection of LayerNorm: it bundles two distinct operations that feel like one move ("standardize") but buy different things. Subtracting the mean recenters the vector and buys *re-centering invariance* — add a constant to every entry and the centered vector is unchanged. Dividing by the standard deviation rescales it and buys *re-scaling invariance* — scale every entry by $\alpha$ and $(a-\mu)/\sigma$ is unchanged. Stabilizing training is fundamentally about controlling *spread*, and the mean is the suspicious operation here: recentering moves the cloud but does nothing to its variance, $\mathrm{var}(a-\mu)=\mathrm{var}(a)$. The thing that actually pins magnitude to a fixed scale regardless of how big the weights grew is the *division by a scale statistic*. So I keep the re-scaling and discard the re-centering as dead weight — and in this model that is exactly the right trade, because the degree of freedom re-scaling pins is precisely the runaway scale I diagnosed.

If I drop the mean I cannot divide by $\sigma$, because $\sigma = \sqrt{\tfrac1n\sum_i(a_i-\mu)^2}$ is *defined* through $\mu$ — it is the spread around the mean. Refusing to compute $\mu$, I need a scale that never references it: the spread around *zero* instead of around the mean, the root-mean-square $\mathrm{RMS}(a)=\sqrt{\tfrac1n\sum_i a_i^2}$. The normalization becomes

$$\bar a_i = \frac{a_i}{\mathrm{RMS}(a)}\,g_i,$$

with a learned gain $g$, no $\mu$ anywhere and a single statistic. Two checks confirm it is the right tool. First, the degenerate case: if the entries happen to have zero mean then $\sigma=\mathrm{RMS}(a)$ and this rule coincides exactly with LayerNorm — so it is not a wild departure, it is LayerNorm with re-centering switched off. Second, it does the one job I need: $\mathrm{RMS}(\alpha a)=|\alpha|\,\mathrm{RMS}(a)$, so $\alpha a / \mathrm{RMS}(\alpha a) = a/\mathrm{RMS}(a)$ — re-scaling invariance, the load-bearing one, survives, and only re-centering, the weaker suspect, is discarded.

Now the part that deviates from textbook RMSNorm, deliberately. The default activation in this slot is SimNorm, which reshapes the 128-dim latent into 16 groups of $\texttt{simnorm\_dim}=8$ and operates within each group; the whole harness — the consistency loss, the heads — is built around a latent that has this group structure as its native shape. The cleanest apples-to-apples swap is to keep the *same partition* and apply the RMS rescale *within each group of 8* rather than over the full 128-vector. So I reshape to $(\ast\text{batch}, 16, 8)$, compute the root-mean-square over the last axis (the 8 entries of a group), divide, multiply by a learnable gain of size 8 shared across groups, and reshape back. This normalizes each of the 16 groups to a common magnitude independently, mirroring SimNorm's grouping so the comparison isolates "softmax-on-a-simplex vs. RMS-rescale" instead of confounding it with "16 groups vs. one block." The gain is per group-element (size 8) so each coordinate within a group can be re-weighted, the same role the learned gain plays in LayerNorm, initialized to ones so the layer starts as a pure RMS rescale.

One numerical choice matters. I put $\varepsilon$ *inside* the square root, $\mathrm{rms}=\sqrt{\mathrm{mean}(x^2)+\varepsilon}$, rather than clamping the norm from below afterward. With groups of only 8 entries the mean-square can get small; adding $\varepsilon$ under the root keeps the division well-conditioned without ever producing a zero denominator, and keeps the sqrt gradient finite at the origin. I use $\varepsilon = 10^{-8}$, small enough not to bias the scale when a group has real magnitude.

I want to be clear-eyed about what this does *not* buy, because it is why this is the first rung and not the last. RMSNorm bounds the *spread* of each group — after the divide each group has root-mean-square one, up to the gain — so the magnitude degree of freedom that drives the runaway loop is controlled and gradients should stay tame. But it does nothing else. It induces no sparsity: every coordinate is generically nonzero, the group is a bounded but dense blob. It induces no competition between coordinates: each entry is rescaled by the same scalar, so there is no pressure to prioritize a few directions. And the learnable gain can partially re-inflate the scale per coordinate, loosening the very bound I imposed. For a latent that must support stable bootstrapped value learning, a bounded-but-shapeless code is a weak representation — the value head reads a dense vector with no structure to exploit. I expect that cost to be invisible on walker-walk and cartpole-swingup, where almost any bounded latent saturates the reward, and visible on cheetah-run, whose running gait is dynamic enough that the value head must read fine structure out of the latent. If the numbers come back saturated on the two easy tasks and visibly lower on cheetah-run, the diagnosis for the next rung writes itself: bounding the magnitude is necessary but not sufficient, and the missing ingredient is *structure* — a within-group competition that biases the latent toward a sparse, overcomplete code.

```python
# EDITABLE region of custom_simnorm.py (lines 16-43) -- step 1: group-wise RMSNorm
class CustomSimNorm(nn.Module):
    """Group-wise RMSNorm baseline for latent representations."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim
        self.eps = 1e-8
        # Learnable gain per group element
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        shp = x.shape
        # Reshape into groups (same as SimNorm)
        x = x.view(*shp[:-1], -1, self.dim)
        # RMS normalization within each group
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x / rms) * self.weight
        return x.view(*shp)

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim}, type=RMSNorm)"
```
