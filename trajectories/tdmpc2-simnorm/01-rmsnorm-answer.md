**Problem.** In a decoder-free latent world model the latent is defined only by self-predictive
consistency, reward regression, and a bootstrapped value loss whose target reads the network's own
latents. That self-referential setup leaves the latent's overall *scale* unpinned, and the value-target
feedback loop drives it — gradients can run away on harder tasks. The final activation of the encoder
and dynamics nets must bound the latent. LayerNorm only controls the pre-activations *inside* the MLP;
the latent is the output of a last linear map *after* it, so it stays unbounded.

**Key idea.** Use RMSNorm as the final activation, in a **group-wise** form matched to this task's
latent structure. LayerNorm bundles two operations — subtract the mean (re-centering) and divide by the
standard deviation (re-scaling) — buying two separate invariances. Only re-scaling controls activation
spread (`var(a−μ)=var(a)`: recentering does not touch variance); re-centering is dead weight here.
Dropping the mean leaves no μ, so divide by the root-mean-square instead of σ:
`āᵢ = aᵢ / √(mean(aᵢ²)) · gᵢ`. RMS scales linearly (`RMS(αa)=|α|·RMS(a)`), so re-scaling invariance —
the load-bearing one — survives; only re-centering is discarded. To make the swap apples-to-apples with
the default SimNorm, keep the same 16-groups-of-8 partition and apply the RMS rescale *within each
group*, with a learnable gain of size 8.

**Why it works.** Dividing each group by its RMS pins each group to a common magnitude, removing the
scale degree of freedom that drives the runaway loop, so gradients stay tame. The per-group structure
mirrors SimNorm so the comparison isolates the normalization, not the grouping. `eps` goes *inside* the
square root (`√(mean(x²)+eps)`) to keep the division and the sqrt gradient well-conditioned for groups
of only 8 entries. Its limit: RMSNorm bounds spread but induces *no* sparsity and *no* competition
between coordinates — the group is a bounded but dense blob, and the gain can re-inflate scale — so the
value head reads a structureless latent. That cost should be invisible on the easy tasks and visible on
cheetah-run.

**Hyperparameters.** `simnorm_dim = 8` (group size, 16 groups over the 128-dim latent); `eps = 1e-8`
inside the root; learnable gain `weight ∈ ℝ⁸` initialized to ones; same module instance as the final
activation of both the encoder MLP and the dynamics MLP; world model ~1M params, 200K training steps.

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
