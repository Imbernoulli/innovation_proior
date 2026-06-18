**Problem.** Group-wise RMSNorm bounded the latent's magnitude but supplied no *structure*: a single
scalar rescales all eight entries of a group identically, so there is no competition between
coordinates and the learned latent stays dense. On the discriminating task (cheetah-run) a bounded-but-
dense code gave the value head too little to read. The fix must keep the boundedness and *add* a
within-group competition that biases the latent toward a sparse, overcomplete code — without an L1
penalty to tune.

**Key idea.** SimNorm (Simplicial Normalization): treat the latent as 16 independent groups of 8 and
project each group onto a simplex with a softmax. Reshape `(*batch,128) → (*batch,16,8)`, softmax over
the last axis so each group becomes eight nonnegatives summing to one — a point on the 7-simplex — then
reshape back. It is the smooth, differentiable relaxation of a hard vector-of-categoricals (VQ-VAE)
code: `τ→0` recovers one-hot groups, `τ→∞` gives trivial uniform groups, and `τ=1` (a plain softmax)
sits in the useful middle. It reuses the exact partition the RMSNorm rung used; the only change is
*within each group*, from "divide by the RMS scalar" to "softmax." Applied as the final activation of
both the encoder and the dynamics net so the consistency loss compares prediction and target in the same
space.

**Why it works.** *Tighter bound, no parameters:* within a group `‖gᵢ‖₁=1` and `‖gᵢ‖₂≤1`, so over 16
groups `‖z‖₁=16` and `‖z‖₂≤4` — structural and exact, with no learnable gain that could re-inflate
scale (unlike RMSNorm). *Structure:* softmax is a zero-sum competition (entries sum to one), so to lower
the loss the network must prioritize a few entries per group — the latent drifts toward soft one-hot,
sparse and competitive, exactly what RMSNorm could not produce. *Expressive:* 16 independent simplices
give up to `8¹⁶` configurations (~48 bits) while every group stays bounded — boundedness and capacity
stop fighting; a single softmax over all 128 would bottleneck to ~7 bits. `τ=1` keeps the sparsity bias
while leaving the softmax Jacobian non-degenerate so gradients flow through the dynamics rollout.

**Hyperparameters.** `simnorm_dim = 8` (group size; 16 groups over the 128-dim latent); temperature
`τ = 1` (plain softmax, no tunable constant); no learnable parameters; same module instance as the final
activation of both the encoder MLP and the dynamics MLP; world model ~1M params, 200K training steps.

```python
# EDITABLE region of custom_simnorm.py (lines 16-43) -- step 2: SimNorm (simplicial normalization)
class CustomSimNorm(nn.Module):
    """SimNorm baseline -- original simplicial normalization from TD-MPC2."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim}, type=SimNorm)"
```
