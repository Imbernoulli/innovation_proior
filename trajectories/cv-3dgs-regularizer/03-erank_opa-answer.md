**Problem.** The compactness L1 (step 2) gained most on the floater-rich outdoor scenes but barely moved
fully-observed bonsai (32.531 → 32.685, +0.154). A size-and-count prior controls *how big* and *how many*
Gaussians there are, not *what shape* each one is: a small needle is still a needle. The residual on
high-detail, well-constrained geometry is **shape** — extremely elongated "needle" Gaussians (one
variance dominant, two near zero) that tile training views but spike in held-out views. The fix is a
parameter-only, differentiable, `O(N)` **shape** term added on top of the compactness L1.

**Key idea.** A Gaussian's shape is the spread of the three squared scales of `Σ = R diag(s₁²,s₂²,s₃²) Rᵀ`.
Measure it with the **effective rank** (Roy & Vetterli, 2007), the differentiable surrogate for matrix
rank:

```
q_j   = s_j² / (s₁² + s₂² + s₃²)     # normalize squared scales into a distribution
H(q)  = - Σ_j q_j log q_j            # Shannon entropy
erank = exp(H)
```

`erank ≈ 1` for a needle, `2` for a disk, `3` for a sphere, smooth between. Disks (`erank ≈ 2`) cover
surface area and survive viewpoint changes; needles do not. Penalize with a **one-sided log-barrier** and
break its scale-invariance with the smallest scale:

- `max(-log(erank - 1 + eps), 0)`: the negative log blows up at the needle limit; `eps = 1e-5` makes the
  value at `erank = 1` a finite wall (`-log(eps) ≈ 11.5`). At `erank = 2` the raw value is `-log(1+eps) <
  0`, so the clamp returns **exactly 0**; for `erank > 2` it stays 0. Steep near the needle, gentle near
  the disk, off above — no hard ban on slender primitives.
- `+ s_min` (smallest scale): `erank` depends only on scale *ratios*, so the barrier alone could be
  satisfied by inflating the two small axes into a blob (density off-surface). Absolute pressure on the
  smallest scale makes *flattening* the cheap way to raise `erank`, yielding a flat disk near the surface.

**Schedule.** Apply the shape prior only after warmup (`step ≥ 7000`), coarse-to-fine: Gaussians start
roughly isotropic (`erank > 2`) and early densification must run first; the compactness L1 runs throughout.

**Densification compatibility (fixed in the pipeline).** The harness renders with `absgrad=True`, so
`DefaultStrategy` splits on the absolute (sum-of-magnitudes) view-space gradient. Disks cover many pixels
whose gradients would cancel under the norm-of-sum rule; the absgrad accumulation keeps disks splitting, so
the shape prior and the densifier pull the same way. This is held fixed, not part of the regularizer.

**Hyperparameters.** `SCALE_REG = OPACITY_REG = 1e-2` (compactness, always on). `ERANK_REG = 1e-2` multiplies
the **whole** shape contribution — `barrier.mean() + s_min.mean()` together — so the flattening term is a
`1e-2`-scale tie-breaker on *how* `erank` is raised, not a full-strength force. `ERANK_WARMUP = 7000`,
`ERANK_EPS = 1e-5`. Full strength `1e-2` scale_opa (not the half-strength a stand-alone erank uses) is kept
here because the extra compactness pressure helps the indoor scene; no anisotropy term (it over-regularizes
stump).

```python
# EDITABLE region of gsplat/custom_regularizer.py — step 3: scale_opa (full) + erank log-barrier (warmup 7000)
import torch
import torch.nn.functional as F

# scale_opa (full strength) + erank log-barrier (warmup at step 7000).
SCALE_REG = 1e-2
OPACITY_REG = 1e-2
ERANK_REG = 1e-2
ERANK_WARMUP = 7000
ERANK_EPS = 1e-5

def compute_regularizer(splats, step, scene_scale):
    """Compactness L1 (always on) + erank log-barrier (after warmup)."""
    s = torch.exp(splats["scales"])                                # [N, 3]
    a = torch.sigmoid(splats["opacities"])                         # [N]

    loss = SCALE_REG * s.mean() + OPACITY_REG * a.mean()

    if step >= ERANK_WARMUP:
        s_sq = s * s
        q = s_sq / (s_sq.sum(dim=-1, keepdim=True) + 1e-12)
        H = -(q * (q + 1e-12).log()).sum(dim=-1)
        erank = H.exp()
        barrier = torch.clamp(-torch.log(erank - 1.0 + ERANK_EPS), min=0.0)
        s_min = s.min(dim=-1).values
        loss = loss + ERANK_REG * (barrier.mean() + s_min.mean())

    return loss
```
