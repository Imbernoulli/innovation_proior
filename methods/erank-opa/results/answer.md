# Effective-rank regularization for 3D Gaussian Splatting (with compactness L1), distilled

A per-scene 3D Gaussian Splatting optimization driven only by a photometric loss is shape-blind:
the optimizer overfits training views with extremely elongated "needle" Gaussians (one variance
dominant, two near zero) that show up as spikes in novel views. The fix is a parameter-only,
differentiable, `O(N)` penalty added to the photometric loss that drives each Gaussian toward a
flat, disk-like shape. It has two parts: a compactness L1 (inherited from 3DGS-MCMC) that controls
size and count, and the **effective-rank shape prior** — a one-sided log-barrier on each Gaussian's
effective rank plus a smallest-scale flattening term.

## Problem it solves

Add a scalar regularizer, computed only from the Gaussian parameters (no depth/normal/feature
supervision), differentiable, `O(N)`, numerically safe, that prevents primitives from collapsing into
needles under the standard novel-view and geometry objectives — without over-flattening or punishing
legitimately thin structures.

## Key idea

A Gaussian's covariance is `Sigma = R diag(s_1^2, s_2^2, s_3^2) R^T`; its shape is the spread of the
three squared scales. Measure that spread with the **effective rank** (Roy & Vetterli 2007), the
real-valued differentiable surrogate for matrix rank:

```
q_i   = s_i^2 / (s_1^2 + s_2^2 + s_3^2)          # normalize squared scales into a distribution
H(q)  = - sum_i q_i log q_i                       # Shannon entropy
erank = exp(H(q))
```

By construction `erank = 1` for a needle (`q ≈ (1,0,0)`), `2` for a disk (`q ≈ (.5,.5,0)`), `3` for a
sphere (`q = (1/3,1/3,1/3)`), and varies smoothly between. For surface reconstruction, planar
disk-like Gaussians (`erank ≈ 2`) cover area and survive viewpoint changes; needles (`erank ≈ 1`) do
not.

Penalize needles with a **one-sided log-barrier** and break its scale-invariance with the smallest
scale:

```
L_erank = sum_k [ lambda_erank * max( -log( erank(G_k) - 1 + eps ), 0 ) + s_{k,3} ]
```

- `max(-log(erank - 1 + eps), 0)`: the underlying negative log blows up at the needle limit; with
  `eps = 1e-5`, the value at `erank = 1` is the finite wall `-log(eps) ≈ 11.5`. At `erank = 2`, the
  raw value is `-log(1 + eps) < 0`, so the clamp returns **exactly 0**; for `erank > 2` it stays
  clamped to zero. For `erank ∈ (1, 2)`, the penalty decays toward zero near the disk and rises
  sharply near the needle, discouraging the pathology without a hard ban on slender primitives.
- `+ s_{k,3}` (smallest scale): `erank` depends only on scale *ratios*, so the barrier alone could be
  satisfied by inflating the two small axes into a blob (density off-surface). Adding absolute
  pressure on the smallest scale makes *flattening* (keep the third axis thin) the cheap way to raise
  `erank`, yielding a flat disk near the surface.

**Schedule:** apply the shape prior only after a warmup (step 7000), coarse-to-fine — Gaussians start
roughly isotropic (`erank > 2`) and early densification must run first; switching the barrier on later
keeps early training stable and refines geometry that already exists. The compactness L1 runs
throughout.

**Densification compatibility (fixed elsewhere in the pipeline):** standard ADC splits on the *norm
of the summed* view-space gradient; disks cover many pixels whose gradients cancel, so they fail to
split. A revised rule using the *sum of norms* (Bulò et al. 2024; Yu et al. 2024) splits disks
properly, so the shape prior does not starve the densifier once disks cover larger screen regions.

## Defaults and why

- `lambda_erank = 1e-2` on the log-barrier, scale/opacity L1 `= 1e-2` each: the photometric loss is
  `~0.03–0.1`; keeping each regularizer in `~1e-4..1e-1` avoids swamping the data term.
- `+ s_3`: the smallest-scale term is added outside `lambda_erank`, matching the loss
  `lambda_erank * barrier + s_3`; it closes the scale-invariance loophole.
- `eps = 1e-5`: floors the barrier at the needle (finite value and gradient) while leaving the
  clamped value exactly zero at `erank = 2`.
- warmup `= 7000`: coarse-to-fine; let geometry and primitive count settle before imposing shape.
- effective rank from **squared** scales: the singular values of `Sigma = R diag(s^2) R^T` are the
  squared scales, not the scales.

## Why effective rank over the alternatives

- vs **integer rank**: non-differentiable and brittle (can't separate near-needle from near-disk in
  floating point); `exp(H)` is smooth and gives gradients.
- vs **flatness / one-small-scale penalties**: a single small scale is ambiguous between disk and
  needle; `erank` folds all three axes into one number and separates them.
- vs **aspect-ratio / pairwise-variance caps**: two-axis comparisons can't distinguish disk from
  needle and over-punish thin structures.
- vs **compactness L1 alone**: controls size and sparsity, not shape — leaves needles intact (which is
  why the two terms are combined, not substituted).
- vs a **symmetric `(2 - erank)^2`**: would punish `erank > 2` (forcing sphericity) and ramp too
  gently near the needle; the clamped log-barrier is one-sided and steep exactly where it must be.

## Working code

Fills the per-step `compute_regularizer` slot; added to the photometric loss unscaled, so it carries
its own weights. Faithful to the canonical `get_effective_rank` (square scales → normalize → entropy →
exp) and the log-barrier + smallest-scale loss.

```python
import torch

SCALE_REG    = 1e-2     # L1 on exp(scales): compactness
OPACITY_REG  = 1e-2     # L1 on sigmoid(opacities): sparsity / prunability
ERANK_REG    = 1e-2     # weight on the effective-rank log-barrier
ERANK_WARMUP = 7000     # coarse-to-fine: shape prior on only after this step
ERANK_EPS    = 1e-5     # floor inside the barrier's log (finite wall at erank = 1)


def compute_regularizer(splats, step, scene_scale):
    """Compactness L1 (always) + effective-rank shape barrier (after warmup).

    splats["scales"]    [N,3] log-scales   -> exp(...) for actual scale
    splats["opacities"] [N]   logits        -> sigmoid(...) for [0,1] opacity
    Returns a scalar tensor; O(N), autograd-safe.
    """
    s = torch.exp(splats["scales"])           # [N, 3] actual scales
    a = torch.sigmoid(splats["opacities"])    # [N]    actual opacities

    loss = SCALE_REG * s.mean() + OPACITY_REG * a.mean()

    if step >= ERANK_WARMUP:
        # erank: singular values of Sigma are the squared scales -> normalize s^2.
        s_sq = s * s                                          # [N, 3]
        q = s_sq / (s_sq.sum(dim=-1, keepdim=True) + 1e-12)   # distribution over 3 axes
        H = -(q * (q + 1e-12).log()).sum(dim=-1)             # Shannon entropy
        erank = H.exp()                                       # ~1 needle, ~2 disk, ~3 sphere

        # one-sided log-barrier: finite wall near erank=1, exactly 0 at erank=2 and above
        barrier = torch.clamp(-torch.log(erank - 1.0 + ERANK_EPS), min=0.0)
        # smallest scale: break scale-invariance -> flatten, don't bloat
        s_min = s.min(dim=-1).values

        loss = loss + ERANK_REG * barrier.mean() + s_min.mean()

    return loss
```

Canonical per-Gaussian effective-rank helper:

```python
def effective_rank(scale):                    # scale: [N, 3] actual (non-log) scales
    D = scale * scale                          # singular values of the covariance = squared scales
    p = D / D.sum(dim=1, keepdim=True)         # normalize into a distribution
    entropy = -(p * torch.log(p)).sum(dim=1)   # Shannon entropy
    return torch.exp(entropy)                  # exp(H): 1 (needle) .. 2 (disk) .. 3 (sphere)
```
