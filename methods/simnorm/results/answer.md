# SimNorm (Simplicial Normalization)

## Problem it solves

In a decoder-free latent world model — an encoder `h: s → z`, latent dynamics `d: (z,a) → z'`,
and reward/value heads reading off `z`, trained by a self-predictive consistency loss
`||d(z,a) − sg(h(s'))||²` plus bootstrapped value regression — nothing pins down the *scale* of
the latent. The consistency loss permits collapse toward small feature magnitudes, while the
bootstrapped value loop can drive the learned latent scale upward. An unconstrained latent
therefore has no structural magnitude bound, and in the TD-MPC baseline this shows up as
exploding gradients on harder tasks. SimNorm bounds and shapes `z` so training stays stable
across many tasks with a *single* hyperparameter set, with no per-task latent normalization
constant.

## Key idea

Treat the latent as `L` independent groups and project each group onto a simplex with a softmax.
Partition `z ∈ R^d` into `L = d/V` groups of size `V`; apply softmax (temperature `τ`) within
each group so every group becomes a nonnegative vector summing to 1 — a point on the
`(V−1)`-simplex — then concatenate. SimNorm is the smooth, differentiable relaxation of a hard
vector-of-categoricals (VQ-style) code: for `softmax(z/τ)`, `τ → 0+` approaches one-hot codes,
`τ → ∞` gives trivial uniform codes, and finite `τ` interpolates. It is used as the **final
activation** of both the encoder and the latent dynamics network, so every latent the model
produces is simplex-valued.

## Why it works

- **Bounded.** Within a group, `||g_i||_1 = 1` and `1/√V ≤ ||g_i||_2 ≤ 1`; over all `L` groups
  `||z°||_1 = L` and `√(L/V) ≤ ||z°||_2 ≤ √L`. The upper bound depends only on `L` — no input- or
  task-dependent constant — so the latent's unbounded magnitude degree of freedom is removed
  structurally.
- **Group-sparse, overcomplete, expressive.** `L` independent simplices give up to `V^L`
  configurations (≈ `L·log₂V` bits) while each group stays bounded — unlike a single global
  softmax over `d`, which would force one dominant coordinate and ≈ `log₂d` bits. The within-group
  softmax is a zero-sum competition (sum fixed to 1, all ≥ 0), so the network must prioritize a
  few entries per group → an inductive bias toward sparsity, *without* an L1 penalty or hard
  `argmax`. Per-group entropy `H(g_i) ∈ [0, ln V]` measures it (0 = one-hot, `ln V` = uniform).
- **Differentiable.** Softmax flows gradients cleanly (no straight-through estimator, no
  codebook, no commitment loss), so it composes with multi-step latent rollouts.
- **Shared geometry for predictor and target.** Putting SimNorm on the final layer of *both* `h`
  and `d` makes the consistency prediction `d(z,a)` and its target `sg(h(s'))` live in the same
  bounded simplicial space; reward/value heads always read a well-scaled latent.

Defaults: group size `V = 8`, temperature `τ = 1` (plain softmax — `τ` is left untuned).

## Final form

Definition. With `L` groups of size `V` (`d = L·V`):

  z° = [g_1, …, g_L],   g_i = softmax(z_{(i)} / τ),   i.e.  g_{ij} = e^{z_{ij}/τ} / Σ_k e^{z_{ik}/τ}

Bound: `||z°||_1 = L` and `√(L/V) ≤ ||z°||_2 ≤ √L`.

Code (PyTorch; shape-preserving, drops in as the final activation `act=` of an MLP):

```python
import torch.nn as nn
import torch.nn.functional as F

class SimNorm(nn.Module):
    """Simplicial Normalization. Partition the input into groups of size `dim` (V)
    and softmax each group, projecting it onto a simplex. Shape-preserving."""
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim          # V; default 8

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim) # (..., d) -> (..., L, V)
        x = F.softmax(x, dim=-1)            # softmax within each group -> point on a simplex
        return x.view(*shp)                 # back to (..., d)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"
```

Canonical TD-MPC2 usage — final activation of every encoder path and the latent dynamics, not the
reward, policy, or Q heads:

```python
# state encoder; task_dim is 0 in single-task runs
out["state"] = mlp(cfg.obs_shape["state"][0] + cfg.task_dim,
                   max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                   cfg.latent_dim, act=SimNorm(cfg))
# RGB encoder path also ends with SimNorm
out["rgb"] = conv(cfg.obs_shape["rgb"], cfg.num_channels, act=SimNorm(cfg))
# dynamics prediction lives in the same bounded space as sg(h(s'))
self._dynamics = mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                     2 * [cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg))
```

The official code uses plain `F.softmax(x, dim=-1)`, i.e. the table's default `τ = 1`. A
temperature-aware variant would divide the logits before the softmax
(`x = F.softmax(x / self.tau, dim=-1)`).
