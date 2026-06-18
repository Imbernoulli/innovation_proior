## Research question

TD-MPC2 learns a **decoder-free latent world model** and plans in it: an encoder maps an observation
to a latent state, a latent dynamics network predicts the next latent from the current latent and an
action, and reward and value heads read off the latent. Nothing reconstructs the observation, so the
latent is defined purely by self-predictive consistency, reward regression, and a bootstrapped value
loss. The geometry of that latent — how its magnitude is bounded and how its coordinates are shaped —
is the single thing being designed here. The one editable object is the **final activation** applied
at the output of *both* the encoder and the dynamics network; everything else about the agent (the
MLP stacks, the planner, the losses, the optimizer) is fixed. The question is which normalization of
the latent makes world-model learning most stable and most sample-efficient across DMControl tasks.

## Prior art before the first rung (latent-geometry lineage)

The first rung reacts to a line of "normalize the representation" ideas, each of which the latent
world model could in principle borrow as its final activation, each with a gap that matters *here*.

- **No final normalization (identity).** Let the last linear map emit the latent unchanged. The
  intermediate LayerNorms inside the MLP control the pre-activations flowing *through* each layer, but
  the network's *output* — the latent actually consumed by dynamics, reward, and value — is produced by
  a last linear map with arbitrary weights, so its magnitude is unconstrained. Gap: under a
  self-predictive plus bootstrapped objective the overall scale of the latent is a free degree of
  freedom the losses do not pin down, and the value-target feedback loop drives it, so gradients can
  run away on harder tasks.
- **L2 normalization (project to a hypersphere).** Divide the latent by its L2 norm, `z ← z/‖z‖₂`,
  optionally times a learnable scalar. Kills the magnitude degree of freedom dead — `‖z‖₂ = 1` by
  construction. Gap: it bounds magnitude and *nothing else*; every point on the sphere is equally
  available, one global division couples all coordinates through a single scalar, and the latent is
  bounded but dense and shapeless — no pressure toward the sparse, structured code a value/reward
  readout tends to prefer.
- **RMSNorm (Zhang & Sennrich 2019).** Drop the mean-subtraction of LayerNorm and keep only the
  scale: divide by the root-mean-square of the entries and apply a learnable gain. Re-scaling
  invariance — the part that actually controls activation/gradient spread — survives; re-centering
  invariance is discarded. Cheaper than LayerNorm and a natural "normalize the magnitude" final
  activation. Gap as a *world-model latent*: like L2 it bounds spread without inducing any sparsity or
  group structure; the gain can re-inflate scale, and on the discriminating DMControl task it shapes a
  geometry the value head learns from less efficiently.
- **Discrete latent codes (VQ-VAE, van den Oord et al. 2017).** Represent the latent as a vector of
  categoricals — split it into groups and replace each group by a one-hot from a learned codebook.
  Bounded, structured, maximally sparse within a group, and overcomplete (up to `Vᴸ` configurations).
  Gap: the one-hot is an argmax — non-differentiable — so it needs a straight-through estimator and a
  commitment loss and drags codebook collapse along; fragile machinery for a recurrent world model
  whose gradients must flow cleanly through the dynamics step after step.

## The fixed substrate

A TD-MPC2 world model (~1M parameters) is frozen and must not be touched: the encoder MLP
(`layers.py: enc()`) mapping observations to a 128-dim latent, the latent dynamics MLP
(`world_model.py: __init__`) predicting the next latent from latent+action, the reward and value
heads, the self-predictive consistency loss `‖d(z,a) − sg(h(s'))‖²`, the bootstrapped value loss
`q = r + γ Q̄(z', p(z'))`, the MPPI/CEM planner, and the optimizer. The latent dimension is **128**
with `cfg.simnorm_dim = 8`, so the latent splits into **16 groups of 8**. The custom normalization is
inserted as the *final activation* of both the encoder MLP stack and the dynamics MLP stack — the same
module instance contract in both places — so the consistency loss compares a prediction and a target
that live in the *same* normalized space.

## The editable interface

Exactly one region is editable — the `CustomSimNorm` class in `custom_simnorm.py` (lines 16–43). The
contract is fixed: `__init__(self, cfg)` may read `cfg.simnorm_dim` (the group size, 8) and register
any parameters; `forward(self, x)` takes a tensor of shape `(*batch_dims, latent_dim)` with
`latent_dim` divisible by `simnorm_dim` and **must return the same shape**. The loop imports
`torch`, `torch.nn as nn`, and `torch.nn.functional as F`. Every method on the ladder is a fill of
exactly this class — same signature, shape-preserving forward — and nothing else changes.

The starting point is the scaffold default: **SimNorm** (the simplicial normalization the task is
named for), reshaping the latent into groups of 8 and applying a softmax within each group.

```python
# EDITABLE region of custom_simnorm.py (lines 16-43) -- default fill (SimNorm)
class CustomSimNorm(nn.Module):
    """Custom normalization for latent state representations in world models.

    Interface contract (same as SimNorm):
        __init__(cfg)  -- cfg.simnorm_dim is the group size (default: 8)
        forward(x: Tensor) -> Tensor  (same shape as input)

    The input tensor has shape (*batch_dims, latent_dim) where latent_dim
    is divisible by simnorm_dim. Your normalization should constrain the
    geometry of the latent space to improve world model learning.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        # Default: SimNorm (simplicial normalization)
        # Reshape into groups of size self.dim and apply softmax
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim})"
```

## Evaluation settings

The model is trained for **200K steps** and evaluated by **episode reward (higher is better)** on
DMControl tasks. The leaderboard reports three environments — **walker-walk**, **cheetah-run**, and a
held-out **cartpole-swingup** — each over three seeds {42, 123, 456}. Two of the three (walker-walk
and cartpole-swingup) are easy enough that every reasonable normalization saturates near the top of
their reward range; **cheetah-run** is the discriminating task, the one where the latent geometry
actually separates methods. The metric is reported per seed and as the mean.
