## Research question

TD-MPC2 learns a decoder-free latent world model: an encoder maps observations to a latent state, a latent dynamics network predicts the next latent from the current latent and an action, and reward and value heads read from the latent. Nothing reconstructs the observation, so the latent is defined only by self-predictive consistency, reward regression, and a bootstrapped value loss. The geometry of that latent — how its magnitude is bounded and how its coordinates are shaped — is the design question here. The only editable object is the final activation applied at the output of both the encoder and the dynamics network; everything else (the MLP stacks, the planner, the losses, the optimizer) is fixed. The question is which final normalization makes world-model learning most stable and sample-efficient across DMControl tasks.

## Prior art / Background / Baselines

A set of representation-normalization ideas can in principle serve as the final activation.

- **No final normalization (identity).** Intermediate LayerNorms control pre-activations inside the MLP, but the final linear map emits the latent with unconstrained magnitude.
- **L2 normalization (hypersphere).** Divide the latent by its L2 norm, optionally times a learnable scalar, so every latent has fixed magnitude.
- **RMSNorm.** Drop mean-subtraction from LayerNorm and keep only the scale: divide by the root-mean-square of the entries and apply a learnable gain.
- **Discrete latent codes (VQ-VAE).** Split the latent into groups and replace each group by a one-hot from a learned codebook; the one-hot argmax is non-differentiable, so straight-through estimation and a commitment loss are required.

## Fixed substrate / Code framework

The TD-MPC2 world model (~1M parameters) is frozen: the encoder MLP mapping observations to a 128-dim latent, the latent dynamics MLP predicting the next latent from latent+action, the reward and value heads, the self-predictive consistency loss, the bootstrapped value loss, the MPPI/CEM planner, and the optimizer. The latent dimension is **128** with `cfg.simnorm_dim = 8`, so the latent splits into **16 groups of 8**. The custom normalization is inserted as the *final activation* of both the encoder MLP stack and the dynamics MLP stack — the same module instance in both places — so the consistency loss compares a prediction and a target that live in the same normalized space.

## Editable interface

Only one region is editable: the `CustomSimNorm` class in `custom_simnorm.py` (lines 16–43). The contract is fixed: `__init__(self, cfg)` may read `cfg.simnorm_dim` (the group size, 8) and register any parameters; `forward(self, x)` takes a tensor of shape `(*batch_dims, latent_dim)` with `latent_dim` divisible by `simnorm_dim` and **must return the same shape**. The loop imports `torch`, `torch.nn as nn`, and `torch.nn.functional as F`. Every fill uses exactly this class — same signature, shape-preserving forward — and nothing else changes.

The starting point is the scaffold default shown below.

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

The model is trained for **200K steps** and evaluated by **episode reward (higher is better)** on DMControl tasks. The leaderboard reports three environments — **walker-walk**, **cheetah-run**, and **cartpole-swingup** — each over three seeds {42, 123, 456}. The metric is reported per seed and as the mean.
