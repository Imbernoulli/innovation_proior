# Context: normalizing the latent representation of a decoder-free latent world model

## Research question

We are training a model-based reinforcement-learning agent that learns an *implicit* world
model entirely in a latent space and plans in that latent space. An encoder maps an observation
`s` to a latent state `z`; a latent dynamics network predicts the next latent `z'` from `(z, a)`;
reward and value heads read off `z`. Crucially there is **no decoder** — nothing reconstructs
the observation to ground `z`. The model is trained instead by a *self-predictive* (joint-
embedding) consistency loss that pulls the predicted next latent toward the encoder's own
(stop-gradient) embedding of the next observation, plus reward and bootstrapped value
regression on top of `z`.

The latent representation's *geometry* is therefore load-bearing: every learning signal flows
through `z`, and `z` is defined only by these self-referential objectives. The concrete problem:
**how should `z` be constrained so that training is stable across many tasks with a single set
of hyperparameters?** A solution must (a) prevent the latent from drifting to pathological
scales under a self-predictive loss, (b) keep gradients well-behaved without per-task tuning of
a normalization constant, and (c) preserve enough representational capacity that planning and
value learning still work. Getting this wrong is not a minor accuracy hit — it is the
difference between a model that trains and one whose gradients diverge.

## Background

**Self-predictive latent objectives have no anchor on the scale of `z`.** When a world model is
trained by reconstruction, the decoder ties the latent to pixels and pins down its meaning. A
decoder-free model trained by a consistency loss of the form `||z'_t − sg(h(s'_t))||²` (BYOL-
style joint-embedding prediction, Grill et al. 2020) has no such anchor: the loss only asks the
predicted latent to match a *moving, self-generated* target. This is exactly the regime where
representation collapse and magnitude drift are documented failure modes — the network can make
two embeddings "agree" by shrinking the representation, while bootstrapped readouts can create
pressure in the opposite direction, all without encoding useful structure. BYOL and its
relatives guard against the collapse-to-constant mode with architectural asymmetry (a predictor
and a stop-gradient EMA target), but they do not, by themselves, bound the *magnitude* of the
embedding.

**In an RL world model the instability is amplified by bootstrapping.** The value head is
trained toward a target `q = r + γ Q̄(z', p(z'))` that is itself a function of latent states the
same network produces. So there is a closed loop: latents feed value targets, value gradients
feed back into the encoder and dynamics, which reshape the latents. If the latent magnitude is
free to grow, this loop has a positive-feedback direction.

**The observed pathology.** A prior latent-MPC world model in this family was implemented as
plain MLPs (ELU activations, no per-layer normalization) and placed **no constraint on the
latent state at all**. On harder continuous-control tasks this model is prone to **exploding
gradients** — the gradient norm climbs over training and learning diverges on some tasks (for
example a walker task that simply destabilizes). This is a measured, pre-existing phenomenon
about an existing system: an unconstrained latent under a self-predictive + bootstrapped loop
does not reliably stay in a usable range.

**What "constrain the latent" could draw on.** Several mature ideas bear on bounding or shaping
a representation:

- *Per-layer normalization* (LayerNorm, Ba et al. 2016) stabilizes the *intermediate*
  activations of an MLP by recentering/rescaling each layer's pre-activations. It controls layer
  statistics but does not impose a fixed geometry on the network's *output* latent.
- *Discrete / categorical latent codes* (VQ-VAE, van den Oord et al. 2017) replace a continuous
  latent with a set of `L` one-hot codes drawn from a learned codebook — a "vector of
  categoricals." This bounds the representation and gives it discrete structure, at the cost of
  a non-differentiable lookup.
- *Sparse, overcomplete representations* from the sparse-coding tradition: representing an input
  with more basis components than its dimensionality, most of them inactive, which has long been
  associated with stability under noise and with interpretable features.
- *Self-supervised representation shaping*: recent SSL work projects an encoder's output through
  a softmax to impart an inductive bias toward group sparsity (Lavoie et al. 2022; see
  Baselines). This was developed for downstream *classification* accuracy and interpretability,
  in a single-shot image-encoding setting — not as a stability mechanism inside a recurrent,
  bootstrapped RL world model.

## Baselines

These are the constraint/normalization strategies on the table, each with its core mechanism
and the specific gap it leaves for our setting.

- **Unconstrained continuous latent (prior latent-MPC world model; Hansen et al. 2022).** The
  encoder and dynamics output a raw real-valued vector `z ∈ R^d`; MLPs with ELU activations, no
  per-layer normalization, no constraint on `z`. *Core idea:* let the consistency, reward, and
  value losses freely shape `z`. *Gap:* with nothing bounding `z`, on harder tasks the latent
  magnitude drifts and gradient norms explode, diverging training; it also needs per-task
  hyperparameter tuning (latent dimension, learning rate, planning iterations) to stay usable.

- **LayerNorm on intermediate activations (Ba et al. 2016).** Recenter and rescale each layer's
  pre-activations by their own mean and variance: `(x − μ)/σ · γ + β`. *Core idea:* stabilize
  the statistics flowing *through* the network. *Gap:* it normalizes *intermediate* features,
  not the geometry of the final latent the dynamics and value heads consume; an MLP can still
  emit a final `z` of arbitrary magnitude and structure after its last LayerNorm + linear map.

- **L2 normalization / hypersphere projection.** Divide `z` by its Euclidean norm so it lies on
  the unit sphere. *Core idea:* fix `||z||₂ = 1`, removing the magnitude degree of freedom.
  *Gap:* it bounds magnitude but imposes no *structure* on the representation — no sparsity
  pressure, no discrete-code bias; all of the unit sphere is equally available, and a single
  global constraint couples all coordinates.

- **Squashing nonlinearities (tanh / sigmoid).** Pass each coordinate through a bounded
  nonlinearity. *Core idea:* clamp each coordinate into a fixed interval. *Gap:* it bounds
  per-coordinate range but creates no competition *between* coordinates and no sparsity bias;
  saturated units also pass near-zero gradient, and the representation stays dense.

- **Discrete categorical codes / VQ-VAE (van den Oord et al. 2017).** Quantize the latent into
  `L` one-hot codes from a learned codebook. *Core idea:* a bounded, discrete, vector-of-
  categoricals representation. *Gap:* the codebook lookup is a hard `argmax` — non-
  differentiable — so it requires a straight-through gradient estimator and an auxiliary
  commitment loss, and it pays codebook-collapse and dead-code pathologies; the hard
  discretization is a strong constraint that can choke information flow.

- **Softmax-projected simplicial embeddings for SSL (Lavoie et al. 2022).** Split an encoder's
  output into `L` groups of size `V` and softmax each group, giving `L` simplex-valued vectors
  concatenated together; temperature controls a group-sparsity inductive bias. *Core idea:* a
  *soft* overcomplete code that biases toward group sparsity without hard discretization, shown
  to improve downstream-classification generalization. *Gap:* it was designed and validated as a
  one-shot image-encoder add-on optimized for classification accuracy/interpretability; whether
  such a projection is the right way to *stabilize a recurrent, bootstrapped latent world model*
  — and how it interacts with a self-predictive consistency loss and TD targets — is not what
  that line establishes.

## Evaluation settings

The natural yardstick is online continuous control. Concretely, DeepMind Control Suite tasks
(for example `walker-walk` and `cheetah-run`) provide proprioceptive-state environments with
dense rewards, fixed episode lengths, and no termination. The baseline system is a small MLP
latent world model, and the replacement should stay in the same few-million-parameter regime
rather than depend on a large generative decoder; the metric is **episode return**
(higher is better), reported as a function of environment steps and averaged over seeds. Broader
protocols in this family span many tasks (locomotion, manipulation) under a *single* shared
hyperparameter set, and track the **gradient norm during training** as a stability diagnostic.
The latent dimension is on the order of 128–512. These settings — tasks, metrics, model scale —
predate any particular normalization choice.

## Code framework

The world model and its training loop already exist; what is missing is the normalization layer
that defines the geometry of the latent `z`. The scaffold below is the existing harness with one
empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentNorm(nn.Module):
    """Final activation that maps a raw MLP output to the latent representation z.
    Used as the last layer of both the encoder and the latent dynamics network."""
    def __init__(self, cfg):
        super().__init__()
        # TODO: parameters of the normalization we will design
        pass

    def forward(self, x):
        # input:  raw real-valued vector (..., d) from the preceding linear layer
        # output: the normalized latent z, SAME shape (..., d)
        # TODO: the normalization we will design
        pass


class NormedLinear(nn.Linear):
    """Existing primitive: Linear -> LayerNorm -> activation (+ optional dropout)."""
    def __init__(self, *args, dropout=0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))


def mlp(in_dim, hidden_dims, out_dim, act=None):
    """Existing primitive: stack NormedLinears, with `act` as the FINAL activation."""
    dims = [in_dim, *hidden_dims]
    layers = [NormedLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
    layers.append(NormedLinear(dims[-1], out_dim, act=act))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """Existing decoder-free latent world model. The encoder and dynamics both
    emit the latent z; reward/value/policy heads read z. The final activation of
    encoder and dynamics is the slot we must fill."""
    def __init__(self, cfg):
        super().__init__()
        self._encoder  = mlp(cfg.obs_dim, [cfg.enc_dim], cfg.latent_dim,
                             act=LatentNorm(cfg))                       # TODO slot
        self._dynamics = mlp(cfg.latent_dim + cfg.action_dim, 2 * [cfg.mlp_dim],
                             cfg.latent_dim, act=LatentNorm(cfg))        # TODO slot
        self._reward   = mlp(cfg.latent_dim + cfg.action_dim, 2 * [cfg.mlp_dim], cfg.num_bins)
        self._Qs       = nn.ModuleList([
            mlp(cfg.latent_dim + cfg.action_dim, 2 * [cfg.mlp_dim], cfg.num_bins)
            for _ in range(cfg.num_q)])
        # policy prior, EMA target Q, etc. omitted for brevity

    def encode(self, s):       return self._encoder(s)
    def next(self, z, a):      return self._dynamics(torch.cat([z, a], dim=-1))

# Training (existing): self-predictive consistency + reward + bootstrapped value.
#   z      = model.encode(s)
#   z_pred = model.next(z, a)
#   L_consistency = F.mse_loss(z_pred, model.encode(s_next).detach())
#   L = L_consistency + reward_loss + value_loss   # value target uses EMA Q on z_pred
```
