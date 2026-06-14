## Research question

Generative modeling transports a simple prior (Gaussian noise) into a data distribution. The
dominant continuous-time recipe trains a neural network to predict a velocity field and then
generates by integrating an ordinary differential equation `dz_t/dt = v(z_t, t)` from noise at
`t = 1` to data at `t = 0`. The sampling rule over an interval is the integral
`z_r = z_t - ∫_r^t v(z_τ, τ) dτ`, approximated numerically — e.g. an Euler step
`z_{t+Δt} = z_t + Δt · v(z_t, t)`. The catch is the number of steps. Even when the per-sample
("conditional") paths are designed to be perfectly straight, the field the network actually has
to learn is the *marginal* velocity, an expectation over all sample pairs that pass through a
given point, and that marginal field is curved. Integrating a curved trajectory accurately needs
many steps; coarse discretization — and in the extreme, a single step — gives wrong samples. So
each generated image costs tens to hundreds of network evaluations (NFE).

The precise goal is a generator that produces high-quality samples in very few steps, ideally
**one** function evaluation, while keeping the conceptual cleanliness and stable training of flow
matching. A good solution would: (1) keep sampling-time computation fixed and cheap instead of
relying on accurate long ODE solves; (2) be *self-contained* — trained from scratch, with no
pre-trained teacher to distill from, no curriculum schedule, no auxiliary heuristics; (3) rest on
a genuine ground-truth target that exists independently of the network, so the optimum does not
depend on architecture and training is stable; and (4) preserve the stable supervised-regression
signal that makes ordinary flow matching attractive. Existing few-step approaches each give up one
or more of these. Closing that gap is the problem.

## Background

By this time, flow matching and diffusion are the engine of large-scale image, video, and audio
generation. The shared structure is a forward path that interpolates data and noise and a network
trained to reverse it.

**The flow-matching path and field.** Given data `x ~ p_data` and prior `ε ~ p_prior`, build a
path `z_t = a_t x + b_t ε` with predefined schedules `a_t, b_t` and time `t ∈ [0,1]`. The
instantaneous velocity of this path is its time derivative `v_t = a_t' x + b_t' ε`. A common,
"rectified" choice is `a_t = 1 - t`, `b_t = t`, which gives the strikingly simple conditional
velocity `v_t = ε - x` — a constant in time along each sample pair. Because many `(x, ε)` pairs
can produce the same point `z_t`, what the network must model is the marginal (expected) velocity
`v(z_t, t) = E[v_t | z_t]`. Generation integrates the ODE `dz_t/dt = v(z_t, t)` from `z_1 = ε`.

**Why few steps are hard.** Even with straight conditional paths, the marginal velocity field
generically curves: at a crossing point the expectation mixes incoming directions. This curvature
is a property of the true field, not of network error, so it cannot be trained away. Numerical ODE
solvers therefore accumulate discretization error under coarse step counts, and one Euler step
from pure noise lands far from the data manifold. This is the diagnostic fact that motivates the
whole few-step line: a good model can still be bottlenecked by the numerical procedure used to
move along its learned field.

**Diffusion roots.** Diffusion models add noise progressively and learn to reverse an SDE, which
admits an equivalent probability-flow ODE. Flow matching can be read as a clean ODE-first
reformulation of the same transport idea, and as a continuous-time normalizing flow. The
practical upshot is identical: iterative sampling along a curved trajectory.

**Endpoint-conditioned attempts.** Several few-step lines condition their networks on endpoint or
interval information rather than only on a single time. That extra conditioning can describe more
than an instantaneous state, but it does not by itself provide a supervised target. The open
question each leaves is how to train such a network without a teacher, a discretization curriculum,
or an extra self-consistency rule that is only a property of the network.

**Forward-mode autodiff is cheap.** Modern frameworks expose a Jacobian-vector-product primitive
(`torch.func.jvp`, `jax.jvp`) that evaluates a directional derivative of a network's output along
a chosen input tangent in roughly the cost of one forward/backward pass — without ever forming the
Jacobian. This makes total time-derivatives of network outputs along a trajectory affordable.

## Baselines

These are the prior methods a new few-step generator would be measured against and would react to.

**Flow Matching (Lipman et al. 2022; Liu et al. 2022, rectified flow; Albergo & Vanden-Eijnden
2022/2023, stochastic interpolants).** Train the instantaneous marginal velocity. The marginal
field `v(z_t, t) = E[v_t | z_t]` is intractable, so the **Conditional Flow Matching** objective
regresses on the per-sample conditional velocity instead:
`L_CFM(θ) = E_{t, x, ε} || v_θ(z_t, t) - v_t ||²`, with `v_t = a_t' x + b_t' ε` (default `ε - x`).
The load-bearing theorem (Lipman et al. 2022, Thm 2): up to a `θ`-independent constant,
`L_CFM` and the intractable marginal `L_FM` are equal, hence `∇_θ L_FM = ∇_θ L_CFM`. So
regressing on the easy per-sample target gives the same gradient as fitting the true marginal
field — you never need the intractable marginalization. Sampling integrates the learned ODE.
**Gap:** the learned field is the *instantaneous* velocity at a point; producing a sample still
requires repeated numerical solver steps along a curved trajectory, so high quality demands many
NFE and a single step is inaccurate.

**Diffusion / score-based models (Song & Ermon 2019; Ho et al. 2020, DDPM; Song et al. 2021,
score-SDE; Karras et al. 2022, EDM).** Learn the score of noised data; sample by reversing an SDE
or its probability-flow ODE. **Gap:** same iterative-integration cost; many steps per sample.

**Consistency Models (Song et al. 2023; and the consistency-training line iCT, ECT, sCT).**
Target one-step generation by learning a *consistency function* `f(x_t, t)` that maps any point on
a probability-flow-ODE trajectory directly to that trajectory's origin `x_ε`, with the boundary
condition `f(x_ε, ε) = x_ε` (which also precludes the trivial `f ≡ 0`). Training enforces
*self-consistency*: outputs at adjacent points on a discretized trajectory must agree,
`d(f_θ(x_{t_{n+1}}, t_{n+1}), f_{θ⁻}(x̂_{t_n}, t_n))`, where `f_{θ⁻}` is a target network whose
weights are an EMA of `θ` and is held under stop-gradient (online/target split borrowed from RL
and momentum-contrastive learning). Consistency distillation needs a pre-trained score model;
consistency *training* removes the teacher via an unbiased score estimator but then needs a
**discretization curriculum** `N(k)` — the number of time-grid points is grown over the course of
training — together with a tuned metric (LPIPS / Pseudo-Huber) and an EMA-rate schedule. A
continuous-time limit avoids choosing `N` but then uses forward-mode autodiff / JVPs.
**Gaps:** (1) the constraint is imposed on the *network's behaviour* — there is no closed-form
ground-truth field that the optimum is pinned to, so different networks can satisfy it differently
and training is delicate; (2) the formulation is anchored at the data side, fixing the origin
`r ≡ 0` for every `t`, so it conditions on a single time variable and cannot express a general
interval `[r, t]`; (3) it leans on the curriculum, EMA target, and metric choices that the gap in
(1) makes necessary.

**Two-time / interval methods (Boffi et al. 2024, flow map; Frans et al. 2024, Shortcut Models;
Inductive Moment Matching, 2025).** Condition on two time variables and characterize endpoint-level
behavior. Flow-map methods model displacement; Shortcut Models add a self-consistency loss, on top
of flow matching, relating flows over different discrete intervals. Inductive Moment Matching
matches the self-consistency of stochastic interpolants across time steps. **Gaps:** these still
introduce *extra* self-consistency assumptions imposed on the network, a teacher-like relation
between interval predictions, or a separate boundary condition. Their training signal remains less
direct than the supervised velocity target used by ordinary flow matching.

**Backbone — Diffusion Transformer (Peebles & Xie 2023, DiT; on ViT, Dosovitskiy et al. 2020).**
A transformer operating on image (or VAE-latent) patches, with adaptive-layer-norm-zero
(adaLN-Zero) conditioning that injects time (and class) information. This is a standard backbone
choice for large-scale image transport models; the architecture is orthogonal to the training
objective being designed.

## Evaluation settings

The natural yardsticks already in use for this regime:

- **Datasets.** Class-unconditional CIFAR-10 (32×32, pixel space); class-conditional ImageNet
  256×256 in the latent space of a pre-trained VAE tokenizer (latent 32×32×4).
- **Backbone.** A DiT/ViT (adaLN-Zero) conditioned on the time variable(s); on CIFAR-10 a
  U-Net from the score-SDE lineage is also standard. For two-time conditioning, each time variable
  is given a positional embedding, passed through a small MLP, and combined.
- **Metric and budget.** Fréchet Inception Distance (FID), lower is better, computed against the
  training set; reported as a function of the number of function evaluations (NFE), with one-step
  (1-NFE) generation the headline regime, alongside 2-step and few-step.
- **Sampling protocol.** Generate from Gaussian noise with a fixed low-step solver — e.g. a
  small number of Euler steps, down to a single step — then score FID on the generated set.
- **Optimization protocol.** Adam, constant or warmed-up learning rate, EMA of weights for
  evaluation; time variables, when used by a model, are typically sampled from simple distributions
  on `[0,1]`, including uniform and logit-normal choices.

## Code framework

The objective plugs into an existing flow-matching training harness: a DiT/U-Net backbone, an
Adam optimizer, a data pipeline that draws image batches and corrupts them along the
interpolation path, and a training loop that builds a regression target and back-propagates a
mean-squared error. What is *not* settled is the regression target for few-step generation. The
substrate below is therefore the generic flow-matching machinery only, with one empty slot where
the target is built. The forward-mode `jvp` primitive is shown as an available library tool, not
yet wired to anything.

```python
import torch
import torch.nn.functional as F
from torch.func import jvp   # forward-mode autodiff primitive that already exists


def make_path(x, eps, t):
    """Conditional flow-matching path and its instantaneous velocity.
    Default rectified schedule a_t = 1 - t, b_t = t."""
    z_t = (1 - t) * x + t * eps          # z_t = a_t x + b_t eps
    v_t = eps - x                         # conditional velocity a_t' x + b_t' eps
    return z_t, v_t


def sample_times(batch_size, device):
    """Draw whatever scalar time coordinates the target will require."""
    # TODO: the time-sampling scheme.
    pass


def build_target(z_t, v_t, times):
    """Construct the regression target the network is trained to match.

    For ordinary flow matching this would just be the instantaneous velocity v_t.
    The few-step target goes here.
    """
    # TODO: the regression target.
    pass


def train_step(net, optimizer, x, device):
    eps = torch.randn_like(x)                       # prior sample
    times = sample_times(x.shape[0], device)
    z_t, v_t = make_path(x, eps, times)             # corrupt along the path

    pred = net(z_t, times)                          # network prediction
    target = build_target(z_t, v_t, times)          # <- the slot we fill

    loss = F.mse_loss(pred, target.detach())        # squared-L2 regression
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def sample(net, shape, device):
    """Generate from noise with a fixed low-step solver. The exact update per
    step depends on what the network predicts -- left open here."""
    z = torch.randn(shape, device=device)
    # TODO: the sampling update implied by whatever the network predicts.
    pass
```

The data pipeline, the path construction, the optimizer, and the MSE regression are all in place;
the single empty slot is the regression target (and the matching sampling update) that turns this
into a few-step generator.
