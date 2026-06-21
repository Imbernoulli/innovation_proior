# Context: few-step generative modeling with flows (circa 2024-2025)

## Research question

Generative models built on flows and diffusions produce state-of-the-art samples in vision,
audio, protein structure, and weather, but they generate by *solving a differential equation*:
starting from noise `x_0 ~ rho_0`, one integrates an ordinary differential equation
`xdot_t = b_t(x_t)` until `t = 1` to land on a data sample `x_1 ~ rho_1`. Numerically solving
that ODE accurately takes tens to hundreds of evaluations of the learned network — one per
integration substep — which is the dominant inference cost. The goal is to generate in
**one or a few** network evaluations instead of hundreds, i.e. to learn a finite-time transport
operator rather than only an instantaneous velocity that must be integrated.

Two broad routes to few-step models exist: **distillation**, which trains the few-step model to
imitate a separately pre-trained many-step model, and **direct training** of a few-step model
from scratch without a teacher. How to learn such a finite-time transport operator effectively is
the central question.

## Background

**Stochastic interpolants and the probability flow.** The modern framework for flow-based
generation (Albergo & Vanden-Eijnden 2023; Albergo, Boffi & Vanden-Eijnden 2023; Lipman et al.
2022; Liu et al. 2022) builds a continuous path between a simple base density `rho_0` (e.g. a
Gaussian) and the data density `rho_1` with a *stochastic interpolant*

```
I_t(x_0, x_1) = alpha_t * x_0 + beta_t * x_1,
```

where `alpha, beta : [0,1] -> [0,1]` are continuously differentiable with `alpha_0 = 1,
alpha_1 = 0, beta_0 = 0, beta_1 = 1`, and `(x_0, x_1)` is drawn from any coupling with the right
marginals. The law `rho_t = Law(I_t)` interpolates the two densities. A standard choice is the
linear interpolant `alpha_t = 1 - t`, `beta_t = t`, which recovers flow matching and rectified
flow; variance-preserving and variance-exploding diffusions arise from other coefficient pairs.
This path is transported by the probability-flow ODE `xdot_t = b_t(x_t)`, `x_0 ~ rho_0`, whose
solution has the same time marginals as the interpolant, `x_t ~ rho_t`. The drift is the
conditional expectation of the interpolant's time derivative,

```
b_t(x) = E[ Idot_t | I_t = x ],   Idot_t = alpha_dot_t * x_0 + beta_dot_t * x_1,
```

— it averages the velocities of all interpolant paths passing through `x` at time `t`. Because a
conditional expectation is the minimizer of a square loss, `b` is learned by a simple regression
(flow matching):

```
L_b(bhat) = integral_0^1 E_{x_0,x_1} | bhat_t(I_t) - Idot_t |^2 dt.
```

Once `bhat` is learned, sampling integrates `xdot_t = bhat_t(x_t)` from `x_0 ~ rho_0` to `t = 1`.

**The integrated object.** Rather than the instantaneous velocity, one can consider the map that
advances a point *along the same ODE trajectory* from one time to another:

```
X_{s,t}(x_s) = x_t   for every (s, t) and every ODE trajectory (x_t).
```

A single application `X_{0,1}(x_0)` with `x_0 ~ rho_0` is a data sample in one shot; composing
`X` over a grid `0 = t_0 < ... < t_k = 1` recovers a few-step sampler that trades compute for
quality. This two-time map satisfies, by its own definition, several structural relations that
fall out of differentiating or composing the jump condition `X_{s,t}(x_s) = x_t`: it reduces to
the identity on the diagonal, `X_{s,s}(x) = x`; it composes, `X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)`
(any two jumps equal one jump); and it is tied to the velocity that generated it. These structural
facts constrain any objective for the two-time map.

**Empirical observations on training the integrated map.** Prior attempts to learn the jump map
directly report several empirical patterns. (i) Objectives that require differentiating the
network's output with respect to its spatial input — i.e. that contain a spatial Jacobian of the
map evaluated through a Jacobian-vector product — are observed to be unstable for large image
networks, often diverging, and need careful schemes and heavy hyperparameter tuning (Lu et al. 2025;
the DreamFusion line notes similar spatial-Jacobian pathologies). (ii) Objectives that build a
large jump by composing two smaller learned jumps and regressing the big one onto the composition
are stable to optimize but accumulate error as the inner jump's mistakes feed the outer jump. (iii)
Loss values at different time pairs `(s, t)` are observed to have gradient norms that differ by
orders of magnitude.

**Adaptive loss weighting.** A relevant tool already on the table is the uncertainty-style
adaptive loss weight from EDM2 (Karras et al. 2024), rooted in multi-task uncertainty weighting
(Kendall & Gal; Sener & Koltun). For a per-condition loss `L`, one trains a scalar `w` and uses
`e^{-w} * L + w`; minimizing over `w` (with `L` held) gives `w = log L`, so the gradient that
reaches the model is `e^{-w} ∇L = ∇L / L` — each condition contributes a *scale-normalized*
gradient. EDM2 applies this across noise levels in single-time diffusion training.

**Architecture.** The standard high-performance image backbone is the EDM2 UNet (Karras et al.
2024): magnitude-preserving convolutions, Fourier/positional time embeddings, FiLM conditioning,
self-attention at coarse resolutions, with input/output preconditioning `c_in = 1/sigma_data`,
`c_out = sigma_data` so the network sees and emits unit-scale signals.

## Baselines

**Flow matching / stochastic interpolants / rectified flow (Albergo & Vanden-Eijnden 2023;
Lipman et al. 2022; Liu et al. 2022).** Learn the velocity `b_t` by the regression `L_b` above,
then integrate the ODE to sample. Core idea and math are exactly the interpolant construction in
Background.

**Diffusion / score-based models (Ho et al. 2020; Song et al. 2020).** Learn a score / noise
predictor and sample via the reverse SDE or the probability-flow ODE.

**Consistency models (Song et al. 2023; Song & Dhariwal 2023).** Learn the single-time jump to
data, `X_{s,1}` in the two-time notation, trained so that points on a common ODE trajectory map
to the same endpoint ("consistency"). Consistency *distillation* uses a pre-trained teacher;
consistency *training* avoids it. Few-step, sometimes one-step, generation. The training signal is
built from the relation describing how the map changes as the start time moves, which brings in the
spatial Jacobian of the map. Consistency *distillation* requires a separately pre-trained teacher.

**Consistency trajectory models (Kim et al. 2024).** Extend consistency models to the full
two-time map `X_{s,t}`, enabling multistep sampling. Uses the same underlying structural relation.

**Shortcut models (Frans et al. 2024).** Train a two-time map and enforce that one big jump
equals two consecutive learned half-jumps (a discretized composition relation), bootstrapping
larger steps from smaller ones within a single model.

**Progressive distillation (Salimans & Ho 2022).** Repeatedly train a student to replicate two
steps of a teacher in one, halving the step count each round, starting from a many-step diffusion
teacher.

**Mean flow / Align your flow (2025).** Recent direct-training and distillation schemes for the
two-time map, using spatial-derivative characterizations of the map.

## Evaluation settings

- **Datasets.** CIFAR-10 (32x32, auto-downloaded via TensorFlow Datasets); CelebA-64 and
  AFHQ-64 (64x64); a 2D synthetic checkerboard for low-dimensional diagnostics. The base
  distribution is Gaussian with variance set adaptively to match the training data.
- **Metric.** Fréchet Inception Distance (FID): Inception-v3 features of generated vs. reference
  images compared by their Gaussian (mean, covariance) statistics; lower is better. Computed
  on-the-fly during training (e.g. 10k samples) and post-hoc on 50k samples for the final number.
- **Number of function evaluations (NFE).** FID is reported as a function of the number of jumps
  used at inference, NFE in {1, 2, 4, 8, 16}, by composing the map over a uniform time grid — the
  whole point being few-step quality.
- **Protocol.** Linear interpolant `alpha_t = 1-t`, `beta_t = t`; EDM2 Config-G UNet (e.g. 128
  base channels, attention at 16x16, dropout 0.13 for CIFAR-10); RAdam; learning-rate warmup then
  square-root decay; gradient clipping at 1.0; EMA of parameters (decay 0.999 / 0.9999) for
  evaluation; from random initialization, no pre-training. Each candidate method must specify how
  it samples time pairs `(s, t)` during training.

## Code framework

The training substrate already exists: a JAX/Flax stochastic-interpolant pipeline driving an EDM2
module that can condition on time arguments and a point, plus a one-time velocity-regression term,
the adaptive weight `w_{s,t}`, and the batch-splitting harness. What is *not* fixed is the
finite-time objective for pairs `(s, t)` with `s < t`, where there is no direct regression target.
That objective is the single empty slot.

```python
import jax
import jax.numpy as jnp

# --- existing stochastic interpolant (linear: alpha=1-t, beta=t) ---
class Interpolant:
    def calc_It(self, t, x0, x1):       # I_t = alpha_t x0 + beta_t x1
        return self.alpha(t) * x0 + self.beta(t) * x1
    def calc_It_dot(self, t, x0, x1):   # Idot_t = alpha_dot_t x0 + beta_dot_t x1
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1


# --- existing candidate network wrapper (EDM2 UNet inside) ---
# The module consumes a point and time information and emits a same-shaped output. The
# finite-time training objective is the unresolved slot.
class CandidateMap:
    def apply(self, params, *args, method=None, **kw):
        # raw network output and optional learned scalar weight w_{s,t}
        ...


def velocity_term(params, x0, x1, label, t, rng, *, interp, X):
    """Velocity regression usable where one time argument suffices."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    bt = X.apply(params, t, It, label, train=True, rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt      # e^{-w} L + w


def offdiagonal_term(params, teacher_params, x0, x1, label, s, t, rng,
                     *, interp, X, stopgrad_type):
    """Objective for the time pairs s < t, where no direct regression target exists."""
    Is = interp.calc_It(s, x0, x1)
    # TODO: the objective we will design here, weighted as e^{-w_{s,t}} * (...) + w_{s,t}.
    raise NotImplementedError


def setup_loss(cfg, net, interp):
    """Existing harness: vmap over the batch, split a fraction eta onto the s = t case
    (flow matching) and (1 - eta) onto the off-diagonal slot, average."""
    ...
```

The velocity term is fixed; the objective for the `s < t` pairs is the slot the method fills,
after which the two pieces are summed and minimized jointly.
