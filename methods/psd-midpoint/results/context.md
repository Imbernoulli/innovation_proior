# Context: few-step generative modeling via flow maps (circa 2024-2025)

## Research question

Generative models built on dynamical systems — flows and diffusions — produce
state-of-the-art samples in vision, audio, language, and the sciences, but they pay for it at
inference time. To draw one sample you integrate an ordinary differential equation
`dot x_t = b_t(x_t)` from a base distribution `rho_0` at `t=0` to the target `rho_1` at `t=1`,
and an accurate solve needs tens to hundreds of evaluations of the learned drift network. In
latency-sensitive settings — real-time control, interactive editing, large-batch serving —
that repeated network evaluation is the bottleneck, and it has driven intense interest in
generating in **one or a few** network passes instead of many.

The object that would make this possible is the **flow map** `X_{s,t}`: the operator that
takes a point `x_s` sitting on a probability-flow trajectory at time `s` directly to the same
trajectory's point `x_t` at time `t`, for an *arbitrary* gap `t-s`. If we had it, a single
application `X_{0,1}(x_0)` with `x_0 ~ rho_0` would be a full sample, no integration at all;
and chaining a handful of jumps over a coarse time grid would trade a little compute for a
little more accuracy. The precise problem is: **how do we train such a two-time map directly
and stably**, ideally as cheaply and robustly as we already train a flow-matching velocity,
and **without** first having to train a separate teacher model and distill it?

The constraints that make this hard are concrete. (1) A learning signal exists naturally only
in the infinitesimal regime `s -> t`, where the map must reduce to the instantaneous velocity
we already know how to regress; away from the diagonal `s != t` there is no externally given
target for a finite-size jump. (2) The model has to behave coherently across arbitrary start
and end times, or a sampler that spends extra evaluations on a finer grid will not reliably
improve over a one-step jump. (3) Whatever objective supplies the finite-gap signal must be
trainable at image scale on a single network with ordinary autodiff, and must not blow up: the
prevailing few-step training schemes are notoriously unstable and need heavy engineering to keep
from diverging. A solution has to give all three at once.

## Background

**Stochastic interpolants and the probability flow.** The base-to-target path is built by
the stochastic-interpolant construction (Albergo & Vanden-Eijnden 2022; Albergo, Boffi &
Vanden-Eijnden 2023), which also underlies flow matching (Lipman et al. 2022) and rectified
flow (Liu et al. 2022). One defines
`I_t(x_0, x_1) = alpha_t x_0 + beta_t x_1`, with continuously differentiable schedules
satisfying `alpha_0 = 1, alpha_1 = 0, beta_0 = 0, beta_1 = 1`, and `(x_0, x_1)` drawn from a
coupling with the right marginals. The standard linear choice is `alpha_t = 1-t`,
`beta_t = t`. The law `rho_t = Law(I_t)` interpolates `rho_0` and `rho_1`, and it is
transported by the probability flow `dot x_t = b_t(x_t)`, whose drift is the conditional
expectation of the interpolant's velocity,
`b_t(x) = E[dot I_t | I_t = x]`. This conditional expectation is exactly the minimizer of a
square-loss regression, so `b` is learnable by

```
L_b(hat_b) = ∫_0^1 E_{x_0,x_1} | hat_b_t(I_t) - dot I_t |^2 dt,
```

and sampling is then a numerical ODE solve of `dot hat_x_t = hat_b_t(hat_x_t)` — the expensive
multi-evaluation step we want to avoid.

**Two-time maps and their boundary structure.** The flow map `X : [0,1]^2 × R^d -> R^d` is the
unique map with `X_{s,t}(x_s) = x_t` along every solution of the probability flow. Two
properties are immediate and pre-given: it is the identity on the diagonal, `X_{t,t}(x) = x`,
and as the gap closes its rate of change recovers the drift — the time-derivative of the map
at coincident times is the velocity field. So whatever finite-jump model we build is pinned at
the diagonal to the very velocity that flow matching already regresses; the open part is
everything off the diagonal, where the gap `t-s` is finite and no direct regression target
exists.

**The acceleration literature this sits in.** Several lines attack few-step generation, each
with a characteristic limitation:

- *Distillation.* Train a velocity/score model first, then run a second algorithm that
  teaches a "student" map to reproduce the "teacher's" trajectory (Salimans & Ho 2022; Boffi,
  Albergo & Vanden-Eijnden 2024). Empirically strong and stable, but two-phase, and the
  student's quality is capped by the frozen teacher.
- *Consistency-style direct training.* Train a map directly to be self-consistent along flow
  trajectories (Song et al. 2023; Song & Dhariwal 2023; Kim et al. 2024). No separate teacher,
  but these objectives are observed to be unstable and to require substantial engineering to
  train at scale (Lu & Song 2024).

**Diagnostic observations about the existing schemes.** Two empirical regularities about the
*prior* few-step methods set up the problem. First, objectives whose gradient passes through the
**spatial Jacobian** of
the map (the `nabla X` term that appears in transport-residual / consistency formulations) are
reported to be unstable for large image networks and to need careful engineering to converge.
Second, recursive bootstrap schemes accumulate error: the training target for a coarse move is
model-generated, so model error and the distribution shift of feeding the model its own outputs
compound as the bootstrap chain lengthens. These are observed costs of the prior art, not yet
anything about a fix.

**Loss-scale heterogeneity across time pairs.** A pre-existing nuisance carried over from
single-time diffusion training: the per-example loss has very different magnitudes at
different noise levels / time pairs, which inflates gradient variance. EDM2 (Karras et al.
2024) addresses this in the single-time case with a learned per-time uncertainty weight — an
estimate of the loss log-variance that, at the optimum, equalizes the contribution of each
time. Any two-time objective inherits an even worse version of this, now over a pair `(s,t)`.

## Baselines

A new flow-map training scheme would be measured against, and reacts to, the following prior
art.

**Flow-matching velocity training (Lipman et al. 2022; Albergo et al. 2023; Liu et al. 2022).**
Regress the drift directly by minimizing `L_b` above. Core algorithm: sample `(x_0, x_1)` and
`t ~ U[0,1]`, form `I_t` and `dot I_t`, and fit `hat_b_t(I_t) -> dot I_t` by least squares.
Stable, simple, the de-facto way to learn the velocity. **Gap:** it learns only the
*instantaneous* dynamics; sampling still requires integrating the ODE, i.e. many network
evaluations, which is precisely the inference cost we want to eliminate.

**Progressive distillation (Salimans & Ho 2022).** Start from a trained many-step teacher; a
student learns to take *one* step that reproduces *two* of the teacher's steps, then the
student becomes the new teacher and the procedure is iterated, halving the step count each
round. Core relation: the student step is fit to a frozen target produced by two consecutive
teacher steps. **Gap:** requires a pre-trained teacher and a staged halving schedule; it is a
multi-phase pipeline, and the student is tied to a frozen teacher rather than learning the map
directly from data.

**Consistency models and consistency trajectory models (Song et al. 2023; Song & Dhariwal
2023; Kim et al. 2024).** Train a map to be invariant along a probability-flow trajectory:
points at different times on the same trajectory must map to the same endpoint
(single-time, `X_{s,1}`) or to a consistent two-time target (`X_{s,t}` for trajectory models).
In continuous time the underlying residual is a transport equation that contains the spatial
derivative of the map. **Gap:** the gradient flows through the spatial Jacobian `nabla X`,
which for high-dimensional image networks is observed to be unstable and to require significant
engineering (special parameterizations, schedules, and regularizers) to train, as
documented by efforts to simplify and stabilize them (Lu & Song 2024).

**Shortcut models (Frans et al. 2024).** Train a single network conditioned on the step size
on a fixed dyadic time grid (`t_{i+2} - t_{i+1} = 2(t_{i+1}-t_i)`). On the diagonal it is fit
like a flow-matching velocity; off the diagonal a self-generated target supplies finite-step
behavior through a fixed recursive grid rule. **Gap:** the construction is tied to a particular
grid and recursion, the bootstrap target inherits the compounding-error cost noted above, and
the scheme is presented as a specific recipe rather than as one instance of a general training
principle for finite-gap maps.

**Flow-map matching with a frozen teacher (Boffi, Albergo & Vanden-Eijnden 2024).** Given a
pre-trained velocity `hat_b`, fit a map by minimizing the squared residual of exact relations
between the map and that teacher velocity. **Gap:** every variant relies on a *frozen,
pre-trained* `hat_b` as teacher, so it is again a two-phase distillation pipeline, not a direct
data-driven training of the map.

## Evaluation settings

The natural yardsticks for few-step image generation, all pre-existing:

- **CIFAR-10** (50,000 training images, 32×32×3), the standard small-scale benchmark.
  Unconditional generation with the linear interpolant `alpha_t = 1-t, beta_t = t` and a
  Gaussian base. Network: an EDM2-style U-Net (Karras et al. 2024) adapted to take the *two*
  times `(s,t)` (in practice re-expressed as `(s, dt = t-s)`), with standard input/output
  preconditioning by `sigma_data` and a learned per-`(s,t)` log-variance weighting head.
- Additional standard benchmarks at higher resolution: **CelebA-64** and **AFHQ-64**, plus a
  synthetic 2-D **checkerboard** density used to study finite-jump training in a controlled,
  low-dimensional setting where the spatial-Jacobian instability is less severe.
- **Metric:** Fréchet Inception Distance (FID) between generated and reference images,
  evaluated at several inference budgets (1, 2, 4, 8, 16 map steps) so that the
  compute-vs-quality trade-off is visible; a held-out set of Inception statistics
  (`cifar_stats.npz`) is precomputed. Lower FID is better, under a fixed training budget.
- **Protocol:** train with a square-root learning-rate schedule, gradient clipping, an
  exponential-moving-average copy of the parameters for evaluation, a batch split between a
  diagonal (flow-matching) portion and an off-diagonal (finite-gap) portion, and FID measured
  on a large pool of generated samples.

## Code framework

The map plugs into a standard two-time flow-map training harness. The interpolant, a network
object `X` with a finite-jump forward call, a diagonal velocity call via `calc_b`, a time
derivative via `partial_t`, and a learned `(s,t)` weight via `calc_weight`, the batch sampler
that draws diagonal and off-diagonal time pairs, the EMA, and the optimizer all already exist.
The diagonal flow-matching term is settled. What is **not** settled — and is exactly the slot to
design — is the off-diagonal term that supplies a learning signal for finite-gap jumps `s != t`.

```python
import jax
import jax.numpy as jnp


# --- existing: interpolant, network, weighted-loss convention ---
# X.apply(params, s, t, x, ...)                    -> candidate finite jump
# X.apply(params, t, x, ..., method="calc_b")      -> diagonal velocity prediction
# X.apply(params, s, t,      method="calc_weight") -> w_{s,t}      (learned log-var weight)


def diagonal_term(params, x0, x1, label, t, rng, *, interp, X):
    """Settled diagonal (flow-matching) term: on s=t the map's velocity must match dot I_t."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt          # EDM2-style (s,t) weighting


def offdiagonal_term(params, teacher_params, x0, x1, label, s, t, u, h, rng,
                     *, interp, X, stopgrad_type):
    """The finite-gap (s != t) training signal we have to design.

    There is no externally given target for a jump of size (t - s). We are handed the
    endpoints, generic sampler values, an auxiliary parameter copy (teacher_params), and a
    stopgradient knob.
    """
    Is = interp.calc_It(s, x0, x1)
    # TODO: the off-diagonal objective we will design.
    pass


# --- existing: split the batch eta:(1-eta) between diagonal and off-diagonal, sum the loss ---
def loss(params, teacher_params, x0, x1, label, s, t, u, h, rng, *, interp, X, cfg):
    diag = diagonal_term(params, x0, x1, label, t, rng, interp=interp, X=X)
    off = offdiagonal_term(params, teacher_params, x0, x1, label, s, t, u, h, rng,
                           interp=interp, X=X, stopgrad_type=cfg.stopgrad_type)
    eta = cfg.diag_fraction
    return eta * diag + (1.0 - eta) * off
```

The single empty slot is `offdiagonal_term`: produce the loss that pins down finite-gap
behavior.
