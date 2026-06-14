## Research question

Modern generative models for images, video, audio, and scientific fields are built on a
*dynamical* recipe: learn a velocity field `b_t(x)` and integrate an ordinary differential
equation `x_dot = b_t(x)` from a base sample (Gaussian noise) `x_0` at `t=0` to a data sample
`x_1` at `t=1`. The samples are excellent, but generation is slow: numerically integrating the
ODE to acceptable accuracy takes tens to hundreds of evaluations of a large network, one per
solver step. For real-time control, interactive editing, and any latency-sensitive deployment
this is the bottleneck.

The object that would remove the bottleneck is the *flow map* `X_{s,t}`: the operator that
takes a point `x_s` lying on an ODE trajectory at time `s` and returns the point `x_t` the same
trajectory reaches at time `t`, for *any* pair `(s,t)`, not just an infinitesimal step. A single
evaluation `X_{0,1}(x_0)` would turn noise into data in one network call; a handful of
compositions `X_{t_0,t_1}, X_{t_1,t_2}, ...` would trade compute for quality on demand. So the
goal is a *principled and practical* way to train `X_{s,t}` directly — ideally as cheaply and
stably as ordinary flow matching, on a single model, from scratch.

The pain that makes this hard: the most reliable way to obtain such a map has been
*distillation* — first train a velocity/score model, then train a second "student" map to
reproduce many integration steps of that frozen "teacher". This works empirically but caps the
student at the teacher's quality and doubles the training pipeline. Direct-training
alternatives that reduce this dependence have been numerically delicate: they need spatial Jacobians
of a large network inside the loss, or they bootstrap large jumps from small ones in ways that
destabilize optimization, and they have demanded extensive per-architecture engineering to
train at all. A solution would need: a single network, no pretrained teacher, a loss whose
minimizer is provably the true flow map, and training dynamics stable enough to use standard
optimizers and large learning rates on image-scale UNets.

## Background

**Stochastic interpolants and the probability flow.** Given data `x_1 ~ rho_1` and base
`x_0 ~ rho_0` (Gaussian), the stochastic-interpolant construction (Albergo, Boffi &
Vanden-Eijnden 2023; Albergo & Vanden-Eijnden 2022) defines a process

```
I_t(x_0, x_1) = alpha_t * x_0 + beta_t * x_1,    alpha_0=1, alpha_1=0, beta_0=0, beta_1=1,
```

interpolating base to data as `t: 0 -> 1`. Its law `rho_t = Law(I_t)` is transported by a
*probability flow* ODE `x_dot_t = b_t(x_t)` whose drift is the conditional expectation of the
interpolant velocity,

```
b_t(x) = E[ I_dot_t | I_t = x ].
```

Because `b_t` is a conditional expectation, it is the minimizer of a plain square-loss
regression — this is exactly flow matching (Lipman et al. 2022) / rectified flow (Liu et al.
2022):

```
L_b(b_hat) = ∫_0^1 E_{x_0,x_1} | b_hat_t(I_t) - I_dot_t |^2 dt.
```

The linear choice `alpha_t = 1-t`, `beta_t = t` gives `I_dot_t = x_1 - x_0`. Training `b_hat`
this way is robust and standard; the cost is at *inference*, where one must integrate the ODE.

**The flow map and why a single evaluation is not enough information.** The flow map is the
unique `X_{s,t}` with `X_{s,t}(x_s) = x_t` along ODE trajectories; equivalently it is the
solution operator of the probability-flow ODE. It satisfies the standard structural facts:
the Lagrangian ODE `∂_t X_{s,t}(x) = b_t(X_{s,t}(x))`; the Eulerian PDE
`∂_s X_{s,t}(x) + ∇X_{s,t}(x) b_s(x) = 0`; and the *semigroup* (composition) identity
`X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)` for any intermediate `u`. On the diagonal it is the identity,
`X_{s,s}(x) = x`, and its infinitesimal time-derivative recovers the drift,
`lim_{s→t} ∂_t X_{s,t}(x) = b_t(x)`. These are the three (equivalent) ways a map can be
*certified* to be the flow map; each is a candidate residual to drive learning.

**Loss-scale imbalance across the time domain (a diagnostic fact).** When a single network is
trained over a whole family of regression sub-problems indexed by a continuous variable
(here the times `(s,t)`), the per-index loss values and their gradient magnitudes differ by
orders of magnitude; the large-gradient indices dominate and inject variance. EDM2 (Karras et
al. 2024) addressed the analogous problem across noise levels with an *uncertainty-style*
adaptive weight (the multi-task form of Kendall, Gal & Cipolla 2017): attach a learned scalar
`w` per index and train `e^{-w} L + w`. For positive `L`, minimizing over `w` alone gives
`w* = log L` and value `1 + log L`, so the construction estimates each index's log-loss-scale
and divides it out, equalizing contributions. This stabilizer applies broadly to time-indexed
regression problems.

**Spatial-Jacobian and bootstrapping instabilities (diagnostic, from prior direct-training
attempts).** Two failure modes are documented for teacher-free map training. (1) Any residual
that contains a spatial derivative `∇X` (the Eulerian route, and the continuous-time limit of
consistency training) requires backpropagating through the network's spatial Jacobian, which is
empirically unstable for large image UNets and has needed dedicated engineering to tame.
(2) Schemes that build large jumps by composing small ones suffer compounding error and
distribution shift as the student feeds its own outputs back as inputs.

## Baselines

**Flow matching / rectified flow / stochastic interpolants (Lipman et al. 2022; Liu et al.
2022; Albergo et al. 2023).** Learn only the instantaneous drift `b_t` by the square loss
`L_b` above; sample by integrating the ODE. Core idea is a clean conditional-expectation
regression with a unique convex minimizer. **Gap:** it learns only the infinitesimal generator,
so producing a sample still requires solving the ODE — many sequential network evaluations at
inference. It says nothing about how to take a *finite* jump in one call.

**Progressive distillation (Salimans & Ho 2022).** Start from a many-step teacher sampler and
train a student to reproduce *two* teacher steps in *one*, then repeat, halving the step count
each round. Core idea: a one-step student can be supervised to match a composition of two
teacher steps. **Gap:** it is inherently two-phase (needs the pretrained teacher), the student's
ceiling is the teacher's quality, and the repeated halving is a multi-stage schedule rather than
a single direct objective.

**Consistency models and consistency trajectory models (Song et al. 2023; Song & Dhariwal
2023; Kim et al. 2024).** Enforce *self-consistency*: points on the same trajectory must map to
the same target. The single-time version learns a map to the endpoint, `X_{s,1}`; the
trajectory version learns the two-time map `X_{s,t}` by matching the map at a point to the map
at a slightly shifted point on the teacher's trajectory. Core idea: a residual between a step
and a marginally-different step pins the map down. **Gap:** in its continuous-time limit this
residual *is* the spatial-Jacobian (Eulerian) condition, so it inherits the
backprop-through-`∇X` instability that has required substantial engineering to stabilize at
image scale; consistency *distillation* additionally needs a pretrained teacher.

**Shortcut models (Frans et al. 2024).** Train a single network conditioned on a step size to
satisfy a *discretized* composition condition (one big step = two half steps). Core idea: bring
the composition identity into a single model with a step-size input. **Gap:** it relies on a
discretized self-consistency at a chosen step granularity rather than a continuous, provably
exact characterization of the map, and is presented largely as an engineering recipe.

**Teacher-based flow map matching / progressive flow map matching (Boffi et al. 2024).** Given a
pretrained velocity teacher, minimize the square residual of the Lagrangian, Eulerian, or
composition condition to distill a flow map; the progressive variant iteratively extends the
valid jump range. Core idea: each structural identity of the map is a distillation residual.
**Gap:** still teacher-dependent (two-phase), and the spatial-derivative variants carry the same
`∇X` instability; the composition variant carries compounding error from bootstrapping.

## Evaluation settings

The natural yardsticks already in place for image generative models. **Datasets:** the
synthetic 2-D checkerboard (for exact, low-dimensional diagnostics), and unconditional image
benchmarks CIFAR-10 (`3x32x32`), CelebA-64, and AFHQ-64. **Metric:** Frechet Inception Distance
(FID) between generated and reference image statistics, computed over a large sample and reported
as a function of the number of function evaluations (NFE) in `{1,2,4,8,16}` so one can read off
one-step vs few-step quality. Reference FID statistics such as `cifar_stats.npz` are precomputed
from the dataset. **Architecture / protocol:** an EDM2-style UNet (Karras et al. 2024) -- for
CIFAR-10, config-G with 128 base channels, channel multipliers `[2,2,2]`, four residual blocks per
resolution, attention at `16x16`, and dropout following EDM recommendations; a linear interpolant
with a Gaussian base whose variance is matched to the data; RAdam with gradient clipping and a
square-root learning-rate decay; and exponential moving averages of parameters for sampling. The
CIFAR-10 configuration uses batch size 512, 204.8M training samples (400,000 optimization steps),
online FID over 10,000 generated samples, and FID over 50,000 generated samples for saved
checkpoints.

## Code framework

The training substrate already exists: a stochastic-interpolant data pipeline, an EDM2 UNet, an
optimizer, an EMA bookkeeper, an outer training loop, and a model wrapper that parameterizes the
two-time map through a single network output. Concretely, the wrapper maps `(s, t, x)` to a
network output `phi_{s,t}(x)` and forms the map by the identity-preserving parameterization

```
X_{s,t}(x) = x + (t - s) * phi_{s,t}(x),
```

so that `X_{s,s}(x) = x` automatically, and the on-diagonal output `phi_{t,t}(x)` plays the role
of the instantaneous drift `b_t(x)`. What is *not* settled is the off-diagonal training signal:
how to supervise `phi_{s,t}` for `s ≠ t` so that the trained `X_{s,t}` is the true flow map. The
diagonal piece is just flow matching. The single empty slot below is that off-diagonal loss
term.

```python
import jax.numpy as jnp


def diagonal_term(params, x0, x1, label, t, rng, *, interp, X):
    """On-diagonal (s = t) flow-matching loss for the implicit drift phi_{t,t} = b_t.

    Standard square-loss regression of the conditional expectation b_t(x) = E[I_dot_t | I_t = x],
    with the EDM2-style per-(s,t) adaptive weight e^{-w} L + w to equalize loss scale."""
    It = interp.calc_It(t, x0, x1)            # I_t = alpha_t x0 + beta_t x1
    It_dot = interp.calc_It_dot(t, x0, x1)    # I_dot_t = alpha_dot_t x0 + beta_dot_t x1
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)  # phi_{t,t}(I_t)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def offdiagonal_term(params, aux_params, x0, x1, label, s, t, aux, rng, *, interp, X):
    """Off-diagonal (s != t) training signal for phi_{s,t}.

    The whole question is what target to regress phi_{s,t} against so that the trained
    X_{s,t} = x + (t-s) phi_{s,t} is the flow map. Same e^{-w} L + w weighting wraps it."""
    Is = interp.calc_It(s, x0, x1)
    # TODO: fill in one off-diagonal training signal.
    #       Given the model X and the sampled points Is, form a weighted scalar loss.
    pass


def combine_terms(params, aux_params, batch, *, interp, X):
    """Same split as the training loss wrapper: diagonal rows first, off-diagonal rows second."""
    diag = diagonal_term(params, batch.x0[:batch.diag_bs], batch.x1[:batch.diag_bs],
                         batch.label_d, batch.t[:batch.diag_bs], batch.rng[:batch.diag_bs],
                         interp=interp, X=X)
    off = offdiagonal_term(params, aux_params, batch.x0[batch.diag_bs:],
                           batch.x1[batch.diag_bs:], batch.label_o,
                           batch.s[batch.diag_bs:], batch.t[batch.diag_bs:],
                           batch.aux[batch.diag_bs:], batch.rng[batch.diag_bs:],
                           interp=interp, X=X)
    return (diag * batch.diag_bs + off * batch.offdiag_bs) / batch.total_bs
```

The diagonal stub is fully determined (flow matching with the adaptive weight); the
off-diagonal `offdiagonal_term` is the one open slot, and the rest of the harness — interpolant,
network, optimizer, EMA, weighting — is fixed scaffolding around it.
