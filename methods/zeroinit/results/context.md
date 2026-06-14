## Research question

Flow-matching image and video generators sample by solving an ordinary differential equation
from a source distribution to a data distribution. A frozen model supplies a time-dependent
velocity field `v_t^theta(x | y)`; generation starts at `t = 0` from noise and ends at `t = 1`
near a clean sample. The condition `y` may be a class label, a text prompt, or another control
signal.

Classifier-free guidance is the standard way to make the condition matter at inference time.
The same network is queried once with the condition and once with the condition dropped, giving
`v_cond` and `v_uncond`, and the sampler integrates a linear mix of those two predictions. The
practical problem is that this inference-time mix is trusted even when the learned velocity field
is not yet a good approximation to the true transport velocity. A useful guidance rule should
therefore keep the drop-in nature of CFG, avoid retraining and extra network calls, and behave
sensibly when the two available velocity predictions are imperfect.

## Background

Score-based diffusion models learn the reverse of a noising process and can be sampled through
a reverse SDE or a probability-flow ODE. Flow matching instead trains a velocity field directly.
The flow view is attractive for large text-to-image and text-to-video systems because it presents
sampling as transport along an ODE and can be paired with standard numerical solvers.

The basic conditional-flow-matching construction uses a linear path. Draw
`x_0 ~ p(x | y)` from the source distribution and `x_1 ~ q(x | y)` from the target distribution,
then set

```text
x_t = (1 - t) x_0 + t x_1.
```

The training loss regresses the velocity network to the displacement:

```text
L_CFM(theta) = E_{t,x_0,x_1} ||v_t^theta(x_t | y) - (x_1 - x_0)||_2^2.
```

The population minimizer is the conditional mean
`v_t^*(x) = E[x_1 - x_0 | x_t = x]`. This optimal velocity is usually unknown on real image
data, but it is available in closed form for a Gaussian source and target. For
`p = N(0, I)`, `q = N(mu, I)`, and the same linear path,

```text
v_t^*(x) = ((2t - 1) / ((1 - t)^2 + t^2)) (x - t mu) + mu.
```

The formula follows from the Gaussian conditional-mean identity with
`Cov(x_t, x_1 - x_0) = (2t - 1) I` and
`Var(x_t) = ((1 - t)^2 + t^2) I`. It gives a diagnostic setting where a learned velocity can be
compared directly to the optimal velocity throughout training. In underfitted flow models, this
kind of comparison can expose time-dependent velocity error near the high-noise/source end of
the trajectory, precisely where guidance has little semantic information to lean on.

## Baselines

**Classifier-free guidance, CFG.** Train one model to produce both conditional and unconditional
predictions by randomly dropping the condition during training. At inference, combine the two
available velocities as

```text
v_guided = (1 - w) v_uncond + w v_cond,
```

with guidance scale `w`. At `w = 1`, the sampler uses the conditional prediction alone; larger
values amplify the conditional direction. The gap is that `w` is a global hand-tuned scalar and
the fixed coefficients do not account for the current sample, timestep, or agreement between
the two predictions.

**Classifier guidance.** Earlier guided diffusion sampling adds a separately scaled classifier
gradient to an unconditional score. It can improve conditional control, but it requires an
extra classifier trained on noisy inputs. Its useful lesson is architectural rather than
operational: the unconditional model output and the condition-improving direction need not be
locked to a single scalar.

**Adaptive and dynamically scaled guidance.** Methods such as adaptive guidance, characteristic
guidance, rectified CFG variants, and CFG++ modify the way guidance is injected to reduce
oversaturation, artifacts, or off-manifold reverse steps at high guidance scale. These methods
focus on known high-scale pathologies of CFG and on the geometry of the reverse step, while the
fixed CFG velocity mix can still be brittle when the learned flow field itself is inaccurate.

**Guidance interval heuristics.** Some samplers restrict guidance to a selected interval of the
sampling schedule because guidance at extreme noise levels is often unreliable. This gives a
coarse binary schedule, but the interval is heuristic and can discard useful guidance in some
parts of the trajectory while still allowing harmful guidance in others.

**Guidance as predictor-corrector.** Analyses of CFG show that the guided estimate should not be
understood as the exact score or velocity of one fixed tilted distribution. That explains why
the CFG direction can be suboptimal, but it leaves open a concrete drop-in replacement for the
velocity mix in flow-matching samplers.

## Evaluation settings

The natural diagnostic and evaluation setup keeps the generator frozen and changes only the
per-step guidance computation:

- A Gaussian or Gaussian-mixture flow-matching toy problem where the optimal velocity can be
  evaluated in closed form, allowing direct learned-versus-optimal velocity comparisons across
  timesteps and training checkpoints.
- Class-conditional ImageNet-256 generation with a DiT/SiT-style flow model, sampled with a
  fixed ODE solver, fixed step budget, and standard generative metrics such as Inception Score,
  FID, sFID, precision, and recall.
- Text-to-image generation with large frozen flow models such as Stable Diffusion 3/3.5,
  Lumina-Next, and Flux, using fixed prompt sets, seeds, solvers, and guidance scales.
- Text-to-video generation with large frozen flow models such as Wan-style systems, using fixed
  prompts, seeds, frame settings, and video-quality metrics.

## Code framework

A guidance rule plugs into the ordinary flow-matching sampling loop. The sampler already owns a
frozen velocity network, a time schedule, and an ODE stepper. Each iteration can obtain the
unconditional and conditional predictions without increasing the model-evaluation budget.

```python
import torch


@torch.no_grad()
def sample(pipeline, cond, uncond, guidance_scale, num_steps):
    x = pipeline.initialize_sample()

    for step, t in enumerate(pipeline.schedule(num_steps)):
        v_uncond, v_cond = pipeline.predict_velocity(x, t, uncond, cond)

        # TODO: fill in the per-step velocity update.
        v = ...

        x = pipeline.ode_step(x, t, v)

    return x
```
