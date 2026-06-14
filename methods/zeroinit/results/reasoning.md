Let me start from the failure mode I can actually feel in a flow sampler. The model gives me a
velocity field. At a sample `x` and time `t`, I can ask for the velocity with the condition and I
can ask for the velocity with the condition dropped:

  v_cond = v_t^theta(x | y),       v_uncond = v_t^theta(x | empty).

Then I integrate an ODE from noise at `t = 0` to data at `t = 1`. If I only use `v_cond`, the
condition is often too weak. So classifier-free guidance forms

  v_guided = (1 - w) v_uncond + w v_cond,      w > 1,

and pushes harder toward the condition. That is useful, but it hides an assumption: the learned
velocities are being treated as if they are accurate enough to amplify. If `v_uncond = v_uncond^*
+ e_uncond` and `v_cond = v_cond^* + e_cond`, then the guided estimate contains the ideal guided
part plus `(1 - w) e_uncond + w e_cond`. The same scale that strengthens the conditional signal
also strengthens the conditional error. When the model is underfitted, a large `w` does not know
which component is semantic control and which component is a wrong velocity.

I need a way to stop guessing about that error. On real images I do not know the true velocity,
but a Gaussian path gives me a measuring stick. Let `x_0 ~ N(0, I)`, `x_1 ~ N(mu, I)`, and
`x_t = (1 - t) x_0 + t x_1`. The optimal flow-matching velocity is
`v_t^*(x) = E[x_1 - x_0 | x_t = x]`. The pair `(x_t, x_1 - x_0)` is jointly Gaussian. Its means
are `E[x_t] = t mu` and `E[x_1 - x_0] = mu`. Its covariance terms are

  Var(x_t) = (1 - t)^2 I + t^2 I = ((1 - t)^2 + t^2) I,

and

  Cov(x_t, x_1 - x_0)
    = Cov((1 - t) x_0 + t x_1, x_1 - x_0)
    = (1 - t)(0 - I) + t(I - 0)
    = (2t - 1) I.

The conditional-mean formula gives

  v_t^*(x)
    = mu + Cov(x_1 - x_0, x_t) Var(x_t)^{-1} (x - t mu)
    = mu + ((2t - 1) / ((1 - t)^2 + t^2)) (x - t mu).

Good. Now I can compare a learned velocity to the exact one in a controlled setting and ask
where the learned field is least trustworthy. The error is not just a global training-quality
number; it changes with `t`, and the source end of the trajectory is a dangerous place because
the sample is still close to noise while guidance is already allowed to push hard.

The fixed CFG mix is also too rigid. It ties the unconditional coefficient to `(1 - w)`, even
though the unconditional prediction itself may be too large, too small, or poorly aligned with
the conditional prediction at this particular `(x, t)`. Classifier guidance already suggests a
useful separation: there can be a separate scalar balancing the baseline direction against the
condition-improving direction. I can introduce that scalar on the unconditional prediction:

  v_s = (1 - w) s v_uncond + w v_cond.

When `s = 1`, this is ordinary CFG. I do not want `s` to become another hand-tuned knob, so I
try to choose it from the two vectors I already have. If the true guided velocity were visible,
I would minimize `||v_s - v^*||^2` over `s`. It is not visible. That is the wall.

Let `delta = w - 1`, so the same guided velocity is

  v_s = v_cond + delta (v_cond - s v_uncond).

Now the unavailable loss is

  ||v_s - v^*||^2 = ||(v_cond - v^*) + delta (v_cond - s v_uncond)||^2.

For any positive constant `lambda`, Young's inequality gives

  ||a + delta b||^2 <= (1 + lambda)||a||^2 + (1 + 1/lambda) delta^2 ||b||^2,

with `a = v_cond - v^*` and `b = v_cond - s v_uncond`. The first term still contains the
unknown truth, but it does not depend on `s`. The only `s`-dependent part of this upper bound is
a positive constant times

  ||v_cond - s v_uncond||^2.

So the scalar choice reduces to a projection problem that uses only the two model predictions.
Differentiate

  g(s) = ||v_cond - s v_uncond||^2.

Then

  g'(s) = -2 v_uncond^T (v_cond - s v_uncond),

and setting this to zero gives

  s^* = (v_cond^T v_uncond) / ||v_uncond||^2.

The second derivative is `2 ||v_uncond||^2`, so this is the least-squares minimizer whenever the
unconditional vector is nonzero. Geometrically, `s^* v_uncond` is the orthogonal projection of
`v_cond` onto the line spanned by `v_uncond`, and `v_cond - s^* v_uncond` is the residual that
the unconditional prediction cannot explain. That is exactly the direction I want guidance to
amplify: not the whole conditional vector, but the conditional part left after the best scalar
match to the unconditional vector is removed.

Numerically, I add a small denominator floor, flatten all non-batch dimensions, compute one dot
product and one squared norm per sample, and broadcast the scalar back. The guided velocity can
then be written either as

  (1 - w) s^* v_uncond + w v_cond

or as

  s^* v_uncond + w (v_cond - s^* v_uncond).

The second form reads like the geometry: rescale the unconditional baseline, then push along the
conditional residual.

That fixes the mix, but it does not answer what to do when the velocity at the beginning is so
bad that even a better mix is not a good step. The closed-form diagnostic lets me ask a sharper
question at `t = 0`: is the guided first-step velocity closer to the optimal first-step velocity
than the zero vector is? In the underfitted regime the diagnostic can satisfy

  ||v_guided(t = 0) - v_0^*||_2^2 >= ||0 - v_0^*||_2^2.

Read that as a decision rule. At the source end, a guided move can be a worse velocity estimate
than no velocity at all. If I take that step, I inject the largest wrong direction exactly when
the trajectory has the least semantic information. If I set the velocity to zero, the ODE update
leaves `x` unchanged and I avoid that particular bad move.

So the beginning of the solver needs an inert prefix. Conceptually I zero the velocity for the
first `K` solver steps and use the optimized guided velocity after that. `K` has to stay small
because the inequality is a statement about the unreliable source end, not the whole trajectory;
once the velocity field becomes informative, continuing to do nothing would simply throw away
useful solver steps.

The sampler code only needs the two predictions it already computes. I flatten each prediction per
batch element, form the projection scale, reshape it for broadcasting, and use the same per-step
zero branch that a pipeline can expose with `zero_steps` as the last zeroed step index.

```python
import torch


def optimized_scale(v_cond_flat, v_uncond_flat, eps=1e-8):
    dot = torch.sum(v_cond_flat * v_uncond_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(v_uncond_flat ** 2, dim=1, keepdim=True) + eps
    return dot / squared_norm


@torch.no_grad()
def sample(pipeline, cond, uncond, w, num_steps, zero_steps=0, use_zero_init=True):
    x = pipeline.initialize_sample()
    for i, t in enumerate(pipeline.schedule(num_steps)):
        v_uncond, v_cond = pipeline.predict_velocity(x, t, uncond, cond)

        batch_size = v_cond.shape[0]
        alpha = optimized_scale(v_cond.view(batch_size, -1), v_uncond.view(batch_size, -1))
        alpha = alpha.view(batch_size, *([1] * (v_cond.dim() - 1))).to(v_cond.dtype)

        if (i <= zero_steps) and use_zero_init:
            v = v_cond * 0.0
        else:
            v = v_uncond * alpha + w * (v_cond - v_uncond * alpha)

        x = pipeline.ode_step(x, t, v)
    return x
```

The algebra checks out: `v_uncond * alpha + w(v_cond - v_uncond * alpha)` collects to
`(1 - w) alpha v_uncond + w v_cond`. At `alpha = 1` and with the zero branch disabled, this is
standard CFG. With the branch written as `i <= zero_steps`, `zero_steps = 0` zeros exactly the
first solver step.

Some samplers expose predicted noise rather than an ODE velocity. In a DDIM-style step,

  x0_hat = (x_t - sqrt(1 - a_t) eps_pred) / sqrt(a_t),
  x_next = sqrt(a_next) x0_hat + sqrt(1 - a_next) eps_renoise.

If I want the initial zero-velocity behavior in this representation, the clean transplant is to
make the initial steps contribute no update at all. In code that is just the branch

  if step < K: continue

before the denoise-and-renoise update. The rest of the sampler can remain the existing guided
DDIM or CFG++-style step:

```python
import torch


@torch.no_grad()
def sample_ddim_zeroinit(pipeline, prompt, cfg_guidance=7.5, K=2):
    uc, c = pipeline.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
    zt = pipeline.initialize_latent()

    for step, t in enumerate(pipeline.scheduler.timesteps):
        if step < K:
            continue

        at = pipeline.alpha(t)
        at_prev = pipeline.alpha(t - pipeline.skip)

        eps_uc, eps_c = pipeline.predict_noise(zt, t, uc, c)
        eps_pred = eps_uc + cfg_guidance * (eps_c - eps_uc)

        z0t = (zt - (1 - at).sqrt() * eps_pred) / at.sqrt()
        zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * eps_uc

    return pipeline.decode(zt)
```

The pieces now fit into one causal chain. CFG is brittle under velocity error because the
guidance scale amplifies prediction errors as well as conditional signal. The Gaussian path
gives an exact velocity,
`v_t^*(x) = ((2t - 1) / ((1 - t)^2 + t^2))(x - t mu) + mu`, so I can locate where that error is
worst. The mix gets a per-sample projection coefficient
`s^* = v_cond^T v_uncond / ||v_uncond||^2`, which leaves guidance to amplify the conditional
residual. The initial steps get zero velocity when the guided estimate is worse than the zero
vector under the first-step diagnostic. I still have a drop-in guidance rule: the same two network
predictions per step, one dot product and one norm for the scale, and an inert prefix at the
unreliable source end.
