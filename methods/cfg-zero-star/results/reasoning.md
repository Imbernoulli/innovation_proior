Let me start from the failure I can actually feel in a guided sampler. The model gives me two
predictions per step — the unconditional `v_uncond` and the conditional `v_cond` — and classifier-free
guidance forms `v_guided = (1 - w) v_uncond + w v_cond` and integrates it. That is useful, but it hides
an assumption I want to make explicit: the learned velocities are being treated as if they are accurate
enough to amplify. Write `v_uncond = v_uncond^* + e_uncond` and `v_cond = v_cond^* + e_cond` for the
truth plus error. Then the guided estimate carries the ideal guided part plus `(1 - w) e_uncond + w
e_cond`. The same scale `w` that strengthens the conditional *signal* also strengthens the conditional
*error*. When the model is underfitted, a large `w` cannot tell which component is semantic control and
which is a wrong velocity. So the brittleness of CFG is not a flaw of the model alone; it is the fixed,
error-blind mix doing the amplifying.

I need a way to stop guessing about that error. On real images I do not know the true velocity, but a
Gaussian path gives me a measuring stick. Let `x_0 ~ N(0, I)`, `x_1 ~ N(mu, I)`, and the linear path
`x_t = (1 - t) x_0 + t x_1`. The optimal flow-matching velocity is the conditional mean
`v_t^*(x) = E[x_1 - x_0 | x_t = x]`, and `(x_t, x_1 - x_0)` is jointly Gaussian. Its means are
`E[x_t] = t mu` and `E[x_1 - x_0] = mu`. Its covariance pieces are `Var(x_t) = (1-t)^2 I + t^2 I =
((1-t)^2 + t^2) I` and `Cov(x_t, x_1 - x_0) = (1-t)(0 - I) + t(I - 0) = (2t - 1) I`. The Gaussian
conditional-mean formula then gives `v_t^*(x) = mu + ((2t - 1)/((1-t)^2 + t^2)) (x - t mu)`. Now I can
compare a learned, guided velocity to the exact one in a controlled setting and ask *where* the learned
field is least trustworthy. The error is not a single global number; it varies with `t`, and the source
end of the trajectory — `t` near the start — is dangerous because the sample is still essentially noise
while guidance is already allowed to push at full strength.

There are two separate problems hiding in CFG, and the diagnostic exposes both. The first is that the
mix is *too rigid*: it ties the unconditional coefficient to `(1 - w)`, even though the unconditional
prediction itself may be too large, too small, or poorly aligned with the conditional prediction at
this particular `(x, t)`. Classifier guidance already suggests the right separation — there can be a
separate scalar balancing the baseline direction against the condition-improving direction. So let me
introduce a scalar `s` on the unconditional prediction: `v_s = (1 - w) s v_uncond + w v_cond`. At
`s = 1` this is ordinary CFG. I do not want `s` to become another hand-tuned knob; I want to *choose* it
from the two vectors I already have.

If the true guided velocity were visible I would minimize `||v_s - v^*||^2` over `s`. It is not. That
is the wall, and the way through it is an inequality, not a heuristic. Let `delta = w - 1`, so the same
guided velocity is `v_s = v_cond + delta (v_cond - s v_uncond)`. Then the unavailable loss is
`||v_s - v^*||^2 = ||(v_cond - v^*) + delta (v_cond - s v_uncond)||^2`. For any positive `lambda`,
Young's inequality gives `||a + delta b||^2 <= (1 + lambda)||a||^2 + (1 + 1/lambda) delta^2 ||b||^2`,
with `a = v_cond - v^*` and `b = v_cond - s v_uncond`. The first term still contains the unknown truth
— but it does not depend on `s`. The only `s`-dependent part of this upper bound is a positive constant
times `||v_cond - s v_uncond||^2`. So the scalar choice reduces to a projection problem that uses *only
the two model predictions*: minimizing the bound on the invisible loss replaces "match the unseeable
truth" with "best least-squares match of the conditional vector by a scalar multiple of the
unconditional vector."

Differentiate `g(s) = ||v_cond - s v_uncond||^2`: `g'(s) = -2 v_uncond^T (v_cond - s v_uncond)`, and
setting it to zero gives `s^* = (v_cond^T v_uncond) / ||v_uncond||^2`. The second derivative is
`2 ||v_uncond||^2 > 0`, so this is the least-squares minimizer whenever `v_uncond` is nonzero.
Geometrically, `s^* v_uncond` is the orthogonal projection of `v_cond` onto the line spanned by
`v_uncond`, and `v_cond - s^* v_uncond` is the residual that the unconditional prediction *cannot*
explain. That residual is exactly the direction I want guidance to amplify: not the whole conditional
vector measured against a possibly mis-scaled baseline, but the conditional part left over after the
best scalar match to the unconditional vector is removed. The guided velocity can then be written as
`(1 - w) s^* v_uncond + w v_cond` or, more revealingly, `s^* v_uncond + w (v_cond - s^* v_uncond)`:
rescale the unconditional baseline, then push along the conditional residual. This is the "optimized
scale," and the star in the method's name marks it. It is per-sample and per-step, costs one dot
product and one squared norm on vectors I already hold, and adds no network evaluation. It also reduces
to ordinary CFG exactly when the two predictions are collinear (`s^* = 1`), so it can only differ where
there is a genuine off-axis component to correct.

I should check this is a real new degree of freedom and not a relabeled `w`, because if `s^*` only
rescaled what `w` already controls it would buy nothing. It does not. `w` is a single global scalar,
fixed for the entire run and applied identically to every sample and every step; `s^*` is computed per
sample and per step from the actual geometry of that step's two predictions. They act on different
objects — `w` sets *how hard* to push along the conditional residual, `s^*` sets *what the residual is*
by fixing the baseline against which it is measured. So the optimized scale is orthogonal to the
guidance scale, not a reparameterization of it.

The optimized mix fixes *how* to combine the predictions, but it does not answer what to do when the
prediction at the very beginning is so bad that even the best mix is not a good step. The closed-form
diagnostic lets me ask a sharper question at `t = 0`: is the guided first-step velocity closer to the
optimal first-step velocity than the *zero* vector is? In the underfitted regime the diagnostic can
satisfy `||v_guided(t = 0) - v_0^*||^2 >= ||0 - v_0^*||^2`. Read that as a decision rule. At the source
end, a guided move can be a *worse* estimate of the right direction than no velocity at all. Taking that
step injects the largest wrong direction exactly when the trajectory carries the least semantic
information; setting the velocity to zero leaves `x` unchanged and provably does no worse than the
guided move — and often better. So the beginning of the solver needs an inert prefix: zero the velocity
for the first `K` steps, then use the optimized guided velocity. `K` must stay small, because the
inequality is a statement about the unreliable *source end*, not the whole trajectory; once the field
becomes informative, continuing to do nothing throws away useful, budget-limited solver steps and
discards the conditional signal I need. This is the "zero-init" half. The two halves are independent
levers — optimized scale corrects the mix on every guided step; zero-init removes the few steps where
no mix is trustworthy — and together they are the full method.

Now make both halves concrete in code. The optimized scale flattens each prediction per batch element,
forms the projection coefficient with a small denominator floor (so a near-zero unconditional
prediction does not blow up `s^*`), reshapes the per-sample scalar to broadcast over the latent's
channel and spatial axes, and casts it back to the prediction dtype. The zero-init prefix is a single
branch on the step index. In the flow/velocity frame the inert step sets `v = 0`; in an eps-prediction
DDIM/CFG++ sampler the clean transplant is to skip the initial denoise-and-renoise updates entirely
with `if step < K: continue`, leaving the latent at its initialization for those steps. The rest of the
sampler is unchanged: the optimized scale lives entirely inside the construction of the guided
prediction, and the renoise is whatever the base sampler uses (for CFG++ that is the unconditional
prediction, which keeps the trajectory on the data manifold).

Let me trace the limits to confirm the construction. If `v_cond` and `v_uncond` are collinear, `s^* =
1` and the optimized mix is exactly CFG (and, on a CFG++ base, exactly the plain CFG++ mix) — so the
method never underperforms the base where the predictions already agree, and only acts where there is a
real off-axis component. If `v_uncond` were orthogonal to `v_cond`, `s^* = 0` and guidance would push
along the full conditional vector, which is the sensible thing when the unconditional baseline carries
no shared component. Both limits are reasonable, which is the reassurance that `s^*` is the *right*
scalar rather than a tuned one. And the implicit-classifier reading from CFG makes the same point from
another angle: there the guided direction is the difference `v_cond - v_uncond` with a *unit*
coefficient on the unconditional term; the optimized scale says that unit coefficient was an unexamined
default, and the difference isolates the conditional-specific component more cleanly when the baseline
is first projected onto the conditional prediction.

So the causal chain in one breath. CFG is brittle under velocity error because the global scale
amplifies prediction error as well as conditional signal, worst at the high-noise source end. The
Gaussian path gives an exact velocity `v_t^*(x) = ((2t - 1)/((1-t)^2 + t^2))(x - t mu) + mu`, which
locates where that error is worst. Minimizing a Young's-inequality upper bound on the invisible
`||v_s - v^*||^2` — the only `s`-dependent term of which is `||v_cond - s v_uncond||^2` — gives the
per-sample projection coefficient `s^* = v_cond^T v_uncond / ||v_uncond||^2`, so guidance amplifies the
conditional *residual* the unconditional prediction cannot explain. The first-step diagnostic shows the
guided move can be worse than the zero move at the source end, so the first `K` steps get zero velocity.
The whole thing stays a drop-in guidance rule: the same two predictions per step, one dot product and
one norm for the scale, an inert prefix at the unreliable start, and no retraining or extra network
evaluation. The eps-space DDIM/CFG++ transplant — optimized-scale mix in the denoise, unconditional
renoise, `if step < K: continue` prefix — is the form filled into the answer's scaffold.

## Minimal reference implementation

A concise Python stub that captures the optimized-scale guidance rule used by CFG-Zero*:

```python
import torch


def optimized_scale(v_cond, v_uncond, eps=1e-8):
    bsz = v_cond.shape[0]
    c = v_cond.view(bsz, -1)
    u = v_uncond.view(bsz, -1)
    s_star = (c * u).sum(dim=1, keepdim=True) / ((u ** 2).sum(dim=1, keepdim=True) + eps)
    return s_star.view(bsz, *([1] * (v_cond.dim() - 1))).to(v_cond.dtype)


def cfg_zero_star_step(v_uncond, v_cond, guidance_scale=3.0):
    alpha = optimized_scale(v_cond, v_uncond)
    return v_uncond * alpha + guidance_scale * (v_cond - v_uncond * alpha)
```
