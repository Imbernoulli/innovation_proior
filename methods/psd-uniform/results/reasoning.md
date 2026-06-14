Let me start from what actually costs me. I can train a velocity field `b_t` for a probability-flow ODE with nothing fancier than a square loss — the drift is a conditional expectation `b_t(x) = E[I_dot_t | I_t = x]` of the interpolant `I_t = alpha_t x_0 + beta_t x_1`, and conditional expectations are exactly what least squares recovers, so `L_b(b_hat) = ∫ E |b_hat_t(I_t) - I_dot_t|^2` has the right minimizer and trains like clockwork. The samples are great. The problem is entirely at inference: to turn a noise draw into a data sample I have to numerically integrate `x_dot = b_t(x)` from `t=0` to `t=1`, and a stiff-ish ODE through a big UNet wants tens to hundreds of solver steps, each one a full network evaluation. That sequential cost is the whole pain. So what I actually want isn't the instantaneous velocity at all; I want the *finite jump* — the operator `X_{s,t}` that takes a point sitting on a trajectory at time `s` and returns where that same trajectory is at time `t`. If I had `X_{0,1}` exactly, one call turns noise into data. If I had `X_{s,t}` for general `(s,t)`, I could also chain a few jumps `X_{0,t_1}, X_{t_1,t_2}, ...` and spend a little compute to clean up the model's errors. One model, one-to-a-few evaluations. That's the prize.

How would I even parameterize a thing like `X_{s,t}`? The hard constraints I know for free: when `s=t` it has to be the identity, `X_{s,s}(x) = x` — a zero-length jump goes nowhere. And it should connect to the velocity I already know how to learn, because integrating the drift over an infinitesimal step is the smallest possible jump. Let me not fight the identity constraint with a penalty; let me build it in. Write `X_{s,t}(x) = x + (t-s) v_{s,t}(x)` where `v_{s,t}` is the network output. Then `X_{s,s}(x) = x` is automatic for any `v`, the constraint is gone. This looks like a one-term Taylor expansion but it isn't an approximation — it's just a shift and a rescale of whatever `X` is, so I lose no expressivity; `v_{s,t}` absorbs all the nonlinearity. And it pays off immediately on the diagonal. Differentiate in `t` and send `s → t`: `∂_t X_{s,t}(x) = v_{s,t}(x) + (t-s) ∂_t v_{s,t}(x)`, so `lim_{s→t} ∂_t X_{s,t}(x) = v_{t,t}(x)`. But the flow map's own definition forces `lim_{s→t} ∂_t X_{s,t}(x) = b_t(x)` — taking a derivative of `X_{s,t}(x_s) = x_t` in `t` along a trajectory gives `∂_t X = x_dot_t = b_t(X_{s,t})`, and at `s=t` that's just `b_t(x)`. So my parameterization hands me `v_{t,t}(x) = b_t(x)` for free: the diagonal of `v` *is* the drift, and I can train it with the very flow-matching loss I already trust. Beautiful — half the job is done by a loss I already have. The whole remaining question is how to supervise `v_{s,t}` off the diagonal, for `s ≠ t`, so that the trained `X_{s,t}` is genuinely the flow map and not just identity-on-the-diagonal nonsense.

What certifies a map as *the* flow map away from the diagonal? I have three structural identities the true map obeys, each a derivative of the jump condition `X_{s,t}(x_s) = x_t`. One: differentiate in `t` to get the Lagrangian ODE `∂_t X_{s,t}(x) = b_t(X_{s,t}(x))` — the map flows forward at the local velocity evaluated at where it's transported you. Two: differentiate in `s` (total derivative, since `x_s` moves too) to get the Eulerian PDE `∂_s X_{s,t}(x) + ∇X_{s,t}(x) b_s(x) = 0`. Three: the semigroup property, `X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)` for any intermediate `u` — two jumps glued end to end equal one big jump, which is just the trivial fact that following a trajectory from `s` to `u` and then `u` to `t` is the same as following it from `s` to `t`. Each of these is zero exactly when the map is correct, so each squared residual is a candidate off-diagonal loss. The velocity these residuals need is `b`, and I just argued `b_t = v_{t,t}`, the diagonal of my own model. So I don't need an external teacher at all — the model teaches itself, distilling its own diagonal into its off-diagonal. Self-distillation. The diagonal gets the real external signal (`I_dot_t` from the data), and the off-diagonal bootstraps off the diagonal.

So which residual? Let me feel out the costs, because at image scale the difference between "trains" and "doesn't train" is exactly this. The Eulerian one has `∇X_{s,t}(x)` in it — a spatial Jacobian of the network. To take a gradient step I'd have to backpropagate through that Jacobian-vector product, and that's the thing that's been documented to blow up for big image UNets; the consistency-trajectory line lives on this Eulerian condition in its continuous limit and has needed a pile of engineering — Fourier-feature surgery, careful preconditioning — just to keep it stable. I'd rather not start there. The Lagrangian one needs `∂_t X` (a time-jvp, cheap) and an evaluation of the teacher velocity `b_t = v_{t,t}` at the *transported* point `X_{s,t}(I_s)` — no spatial Jacobian, much friendlier. The semigroup one needs *no derivative of the map at all*: just evaluate the map three times — the big jump `X_{s,t}`, and the two small jumps `X_{s,u}` and `X_{u,t}` — and ask them to agree. No `∇X`, no `∂_t` through the teacher, just function values. That's the cheapest and the most obviously stable thing to differentiate. And it has a pedigree: progressive distillation trained a student to do in one step what a teacher does in two, then halved repeatedly. This is the same composition idea — one big jump equals two small jumps — but turned into a single direct objective with the model as its own teacher, and continuous in the split point rather than a discrete halving schedule. Let me build the off-diagonal loss on the semigroup.

The raw objective writes itself: penalize the composition residual on `X`,

  L = ∫∫∫ E_{x_0,x_1} | X_{s,t}(I_s) - X_{u,t}( X_{s,u}(I_s) ) |^2 du ds dt,

over `0 ≤ s ≤ t ≤ 1` (the upper triangle is all I need to go noise → data) and `u ∈ [s,t]`. Is its minimizer right? The true flow map makes the integrand identically zero by the semigroup property, so it hits the global lower bound; and conversely, if `v` is continuous and the composition residual is zero for all intermediate `u`, then a Taylor expansion of the semigroup in a small step `u = t`, `t' = t+h` recovers the Lagrangian ODE `∂_t X_{s,t} = b_t(X_{s,t})` — `X_{s,t+h}(x) = X_{t,t+h}(X_{s,t}(x)) = X_{s,t}(x) + h v_{t,t}(X_{s,t}(x)) + o(h)`, divide by `h`, send `h→0`, and there's the Lagrangian equation, whose well-posed solution is the flow map. So zero composition residual plus correct diagonal forces the true flow map. (The continuity is load-bearing: without it there's a junk solution `v_{s,t} = b_t` on the diagonal and `0` off it, i.e. `X_{s,t}=x` everywhere, which satisfies the semigroup trivially; demanding continuity in `(s,t)` rules it out.) Good — the objective is exact.

Now let me actually try to optimize this and watch where it hurts. I sample `s < t`, sample an intermediate `u ∈ [s,t]`, evaluate three times, square the difference. Let me look hard at the *scale* of that residual, because I've been burned by losses whose magnitude depends on the sampled index. Substitute the parameterization into the residual. The big jump is `X_{s,t}(x) = x + (t-s) v_{s,t}(x)`. The composed jump: first `X_{s,u}(x) = x + (u-s) v_{s,u}(x)`, then

  X_{u,t}(X_{s,u}(x)) = X_{s,u}(x) + (t-u) v_{u,t}(X_{s,u}(x)) = x + (u-s) v_{s,u}(x) + (t-u) v_{u,t}(X_{s,u}(x)).

Subtract:

  X_{s,t}(x) - X_{u,t}(X_{s,u}(x)) = (t-s) v_{s,t}(x) - (u-s) v_{s,u}(x) - (t-u) v_{u,t}(X_{s,u}(x)).

Every term carries a time-gap prefactor. If I think of `u` as splitting `[s,t]` by a fraction — and I will, in a second — then `u-s` and `t-u` are both proportional to `t-s`, so the *whole* residual scales like `(t-s)`, and the squared loss scales like `(t-s)^2`. That's a real problem, not a cosmetic one. For a pair `(s,t)` close together the loss and its gradient are tiny; for a far-apart pair they're large. The optimizer then sees an effective learning rate that silently depends on how big a jump I happened to sample, which is pure injected variance — the far jumps shout, the near jumps whisper, and the network gets yanked around by the sampling of `(s,t)` rather than by the actual error in the map. I don't want to learn `X` with a `(t-s)`-dependent gain. Wall.

Could I just reweight by `1/(t-s)^2`? That divides by something that goes to zero on the diagonal — numerically nasty and it doesn't remove the structural coupling, it papers over it. Better to kill the `(t-s)` at the source. The residual I wrote is a residual on `X`, but `X` is just `x` plus `(t-s)` times `v`; the constant `x` cancels in the difference (it appears on both sides), so the residual is entirely a statement about the `v`'s. Let me ask the semigroup directly in terms of `v`. From `x + (t-s) v_{s,t}(x) = x + (u-s) v_{s,u}(x) + (t-u) v_{u,t}(X_{s,u}(x))`, the `x` drops and I can solve for `v_{s,t}`:

  v_{s,t}(x) = ((u-s)/(t-s)) v_{s,u}(x) + ((t-u)/(t-s)) v_{u,t}(X_{s,u}(x)).

The slope of the big jump is a *convex combination* of the slope of the first small jump and the slope of the second — weighted by the fraction of the interval each covers. Of course: average velocity over `[s,t]` is the length-weighted average of the average velocities over `[s,u]` and `[u,t]`. Now the prefactors are *ratios* of time gaps, which are `O(1)`, not `O(t-s)`. If I regress `v_{s,t}` onto this combination, the residual is a difference of velocities, scale-free in `t-s`. The `(t-s)^2` is gone — I've preconditioned the loss by reading the composition off the slopes instead of the points.

Let me make the weights concrete and clean. Parameterize the split as `u = gamma s + (1-gamma) t` for `gamma ∈ [0,1]`. Then `u - s = (1-gamma)(t-s)` and `t - u = gamma (t-s)`, so

  (u-s)/(t-s) = 1 - gamma,    (t-u)/(t-s) = gamma,

and the relation collapses to

  v_{s,t}(x) = (1 - gamma) v_{s,u}(x) + gamma v_{u,t}( X_{s,u}(x) ).

(Worth a sanity check that I haven't flipped the weights: `gamma = 0` puts `u = t`, the second segment `[u,t]` has zero length and should contribute nothing — and indeed its coefficient `gamma = 0` kills it, leaving `v_{s,t} = v_{s,t}`. `gamma = 1` puts `u = s`, the first segment vanishes, coefficient `1-gamma = 0` kills *it*. The weights track the segment lengths correctly.) So the off-diagonal target for the student `v_{s,t}(I_s)` is the slope-combination on the right, and I'll minimize

  L_off^{s,t} = E | v_{s,t}(I_s) - [ (1-gamma) v_{s,u}(I_s) + gamma v_{u,t}( X_{s,u}(I_s) ) ] |^2,

with `u = gamma s + (1-gamma) t`. Note the second slope is evaluated at the *transported* point `X_{s,u}(I_s)`, not at `I_s` — because the second jump starts where the first one landed. That nonlinearity (a network evaluated at another network's output) is exactly the bootstrapping; I'll have to be careful with it.

Now `gamma`. I have a free scalar that picks where to split each interval, and I get to choose its distribution. The clean minimizer argument goes through for any proposal over `gamma` with full support on `[0,1]`: zero residual for all supported splits forces the map, and the distribution averages the signal over many splits. So what do I gain by spreading it? If I always split at the midpoint, `gamma = 1/2`, I only ever teach the map "one full jump = two equal half-jumps." That's a clean, low-variance engineering variant, but it only exercises the composition at a 50/50 split; the map never sees, in training, that a `[s,t]` jump should also equal a `10/90` composition or a `90/10` one. If instead I draw `gamma ~ U([0,1])` uniformly, every batch supervises the map at a different, randomly chosen split, so over training the map is forced to be self-consistent under all ways of breaking a jump into two, at the cost of more variance per step because the target now also fluctuates with `gamma`. I'll take the uniform proposal here — draw `gamma ~ U([0,1])`, set `u = gamma s + (1-gamma) t` — and lean on the weight function below to absorb the extra variance. In code I'll carry the split fraction as a variable `h` (this is just my `gamma`), draw `h ~ U([0,1])`, and form `u = h s + (1-h) t`, so the weights become `(1-h)` and `h` exactly as derived.

There's a subtlety I have to get right or this whole thing chases its own tail: the *direction* of information flow. The loss `L_off` is nonconvex in `v` (the second term has `v` composed with `X` which is `v` again), and it has a symmetric look — it would be just as happy to drag the diagonal-derived slopes toward a wrong `v_{s,t}` as to pull `v_{s,t}` toward the correct composition. But there's a clear asymmetry in *where the real signal lives*: only the diagonal `v_{t,t}` has an external teacher, the data velocity `I_dot_t`. The off-diagonal has no ground truth of its own; it can only be *correct* by inheriting from the diagonal. So I want the gradient to push `v_{s,t}` toward the composed target, and emphatically *not* push the composed target (which is built from `v_{s,u}` and `v_{u,t}`, themselves closer-to-diagonal slopes) toward `v_{s,t}`. The clean way to enforce a one-way teacher is the stopgradient: treat the right-hand combination as a frozen target,

  L_off^{s,t} = E | v_{s,t}(I_s) - sg[ (1-gamma) v_{s,u}(I_s) + gamma v_{u,t}( X_{s,u}(I_s) ) ] |^2.

This makes the composed slope behave exactly like a pretrained distillation teacher — the off-diagonal adapts entirely to it, never the reverse — even though it's the same network. It also quietly removes the worst of the bootstrapping instability: with the target stopgradded, I never backpropagate through the inner `X_{s,u}` and its (implicit) spatial Jacobian, so the gradient is a plain residual on the single forward evaluation `v_{s,t}(I_s)`. That's the "convex"-style placement: stop the gradient on both teacher evaluations, learn only through the student `v_{s,t}`. I did briefly consider letting the teacher track a slow EMA of the weights instead of being the instantaneous network — that's the other classic way to make a "frozen" teacher. But an EMA teacher adds a decay hyperparameter and separates the target from the current diagonal signal; the instantaneous-stopgradient version (an EMA with decay zero, in effect) is simpler and keeps the teacher exactly aligned with the current model, so I'll evaluate the teacher at the current weights with the gradient stopped.

Now the scale problem across `(s,t)` that I haven't fully closed. I removed the `(t-s)^2` coupling by going to the slope form, but different regions of the `(s,t)` square still produce loss values and gradient norms that differ a lot — the regression difficulty just isn't uniform over the time domain, and the loud regions will dominate and inject variance. This is the same disease that shows up whenever one network is trained over a continuum of regression sub-problems indexed by a time-like variable, and there's a known cure I can lift: attach a learned scalar `w_{s,t}` to each index and train, instead of `L`, the quantity `e^{-w_{s,t}} L + w_{s,t}`. Let me check what `w` does by minimizing over it alone, holding a positive `L` fixed: `∂/∂w [ e^{-w} L + w ] = -e^{-w} L + 1 = 0` gives `e^{-w*} = 1/L`, i.e. `w* = log L`, and the value at that optimum is `1 + log L`. So `w` learns the log of each index's loss scale, and after dividing it out (`e^{-w} L → 1` at the optimum) every `(s,t)` contributes on the same order — `w` is an estimate of the per-`(s,t)` log-variance, and the construction self-normalizes the loss surface. That's exactly what I need to tame both the residual heterogeneity over the square and the extra variance my uniform `gamma` injects; it is the mechanism that should let me use a larger learning rate without the loud time-regions destabilizing the step. I'll wrap *both* the diagonal and off-diagonal terms in `e^{-w} (·) + w` with the same learned weight head, training `w` jointly with `v`.

Let me also decide how to spend the batch between diagonal and off-diagonal. The diagonal flow-matching term costs one network evaluation; the off-diagonal composition term costs three (the big jump, and the two small jumps). And there's a limiting fact that tells me the two terms aren't really separate jobs: as `s → t`, the composition residual degenerates into the flow-matching residual (the Taylor expansion above shows the semigroup recovers the Lagrangian condition at order `h`, which recovers `b` at the diagonal), so the off-diagonal loss *is* the smooth extension of the diagonal loss into the interior of the square. That means I only need the cheap `L_b` exactly on the diagonal `s=t`, and the off-diagonal term everywhere else. I'll sample a fraction `eta` of the batch on the diagonal (times `t ~ U([0,1])`, `s=t`) and `1-eta` in the upper triangle (draw two uniforms, take min and max so `s ≤ t`). Most of the learning signal — the actual velocity — comes from the diagonal, and distilling it into finite jumps is the smaller, more expensive part; so I want the majority of the batch on the cheap, information-rich diagonal. A diagonal fraction around `eta = 0.75` puts three-quarters of the compute on learning the flow and a quarter on distilling it into the map, which also keeps the per-step cost down given the off-diagonal term's `3×` evaluation count. `eta` doubles as the knob that trades training cost against how aggressively I distill.

Let me write the diagonal term concretely so the two halves match. With the linear interpolant `alpha_t = 1-t`, `beta_t = t`, the interpolant is `I_t = (1-t) x_0 + t x_1` and its velocity is `I_dot_t = x_1 - x_0`. The diagonal loss is flow matching on `v_{t,t}`,

  L_b^t = E | v_{t,t}(I_t) - I_dot_t |^2,

wrapped in the same weight: `e^{-w_{t,t}} L_b^t + w_{t,t}`. On the diagonal `s=t` so the time-gap prefactor in `X` is zero and `v_{t,t}` is read straight from the network — that's why the diagonal evaluation is one forward pass. In code the network exposes `calc_b(t,x) = v_{t,t}(x)` precisely as `calc_phi(t,t,x)`, and `X.apply(..., return_X_and_phi=True)` returns both `X_{s,t}(x) = x + (t-s) v_{s,t}(x)` and the raw slope `v_{s,t}(x)` so I can use the slope form directly off-diagonal.

Let me put the whole off-diagonal computation into the code I'd actually run, in JAX, filling the one empty slot. The three evaluations: the student slope `v_{s,t}(I_s)` (gradients flow), and the two teacher slopes `v_{s,u}(I_s)` and `v_{u,t}(X_{s,u}(I_s))` (gradients stopped). Form the convex-combination teacher with weights `(1-h, h)` for `h = gamma`, take the squared residual, wrap in the weight.

```python
import jax
import jax.numpy as jnp


def psd_term(
    params, teacher_params,
    x0, x1, label,
    s, t, u, h,           # h is the split fraction gamma; u = h*s + (1-h)*t
    rng, *, interp, X,
    psd_type, stopgrad_type,
):
    """Off-diagonal progressive self-distillation loss for v_{s,t} via the semigroup.

    Big jump should equal two small jumps glued at u; read off the slopes so the
    (t-s)^2 scaling cancels, then regress v_{s,t} onto the convex-combination teacher."""
    Is = interp.calc_It(s, x0, x1)                       # I_s = alpha_s x0 + beta_s x1

    # student: the full jump's slope v_{s,t}(I_s)  (gradients flow through this)
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )

    # teacher: the two-segment composition, slopes of [s,u] and [u,t]
    if stopgrad_type == "convex":
        # freeze the teacher so information flows diagonal -> off-diagonal, not back;
        # also avoids backprop through the inner X_{s,u} (the bootstrap) for stability
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(teacher_params, s, u, Is, label, train=False,
                    rngs=rng, return_X_and_phi=True)
        )
        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(teacher_params, u, t, X_su, label, train=False,   # starts where seg 1 landed
                    rngs=rng, return_X_and_phi=True)
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(params, s, u, Is, label, train=False,
                               rngs=rng, return_X_and_phi=True)
        X_ut, phi_ut = X.apply(params, u, t, X_su, label, train=False,
                               rngs=rng, return_X_and_phi=True)
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    # convex-combination teacher: v_{s,t} = (1-gamma) v_{s,u} + gamma v_{u,t}(X_{s,u})
    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut          # gamma ~ U([0,1]) drawn into h
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)                # gamma = 1/2 fixed
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")

    psd_loss = jnp.sum((student - teacher) ** 2)         # convex mode freezes teacher above

    # EDM2-style adaptive weight: e^{-w} L + w equalizes loss scale across (s,t)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * psd_loss + weight_st
```

and the diagonal half it pairs with:

```python
def diagonal_term(params, x0, x1, label, t, rng, *, interp, X):
    """Flow matching on the diagonal: v_{t,t} = b_t = E[I_dot_t | I_t]."""
    It = interp.calc_It(t, x0, x1)                       # I_t
    It_dot = interp.calc_It_dot(t, x0, x1)               # I_dot_t = x1 - x0 for linear interp
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)  # v_{t,t}(I_t)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt
```

I draw the split fraction in the sampler as `h ~ U([0,1])` and set `u = h s + (1-h) t`, exactly the `gamma`-parameterization. I keep the diagonal rows first and the off-diagonal rows second; the loss wrapper `vmap`s `diagonal_term` over the first slice and `psd_term` over the second, then returns `(diag_loss * diag_bs + offdiag_loss * offdiag_bs) / total_bs`. I pass the current training params in as `teacher_params`; in the convex setting the teacher evaluations are stopped, so the parameters are instantaneous but the target has no gradient. With `eta = 0.75` of the batch on the diagonal, I run the update with gradient clipping and a square-root-decayed learning rate, while EMA copies of the weights are tracked for sampling.

Let me retrace the causal chain so I trust it. The cost was sequential ODE integration at inference, so I went after the finite-jump operator `X_{s,t}` instead of the instantaneous drift. I parameterized `X_{s,t}(x) = x + (t-s) v_{s,t}(x)` to bake in the identity-on-the-diagonal constraint, which as a bonus made `v_{t,t} = b_t` so the diagonal trains by ordinary flow matching with no teacher. For the off-diagonal I picked the semigroup characterization — one big jump equals two small jumps — because alone among the three certifications it needs no spatial Jacobian and no time-derivative of the teacher, just three function evaluations, which is the cheapest and most stable thing to differentiate; and because the model can be its own teacher, I get direct training with no pretrained model. Writing the composition residual on `X` exposed a `(t-s)^2` scaling that makes the effective learning rate depend on the sampled jump size — pure variance — so I cancelled it by reading the semigroup off the *slopes*, giving the convex-combination relation `v_{s,t} = (1-gamma) v_{s,u} + gamma v_{u,t}(X_{s,u})` with `u = gamma s + (1-gamma) t`. I drew `gamma ~ U([0,1])` to supervise the map under all splits of a jump, not just the midpoint. I stopgradded the composed target so information flows one way, diagonal → off-diagonal, matching the distillation setting and avoiding backprop through the bootstrap. I wrapped both terms in the EDM2 `e^{-w} L + w` weight, whose finite optimum for positive `L` is `w* = log L`, to self-normalize the loss scale across `(s,t)` and run a larger learning rate. And I spent most of the batch (`eta = 0.75`) on the cheap, signal-rich diagonal, using the off-diagonal — which degenerates into flow matching as `s → t` — as its smooth extension into the interior. The result is a single model trained from scratch: its diagonal learns the real velocity, the semigroup slope residual distills that velocity into finite jumps, and generation can use as few as one network evaluation.
