Let me start from what actually hurts. I have a flow-matching model that works beautifully when I give it enough sampling steps, and falls apart when I demand one. The structure is clean: I build a path between data and noise, `z_t = a_t x + b_t ε`, take the rectified schedule `a_t = 1 - t`, `b_t = t` so `z_t = (1-t)x + tε`, and the instantaneous velocity is the time derivative `v_t = a_t' x + b_t' ε = ε - x`, a constant in time along each `(x, ε)` pair. I train a network to fit the marginal velocity `v(z_t, t) = E[v_t | z_t]` — the expectation over all pairs that pass through `z_t` — and I generate by integrating `dz_t/dt = v(z_t, t)` from `z_1 = ε` down to `z_0`. The integral form is what I actually care about: `z_r = z_t - ∫_r^t v(z_τ, τ) dτ`. And there's the rub. Even though each conditional path is dead straight, the marginal field curves — at a point where two pairs cross, the expectation mixes their directions, and that curvature is a property of the true field, not of any approximation error in my net. So `∫_r^t v dτ` over a curved trajectory needs many Euler steps to approximate; one step from pure noise lands nowhere near the data. I have the integrand `v` at every point, and what I want is the integral.

So the honest reframing: I keep trying to learn `v`, the *instantaneous* velocity, and then at sampling time I'm forced to reconstruct an integral from it numerically. What if I stop fighting the integral at sampling time and instead learn the thing that already *is* the integral, divided by the interval — the quantity that, multiplied by the interval, gives the displacement directly? Define the average velocity over `[r, t]`:

  u(z_t, r, t) ≜ (1/(t - r)) ∫_r^t v(z_τ, τ) dτ.

This is just displacement over elapsed time. It's a field induced by `v` — it doesn't reference any neural network — exactly the way the marginal `v` is induced by the conditional velocities. And if I had it, sampling collapses: over any interval, `z_r = z_t - (t - r) u(z_t, r, t)`, because `(t-r)u` is by construction the displacement `∫_r^t v dτ`. One step from noise to data is then just `z_0 = z_1 - u(z_1, 0, 1)` with `z_1 = ε`. No integral to approximate at inference — the integral has been absorbed into the learned object. That's the whole appeal: I move the hard part from sampling time, where I can only afford a few evaluations, into training time, where I have a full budget.

Before I get excited, let me sanity-check the object. As the interval shrinks, `r → t`, the average of `v` over an infinitesimal window is just `v` at the point: `lim_{r→t} u(z_t, r, t) = v(z_t, t)`. Good — at zero interval the average velocity *is* the instantaneous velocity, so the new field contains the old one as a boundary. And there's a structural property I get for free, which is worth pausing on because the few-step literature works hard to *impose* something like it. Take any intermediate time `s` in `[r, t]`. The integral is additive: `∫_r^t v dτ = ∫_r^s v dτ + ∫_s^t v dτ`. Translate that into displacements: `(t-r)u(z_t, r, t) = (s-r)u(z_s, r, s) + (t-s)u(z_t, s, t)`. So taking one big jump over `[r, t]` is automatically consistent with taking two smaller consecutive jumps over `[r, s]` and `[s, t]`. A network that genuinely approximates this `u` *inherits* that consistency from the definition — I don't have to add a self-consistency loss to enforce it. That's a sharp contrast with the consistency-model way of doing things, where the consistency of outputs across the trajectory is precisely the thing the loss must hammer in, because there the target is a property of the network's behaviour rather than a known field. Here the field exists, the consistency is a theorem about it, and any accurate fit gets it as a side effect. The optimum is pinned to a ground-truth `u` that doesn't care which network I use, which is exactly the stability property I wanted.

Now the obstacle. I can't train against `u` directly — its definition is an integral, `(1/(t-r))∫_r^t v dτ`, and evaluating that during training means numerically integrating along the trajectory for every sample, which is the cost I was trying to escape. I have the integrand, not the integral. I need to turn this integral definition into something local — something I can compute from `v` at a single point plus derivatives I can get cheaply.

Let me write the definition without the division, as a displacement:

  (t - r) u(z_t, r, t) = ∫_r^t v(z_τ, τ) dτ.

The integral is on the right, with `t` as the upper limit. Differentiation kills integrals — the fundamental theorem of calculus turns `d/dt ∫_r^t v dτ` into just `v(z_t, t)` (treating `r` as independent of `t`, so the lower limit contributes nothing). So differentiate both sides with respect to `t`. On the left, `(t - r) u` is a product of `(t-r)` and `u`, both depending on `t`, so the product rule gives

  d/dt [(t - r) u] = (d/dt (t - r)) · u + (t - r) · d/dt u = u + (t - r) d/dt u,

since `d/dt (t - r) = 1`. On the right, the FTC gives `v(z_t, t)`. Set them equal:

  u(z_t, r, t) + (t - r) d/dt u(z_t, r, t) = v(z_t, t).

Rearrange to isolate `u`:

  u(z_t, r, t) = v(z_t, t) - (t - r) d/dt u(z_t, r, t).

There it is — a purely differential relation between the average velocity and the instantaneous velocity, with the integral gone. The instantaneous `v` is the only ground-truth signal that appears; the rest is a derivative of `u` itself. I'll call this the identity that `u` must satisfy. Notice the shape: `u` equals `v` minus a correction `(t-r) d/dt u`, and at `r = t` the correction vanishes and `u = v` exactly — consistent with the boundary I checked, and a hint that this whole thing degenerates to ordinary flow matching at zero interval. That's reassuring: I'm not throwing away flow matching, I'm extending it.

But I should be careful here, because differentiating an integral equation is a one-way street in general. I derived "the identity holds" *from* the definition, so the identity is a necessary condition. Is it sufficient? Equality of derivatives normally only pins down functions up to an additive constant — `dF/dt = dG/dt` gives `F = G + C`, not `F = G`. If the identity I just wrote were only necessary, a network satisfying it could still be the wrong field, off by a constant. Let me check whether the constant is actually forced to zero here. Work with the displacement `S(z_t, r, t) ≜ (t - r) u(z_t, r, t)`. The identity, multiplied through, says `d/dt S = v`. As an abstract statement about an unknown `S`, that only gives

  d/dt S = v ⟹ S + C_1 = ∫_r^t v dτ + C_2,

i.e. `S = ∫_r^t v dτ + (C_2 - C_1)` for some constant. Now evaluate at `t = r`. By the *definition* of `S`, `S = (t-r)u`, so `S|_{t=r} = 0 · u = 0`. And the integral `∫_r^t v dτ` also vanishes at `t = r` (empty interval). Plug `t = r` into `S = ∫ v dτ + (C_2 - C_1)`: `0 = 0 + (C_2 - C_1)`, so `C_2 = C_1`, the constant cancels, and `S = ∫_r^t v dτ` exactly — which is precisely the definition of `u`. So a network that satisfies the identity *does* recover the average-velocity definition; the identity is sufficient, not just necessary. And the reason it works is that I'm modeling `u`, which carries the factor `(t-r)` that forces `S|_{t=r} = 0` automatically. If I had instead chosen to parameterize the displacement `S` directly, that boundary `S|_{t=r} = 0` would *not* be automatic — I'd have to enforce it as an extra constraint by hand, exactly the kind of bolted-on boundary condition the flow-map and shortcut approaches carry. Modeling the average velocity, with its built-in `1/(t-r)` scaling, makes the boundary free. That settles a design choice I might have gotten wrong: model `u`, not `S`.

Now I need the correction term `d/dt u` to actually be computable. That `d/dt` is a *total* derivative along the trajectory — `u(z_t, r, t)` depends on `t` through three channels: directly through its third argument, through `z_t` which moves with `t`, and (in principle) through `r`. Expand by the chain rule:

  d/dt u(z_t, r, t) = (dz_t/dt) ∂_z u + (dr/dt) ∂_r u + (dt/dt) ∂_t u.

The three coefficients are exactly what the setup gives me: `dz_t/dt = v(z_t, t)` is the ODE itself; `dr/dt = 0` because I'm holding `r` independent of `t`; and `dt/dt = 1`. So two terms survive:

  d/dt u(z_t, r, t) = v(z_t, t) · ∂_z u + ∂_t u.

This is a directional derivative of the function `u` — the Jacobian `[∂_z u, ∂_r u, ∂_t u]` contracted with the tangent vector `(v, 0, 1)`. That's a Jacobian-vector product. And the JVP is precisely what forward-mode autodiff computes in roughly one extra pass without ever forming the Jacobian — `torch.func.jvp` or `jax.jvp`, feed the function `u`, the point `(z_t, r, t)`, and the tangent `(v, 0, 1)`, and it returns both `u` and `d/dt u` together. The data tangent along `∂_z u` is `d`-dimensional (the full image), while the tangents along `∂_r u` and `∂_t u` are one-dimensional, but those two scalars are what tie all the `(r, t)` coordinates of the field together, so they're not optional even though they're small. So the expensive-looking total derivative is, mechanically, one forward-mode evaluation.

Now put a network in. I parameterize `u_θ(z_t, r, t)` and want it to satisfy the identity. Reading the identity as "the left side should equal the right side," the right side `v - (t-r) d/dt u` is a target the network's output `u_θ` should match. Replace the true `u`'s derivatives in that correction by the network's own derivatives — I only have `u_θ`, not the true `u` — so the regression target is

  u_tgt = v - (t - r) (v · ∂_z u_θ + ∂_t u_θ),

and the loss is the squared distance between the prediction and this target:

  L(θ) = E || u_θ(z_t, r, t) - u_tgt ||²,   with u_tgt computed from u_θ.

Two things make me nervous about this, and I want to deal with both before I trust it. First: `u_tgt` contains derivatives of `u_θ`, i.e. it depends on `θ`. If I let gradients flow through `u_tgt`, then minimizing `||u_θ - u_tgt||²` would back-propagate through the JVP, which itself is a first derivative of the network — so the parameter gradient would involve a *second* derivative of the net, double-backprop, higher-order optimization, expensive and finicky. The fix is to freeze the target: apply a stop-gradient to `u_tgt`, treating it as a constant when I differentiate the loss with respect to `θ`. Then the loss is an ordinary regression of `u_θ` onto a fixed value, first-order SGD only; the JVP is computed to build the target value, but the outer parameter update does not differentiate through that JVP. This stop-gradient-on-the-target move is the same device the consistency-training line uses to make its self-consistency loss optimizable; here it's even cleaner because the target is a genuine field value, not a slowly-moving EMA of the network. And I should check the stop-gradient doesn't change the fixed point: if `u_θ` reaches zero loss, then `u_θ = u_tgt = v - (t-r)(v ∂_z u_θ + ∂_t u_θ)`, which is exactly the identity, which by the sufficiency argument above means `u_θ` equals the true average velocity. So freezing the target costs nothing at the optimum — it's purely an optimizability convenience.

Second worry: the identity has the *marginal* velocity `v(z_t, t) = E[v_t | z_t]` in it, and that's intractable — the same marginalization that makes plain flow matching's true objective uncomputable. But this is a solved problem for the instantaneous field, and I can borrow the solution with one extra check. Conditional flow matching says that regressing on the per-sample conditional velocity `v_t` gives, up to a `θ`-independent constant, the same gradient as regressing on the intractable marginal `v` — `∇_θ L_FM = ∇_θ L_CFM`. The mechanism is that `v(z_t, t)` is exactly the conditional expectation of `v_t` given `z_t`, so an L2 regression onto `v_t` has its minimizer at that conditional mean and the same gradient in expectation. My target uses `v` twice, once as the explicit leading term and once as the tangent inside the JVP. After I stop-gradient the target, the network derivatives inside that target are frozen for the outer loss, so for fixed `(z_t, r, t)` the target is an affine function of the velocity: `(I - (t-r)∂_z u_θ) v - (t-r)∂_t u_θ`. Taking the conditional expectation over sample pairs that produce the same `z_t` therefore replaces `v_t` by exactly the marginal `v` and leaves only a `θ`-independent noise term in the squared regression. So I substitute the conditional `v_t = ε - x` in both places — the explicit term and the JVP tangent — because I know it per sample for free. The target becomes

  u_tgt = v_t - (t - r) (v_t · ∂_z u_θ + ∂_t u_θ),   v_t = ε - x,

and the JVP tangent's data component is `v_t` as well. This is the move that makes the whole thing trainable: the only ground-truth signal is the per-sample instantaneous velocity `ε - x`, no integral, no marginalization.

Let me write the training step out concretely to make sure every piece is mechanical. Sample a data batch `x` and noise `ε`. Sample the two times `r, t`. Form the path point `z = (1-t)x + tε` and the conditional velocity `v = ε - x`. Call the JVP: `u, dudt = jvp(fn, (z, r, t), (v, 0, 1))`, which returns the network's `u_θ(z, r, t)` and the total time derivative `d/dt u_θ` in one shot. Form the target `u_tgt = v - (t - r) · dudt`, stop-gradient it, and the error is `u - sg(u_tgt)`, with `loss = ||error||²`. That's it — it looks *exactly* like flow matching, except the matching target carries the extra `-(t-r)(v ∂_z u_θ + ∂_t u_θ)` correction that comes from the average-velocity view. And the degeneration is now explicit: if I restrict to `r = t`, the factor `(t-r) = 0` wipes out the correction, `u_tgt = v`, and I'm doing plain conditional flow matching, regressing the instantaneous velocity. So this is a strict generalization — flow matching is the `r = t` slice.

That degeneration also tells me a knob I have to set: how often to sample `r = t` versus `r ≠ t`. At `r = t` the model just learns the instantaneous velocity, anchoring it to the well-behaved flow-matching objective; at `r ≠ t` it learns to propagate that into interval behavior through the correction term that links the `(r, t)` coordinates. If I always set `r = t`, I never train the off-diagonal interval behavior and the sampler has only the ordinary instantaneous field. If I never set `r = t`, I weaken the clean instantaneous anchor that the correction term is built from. So I want a mixture: some samples pinned to the flow-matching slice, the rest off the diagonal. In code I'll make that explicit as `fm_proportion`, the fraction with `r = t`; the complementary fraction has `r < t`.

The other sampling-side choice is the distribution over `(r, t)` themselves. Drawing both uniformly on `[0,1]` is the naive option, but the informative part of the trajectory — where the field is most curved and the model most needs resolution — sits in the middle times, not at the endpoints where the path is nearly pure noise or nearly clean data. A logit-normal draw — sample a Gaussian `N(μ, σ)` and squash it through the logistic function into `(0,1)` — concentrates mass in that mid-region and is what large-scale flow-matching training already favours. I draw two values from it per sample and assign the larger to `t`, the smaller to `r`, so `t ≥ r` always (the interval has a definite orientation). And I should decide what the network conditions on: it doesn't have to take `(r, t)` literally. The field is `u_θ(z, r, t)`, but `(t, t - r)` carries the same information and is arguably a more natural basis — current time and interval width — so I let the net take `(t, Δt)` with `Δt = t - r`, embedding each with a positional embedding through a small MLP and combining them. The JVP is still taken with respect to `u_θ(·, r, t)`; the reparameterization is just how the two scalars are fed in.

Now the loss metric. I wrote squared L2, `||Δ||²` with `Δ = u_θ - u_tgt`, and that's the clean floor — the simplest objective and the one I'll take as the base. There's a known generalization I should at least understand even if I don't switch to it: a powered loss `||Δ||^{2γ}`. Its gradient is `γ (||Δ||²)^{γ-1} · d/dθ ||Δ||²`, so minimizing `||Δ||^{2γ}` is the same as minimizing the plain squared loss `||Δ||²` weighted by a loss-adaptive factor `w ∝ ||Δ||^{2(γ-1)}`. In practice that's implemented as `w = 1/(||Δ||² + c)^p` with `p = 1 - γ` and a small `c` to avoid dividing by zero, and the weight is stop-gradiented so it only reweights, not redirects, the gradient; `p = 0.5` recovers something close to a Pseudo-Huber loss. But the pure squared-L2 case is `p = 0`, `w = 1`, and that's the base form — no adaptive reweighting, just `mean(||u_θ - sg(u_tgt)||²)`. I'll land on that as the floor and treat the adaptive `p` as an optional knob layered on top, not part of the core method.

Let me write the algorithm I'd actually ship for the pure squared-L2 base, filling the empty slot in the flow-matching harness — the regression target and the sampling update. The only real machinery beyond ordinary flow matching is the single `jvp` call and the stop-gradient on its product:

```python
import torch
from torch.func import jvp


def sample_t_r(batch_size, device, p_mean=-0.4, p_std=1.0, fm_proportion=0.75):
    """Draw (t, r): logit-normal, larger -> t, smaller -> r; fm_proportion keeps r = t.
    r = t reduces the target to plain flow matching; r != t trains interval behavior."""
    def logit_normal(n):
        u = torch.randn(n, device=device) * p_std + p_mean
        return torch.sigmoid(u)
    t = logit_normal(batch_size)
    r = logit_normal(batch_size)
    t, r = torch.maximum(t, r), torch.minimum(t, r)        # orient the interval t >= r
    use_fm = torch.rand(batch_size, device=device) < fm_proportion
    r = torch.where(use_fm, t, r)                           # r = t gives the flow-matching slice
    return t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)


def meanflow_loss(net, x):
    """One training step of MeanFlow with the pure squared-L2 (MSE) objective."""
    B = x.shape[0]
    device = x.device
    t, r = sample_t_r(B, device)

    eps = torch.randn_like(x)                              # prior sample
    z_t = (1 - t) * x + t * eps                           # conditional path z_t = a_t x + b_t eps
    v = eps - x                                            # conditional velocity v_t = eps - x

    # u_theta(z, r, t) conditioned on (t, t - r); JVP gives u and the total
    # time derivative d/dt u = v * d_z u + d_t u
    # tangent matches the (z, t, r) argument order of u_fn: (dz/dt, dt/dt, dr/dt) = (v, 1, 0)
    def u_fn(z, t_, r_):
        return net(z, t_, t_ - r_)                        # network takes (z, t, delta_t)

    primals = (z_t, t, r)
    tangents = (v, torch.ones_like(t), torch.zeros_like(r))
    u, dudt = jvp(u_fn, primals, tangents)               # one forward-mode pass

    # MeanFlow Identity as a regression target; stop-gradient avoids double-backprop
    interval = (t - r).clamp(0.0, 1.0)                    # no-op for valid sampled times
    u_tgt = v - interval * dudt
    u_tgt = u_tgt.detach()

    # Pure MSE on the mean-velocity prediction -- the floor objective
    loss = (u - u_tgt).flatten(1).pow(2).mean(dim=1).mean()
    return loss


@torch.no_grad()
def sample_one_step(net, shape, device):
    """1-NFE generation: z_0 = z_1 - u_theta(z_1, r=0, t=1)."""
    z = torch.randn(shape, device=device)
    t = torch.ones(shape[0], 1, 1, 1, device=device)
    r = torch.zeros(shape[0], 1, 1, 1, device=device)
    interval = t - r
    u = net(z, t, interval)                               # average velocity over [0, 1]
    return z - interval * u                               # one step covers the whole interval
```

Let me trace the causal chain end to end. I was stuck because flow matching learns the instantaneous velocity but sampling needs its integral over a curved marginal field, so one step is inaccurate and quality costs many evaluations. Instead of reconstructing the integral at sampling time, I defined the average velocity `u = (1/(t-r))∫_r^t v dτ`, a network-independent field whose product with the interval is exactly the displacement, so sampling over `[r,t]` is the single step `z_r = z_t - (t-r)u` and one-step generation is `z_0 = z_1 - u(z_1,0,1)`. The boundary `r→t` gives `u=v`, and integral additivity hands me consistency for free, so no self-consistency heuristic is needed and the optimum is pinned to a true field. Using `u` directly was intractable because it's an integral, so I differentiated the displacement form `(t-r)u = ∫_r^t v dτ` in `t` — product rule on the left, fundamental theorem of calculus on the right — to get the integral-free identity `u = v - (t-r) d/dt u`, and I checked it's sufficient as well as necessary because modeling `u` (not the displacement) forces the integration constant to cancel through `S|_{t=r}=0`. The total derivative `d/dt u = v ∂_z u + ∂_t u` is a Jacobian-vector product with tangent `(v,0,1)`, one cheap forward-mode pass. Parameterizing `u_θ` and regressing it onto `v - (t-r) d/dt u_θ` gives the loss; stop-gradient on the target avoids double-backprop and doesn't move the optimum; because the stopped target is affine in the velocity, the intractable marginal `v` can be replaced by the conditional `v_t = ε - x` in both the target and the JVP tangent by the conditional-flow-matching gradient equivalence. At `r=t` the correction vanishes and it's exactly flow matching, so I mix a fraction of `r=t` samples to keep the instantaneous anchor, draw `(r,t)` logit-normal to put resolution in the curved mid-times, condition on `(t, t-r)`, and take the pure squared-L2 loss as the base objective. The result drops into the flow-matching training loop as one extra `jvp` call and a stop-gradient — a strict, principled generalization of flow matching that learns the integral instead of approximating it at inference.
