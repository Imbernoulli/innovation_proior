Let me start from the thing that actually bugs me. I want a generative model that maps pure Gaussian noise to data, and I have a beautiful object for it: a continuous normalizing flow. I pick a time-dependent vector field `v_t(x;θ)` — a neural net — and let it carry points along the ODE `dφ_t/dt = v_t(φ_t)`, `φ_0(x)=x`. As `t` runs from 0 to 1, the prior `p_0 = N(0,I)` gets pushed forward into some `p_1`, and if I'm lucky `p_1 ≈ q`, the data. This is lovely because it's deterministic, invertible, and gives me exact likelihoods through the instantaneous change of variables: along a trajectory from `x_0`, `d/dt log p_t(φ_t(x_0)) = -div(v_t(φ_t(x_0)))`, so `log p_1(φ_1(x_0)) = log p_0(x_0) - ∫_0^1 div(v_t(φ_t(x_0))) dt`. So in principle I can both sample and score arbitrarily.

The trouble is training. The natural objective is maximum likelihood: push data back to noise, accumulate the divergence, maximize `log p_1` on data. But to even *evaluate* that objective I have to numerically solve the ODE — a long chain of network calls per example — and then backprop through (or adjoint-solve) that whole integration, every gradient step. It's sequential and slow, and the divergence is `O(d²)` unless I Hutchinson-estimate it. This is exactly why continuous flows have been stuck at tiny images. So the real question isn't "is the model expressive" — it obviously is, it can represent any path — the question is: can I get a training signal *without simulating the ODE in the loop*, and without biasing the gradient?

Let me write down what I'd actually *want* to regress on. Suppose I somehow knew a target probability path `p_t`, from noise at `t=0` to (approximately) data at `t=1`, and I knew the vector field `u_t` that generates it. Then I wouldn't need likelihoods at all — I'd just fit my network field to the true field directly:

```
L_FM(θ) = E_{t~U[0,1], x~p_t(x)} || v_t(x) - u_t(x) ||² .
```

If this hits zero, `v_t = u_t` everywhere along the path, so my flow reproduces `p_t`, and at `t=1` I land on the chosen data-end distribution. No ODE solve in the loss — it's a plain regression. This is the thing I want. Call it flow matching.

But stare at it: I have no idea what `p_t` should be, and far worse, I have no closed form for `u_t`. What does it even mean for `u_t` to "generate" `p_t`? That's the continuity equation: `u_t` generates `p_t` iff `∂_t p_t + div(p_t u_t) = 0`. It's just conservation of probability mass — mass is advected, not created. So `u_t` is implicitly tied to `p_t` through a PDE. Picking `p_t` out of thin air and then solving that PDE for `u_t` in `d` dimensions is hopeless. So `L_FM` as written is a fantasy: both arguments of the regression are unknown. Wall.

OK, back up. The reason `u_t` is hopeless is that `p_t` is a single global object stitched over the whole unknown data distribution. What if I don't try to specify the path globally at all, but build it up out of pieces I *can* write down — one piece per data example?

Take a single data point `x_1`. I can easily design a *conditional* path `p_t(x|x_1)` that does the right thing at the endpoints: at `t=0` it's the prior `p(x)=N(0,I)` (same for every `x_1`), and at `t=1` it's a tiny blob concentrated at `x_1`, say `N(x_1, σ²I)` with `σ` small. That's trivial — I'm just describing a Gaussian sliding from the origin out to `x_1` and shrinking. Now if I mix these over the data,

```
p_t(x) = ∫ p_t(x|x_1) q(x_1) dx_1 ,
```

then at `t=1`, `p_1(x) = ∫ p_1(x|x_1) q(x_1) dx_1`, which is a mixture of tiny blobs at every data point — that's essentially `q` itself (a kernel density estimate that sharpens as `σ→0`). And at `t=0` every conditional is the same prior, so the mixture is the prior. So this mixture `p_t` *is* a valid path from noise to data, and I never had to know `q`'s density — only how to sample `x_1` from it. Good, that's `p_t` handled.

Now the field. I need `u_t` generating this mixture `p_t`. For a single conditional path I can hope to know its generating field `u_t(x|x_1)` (I'll construct it below). The question is how the conditional fields combine into the marginal field. Mixing densities is just a weighted sum — but velocities don't add like densities. So how do I aggregate? Let me think about it as: at a point `x` and time `t`, multiple conditionals could have deposited mass here, and each wants to move it in its own direction `u_t(x|x_1)`. The net velocity should be the *mass-weighted average* of those directions — weighted by how much each conditional contributes to the density at `x`. The contribution of `x_1` to the density at `x` is `p_t(x|x_1) q(x_1)`, and the total is `p_t(x)`, so the weights are the posterior `p_t(x|x_1) q(x_1) / p_t(x)`. That gives

```
u_t(x) = ∫ u_t(x|x_1) · [ p_t(x|x_1) q(x_1) / p_t(x) ] dx_1 .
```

That's a guess from physical intuition. I need to *prove* this `u_t` actually generates the mixture `p_t`. Let me just check the continuity equation directly. Differentiate the mixture in time:

```
d/dt p_t(x) = ∫ (d/dt p_t(x|x_1)) q(x_1) dx_1 .
```

Each conditional satisfies its own continuity equation, `d/dt p_t(x|x_1) = -div( u_t(x|x_1) p_t(x|x_1) )`, so

```
d/dt p_t(x) = -∫ div( u_t(x|x_1) p_t(x|x_1) ) q(x_1) dx_1
            = -div( ∫ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1 ) ,
```

pulling the divergence (a spatial operator) outside the `x_1` integral. Now look at the integral inside the divergence: `∫ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1`. By my definition of `u_t`, that integral is exactly `u_t(x) p_t(x)` — the `p_t(x)` denominator in `u_t` cancels against... wait, let me be careful. My `u_t(x)` is `(1/p_t(x)) ∫ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1`. So `u_t(x) p_t(x) = ∫ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1`. Exactly the bracket. So

```
d/dt p_t(x) = -div( u_t(x) p_t(x) ) ,
```

which *is* the continuity equation for `(p_t, u_t)`. So `u_t` generates `p_t`. The posterior-averaging guess was right, and the reason it works is precisely that the `1/p_t` weighting is what turns the sum of per-example mass-fluxes into a single divergence. (The Leibniz swap of `d/dt` and `∫`, and pulling `div` out, need mild regularity — integrands decaying fast enough — which is fine for Gaussians.)

So now I have an honest `p_t` and an honest `u_t`, both defined as integrals over the data. Can I plug them into `L_FM`? No — `u_t(x)` is that posterior integral, intractable; I can't even evaluate it at one point without integrating over all of `q`. So `L_FM` is *still* unusable. Same wall, dressed differently.

But now I have a lever I didn't have before: I have a closed form for the *conditional* field `u_t(x|x_1)`, and the marginal field is its posterior average. This smells exactly like the situation with score matching — there, the marginal score `∇log p_t(x)` is intractable, but people regress against the conditional score `∇log p_t(x|x_1)` and it just works, because the conditional target averages (over the posterior) to the marginal one. Let me see if the same trick applies to regressing a velocity field, not a score.

So instead of `L_FM`, propose the conditional version — regress against the *conditional* field, sampling `x` from the conditional path:

```
L_CFM(θ) = E_{t~U[0,1], x_1~q(x_1), x~p_t(x|x_1)} || v_t(x) - u_t(x|x_1) ||² .
```

This one I can actually compute: sample `t`, sample a real data point `x_1`, sample `x` from the conditional Gaussian, and I know `u_t(x|x_1)` in closed form. Unbiased, one network call, no integration. The only question that matters: does minimizing this give me the same network as minimizing `L_FM`? If `v_θ` is expressive, `L_CFM` is pointwise minimized by `v_t(x) = E[u_t(x|x_1) | x_t = x]`, the posterior mean — which I just showed equals the marginal `u_t(x)`. So the *minimizer* matches. But I want more than that; I want the whole gradient to match so that *any* optimization of `L_CFM` is optimization of `L_FM`. Let me prove the gradients are identical.

Expand both squared norms. Bilinearity of the inner product:

```
|| v_t(x) - u_t(x)    ||² = ||v_t(x)||² - 2⟨v_t(x), u_t(x)⟩    + ||u_t(x)||²
|| v_t(x) - u_t(x|x_1)||² = ||v_t(x)||² - 2⟨v_t(x), u_t(x|x_1)⟩ + ||u_t(x|x_1)||² .
```

The last term in each is independent of `θ` (the targets don't depend on the network), so it drops under `∇_θ`. I just need the first two terms to have equal expectations under the two different sampling distributions.

First term, `||v_t(x)||²`. Under `L_FM` the expectation is over `x~p_t(x)`:

```
E_{p_t(x)} ||v_t(x)||² = ∫ ||v_t(x)||² p_t(x) dx
                       = ∫ ||v_t(x)||² [∫ p_t(x|x_1) q(x_1) dx_1] dx
                       = E_{q(x_1), p_t(x|x_1)} ||v_t(x)||² .
```

I just unfolded the marginal `p_t` into its mixture and swapped the order of integration. So the `||v||²` term is literally the same expectation whether I sample `x` from the marginal or from `(x_1, conditional)`. Good.

Cross term, `⟨v_t(x), u_t(x)⟩`. Under `L_FM`:

```
E_{p_t(x)} ⟨v_t(x), u_t(x)⟩
  = ∫ ⟨ v_t(x), u_t(x) ⟩ p_t(x) dx
  = ∫ ⟨ v_t(x), (1/p_t(x)) ∫ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1 ⟩ p_t(x) dx .
```

The `1/p_t(x)` inside the inner product meets the `p_t(x)` from the measure and they annihilate:

```
  = ∫ ⟨ v_t(x), ∫ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1 ⟩ dx
  = ∫∫ ⟨ v_t(x), u_t(x|x_1) ⟩ p_t(x|x_1) q(x_1) dx_1 dx
  = E_{q(x_1), p_t(x|x_1)} ⟨ v_t(x), u_t(x|x_1) ⟩ ,
```

pulling `v_t(x)` (constant in `x_1`) inside and swapping integration order. So the cross term, too, is the same expectation under both samplings. Both `θ`-dependent terms agree; the leftover `||·||²` terms differ but are `θ`-independent constants. Therefore

```
L_FM(θ) = L_CFM(θ) + const ,   and   ∇_θ L_FM = ∇_θ L_CFM .
```

That is the leverage I needed. Optimizing the tractable per-example loss is *exactly* optimizing the intractable global one. I never need the marginal path or the marginal field — I only need to design good conditional paths `p_t(x|x_1)` and write down their conditional fields `u_t(x|x_1)`. And notice the cancellation in the cross term is the *same* `1/p_t` algebra that made the marginal field generate the marginal path; it's one identity doing double duty.

So now everything reduces to: choose a conditional path, and get its conditional field. Let me make the conditional path Gaussian — that's the family I can sample from trivially and where I'll have closed forms:

```
p_t(x|x_1) = N( x | μ_t(x_1), σ_t(x_1)² I ) ,
```

with a time-dependent mean `μ_t(x_1)` and scalar std `σ_t(x_1)`. The boundary conditions I argued for fix the ends: at `t=0` I want the shared prior `N(0,I)`, so `μ_0(x_1)=0`, `σ_0(x_1)=1`; at `t=1` I want a concentrated blob at `x_1`, so `μ_1(x_1)=x_1`, `σ_1(x_1)=σ_min` with `σ_min` small. Anything smooth in between is allowed.

Now I need `u_t(x|x_1)`. A path has infinitely many generating fields — I can add a correction `r_t` whenever its probability flux has zero divergence, `div(p_t r_t)=0`, and the continuity equation still sees the same density evolution. Those corrections spin mass around without changing `p_t`; they're wasted motion and would make the field harder to fit and the trajectories curlier. So I want the *canonical*, swirl-free field — the one from the simplest flow that realizes this Gaussian. The simplest map that turns a standard normal into `N(μ_t, σ_t²I)` is the affine reparameterization

```
ψ_t(x) = σ_t(x_1) x + μ_t(x_1) ,
```

i.e. scale by `σ_t`, shift by `μ_t`. If `x~N(0,I)` then `ψ_t(x) ~ N(μ_t, σ_t²I)` — so `ψ_t` pushes the prior to the conditional, `[ψ_t]_* p = p_t(·|x_1)`. This `ψ_t` is a genuine flow (smooth, invertible since `σ_t>0`), and a flow has a *unique* generating field, recovered from its own defining ODE `d/dt ψ_t(x) = u_t(ψ_t(x)|x_1)`. Let me solve for `u_t`. Write `w_t(·) = u_t(·|x_1)` to declutter. Since `ψ_t` is invertible, put `y = ψ_t(x)`, so `x = ψ_t⁻¹(y)` and the ODE reads `ψ_t'(ψ_t⁻¹(y)) = w_t(y)`. Two pieces: invert the affine map,

```
ψ_t⁻¹(y) = (y - μ_t(x_1)) / σ_t(x_1) ,
```

and differentiate `ψ_t` in time at fixed `x`,

```
ψ_t'(x) = σ_t'(x_1) x + μ_t'(x_1) ,
```

where `'` is `d/dt`. Now evaluate `ψ_t'` at `x = ψ_t⁻¹(y)`:

```
w_t(y) = σ_t'(x_1) · (y - μ_t(x_1))/σ_t(x_1) + μ_t'(x_1) ,
```

so, renaming `y→x`,

```
u_t(x|x_1) = (σ_t'(x_1)/σ_t(x_1)) ( x - μ_t(x_1) ) + μ_t'(x_1) .
```

There it is — a closed form for the conditional field in terms of `μ_t, σ_t` and their time-derivatives. Every ingredient of `L_CFM` is now elementary: sample `x_1`, sample `ε~N(0,I)`, set `x = σ_t ε + μ_t`, and the target is `u_t(x|x_1)` above. In fact, since I'm sampling `x` via `ψ_t(x_0)` with `x_0~N(0,I)`, the target is just `d/dt ψ_t(x_0) = σ_t' x_0 + μ_t'`, which is even cleaner — no inversion needed at training time. So

```
L_CFM(θ) = E_{t, q(x_1), p(x_0)} || v_t(ψ_t(x_0)) - (σ_t' x_0 + μ_t') ||² .
```

What should `μ_t, σ_t` be? The formula is general, so different choices give different methods. Let me first check that the diffusion paths fall out — that would tell me diffusion is just one corner of this design space and I never needed to reason about SDEs at all.

Take the Variance Exploding path: a process that just keeps injecting noise, conditional marginal `N(x_1, σ_t²I)` with `σ_t` increasing. This is not going to satisfy the exact `N(0,I)` boundary I ultimately want — it only makes the noise end approximately independent of `x_1` by taking the variance large — but it should still drop out of the Gaussian formula. In my (noise→data) time it reads `μ_t(x_1)=x_1`, `σ_t(x_1)=σ_{1-t}` (so std is large near `t=0`, the noise end, and small near `t=1`). Then `μ_t'=0`, `σ_t' = -σ_{1-t}'`, and my formula gives

```
u_t(x|x_1) = (-σ_{1-t}'/σ_{1-t})(x - x_1) ,
```

a pure radial contraction toward `x_1`. Take Variance Preserving — the DDPM-style process — whose conditional marginal is `N(α_{1-t} x_1, (1-α_{1-t}²) I)` with `α_s = e^{-T(s)/2}`, `T(s)=∫_0^s β(r)dr`. So `μ_t = α_{1-t} x_1`, `σ_t = √(1-α_{1-t}²)`. Let `s=1-t` and `a=α_s`. Since `dα_s/ds=-(T'(s)/2)α_s`, differentiating with respect to my reversed time gives `da/dt=(T'(s)/2)a`, and therefore `σ_t'=-a(da/dt)/√(1-a²)`. Plugging those into the Gaussian-field formula gives

```
u_t(x|x_1) = (T'(s)/2) [ (a x_1 - a² x) / (1-a²) ]
           = -(T'(1-t)/2) [ (e^{-T(1-t)} x - e^{-T(1-t)/2} x_1) / (1 - e^{-T(1-t)}) ] .
```

Now is this field actually the right one for the diffusion path? Diffusion gives me a *stochastic* process; I should sanity-check against its deterministic counterpart. An SDE `dy = f_t dt + g_t dw` has marginals obeying Fokker–Planck, `∂_t p_t = -div(f_t p_t) + (g_t²/2) Δ p_t`. Rewrite the Laplacian term as a divergence: `(g_t²/2) Δ p_t = (g_t²/2) div(∇p_t) = div( (g_t²/2)(∇p_t/p_t) p_t ) = div( (g_t²/2 ∇log p_t) p_t )`. So `∂_t p_t = -div( [f_t - (g_t²/2)∇log p_t] p_t )`, which is the continuity equation with deterministic field `w_t = f_t - (g_t²/2)∇log p_t`. For the VP SDE `f_t(y) = -(T'(t)/2) y`, `g_t = √(T'(t))`, and the conditional `∇log p_t(y|y_0)` of a Gaussian is `-(y-μ)/σ²`; substituting and then flipping time (a field `u_t` generating `p_t` becomes `-u_{1-t}` generating the reversed path `p_{1-t}`, which I can verify in one line from the continuity equation) reproduces exactly the `u_t(x|x_1)` above. So my construction coincides with the diffusion probability-flow field on diffusion paths — but I got it by a one-line affine derivation, with no SDE, no Fokker–Planck, no time-reversal needed in the main argument. Diffusion is just the choice `μ_t = α_{1-t}x_1`, `σ_t = √(1-α_{1-t}²)`.

And here's a wrinkle that bugs me about diffusion paths now that I can see the whole family: they never actually reach a clean Gaussian in finite time. `α_{1-t}` only hits the noise end asymptotically, so at `t=0` the conditional isn't exactly `N(0,I)` — people fudge it with an approximation and a small `ε` cutoff. That's a defect inherited from the SDE, not something I have to live with. Since I get to *pick* `μ_t, σ_t` directly, I can just demand the boundary conditions hold exactly and choose the in-between freely. So let me ask: of all the paths, which is the *nicest* — easiest to fit, cheapest to integrate at sampling time?

Look at what makes the diffusion field annoying: `u_t(x|x_1)` depends on `x` and `t` in a tangled, time-varying way (those `e^{-T(1-t)}` factors), so the field's direction *rotates* as `t` advances — the trajectories curve and can even overshoot the target and backtrack. The regression target keeps changing direction; the solver has to take many small steps. What if I force the simplest possible time-dependence? Let the mean and std move *linearly* in time:

```
μ_t(x_1) = t x_1 ,   σ_t(x_1) = 1 - (1 - σ_min) t .
```

Check the boundaries: at `t=0`, `μ=0`, `σ=1` — the standard normal, exactly. At `t=1`, `μ=x_1`, `σ=σ_min` — a blob at `x_1`, exactly. No asymptotics, no cutoff. And the derivatives are constants: `μ_t' = x_1`, `σ_t' = -(1-σ_min)`. Drop them into the conditional-field formula:

```
u_t(x|x_1) = (σ_t'/σ_t)(x - μ_t) + μ_t'
           = ( -(1-σ_min) / (1-(1-σ_min)t) )( x - t x_1 ) + x_1 .
```

Combine over the common denominator: the numerator is `-(1-σ_min)(x - t x_1) + x_1(1-(1-σ_min)t) = -(1-σ_min)x + (1-σ_min)t x_1 + x_1 - (1-σ_min)t x_1 = x_1 - (1-σ_min)x`. The `t x_1` terms cancel. So

```
u_t(x|x_1) = ( x_1 - (1-σ_min) x ) / ( 1 - (1-σ_min) t ) ,
```

defined for *all* `t∈[0,1]` (the diffusion field, by contrast, blows up where `1-e^{-T(1-t)}→0`). And in the cleaner `x_0` parameterization, where `x = ψ_t(x_0) = (1-(1-σ_min)t)x_0 + t x_1`, the target collapses even further. Substitute: `x_1 - (1-σ_min)x = x_1 - (1-σ_min)[(1-(1-σ_min)t)x_0 + t x_1]`. If I write `c=1-σ_min`, this numerator is `(1-ct)x_1 - c(1-ct)x_0 = (1-ct)(x_1-cx_0)`, so the same factor as the denominator cancels and the whole thing equals

```
u_t(ψ_t(x_0)|x_1) = x_1 - (1-σ_min) x_0 ,
```

which is *constant in `t`*. So the CFM loss for this path is just

```
L_CFM(θ) = E_{t, q(x_1), p(x_0)} || v_t(ψ_t(x_0)) - ( x_1 - (1-σ_min) x_0 ) ||² ,
```

regress the network onto a fixed vector `x_1 - (1-σ_min)x_0` per noise/data pair. The field has constant direction in time — it factors as `g(t) h(x|x_1)` — which means the conditional trajectory is a *straight line at constant speed* from the noise sample to (near) the data point.

Why is straight-and-constant-speed the right thing to want, beyond "it's simple"? The regression target stops rotating with `t`, so the network has a much easier function to fit — at fixed `(x_0,x_1)` it's literally a constant. And sampling integrates `dx/dt = v_t(x)` with a numerical solver, and a solver spends its step budget fighting curvature; on straight constant-speed paths even a coarse, few-step solver is nearly exact, so I get low-NFE sampling for free. The curved diffusion path wastes solver steps and can overshoot; the straight path can't.

And there's a deeper reason this particular linear choice is special, not arbitrary. The flow `ψ_t(x_0) = (1-(1-σ_min)t)x_0 + t x_1` is exactly the *optimal-transport displacement interpolation* between the two endpoint Gaussians `p_0(·|x_1)=N(0,I)` and `p_1(·|x_1)=N(x_1,σ_min²I)`. The Wasserstein-2 geodesic between distributions moves mass along straight lines at constant speed — that's the definition of displacement interpolation — and for two Gaussians with one being standard it is precisely this affine map. So the "simplest" choice I reached by demanding linear `μ_t,σ_t` is the *geometrically optimal* one: it transports each conditional with minimal kinetic energy. (One honest caveat: the conditional flow being OT-optimal does not make the *marginal* flow an OT map — the mixture mixes — but I'd still expect the marginal field to stay relatively simple, since it's a posterior average of these simple straight fields.)

So the recipe is settled. I never construct a global path or field; I never simulate an ODE during training; the gradient is unbiased and exact; and I can pick *any* `μ_t,σ_t` — diffusion paths drop out as one choice, but the linear/OT path is simpler, straighter, cheaper to sample, and optimal in the transport sense. Let me write it as code, mirroring the structure of the framework I started from: a path object that, given `(x_0,x_1,t)`, returns the point to feed the network and the vector to regress onto; a training loop that just MSEs the network against that target; and sampling that integrates the trained field from noise to data.

```python
import torch


def pad_t_like_x(t, x):
    """Reshape a batch of scalar times so it broadcasts over x."""
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def sample_prior_like(x):
    return torch.randn_like(x)


@torch.no_grad()
def odeint(field, x0, t0, t1, steps=100):
    x = x0
    dt = (t1 - t0) / steps
    n = x0.shape[0]
    for i in range(steps):
        t = torch.full((n,), t0 + i * dt, device=x.device, dtype=x.dtype)
        x = x + field(x, t) * dt
    return x


class TrainingPairBuilder:
    """Gaussian conditional path with linearly moving mean/std:
    mu_t = t*x1, sigma_t = 1 - (1 - sigma_min) t."""

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def compute_mu_t(self, x0, x1, t):
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        return 1 - (1 - self.sigma_min) * t

    def sample_xt(self, x0, x1, t):
        # x0 is the standard-normal draw; xt ~ N(t*x1, sigma_t^2 I).
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = pad_t_like_x(self.compute_sigma_t(t), x1)
        return mu_t + sigma_t * x0

    def compute_conditional_flow(self, x0, x1, t, xt):
        # u_t(x_t|x1) = (x1 - (1 - sigma_min) x_t) / sigma_t.
        # For xt from sample_xt this is x1 - (1 - sigma_min) x0, constant along the line.
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma_min) * xt) / (1 - (1 - self.sigma_min) * t)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

def train_step(pair_builder, v_theta, opt, x1):
    x0 = sample_prior_like(x1)                 # prior sample
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    t, xt, ut = pair_builder.sample_location_and_conditional_flow(x0, x1, t)
    loss = ((v_theta(xt, t) - ut) ** 2).mean() # the CFM objective
    opt.zero_grad(); loss.backward(); opt.step()
    return loss

def sample(v_theta, n, shape, steps=100, device="cpu"):
    # integrate dx/dt = v_theta(x, t) from noise (t=0) to data (t=1); straight paths -> few steps suffice
    x = torch.randn(n, *shape, device=device)
    return odeint(lambda x, t: v_theta(x, t), x, 0.0, 1.0, steps=steps)
```

The causal chain, end to end: I wanted to train a continuous flow without solving its ODE in the loop, so I tried regressing the network field onto the true generating field (flow matching) — but the true marginal field is an intractable integral over the data. I rescued the path by building it as a mixture of per-example Gaussian conditionals, proved (via the continuity equation) that the posterior-average of conditional fields generates that mixture, and then proved (via expanding the squared loss and watching the marginal density cancel in the cross term) that regressing on the *conditional* field has the identical gradient — so I can train per-example with closed-form targets, unbiased, one network call, no simulation. Choosing Gaussian conditionals with a canonical affine flow gave a one-line formula for the conditional field; plugging in the diffusion schedules recovers diffusion as a special case, while choosing the mean and std to move linearly gives a constant-direction field whose straight, constant-speed trajectories are the optimal-transport interpolant — easier to fit and far cheaper to integrate at sampling time.
