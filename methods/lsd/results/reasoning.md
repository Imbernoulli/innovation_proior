Let me start from the thing that actually hurts. I have a flow-based generative model that works: I built a stochastic interpolant `I_t = alpha_t x_0 + beta_t x_1` between a Gaussian `rho_0` and the data `rho_1`, with `alpha_0=1, alpha_1=0, beta_0=0, beta_1=1`, and I learned the drift of the probability flow `xdot_t = b_t(x_t)` by the square-loss regression `L_b(bhat) = int_0^1 E|bhat_t(I_t) - Idot_t|^2 dt`, which works because `b_t(x) = E[Idot_t | I_t = x]` is exactly the conditional expectation a square loss recovers. The samples are great. The problem is purely the *cost of sampling*: to draw one `x_1` I integrate the ODE from `t=0` to `t=1`, and to integrate it accurately I evaluate the network at tens, sometimes hundreds, of substeps. The velocity `b_t` only tells me the instantaneous direction; to move a finite distance I have to keep re-asking it. Every substep is a full forward pass of a large UNet. So inference is one or two orders of magnitude slower than I'd like, and that kills it for anything interactive.

So what do I actually want? Not the velocity at a point — I want the *result of integrating* it. I want a map that, given a point on an ODE trajectory at time `s`, hands me the point on the same trajectory at time `t`, in one shot: `X_{s,t}(x_s) = x_t`, for every `(s,t)` and every trajectory. If I had that, then `X_{0,1}(x_0)` with `x_0 ~ rho_0` is a data sample in a *single* network evaluation. And it degrades gracefully: if the one-shot map is imperfect I can compose a handful of smaller jumps over a grid `0 = t_0 < ... < t_k = 1`, spending a few evaluations to buy quality. That's exactly the few-step regime I'm after. The object I want is this two-time jump map `X`, not the velocity.

How do people get such a map today, and why am I not happy with any of it? Two routes. One is *distillation*: train this fast jump map to imitate a separately pre-trained slow many-step model. It's stable and it works — but it's a two-phase pipeline (train the slow model, then train the fast one), it needs two models around, and the student can never be better than the teacher it's copying. The other route is *direct training* of the jump map from scratch — no teacher. That's what I want in principle, but the objectives people use for it are observed to be touchy: they diverge on large image networks and need a lot of dataset-specific babysitting to converge at all. Consistency models and their two-time extension (consistency trajectory models) live here; shortcut models live here. So the field's state is: either pay the teacher tax, or fight an unstable optimizer. Neither is a clean answer, and crucially there's no unifying account telling me *which* residual I should be minimizing, *why* its minimizer is exactly the jump map, and *what* I get for driving the loss down. Let me try to derive that from scratch, because if I can find the residual whose unique zero is the true map, and pick the one that dodges the known instabilities, I get direct training with distillation-grade stability and no teacher.

The first question is the most basic one: what do I even know about this map `X_{s,t}` for free, just from its definition `X_{s,t}(x_s) = x_t`? Let me differentiate that jump condition, because the map is defined *on trajectories* of `xdot_t = b_t(x_t)`, and trajectories are differentiable. Hold `s` and the trajectory fixed, vary `t`. The left side's `t`-derivative is `partial_t X_{s,t}(x_s)`. The right side is `xdot_t = b_t(x_t)`, and `x_t = X_{s,t}(x_s)`. So

```
partial_t X_{s,t}(x_s) = xdot_t = b_t(x_t) = b_t(X_{s,t}(x_s)).
```

Because `x_s` was an arbitrary point (every point is on some trajectory), this holds for all `x`:

```
partial_t X_{s,t}(x) = b_t(X_{s,t}(x)).
```

That's clean — it says the map, as a function of its end time, is *itself* an integral curve of the same velocity field `b`, just started from `X_{s,t}(x)` instead of from a noise sample. It's a Lagrangian description: follow the particle. Now do the same with `s`. Differentiate `X_{s,t}(x_s) = x_t` totally in `s`. The right side `x_t` doesn't depend on `s` (the endpoint is fixed once the trajectory and `t` are fixed), so its `s`-derivative is zero. The left side, by the chain rule, has two pieces — the explicit `s`-dependence of the map and the dependence through its moving argument `x_s`, whose `s`-derivative is `xdot_s = b_s(x_s)`:

```
0 = d/ds X_{s,t}(x_s) = partial_s X_{s,t}(x_s) + (grad X_{s,t})(x_s) . xdot_s
                      = partial_s X_{s,t}(x_s) + (grad X_{s,t})(x_s) . b_s(x_s),
```

so `partial_s X_{s,t}(x) + grad X_{s,t}(x) . b_s(x) = 0`. That's an Eulerian description — a transport PDE for the map in its start time, with the spatial gradient `grad X` of the map appearing. And there's a third fact that needs no derivative at all: jumping `s -> u` then `u -> t` along the same trajectory is the same as jumping `s -> t`,

```
X_{u,t}(X_{s,u}(x)) = X_{s,t}(x),
```

the semigroup / composition property, which is immediate because `X_{u,t}(X_{s,u}(x_s)) = X_{u,t}(x_u) = x_t = X_{s,t}(x_s)`. So before I've chosen any training scheme I have three structural identities the true map must satisfy: a Lagrangian ODE in `t`, an Eulerian PDE in `s`, and a semigroup law. Each is a *residual I could square and minimize*. That's promising — three candidate objectives, all describing the same object.

But there's a chicken-and-egg problem staring at me. Every one of those identities references `b`, the velocity, or references the map composed with itself. If I'm doing *direct* training with no teacher, where does `b` come from? I don't want to pre-train a separate velocity model — that's the teacher I'm trying to kill. Let me look harder at the Lagrangian identity, because it has the velocity `b_t` sitting in it, and ask whether the map secretly already contains its own velocity.

Take the Lagrangian identity `partial_t X_{s,t}(x) = b_t(X_{s,t}(x))` and push `s -> t`. On the diagonal the map is the identity — jumping from `t` to `t` does nothing, `X_{t,t}(x) = x` — so the right side becomes `b_t(X_{t,t}(x)) = b_t(x)`. Therefore

```
lim_{s -> t} partial_t X_{s,t}(x) = b_t(x).
```

There it is. The velocity is *not* a separate object I need to supply — it's the time-derivative of the map evaluated on its own diagonal. The map already knows its velocity; I just read it off at `s = t`. This is the hinge the whole thing turns on: a single network for `X` simultaneously encodes `b` (at `s=t`) and the integrated jump (at `s != t`). If I can train the diagonal to be a correct velocity *and* train the off-diagonal to be consistent with that diagonal, I never need an external teacher — the diagonal *is* the teacher for the off-diagonal.

Now I have to choose how to parameterize `X` so that this structure is exact and not just approximate. The naive thing is to let a network output `X_{s,t}(x)` directly. But then `X_{s,s}(x) = x` is something I'd have to *learn*, and the velocity readout `lim_{s->t} partial_t X` would be some limit I'd have to take numerically — both fragile. Let me instead bake the diagonal structure into the parameterization. I want `X_{s,s}(x) = x` to be exact and `lim_{s->t} partial_t X` to be a clean network call. Write

```
X_{s,t}(x) = x + (t - s) v_{s,t}(x),
```

where `v` is the thing the network actually outputs. This looks like a first-order Taylor step but it isn't an approximation — it's just a shift-and-rescale of `X`, so it costs me no expressivity: any `X` with `X_{s,s}=x` can be written this way with `v_{s,t}(x) = (X_{s,t}(x) - x)/(t-s)`, and `v` absorbs whatever `X` was. What do I gain? Two things, both exact. First, `X_{s,s}(x) = x + 0 = x` automatically — the boundary is free, no learning, no error. Second, differentiate: `partial_t X_{s,t}(x) = v_{s,t}(x) + (t-s) partial_t v_{s,t}(x)`, and as `s -> t` the `(t-s)` term dies, leaving `lim_{s->t} partial_t X_{s,t}(x) = v_{t,t}(x)`. Combine with the tangent fact `lim_{s->t} partial_t X = b_t`:

```
v_{t,t}(x) = b_t(x).
```

The diagonal of my network output `v_{s,t}` *is* the velocity field. So I can train `v_{t,t}` with the ordinary flow-matching loss `L_b` I already trust — no change there — and `v_{s,t}` off the diagonal is the slope of the secant line between two points on a trajectory, which the tangent identity says converges to the velocity as the two points merge. Geometrically `v_{s,t}(x)` is "the average slope to get from `x` at time `s` to where I'll be at time `t`," and the diagonal is the instantaneous slope. Beautiful — one network, two jobs, and the boundary conditions are structural rather than learned.

So the diagonal is settled: flow matching on `v_{t,t}`. The whole game is now the off-diagonal — how do I force `v_{s,t}` for `s < t` to be the genuine secant of the true flow, using only the diagonal as a teacher? I have three identities to turn into residuals. Let me also be careful and check the converse: if I impose `v_{t,t} = b_t` *and* one of the three identities, is the resulting `X` necessarily the true flow map, or could there be impostors? I need *uniqueness of the minimizer*, otherwise minimizing the residual could land me on garbage. Take the Lagrangian one. With `v_{t,t} = b_t`, the residual identity `partial_t X_{s,t}(x) = v_{t,t}(X_{s,t}(x)) = b_t(X_{s,t}(x))` is exactly the ODE `partial_t X_{s,t} = b_t(X_{s,t})` with initial condition `X_{s,s}(x) = x` (which my parameterization gives for free). Under a one-sided Lipschitz condition on `b` — `(b_t(x)-b_t(y)).(x-y) <= C|x-y|^2`, the standard Cauchy–Lipschitz hypothesis — that initial-value problem has a *unique* solution, and the true flow map is one solution, so it's the *only* one. Good: zero Lagrangian residual plus correct diagonal forces `X` to be the true map. The Eulerian case is the same kind of argument: the PDE `partial_s X + grad X . b_s = 0` says `d/ds X_{s,t}(x_s) = 0` along trajectories, so integrating from `s` to `t` gives `X_{s,t}(x_s) = X_{t,t}(x_t) = x_t`, the jump condition. The semigroup case needs a little more care — the *trivial* map `X_{s,t}(x) = x` (i.e. `v_{s,t} = 0` for `s != t`, `b_t` on the diagonal) satisfies composition vacuously, so I need continuity of `v` in `(s,t)` to rule it out: Taylor-expanding the infinitesimal semigroup `X_{s,t+h}(x) = X_{t,t+h}(X_{s,t}(x)) = X_{s,t}(x) + h b_t(X_{s,t}(x)) + o(h)` recovers the Lagrangian ODE, whose unique solution is again the true map. So all three, *given the correct diagonal and continuity*, have the true flow map as the unique minimizer. Reassuring — any of the three is a legitimate direct-training objective. I'll write each as a squared residual added to `L_b`:

```
L_LSD = int_0^1 int_0^t E | vhat_{t,t}(Xhat_{s,t}(I_s)) - partial_t Xhat_{s,t}(I_s) |^2 ds dt   (Lagrangian)
L_ESD = int_0^1 int_0^t E | partial_s Xhat_{s,t}(I_s) + grad Xhat_{s,t}(I_s) . vhat_{s,s}(I_s) |^2 ds dt   (Eulerian)
L_PSD = int_0^1 int_0^t int_s^t E | Xhat_{s,t}(I_s) - Xhat_{u,t}(Xhat_{s,u}(I_s)) |^2 du ds dt   (semigroup)
```

and the total objective is `L_sd = L_b + L_dist`. Why is the off-diagonal expectation taken over `I_s` (the interpolant at the start time `s`)? Because that's the distribution of points the map will actually act on at time `s` — I want consistency to hold where I'll evaluate it, not at arbitrary points.

Let me confirm that minimizing `L_b + L_dist` really does land on the true map and not just somewhere with small loss. Lower-bound it: `L_b(vhat) >= L_b(b)` always, since `L_b` is convex in its argument with the conditional expectation `b` as the unique global minimizer; and each `L_dist >= 0` since it's a sum of squares. So `L_sd = L_b + L_dist >= L_b(b)`. Now the *true* map, with `v_{t,t} = b` and zero off-diagonal residual (it satisfies the identity exactly), gives `L_sd = L_b(b) + 0 = L_b(b)`, hitting the lower bound. So the true map is a global minimizer, and conversely any global minimizer must have `L_b(vhat) = L_b(b)` (so the diagonal is exactly `b`) and `L_dist = 0` (so the identity holds), which by the uniqueness argument above forces it to be the true map. The minimizer is correct and unique. So mathematically the three are equivalent — same minimizer. The choice between them has to come from *optimization*, not from correctness. Which residual does an actual optimizer descend cleanly?

Now I confront the empirical facts I know about training these things. Two failure modes are well documented for direct training of jump maps. Let me look at the ESD residual: `partial_s Xhat + grad Xhat . vhat_{s,s}`. That `grad Xhat` is the *spatial Jacobian of the map* — the derivative of the network output with respect to its high-dimensional image input. To get a gradient of this loss with respect to the parameters I have to backpropagate through that spatial Jacobian (it shows up as a Jacobian-vector product through the network's spatial input). For a large image UNet this is exactly the operation that's observed to blow up — backprop through a spatial JVP of a big generative network is unstable, diverges, needs heavy engineering and tuning to tame. That's not a hunch; it's the recurring pathology behind the engineering effort that consistency-style training needs, and the same kind of spatial-Jacobian trouble noted in distillation-through-rendering work. The ESD residual structurally *requires* this object. Strike one against Eulerian.

Now the PSD residual: `Xhat_{s,t}(I_s) - Xhat_{u,t}(Xhat_{s,u}(I_s))`. To form the target I compose two learned jumps — first `s -> u`, then feed that output into `u -> t`. No spatial Jacobian appears, which is nice. But two other things go wrong, and they're the other documented failure mode. First, the inner jump `Xhat_{s,u}` is imperfect early in training, so its errors get *fed into* the outer jump `Xhat_{u,t}`, and I'm regressing the big jump onto a target built from two error-prone small jumps — errors compound. Second, the input to the outer jump is `Xhat_{s,u}(I_s)`, a *model output*, whose distribution drifts away from anything the model was trained to handle on its diagonal; the target lives on a shifting, off-distribution support. This is the compounding-error / distribution-shift difficulty inherent to bootstrapping large steps from small ones. I can even see why I *can't* get a clean Wasserstein guarantee for it the way I will for the others: the bound would have to control how composition amplifies error, and composition of learned maps has no contraction I can lean on. So PSD is stable to *optimize* (no spatial Jacobian) but pays in *sample quality*. Strike against semigroup.

So look at what's left — the Lagrangian mismatch between `vhat_{t,t}(Xhat_{s,t}(I_s))` and `partial_t Xhat_{s,t}(I_s)`. What derivatives does it contain? `partial_t Xhat` — that's a derivative with respect to the scalar *time* `t`, not with respect to the spatial input. A time-derivative is a one-dimensional JVP: differentiate the network output as I push the scalar `t`, which `jax.jvp` computes at roughly 1.5x the cost of a forward pass and, helpfully, returns `Xhat` itself as a byproduct (the JVP gives the primal and the tangent together). No spatial Jacobian `grad X` anywhere. And there's no composition of two learned jumps — the target `vhat_{t,t}(Xhat_{s,t}(I_s))` is a single network call at the transported point, using the *diagonal* velocity as the teacher. So LSD has *neither* of the two failure modes: no spatial Jacobian (unlike ESD), and no bootstrapping of small steps into large (unlike PSD). It threads the needle. That's the case for choosing the Lagrangian residual — not because it's "more correct" (all three share the minimizer) but because its residual is built from exactly the cheap, stable operations and avoids exactly the two expensive, unstable ones.

Let me make the residual concrete and count the cost, because I want to be sure I'm not fooling myself. For an off-diagonal pair `(s,t)` and a start point `I_s = alpha_s x_0 + beta_s x_1`: I do one `jvp` in `t` of the map, which gives me `Xhat_{s,t}(I_s)` and `partial_t Xhat_{s,t}(I_s)` together (~1.5 forward passes); then one ordinary network call to get the teacher velocity at the transported point, `vhat_{t,t}(Xhat_{s,t}(I_s))` — which is just `calc_b` at `(t, Xhat_{s,t}(I_s))`. That's it: about `(1 + 1.5) = 2.5` forward-pass-equivalents per off-diagonal sample. The implementation can square either sign, and I store it as `r = vhat_{t,t}(Xhat_{s,t}(I_s)) - partial_t Xhat_{s,t}(I_s)` so the target appears first. The LSD loss is `|r|^2`, summed over space. Compare ESD (same JVP cost but the unstable spatial Jacobian) and PSD (three full network calls, `3B`, plus the compounding problem). LSD is both cheap and stable. Settled: Lagrangian.

Now the teacher question, which is subtle and where stopgrad enters. In the off-diagonal mismatch `vhat_{t,t}(Xhat_{s,t}) - partial_t Xhat_{s,t}`, *both* terms depend on the same parameters. If I just let gradients flow through everything, what's the optimizer free to do? It can satisfy the residual by moving `vhat_{t,t}` toward `partial_t Xhat_{s,t}` instead of the other way around — i.e. it can corrupt the diagonal velocity (my one source of real, externally-grounded signal from `L_b`) to match a possibly-wrong off-diagonal map. That's backwards. The information should flow *from* the diagonal, where `Idot_t` gives a genuine learning signal, *to* the off-diagonal, which has no external signal of its own and must inherit it. In the original distillation setting this asymmetry is enforced for free: the teacher `b` is a frozen pre-trained model, so the off-diagonal *must* adapt to it. I'm replacing that frozen teacher by my own diagonal `vhat_{t,t}` — a self-consistent implicit teacher — so I need to *manufacture* the same "frozen teacher" effect. The tool is the stopgradient operator: wrap the teacher term so its parameters are treated as constants during backprop. So the recommended residual is

```
r = sg[ vhat_{t,t}(Xhat_{s,t}(I_s)) ] - partial_t Xhat_{s,t}(I_s),
```

where `sg[.]` stops the gradient. Now the off-diagonal `partial_t Xhat_{s,t}` is pulled toward the (frozen-for-this-step) teacher, and the teacher itself is only ever trained by `L_b` on the diagonal. Information flows the right way. Concretely this is "take the gradient of the two-parameter loss `L(theta, phi) = |vhat^phi_{t,t}(Xhat^phi) - partial_t Xhat^theta|^2` with respect to the student `theta` and then set `phi = theta`." I could also let `phi` be an EMA of `theta` — a literal slowly-updated teacher copy with decay `delta`, `phi_k = delta phi_{k-1} + (1-delta) theta_k` — but that adds lag to the very signal the off-diagonal is trying to follow. The EMA is better used for sampling, where it smooths last-iterate noise, than as the teacher inside the loss. So I'll use the instantaneous teacher `phi = theta` with `sg`, i.e. `delta = 0`.

There's a second place a stopgradient helps, and it's about that spatial-Jacobian instability sneaking back in through a side door. The teacher term is `vhat_{t,t}(Xhat_{s,t}(I_s))` — a network `vhat_{t,t}` evaluated *at* the transported point `Xhat_{s,t}(I_s)`, which is itself a network output. If I let gradients flow through the argument, I'm differentiating the teacher network with respect to its spatial input through the map — that's a spatial JVP again, the exact thing I picked LSD to avoid. So for the high-dimensional image case I also stop the gradient on the transported point before feeding it to the teacher: compute `Xst = Xhat_{s,t}(I_s)`, then `sg[Xst]`, then `vhat_{t,t}(sg[Xst])`, and stop the whole teacher output too. This is the "convex"-style configuration — it makes the teacher a genuine frozen evaluation with no backprop through a spatial Jacobian, matching the original frozen-teacher distillation exactly. (For low-dimensional toy problems with small nets the instability isn't a concern and full gradients are fine, but for images this stopgrad placement is what makes it train.) I should be honest with myself that the convergence behavior of any `sg`-laden objective is hard to characterize a priori — `sg` breaks the "this is the gradient of a single scalar loss" story — so this is an empirical recommendation grounded in the distillation analogy, not a theorem. But it's the configuration that makes large-scale training stable.

Two more practical issues I can already see will bite, both about variance across the time pairs `(s,t)`. First, the residual's *scale* varies wildly across `(s,t)`: a tiny jump `t ~ s` and a huge jump `t >> s` produce loss values, and gradient norms, that differ by orders of magnitude, and that heterogeneity injects variance into every minibatch update and forces me to a tiny learning rate. I want each `(s,t)` to contribute a *scale-normalized* gradient. The clean tool is the uncertainty-style adaptive weight: introduce a small learned scalar function `w_{s,t}` and replace the bare loss `L^{s,t}` by

```
e^{-w_{s,t}} L^{s,t} + w_{s,t}.
```

Why this form? Minimize over `w_{s,t}` holding `L^{s,t}` fixed: `d/dw [e^{-w} L + w] = -e^{-w} L + 1 = 0`, so `w* = log L`, and at that optimum the term is `1 + log L` and — this is the point — the gradient that reaches the *model* parameters is `e^{-w} ∇L = e^{-log L} ∇L = ∇L / L`. Each `(s,t)`'s contribution is divided by its own current loss magnitude, so all time pairs push the model on a common scale regardless of whether they're tiny jumps or huge ones. `w_{s,t}` is, literally, a running estimate of the log-variance of the loss at `(s,t)`, learned jointly with the model. This is the EDM2 adaptive weight generalized from one time to the two-time `(s,t)` setting; the diagonal uses `w_{t,t}`. So the per-sample off-diagonal loss is `e^{-w_{s,t}} |r|^2 + w_{s,t}`, and the per-sample diagonal loss is `e^{-w_{t,t}} |vhat_{t,t}(I_t) - Idot_t|^2 + w_{t,t}`.

Second, *which* `(s,t)` do I sample, and how much budget on each? The full objective is an integral over the diagonal, where `L_b` supplies the real interpolant target, plus the upper triangle `s < t`, where the distillation residual supplies consistency for finite jumps. The limiting identities say the diagonal object must become the velocity, but the self-distillation residual itself is degenerate exactly on the diagonal: with `X_{s,t} = x + (t-s)v_{s,t}`, `partial_t X_{t,t}(x) = v_{t,t}(x)`, so the squared mismatch at `s=t` is zero whether or not `v_{t,t}` equals the true `b_t`. That means the diagonal flow-matching term is not optional; it is the only place the external target `Idot_t` enters. So I draw a mixture: a fraction `eta` of the batch placed uniformly on the diagonal (trained by `L_b`), and `1-eta` placed uniformly on the upper triangle (trained by the LSD residual). What's `eta`? The diagonal term is cheap — one network evaluation. The off-diagonal term is expensive — the JVP plus the teacher call, `~2.5x`. And the diagonal is learning the flow itself, the thing the off-diagonal distills, so it deserves the majority of the budget. I'll put `eta = 0.75`: three-quarters of the batch learns the flow, one-quarter distills it into the map. `eta` doubles as a compute knob — more diagonal means cheaper steps.

Let me also pin down the architecture-level pieces so the implementation is grounded and not floating. The network outputs `v_{s,t}(x)` with EDM2 preconditioning: `c_in = 1/sigma_data`, `c_out = sigma_data`, so `v_{s,t}(x) = c_out * UNet(c_in * x, s, t-s)` — the UNet sees a unit-scale input and the output is scaled back. I embed `s` and the *gap* `dt = t - s` rather than `s` and `t` directly (the map's behavior is more naturally a function of how far it jumps), add the two embeddings, and FiLM-condition the UNet on them. I use deterministic positional embeddings for time, matching the EDM2-style module and keeping the time encoding fixed. The interpolant is linear, `alpha_t = 1 - t`, `beta_t = t`, so `Idot_t = x_1 - x_0` (constant), with a Gaussian base whose variance is matched to the data. Gradient clipping at 1.0, RAdam, warmup then square-root learning-rate decay, EMA of parameters for sampling — standard. And I train from random init with no pre-training, because the whole point is to avoid the teacher; the diagonal `L_b` *is* the pre-training, happening simultaneously.

Before I write the code, let me make sure driving `L_sd` down actually buys me a good *sampler*, not just a small number — I want a guarantee that the loss value controls the generated distribution. Suppose `L_b(vhat) + L_LSD(vhat) <= eps`, so each piece is `<= eps`. First, what does `L_b <= eps` buy for the diagonal velocity? Expand:

```
L_b(vhat) = int_0^1 E |vhat_{t,t}(I_t) - Idot_t|^2 dt.
```

Add and subtract `b_t(I_t)` and use that `b_t = E[Idot_t | I_t]` is the conditional mean, so the cross term vanishes by the tower property:

```
L_b(vhat) = int_0^1 E |vhat_{t,t}(I_t) - b_t(I_t)|^2 dt  +  int_0^1 E[ |Idot_t|^2 - |b_t(I_t)|^2 ] dt.
```

The second integral is, again by the tower property, `int E[ |Idot_t - b_t(I_t)|^2 | I_t ] dt`, a *conditional variance* — it's `>= 0` and independent of `vhat`. So

```
int_0^1 E |vhat_{t,t}(I_t) - b_t(I_t)|^2 dt  <=  L_b(vhat)  <=  eps.
```

My learned diagonal velocity is within `eps` (in mean-square, along the path) of the true `b`. Now stitch two transport estimates. Consider the flow generated by my *diagonal* velocity, `xdot = vhat_{t,t}(x)`, with law `rhohat^v_1` at `t=1`. A standard Grönwall-type bound for flow matching (Albergo–Vanden-Eijnden) turns the velocity `L^2` error into a Wasserstein bound: `W_2^2(rho_1, rhohat^v_1) <= e^{1 + 2 Lhat} eps`, where `Lhat` is the spatial Lipschitz constant of `vhat_{t,t}`. That controls the gap between the *true* data and what my *implicit velocity* would produce if I integrated it. But I'm not going to integrate it — I'm going to apply the *map* `Xhat_{0,1}` in one shot, with law `rhohat_1`. The LSD residual being `<= eps` is exactly the statement that my map is close to integrating its own implicit velocity, and the flow-map-matching Lagrangian guarantee gives `W_2^2(rhohat^v_1, rhohat_1) <= e^{1 + 2 Lhat} eps`. Now triangle-inequality the two and use `(a+b)^2 <= 2a^2 + 2b^2`:

```
W_2^2(rho_1, rhohat_1) <= 2( W_2^2(rho_1, rhohat^v_1) + W_2^2(rhohat^v_1, rhohat_1) )
                       <= 2( e^{1+2Lhat} eps + e^{1+2Lhat} eps ) = 4 e^{1 + 2 Lhat} eps.
```

So the one-step generated distribution is within `4 e^{1+2Lhat} eps` of the data in squared 2-Wasserstein — the sample quality improves systematically as I drive the loss down. (The same machinery gives a bound for ESD too, `2e(1 + e^{2Lhat}) eps`; the point is that LSD and ESD *both* have such a guarantee while the composition-based PSD does not, which is the theoretical shadow of PSD's compounding-error problem.) That's the validation I wanted: the residual I chose isn't just zero-at-the-truth, its *value* certifies the sampler.

Now let me write it as the actual training term, filling the off-diagonal slot. The diagonal term is settled flow matching with the adaptive weight; the off-diagonal term is the LSD residual. I'll write them in the JAX/Flax idiom the harness uses, where the flow-map module `X.apply(...)` exposes `method="calc_b"` for the diagonal velocity `v_{t,t}`, `method="partial_t"` for the simultaneous `(Xhat_{s,t}, partial_t Xhat_{s,t})` via `jvp` in `t`, and `method="calc_weight"` for `w_{s,t}`.

```python
import jax
import jax.numpy as jnp


def diagonal_term(params, x0, x1, label, t, rng, *, interp, X):
    """Flow matching on the diagonal s = t: train v_{t,t} = b_t against Idot."""
    It = interp.calc_It(t, x0, x1)            # I_t = alpha_t x0 + beta_t x1
    It_dot = interp.calc_It_dot(t, x0, x1)    # Idot_t = alpha_dot_t x0 + beta_dot_t x1
    # v_{t,t}(I_t) is the diagonal velocity readout = calc_b at (t, I_t)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")    # w_{t,t}
    # uncertainty weighting: e^{-w} L + w  =>  model gets a scale-normalized gradient
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def lsd_term(params, teacher_params, x0, x1, label, s, t, rng, *, interp, X, stopgrad_type):
    """Off-diagonal Lagrangian mismatch: v_{t,t}(X_{s,t}(I_s)) - partial_t X_{s,t}(I_s)."""
    Is = interp.calc_It(s, x0, x1)            # start point at time s

    # one jvp in t gives BOTH the map and its time-derivative (the residual's first term)
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )

    if stopgrad_type == "convex":
        # frozen-teacher emulation: no gradient through the transported point (no spatial
        # Jacobian) and no gradient through the teacher velocity -> info flows diag -> offdiag
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval = jax.lax.stop_gradient(
            X.apply(teacher_params, t, Xst_Is, label, train=False, method="calc_b", rngs=rng)
        )
    elif stopgrad_type == "none":
        # full gradient (used only for low-dimensional / small-net problems)
        b_eval = X.apply(params, t, Xst_Is, label, train=False, method="calc_b", rngs=rng)
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    weight_st = X.apply(params, s, t, method="calc_weight")     # w_{s,t}
    error = b_eval - dt_Xst                   # residual r = v_{t,t}(X_{s,t}) - partial_t X_{s,t}
    lsd_loss = jnp.sum(error ** 2)
    return jnp.exp(-weight_st) * lsd_loss + weight_st          # e^{-w} |r|^2 + w
```

The full objective `setup_loss` then vmaps these over the batch and splits it: a fraction `eta = 0.75` of samples go through `diagonal_term` (flow matching on `s=t`), the rest through `lsd_term` on uniformly-drawn upper-triangle pairs `(s,t)`, and the two contributions are summed, weighted by their batch counts, and averaged — a single combined gradient updates both `v` and the weight `w`.

The pain is sampling cost: integrating the velocity needs many network calls. So I want the *integrated* two-time jump map `X_{s,t}` instead of the velocity. Differentiating the map's own definition `X_{s,t}(x_s)=x_t` in `t`, in `s`, and composing gives three structural identities (Lagrangian ODE, Eulerian PDE, semigroup) the true map must satisfy. Pushing the Lagrangian one to the diagonal reveals the tangent condition `lim_{s->t} partial_t X = b_t` — the velocity is the map's diagonal time-derivative, so one network can be both. The parameterization `X_{s,t}(x) = x + (t-s) v_{s,t}(x)` makes the boundary `X_{s,s}=x` exact and turns the tangent condition into `v_{t,t}=b`, so the diagonal trains by ordinary flow matching `L_b`. Each of the three identities, squared and added to `L_b`, has the true map as its unique minimizer (uniqueness from one-sided-Lipschitz ODE theory / continuity). They're correctness-equivalent, so I choose by optimization: the Eulerian residual carries a spatial Jacobian that destabilizes large image nets, the semigroup residual composes learned jumps and compounds error with distribution shift, but the *Lagrangian* mismatch `v_{t,t}(X_{s,t}) - partial_t X_{s,t}` has neither — only a cheap time-JVP and a single teacher call. A stopgradient on the teacher (and, for images, on the transported point) emulates a frozen teacher so signal flows diagonal -> off-diagonal and no spatial Jacobian leaks back. The EDM2-style adaptive weight `e^{-w}L+w` normalizes the gradient scale across the heterogeneous time pairs, and the `eta=0.75` diagonal/triangle split puts the budget on the cheap flow-matching signal that the off-diagonal distills. Finally, a triangle-inequality stitch of two transport bounds certifies that `W_2^2(rhohat_1, rho_1) <= 4 e^{1+2Lhat}(L_b + L_LSD)`, so driving the loss down provably improves the one-step sampler — direct training of the jump map, distillation-grade stability, no teacher.
