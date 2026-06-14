The uniform split came back at FID 14.99 on train_small, and the number reads exactly like the cost I
flagged when I chose it. The semigroup residual trained — no divergence, which was the whole reason to
open on the residual with no spatial Jacobian and no map derivative — so the floor is real, not a
crash. But 14.99 is the worst FID I expect to see on this ladder, and the reason is the one I named: the
uniform variant draws the split `gamma ~ U([0,1])` fresh every step, so for a fixed pair `(s,t)` the
teacher target `(1-gamma) v_{s,u} + gamma v_{u,t}(X_{s,u})` is itself a random variable, rattling with
`gamma`. The student `v_{s,t}(I_s)` is being regressed onto a moving target, and that variance sits on
top of the bootstrap noise I already cannot remove. The uncertainty weight `e^{-w}L + w` absorbed enough
of it to keep training stable, but absorbing variance is not the same as removing it: a noisier target
means a blurrier fixed point, and a blurrier flow map means a higher FID. So the diagnosis is precise —
the uniform variant's defining strength, supervising the map under *every* possible split of a jump, is
also what is costing it, because each batch only gets a noisy single-sample estimate of "consistency
under all splits," and the noise propagates straight into the learned map. The fix is not to change the
residual — the semigroup residual is sound and stable — but to change *how I sample the split*.

Let me re-derive the off-diagonal target so I am precise about where the variance enters and what I can
do about it. The map is `X_{s,t}(x) = x + (t-s) v_{s,t}(x)`, the diagonal `v_{t,t} = b_t` is pinned by
flow matching, and the off-diagonal target comes from the semigroup identity
`X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)`. Expanding under the parameterization and cancelling the constant `x`
gives the slope relation `(t-s) v_{s,t} = (u-s) v_{s,u} + (t-u) v_{u,t}(X_{s,u})`, and dividing by
`(t-s)` — the step that killed the `(t-s)^2` learning-rate coupling on the last rung — leaves a convex
combination, `v_{s,t} = ((u-s)/(t-s)) v_{s,u} + ((t-u)/(t-s)) v_{u,t}(X_{s,u})`. With
`u = gamma s + (1-gamma) t` the coefficients become `1-gamma` and `gamma`, so the target is
`(1-gamma) v_{s,u} + gamma v_{u,t}(X_{s,u})`. This is exact for the true map at *every* `gamma` — that
is the full-support theorem the uniform variant leaned on. But "exact for every `gamma`" is a statement
about the converged optimum, not about each gradient step. During training, what the optimizer actually
sees per step is one Monte-Carlo sample of `gamma`, and a single sample of a target that depends on
`gamma` is a high-variance estimate. The uniform variant pays for full coverage with per-step noise. I
saw the price: 14.99.

So the trade is laid bare. The full-support proof requires only that the proposal over `gamma` has full
support; it says nothing about needing to *sample* `gamma` randomly. What if I collapse the proposal to a
single deterministic value? I give up the clean full-support uniqueness theorem — a deterministic split
no longer pins the map by itself — but I keep something nearly as strong: the slope relation is still an
*exact* identity of the true map at that one split, so the true map still has zero residual there, and
paired with the diagonal flow-matching term (which carries the real external signal and is unchanged),
the combined objective still drives toward the correct map. What I gain is a *sharp, fixed* teacher: for
each `(s,t)` the target no longer fluctuates, the regression has one well-defined fixed point per pair,
and the only remaining noise is the data and the bootstrap — exactly the variance the uncertainty weight
was designed for. That is the practical stability trade I am now willing to make, because the uniform
variant's number tells me its extra target variance is the binding cost, not the residual itself.

Which single `gamma`? If I have to pick one decomposition of `[s,t]` into two sub-jumps, the natural
choice is the **midpoint**, `gamma = 1/2`, so `u = (s+t)/2` and the teacher collapses to the even
average `0.5 (v_{s,u} + v_{u,t}(X_{s,u}))`. The case for the midpoint specifically, and not say
`gamma = 1/4`, is symmetry and balance. With `gamma = 1/2` the two sub-jumps `[s,u]` and `[u,t]` have
equal length, so neither leg dominates the composition and the teacher weights are an even `1/2, 1/2`. A
lopsided split would make one leg a near-full jump — almost as hard to get right as the full jump I am
trying to teach, which defeats the purpose of decomposing — and the other a near-trivial jump carrying
almost no information. The midpoint puts the auxiliary node at the geometric center, the place that
maximizes how much *easier* each sub-jump is than the whole while keeping the two legs balanced, so the
inner jump `X_{s,u}` is the most accurate it can be for a given total gap, which directly limits the
bootstrap error that the inner jump feeds into the outer jump. There is also a structural reason the
midpoint is the right deterministic choice: it is exactly the recursive structure of the dyadic-grid
bootstrap schemes — two equal half-steps composing into one full step — except here it falls out
continuously for arbitrary `(s,t)` rather than being tied to a fixed grid. So the choice is principled,
not arbitrary: of all deterministic splits, the midpoint is the one that minimizes the inner-jump
difficulty and keeps the composition symmetric.

Now let me be honest about what the midpoint does *not* fix, because the uniform result already showed me
the residual's irreducible weakness. The target `v_{u,t}(X_{s,u}(I_s))` evaluates the second sub-slope
at the *output* of the first sub-jump — a network evaluated at another network's output, which is the
bootstrap. Early in training the inner jump `X_{s,u}` is wrong, so I regress onto a target built from a
composition of two wrong sub-jumps, and the inner error feeds the outer call on a shifting input support.
The midpoint reduces this by making the inner jump the easiest balanced sub-jump, but it does not
eliminate it — bootstrapping is baked into "use yourself as your own teacher," and it is the structural
reason I will *not* get a clean Wasserstein guarantee for this scheme the way a derivative-based residual
would give one. That caveat is the same for uniform and midpoint; what midpoint changes is only the
target variance. So I should expect the midpoint to beat the uniform's 14.99 by removing the swept-split
noise, but I should *not* expect it to suddenly acquire a loss-to-sample-error bound it never had.

The rest of the machinery carries over unchanged from the uniform rung, and for the same reasons, so I
will keep it and only swap the split. The information-flow asymmetry is identical: only the diagonal has
an external teacher, so the off-diagonal must inherit, so the composed teacher is stopgradded
(`convex`) — the student slope `v_{s,t}(I_s)` from `params` carries the gradient, both sub-slopes from
`teacher_params` are stop-gradiented, which also avoids backprop through the inner jump's spatial
Jacobian. (The `none` mode that runs the teacher through `params` with no stopgrad stays available for
the small-scale regime but is not what these image baselines use.) The teacher is instantaneous,
`phi = theta`, decay zero — an EMA would lag the very diagonal signal the off-diagonal is chasing, and is
better spent on sampling. The uncertainty weight `e^{-w_{s,t}} L + w_{s,t}` stays on both terms: its
finite optimum for positive `L` is `w* = log L`, so the model receives the scale-normalized gradient
`grad L / L`, equalizing the heterogeneous residual magnitudes across the time-square. Here it has *less*
variance to absorb than on the uniform rung, since the deterministic split removed the `gamma` noise — so
if anything I expect the weight head to settle more cleanly and the effective learning rate to be a touch
more forgiving. And the batch split is unchanged: `eta = 0.75` on the diagonal (the cheap,
externally-grounded flow-matching term, one network evaluation, and the smooth `s -> t` limit of the
off-diagonal term), `0.25` in the upper triangle (`s,t` from two ordered uniforms), with `u` now set to
the midpoint `(s+t)/2` instead of a sampled `h s + (1-h) t`. In the harness this is the single switch
`psd_type = "midpoint"`, which replaces `(1-h) phi_su + h phi_ut` by `0.5 (phi_su + phi_ut)` and ignores
the sampled `h`; the sampler sets `u` to the midpoint. Everything else in `psd_term` — the student/teacher
routing, the stopgrad, the weight — is the same code. The full module is in the answer.

Let me also sanity-check what this specializes to, because seeing it sit inside known recipes tells me I
have the right object and not a fourth isolated trick. Restrict `(s,t)` to a dyadic grid and fix the
intermediate to the grid midpoint, and the off-diagonal term becomes "one full grid step matches two half
grid steps composed," recursively over the grid — exactly the discrete bootstrap recipe of shortcut
models. So that prior recipe is the grid-restricted special case of this midpoint consistency loss, and
the present scheme is its continuous-time generalization for arbitrary `(s,t)`. The frozen-teacher
progressive-distillation idea — one student step reproducing two teacher steps on a halving schedule — is
the same composition relation but with an external pre-trained teacher instead of the model's own current
parameters. Both fall out as special cases, which is the reassurance that the midpoint is a principled
point in the design space, not a guess.

So the delta from the uniform rung is one line of intent and one switch of code: keep the semigroup slope
residual, keep the convex stopgrad, keep the uncertainty weight, keep the `0.75/0.25` split, but fix the
intermediate node at the midpoint instead of sampling it. The falsifiable expectation against the
measured uniform result: removing the swept-split target variance should *lower* the train_small FID
below 14.99 — I expect a clear improvement, into the low double digits, because the regression now chases
a sharp fixed target instead of a rattling one. I do *not* expect it to reach the best possible number on
this ladder, because the bootstrap weakness survives the change: the target is still a composition of the
model's own jumps, with no clean accuracy bound, so the midpoint should land *better than uniform but
short of a derivative-based residual*. If the midpoint comes in below 14.99 but still above a single-call
derivative target, that is the signal that the next move is to abandon composition entirely for the
Lagrangian residual — trading three evaluations and a bootstrap for one teacher call and a clean
time-JVP target. The midpoint's number is the bar that the Lagrangian rung will have to clear.
