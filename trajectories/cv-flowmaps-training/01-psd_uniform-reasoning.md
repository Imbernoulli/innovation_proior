The harness hands me a flow-map module whose diagonal already trains itself for free, and four empty
loss-term bodies whose default fill is deliberately inert. So the real question on this first rung is
narrow and concrete: with the diagonal pinned, what is the *cheapest, most obviously stable* way to
supervise the off-diagonal `v_{s,t}` for `s != t`, where there is no external target? I want to begin
with the residual that asks the least of the optimizer, because if even that does not train on the
large-UNet scales the benchmark uses, nothing more delicate will.

Let me re-derive what the scaffold gives me so I know exactly what I am building on. The map is
parameterized as `X_{s,t}(x) = x + (t-s) v_{s,t}(x)`, and this is not a Taylor truncation — it is an
exact shift-and-rescale of whatever `X` is, so `v_{s,t}` absorbs all the nonlinearity and I lose no
expressivity. Two facts fall out of it immediately. First, `X_{s,s}(x) = x` holds for any `v`, so the
identity-on-the-diagonal boundary is structural, never learned. Second, differentiate in `t` and send
`s -> t`: `partial_t X_{s,t}(x) = v_{s,t}(x) + (t-s) partial_t v_{s,t}(x)`, whose limit is
`v_{t,t}(x)`. But the map's own defining relation `X_{s,t}(x_s) = x_t`, differentiated along a
trajectory, gives `lim_{s->t} partial_t X_{s,t}(x) = b_t(x)`, the velocity. So the diagonal of the
network output *is* the drift, `v_{t,t} = b_t`, and the harness's `calc_b` reads exactly that. The
diagonal therefore trains by ordinary flow matching against `Idot_t` — the one term in the whole
objective that has a genuine external target — and `diagonal_term` is settled. Geometrically
`v_{s,t}(x)` is the slope of the chord between `x_s` and `x_t` on one trajectory, and the diagonal says
the chord between two infinitesimally close points is the instantaneous velocity. The entire open
problem is to determine the chord for far-apart points, using only the diagonal as a teacher.

What certifies a chord as correct away from the diagonal? The true map obeys three structural
identities, each a derivative of `X_{s,t}(x_s) = x_t`. Differentiating in `t` gives the Lagrangian ODE
`partial_t X_{s,t}(x) = b_t(X_{s,t}(x))`. Differentiating totally in `s` (the argument `x_s` moves too)
gives the Eulerian PDE `partial_s X_{s,t}(x) + grad X_{s,t}(x) . b_s(x) = 0`. And no derivative at all
is needed for the third: jumping `s -> u` then `u -> t` along the same trajectory lands where jumping
`s -> t` lands, so `X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)` for any intermediate `u`. Each is zero exactly
when the map is correct, so each squared residual is a candidate off-diagonal loss; and since each
references the velocity `b`, which is just my own diagonal `v_{t,t}`, the model can be its own teacher
with no pre-trained model at all. The harness exposes a slot for each — `lsd_term` for the Lagrangian,
`esd_term` for the Eulerian, `psd_term` for the semigroup. The choice between them, for this first
rung, has to come from optimization, because they share the same minimizer.

Look at what each residual *costs to differentiate*, because at the image scale this benchmark runs,
that is the difference between "trains" and "diverges." The Eulerian residual carries `grad X_{s,t}` —
the spatial Jacobian of the network. A parameter gradient through that backpropagates a Jacobian-vector
product through a large image UNet, which is exactly the operation the consistency-trajectory line has
needed heavy engineering to keep from blowing up; the harness's `esd_term` slot even has to call
`jax.jvp` against the *spatial* input to form it. I do not want to open on the residual most likely to
diverge. The Lagrangian residual is friendlier — it needs `partial_t X` (a time-JVP, which the module's
`partial_t` gives cheaply, returning `X_{s,t}` for free) and one teacher-velocity call at the
transported point — but it still evaluates the teacher *at* a model output and reasons through a
time-derivative. The semigroup residual is the cheapest and the most obviously stable thing to ask an
optimizer to descend: it needs *no derivative of the map at all*, just three function evaluations — the
big jump `X_{s,t}`, and the two small jumps `X_{s,u}` and `X_{u,t}` — and asks them to agree. No
`grad X`, no `partial_t` through a teacher, only function values. That is where I start: the semigroup
residual, filling `psd_term`. It also has a clear pedigree — progressive distillation trained a student
to do in one step what a teacher does in two, then halved repeatedly; this is the same composition idea
made into a single direct objective with the model as its own teacher.

The raw objective is the squared composition residual on `X`, averaged over the upper triangle
`0 <= s <= t <= 1` and over the intermediate `u`. Its minimizer is right: the true map zeroes the
integrand by the semigroup property, and conversely, if `v` is continuous and the residual is zero for
all `u`, a Taylor expansion of the semigroup in a small step recovers the Lagrangian ODE, whose
well-posed solution is the flow map. (Continuity is load-bearing — without it the trivial map
`X_{s,t} = x` satisfies composition vacuously.) So the objective is exact. But let me actually look at
the *scale* of that residual before I trust it, because I have been burned by losses whose magnitude
depends on the sampled index. Substitute the parameterization. The big jump is
`X_{s,t}(x) = x + (t-s) v_{s,t}(x)`; the composed jump is
`x + (u-s) v_{s,u}(x) + (t-u) v_{u,t}(X_{s,u}(x))`. Subtract: the constant `x` cancels, and every
surviving term carries a time-gap prefactor `(t-s)`, `(u-s)`, or `(t-u)`. So the `X`-space residual
scales like `(t-s)`, and the squared loss like `(t-s)^2`. That is a real problem: a pair `(s,t)` close
together produces a whisper-quiet gradient, a far-apart pair a loud one, and the optimizer sees an
effective learning rate that silently depends on how big a jump I happened to sample — pure injected
variance, with the far jumps yanking the network around and the near jumps ignored.

I do not want to reweight by `1/(t-s)^2` (it divides by something that vanishes on the diagonal and
only papers over the coupling). Better to kill the `(t-s)` at the source. The residual on `X` is
entirely a statement about the `v`'s once `x` cancels, so solve the semigroup directly for `v_{s,t}`:
`v_{s,t}(x) = ((u-s)/(t-s)) v_{s,u}(x) + ((t-u)/(t-s)) v_{u,t}(X_{s,u}(x))`. The slope of the big jump
is a *convex combination* of the two sub-slopes, weighted by the fraction of the interval each covers —
of course: the average velocity over `[s,t]` is the length-weighted average of the average velocities
over `[s,u]` and `[u,t]`. The coefficients are now ratios of gaps, `O(1)`, not `O(t-s)`. Regressing
`v_{s,t}` onto this combination gives a residual that is a difference of velocities, scale-free in
`t-s`: the `(t-s)^2` is gone. This is exactly why the harness's `psd_term` reads off the implicit slope
`phi = v_{s,t}` via `return_X_and_phi=True` rather than differencing the mapped points — the slope form
is the preconditioned form. Parameterize the split as `u = gamma s + (1-gamma) t`; then
`(u-s)/(t-s) = 1-gamma` and `(t-u)/(t-s) = gamma`, so the target collapses to
`(1-gamma) v_{s,u}(x) + gamma v_{u,t}(X_{s,u}(x))`. (Sanity check: `gamma=0` puts `u=t`, the second
segment has zero length and its coefficient `gamma=0` kills it; `gamma=1` puts `u=s` and the first
segment's coefficient `1-gamma=0` kills it. The weights track segment lengths.) In the harness this
split fraction arrives as `h` and the intermediate time as `u`, so the uniform teacher is exactly
`(1-h) phi_su + h phi_ut`.

Now the choice that names this rung: how to pick `gamma`. The correctness proof holds for any proposal
with full support on `[0,1]`. If I always split at the midpoint I only ever teach "one full jump = two
equal half-jumps" — a clean, low-variance lesson, but the map never sees in training that an `[s,t]`
jump should also equal a 10/90 or a 90/10 composition. If instead I draw `gamma ~ U([0,1])` uniformly,
every batch supervises the map at a different, randomly chosen split, so over training the map is forced
to be self-consistent under *all* ways of breaking a jump into two. That is the **uniform** variant, and
it is the natural thing to try first: it is the most faithful sampler of the full-support theorem, with
no special split singled out. The cost is variance — the teacher target is now a function of `gamma`,
so as `gamma` rattles the target rattles too, on top of the noise already present. I will lean on the
weight head to absorb that, and I am open to the cost being real: if the extra target variance hurts,
fixing the split is the obvious next move. So for this rung: `psd_type = "uniform"`, `h ~ U([0,1])`,
`u = h s + (1-h) t`.

Two more pieces the bare scaffold default leaves out, and both matter. First, the direction of
information flow. The off-diagonal loss is nonconvex in `v` and symmetric-looking — it would be just as
happy to drag the diagonal-derived slopes toward a wrong `v_{s,t}` as to pull `v_{s,t}` toward the
correct composition. But only the diagonal `v_{t,t}` has an external teacher (the data velocity). The
off-diagonal has no ground truth of its own and can only be correct by inheriting from the diagonal. So
the gradient must push `v_{s,t}` toward the composed target and *not* push the target back. The clean
enforcement is a stopgradient on the composed teacher: treat the combination as frozen. This makes the
self-teacher behave like a pre-trained distillation teacher even though it is the same network, and it
quietly removes the worst of the bootstrap instability — with the target stopgradded I never
backpropagate through the inner `X_{s,u}` and its implicit spatial Jacobian, so the gradient is a plain
residual on the single forward evaluation `v_{s,t}(I_s)`. That is the `convex` placement: the student
`phi_st` is computed from `params` and carries the gradient, while both teacher slopes `phi_su`,
`phi_ut` are evaluated and stop-gradiented. (The bare scaffold hard-wires a stopgradient but does it
unconditionally with no `stopgrad_type` branch and no real teacher-parameter routing; the baseline
restores the proper convex routing through `teacher_params`, leaving a `none` mode that runs pure
self-consistency through `params` for the small-scale regime.) I keep the instantaneous teacher
(`phi = theta`, decay zero) rather than a slow EMA copy, because an EMA adds a lag hyperparameter and
separates the target from the current diagonal signal; the EMA is better spent on sampling.

Second, the loss-scale heterogeneity that survives the slope preconditioning. Even with `(t-s)^2`
divided out, different regions of the time-square are intrinsically harder and noisier, and if I just
average raw squared residuals the loud regions hijack the gradient. The fix is uncertainty weighting,
generalized to two times: attach the learned head `w_{s,t}` (the harness's `calc_weight`) and replace
the bare loss by `e^{-w_{s,t}} L + w_{s,t}`. Minimizing over `w` alone for positive `L`:
`d/dw[e^{-w} L + w] = -e^{-w} L + 1 = 0` gives `w* = log L`, value `1 + log L`, and the gradient that
reaches the model is `e^{-w} grad L = grad L / L` — each `(s,t)` divided by its own current loss
magnitude, so all time pairs push the model on a common scale. `w_{s,t}` is literally a running estimate
of the per-pair log-variance, and this is exactly what should absorb the extra variance the uniform
`gamma` injects, letting me run a usable learning rate. I wrap *both* the diagonal and the off-diagonal
term in this form (the diagonal with `w_{t,t}`). The bare scaffold default omits the weight entirely —
returning raw `jnp.sum(...)` — which is precisely the thing this baseline restores.

Finally, the batch split. The diagonal term costs one network evaluation and carries the only external
truth; the off-diagonal composition costs three (the big jump and two small jumps). And as `s -> t` the
composition residual degenerates into the flow-matching residual, so the off-diagonal term is the smooth
extension of the diagonal into the interior of the square, not a separate gadget. So I want most of the
batch on the cheap, information-rich diagonal: `eta = 0.75` on the diagonal (times `t ~ U([0,1])`,
`s=t`), `1-eta = 0.25` in the upper triangle (draw two uniforms, take min and max), with `u` set by the
sampled split. The full module — `diagonal_term` plus `psd_term` in its `uniform`/`convex` form, both
uncertainty-weighted — is in the answer.

So this rung lands the semigroup residual in `psd_term`, in the uniform variant: the diagonal learns the
real velocity by flow matching, the off-diagonal distills it into finite jumps via the convex-combination
slope identity with a *uniformly sampled* split, stopgradded one-way, uncertainty-weighted, three
quarters of the batch on the diagonal. It is the most faithful sampler of the full-support consistency
theorem and the off-diagonal residual least likely to diverge (no spatial Jacobian, no map derivative),
so it is the right floor to measure first. What I expect, and what I will test against the other variants
to come: the uniform split should train stably and give a real but unremarkable FID, because its
strength — supervising every possible split — is also its weakness, a noisier teacher than a single
fixed split would give. If the swept-split variance is the binding cost, a deterministic midpoint split
should sharpen the target and lower FID; and if the Lagrangian residual's cleaner derivative-based
target beats the composition's bootstrap, LSD should beat both. The number this rung posts is the bar
those two have to clear.
