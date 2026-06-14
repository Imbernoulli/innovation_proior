The midpoint split came in at FID 10.79 on train_small, a clear step down from the uniform variant's
14.99, and the size of that drop confirms the diagnosis exactly: most of the uniform variant's deficit
was the swept-split target variance, and fixing the split to the midpoint removed it. So the
semigroup residual itself is sound — both PSD variants trained stably, which is why I opened the ladder
there. But 10.79 is where the midpoint *stalls*, and the reason it stalls is the weakness I flagged when
I built it and could not remove: the bootstrap. The midpoint target is still
`0.5 (v_{s,u}(I_s) + v_{u,t}(X_{s,u}(I_s)))`, and the second term evaluates a sub-slope at the *output*
of the first sub-jump — a network at another network's output. Early in training the inner jump
`X_{s,u}` is wrong, so the outer call sees an off-distribution input and the inner error propagates into
the target; even at convergence the target is a composition of the model's own jumps, with no clean
loss-to-accuracy bound. The midpoint minimized the inner-jump difficulty (equal balanced legs) but it
could not make the composition stop being a composition. That is what 10.79 is: the best the
self-composed target can do once the split variance is gone. To go lower I have to attack the bootstrap
itself — get an off-diagonal target that is *not* built by composing two learned jumps.

Let me go back to the three structural identities and ask which one gives a target with no composition.
The map obeys the Lagrangian ODE `partial_t X_{s,t}(x) = b_t(X_{s,t}(x))`, the Eulerian PDE
`partial_s X_{s,t}(x) + grad X_{s,t}(x) . b_s(x) = 0`, and the semigroup
`X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)`. PSD used the semigroup, and the semigroup is the *only* one of the
three whose residual composes two learned jumps — that is precisely the source of the bootstrap I am now
stuck behind. The other two characterizations express the off-diagonal map against a *derivative* and a
*single* velocity evaluation, with no second learned jump fed into a first. So the move off 10.79 is to
abandon composition entirely and switch to a derivative-based residual. The harness gives me both: the
Eulerian residual in `esd_term`, the Lagrangian residual in `lsd_term`. The minimizers are identical to
PSD's — all three share the true map — so again the choice is pure optimization, and the optimization
behavior is exactly what 10.79 is telling me to weigh.

Look at the Eulerian residual first, because it is the most direct cure for the bootstrap and I want to
rule it in or out cleanly. Its residual carries `grad X_{s,t}` — the spatial Jacobian of the map. To
take a parameter gradient I have to backpropagate through a Jacobian-vector product against the
*spatial* input of a large image UNet; the harness's `esd_term` even forms it by calling `jax.jvp`
against `Is`. That is the operation the consistency-trajectory line has needed heavy engineering to keep
from diverging, and on the EDM2-scale UNets these benchmarks run, it is the textbook way to make
training blow up. So ESD trades PSD's bootstrap for an instability that is *worse*, not better — I would
be swapping a residual that trains-but-stalls for one that may not train at all on the large preset.
That is not the trade I want. The Eulerian residual fixes the bootstrap but reintroduces exactly the
spatial-Jacobian fragility I avoided by opening on PSD in the first place. Strike it.

That leaves the Lagrangian residual, and it is the one that has *neither* of the two failure modes I
have now seen or reasoned through. Push the Lagrangian identity to the diagonal: with
`X_{s,t}(x) = x + (t-s) v_{s,t}(x)`, `lim_{s->t} partial_t X_{s,t}(x) = v_{t,t}(x)`, and the map's
defining relation gives `lim_{s->t} partial_t X_{s,t}(x) = b_t(x)`, so `v_{t,t} = b_t` — the same
diagonal-is-velocity fact the diagonal term already trains. The Lagrangian identity off the diagonal is
`partial_t X_{s,t}(x) = b_t(X_{s,t}(x))`, i.e. the mismatch
`v_{t,t}(X_{s,t}(I_s)) - partial_t X_{s,t}(I_s)`. What derivatives does that contain? `partial_t X` is a
derivative with respect to the *scalar time* `t`, not the spatial input — a one-dimensional JVP, which
the module's `partial_t` computes at roughly 1.5x a forward pass and, helpfully, returns `X_{s,t}` as a
byproduct (the JVP gives primal and tangent together). No `grad X`, so no spatial Jacobian, so none of
ESD's instability. And the target `v_{t,t}(X_{s,t}(I_s))` is a *single* network call at the transported
point, using the diagonal velocity as the teacher — no composition of two learned jumps, so none of
PSD's bootstrap. The Lagrangian residual threads the needle exactly between the two failures the ladder
has exhibited: it is as stable to differentiate as PSD (no spatial Jacobian) and as free of compounding
error as ESD would be (no self-composed target). That is the case for `lsd_term`, and it is a case made
*by* the two measured results below it, not in the abstract: PSD trains but stalls at 10.79 because of
the bootstrap; ESD would cure the bootstrap but destabilize on the Jacobian; LSD has neither problem.

Let me make the residual concrete and confirm the minimizer is right, because abandoning composition
should not cost me correctness. The off-diagonal LSD loss is the squared mismatch
`|v_{t,t}(X_{s,t}(I_s)) - partial_t X_{s,t}(I_s)|^2`, added to the unchanged diagonal flow-matching
term. With `v_{t,t} = b_t` from the diagonal, the residual is exactly the statement
`partial_t X_{s,t} = b_t(X_{s,t})` with initial condition `X_{s,s}(x) = x` (free from the
parameterization), which under a one-sided Lipschitz condition on `b` is an initial-value problem with a
*unique* solution — the true flow map. So zero Lagrangian residual plus the correct diagonal forces the
true map, the same uniqueness PSD had, but now without the bootstrap that denied PSD a clean accuracy
bound. In fact the Lagrangian residual *does* admit the bound PSD lacked: if the diagonal flow-matching
loss and the LSD residual are each below `eps`, a Grönwall-type flow-matching bound turns the diagonal
velocity error into a Wasserstein bound on the implicit-velocity flow, the flow-map-matching Lagrangian
guarantee bounds the gap between that flow and the one-step map, and a triangle inequality stitches them
into `W_2^2(rhohat_1, rho_1) <= 4 e^{1+2Lhat} eps`. So driving the LSD loss down provably improves the
one-step sampler — exactly the certificate the composition-based midpoint could not offer, and the
theoretical shadow of why I expect LSD to edge below 10.79.

Now the teacher placement, which is where I have to be careful so the residual does not corrupt the one
source of real signal. Both terms of the mismatch `v_{t,t}(X_{s,t}) - partial_t X_{s,t}` depend on the
same parameters, and if I let gradients flow through everything the optimizer can satisfy the residual by
moving the diagonal velocity `v_{t,t}` toward the off-diagonal map instead of the other way around —
backwards, since only the diagonal has the external `Idot` signal. So I stopgradient the teacher: wrap
`v_{t,t}(X_{s,t}(I_s))` so its parameters are constants during backprop, making the diagonal a frozen
teacher exactly as in distillation, with information flowing diagonal -> off-diagonal. That is the same
one-way-teacher logic PSD used, and for the same reason. There is a second stopgradient that matters
specifically for images, and it is about the spatial Jacobian sneaking back through a side door: the
teacher is `v_{t,t}` evaluated *at* the transported point `X_{s,t}(I_s)`, itself a network output, so if
I let gradients flow through that argument I differentiate the teacher with respect to its spatial input
— a spatial JVP, the exact thing I picked LSD to avoid. So for the image case I also stop the gradient on
the transported point before the teacher call. This is the `convex` configuration in the harness: the
`partial_t` call gives `(X_{s,t}, partial_t X_{s,t})` from `params` (the off-diagonal carries the
gradient through `partial_t X`), then the transported point is stop-gradiented and fed to a
`calc_b` evaluated under `teacher_params` and stop-gradiented again — a genuine frozen teacher with no
spatial-Jacobian backprop. (A `none` mode runs the teacher through `params` with full gradient for the
low-dimensional regime, but the image baselines use `convex`.) The teacher is instantaneous,
`phi = theta`, decay zero, with the EMA reserved for sampling — same choice as PSD, same reasoning: an
EMA would lag the diagonal signal the off-diagonal is chasing.

The remaining machinery carries over unchanged, and I keep it for the reasons the ladder has already
validated. The diagonal term is the same uncertainty-weighted flow matching against `Idot_t`. The
uncertainty weight `e^{-w_{s,t}} L + w_{s,t}` stays on both terms — its finite optimum `w* = log L` for
positive `L` hands the model the scale-normalized gradient `grad L / L`, equalizing the heterogeneous
residual magnitudes across the time-square; here it weights the LSD residual instead of the PSD residual,
but the form and the head (`calc_weight`) are identical. The batch split is unchanged, `eta = 0.75` on
the diagonal and `0.25` on upper-triangle pairs, with the cost accounting if anything *better* than PSD's:
the LSD off-diagonal sample costs one `partial_t` JVP (~1.5 forward passes, returning `X` for free) plus
one teacher `calc_b` call, about 2.5 forward-equivalents, versus PSD's three full evaluations for the big
jump and two small jumps. So switching `psd_term` for `lsd_term` is also slightly cheaper per off-diagonal
sample, which under the fixed 50000-step budget is a small free lift. In the harness this is the switch
from `psd_term`/`psd_type` to `lsd_term`/`stopgrad_type="convex"`, selected by
`FLOWMAPS_BENCH_SLURM_ID=0`; the diagonal term, the weight head, and the batch split are byte-for-byte
the same code. The full module is in the answer.

So the delta from the midpoint rung is a change of *which* identity supervises the off-diagonal: drop the
semigroup composition (the bootstrap that capped PSD at 10.79) and the Eulerian residual (the spatial
Jacobian that would destabilize ESD), and take the Lagrangian residual, whose target is a single
frozen-teacher velocity at the transported point with only a cheap time-JVP — neither failure mode. Keep
the convex stopgrad, the uncertainty weight, the `0.75/0.25` split. The falsifiable expectation against
the measured midpoint result: removing the bootstrap should let the off-diagonal target be a clean,
single-call regression with a real loss-to-Wasserstein certificate, so LSD should come in *below* 10.79
on train_small — I expect a modest but real improvement, into the low ten / high nine range, not a
landslide, because the midpoint already removed the larger split-variance deficit and what is left to
recover is the smaller bootstrap penalty. If LSD lands at or below 10.79 it confirms that the
composition, not the residual family, was the binding constraint, and that the derivative-based Lagrangian
target is the strongest fill of this edit surface. If it somehow does not beat the midpoint, that would
say the bootstrap penalty was already negligible at 50000 steps and the time-JVP target carries its own
noise — but the certificate and the cheaper per-sample cost both point the other way. The midpoint's
10.79 is the bar; the Lagrangian residual is the move to clear it.
