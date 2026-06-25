The global-batch loss landed where I expected — imbalance in the same good band as the Switch loss,
cross-entropy no worse — and the previous rung ended by naming exactly what it leaves undone. The
smooth `f·P` penalty equalizes the *average* usage, but I want to see whether it actually has any
grip on the tail before I trust that complaint. The term is `N·Σ_i f_i·P_i`. At perfect balance
every `f_i = P_i = 1/N`, so it equals `N · N · (1/N)·(1/N) = 1`; at full collapse onto one expert it
is `N·1·1 = N`. So it lives in `[1, N]` and is minimized at uniform — good, it is a real balancing
term. But consider an expert at a tenth of its fair share against one at nine-tenths: both enter the
sum only through their own small `f_i·P_i`, and the cold one's product is the *smaller* of the two.
The gradient that would resurrect a nearly-dead expert is therefore weakest precisely in the tail
where I need it most. Balancing the mean is not the same as resurrecting the dying. I want a term
that singles out the under-utilized experts and pushes specifically on them — and I have to be
careful, because the obvious ways to do that are the ways that wreck specialization.

Let me think about what a targeted under-use penalty should and should not do. It should fire on an
expert that has fallen below some floor of usage, and leave alone every expert at or above its fair
share. That shape is a one-sided penalty — a hinge: `max(0, floor − f_i)`, zero above the floor and
linear below it. An expert comfortably above the floor contributes nothing; one below it contributes
in proportion to how far below. That is the opposite behavior to the smooth `f·P` term, which
spreads its attention evenly: the hinge ignores the healthy experts entirely and concentrates all
its force on the cold tail. The floor should be a small fraction of the uniform share `1/N` — I do
not want to demand that every expert hit `1/N` exactly, only that none wither below a minimum. A few
percent of `1/N` feels right. With `N=8`, `1/N = 0.125`, so a floor of, say, `0.064/N = 0.008` is
6.4% of the uniform share — an expert has to drop below one-fifteenth of its fair usage before the
hinge even notices it. That is a genuine emergency threshold, not a nudge toward perfection.

A hinge by itself is dangerous, though, and I have to see the danger before I trust it. If it fires
hard whenever an expert is below the floor, it will fire even when the router is already healthy and
the under-use is just benign variation — a momentary dip, the natural roughness of a finite batch.
Pushing hard there would do exactly what the micro-batch loss did: flatten legitimate structure,
raise the cross-entropy. So the hinge needs a gate that asks *is this real collapse, or normal
variation?* There is a clean signal for that already in the router: the entropy of its distribution
over experts. When the router is peaked — low entropy, most mass on a few experts — that is the
collapse regime and a cold expert really is being starved; the rescue should be strong. When the
router is near-uniform — high entropy — the system is healthy, any momentary under-use is noise, and
the hinge should barely fire. So I want a weight that is large when entropy is low and small when it
is high: a complement of the normalized entropy.

Let me pin the weight down and check its endpoints by hand, because the whole point of the gate is
its behavior at the two limits. Normalize the router entropy by its maximum `log N` so it runs in
`[0,1]`; take `1 − H/log N` so peaked routers score near one and uniform routers near zero; and
offset by a half so the weight never quite vanishes — `s = 0.5 + (1 − H/log N)`. At a uniform router
`H = log N`, so `s = 0.5 + (1 − 1) = 0.5`. At a router fully collapsed onto one expert `H = 0`, so
`s = 0.5 + (1 − 0) = 1.5`. Between them it is monotone in peakedness — if the mass sits evenly on
half the experts, `H = log 4` and `s = 0.5 + (1 − log4/log8) = 0.83`; on two experts, `H = log 2`
and `s = 1.17`. So the gate triples the rescue strength from a healthy router (`0.5`) to a fully
collapsed one (`1.5`), and never drops the floor enforcement to exactly zero. That is the modulation
I wanted: mostly idle, surging only as the router peaks.

Now the implementation point that decides whether any of this is a real training signal. The hinge
is naturally written on the count `f_i`, and `f_i` comes from a `bincount` of the top-K selections —
it is non-differentiable, its gradient is identically zero, just like the bare count penalty of the
first balancing rung. The count can only *select* which experts are under the floor; the gradient
that actually moves usage has to flow through the differentiable router probability `P_i`. The
obvious patch is to keep the `f`-based test for membership but apply the penalty to `P` of the
selected experts: `Σ_i under_i · max(0, τ − P_i)`, where `under_i = [τ − f_i > 0]` is detached. Let
me trace this on a concrete collapsed layer to make sure it does what I think.

Take `N=8`, `τ=0.008`, and a heavily skewed routing of 100 tokens (top-2) so that experts 6 and 7
each get only 0.5% of the slots — both below the floor, `under = {6,7}`. Now I need the router
probabilities `P`. Suppose the router is peaked but the two cold experts still carry *moderate*
probability mass — say `P_6 ≈ 0.13`, `P_7 ≈ 0.20` — even though almost no tokens were finally routed
to them. Then `max(0, τ − P_i) = max(0, 0.008 − 0.13) = 0` for expert 6 and `0` for expert 7. The
whole hinge is zero, the gradient is zero — *even though both experts are flagged as under the
floor*. That stopped me. I had been telling myself the hinge "pushes on the under-used experts," but
the trace says membership in `under` is necessary and not sufficient: the term only produces a
gradient when a flagged expert *also* has its router probability below `τ`. An expert that is starved
in *realized usage* but still holds real probability mass gets nothing from this term.

Let me re-run with an expert that is starved in both senses. Put expert 7's logit far below the rest
so `P_7 ≈ 0.0001`, keep it under the usage floor. Now `max(0, τ − P_7) = 0.008 − 0.0001 = 0.0079`,
the hinge is nonzero, and `s` for this peaked router comes out around `1.4`. Backpropagating, the
gradient on expert 7's logit is negative (about `−8·10⁻⁵`) and the gradients on the high-mass
experts are small and positive — the term raises the probability mass on the doubly-starved expert
and skims a little off the dominant ones. So the term *is* a real rescue, but a sharper one than my
prose suggested: it targets experts the router has nearly stopped proposing at all (`P` near zero),
not merely experts that happened to lose the top-K lottery this batch. On reflection that is the
more conservative behavior, and the better one — an expert with healthy `P` but low realized `f` is
not dying, it is just losing close contests, and leaving it alone is correct. The `f`-gate keeps the
term from chasing every transient dip; the `P`-clamp keeps it from acting on experts that are not
actually being abandoned. I would not have predicted the second gate from the prose, but having seen
it I think it is load-bearing, not incidental.

I should be candid about where the exact constants come from, because I did not derive them. The
half-offset in `s`, the floor coefficient `0.064`, the `0.1` weight on the hinge relative to the
global term — these are not first-principles numbers. They are the output of an *evolutionary search*
over the loss function: ShinkaEvolve evolved the Python of the balancing loss, scored by the same
fitness I am using, `r = −(L_CE + L_imb)`, on real MoE pretraining, and converged to this form. What
I can do here is exactly what I just did — check that each piece plays the role the mechanism needs.
The endpoint check confirmed the entropy gate spans `0.5` to `1.5`; the floor arithmetic confirmed
`τ` is a genuine emergency threshold at 6.4% of fair share; the gradient trace confirmed the hinge
becomes a real, signed training signal through `P`, and revealed the second `P<τ` condition I had
glossed. The `0.1` coefficient I cannot independently justify beyond noting its job: keep the rescue
from overwhelming the global `[1,N]`-scale term — and a tenth of a term that is itself bounded by
`τ` per expert is plainly subordinate to it, which is at least consistent.

So the rung is: keep the global-batch `f·P` term unchanged, averaged over layers with the `N_E`
scale; add a `0.1`-weighted, entropy-modulated hinge that fires only on experts below the usage
floor *and* below the probability floor, also averaged over layers, with its gradient routed through
those experts' probabilities. I expect this to hold the cross-entropy where the global-batch loss
had it — the hinge is idle when the router is healthy, by the `s=0.5` floor and the `P<τ` gate both
— and to push the imbalance lower than the global term managed, because the cold tail is now actively
rescued rather than merely averaged over. I would want to confirm that expectation on the actual
runs rather than assert it; the mechanism checks out piece by piece, but whether the joint
cross-entropy/imbalance point is genuinely better than the global term alone is a measurement, not a
derivation. This is the loss the program-evolution search settled on against this exact fitness, at
scale on a 556M-active-82M MoE over two-billion-plus FineWeb tokens; my run is a small reproduction
of its mechanism, not its scale.
