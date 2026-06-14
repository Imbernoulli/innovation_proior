The geometric ladder told me exactly what I suspected, and it told me in numbers. Log-linear landed at
FID 5.634 on edges2handbags, 13.748 on ImageNet inpainting, 17.048 on DIODE — geometric-mean 10.97, by
a wide margin the worst placement I have tried. Read the shape: DIODE, where the budget is even tighter
(NFE = 3, so `n = 2` and the geometric ratio is `γ = (1.0/1e-4)^{1/2} = 100`), is the catastrophe at
17.0, and ImageNet inpainting at 13.7 is the second worst. Those are exactly the two workloads where the
high-`t` opening step is most violent: with γ = 100 the DIODE schedule is `{1.0, 0.01, 1e-4}`, so a
*single* deterministic denoiser call has to carry the state across `1.0 → 0.01` — 99% of the entire
interval — and ImageNet's I2SB bridge carries the most high-`t` conditioning structure (a hard
inpainting mask) for that one coarse jump to mangle. The diagnosis writes itself: this is not a near-data
problem, it is the lone oversized high-`t` step I flagged. The geometric spacing packed three or four of
its few stops against `t_min` and trusted one enormous jump to traverse the entire high-`t` half, and on
a frozen DBIM marcher that does not re-equilibrate, that jump's integration error is injected at the top
and carried all the way down. The overlap-equalizing property that made geometric optimal in annealed
sampling bought me nothing, because this sampler has no warm-start handoff to protect — it just punished
me for leaving one interval far too large. So the sampler's actual currency is the opposite of what
geometric optimizes: it is not "equalize overlap," it is "do not leave any single step oversized across a
region where the trajectory bends."

That is a precise lesson, and it points straight at the most disciplined response: stop betting on where
the curvature lives at all, and instead place the nodes so that *no* interval is oversized. That is the
uniform grid — constant spacing in `t`. Let me derive why this is the right move from the loglinear
failure, rather than just retreating to the scaffold default by reflex.

Start from what the deterministic DBIM update actually is. With η = 0 the sampler marches a fixed update
rule along `t` from `t_max` to `t_min`, computing its step coefficients from the bridge's own
`get_abc`/`get_alpha_rho` at the current and next scheduled times — this is, structurally, Euler-style
integration of the bridge's probability-flow trajectory, with the schedule playing the role of the
integrator's node placement. The local error of a single step on an interval of length `Δt_i` scales
like `(Δt_i)^2` times the local trajectory curvature `C_i`, so the total accumulated error is roughly
`Σ_i (Δt_i)^2 C_i`. This is the right object: it is quadratic in the step length, which is exactly why
loglinear's one giant `Δt` near `t_max` dominated — a single `(Δt)^2` term with a huge `Δt` swamps
everything the tightly-packed low-`t` steps could save.

Now, *if I knew the curvature* `C_i`, the optimal placement would minimize `Σ_i (Δt_i)^2 C_i` subject to
`Σ_i Δt_i = t_max − t_min`. The Lagrange condition is `2 C_i Δt_i = λ`, i.e. `Δt_i ∝ 1/C_i` — shorten
the steps where the trajectory is hard. Every warped schedule is exactly such a bet: the EDM power-law,
log-linear, and cosine each bake in a fixed assumption about *where* curvature lives plus a shape knob.
Log-linear bet "curvature is concentrated at low `t`, so make the steps shrink multiplicatively toward
`t_min`," and the bet was wrong in the way that matters most — it implied a single huge step at high `t`
where, on these conditioned bridges, there is real curvature to resolve. So the honest question is: do I
actually know where the curvature lives well enough to place another bet? Across three different bridges
— a VP edges bridge, an I2SB inpainting bridge, a VP depth bridge — with the requirement that I carry no
per-dataset constants, the answer is no. I have one failed bet that says "low-`t`-heavy is wrong here,"
which is information about what *not* to do, but not enough to confidently place the opposite bet.

With no reliable curvature information, the problem is no longer "minimize `Σ (Δt_i)^2 C_i` for a known
`C`"; it is **minimax**. I have to choose the grid before I see the curvature, against an adversary who
can put a fixed curvature budget `Σ_i C_i = B` wherever it hurts most. Take any non-uniform grid. It has
some largest interval `Δt_max > (t_max − t_min)/n`. The adversary dumps the entire curvature budget into
that interval, and the error becomes `≈ (Δt_max)^2 B` — driven by the *square* of the biggest step. The
only way to deny the adversary an oversized interval to exploit is to make every interval the same size:
all `Δt_i = (t_max − t_min)/n`. The uniform grid is the unique placement with no largest interval to
attack; it minimizes the worst case. And this is exactly the diagnosis of the loglinear failure stated
as a design principle: loglinear *was* the non-uniform grid with one enormous `Δt_max`, and the bridge's
high-`t` curvature *was* the adversary that dumped error into it. The uniform grid removes the very
vulnerability that produced the 17.0 on DIODE and the 13.7 on ImageNet.

I should pause on one objection, because the warped schedules can claim to be uniform too. The EDM
power-law is uniform in `t^{1/ρ}`; log-linear is uniform in `log t`; cosine is uniform in a cosine
coordinate. Spacing evenly in a transformed coordinate is just a warp of the grid in `t`, and it
presupposes the trajectory is smoothest in *that* coordinate — which is a curvature assumption. The
frozen DBIM update is written in `t`: it indexes the bridge's `get_abc`/`get_alpha_rho` coefficients by
`t` and queries `D(x_t, t)` at `t`, so `t` is the integrator's native variable. Constant `Δt`
presupposes nothing about the warp. Given that loglinear's particular warp just blew up, the
warp-agnostic stance — even spacing in the sampler's own `t` — is the principled retreat, not a lazy one.

There is a second, independent reason uniform is the right rung here, and it is about the constraint that
the schedule carry no dataset-tuned constants. Uniform spacing has **no shape hyperparameter** at all —
only the count `n`, which the budget fixes. Every warped alternative has a shape knob (ρ for the
power-law, the implicit base for log-linear, the period for cosine) that, to be set well, wants to be
tuned to a workload's true curvature — exactly the per-dataset tuning the task forbids and exactly the
move that, done blindly, produced loglinear's collapse. A hyperparameter-free placement transfers across
the three workloads unchanged by construction. So uniform is simultaneously the minimax-optimal grid
under no curvature knowledge and the only placement that needs no cross-workload tuning. For a rung whose
job is to establish a trustworthy floor after a bet went badly wrong, that is precisely the right object.

The fill is the simplest one on the ladder, and it is also the scaffold default — which is worth being
honest about: this rung is not a new gadget, it is the disciplined baseline that the failed warp should
have been measured against in the first place. Constant spacing in `t` from `t_max` to `t_min` over
`n + 1` nodes is `t_i = t_max + (i/n)(t_min − t_max)`. `torch.linspace(t_max, t_min, n + 1)` realizes it
directly and — unlike the geometric fill — needs *no* terminal correction, because `linspace` hits both
endpoints exactly: `t_0 = t_max`, `t_n = t_min` to machine precision, with a constant negative step in
between. Length `n + 1`, strictly decreasing, terminal exactly `t_min`, on `device` — every contract
clause holds with nothing to patch. That endpoint-exactness is itself a small but real advantage over
loglinear, where I had to pin the terminal node by hand to undo a `log`/`exp` round-trip error. The full
scaffold module is in the answer.

So the delta from the loglinear rung is a reversal of the bet: where loglinear crammed the budget against
`t_min` and left one ruinous step at the top, uniform spreads the budget so that *every* step is the same
size — `Δt = (t_max − t_min)/n` everywhere — denying any single interval the chance to be the oversized
one the bridge's high-`t` curvature exploited. Here is what I expect against loglinear's measured
numbers. The catastrophic workloads should recover most: DIODE at NFE = 3, where loglinear's γ = 100
schedule put 99% of the interval into one step, should improve dramatically once that step is broken into
two equal halves; ImageNet inpainting, the second-worst at 13.7, should also drop sharply because its
high-`t` mask structure no longer sits under one giant jump. edges2handbags, where loglinear was already
the least bad (5.634), should improve more modestly — its VP bridge curves less violently at high `t`, so
it suffered least from the oversized step and has less to gain from removing it. I do *not* expect uniform
to be the best schedule attainable: a placement that correctly matched the bridge's real curvature would
beat it, and the minimax argument concedes exactly that — uniform is the disciplined "no bet," the floor
that any genuinely curvature-aware schedule must clear. If uniform clears loglinear comfortably across
all three workloads while ending exactly at `t_min` with no terminal patch, that confirms the lesson the
geometric collapse taught me — the few-step bridge sampler rewards *evenness*, not overlap — and it gives
me a clean, hyperparameter-free baseline from which to ask the next, sharper question: where, if anywhere,
does this bridge curve enough that I should deliberately break the evenness and spend resolution there.
