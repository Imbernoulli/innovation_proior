The hierarchical lifts kept paying, and each rung confirmed the same recipe: upscale the optimized profile
for free, kick to break the block plateau, refine with annealed-soft-max search while keeping the best true
overlap. `0.381240` at `24` cells, then `0.381076` at `120` — a modest drop from lifting, and the falsifiable
prediction held: adding pieces lowered the bound, so the cap was resolution, not a missed basin. The
per-rung gain also collapsed, from `0.118` in the coarse rung to about `1.6×10⁻⁴` in the middle one, a shrink
of roughly seven hundred-fold — the signature of a search closing on a hard limit. The published frontier
lives at several hundred cells (AlphaEvolve's `0.380924`, the AutoEvolver record `0.38086945`), so the
natural next move is to lift once more, to the several-hundred-cell scale the records use, and
grind there with a longer, sharper refinement plus an exact minimax polish. That band is exactly where
careful gradient refinement should approach the step-function frontier, so it is the target for this endpoint
rung: push toward and, if I can, through `0.3810` toward `~0.3809`, while being honest that the absolute
record was bought with orders of magnitude more compute than I have here.

I take the optimized `120`-cell profile and upscale `×5` to `600`. As before the upscale is free — same step
function, same overlap, same `C`, verified by the same peak-scaling argument — and as before the upscaled
point is a degenerate plateau of repeated blocks (now `5`-wide, so `120·4 = 480` flat directions), which I
kick to break the block symmetry and give the optimizer traction. Then I have to confront the scaling wall I
have been deferring since the coarse rung's aside, because it changes the whole shape of the endpoint.

The wall is the solver. The coarse and middle rungs leaned on constrained SLSQP against the soft-max
surrogate, and that was the right tool at two dozen and a hundred-odd cells. At `600` cells it becomes the
bottleneck, and I can put numbers on why. SLSQP finite-differences the surrogate, costing `O(n)` objective
evaluations per gradient, each an `O(n²)` cross-correlation, so a single gradient is `O(n³)` — about `2×10⁸`
operations at `n = 600` against `7×10⁵` for one correlation — and on top of that its internal quadratic
subproblem is itself super-linear in the number of variables. In wall-clock terms one annealed SLSQP ladder
at `600` variables runs a couple of minutes, and worse, from a good starting point it barely moves, because
the surrogate optimum it chases has essentially coincided with where the lifted profile already sits. So
leaning on SLSQP at this resolution buys almost nothing for a large cost. I have to switch the large-`n`
optimizer to something that scales.

Before I commit to that switch, I want to walk the alternatives and reject them on arithmetic rather than
taste. Option one is to keep SLSQP and just spend more: at a couple of minutes per annealed ladder, a
basin-hopping budget like the middle rung's `20` hops would run to the better part of an hour, and I already
argued the solver barely moves from a good start, so I would pay hours to hold roughly the same value — a
bad trade. Option two is to abandon the lift and do a fresh multi-start directly at `600` cells. But rung 2's
random-profile measurement is decisive here: random balanced starts average around `0.6`, worse than the flat
floor, and the fraction `p` of starts flowing into the good basin shrinks sharply with `n` — at `600`
variables `p` is effectively negligible, so a fresh multi-start would need an astronomical number of restarts
to match what the free upscale hands me for nothing. The lift is the *only* affordable way to reach record
resolution with the good structure intact. Option three is to lift even further, to `1200` cells or more, on
the theory that resolution is still the cap. But the records themselves live at several hundred cells, and the
per-rung gain has already collapsed seven-hundred-fold; if the `600`-cell result turns out to be a basin
floor rather than a resolution cap, finer grids would only draw the same trapped envelope more smoothly at
higher cost. So `600` is the right resolution — matching the records, where I can test whether the frontier
is a basin or a grid — and the lift-plus-scalable-optimizer is forced, not chosen.

The scalable replacement is a projected-gradient method on the smooth soft-max bound with an *analytic*
gradient, run as `β`-annealed Adam. The gradient is cheap because it collapses into correlations. Writing the
soft-max weights `w_k = exp(β(c_k − m)) / Σ_j exp(β(c_j − m))`, the surrogate's gradient is `∂C̃/∂v_a =
(2/n) Σ_k w_k · ∂c_k/∂v_a`, and the per-shift gradient is `∂c_k/∂v_a = (1 − v_{a−k}) − v_{a+k}` — the first
term from `v_a` appearing as the `v_i` factor, the second from `v_a` appearing as the complemented factor. I
checked that formula against finite differences and it is exact. Summed against the weights, the two pieces
are themselves cross-correlations of the weight vector `w` with `1 − v` and with `v`, so the whole analytic
gradient is one or two `O(n²)` correlations — no `O(n)` blow-up from finite differencing. Each Adam step is
then one correlation to form `c`, a softmax to form `w`, and a couple of correlations for the gradient: I
measured this at about `2×10⁻⁴` seconds per step at `n = 600`, so tens of thousands of steps cost seconds
rather than minutes. That is the substitution the analogous step-function frontier used, and it is what makes
a long grind at record resolution affordable at all.

Why Adam specifically, rather than plain projected gradient descent on the same analytic gradient? The
soft-max minimax is badly conditioned: the cells sitting under the binding shifts see large gradient
components while the cells in the quiet regions barely move, so a single global step size is either too big
for the active cells or too small for the rest. Adam's per-coordinate adaptive scaling absorbs that
conditioning — each height gets a step tuned to its own gradient history — which matters when I want tens of
thousands of steps to keep making progress rather than stalling on the stiff directions. After each Adam step
I project back onto the box-and-sum constraint set with the same clip-and-redistribute operator as the
earlier rungs, so the iterate never drifts off feasibility; the projection is `O(n)` per step and does not
change the cost story. The kicks during the grind are injected periodically rather than after full solves, so
the run alternates long stretches of Adam refinement with small feasibility-preserving perturbations — mild
restarts that keep the iterate off shallow saddles while the annealing `β` and the decaying effective step
push the late phase toward pure refinement.

The exact subgradient polish is complementary to the Adam grind, not a repetition of it. Adam descends the
*soft* surrogate, whose weighted gradient smears attention across the near-tied band; the polish descends the
*true* max by taking a hard active set — shifts within `10⁻⁹` of the maximum — and stepping down their
averaged exact gradient. Because the true minimax objective is non-smooth at the tied plateau, a fixed step
would chatter across the kinks, so the polish uses a decaying learning rate `lr₀/(1 + it/500)` over `2000`
iterations, the standard schedule that makes a subgradient method actually converge on a non-smooth
objective: large early steps to reach the plateau, shrinking steps to settle onto it without oscillating.
Averaging the exact gradients over the whole active set is the crucial choice — it seeks the direction that
lowers every binding shift at once, which is the only kind of move that can lower the maximum when many
shifts are tied, and it is also the move that reveals a local optimum when no such direction is feasible,
because then the averaged step projects to near zero.

The tolerance that defines "active" is itself a judgement call, and it has to be tight. I take shifts within
`10⁻⁹` of the true maximum as active. If I set it much looser — say `10⁻⁴` — I would sweep in shifts that are
not actually binding, dilute the averaged gradient with directions that do not need lowering, and step in a
compromise direction that fails to press the true worst shifts down; the polish would stall above the real
optimum. If I set it far tighter than floating-point noise, the active set would fragment as round-off jitters
which shifts count as the exact maximum, and the averaged direction would flicker between iterations. A
tolerance at the `10⁻⁹` level, scaled by the magnitude of the maximum, sits in the band where it captures the
genuinely tied ridge and nothing else — and, tellingly, the width of the active set at that tolerance is a
direct readout of how flat the envelope has become. If I see hundreds of shifts inside `10⁻⁹` of the max at
`600` cells, that is not a numerical artifact; it is the broad tied plateau the conservation argument
predicts, and the same object whose size decides, through the dimension count, whether any descent direction
survives at all.

Three things change at `600` cells, and each is load-bearing. First, the `β` schedule. Naively the soft-max
gap is bounded by `(2/n)·log(2n−1)/β`, which actually *shrinks* with `n` because `2/n` falls faster than
`log(2n−1)` grows — so on the count of shifts alone I would not need a sharper `β`. But that worst-case bound
is only tight when the near-worst band is wide, and here it is: rung 3 predicted, and I expect to confirm,
that the active set of near-tied shifts *grows* as the envelope flattens. When `A` shifts sit essentially at
the maximum, the soft-max reads `m + (1/β)log A` rather than `m`, so the effective gap is `(2/n)·log(A)/β`,
and with `A` now in the hundreds `log A` climbs toward its ceiling `log(2n−1)`. So the reason to sharpen `β`
at `600` cells is not the shift count but the flat envelope: I anneal `β` into the thousands, where the gap
falls to the `10⁻⁶` scale (`β ≈ 3000` gives about `8×10⁻⁶`), comfortably below the ten-thousandths I am
fighting over, so the surrogate does not average the true max down toward its own near-tied plateau. Second,
periodic kicks *during* the long grind, not just at the upscale: over many passes Adam can settle into a
shallow basin, and a small shrinking kick between passes acts like a mild restart that keeps it exploring
while the late phase stays pure refinement. Third, the exact subgradient polish at the end. The soft-max and
the true `max` diverge slightly, so I finish by descending the genuine minimax directly: at each iteration I
find the active set of shifts within a tiny tolerance of the maximum, form the exact gradient of each active
`c_k` using the same `(1 − v_{a−k}) − v_{a+k}` formula, average those gradients over the active set, take a
decaying step, and re-project to feasibility. Distributing the step across *all* active shifts is what lets
the polish push the binding constraints down together rather than trading one for another — a Chebyshev-style
descent on the tied plateau. Throughout I keep the best *true* overlap ever seen, because the surrogate-best
and true-best are not the same vector.

Now I have to be honest about what this is likely to reach, and the conservation picture makes me cautious.
Rung 3 warned that as resolution rises the active set widens, and a very large active set is precisely the
condition under which local descent has nowhere to go. Here is the mechanism sharpened: at the near-binary
spiky profile the total overlap `n²/4` is conserved, and once hundreds of shifts are tied at the worst level,
lowering the maximum requires lowering *all* of them at once — but conservation says the mass I remove from
the active shifts has to reappear somewhere, and it reappears by pushing the next-highest inactive shifts up
until they join the active set. If the active set is large enough that no feasible direction (respecting the
box and the sum) lowers every active shift simultaneously, the descent cone is empty and I am at a local
optimum: the subgradient step projects to nearly zero, sharper `β` refines inside the same basin, and a fresh
`600`-cell start would only find a different profile with the same trapped structure.

I can make the "large enough" precise with a dimension count, and it is sobering. To strictly lower the
maximum I need a feasible direction `d` with `∇c_k · d < 0` for every active shift `k`, together with `d · 1
= 0` from the sum constraint and non-negativity components wherever a height is pinned at a box corner. That
is a system of `A` homogeneous strict inequalities inside the tangent space of the feasible polytope, whose
dimension is at most `n − 1` after the sum equality (and less once pinned cells remove their coordinates).
When `A` is small relative to `n − 1` such a direction generically exists — there is room to push all active
shifts down together. But as `A` grows toward `n`, the `A` gradient vectors `∇c_k` start to positively span
the whole tangent space, and once they do, no single direction can have negative inner product with all of
them — the descent cone collapses to nothing. At `600` cells with a near-binary profile, rung 3's prediction
that the active set widens with resolution means `A` may well reach into the hundreds, the same order as
`n`, which is exactly the regime where this collapse happens. So the conservation law (fixed total, so
lowering active shifts refills others) and the dimension count (too many binding constraints for the
available directions) point the same way: the `600`-cell profile is plausibly pinned at a genuine local
optimum, and no amount of sharper `β` or harder polishing inside that basin will move it. So I genuinely do not
know whether the sharp-`β` grind and the polish will push below `0.3810764` or merely hold it. The
reorganizing passes should hold the `120`-cell value exactly, since upscaling is free and the lift only adds
freedom; the question is whether the extra freedom at `600` cells opens a descent direction or just lets the
same envelope be drawn more smoothly. I expect the latter is likely — that this profile is at or very near a
robust local optimum whose floor a single bounded constructor's bias selects — but I will not assert it; the
evaluator on the returned `600` heights is what settles it, and I report whatever number it returns, not a
hoped-for one.

There is a quantitative check that sharpens the "different basin, not worse flattening" reading, and I can
do it against the conservation floor. At `n = 600` the averaging floor is `C ≥ 600/(2·1199) = 0.25021`, so a
value of `0.3810764` corresponds to a peak-to-mean overlap ratio of `0.3810764/0.25021 ≈ 1.523`. The
published record sits at a ratio of about `0.38087/0.2502 ≈ 1.522` — the averaging floor is `≈ 0.2502` at
several-hundred-cell resolution, essentially the same denominator as mine. Those two ratios are essentially
identical: my envelope, if it holds at `0.38108`, is *just as flat* as the record's — I am not losing because
my worst overlap towers any higher above the mean than theirs does. The record is lower purely because it is
a *different* flat configuration, one whose common plateau level sits a couple ten-thousandths beneath mine.
That is precisely the fingerprint of two distinct local optima of comparable quality rather than a
resolution deficit: same flatness, different floor. It tells me the remaining gap is not something I can
flatten my way into from where I stand.

What I can predict with more confidence is the surrounding geometry. If the grind holds near `0.38108`, that
sits about `1.9×10⁻⁴` above the AutoEvolver record `0.38086945` and `1.6×10⁻⁴` below the AlphaEvolve and
Haugland landmarks near `0.38092`, and about `2.1×10⁻³` above White's provable floor `0.379005`. The record
and the AlphaEvolve value were both found by large evolutionary searches with orders of magnitude more
compute than a single `~85`-second gradient run on a few hundred cells, so a residual of a couple
ten-thousandths to the record is exactly what I would expect a single bounded constructor to leave on the
table — not a failure of tuning but the frontier of the method. The gap from the low `0.3810`s to `0.38087`
is the part of this seventy-year-old problem that careful hierarchical gradient refinement does not close,
with White's `0.379005` standing below as the floor the true constant cannot cross.

So the endpoint of this ladder is the step-function frontier that a single bounded hierarchical-gradient
constructor can actually reach — close to, and read honestly against, the best published results. I upscale
`120 → 600` for free, switch the solver from SLSQP to fast `β`-annealed analytic-gradient Adam with periodic
kicks, and finish with an exact-minimax subgradient polish, keeping the best true overlap throughout.

I want the reporting to be guarded so that this rung can never quietly regress. The upscale gives me the
`120`-cell value at `600` cells for free, so I hold that lifted vector as a floor and return the polished
result only if its true overlap is strictly lower — otherwise I fall back to the lifted vector. This makes
the endpoint monotone by construction: the reported bound is at most the `120`-cell `0.3810764`, and it
improves on it only if the grind genuinely finds something the evaluator confirms is lower. That discipline
matters precisely because I suspect the profile is a robust local optimum — if the Adam grind and the polish
merely reshuffle the tied plateau without lowering it (or, through the non-smooth chatter, momentarily raise
it), the guard ensures I still report the honest lifted value rather than a degraded one. The number that
comes out is therefore a true, feasibility-checked upper bound on `C5`, equal to the lifted value if the
basin holds and lower only if it does not — never worse. The
honest measured number I report is whatever the evaluator returns on the returned `600`-cell vector, and the
remaining gap to `0.38087` is the measure of how tightly squeezed — and how much still-contested at the fifth
decimal — this problem remains. If the grind holds the value rather than lowering it, that is itself the
finding: that `0.38108` is a basin floor, not a resolution cap, and crossing below it needs something other
than a better local optimizer.

The whole endpoint is cheap by design — the upscale is free, the Adam grind is seconds at `2×10⁻⁴` per step,
and the polish is `2000` cheap iterations, so the total run is on the order of a minute and a half. That
frugality is itself part of the honest framing: the published records came from searches costing orders of
magnitude more, and a minute-and-a-half single-trajectory constructor landing within a couple ten-thousandths
of them is either a triumph of the hierarchical recipe or a sign that the recipe and the record occupy
different basins — and the dimension count above says it is the latter. The clean falsifiable statement, in
the two quantities the evaluation tracks, is this: at `pieces_n = 600`, if the frontier is a resolution cap
the `upper_bound` should fall below the `120`-cell `0.3810764`; if it is a basin floor the `upper_bound`
should equal `0.3810764` to many digits, because upscaling preserved it and nothing local moved it. Those are
mutually exclusive outcomes, and whichever the evaluator returns is the real finding of this rung. I expect
the value to hold — that the sharp-`β` Adam, a fresh multistart, SLSQP, and the exact subgradient polish will
all refine inside the one basin and none will lower it — but I let the measurement, not my expectation, write
the number. Whichever way it falls, the endpoint's job is to establish honestly where a single bounded
hierarchical-gradient constructor stops, and to say clearly whether what stops it is a shortage of pieces or
the walls of a basin — a distinction the returned bound at `600` cells settles on its own.
