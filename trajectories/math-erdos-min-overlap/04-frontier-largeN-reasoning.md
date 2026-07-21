The hierarchical lifts kept paying, and each step confirmed the recipe: upscale the optimized profile for
free, kick to break the block plateau, refine with annealed-soft-max search keeping the best true overlap.
`0.381240` at `24` cells, then `0.381076` at `120` — a modest drop from lifting, so the cap was
resolution, not a missed basin, as predicted; and the per-step gain collapsed from `0.118` to about
`1.6×10⁻⁴`, the signature of a search closing on a hard limit. The published frontier lives at several
hundred cells (AlphaEvolve's `0.380924`, the record `0.38086945`), so the next move is to lift once more to
that scale and grind there with a longer, sharper refinement plus an exact minimax polish — pushing toward
`~0.3809`, while being honest that the record was bought with orders of magnitude more compute than I have
here.

I take the `120`-cell profile and upscale `×5` to `600`. As before the upscale is free — same step
function, same `C` — and parks me on a degenerate plateau of `5`-wide blocks (`480` flat directions) that I
kick to give the optimizer traction. Then I confront the scaling wall I have deferred since the coarse
pass. SLSQP was the right tool at two dozen and a hundred-odd cells, but at `600` it is the bottleneck: it
finite-differences the surrogate at `O(n)` evaluations of an `O(n²)` correlation, so one gradient is
`O(n³)` — about `2×10⁸` operations — on top of a super-linear QP, running a couple of minutes per annealed
ladder, and from a good starting point it barely moves because the surrogate optimum has essentially
coincided with where the lifted profile sits. Spending more SLSQP buys almost nothing; a fresh multi-start
at `600` is worse, since random balanced starts average around `0.6` and the fraction reaching the good
basin is negligible at this dimension; and lifting further to `1200` cells only draws the same envelope
more smoothly if the profile is already in its basin. So the free upscale plus a *scalable* optimizer is
forced.

The scalable replacement is `β`-annealed Adam on the *analytic* soft-max gradient. Writing the soft-max
weights `w_k = exp(β(c_k − m))/Σ_j exp(β(c_j − m))`, the gradient is `∂C̃/∂v_a = (2/n) Σ_k w_k ∂c_k/∂v_a`
with `∂c_k/∂v_a = (1 − v_{a−k}) − v_{a+k}` — the first term from `v_a` as the `v_i` factor, the second from
`v_a` complemented; I checked it against finite differences and it is exact. Summed against the weights,
the two pieces are themselves correlations of `w` with `1 − v` and with `v`, so the whole gradient is one
or two `O(n²)` correlations — no `O(n)` finite-difference blow-up. Each Adam step is a correlation for `c`,
a softmax for `w`, and a couple of correlations for the gradient: about `2×10⁻⁴` seconds at `n = 600`, so
tens of thousands of steps cost seconds. Adam rather than plain projected descent because the minimax is
badly conditioned — cells under the binding shifts see large gradients while quiet cells barely move — and
Adam's per-coordinate scaling absorbs that so a long run keeps making progress rather than stalling on the
stiff directions. After each step I re-project with the same clip-and-redistribute operator, `O(n)` per
step.

The exact subgradient polish is complementary, not a repeat. Adam descends the *soft* surrogate, whose
weighted gradient smears across the near-tied band; the polish descends the *true* max by taking a hard
active set — shifts within `10⁻⁹` of the maximum — and stepping down their averaged exact gradient. Because
the true objective is non-smooth at the tied plateau, a fixed step would chatter, so the polish uses a
decaying rate `lr₀/(1 + it/500)` over `2000` iterations — large early steps to reach the plateau, shrinking
steps to settle without oscillating. Averaging over the whole active set is the crucial choice: it seeks
the direction that lowers every binding shift at once, the only kind of move that lowers the max when many
shifts are tied, and it also reveals a local optimum when no such direction is feasible, because then the
step projects to near zero. The `10⁻⁹` tolerance has to be tight — loosen it to `10⁻⁴` and I sweep in
non-binding shifts, dilute the direction, and stall above the optimum; tighten it far below floating-point
noise and the active set flickers with round-off. At that tolerance the width of the active set is also a
direct readout of how flat the envelope has become.

The `β` schedule sharpens at `600` cells, though not for the obvious reason. The worst-case gap
`(2/n)·log(2n−1)/β` actually shrinks with `n`, so the shift count alone would not force a sharper `β`. But
that bound is only tight when the near-worst band is wide, and here it is: as the envelope flattens, an
active set of `A` shifts sits at the max and the soft-max reads `m + (1/β)log A`, so the effective gap is
`(2/n)·log(A)/β` with `A` in the hundreds. So I anneal `β` into the thousands, where the gap falls to the
`10⁻⁶` scale, below the ten-thousandths I am fighting over.

Now I have to be honest about what this reaches, and the conservation picture makes me cautious. Once
hundreds of shifts are tied at the worst level, lowering the max requires lowering *all* of them at once —
but conservation says the mass stripped off the active shifts reappears by pushing the next-highest
inactive shifts up until they join the active set. A dimension count makes "large enough active set"
precise. To strictly lower the max I need a feasible direction `d` with `∇c_k · d < 0` for every active
shift, plus `d · 1 = 0` from the sum constraint, inside a tangent space of dimension at most `n − 1` (less
once pinned cells drop out). When the active count `A` is small relative to `n − 1` such a direction
generically exists; but as `A` grows toward `n`, the `A` gradients `∇c_k` positively span the tangent
space, and once they do no single direction has negative inner product with all of them — the descent cone
collapses and I am at a local optimum, where the subgradient step projects to nearly zero and sharper `β`
only refines inside the same basin. The prediction that the active set widens with resolution means `A` at
`600` cells may well reach the hundreds, the same order as `n` — exactly the collapse regime. So I
genuinely do not know whether the grind pushes below `0.3810764` or merely holds it: the reorganizing
passes should hold the `120`-cell value exactly since upscaling is free, and the question is whether the
extra freedom opens a descent direction or just redraws the same envelope more smoothly. I expect the
latter but will not assert it; the evaluator on the returned `600` heights settles it.

A quantitative check sharpens the "different basin, not worse flattening" reading. At `n = 600` the
averaging floor is `C ≥ 600/(2·1199) = 0.25021`, so `0.3810764` is a peak-to-mean ratio of about `1.523`;
the record sits at about `0.38087/0.2502 ≈ 1.522`, essentially the same denominator and the same ratio. My
envelope, if it holds, is *just as flat* as the record's — I am not losing because my worst overlap towers
higher above the mean. The record is lower purely because it is a different flat configuration whose common
plateau sits a couple ten-thousandths beneath mine. That is the fingerprint of two distinct local optima of
comparable quality, not a resolution deficit, and it tells me the remaining gap is not something I can
flatten my way into from here.

So I upscale `120 → 600` for free, switch from SLSQP to fast `β`-annealed analytic-gradient Adam with
periodic kicks, and finish with the exact-minimax subgradient polish, keeping the best true overlap
throughout. And I guard the reporting so this step can never quietly regress: I hold the lifted vector as a
floor and return the polished result only if its true overlap is strictly lower, otherwise I fall back to
the lifted vector. That makes the endpoint monotone — the reported bound is at most the `120`-cell
`0.3810764` and improves on it only if the grind genuinely finds something the evaluator confirms is lower
— which matters precisely because I suspect a robust local optimum: if the polish merely reshuffles the
tied plateau or momentarily raises it through non-smooth chatter, the guard still reports the honest lifted
value. If the grind holds `0.38108`, that sits about `1.9×10⁻⁴` above the record and `2.1×10⁻³` above
White's floor, and both records were found by large evolutionary searches with orders of magnitude more
compute than an `~85`-second gradient run — so a residual of a couple ten-thousandths is the frontier of a
single bounded constructor, not a tuning failure. Whether the bound falls below `0.3810764` or holds it to
many digits is itself the finding this step returns: a resolution cap in the first case, a basin floor in
the second.
