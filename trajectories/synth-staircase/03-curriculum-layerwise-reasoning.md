Mean-field SGD did exactly what its mechanism predicted, and the numbers split by leap with surgical
cleanliness. On `h1` it beat the lazy floor: `test_mse_h1` fell from NTK's `2.78` to `2.45`, recovery from
`0.96` to `0.89`, `score_h1` rose from `0.062` to `0.086` — feature learning climbed at least the
degree-1 and part of the degree-2 stair that no frozen kernel could reach. But on `h2` it landed at
`test_mse_h2 = 3.00`, recovery `1.00`, `score_h2 = 0.050` — *identical* to NTK's trivial-predictor row.
And on `h3`: `test_mse_h3 = 1.00`, recovery `0.9995`, `score_h3 = 0.368` — again exactly the trivial
`exp(-1)` of a unit-variance target, recovering nothing. So the aggregate barely moved (`0.105 -> 0.117`),
and the diagnosis is precise: mean-field SGD's flow starts at the origin in the signal directions, and a
coordinate only gets a first-order gradient once the lower-degree supports beneath it are lit. `h1` has a
degree-1 entry point that starts the cascade; `h2` leaps by 2 at its first support and `h3` leaps straight
to degree 3, so both leave coordinates frozen at the origin with *zero gradient* — and `score_h3 = 0.368`
is not a near-success, it is the deceptive trivial number I flagged from the start. The failure is not the
optimizer; plain SGD on a flat saddle goes nowhere because the gradient is genuinely zero there. The cure
has to *manufacture* the staircase the leap-2 and leap-3 targets do not have — a way to lift coordinates
the bare flow leaves at the origin.

So the question that drives this rung: how do I get off a saddle where the first-order signal in the
relevant coordinates is zero? Three mechanisms compose into the answer, and each attacks a different part
of why mean-field SGD stalled.

First mechanism: layer-wise alternation to drive saddle-to-saddle dynamics. The reason the bare flow
freezes on `h2`/`h3` is that the first-layer and second-layer updates happen *simultaneously* — the readout
`a` and the features `u` move together, and at the origin neither has a gradient in the leftover
coordinates. The saddle-to-saddle picture says learning a hierarchical target should proceed as a sequence
of *plateaus*: the dynamics sits near a saddle, slowly aligns one layer, then a fast transition lifts the
next component, then another plateau. The way to *force* that structure rather than wait for it is to
alternate which layer moves. In each step I do two sub-updates: first a *feature step* that updates only
the first layer with the readout frozen, then a *readout step* that updates only the second layer with the
features frozen. The feature step, with the readout held fixed at its current (now nonzero, diverse)
values, gets a gradient on `u` even where the joint flow had none, because the residual `(h* - fhat)`
projected through the *fixed* readout exposes correlations the simultaneous update averaged away. Then the
readout step recombines those freshly-aligned features into the output. Alternating these is the
saddle-to-saddle mechanism made explicit: each layer in turn is pushed to follow the next-easiest monomial
in the leap ordering, instead of both layers idling together on the plateau. Concretely, in `train_step` I
freeze the readout's `requires_grad`, take one optimizer step on the feature loss, unfreeze it; then freeze
the features, take one step on the readout loss, unfreeze. Each sub-update samples the *same* batch
`(x,y)` (the harness hands one batch per `train_step`), so it is two gradient steps on one batch with
complementary parameter masks. I return the mean of the two sub-losses as the reported training loss.

Second mechanism: a larger, *adaptive* step to actually escape the saddle once a gradient appears. Even
with alternation, the gradient that lifts a leftover coordinate off the origin is tiny — it is the
high-order correlation that the homogeneity argument showed vanishes to order `2^{k-1}` at the saddle. A
fixed-`lr` SGD step on a gradient that small moves nowhere in `T=4000` steps. Adam is the natural lever:
it normalizes each coordinate's update by a running estimate of its gradient magnitude, so a *consistently
small but nonzero* gradient — exactly the signal that escapes a saddle — gets amplified to an `O(lr)` step
regardless of its raw scale. This is the known practical mechanism for accelerating escape from
high-dimensional saddles where vanilla SGD crawls. So I replace plain SGD with Adam, `betas=(0.9,0.999)`,
`eps=1e-8`. But Adam's per-coordinate normalization also means the *step size* is no longer the
self-correcting `eta=1/2` of the mean-field flow — it is set by the `lr`, and I need a `lr` large enough to
cross the plateau in the budget but not so large it destabilizes the readout regression. A per-layer split
handles this: the first layer (which must escape the saddle) gets the full `lr = 1e-2`, while the readout
gets `lr = 1e-2 / sqrt(M)` so its update scale matches the `1/M`-normalized output and the linear
readout-phase stays stable. The larger first-layer LR is the "leap-1 warm-up" that picks up the
low-frequency monomial quickly; the smaller readout LR keeps the recombination phase from oscillating.

Third mechanism: the parametrization that makes width-scaling preserve feature learning — mu-P-style
init. The mean-field recipe used `w ~ N(0, I_d)` and the `1/M` readout. With Adam and a per-layer LR I want
the initialization to keep the signal weights at the near-origin saddle the saddle-to-saddle story needs,
while keeping the pre-activation `<w,x>` at `O(1)` so the activation's low-order derivatives (the `m_r`
that drive the cascade) are actually exercised. The mu-P prescription is `w ~ N(0, 1/d)` (so
`<w,x> = O(1)` over `x in {+1,-1}^d`, and the signal block starts at `O(1/sqrt(d)) ≈ 0.1`, i.e. near the
saddle), a zero first-layer bias, and a readout drawn `N(0,1)` with the `1/M` output normalization carried
over from the mean-field scaling. This is the parametrization under which the feature-learning dynamics
does not degenerate as width changes — the readout and feature updates stay balanced. I keep the shifted
sigmoid `sigma(x) = sigmoid(x - 0.5)`, because the cascade-keeps-alive argument is unchanged: I still need
`sigma^{(r)}(0) != 0` for all low `r`, and a symmetric activation would zero half the chain regardless of
how cleverly I alternate the layers. The shift is what makes the low derivatives nonzero around the origin.

Let me reason carefully about what this composite can actually reach, because I should not overclaim. The
leap-complexity picture says saddle-to-saddle SGD on a low-leap target should learn it in roughly
`d^{max(leap,2)}` steps. For `h1` (leap-1) that is `d^2 = 10^4` — comfortably inside my `n = 6·10^5`
budget. For `h2` (leap-2) that is `d^2 = 10^4` again (the `max(leap,2)` floor is `2`), also inside budget,
*if* the saddle-escape actually fires. For `h3` (leap-3) that is `d^3 = 10^6`, which is *above* my budget
`n = 6·10^5`. So even the strongest baseline has a structural ceiling: `h3`'s single leap-3 monomial sits
at a sample complexity my fixed budget cannot reach, so I should *not* expect `h3` to be fully learned — at
best partially. But `h1` and `h2`, both at the `d^2` threshold, are within reach if the alternation +
Adam + mu-P combination genuinely escapes the saddle that froze mean-field SGD. This is the honest version
of the claim: layer-wise saddle-to-saddle training should learn any low-leap function in
`d^{max(leap,2)}` steps, so it is the natural upper bound on top of the leap-1-only mean-field baseline —
but the leap-3 monomial is past the budget and stays the hardest case.

Now the falsifiable expectations against the mean-field numbers, target by target. On `h1`: mean-field SGD
already partially climbed it (`test_mse 2.45`, `score 0.086`); with Adam amplifying the slow degree-3 stair
and alternation cleanly separating feature alignment from readout fitting, I expect `h1` to improve
substantially — `test_mse_h1` should drop well below `2.45` and `score_h1` rise well above `0.086`, plausibly
to the `0.5-0.7` range if the full staircase (including the `z1z2z3` term that the bare `t^4` cascade
barely touched in budget) gets picked up. On `h2`, the decisive test: mean-field SGD was *flat* at the
trivial `test_mse 3.00 / score 0.050`. If the saddle-to-saddle machinery works, `h2` should move off that
floor for the first time on the ladder — a leap-2 target is exactly what alternation + Adam is for — so I
expect `score_h2` to rise meaningfully above `0.050` (even partial recovery of one of its three degree-2
monomials would show). If `h2` *stays* at `0.050`, the composite has failed at its central job and the
saddle-escape did not fire. On `h3`: I am most cautious here, because the budget argument says `d^3` is out
of reach — I expect `h3` to improve only modestly over the trivial `score 0.368`, recovering perhaps part
of the single monomial but not driving `test_mse_h3` to zero; a large jump on `h3` would be a genuine
surprise. The aggregate is a geometric mean, so even a strong `h1` and a partial `h2` with a stuck `h3`
should lift the overall score well above the `0.105-0.117` the first two rungs posted — that is the bar
this rung has to clear. The distilled fill — mu-P init, shifted sigmoid, per-layer Adam, layer-wise
alternating `train_step` — is in the answer.
