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

Before I invent machinery, let me look harder at *what kind* of critical point I am stuck at, because the
right cure depends on it. When I linearized the mean-field flow for `h2` at the origin, the leftover pair
obeyed `d/dt (u_1,u_2) = kappa [[0,1],[1,0]] (u_1,u_2)` with eigenvalues `±kappa`. That is a *strict*
saddle: it has a genuine unstable eigendirection `u_1 = u_2`. The flow stayed put not because there was no
way out, but because it started with an *exactly zero* perturbation along that direction and `e^{κt}·0 = 0`.
`h3` is worse: there the driver `E[z1z2z3 sigma'(u(z1+z2+z3))z_i]` vanishes to leading order `u^2` (the
monomial `z2 z3` first surfaces at the quadratic term of the expansion), so the Jacobian at the origin is
identically zero — a degenerate, monkey-saddle-like flat spot with no *linear* unstable direction at all,
only a higher-order one. This distinction reframes the whole rung. For
`h2` I do not need to create curvature; I need to (i) *seed* a nonzero perturbation along the unstable
direction that already exists, and (ii) *grow* it fast enough to matter inside `T=4000` steps. For `h3` even
that will be hard, because the instability only appears at cubic order.

That immediately rules out the generic saddle-escape tricks, and it is worth walking them to see why. Adding
explicit noise / a Langevin term to jiggle off the saddle is the textbook move, but SGD *already* injects
batch noise at `b=150`, and it did nothing for `h2` — an isotropic perturbation seeds the unstable
direction only through its `O(1/sqrt d)` projection and then has to grow it through a `kappa` that the
saturated activation makes tiny, so it never wins the race against the budget. A second-order / negative-
curvature step (find the most-negative Hessian eigenvector and descend it) would help a *strict* saddle,
but it is useless on `h3`, whose Hessian at the origin is exactly zero — there is no negative curvature to
find, which is precisely why `h3` will be the ceiling no matter what I do. And simply cranking the mean-field
step size does not help either: multiplying a zero (or `O(u^3)`) gradient by a bigger `lr` is still zero.
The lesson is that I must *manufacture structure*, not just add energy: seed the unstable direction, keep the
activation in the regime where `kappa` is `O(1)`, and use an optimizer that grows a persistently-small
gradient at a rate set by `lr` rather than by the gradient's vanishing magnitude. Three mechanisms deliver
exactly those three things, and each attacks a different part of why mean-field SGD stalled.

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

I should check this alternation is not secretly a no-op, because a plausible worry is that two sub-steps on
the *same* batch just add up to one bigger joint step and I have gained nothing. They do not, and the reason
is the update *ordering*. A simultaneous (Jacobi) update computes `grad_fc1` and `grad_fc2` at the *same*
point and moves both; my alternation is a Gauss-Seidel sweep — I move `fc1` first, and then recompute
`grad_fc2` at the *already-updated* features before moving `fc2`. The readout step therefore sees the freshly
aligned features, so the two sub-steps are genuinely coupled rather than parallel, and Gauss-Seidel can
converge (or escape a saddle) where the Jacobi iteration of the same operator merely oscillates or stalls.
Concretely, in the feature step the readout is held at its current diverse, `O(1)` values, so the residual
`(h* - fhat)` is projected onto the features through a *fixed* linear map — and that projection is exactly
what lets the feature step follow the strict saddle's unstable direction instead of having the motion along
it cancelled by a readout that is simultaneously chasing the still-wrong features. The alternation does not
manufacture curvature out of nothing; it keeps the descent aligned with the `±kappa` escape direction that
was already there.

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

Let me put numbers on why Adam is the right lever and not a decoration. With the seed sitting at
`|u| ≈ 0.1` (the mu-P scale I fix below), the gradient that lifts a leftover coordinate is a
power of `|u|` times an `O(1)` constant: for `h2`'s strict saddle the escape gradient is linear,
`O(|u|) ≈ 0.1`, while for `h3`'s degenerate one it is a higher power, `O(|u|^2) ≈ 10^{-2}` or smaller. A
plain-SGD step moves those in proportion to the gradient, `lr·0.1` versus `lr·10^{-2}` per step, so the
higher stair crawls exactly when it can least afford to. Adam divides each coordinate's raw gradient by
`sqrt(v) ≈ |grad|` (its running RMS), so a *consistently signed* small gradient becomes a step of size
`≈ lr` regardless of whether its magnitude is `0.1` or `10^{-2}`. That is the whole point: Adam
converts "persistently small and nonzero" into "`O(lr)` per step," which is exactly the signature of a
saddle-escape direction. The cost is that the step is now `lr`-controlled, not the self-scaling `eta=1/2` of
the bare flow, so I have to pick `lr` and split it by layer. I want the feature layer to cross the plateau
inside budget: `lr_fc1 = 10^{-2}` moves `u` about `10^{-2}·4000 = 40` units of *normalized* travel over the
run, comfortably enough to lift an `O(1)` coordinate. For the readout, the dimensional argument sets the
ratio: in the readout-fit phase the `M` coordinates move coherently to reduce the residual, so their net
effect on the `1/M`-normalized output is `(1/M)·M·(lr_fc2) = lr_fc2` per step. To keep that from overrunning
the feature step and oscillating the linear phase, I want the readout's output move a factor `~sqrt(M)`
below the feature step, giving `lr_fc2 = lr_fc1 / sqrt(M) = 10^{-2}/10 = 10^{-3}`. That is the standard
mu-P downscaling of the last layer, arrived at here from the requirement that features lead and the readout
follow.

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

The `1/d` variance is not a cosmetic swap from the mean-field `N(0,I_d)`; it is what rescues `kappa` from
the saturated tail, and this is computable. Under `w ~ N(0,I_d)` the pre-activation `<w,x>` over
`x in {+1,-1}^d` has standard deviation `sqrt(||w||^2) ≈ sqrt(d) ≈ 10`, so in the reduced dynamics the
noise coordinates act as a Gaussian smoothing width `s = ||v|| ≈ sqrt(d-P) ≈ 9.8`. The cascade drivers are
`E_G[sigma'(<u,z> + sG)]`, and smoothing a unit-area bump `sigma'` by a Gaussian of width `s` shrinks its
peak to `O(1/s)`: at `s ≈ 10` the effective driver — and hence `kappa` — is suppressed roughly tenfold,
which is a large part of why the finite-width mean-field run never grew `h2`'s perturbation in budget. The
mu-P `w ~ N(0,1/d)` gives `||w||^2 ≈ 1`, pre-activation std `≈ 1`, and `s = ||v|| ≈ sqrt((d-P)/d) ≈ 0.98`,
so the drivers return to `O(1)` — a concrete `~10x` boost to the escape rate, sitting the activation where
`sigma'(O(1)) ≈ 0.2` rather than in the tail where `sigma'(10) ≈ 4.5·10^{-5}`. And the *same* `1/d` scale
sets the seed: each signal weight starts at `|u_i^0| ≈ sqrt(1/d) = 0.1`, off the *exact* origin but near it —
precisely the nonzero perturbation along the `±kappa` unstable direction that the `d -> infinity` mean-field
flow lacked. So mu-P does double duty: it seeds the escape direction and it keeps `kappa` un-saturated so
Adam has something of `O(1)` scale to normalize and grow.

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

The `max(leap,2)` exponent lines up exactly with the saddle geometry I just described, which is a reassuring
consistency check rather than a coincidence. `h2`'s strict saddle has an `O(u)` unstable direction; seeding
it at `0.1` and growing it through Adam takes on the order of the `d^2 = 10^4` samples the leap-2 bound
names — and `10^4` is a factor `60` inside my `n = b·T = 6·10^5` budget, so `h2` has real room to escape.
`h3`'s degenerate saddle has only an `O(u^3)` instability, and the leap-3 bound puts it at `d^3 = 10^6`;
against `n = 6·10^5` the ratio is `6·10^5 / 10^6 = 0.6`, so I have about `60%` of the samples the single
leap-3 monomial needs. That is why I expect `h3` to move, if at all, only partway — enough perhaps to dent
its recovery, not enough to drive `test_mse_h3` to zero. The two-substeps-per-batch alternation does not
buy me around this: it is two gradient steps but on the *same* `4000` batches, so the sample budget
`n = 6·10^5` is unchanged; what it buys is a better-conditioned escape per sample, not more samples.

One limiting check before I read off predictions: does the composite risk *breaking* `h1`, where the bare
flow already worked? It should not, and the reason is that at a non-degenerate point — which is exactly what
`h1`'s active cascade is, every lit coordinate having a genuine first-order driver — the Gauss-Seidel
alternation agrees with joint descent to first order (it is ordinary block-coordinate descent on a smooth
objective), and mu-P + Adam only rescale the step. So `h1` keeps climbing, and in fact should climb
*further* than mean-field managed: the piece that stalled there was the degree-3 `z1z2z3` stair, whose
`u_1 u_2`-gated gradient was genuinely nonzero but crawling at the `t^4` rate that plain SGD takes at face
value. Adam's normalization is precisely the fix for a small-but-nonzero gradient, so the `h1` degree-3
stair that barely moved under the bare flow should now finish inside budget. The composite therefore
strictly dominates the mean-field recipe on `h1` and adds the leap-2 escape on `h2` — it does not trade one
for the other. That is the consistency I want before trusting the predictions.

Now the falsifiable expectations against the mean-field numbers, target by target. On `h1`: mean-field SGD
already partially climbed it (`test_mse 2.45`, `score 0.086`); with Adam amplifying the slow degree-3 stair
and alternation cleanly separating feature alignment from readout fitting, I expect `h1` to improve
substantially — `test_mse_h1` should drop well below `2.45` and `score_h1` rise well above `0.086`, plausibly
to the `0.5-0.7` range if the full staircase (including the `z1z2z3` term that the bare `t^4` cascade
barely touched in budget) gets picked up. On `h2`, the decisive test: mean-field SGD was *flat* at the
trivial `test_mse 3.00 / score 0.050`. If the saddle-to-saddle machinery works, `h2` should move off that
floor for the first time on the ladder — a leap-2 target is exactly what alternation + Adam is for — so I
expect `score_h2` to rise meaningfully above `0.050` (even partial recovery of one of its three degree-2
monomials would show). The recovery column is the sharp tell here: mean-field left it pinned at `1.00`
(learned nothing), and since `h2` has three unit monomials, cleanly learning one of the three would pull
`fourier_recovery_h2` down toward `2/3 ≈ 0.67`, and learning all three toward `0`. So a recovery that
breaks below `1.0` is the unambiguous fingerprint that the saddle-escape fired, independent of exactly how
much `test_mse` falls. If `h2` *stays* at `0.050` with recovery still `1.00`, the composite has failed at
its central job and the saddle-escape did not fire. On `h3`: I am most cautious here, because the budget argument says `d^3` is out
of reach — I expect `h3` to improve only modestly over the trivial `score 0.368`, recovering perhaps part
of the single monomial but not driving `test_mse_h3` to zero; a large jump on `h3` would be a genuine
surprise. `h3` is the one target with no trivial-predictor ambiguity in its recovery — it has a single
monomial, so *any* `fourier_recovery_h3` below `1.0` is unambiguous evidence the degenerate saddle gave
ground, however partially. Let me size the aggregate this all implies, since a geometric mean can be
counterintuitive. Writing it as `exp(-(mse_h1 + mse_h2 + mse_h3)/3)`, it is the exponential of the mean
test MSE, so it rewards pulling *down the worst environment*. Even a deliberately conservative escape —
say `h1` improved to a score near `0.5`, `h2` only partway off its floor to `~0.2`, and `h3` essentially
stuck near its trivial `0.37` — gives a geometric mean `(0.5·0.2·0.37)^{1/3} ≈ 0.33`, already roughly
three times the `0.105-0.117` the first two rungs posted. If instead `h2` escapes cleanly and `h3` dents,
the aggregate climbs further still. So the bar this rung has to clear is low relative to what a working
saddle-escape delivers, and the single decisive signal to watch remains `h2`'s recovery breaking below
`1.0` — that is where the mean-field ladder was flat, and where this composite either fires or does not. The distilled fill — mu-P init, shifted sigmoid, per-layer Adam, layer-wise
alternating `train_step` — is in the answer.
