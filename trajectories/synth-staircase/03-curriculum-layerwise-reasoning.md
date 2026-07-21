Mean-field SGD did exactly what its mechanism predicted, and the numbers split by leap with surgical
cleanliness. On `h1` it beat the lazy floor: `test_mse_h1` fell from NTK's `2.78` to `2.45`, recovery from
`0.96` to `0.89`, `score_h1` rose from `0.062` to `0.086` — feature learning climbed at least the degree-1
and part of the degree-2 stair that no frozen kernel could reach. But on `h2` it landed at
`test_mse_h2 = 3.00`, recovery `1.00`, `score_h2 = 0.050` — *identical* to NTK's trivial-predictor row. And
on `h3`: `test_mse_h3 = 1.00`, recovery `0.9995`, `score_h3 = 0.368` — again the trivial `exp(-1)` of a
unit-variance target, recovering nothing. The aggregate barely moved (`0.105 -> 0.117`), and the diagnosis
is precise: mean-field SGD's flow starts at the origin in the signal directions, and a coordinate gets zero
first-order gradient until the supports beneath it are lit. `h1` has a degree-1 entry point that starts the
cascade; `h2` leaps by 2 and `h3` straight to degree 3, so both leave coordinates frozen at the origin with
zero gradient — and `score_h3 = 0.368` is not a near-success but the deceptive trivial number I flagged from
the start. The failure is not the optimizer; plain SGD on a flat saddle goes nowhere because the gradient is
genuinely zero. The cure has to *manufacture* the staircase the leap-2 and leap-3 targets lack.

The right cure depends on *what kind* of critical point I am stuck at. When I linearized the mean-field flow
for `h2` at the origin, the leftover pair obeyed `d/dt (u_1,u_2) = kappa [[0,1],[1,0]] (u_1,u_2)`, eigenvalues
`±kappa` — a *strict* saddle with a genuine unstable eigendirection `u_1 = u_2`. The flow stayed put not
because there was no way out, but because it started with an exactly zero perturbation along it and
`e^{κt}·0 = 0`. `h3` is worse: there the driver `E[z1z2z3 sigma'(u(z1+z2+z3))z_i]` vanishes to leading order
`u^2` (the monomial `z2z3` first surfaces at the quadratic term), so the Jacobian at the origin is
identically zero — a degenerate, monkey-saddle flat spot with no *linear* unstable direction, only a
higher-order one. This reframes what I need: for `h2` I need to (i) seed a nonzero perturbation along the
unstable direction that already exists and (ii) grow it fast enough to matter inside `T=4000` steps; for `h3`
even that will be hard, because the instability only appears at cubic order.

That rules out the generic saddle-escape tricks. Adding explicit noise / a Langevin term is the textbook
move, but SGD already injects batch noise at `b=150` and it did nothing for `h2` — an isotropic perturbation
seeds the unstable direction only through its `O(1/sqrt d)` projection and then has to grow it through a
`kappa` the saturated activation makes tiny, so it never wins the race against the budget. A second-order /
negative-curvature step would help a strict saddle but is useless on `h3`, whose Hessian at the origin is
exactly zero — no negative curvature to find, which is why `h3` is the ceiling no matter what I do. And
cranking the step size multiplies a zero (or `O(u^3)`) gradient by a bigger `lr` and stays zero. The lesson:
*manufacture structure*, not just add energy — seed the unstable direction, keep the activation where
`kappa` is `O(1)`, and use an optimizer that grows a persistently-small gradient at a rate set by `lr`
rather than by the gradient's vanishing magnitude. Three mechanisms deliver exactly those three things.

First, layer-wise alternation to drive saddle-to-saddle dynamics. The bare flow freezes on `h2`/`h3` because
the first- and second-layer updates happen *simultaneously* — readout `a` and features `u` move together, and
at the origin neither has a gradient in the leftover coordinates. The saddle-to-saddle picture says learning
a hierarchical target should proceed as a sequence of plateaus: sit near a saddle, slowly align one layer, a
fast transition lifts the next component, then another plateau. The way to *force* that structure is to
alternate which layer moves. Each `train_step` does two sub-updates on the same batch: a *feature step*
updating only the first layer with the readout frozen, then a *readout step* updating only the second layer
with the features frozen. Crucially this is a Gauss-Seidel sweep, not a Jacobi one: a simultaneous update
computes `grad_fc1` and `grad_fc2` at the same point, whereas here I move `fc1` first and then recompute
`grad_fc2` at the *already-updated* features. The feature step, with the readout held at its current nonzero
diverse values, gets a gradient on `u` even where the joint flow had none — the residual `(h* - fhat)`
projected through the *fixed* readout exposes correlations the simultaneous update averaged away, and follows
the strict saddle's `±kappa` escape direction instead of having motion along it cancelled by a readout
chasing still-wrong features. Gauss-Seidel can escape a saddle where the Jacobi iteration of the same
operator merely stalls. I return the mean of the two sub-losses as the reported training loss.

Second, a larger *adaptive* step to actually escape once a gradient appears. Even with alternation, the
gradient that lifts a leftover coordinate is tiny — the high-order correlation the homogeneity argument
showed vanishes to order `2^{k-1}` at the saddle — and a fixed-`lr` SGD step on it moves nowhere in `T=4000`.
Adam normalizes each coordinate's update by a running estimate of its gradient magnitude, so a consistently
small but nonzero gradient — exactly the signature of a saddle-escape direction — gets amplified to an
`O(lr)` step regardless of raw scale. Put numbers on it: with the seed at `|u| ≈ 0.1` (the mu-P scale below),
`h2`'s strict saddle has an escape gradient `O(|u|) ≈ 0.1` while `h3`'s degenerate one is `O(|u|^2) ≈
10^{-2}` or smaller, so plain SGD would crawl on the higher stair exactly when it can least afford to. Adam
divides each raw gradient by `sqrt(v) ≈ |grad|`, turning both into a step of size `≈ lr`. The cost is that
the step is now `lr`-controlled, not the self-scaling `eta=1/2` of the bare flow, so I pick `lr` and split it
by layer. `lr_fc1 = 10^{-2}` moves `u` about `10^{-2}·4000 = 40` units of normalized travel over the run,
comfortably enough to lift an `O(1)` coordinate. For the readout, `M` coordinates move coherently so their
net effect on the `1/M`-normalized output is `(1/M)·M·lr_fc2 = lr_fc2` per step; to keep it a factor `~sqrt(M)`
below the feature step and stop the linear phase oscillating, `lr_fc2 = lr_fc1 / sqrt(M) = 10^{-3}` — the
standard mu-P last-layer downscaling, arrived at here from "features lead, readout follows." The larger
first-layer LR is the leap-1 warm-up; the smaller readout LR keeps recombination stable.

Third, the initialization — mu-P-style. The mean-field recipe used `w ~ N(0, I_d)` and the `1/M` readout.
With Adam and per-layer LRs I want the signal weights at the near-origin saddle the saddle-to-saddle story
needs, while keeping `<w,x>` at `O(1)` so the activation's low-order derivatives (the `m_r` driving the
cascade) are actually exercised. The mu-P prescription: `w ~ N(0, 1/d)`, zero first-layer bias, readout
`N(0,1)`, `1/M` output. The `1/d` variance is not cosmetic — it rescues `kappa` from the saturated tail, and
this is computable. Under `w ~ N(0,I_d)` the pre-activation `<w,x>` has std `sqrt(||w||^2) ≈ sqrt(d) ≈ 10`,
so in the reduced dynamics the noise coordinates act as a Gaussian smoothing width `s = ||v|| ≈ 9.8`; the
cascade drivers `E_G[sigma'(<u,z> + sG)]` are a unit-area bump smoothed by that width, shrinking its peak to
`O(1/s)`, so at `s ≈ 10` the effective driver and hence `kappa` is suppressed roughly tenfold — a large part
of why the finite-width mean-field run never grew `h2`'s perturbation in budget. The mu-P `N(0,1/d)` gives
`||w||^2 ≈ 1`, pre-activation std `≈ 1`, `s ≈ 0.98`, so the drivers return to `O(1)` — a `~10x` boost,
sitting the activation where `sigma'(O(1)) ≈ 0.2` rather than the tail where `sigma'(10) ≈ 4.5·10^{-5}`. And
the *same* `1/d` scale sets the seed: each signal weight starts at `|u_i^0| ≈ sqrt(1/d) = 0.1`, off the exact
origin but near it — precisely the nonzero perturbation along the `±kappa` unstable direction the
`d -> infinity` flow lacked. So mu-P does double duty: it seeds the escape direction and keeps `kappa`
un-saturated so Adam has something of `O(1)` scale to normalize and grow. I keep the shifted sigmoid
`sigma(x) = sigmoid(x - 0.5)`, because the cascade still needs `sigma^{(r)}(0) != 0` for all low `r` and a
symmetric activation would zero half the chain no matter how cleverly I alternate the layers.

What can this composite actually reach? Saddle-to-saddle SGD learns a leap-`k` function in roughly
`d^{max(k,2)}` steps. For `h1` (leap-1) that is `d^2 = 10^4`, comfortably inside `n = 6·10^5`. For `h2`
(leap-2) also `d^2 = 10^4` (the `max(k,2)` floor is `2`), a factor `60` inside budget — so if the
saddle-escape fires, `h2` has real room. For `h3` (leap-3) it is `d^3 = 10^6`, *above* budget: the ratio
`6·10^5 / 10^6 = 0.6` gives me about `60%` of the samples the single leap-3 monomial needs, so I expect `h3`
to move, if at all, only partway — enough perhaps to dent its recovery, not to drive `test_mse_h3` to zero.
The two sub-steps per batch do not buy around this: they are two gradient steps on the same `4000` batches,
so `n` is unchanged; what they buy is a better-conditioned escape per sample, not more samples.

The composite should not *break* `h1`, where the bare flow already worked. At a non-degenerate point — which
`h1`'s active cascade is, every lit coordinate having a genuine first-order driver — the Gauss-Seidel
alternation agrees with joint descent to first order (ordinary block-coordinate descent on a smooth
objective), and mu-P + Adam only rescale the step. So `h1` keeps climbing, and should climb *further*: the
piece that stalled there was the degree-3 `z1z2z3` stair, whose `u_1 u_2`-gated gradient was nonzero but
crawling at the `t^4` rate plain SGD takes at face value — and Adam's normalization is exactly the fix for a
small-but-nonzero gradient, so that stair should now finish inside budget. The composite dominates the
mean-field recipe on `h1` and adds the leap-2 escape on `h2`, rather than trading one for the other.

The predictions, target by target. On `h1`: mean-field already reached `test_mse 2.45 / score 0.086`; with
Adam amplifying the slow degree-3 stair and alternation separating feature alignment from readout fitting, I
expect a substantial improvement — `test_mse_h1` well below `2.45` and `score_h1` well above `0.086`,
plausibly a large fraction of the way to full recovery if the `z1z2z3` term the bare `t^4` cascade barely
touched now gets picked up. On `h2`, the decisive test: mean-field was flat at the trivial `3.00 / 0.050`. If
the machinery works, `h2` moves off that floor for the first time on the ladder. The recovery column is the
sharp tell — mean-field left it pinned at `1.00`; since `h2` has three unit monomials, cleanly learning one
pulls `fourier_recovery_h2` toward `2/3 ≈ 0.67` and all three toward `0`, so *any* recovery breaking below
`1.0` is the unambiguous fingerprint that the saddle-escape fired, independent of how far `test_mse` falls.
If `h2` stays at `0.050` with recovery `1.00`, the composite failed at its central job. On `h3` I am most
cautious — the budget argument puts `d^3` out of reach, so I expect only modest improvement over the trivial
`0.368`, recovering part of the single monomial but not driving `test_mse_h3` to zero; a large jump would be
a genuine surprise, and here `h3`'s single monomial means any `fourier_recovery_h3 < 1.0` is unambiguous
evidence the degenerate saddle gave ground. For the aggregate, `exp(-(mse_h1+mse_h2+mse_h3)/3)` rewards
pulling down the worst environment: even a conservative escape — `h1` near `0.5`, `h2` only partway off its
floor to `~0.2`, `h3` stuck near `0.37` — gives a geometric mean `(0.5·0.2·0.37)^{1/3} ≈ 0.33`, already
roughly three times the `0.105-0.117` of the two earlier strategies, and the whole question turns on whether `h2`'s
recovery finally breaks below `1.0`. The fill — mu-P init, shifted sigmoid, per-layer Adam, layer-wise
alternating `train_step`.
