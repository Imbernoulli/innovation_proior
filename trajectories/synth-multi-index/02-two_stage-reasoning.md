The floor came back exactly as the information-exponent argument said it would, and the numbers are
worth reading closely because they tell me *which* failure I am looking at. Vanilla joint SGD left
`subspace_err` at 1.95 (r2), 2.37 (r3), 2.74 (r4). Line those up against `√(2r)` = 2.00/2.449/2.828: the
gaps are `0.05`/`0.08`/`0.09`, i.e. `2.3%`/`3.2%`/`3.1%` below the uncorrelated value. That is *within
sampling of* two rank-`r` orthogonal projectors that share no directions — the first-layer row-span and
the teacher subspace are still essentially uncorrelated after 8000 steps, the sub-percent-to-3% dip is
just the finite-`d` `r²/d` overlap correction, not real alignment. The first layer did not move toward
`V*` in any way that counts. And `test_mse` sat at 10.4/9.8/9.1 — *above* the `Var[g] = 6` label
variance I predicted as the do-nothing floor. Read that as a ratio: `test_mse/Var[g]` = `1.73`/`1.63`/
`1.52`, so the joint dynamics is not merely failing to explain the `6` of label variance, it is
*injecting* another `50`–`70%` of prediction variance on top — the readout chasing near-random,
co-adapting features is actively making the fit worse than a constant predictor would. The `score` is
`5·10⁻⁶`/`1.4·10⁻⁵`/`1.7·10⁻⁵`, three to four orders below anything useful, and it lands right where the
two saturated exponentials put it: `exp(−subspace_err²/r)·exp(−test_mse) ≈ exp(−2)·exp(−9) ≈ 1.7·10⁻⁵`
on r4, matching the measured `1.7·10⁻⁵`. The `exp(−2)` subspace factor is the ceiling every one of those
scores is pinned under.

The seed structure tells me this is not a variance problem I can fix by trying harder. On r2 the
per-seed `subspace_err` is 1.993/1.928/1.943 — a spread of `0.065` around `1.95`, all three within `3.5%`
of `√4 = 2.0`. It is not that one lucky seed escaped and two got stuck; *every* seed is pinned at the
uncorrelated value. Same on r3 (2.361/2.417/2.336) and r4 (2.752/2.755/2.725), spreads under `0.03`.
This is the `d²`-wall seed-for-seed: a *structural* dead end, not a luck-of-the-init fluctuation. More
seeds or a warmer restart would print the same three numbers. So whatever I do next cannot be "the same
gradient, more carefully" — it has to change the mechanism.

So the diagnosis is sharp, and it has two intertwined causes I should attack together. First, the net is
conflating two jobs — finding `V*` (nonconvex, the hard part) and fitting the link (convex, once the
features are good) — and training them jointly means the readout is fitting moving, near-random features
the whole time, which is exactly why the MSE overshoots its own `6` floor by `50`–`70%`. Second, the
first-layer signal is third-order-weak: at a random start the population degree-3 correlation is `O(1/d)`
and a single `lr = 5e-2` step moves a row only `~4·10⁻⁴` into `V*`. The cure for the first is to
*separate the stages*: train the first layer alone to align with `V*`, then freeze it and solve the
readout in closed form. The cure for the second is to make the first-layer phase as clean and
well-conditioned as I can, so whatever alignment signal exists is not diluted by a co-adapting readout or
by runaway row norms.

Separating the stages is one of several moves the hooks allow, so weigh the others first. Keep joint SGD
but go online — fresh Gaussian batches every step instead of reshuffling `4096` fixed points — and I kill
the finite-sample noise floor but not the `d²` escape time: the drift is still `O(m²)` at an `O(1/√d)`
start, so I still need `~16k` steps to leave the equator and I have `8k`. Fresh data lowers the noise
ceiling, not the step-count wall. The giant-step leap — one enormous first-layer step `η ~ √(n/d)` to
cancel the `d^{-(s-1)/2}` smallness in a single shot — is the trick the single-index analysis uses on
fresh population batches, but here the empirical degree-3 correlation on 128 points has sampling std
`~1/√128 ≈ 0.088`, dwarfing the `~5.9·10⁻³` population signal (SNR `≈ 0.07`), so one giant step amplifies
sampling noise, not alignment: a coin flip, not a rotation. So separate the stages *and* run many moderate
steps rather than one giant one, so the alignment accumulates over thousands of mini-batch correlations
that partially average the `0.088` noise, while the closed-form readout solve removes the co-adaptation
that inflated the MSE — the only route that touches both failure causes at once.

Let me build the first-layer phase concretely. Freeze the readout `a` and look at the gradient on a row
`w_i` alone. With squared loss `(1/2)(y − f̂)²`, the row gradient is `g_i = a_i (1/n) Σ_ν x^ν σ'(⟨w_i,
x^ν⟩)(f̂(x^ν) − y^ν)`. The nuisance is `f̂`, the net's own output, polluting the residual. The cleanest
thing would be a residual that is just `−y`, so the gradient is a pure correlation between the
activation-weighted input and the labels — the object Stein's lemma turns into `a_i E[σ'(⟨w_i,x⟩) ∇_x y]`,
a contraction of the teacher's Hermite-3 tensor against the row that rotates `w_i` into `V*`. In this
harness I do not get to engineer a zero-output symmetric initialization the way a from-scratch derivation
would; the scaffold builds the net and hands it to `init_model`, and the readout has to start
*somewhere*. So I do the next best thing the contract allows: spherical init on the first-layer rows
(each `w_i` placed uniformly on `S^{d-1}`, the clean `O(1/√d)`-foot-in-`V*` start), and a *random ±1
sign* readout scaled by `1/√W` — symmetric in expectation, small in magnitude, so it gives stage 1 a
usable signal without dominating it. With `W = 256`, that scale is `1/16 = 0.0625` per unit, so the
readout output `f̂ = Σ_j a_j σ(⟨w_j,x⟩)` starts at `O(√W · (1/√W) · O(1)) = O(1)` but *sign-balanced*,
so its contamination of the residual is second-order in the small aligned part. The first-layer rows are
what I want to rotate; the readout is held essentially fixed while I do it.

How do I hold the readout fixed? The contract gives me `lr_outer`, and I set it to a hair above zero (the
normalizer rejects exactly zero — `lr_outer = 1e-8`), so the optimizer's outer group barely moves. But
"barely" is not "not at all," and a leaking readout would re-introduce the co-adaptation I am trying to
kill. So inside `training_step`, during stage 1, after `loss.backward()` I *zero the output layer's
gradient* — both weight and bias — before `optimizer.step()`. That is the literal enforcement of "freeze
the readout": the inner rows get their squared-loss gradient, the readout gets nothing. This is a real
difference from the abstract two-stage recipe, which fits a clean correlation loss on a zero-output net;
here I am running the *squared-loss* gradient on the live net with the readout pinned, so the residual is
`(f̂ − y)` with `f̂` the small-random-readout output, not `−y`. It is a slightly noisier alignment signal
than the pure correlation, but it is the honest one the harness exposes, and with the readout at scale
`0.0625` the `f̂` contamination is second-order against the label term.

The sign symmetry is what keeps the `f̂` contamination incoherent. In the residual `f̂ − y`, the `f̂` term
contributes `E[a_i σ'(⟨w_i,x⟩) x f̂(x)]`; with the `a_j` i.i.d. `±1/√W`, the diagonal `j = i` piece points
essentially along `w_i` itself (a self-rescaling, no `V*` rotation), and the off-diagonal `j ≠ i` pieces
carry random signs `a_i a_j` that average to an `O(1/√W)` fluctuation with no coherent `V*` component. So
the *only* coherent pull into `V*` comes from the `−y` label term — the correlation I want — and a small
random `±1/√W` readout is a usable stand-in for the zero-output net the clean derivation would use.

Now the step size and the budget split. A from-scratch analysis would take a *giant* step to cancel the
smallness of the leap term in one shot; I argued above I cannot do that safely on finite batches. So
instead of one giant step I take *many* moderate steps. I set `lr_inner = 1e-1`, twice the floor's rate,
and run the first layer for the first 80% of the budget — `0.8 · 8000 = 6400` steps — letting the
alignment accumulate over thousands of mini-batch correlations rather than betting it all on one. That
partial-averaging is exactly the noise reduction I need: `6400` steps at batch 128 touch far more of the
correlation structure than a single batch, and the moderate rate keeps each step from being dominated by
its own `0.088` sampling fluctuation. Why `0.8` and not an even split? The stage-2 ridge solve is a
*single* closed-form operation — it needs one step to run, and the trailing steps just re-report the
frozen train loss — so every step I hand stage 2 beyond the first is wasted, whereas every step I hand
stage 1 buys more averaging on a borderline-SNR signal. Giving stage 1 the lion's share and leaving a
`20%` tail says "spend the budget where the marginal step still moves the subspace, reserve only a sliver
for the solve that does not iterate"; I keep the `1600`-step tail rather than literally one step because
the harness runs `training_step` to the end regardless, and a comfortable margin guarantees the solve has
fired well before evaluation. As for `lr_inner = 1e-1`: at SNR `≈ 0.83` the coherent per-step increment
into `V*` is `~lr · signal ≈ 0.1 · 5.9·10⁻³ ≈ 5.9·10⁻⁴`, small enough that renormalization keeps each
step a near-rotation rather than a jump, yet large enough that `6400` of them can compound; doubling the
floor's rate roughly halves the steps-to-escape without pushing a single step into the regime where the
`0.088` noise dominates the `0.006` signal. Each step also re-normalizes the rows back to the unit sphere:
after the gradient update I divide each row by its norm. This is the load-bearing hygiene from the
spherical picture — I care about the row's *orientation* relative to `V*`, not its length, and
renormalizing stops the repeated steps from inflating `‖w_i‖` and muddying the `‖Π* w_i‖/‖w_i‖` story
(and with it the `subspace_err` SVD, which reads singular directions of `W_in` and would be skewed by a
few blown-up rows). Keeping the rows on the sphere is what lets a long sequence of moderate steps act
like a clean rotation toward `V*` instead of a drift in norm.

There is a quantitative reason to expect this to do *better than the floor but not clean*, and it comes
from the same noise-floor arithmetic that killed rung 1, now recomputed for the larger pool. The floor
averaged its degree-3 correlation over `4096` fixed points, giving a sampling noise `~1/√4096 ≈ 0.0156`
against a signal `~5.9·10⁻³` — SNR `≈ 0.38`, sub-noise, hopeless. Here the enlarged `n = 20{,}000` pool
sets the *effective* independent-sample count of the averaged correlation to `~20000`, so the noise
floor drops to `1/√20000 ≈ 0.0071`. Against the same `~5.9·10⁻³` population signal that is SNR `≈ 0.83`
— right at the boundary. That single number captures the whole hope and limit of this rung: the averaged
alignment signal has climbed from a hopeless `0.38·noise` to a *marginal* `0.83·noise`, borderline
detectable, so stage 1 should pull *some* rows a real fraction toward `V*` but cannot be expected to
resolve the subspace cleanly. The larger pool is doing double duty — it conditions the stage-2 Gram and
it lifts the stage-1 correlation SNR to the edge of detectability — but `0.83` is not `≫ 1`, so I am
buying partial recovery, not a solved subspace.

And even granting the stage works, what it can recover is limited by the same decoupled structure that
sank the floor: with no cross terms there is no staircase to condition on, so stage 1 is really `r`
independent third-order alignment problems run in parallel across `256` rows. Whether any row's degree-3
correlation on finite batches rises above the `0.088` sampling floor inside 6400 steps is genuinely
uncertain. I expect *partial* alignment at best — better than the floor's zero, because freezing the
readout and renormalizing removes the two things that were actively hurting joint SGD, but far from full
recovery, because the degree-3 signal is weak and there is no ladder.

Stage 2 is the part I am confident in, and it is why separating the stages helps the MSE regardless of
how well stage 1 does. For the last 20% of the budget — `1600` steps — I freeze `W_in` entirely and fit
the readout in *closed form*. With the features fixed at `φ(x) = ReLU(W_in x)`, the predictor `f̂ =
a·φ(x) + b` is *linear* in `(a, b)`, so fitting it to least squares is convex — ordinary ridge
regression, no backprop, no co-adaptation, the global optimum of the readout given whatever subspace
stage 1 found. I solve it once: form the post-ReLU feature matrix on a *cached, larger* training set (I
enlarge `make_dataset` to `n = 20{,}000` and stash it globally, since the ridge solve wants far more rows
than the 128-point loop batch to condition the Gram matrix), append a ones column for the bias, and solve
`(ΦᵀΦ + λI) a = Φᵀy`. The shapes make the conditioning argument concrete: `Φ` is `20000 × 257` (256
features plus the bias column), the Gram `ΦᵀΦ` is `257 × 257`, and with `20000 ≫ 257` rows the Gram is
comfortably full-rank — the regime where ridge is stable. I set `λ = 10⁻³·n = 20`, small relative to the
diagonal of `ΦᵀΦ` (each ReLU feature has second moment `O(1)`, so diagonal entries are `~n·O(1) =
O(20000)`), so the regularizer only floors the smallest eigenvalues without biasing the fit. The
`min(20k, max_train_examples)` cap keeps me inside the harness's `200k` limit. After the solve, the
remaining stage-2 steps just report the train loss without touching the net — the readout is already
optimal for the frozen features.

This is the decisive structural difference from the floor, and it sets a clean expectation: stage 2 gives
the readout the *best possible* fit on the features stage 1 produced, so the MSE can only be as good as
the subspace allows but will not be made *worse* by a co-adapting readout the way joint SGD's was. The
lower bound is exact — with frozen features the readout cannot beat the best predictor that depends only
on the learned subspace `U`, `E[(g − f̂)²] ≥ E[Var(g | P_U x)] − o(1)` — so everything still rides on how
much of `V*` stage 1 recovered. Quantify that bound: if the learned subspace captures the teacher
directions to an aggregate fraction `ρ`, the residual variance is roughly `Var[g]·(1 − ρ) = 6(1 − ρ)`.
With near-random `U`, `ρ ~ r/d ≈ 0.02` and the residual is `≈ 5.9` — the MSE cannot beat `~6` by fitting
the link, it can only stop *overshooting* it. `6` is a hard reference line: the best constant predictor
`E[y] = 0` already achieves `Var[y] = 6`, and ridge on `256` near-random ReLU features barely touches the
cubic's degree-3 eigenspace, so the MSE sits just *below* `6` only if stage 1 rotated the features into
`V*` and essentially *at* `6` if it did not. Landing a few tenths under it means real (if partial)
alignment; landing at or just above means the gain was pure variance de-inflation from killing the
co-adaptation. Stage 1 sets the ceiling, stage 2 reaches it, and if stage 1 finds nothing the best stage 2
can do is haul the MSE back down from the floor's inflated `9`–`10` to near `6`, no further.

So the expectations against the floor's numbers. `subspace_err` should drop below `√(2r)` — removing the
co-adaptation and norm-inflation that pinned joint SGD there, partial degree-3 alignment should pull the
projector distance down, most where there are fewest cold directions to find (r2). `test_mse` should fall
below the floor's 9–10 toward but not all the way to `Var[g] = 6`, since the `6(1 − ρ)` bound caps the
gain at whatever fraction `ρ` of `V*` stage 1 recovered. `score` should rise roughly an order of magnitude
into the `1·10⁻⁴` range, driven mostly by the MSE recovery, since the subspace exponential stays near
`exp(−2)` while the subspace is far from clean. The sharp branch: if `subspace_err` stays pinned at
`√(2r)` and only the MSE moves, stage 1's degree-3 climb found *nothing* and the whole gain is convex
readout fitting on random features — which would force me to abandon gradient-based alignment and find the
subspace some way that does not depend on climbing a degree-3 ladder that is not there, since the wall is
the gradient *method*, not the problem's information content. The full two-stage module is in the answer.
