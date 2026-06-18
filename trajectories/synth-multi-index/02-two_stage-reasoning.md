The floor came back exactly as the information-exponent argument said it would, and the numbers are
worth reading closely because they tell me *which* failure I am looking at. Vanilla joint SGD left
`subspace_err` at 1.95 (r2), 2.37 (r3), 2.74 (r4) — essentially `√(2r)` = 2.0/2.4/2.8 to three figures.
That is the value two rank-`r` orthogonal projectors take when they share *no* directions: the
first-layer row-span and the teacher subspace are still uncorrelated after 8000 steps. The first layer
did not move toward `V*` at all. And `test_mse` sat at 10.4/9.8/9.1 — *above* the `Var[g] = 6` label
variance I expected as the do-nothing floor, which means the untamed joint dynamics is not merely
failing to fit; it is injecting prediction variance on top of the variance it cannot explain. The
`score` is `5e-6`/`1.4e-5`/`1.7e-5`, three to four orders below anything useful. This is the `d²`-wall
seed-for-seed: per-seed `subspace_err` barely fluctuates (1.93–1.99 on r2), so it is not a variance
problem I can fix with more seeds or a luckier init — it is a *structural* dead end. Joint SGD has no
mechanism to amplify a degree-3 correlation inside the budget, and the readout, chasing near-random
features, is actively making the MSE worse.

So the diagnosis is sharp: this is an optimization failure with two intertwined causes, and I should
attack both. First, the net is conflating two jobs — finding `V*` (nonconvex, the hard part) and
fitting the link (convex, once the features are good) — and training them jointly means the readout is
fitting moving, near-random features the whole time, which is exactly why the MSE overshoots its own
floor. Second, the first-layer signal is third-order-weak and a single `lr = 5e-2` step cannot move a
row `O(1)` into `V*`. The cure for the first is to *separate the stages*: train the first layer alone to
align with `V*`, then freeze it and solve the readout in closed form. The cure for the second is to
make the first-layer phase as clean and well-conditioned as I can so whatever alignment signal exists
is not diluted by a co-adapting readout or by runaway row norms.

Let me build the first-layer phase. Freeze the readout `a` and look at the gradient on a row `w_i`
alone. With squared loss `(1/2)(y − f̂)²`, the row gradient is `g_i = a_i (1/n) Σ_ν x^ν σ'(⟨w_i,
x^ν⟩)(f̂(x^ν) − y^ν)`. The nuisance is `f̂`, the net's own output, polluting the residual. The cleanest
thing would be a residual that is just `−y`, so the gradient is a pure correlation between the
activation-weighted input and the labels — the object Stein's lemma turns into a contraction of the
teacher's Hermite tensor against the row, the part that rotates `w_i` into `V*`. In this harness I do
not get to engineer a zero-output symmetric initialization the way a from-scratch derivation would; the
scaffold builds the net and hands it to `init_model`, and the readout has to start *somewhere*. So I do
the next best thing the contract allows: spherical init on the first-layer rows (each `w_i` placed
uniformly on `S^{d-1}`, the clean `O(1/√d)`-foot-in-`V*` start), and a *random ±1 sign* readout scaled
by `1/√W` — symmetric in expectation, small in magnitude, so it gives stage 1 a usable signal without
dominating it. The first-layer rows are what I want to rotate; the readout is held essentially fixed
while I do it.

How do I hold the readout fixed? The contract gives me `lr_outer`, and I set it to a hair above zero
(the normalizer rejects exactly zero), so the optimizer's outer group barely moves. But "barely" is not
"not at all," and a leaking readout would re-introduce the co-adaptation I am trying to kill. So inside
`training_step`, during stage 1, after `loss.backward()` I *zero the output layer's gradient* — both
weight and bias — before `optimizer.step()`. That is the literal enforcement of "freeze the readout":
the inner rows get their squared-loss gradient, the readout gets nothing. This is a real difference from
the abstract two-stage recipe, which fits a clean correlation loss on a zero-output net; here I am
running the *squared-loss* gradient on the live net with the readout pinned, so the residual is `(f̂ −
y)` with `f̂` the small-random-readout output, not `−y`. It is a slightly noisier alignment signal than
the pure correlation, but it is the honest one the harness exposes, and with the readout small the
`f̂` contamination is second-order.

Now the step size. A from-scratch analysis would take a *giant* step `η ~ p√(n/d)` to cancel the
`d^{-(s-1)/2}` smallness of the leap term in one shot. I cannot do that safely here: the budget is 8000
steps on a *fixed* `n = 4096`-scale dataset reshuffled into batches of 128, not fresh population batches,
so a giant single step on a finite batch would amplify sampling noise, not signal — the empirical
degree-3 correlation on 128 points is dominated by fluctuations unless I average. So instead of one
giant step I take *many* moderate steps. I set `lr_inner = 1e-1`, twice the floor's rate, and run the
first layer for the first 80% of the budget (6400 steps), letting the alignment accumulate over
thousands of mini-batch correlations rather than betting it all on one. Each step also re-normalizes the
rows back to the unit sphere: after the gradient update I divide each row by its norm. This is the
load-bearing hygiene from the spherical picture — I care about the row's *orientation* relative to `V*`,
not its length, and renormalizing stops the repeated steps from inflating `‖w_i‖` and muddying the
`‖Π* w_i‖/‖w_i‖` story (and with it the `subspace_err` SVD). Keeping the rows on the sphere is what lets
a long sequence of moderate steps act like a clean rotation toward `V*` instead of a drift in norm.

The trouble I have to be clear-eyed about: even granting this stage works, what can it recover? The
single-step leap analysis says one giant step into `V_ℓ*` reaches only the directions present *at the
leap order*. Here the leap order is 3 and the link is a *decoupled* `Σ He₃` with no cross terms, so
there is no staircase — each teacher direction must be picked up cold from its own degree-3 correlation,
and there is no lower-degree rung to expose the next one by conditioning. So the multi-step climb that
rescues `ℓ=1` staircases (condition on a found direction, the next becomes first-order visible) has
*nothing to climb* here. My moderate-step stage 1 is really `r` independent third-order alignment
problems run in parallel across `256` rows, and whether any row's degree-3 correlation on finite batches
rises above the sampling floor inside 6400 steps is genuinely uncertain. I expect *partial* alignment
at best — better than the floor's zero, because freezing the readout and renormalizing removes the two
things that were actively hurting joint SGD, but far from full recovery, because the degree-3 signal is
still weak and there is no ladder.

Stage 2 is the part I am confident in, and it is why separating the stages helps the MSE regardless of
how well stage 1 does. For the last 20% of the budget I freeze `W_in` entirely and fit the readout in
*closed form*. With the features fixed at `φ(x) = ReLU(W_in x)`, the predictor `f̂ = a·φ(x) + b` is
*linear* in `(a, b)`, so fitting it to least squares is convex — ordinary ridge regression, no backprop,
no co-adaptation, the global optimum of the readout given whatever subspace stage 1 found. I solve it
once: form the post-ReLU feature matrix on a *cached, larger* training set (I enlarge `make_dataset` to
`n = 20{,}000` and stash it globally, since the ridge solve wants more rows than the 128-point loop
batch to condition the Gram matrix), append a ones column for the bias, and solve `(ΦᵀΦ + λI) a = Φᵀy`
with `λ = 10⁻³·n`. The `min(20k, max_train_examples)` cap keeps me inside the harness's `200k` limit.
After the solve, the remaining stage-2 steps just report the train loss without touching the net — the
readout is already optimal for the frozen features.

This is the decisive structural difference from the floor, and it sets a clean expectation: stage 2
gives the readout the *best possible* fit on the features stage 1 produced, so the MSE can only be as
good as the subspace allows but will not be made *worse* by a co-adapting readout the way joint SGD's
was. The lower bound is exact — with frozen features the readout cannot beat the best predictor that
depends only on the learned subspace `U`, `E[(g − f̂)²] ≥ E[Var(g | P_U x)] − o(1)` — so everything
still rides on how much of `V*` stage 1 recovered. The two stages are not independent knobs: stage 1
sets the ceiling, stage 2 reaches it.

So the falsifiable expectations against the floor's numbers. First, `subspace_err` should *drop below*
the floor's `√(2r)`: freezing the readout and renormalizing the rows removes the co-adaptation and
norm-inflation that pinned joint SGD at exactly `√(2r)`, so even partial degree-3 alignment should pull
the projector distance down — I expect somewhere in the 1.5–2.6 band rather than locked at
2.0/2.4/2.8, with the most movement on r2 (one fewer cold direction to find) and the least on r4.
Second, `test_mse` should fall *below* the floor's 9–10 toward (but not all the way to) `Var[g] = 6`,
because the closed-form ridge solve stops the readout from inflating the prediction variance even on
imperfect features — a few points below 6 if any subspace was found, near 6 if essentially none was.
Third, `score`, being the product of two exponentials in those two quantities, should rise by roughly
an order of magnitude over the floor's `~1e-5` — into the `1e-4` range — driven mostly by the MSE
recovery, since the subspace is still far from clean. If instead `subspace_err` stays pinned at `√(2r)`
and only the MSE moves, that tells me stage 1's degree-3 climb found *nothing* and the entire gain is
the convex readout fit on random features — which would say the next rung must abandon gradient-based
alignment for the cubic and estimate the subspace by a *moment* method that sees degree-3 structure
directly, rather than waiting for SGD to climb a ladder that is not there. The full two-stage module —
spherical init, frozen-readout stage 1 with spherical reprojection, and the stage-2 ridge solve — is in
the answer.
