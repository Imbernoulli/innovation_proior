The vanilla-SGD run told me exactly what I expected and gave me the split in numbers. On `relu-d100`
all three seeds pinned at `direction_recovery ~ 0.998` (`0.998`, `0.998`, `0.998`) with `test_mse ~
0.0017` — the `k=1` easy regime where ordinary SGD already finds the direction, and where I should not
expect to do better. The two hard links are where the floor showed. On `hermite-d100` (`k=3`) recovery
averaged `0.656`, with per-seed values {`0.615`, `0.736`, `0.616`} — the network found *some* alignment
but landed far short of 1, well above the `0.1` chance floor yet nowhere near the pole, and `test_mse`
stayed at `0.16`, two orders of magnitude above the relu floor. And `sign-d100` was the real failure:
recovery {`0.049`, `0.156`, `0.426`} for a mean of `0.210`, with seed 42 sitting *at* the `0.1` chance
level — that run recovered essentially nothing — and `test_mse ~ 0.44`, close to the `Var(sign) = 1`
trivial-predictor ceiling. So the diagnosis is confirmed and sharp.

Let me read the spreads, not just the means, because the shape of the variance is the real tell. On
relu the three seeds agree to the fourth decimal — the easy landscape funnels every run to the same
pole, variance essentially zero. On hermite the range is `0.736 - 0.615 = 0.12` around a mean of `0.66`,
a coefficient of variation of about `0.09`: moderate spread, every run stuck in the same mediocre band.
On sign the range is `0.426 - 0.049 = 0.38` — *larger than the mean itself* — with a coefficient of
variation near `0.75`. That is not a link the recipe solves badly and consistently; it is a lottery. And
a lottery is exactly the signature of a landscape with no benign structure: outcome depends on whether a
run's random rows happened to carry enough early overlap before the head locked onto them, precisely the
coupling stall I sketched at the control. So the two hard links fail for two *different* reasons that the
numbers now let me separate: hermite is a signal problem (consistent low recovery, moderate spread — the
`k=3` direction signal is below the mini-batch noise on every seed alike), sign is a landscape problem
(wild spread — the `k=1` signal is present but the rough, entangled landscape lets each run settle
wherever its lucky rows started). This is not a learning-rate problem: the same SGD that nails relu
cannot move the rows on either hard link, and no step size fixes a coin flip or a lottery.

The `test_mse` column corroborates the split and rules out the lazy alternative explanation that the net
simply lacks capacity. Against the trivial-predictor ceilings `Var(y)` — `0.34` for relu, `1` for sign,
`1` for hermite — relu's `0.0017` is a near-perfect fit, hermite's `0.16` explains most of the variance
but leaves a sixth unexplained, and sign's `0.44` leaves nearly half. Now, `W = 256` ReLU features are
ample to represent a cubic-order link or a step in one dimension, so the residual on the hard links is
not a capacity shortfall — it is the loss registering a *direction* that was never found: a head fitting
features whose rows point mostly off `theta*` cannot drive the noise-free test error down, no matter how
convex the head-fit is. That is the tell that this is a first-layer problem, and it is why my fix lives
in what the first layer's parameters are allowed to do, not in the head or the width.

So the fix I want is structural, not a knob: reshape the optimisation landscape so that the direction
search stops fighting the link fit. Let me reason from the geometry. The trouble is that the first layer
has `W = 256` rows, each a free vector in `R^d`, and each one is simultaneously asked to (i) point
toward `theta*` and (ii) build, together with the head, a basis rich enough to represent the univariate
link `g`. Those are different jobs on different scales — the direction is a needle in `d` dimensions, the
link is a one-dimensional function — and entangling them is what wrecks the rate. I want to *decouple*
them inside the same fixed two-layer ReLU net, using only the levers the harness exposes: how I
initialise `fc1` and `fc2`, what I freeze, and what the optimiser touches.

Here is the structural observation that does it. A hidden ReLU neuron computes `ReLU(<w_j, x> + b_j)`.
Two things parameterise it: the *direction* `w_j` (where it looks in input space) and the *bias* `b_j`
(the threshold along that direction). Now think about what each is *for* in a single-index model. The
non-parametric job — approximating the univariate link `g(u)` for `u = <theta*, x>` — is exactly the job
of a *spread of thresholds* along the relevant direction: a bank of functions `{ReLU(u - b_j)}` with
varied `b_j` is a one-dimensional spline basis that can fit any reasonable `g`. That is a *kernel* job, a
random-feature job; the biases are the random-feature sampling of that one-dimensional kernel. The
high-dimensional job — finding `theta*` — is the job of the *directions* `w_j`. So the two jobs live in
two different sets of parameters, and the lesson of lazy-versus-rich training (Chizat, Oyallon & Bach
2019) is exactly that a part of the model that stays at initialisation and acts like a fixed kernel is
the lazy part, while a part that moves and learns features is the rich part. The non-parametric link fit
should be lazy; the direction search should be rich. The clean move, then, is to **freeze the biases at
their random init and train only the directions and the head**. That is the one essential change to the
recipe.

Let me check that freezing the biases actually collapses the landscape, because if it does that is the
justification, not an analogy. Write the population loss `L = E[(G(x) - y)^2]` for the network
`G(x) = sum_j a_j ReLU(<w_j, x> + b_j)`, and decompose against the Gaussian in Hermite polynomials. The
key identity is that a degree-`p` feature along a direction `w` overlaps the target's degree-`p`
component by exactly `<w, theta*>^p`. Now decompose a single row against the truth: `w_j = m_j theta* +
w_j^perp` with `m_j = <w_j, theta*>`, so the pre-activation is `<w_j, x> = m_j u + <w_j^perp, x>` where
`u = <theta*, x>`. If the biases `b_j` are *frozen*, they do not depend on the directions, so `w_j`
enters the loss only through this decomposition, and by rotational symmetry of the isotropic Gaussian
the perpendicular part `w_j^perp` enters *only through its norm* — there is no other privileged direction
for it to align with, because `theta*` is the sole signal direction. So the loss depends on each row
through just two scalars, `(m_j, ||w_j^perp||)`, and its gradient lives entirely in the two-plane
`span{theta*, w_j}`. The only *truth-informative* component of that gradient is the one along `theta*`:
it pushes `m_j` up or down, rotating the row toward or away from the pole. The residual component merely
rescales the row's own perpendicular length, a nuisance coordinate with no preferred target. Keep the
rows on the sphere and that nuisance is pinned, so the whole high-dimensional search collapses to a
scalar flow in the overlaps `m_j`. *That* is what freezing the biases buys, and it is precisely what was
missing under vanilla SGD: there the biases were also trained, so `b_j` co-adapted with the directions,
opening a second coupling channel — the threshold moving to fit the link locally while the direction was
still near the equator — that dragged the non-parametric problem back into the high-dimensional
dynamics and re-entangled the two jobs I now want separate.

In the ideal limit (infinitely many, unregularised random-feature thresholds) the projected loss is
strictly decreasing in `|m|`, with only two kinds of critical point: the equator `m = 0` and the poles
`m = +-1`. No spurious local minima — gradient flow slides from the random start up to a pole. With
finitely many *frozen* random biases the picture is preserved as long as the random-feature
approximation error of the link is small; the bias variance has to be large enough that the
one-dimensional random-feature space is rich enough to approximate smooth links with a polynomial rate.
So the biases should be drawn with `O(1)` spread, not tiny — wide enough thresholds to span the range of
`u = <theta*, x> ~ N(0,1)`.

Let me pin the bias scale with arithmetic rather than leave it at "O(1)", because it is a real choice and
I have `W = 256` thresholds to place. The relevant coordinate is `u = <theta*, x> ~ N(0, 1)`, whose bulk
lives in `[-1, 1]` — that interval holds `erf(1/sqrt2) ~ 0.68` of the mass, i.e. about two-thirds of all
inputs, with the tails beyond `+-2` carrying under five percent. A ReLU threshold bank `{ReLU(u - b_j)}`
resolves the link exactly where its knots `b_j` sit: knots in `[-1, 1]` give fine resolution across the
high-mass core, and `256` of them there means a knot spacing of `~ 2/256 ~ 0.008 sigma`, far finer than
any smooth link needs. If instead I spread the same `256` knots over `[-3, 3]` I would be spending
resolution on the sparse tails — spacing `~ 0.023 sigma` in the core, three times coarser where almost
all the mass and almost all the link's action are. So a wide-but-not-extreme spread is right, and
`Uniform(-1, 1)` places the thresholds squarely on the core. I draw `fc1.bias ~ Uniform(-1, 1)` and
freeze it.

Now ground this in the harness exactly, because the harness does *not* give me the full apparatus the
cleanest analysis uses, and I have to be honest about what I keep and what I drop. The fixed
`TwoLayerMLP` is a *wide* net — `Linear(d, 256)` with all 256 rows independent, then ReLU, then
`Linear(256, 1)`. The analysis I just sketched is cleanest for a *tied* architecture (one shared inner
direction, neurons differing only in bias and sign) with a time-scale separation (search the direction
first with a small/sparse readout, then turn the head on) and a final fresh-sample readout refit. The
harness exposes none of that cleanly: I cannot tie the rows (the model is fixed wide), I am handed
mini-batches not a controllable two-phase schedule, and a separate closed-form refit is a move I am
deliberately *not* taking at this rung. So I keep the *one essential move* — freeze the biases — and
apply it to the standard wide MLP, accepting that the wide net's many independent rows are a noisier
realisation of the collapsed landscape than the tied net would be. Concretely the recipe becomes:
initialise the first-layer rows on the unit sphere; draw the biases `~ Uniform(-1, 1)` and *freeze* them
with `requires_grad_(False)`; init the readout small and uniform as before; build the optimiser over
*only the parameters with `requires_grad`*, so the frozen biases are excluded automatically; and run the
same SGD-with-momentum mean-squared-error mini-batch loop. No finalize.

Two grounding details matter, and the first is more than cosmetic. Putting `fc1.weight` on the unit
sphere rather than at Kaiming's `sqrt(2/d)` scale does two jobs at once. Obviously, with the biases
frozen the only thing the optimiser controls in the first layer is the *direction* of each row, so I
want every row to start as a comparable unit probe — Kaiming's rows have norm `~ sqrt(2) ~ 1.41` and
would let row norms drift and muddy the "direction-only" reading. But the subtler job is scale-matching
the pre-activation to the frozen threshold. With a unit row, `<w_j, x> ~ N(0, 1)`, so the pre-activation
and the frozen bias `b_j ~ Uniform(-1, 1)` live on the *same* scale — the threshold bites right in the
bulk of the activation, and the ReLU actually switches on and off across the data. With a Kaiming row of
norm `sqrt(2)`, the pre-activation has standard deviation `1.41`, so a bias of `+-1` only reaches
`+-0.7` standard deviations of the activation — the frozen thresholds would sit too close to the centre,
under-covering the tails and wasting the bank's dynamic range. So the unit-sphere init is what makes the
frozen `Uniform(-1, 1)` biases a *usable* one-dimensional basis, not merely a tidy one. The second
detail is the freeze mechanics: set `requires_grad_(False)` on the bias tensor and construct the
optimiser from `[p for p in net.parameters() if p.requires_grad]`; the harness's direction estimator
`normalize(sum_j |a_j| w_j)` then reads off the readout-weighted rows, which now move *only* in
direction.

Before I commit I should rule out the tempting alternative, which is to freeze the *directions* instead
of the biases — the pure random-feature / kernel recipe of Rahimi & Recht and Jacot et al., train only
the head on fixed random features. It is worth a moment because it is the exact opposite freeze and it
looks like it would also "collapse the landscape" by making the head-fit convex. But it collapses the
wrong thing: if `fc1.weight` never moves, the rows stay at their random init forever, and the direction
estimator `normalize(sum_j |a_j| w_j)` reads a readout-weighted sum of *fixed random* rows, pinned at
overlap `~ 1/sqrt(d) = 0.1` — the chance floor — no matter how well the head fits. A random-feature
model can push `test_mse` down by memorising through `256` fixed features but structurally *cannot*
recover `theta*`, because recovery lives in the directions and I would have frozen exactly those. So the
freeze has to be on the biases, keeping the directions free: that is the only split that leaves the
high-dimensional search alive while lazifying the one-dimensional link fit. Freezing the biases removes
`W = 256` trainable scalars and leaves the `256 x 100` direction matrix and the head trainable — the
opposite budget from the random-feature model, and the correct one.

Let me trace the collapsed flow on a single sign row to be sure the mechanism is real and not just a
relabelling. Start a unit row at overlap `m = 0.1`. On a `k=1` link the informative gradient component
along `theta*` is `O(mu_1) = O(0.8)` and, crucially, it does not vanish as `m -> 0` — the drift is
present at the equator, which is what `k=1` means. So each step adds a clean push toward the pole of size
order `mu_1` times the step, and because the frozen-bias landscape has stripped away the perpendicular
"other directions" the row could have wandered into, that push is not diluted across `d` competing
gradients — it goes straight into `m`. The `256`-sample noise still jitters the step by `~ 0.625`, but
now the jitter is around a *nonzero mean drift* rather than around a sub-threshold one, so the row
ratchets up: `m` climbs `0.1 -> 0.2 -> ...` toward `1`, and does so from *any* start because the only
critical points left are the equator (unstable) and the poles (stable). That is the scalar flow the
Hermite decomposition promised, seen on one row. The contrast with hermite is entirely in the drift
size: there the along-`theta*` push scales as `m^2`, so near the equator it is `~ 0.01`, back below the
noise, and the same clean landscape cannot ratchet a row it cannot budge.

There is a second reason the wide net should read out well on sign once the flow works, and it is worth
computing because it is where the `W = 256` rows finally help rather than hurt. The estimator is
`theta_hat = normalize(sum_j |a_j| w_j)`. Suppose the collapsed flow brings *every* row to a common
overlap `m` with `theta*` — same pole — plus an independent perpendicular residual of norm
`sqrt(1 - m^2)`. Then the along-`theta*` part of the sum is coherent, `(sum_j |a_j|) m`, while the `256`
perpendicular residuals point in independent directions and partially cancel, shrinking by `~ sqrt(256) =
16`. The estimator overlap is then `m / sqrt(m^2 + (1 - m^2)/256)`, which for a modest per-row `m = 0.5`
already evaluates to `~ 0.994`, and for `m = 0.3` to `~ 0.98`. So the width is an *amplifier*: even
loosely aligned rows, if they share a pole, aggregate into a near-perfect direction read. But this
amplifier has a precondition — the rows must agree on *which* pole. If the lottery sends some rows to
`+theta*` and others to `-theta*`, the `|a_j|` weighting cannot rescue it (the head flips sign through
`a_j`, but `|a_j|` keeps both contributions positive in the row sum only if the rows themselves agree),
and the coherent term collapses. That is exactly why the seed spread, not just the mean, is the quantity
to watch on sign: the collapsed landscape's whole job is to make all `256` rows pick the same pole every
run, at which point the width converts modest per-row alignment into high recovery.

Let me now state the falsifiable expectations against the vanilla-SGD numbers, link by link, because that
is what tells me whether the landscape collapse is real here.

On `relu-d100` (`k=1`) I expect *no change* — vanilla SGD already hit `0.998`, the easy regime saturates
regardless of how the biases are handled, so frozen-bias should also land near `0.998`. If it dropped
here, the unit-sphere init or the frozen biases would be hurting the easy case, which would be a red
flag that the scale-matching argument is wrong.

On `hermite-d100` (`k=3`) this is the interesting test. The landscape-collapse argument says freezing the
biases makes the informative part of each row's gradient colinear with `theta*`, which should help the
direction search *when the per-step signal is present*. But the collapse does nothing about the signal's
*size*: on hermite that signal is still `~ m^2 ~ d^{-1} ~ 0.01` per step, still buried under the
`256`-sample noise `~ 0.625`, at the same SNR `~ 0.016` as before, because a cleaner landscape reshapes
where the gradient points, not how loud it is. I have not added the one thing that would change the
size — aggregating the weak signal over far more than `256` samples — so I do not expect a dramatic jump.
The honest prediction is a *modest* improvement or a wash relative to vanilla SGD's `0.656`: the
collapsed landscape is cleaner, but the wide net's many independent rows, driven by the same noisy
`256`-sample gradients, still cannot surface a signal forty times below their noise. If frozen-bias
lands roughly *at or slightly below* vanilla SGD on hermite, that is consistent with the diagnosis — the
missing piece is not the bias freeze but the signal aggregation. If it jumped to near 1, the freeze
alone would be doing the whole job, which the SNR arithmetic says it cannot.

On `sign-d100` (`k=1`, non-smooth) I expect the clearest *gain* from this rung. The link is `k=1`, so the
direction signal is present at first order (`mu_1 = sqrt(2/pi) ~ 0.8`, if anything larger than relu's
`0.5`) — the problem under vanilla SGD was never the signal size but the rough, entangled landscape and
the joint dynamics fitting the discontinuous target locally with unaligned rows (recovery as low as
`0.049`, dead on the chance floor). Collapsing the landscape by freezing the biases should make the
direction search far more reliable here: every run should be pulled toward the same pole rather than
landing wherever its random rows happened to start. So I expect sign recovery to rise meaningfully above
vanilla's `0.210` mean and, importantly, to *tighten* across seeds — the `0.38`-wide spread {`0.049`,
`0.156`, `0.426`} should shrink toward the tight agreement I see on relu, as the benign landscape removes
the dependence on lucky init. That tightening is the specific thing I am betting on; if instead the
spread stays wide or grows, it would mean the wide net's independent rows re-introduce a lottery that the
tied-architecture analysis does not have, and that the landscape collapse is only loosely realised on a
`256`-row net.

So the delta from step 1 is one line of substance — freeze the biases (and switch to a unit-sphere
first-layer init so the rows are clean, scale-matched direction-only probes) — and the prediction is
asymmetric: relu unchanged at the ceiling, sign improved and tightened because `k=1` only ever needed a
clean landscape, hermite roughly flat because the freeze collapses the landscape but does not aggregate
the weak `k=3` signal. Whatever the precise numbers, the structure they reveal points straight at step 3:
if hermite stays stuck, the remaining problem is that the third-order direction signal is below the
mini-batch noise, and the fix is to stop crawling with noisy `256`-sample steps and instead sum the weak
signal over the entire training set in one aggregated pass — the full-batch signal aggregation the
`256`-sample loop can never do — surfacing a `k=3` signal that mini-batch SGD, however clean its
landscape, will always leave on the table.
