The vanilla-SGD run gave me the split in numbers. On `relu-d100` all three seeds pinned at
`direction_recovery ~ 0.998` with `test_mse ~ 0.0017` — the `k=1` easy regime where ordinary SGD
already finds the direction and where I should not expect to do better. The two hard links are where
the floor showed. On `hermite-d100` (`k=3`) recovery averaged `0.656`, per-seed {`0.615`, `0.736`,
`0.616`} — some alignment, but far short of 1 and well above the `0.1` chance floor, with `test_mse ~
0.16`, two orders of magnitude above the relu floor. And `sign-d100` was the real failure: recovery
{`0.049`, `0.156`, `0.426`} for a mean of `0.210`, with seed 42 sitting *at* the `0.1` chance level —
that run recovered essentially nothing — and `test_mse ~ 0.44`, close to the `Var(sign) = 1`
trivial-predictor ceiling.

The spreads, not just the means, are the real tell. On relu the seeds agree to the fourth decimal —
the easy landscape funnels every run to the same pole. On hermite the range is `0.736 - 0.615 = 0.12`
around a mean of `0.66`, a coefficient of variation of about `0.09`: moderate spread, every run stuck
in the same mediocre band. On sign the range is `0.426 - 0.049 = 0.38` — *larger than the mean itself*
— a coefficient of variation near `0.75`. That is not a link the recipe solves badly and consistently;
it is a lottery, the signature of a landscape with no benign structure, where the outcome depends on
whether a run's random rows carried enough early overlap before the head locked onto them. So the two
hard links fail for two *different* reasons the numbers now separate: hermite is a signal problem
(consistent low recovery — the `k=3` direction signal is below the mini-batch noise on every seed
alike), sign is a landscape problem (wild spread — the `k=1` signal is present but the rough, entangled
landscape lets each run settle wherever its lucky rows started). Neither is a learning-rate problem:
the same SGD that nails relu cannot move the rows on either hard link, and no step size fixes a coin
flip or a lottery.

The `test_mse` column rules out the lazy explanation that the net simply lacks capacity. Against the
`Var(y)` ceilings — `0.34` for relu, `1` for sign, `1` for hermite — relu's `0.0017` is a near-perfect
fit, hermite's `0.16` leaves a sixth unexplained, sign's `0.44` nearly half. But `W = 256` ReLU
features are ample to represent a cubic-order link or a step in one dimension, so the residual on the
hard links is not a capacity shortfall — it is the loss registering a *direction* that was never found:
a head fitting features whose rows point mostly off `theta*` cannot drive the noise-free test error
down, however convex the head-fit is. This is a first-layer problem, so my fix lives in what the first
layer's parameters are allowed to do, not in the head or the width.

I want a structural fix, not a knob: reshape the landscape so the direction search stops fighting the
link fit. The first layer has `W = 256` rows, each a free vector in `R^d`, and each is asked to do two
jobs at once — point toward `theta*`, and build with the head a basis rich enough to represent the
univariate link `g`. Those are different jobs on different scales — a needle in `d` dimensions versus a
one-dimensional function — and entangling them is what wrecks the rate. I want to decouple them inside
the same fixed two-layer net, using only the levers the harness exposes: how I initialise `fc1` and
`fc2`, what I freeze, and what the optimiser touches.

Here is the structural observation. A hidden ReLU neuron computes `ReLU(<w_j, x> + b_j)`, parameterised
by a *direction* `w_j` and a *bias* `b_j` (the threshold along that direction). In a single-index model
the non-parametric job — approximating `g(u)` for `u = <theta*, x>` — is exactly a *spread of
thresholds* along the relevant direction: a bank `{ReLU(u - b_j)}` with varied `b_j` is a
one-dimensional spline basis that can fit any reasonable `g`. That is a kernel job; the biases are the
random-feature sampling of that one-dimensional kernel. The high-dimensional job — finding `theta*` —
is the job of the directions `w_j`. So the two jobs live in two different sets of parameters, and the
lesson of lazy-versus-rich training (Chizat, Oyallon & Bach 2019) is that a part that stays at
initialisation and acts like a fixed kernel is the lazy part, while a part that moves and learns
features is the rich part. The non-parametric link fit should be lazy; the direction search should be
rich. The clean move is to **freeze the biases at their random init and train only the directions and
the head** — the one essential change to the recipe.

Freezing the biases collapses the landscape, and that is the justification rather than an analogy.
Write the population loss for `G(x) = sum_j a_j ReLU(<w_j, x> + b_j)` and decompose a single row
against the truth: `w_j = m_j theta* + w_j^perp` with `m_j = <w_j, theta*>`, so `<w_j, x> = m_j u +
<w_j^perp, x>`. If the biases are frozen they do not depend on the directions, so `w_j` enters the loss
only through this decomposition, and by rotational symmetry of the isotropic Gaussian the perpendicular
part enters *only through its norm* — there is no other privileged direction for it to align with,
because `theta*` is the sole signal direction. So the loss depends on each row through just two scalars
`(m_j, ||w_j^perp||)`, and its gradient lives entirely in the two-plane `span{theta*, w_j}`. The only
truth-informative component pushes `m_j` up or down, rotating the row toward or away from the pole; the
residual component merely rescales the row's perpendicular length, a nuisance with no preferred target.
Keep the rows on the sphere and that nuisance is pinned, so the whole high-dimensional search collapses
to a scalar flow in the overlaps `m_j`. That is precisely what was missing under vanilla SGD: there the
biases were *also* trained, so `b_j` co-adapted with the directions, opening a second coupling channel
— the threshold moving to fit the link locally while the direction was still near the equator — that
re-entangled the two jobs.

In the ideal limit (infinitely many unregularised random-feature thresholds) the projected loss is
strictly decreasing in `|m|`, with only two kinds of critical point: the equator `m = 0` and the poles
`m = +-1`. No spurious local minima — gradient flow slides from the random start up to a pole. With
finitely many frozen random biases the picture holds as long as the random-feature approximation error
of the link is small, which needs the bias variance large enough that the one-dimensional feature space
can approximate smooth links. So the biases want `O(1)` spread, wide enough to span the range of `u =
<theta*, x> ~ N(0,1)`.

That leaves a real choice of scale, and I have `256` thresholds to place. The coordinate `u ~ N(0, 1)`
has its bulk in `[-1, 1]`, which holds `erf(1/sqrt2) ~ 0.68` of the mass, with the tails beyond `+-2`
carrying under five percent. A ReLU bank `{ReLU(u - b_j)}` resolves the link where its knots sit:
`256` knots in `[-1, 1]` give a spacing of `~ 2/256 ~ 0.008 sigma`, far finer than any smooth link
needs; spread the same knots over `[-3, 3]` and I waste resolution on the sparse tails, `~ 0.023
sigma` in the core where almost all the mass and the link's action live. So `fc1.bias ~ Uniform(-1,
1)`, frozen — a wide-but-not-extreme spread on the core.

I have to be honest that the harness does not give me the full apparatus the cleanest analysis uses.
The fixed `TwoLayerMLP` is a *wide* net — `Linear(d, 256)` with all `256` rows independent — whereas
the cleanest scalar-flow analysis is for a *tied* architecture (one shared inner direction, neurons
differing only in bias and sign), with a time-scale separation (search the direction first with a
sparse readout, then turn the head on) and a fresh-sample readout refit. The harness exposes none of
that cleanly: I cannot tie the rows, I am handed mini-batches not a two-phase schedule, and a separate
closed-form refit is a move I am deliberately not taking at this rung. So I keep the one essential move
— freeze the biases — and apply it to the standard wide MLP, accepting that its many independent rows
are a noisier realisation of the collapsed landscape than the tied net would be. Concretely: put the
first-layer rows on the unit sphere; draw the biases `~ Uniform(-1, 1)` and freeze them with
`requires_grad_(False)`; init the readout small and uniform; build the optimiser over only the
`requires_grad` parameters (excluding the frozen biases automatically); run the same SGD-with-momentum
mean-squared-error loop. No finalize.

The unit-sphere first-layer init is more than cosmetic, and it does two jobs. With the biases frozen
the optimiser controls only the *direction* of each row, so I want every row to start as a comparable
unit probe — Kaiming's rows have norm `~ sqrt(2)` and would let row norms drift and muddy the
direction-only reading. The subtler job is scale-matching the pre-activation to the frozen threshold:
with a unit row `<w_j, x> ~ N(0, 1)`, so the pre-activation and the frozen bias `~ Uniform(-1, 1)`
live on the *same* scale — the threshold bites in the bulk of the activation and the ReLU switches on
and off across the data. With a Kaiming row of norm `sqrt(2)` the pre-activation has std `1.41`, so a
bias of `+-1` reaches only `+-0.7` std — the frozen thresholds would sit too close to the centre,
under-covering the tails and wasting the bank's dynamic range. So the unit-sphere init is what makes
the frozen `Uniform(-1, 1)` biases a *usable* one-dimensional basis, not merely a tidy one.

The freeze has to be on the biases, not the directions: freezing the directions instead is the pure
random-feature recipe, and by the same chance-level argument as at the control it pins recovery at the
`0.1` floor — a head fitting `256` fixed random rows drives `test_mse` down by memorising but cannot
recover `theta*`, because recovery lives in the directions I would have frozen. Freezing the biases is
the opposite budget and the correct one: it removes `256` trainable scalars and leaves the `256 x 100`
direction matrix and the head alive, lazifying the one-dimensional link fit while keeping the
high-dimensional search rich.

On a `k=1` sign row the collapsed flow is real, not a relabelling: start a unit row at `m = 0.1`; the
informative gradient along `theta*` is `O(mu_1) = O(0.8)` and does *not* vanish as `m -> 0` — the drift
is present at the equator, which is what `k=1` means. The `256`-sample noise still jitters the step by
`~ 0.625`, but now the jitter is around a nonzero mean drift rather than a sub-threshold one, and the
frozen-bias landscape has stripped away the perpendicular directions the row could have wandered into,
so the push goes straight into `m`: the row ratchets `0.1 -> 0.2 -> ...` toward `1` from *any* start,
since the only critical points are the unstable equator and the stable poles. The contrast with hermite
is entirely in the drift size: there the along-`theta*` push scales as `m^2 ~ 0.01` near the equator,
back below the noise, and the same clean landscape cannot ratchet a row it cannot budge.

There is a second reason the wide net should read out well on sign once the flow works, and it is where
the `256` rows finally help. The estimator is `normalize(sum_j |a_j| w_j)`. If the flow brings *every*
row to a common overlap `m` plus an independent perpendicular residual of norm `sqrt(1 - m^2)`, the
along-`theta*` part of the sum is coherent, `(sum_j |a_j|) m`, while the `256` perpendicular residuals
point in independent directions and cancel down by `~ sqrt(256) = 16`. The estimator overlap is then `m
/ sqrt(m^2 + (1 - m^2)/256)`, which for a modest per-row `m = 0.5` already gives `~ 0.994`, and for `m
= 0.3` gives `~ 0.98`. So the width is an *amplifier*: loosely aligned rows, if they share a pole,
aggregate into a near-perfect direction read. But the amplifier has a precondition — the rows must
agree on *which* pole. If the lottery sends some rows to `+theta*` and others to `-theta*`, the `|a_j|`
weighting cannot rescue it and the coherent term collapses. That is why the seed spread, not just the
mean, is the quantity to watch on sign: the collapsed landscape's whole job is to make all `256` rows
pick the same pole every run.

So the falsifiable expectations against the vanilla numbers, link by link. On `relu-d100` (`k=1`) I
expect *no change* — vanilla already hit `0.998`, the easy regime saturates regardless of the bias
treatment; a drop here would mean the unit-sphere init or the frozen biases are hurting the easy case
and the scale-matching argument is wrong. On `hermite-d100` (`k=3`) the collapse makes each row's
informative gradient colinear with `theta*` but does nothing about the signal's *size*: it is still `~
m^2 ~ 0.01` per step under the same `0.625` noise, the same SNR `~ 0.016`, because a cleaner landscape
reshapes where the gradient points, not how loud it is. So I expect a wash or at most a modest change
relative to `0.656` — if it jumped to near 1 the freeze alone would be doing the whole job, which the
SNR arithmetic says it cannot. On `sign-d100` (`k=1`, non-smooth) I expect the clearest gain: the
signal was always present (`mu_1 ~ 0.8`), the problem was the rough entangled landscape, and collapsing
it should pull every run toward the same pole rather than wherever its random rows started. I expect
sign recovery to rise above `0.210` and, importantly, to *tighten* across seeds — the `0.38`-wide
spread should shrink toward the tight agreement I see on relu. That tightening is the specific bet; if
the spread instead stays wide or grows, it means the wide net's independent rows re-introduce a lottery
the tied-architecture analysis does not have, and the collapse is only loosely realised on a `256`-row
net.

The delta from step 1 is one line of substance — freeze the biases, on a scale-matched unit-sphere init
— and the prediction is asymmetric: relu unchanged at the ceiling, sign improved because `k=1` only
ever needed a clean landscape, hermite roughly flat because the freeze collapses the landscape but does
not aggregate the weak `k=3` signal. If hermite does stay stuck, the remaining problem is that the
third-order signal is below the mini-batch noise, and the fix is to stop crawling with noisy
`256`-sample steps and sum the weak signal over the entire training set in one aggregated pass — the
full-batch aggregation the `256`-sample loop can never do.
