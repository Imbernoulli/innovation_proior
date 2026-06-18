The vanilla-SGD run told me exactly what I expected and gave me the split in numbers. On `relu-d100`
all three seeds pinned at `direction_recovery ~ 0.998` with `test_mse ~ 0.0017` — the `k=1` easy
regime where ordinary SGD already finds the direction, and where I should not expect to do better.
The two hard links are where the floor showed. On `hermite-d100` (`k=3`) recovery averaged `0.656`,
with per-seed values {0.615, 0.736, 0.616} — the network found *some* alignment but landed far short
of 1, and `test_mse` stayed at `0.16`, two orders of magnitude above the relu floor. And `sign-d100`
was the real failure: recovery {0.049, 0.156, 0.426} for a mean of `0.210`, barely above chance on
seeds 42 and 123, with `test_mse ~ 0.44`. So the diagnosis is confirmed and sharp. This is not a
learning-rate problem — the same SGD that nails relu cannot move the rows toward `theta*` on the hard
links. It is two coupled problems: on hermite the per-mini-batch direction signal (`~ d^{-1}`) is
drowned by the 256-sample gradient noise, and on both hard links the first layer and the readout are
trained jointly, so the high-dimensional direction search and the one-dimensional link fit interfere —
the head fits the noisy targets locally with rows that never align. The seed-to-seed spread (sign
ranging 0.049 to 0.426) is the tell: outcome depends on whether a run's random rows happened to carry
enough early overlap, exactly what you see when the landscape has no benign structure pulling every
run to the same place.

So the fix I want is structural, not a knob: reshape the optimisation landscape so that the
direction search stops fighting the link fit. Let me reason from the geometry. The trouble is that
the first layer has `W = 256` rows, each a free vector in `R^d`, and each one is simultaneously asked
to (i) point toward `theta*` and (ii) build, together with the head, a basis rich enough to represent
the univariate link `g`. Those are different jobs on different scales — the direction is a needle in
`d` dimensions, the link is a one-dimensional function — and entangling them is what wrecks the rate.
I want to *decouple* them inside the same fixed two-layer ReLU net, using only the levers the harness
exposes: how I initialise `fc1` and `fc2`, what I freeze, and what the optimiser touches.

Here is the structural observation that does it. A hidden ReLU neuron computes `ReLU(<w_j, x> + b_j)`.
Two things parameterise it: the *direction* `w_j` (where it looks in input space) and the *bias* `b_j`
(the threshold along that direction). Now think about what each is *for* in a single-index model.
The non-parametric job — approximating the univariate link `g(u)` for `u = <theta*, x>` — is exactly
the job of a *spread of thresholds* along the relevant direction: a bank of functions
`{ReLU(u - b_j)}` with varied `b_j` is a one-dimensional spline basis that can fit any reasonable
`g`. That is a *kernel* job, a random-feature job; the biases are the random-feature sampling of that
one-dimensional kernel. The high-dimensional job — finding `theta*` — is the job of the *directions*
`w_j`. So the two jobs live in two different sets of parameters, and the lesson of lazy-versus-rich
training (Chizat, Oyallon & Bach 2019) is exactly that a part of the model that stays at
initialisation and acts like a fixed kernel is the lazy part, while a part that moves and learns
features is the rich part. The non-parametric link fit should be lazy; the direction search should be
rich. The clean move, then, is to **freeze the biases at their random init and train only the
directions and the head**. That is the one essential change to the recipe.

Let me check that freezing the biases actually collapses the landscape, because if it does that is the
justification, not an analogy. Write the population loss `L = E[(G(x) - y)^2]` for the network
`G(x) = sum_j a_j ReLU(<w_j, x> + b_j)`, and decompose against the Gaussian in Hermite polynomials.
The key identity is that a degree-`p` feature along a direction `w` overlaps the target's degree-`p`
component by exactly `<w, theta*>^p`. If the biases `b_j` are *frozen*, they do not depend on the
directions, so the only place a direction `w_j` enters the loss is through the scalar overlap
`m_j = <w_j, theta*>`. The `d`-dimensional dependence of the loss on each row collapses onto a single
number per row. And the gradient with respect to `w_j` becomes colinear with `theta*` — it can only
push `m_j` up or down, i.e. rotate the row toward or away from the truth. The high-dimensional search
turns into a scalar flow in the overlaps. *That* is what freezing the biases buys, and it is precisely
what was missing on the hard links under vanilla SGD: there the biases were also being trained, so they
dragged the non-parametric problem back into the high-dimensional dynamics and re-entangled the two
jobs I now want separate.

In the ideal limit (infinitely many, unregularised random-feature thresholds) the projected loss is
strictly decreasing in `|m|`, with only two kinds of critical point: the equator `m = 0` and the poles
`m = +-1`. No spurious local minima — gradient flow slides from the random start up to a pole. With
finitely many *frozen* random biases the picture is preserved as long as the random-feature
approximation error of the link is small; the bias variance has to be large enough that the
one-dimensional random-feature space is rich enough to approximate smooth links with a polynomial rate
(in the analysis this is the condition `tau > 1` on the bias scale). So the biases should be drawn
with `O(1)` spread, not tiny — wide enough thresholds to span the range of `u = <theta*, x> ~ N(0,1)`.

Now ground this in the harness exactly, because the harness does *not* give me the full apparatus the
analysis uses, and I have to be honest about what I keep and what I drop. The fixed `TwoLayerMLP` is a
*wide* net — `Linear(d, 256)` with all 256 rows independent, then ReLU, then `Linear(256, 1)`. The
analysis I just sketched is cleanest for a *tied* architecture (one shared inner direction, neurons
differing only in bias and sign) with a time-scale separation (search the direction first with a
small/sparse readout, then turn the head on) and a final fresh-sample ridge refit of the readout. The
harness exposes none of that cleanly: I cannot tie the rows (the model is fixed wide), I am handed
mini-batches not a controllable two-phase schedule, and the standard `finalize` refit is a separate
move I am deliberately *not* taking at this rung (it belongs to the next one). So I keep the *one
essential move* — freeze the biases — and apply it to the standard wide MLP, accepting that the wide
net's many independent rows are a noisier realisation of the collapsed landscape than the tied net
would be. Concretely the recipe becomes: initialise the first-layer rows on the unit sphere (so every
row starts as a clean random probe of `S^{d-1}`, comparable scale, correlation `~1/sqrt(d)` with
`theta*`); draw the biases `~ Uniform(-1, 1)` — the `O(1)` spread of thresholds that makes the frozen
bank a usable one-dimensional basis — and *freeze* them with `requires_grad_(False)`; init the readout
small and uniform as before; build the optimiser over *only the parameters with `requires_grad`*, so
the frozen biases are excluded automatically; and run the same SGD-with-momentum mean-squared-error
mini-batch loop. No finalize.

Two grounding details matter. First, the unit-sphere init of `fc1.weight` (rather than Kaiming) is
deliberate: with the biases frozen, the only thing the optimiser controls in the first layer is the
*direction* of each row, so I want every row to start as a comparable unit probe — Kaiming's `sqrt(2/d)`
scale would let row norms drift and muddy the "direction-only" reading. Second, freezing happens by
setting `requires_grad_(False)` on the bias tensor and then constructing the optimiser from
`[p for p in net.parameters() if p.requires_grad]`; the harness's direction estimator
`normalize(sum_j |a_j| w_j)` then reads off the readout-weighted rows, which now move *only* in
direction.

Let me now state the falsifiable expectations against the vanilla-SGD numbers, link by link, because
that is what tells me whether the landscape collapse is real here.

On `relu-d100` (`k=1`) I expect *no change* — vanilla SGD already hit `0.998`, the easy regime saturates
regardless of how the biases are handled, so frozen-bias should also land near `0.998`. If it dropped
here, the unit-sphere init or the frozen biases would be hurting the easy case, which would be a red
flag.

On `hermite-d100` (`k=3`) this is the interesting test. The landscape-collapse argument says freezing
the biases makes the gradient on each row colinear with `theta*`, which should help the direction
search even when the per-step signal is weak. But I have *not* added the two things that make the
hard-link rate actually optimal — the full-batch signal aggregation and the time-scale separation —
so I do not expect a dramatic jump. The honest prediction is a *modest* improvement or a wash relative
to vanilla SGD's `0.656`: the collapsed landscape is cleaner, but the wide net's many independent rows,
driven by the same noisy 256-sample gradients, still struggle to surface the `~d^{-1}` third-order
signal. If frozen-bias lands roughly *at or slightly below* vanilla SGD on hermite, that is consistent
with the diagnosis — the missing piece is not the bias freeze but the signal aggregation, which is
exactly what the next rung supplies. If it jumped to near 1, the freeze alone would be doing the whole
job, which I doubt.

On `sign-d100` (`k=1`, non-smooth) I expect the clearest *gain* from this rung. The link is `k=1`, so
the direction signal is present at first order — the problem under vanilla SGD was the rough,
entangled landscape and the joint dynamics fitting the discontinuous target locally with unaligned
rows (recovery as low as 0.049). Collapsing the landscape by freezing the biases should make the
direction search far more reliable here: every run should be pulled toward the same pole rather than
landing wherever its random rows happened to start. So I expect sign recovery to rise meaningfully
above vanilla's `0.210` mean and, importantly, to *tighten* across seeds — the seed-to-seed spread
{0.049, 0.156, 0.426} should shrink as the benign landscape removes the dependence on lucky init.

So the delta from step 1 is one line of substance — freeze the biases (and switch to a unit-sphere
first-layer init so the rows are clean direction-only probes) — and the prediction is asymmetric:
relu unchanged at the ceiling, sign improved and tightened because `k=1` only ever needed a clean
landscape, hermite roughly flat because the freeze collapses the landscape but does not aggregate the
weak `k=3` signal. Whatever the precise numbers, the structure they reveal points straight at step 3:
if hermite stays stuck, the remaining problem is that the third-order direction signal is below the
mini-batch noise, and the fix is to stop crawling with noisy 256-sample steps and instead take *one
giant full-batch step* that sums the signal over all `n_train` samples — and pair it with the
closed-form readout refit the harness's `finalize` hook is sitting there waiting for.
