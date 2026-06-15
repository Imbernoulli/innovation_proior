The conformal rung did precisely the one thing it promised and nothing more, and the numbers say so
cleanly. COMPAS coverage moved from 0.7754 up to 0.7810 — the `(n+1)` rank correction pulled the unlucky
calibration draw back toward the 0.80 target, exactly the under-coverage fix I predicted. And everywhere
the plain quantile was already on target the conformal rank landed on the same calibration score: Adult
is bit-identical to step 1 (selective risk 0.069344, worst-group 0.088213, AUROC 0.853188, coverage
0.795578), Law-School likewise identical (0.369054 / 0.413831 / 0.614418 / 0.799731). AUROC did not move
on any dataset, because I never touched the order — only the cut. So the calibration-of-the-cut problem
is, for practical purposes, solved. But read the axis I deliberately left alone: the deferral-rate gaps
are still 0.239 / 0.150 / 0.237, essentially where the confidence floor put them. A pooled threshold,
however principled, controls the marginal accept event and cannot equalize per-group coverage — I said
that going in, and the gaps confirm it. And the *selective risk itself* barely moved: COMPAS even ticked
slightly worse (0.2812 → 0.2828), because the conformal cut accepts a marginally different set. Two
defects remain, and they are different in kind. One is the subgroup gap. The other — the one I want to
attack at this rung — is that *the score is still raw `max(probs)`*, and on the hard datasets that score
is a weak correctness-ranker: COMPAS AUROC 0.630, Law-School 0.614, both far from the 0.853 the model
manages on Adult. The cut is now well-placed on a mediocre ordering. The natural next move is to improve
the *ordering itself*.

So the question is: can I build a better per-point acceptance score than the bare maximum class
probability — one that ranks correct-above-incorrect better, especially where `max(probs)` is weak? Let
me first be honest about what I am *not* allowed to do, because the obvious idea is off the table. The
deepest version of "learn to defer" trains the predictor jointly with a reject head against a combined
predictor-plus-expert loss, so the classifier can *adapt* — give up on the region the reviewer handles
well and specialize elsewhere. That joint training is genuinely the strongest form, but it requires
retraining the base classifier, and here the base model and the entire pipeline are frozen. I get
calibration probabilities, labels, subgroup ids, optionally raw features — that is the whole surface. So
the joint, consistent surrogate is structurally unavailable; what *is* available is a *post-hoc* learned
gate that sits on top of the frozen model's outputs and rescores them. That restriction is not a
weakening I am grudgingly accepting; it dictates the entire shape of the method, and I want to derive the
post-hoc gate that the frozen pipeline actually admits.

Reframe the acceptance decision as its own little supervised problem — this is the move that unlocks it.
At decision time I want to accept the points the base model will get right and defer the ones it will get
wrong. So define a fresh binary target on the calibration set: `correct_i = 1[argmax_j probs_i = y_i]`,
the bit "did the base model predict this calibration point correctly." I *have* the calibration labels —
the harness hands `y_true` to `fit`, and the floor and the conformal rung both ignored them. Train a
compact classifier to predict `correct` from features I *can* compute at test time (no labels there), and
use its predicted `P(correct | features)` as the acceptance score. This is a *meta*-classifier: it does
not predict the task label, it predicts the base model's correctness, which is exactly the thing the
acceptance score is supposed to rank. `max(probs)` is one hand-chosen feature of correctness; a learned
gate can combine *several* and weight them from data.

What features? They must be functions of what `predict_accept` sees at test — the probability vector,
the subgroup id, and the raw features — never the label. The harness even hands me a ready helper,
`_confidence_features`, which assembles exactly the right small table from a probability matrix:
`p1 = probs[:,1]` (the positive-class probability), `max_prob = max(probs)` (the floor's whole score),
`margin = |p1 − p0|` (how decisively the top class beat the runner-up), `entropy = −Σ p log p` (the
spread of the distribution), and, when available, the subgroup id and the first raw feature `X[:,0]`. Why
this set and not just `max_prob`? Because they capture *different facets* of reliability that
`max(probs)` alone conflates. Margin and max-prob agree in the binary case up to a monotone map, but
entropy responds to the *shape* of the distribution and the raw feature `X[:,0]` and the group id let the
gate learn that confidence means different things in different regions — that, say, a 0.7 max-prob on one
subgroup is more trustworthy than a 0.7 on another. A single monotone score cannot express that; a
learned weighting over several features can. Crucially, on Adult where `max(probs)` is already a strong
ranker (AUROC 0.853) the extra features have little to add and the gate should essentially reproduce the
floor; on COMPAS and Law-School where `max(probs)` is weak (0.630, 0.614), the gate has *room* to find a
better combination — that is precisely where I am hoping to move the needle.

Now the meta-classifier itself. I want it compact, offline, and robust to the fixed-pipeline constraint
— no deep net to babysit, fits in milliseconds on a few thousand calibration points. Logistic regression
is the right tool: it outputs a calibrated-in-the-logistic-sense `P(correct | features)`, it is convex
(one optimum, no seed lottery), and a `StandardScaler` in front handles the wildly different scales of
the features (probabilities in `[0,1]`, entropy in nats, a raw feature in whatever units). One subtlety
matters here: the `correct` target is *imbalanced* — the base model is right far more often than wrong
(its accuracy is well above 50% on all three datasets), so a naive fit would learn "predict correct
almost always" and the predicted `P(correct)` would compress into a narrow high band, ruining its use as
a *ranking* score across the full coverage range. So I set `class_weight="balanced"`, which reweights the
rare "wrong" class up to parity, forcing the logistic to actually separate the wrong-prone points rather
than ignore them. That is the difference between a gate that ranks and a gate that just says "yes" to
everything. I give it `max_iter=1000` and the `lbfgs` solver so the convex fit converges, and seed it
with the policy's `random_state` for reproducibility even though convexity makes it nearly seed-free.

With the score defined, the cut is the *same* device as the floor — and this is the design decision I
want to be explicit about, because I could have used the conformal rank here too. The acceptance score is
now `meta.predict_proba(features)[:, 1]`, the predicted probability of correctness, higher = more
acceptable. To hit the target coverage I set the threshold to the `(1 − target_coverage)` quantile of
*that* score over the calibration set — accept a `c`-fraction by deferring the lowest-predicted-correct
`(1 − c)`-fraction. I deliberately reuse the plain quantile rather than the conformal `(n+1)` correction:
the conformal guarantee is a clean statement *for an exchangeable score*, and the predicted-correctness
score is a learned function of the calibration data, so its exchangeability with a fresh point is no
longer the clean i.i.d. story conformal needs (the meta-model saw the calibration points). The honest cut
for a learned score is the empirical coverage quantile, and that is what I take — the same `quantile =
clip(1 − target_coverage, 0, 1)`, the same `np.quantile`, the same `score ≥ threshold` accept test. So
this rung swaps the *score* (raw max-prob → learned `P(correct)`) and keeps the cut machinery of step 1.

I want to predict the failure mode honestly, because there is a real one and it is the seam to the next
rung. The meta-classifier is fit to predict correctness *marginally*, pooling all subgroups, and it gets
the subgroup id as just one feature among several. A logistic regression will use the group feature only
insofar as it linearly improves the *overall* correctness prediction; it has no objective term that says
"defer at equal rates across groups." So the learned score can, and likely will, *concentrate* deferrals
on whichever subgroup it judges least-likely-correct — and because the gate now has more expressive power
than a single monotone threshold, it can concentrate deferrals *more sharply* than the floor did. In
other words I expect a better correctness ranking on the hard datasets to come *at the cost of* a worse
deferral-rate gap, because nothing in the objective protects per-group coverage. That is the precise
tension I cannot resolve at the score level: improving the marginal ranking and equalizing per-group
deferrals are different objectives, and a marginal meta-classifier optimizes the first while being free
to worsen the second. If the gaps blow up, that is not a bug — it is the diagnosis that hands the baton to
a rung whose *cut* is group-aware rather than whose *score* is smarter.

There is also a quieter risk: the meta-classifier is fit on the *same* calibration set whose quantile
sets the threshold, so its in-sample `P(correct)` is optimistic, and the test-time ranking can be a touch
worse than the calibration ranking suggests — a mild over-fitting of the gate. With only logistic
regression on five-ish features this should be small (low capacity, balanced weighting), but it means I
should not expect the AUROC to leap; a modest move, if any, is the realistic hope. And I should keep the
fallback the contract implies: if the meta-model somehow failed to fit (`meta_model_ is None`), the
acceptance score must degrade gracefully to `max(probs)` so the policy never crashes — the floor is the
safe default underneath the learned score.

So the falsifiable expectations against the conformal numbers. (1) On the *hard* datasets the AUROC has
room to move and the selective risk has room to drop, because the learned score combines features the
floor's `max(probs)` ignored — COMPAS selective risk could dip below 0.2828 and Law-School below 0.3691,
and if the gate genuinely re-ranks, the worst-group selective risk on COMPAS (now ~0.291) and Law-School
(~0.414) could come down a touch too. (2) On Adult, where `max(probs)` is already a strong ranker, the
gate should essentially reproduce the floor — AUROC near 0.853, selective risk near 0.069 — because the
extra features add little; if Adult got *worse*, that would mean the balanced weighting or the
overfitting hurt a score that was already good. (3) The deferral-rate gaps should *not* improve and may
*worsen* relative to conformal (especially on the datasets where the gate re-ranks most), because the
marginal meta-classifier has no per-group coverage objective — a worsened gap here is the expected price
of a smarter score and the explicit motivation for making the *cut* group-aware next. If instead the gaps
shrank, my "the score is marginal, the cut is group-blind" account would be wrong. This rung is the
better-ordering sibling of the two threshold rungs: it learns the acceptance score from the calibration
labels the earlier rungs threw away, keeps the coverage-quantile cut, and leaves the subgroup deferral
problem standing — sharper, even — for the rung that finally moves the cut per group (the full scaffold
fill is in the answer).
