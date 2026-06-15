The learned gate confirmed the exact tension I warned it would, and the deferral-rate gaps are the
tell-tale. On Adult the gap *jumped* from the floor's 0.239 to 0.273, and on Law-School from 0.237 to a
striking 0.324 — the learned score, having more expressive power than a single monotone threshold,
concentrated deferrals on one subgroup *more sharply* than `max(probs)` ever did. Meanwhile the things I
hoped it would buy came in muted: COMPAS selective risk did edge down (0.2828 → 0.2800) and its
worst-group risk and gap improved (0.291 → 0.290, 0.150 → 0.119), and Law-School worst-group risk dipped
(0.414 → 0.405); but AUROC barely moved anywhere (Adult even slipped 0.853 → 0.852 from the balanced
re-weighting), and on Adult the selective risk got slightly worse (0.069 → 0.070). So the learned-score
rung was, on balance, a wash on accuracy and a *loss* on fairness — it traded a marginally smarter
ordering on the hard datasets for a worse concentration of deferrals on the easy ones. That is the
clearest possible signal about where the real defect lives. Every rung so far — the global confidence
floor, the conformal cut, the learned gate — shares one structural property, and the gaps it produces are
all large because of it: a *single decision boundary applied across subgroups*. The floor and conformal
use one boundary on `max(probs)`; the learned gate uses one boundary on `P(correct)`. Improving the score
under one shared boundary cannot fix, and can worsen, the deferral gap. So this rung stops touching the
score and attacks the boundary's group structure directly.

Let me first understand *why* a single shared boundary magnifies the gap, because I want the fix to follow
from the mechanism rather than be a guess. Set up the margin distribution properly, since the whole
accept/defer behavior is a functional of it. Take a confidence score and let `F` be the CDF of the signed
margin — positive on correct predictions, negative on incorrect ones. At a confidence threshold `τ`, a
predicted-and-correct point has margin `≥ τ`, the rule defers on `−τ < m < τ`, and a predicted-and-wrong
point has margin `≤ −τ`. So coverage is `1 − F(τ) + F(−τ)` and selective accuracy is `A_F(τ) =
(1 − F(τ)) / (1 − F(τ) + F(−τ))`. I want to know how `A_F` moves as I raise `τ` — does deferring the
least-confident points always help a group? Differentiating `A_F` (quotient rule, the value terms
cancel) gives `dA_F/dτ ∝ f(−τ)(1 − F(τ)) − f(τ)F(−τ)`, so accuracy increases at `τ` iff
`f(−τ)/F(−τ) ≥ f(τ)/(1 − F(τ))`. That is a clean monotonicity test. For a roughly symmetric, mildly
log-concave margin distribution this test is governed by whether the group's *full-coverage accuracy* sits
above or below 50%: if a group is above 50% at full coverage, raising the threshold *increases* its
selective accuracy; if it is below 50%, raising the threshold moves selective accuracy the *wrong way*.

That is the whole trap, and it is exactly the high-stakes-tabular regime. When the base model leans on a
feature that works for the majority subgroup and fails for a minority one — the textbook subgroup shift —
the worst subgroup's margin distribution is shifted left, packed with confident-but-wrong points, and its
full-coverage accuracy can sit at or below 50%. The average margin distribution, dominated by the
majority, sits comfortably above 50% and climbs nicely as the threshold rises. So one shared threshold
rides a curve that goes *up* on average while the worst subgroup's curve goes *down* — and because the
worst subgroup's confidences are systematically lower, that single threshold defers a *larger* fraction
of it. The shared boundary is not merely suboptimal on the worst subgroup; it can be anti-optimal, and it
pours the deferrals onto the very subgroup I most need to keep covered. The learned gate made this worse
precisely because re-ranking by `P(correct)` sharpened the boundary's ability to single out a subgroup;
the floor and conformal made it large; none of them could make it *small*, because none of them lets the
boundary differ by group.

So I need a reference for what "fair across subgroups" should even mean for an accept/defer decision, to
aim at. Frame predict-vs-defer as a meta-classification: a true positive is "predict and correct," a
false positive is "predict and wrong." Equalized odds for this meta-task asks every subgroup to be
predicted on at the same rate, separately conditional on correctness and on incorrectness. The
construction that achieves it keeps each threshold's total correct/incorrect counts but redistributes
them across subgroups in proportion to each subgroup's full-coverage share — so acceptance becomes
group-agnostic inside the correct and incorrect pools. That reference is the ideal, but it needs the
*labels* at decision time (to sort points into correct/incorrect) and a random redistribution, so it is
not directly implementable as a test-time rule. What it tells me, though, is what the shared boundary is
failing to do: it is not distributing acceptance pressure evenly across the subgroup structure. The
metric I *can* control through the policy interface is the per-subgroup *deferral rate*, so the
implementable analogue of the equalized-odds reference is **equal coverage per subgroup**. That is a
weaker promise than full equalized odds — it equalizes total coverage, not coverage-conditional-on-
correctness — but it is exactly the quantity the `deferral_rate_gap` metric reads, and it is the most the
fixed pipeline lets me enforce.

Now the rule falls out. I already have the score `s(x) = max(probs)` — and I keep it, because the
learned-gate experiment showed that changing the *score* under a shared boundary does not fix the gap, so
I should change the *boundary* and leave the score where it ranks best (it preserves the base model's
AUROC, which I do not want to disturb again). For a single global threshold I set it to the
`(1 − target_coverage)` quantile of `s` over *all* calibration points, which makes the *global* coverage
equal the target. The fix is to do that *per subgroup*: for each subgroup `g`, set its threshold to the
`(1 − target_coverage)` quantile of `s` over the calibration points *in that subgroup*. Then by
construction a `target_coverage` fraction of each subgroup's own points lies above its own threshold —
every subgroup is covered at the same rate, so the per-group deferral rates are all `1 − target_coverage`
on calibration and the gap collapses toward zero. At test, accept iff `s(x) ≥ threshold_group(g)`: the
boundary is now group-local. This is the implementable, group-aware coverage rule the reference's failure
mode points at — no labels at decision time, no redistribution, just one quantile per subgroup so the
coverage comes out equal.

Why a *quantile* per group and not, say, a learned per-group cost or a fitted per-group model? Because the
quantile pins each subgroup's coverage to the target *exactly* and offline — no extra model to fit,
trivial compute, robust to the fixed-pipeline constraint. It is the same device the floor used, applied
within each group instead of across all groups. And I must handle the degenerate case the harness can
produce: a subgroup that never appeared in calibration (some sex×race or race×gender cells are tiny) has
no per-group quantile, so I fall back to the *global* quantile — the best coverage-matched default I have
for an unseen subgroup. So `fit` computes the global quantile first (as the fallback and as
`threshold_`), then loops the unique calibration groups computing a per-group quantile for each;
`predict_accept` looks up each test point's group threshold, defaulting to the global one when the group
is unseen.

One property I get for free, and it matters for the AUROC metric I have been protecting. Per-subgroup
thresholding only *moves the decision boundary* per subgroup; it does not change the underlying score
`s(x) = max(probs)` that correctness is ranked by. So the AUROC of the acceptance score against
correctness is just the base softmax-response score's AUROC — preserved by construction, because
re-thresholding is a monotone, group-local shift that does not touch the global ranking the metric reads.
I keep the floor's AUROC (0.853 / 0.630 / 0.614, untouched) *and* equalize coverage — exactly the
combination the learned gate could not deliver, since it bought a re-ranking at the cost of the gap.

I want to be honest about this rule's scope, because it is not a fairness panacea and the next reader
should know what it does *not* do. Equalizing *coverage* does not equalize *selective accuracy*. If the
worst subgroup is genuinely hard — its full-coverage error is high because the frozen base model is bad
on it — moving thresholds cannot repair that; the worst-group selective risk will stay roughly where the
base model put it. So I expect the `worst_group_selective_risk` to be largely *unmoved* (Adult ~0.09,
COMPAS ~0.29, Law-School ~0.41), and the overall `selective_risk_at80` may even tick *up* slightly,
because forcing equal coverage on the worst subgroup means accepting some of its lower-confidence points
that the global threshold would have deferred — I am trading a hair of average accuracy for fairness, on
purpose. The honest scope of this rung is: it removes the deferral-rate disparity the shared boundary
injects, it keeps the SR score's correctness ranking, and it stops the worst subgroup from being silently
over-deferred. Closing the full-coverage accuracy gap is a training-time problem (group-DRO and friends),
which this rule deliberately does not touch because the base model is fixed.

So the falsifiable expectations against the learned-gate numbers — and against the whole ladder, since
this is the rung that finally moves the boundary per group. (1) The deferral-rate gaps should *collapse*:
Law-School from 0.324 down toward ~0.05, Adult from 0.273 down toward ~0.16, COMPAS from 0.119 down toward
~0.07 — by construction equal per-group coverage drives the gap to near the finite-sample floor (it will
not hit exactly zero because per-group quantiles interpolate on discrete calibration points and the tiny
cells fall back to the global cut). If the gaps did *not* shrink, the per-group quantile is not doing what
the algebra says. (2) AUROC should be *identical* to the floor on every dataset (0.853188 / 0.630096 /
0.614418), because the score is unchanged — only the cut moved; any AUROC drift would mean I accidentally
disturbed the ordering. (3) The selective risk and worst-group risk should be *close to* the floor's,
perhaps a touch worse (Adult selective risk could rise from ~0.069 toward ~0.073, worst-group from ~0.088
toward ~0.106), because equalizing coverage on the worst subgroup accepts some of its weaker points —
the deliberate accuracy-for-fairness trade. (4) Actual coverage should land near 0.80 overall, since
each subgroup is held at the target. This rung is the group-aware-cut sibling of the three single-boundary
rungs: same SR score the floor used, the floor's exact coverage quantile applied *within each subgroup*
with a global fallback, the base model's AUROC preserved — and the deferral-rate gap, the one axis no
earlier rung could touch, finally repaired (the full scaffold fill is in the answer).
