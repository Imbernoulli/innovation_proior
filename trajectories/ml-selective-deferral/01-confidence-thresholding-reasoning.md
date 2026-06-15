The acceptance rule is the whole point, but it bolts onto a fixed classifier, and with the simplest
possible rule that classifier is the floor — so the thing to start from is just deciding accept-or-defer
at a target coverage with no machinery at all. The base model is frozen, the splits are frozen, the
metrics are computed for me; the only object I get to write is the `SelectivePolicy`, and the cheapest
honest fill is the one the scaffold already sketches. Let me settle why that is the right floor and what
it must fail to do, because that failure is what every later rung reacts to.

Name the two quantities I am trading before touching code, because the whole problem is a trade-off and I
want the axes right. The fraction of test points I actually answer on — coverage. And the error rate
measured *only* over the points I answered — the selective risk. They pull in the obvious direction: if I
defer more, coverage drops, and the selective risk should also drop, because the points I am dropping
ought to be the ones I was least sure of — *if* I defer on the right points. That "if" is the entire
game. Deferring at random lowers coverage and does nothing for risk; deferring on exactly the points
most likely to be wrong buys the steepest drop in risk per unit of coverage given up. So I really need
two separable things: a per-point quantity that orders points from most-trustworthy to least, and a way
to cut that ordering at a point that hits the coverage I am told to hit. The harness mirrors this split
exactly — `acceptance_score` produces the per-point ordering, `fit` plus `predict_accept` produce the
cut — so the contract is already shaped around the right decomposition, and my job at step 1 is to fill
each half with the most defensible default.

Take the score first. What per-point number orders the base model's predictions by reliability? Chow
worked the ideal case out sixty years ago: if I had the true posterior `P(y|x)`, the Bayes-optimal
reject rule under 0/1 loss is ambiguity rejection — reject when no class is dominant enough, i.e. when
`max_y P(y|x)` is below a threshold, accept when it clears it, and the error and reject rates move
monotonically against each other as the threshold sweeps. The structural lesson is that the thing to
threshold is the *maximum posterior*: the most-likely class's probability is exactly how unambiguous the
prediction is — near 1 the input is clear, near `1/k` the model is guessing. I do not have `P(y|x)`. What
I have that is *shaped* like a posterior is the base model's probability vector — nonnegative, sums to
one, one entry per class. So the natural move is Chow's rule with the only posterior-surrogate I am
handed: `κ(x) = max_j probs(x)_j`, the maximum class probability. That is precisely the scaffold
default's `acceptance_score`, and I will keep it.

I should push on whether that is legitimate, because the well-known objection sits right here and I do
not want to wave it away. Modern classifiers are over-confident: the max class probability can sit high
on points the model gets wrong, so it is *not* a calibrated estimate of `P(correct)`. If I were claiming
`max_j probs_j` literally equals the probability of being right, this would sink me. But look at what the
acceptance rule actually needs. The selection only ever compares points to each other and to one cut
point; I never need the absolute value of `κ` to be a probability, only its *ordering* to be right — more
reliable predictions getting higher `κ`. Calibration is a strictly stronger property than ranking, and
it is the wrong property to demand: a monotone-but-miscalibrated score ranks perfectly and selects
perfectly. The calibration critique kills the "softmax = probability" reading and leaves the
"softmax = reliability ranking" reading untouched, and the second is all I use. And the ranking is in
fact good: across trained classifiers the max class probability separates correct from incorrect
predictions well above chance, which is exactly the property the whole construction rests on, and is the
property the `auroc` metric will read off directly. So `κ = max(probs)` is not a convenience hack; it is
the defensible default score, and the harness even hands me richer features (`_confidence_features`)
I am deliberately *not* using yet — that is a lever for a later rung, not the floor.

Now I want to check that a single *global* threshold is the right shape for the cut, not merely a
convenient one, because I could imagine wanting something cleverer — different cuts in different regions,
or per subgroup. Picture it abstractly: for a fixed classifier, the best selection rule at a given
coverage admits, among all subsets of that size, the subset with the lowest total error. If `κ` ranks
points by reliability, then "admit the lowest-error subset of size matching my coverage" is exactly
"admit the top-ranked prefix of that size," and a prefix of a sorted list is precisely what a threshold
carves out. Once the ranking is fixed, the only remaining freedom is how deep to go — one number. So if
`κ` genuinely orders points by reliability, a single global threshold is not a heuristic shortcut, it is
the optimal selection function *for that score*. That is the reason to start here: the floor is not a
weak strawman, it is the optimal rule under one assumption — that one ranking and one cut suffice for
every point regardless of which subgroup it belongs to. Holding that assumption up to the light is what
later rungs will do.

So the cut. The operator hands me a target *coverage* — answer on 80% of points. I want the threshold
`θ` such that the fraction of points with `κ(x) ≥ θ` is about 0.8. That is just a quantile. If 80% must
be accepted then 20% must be deferred, and the deferred ones are the lowest-`κ` 20%, so `θ` is the 20th
percentile of the score distribution — the `(1 − coverage)` quantile. Where do I compute it? Not on the
test set I will report on — that would be peeking, and the coverage would look on-target only because I
forced it to on those very points. Not on the fit data the base model trained on — the model is
over-confident there, so its score distribution is not the one it will face. I need the held-out
**calibration** set, which the harness hands `fit` for exactly this purpose: compute the scores on
calibration, take the `(1 − coverage)` empirical quantile, freeze it, and apply it unchanged at test.
Up to finite-sample interpolation and score ties, the calibration accept-rate is then the target by
construction, and because calibration is an i.i.d. draw the same `θ` gives approximately the target
coverage on test.

Before I write it I make the direction conventions airtight, because an off-by-one in the inequality or a
flipped quantile silently inverts the whole thing. `κ` is a confidence: higher means more reliable, so I
*accept* high `κ` and *defer* low `κ`; the accept predicate is `κ(x) ≥ θ`. To accept a fraction `c` I
place `θ` so a `c`-fraction lies at or above it, i.e. `θ` at the `(1 − c)` quantile from the bottom: with
`c = 0.8`, `θ` is the 0.2 quantile, the value below which 20% of scores fall — `np.quantile(scores,
1 − c)`. Writing `np.quantile(scores, c)` would cut at the 80th percentile and accept only the top 20%,
the exact opposite coverage. So `quantile = 1 − target_coverage`, clipped into `[0,1]` for a degenerate
coverage, `θ = np.quantile(scores, quantile)`, accept `score ≥ θ`. That is the entire policy: rank by
max probability, cut at the coverage quantile on held-out data, accept above the cut. It is training-free,
one pass, offline — exactly what the fixed-pipeline constraint wants, no extra model to fit, trivial
compute.

I also note what this floor leaves on the table, because that is what the ladder is for. The harness
hands `fit` the calibration *labels* `y_true` and the *subgroup ids* `groups`, and `predict_accept` the
test groups — and the global-threshold rule uses *none* of them. The labels could anchor a learned
correctness score; the groups could anchor a per-subgroup cut. The floor deliberately ignores both,
which is exactly why it is the floor: one ranking, one cut, group-blind, label-blind. The richer signals
sit unused in the contract, waiting.

Now reason about what this floor must do, because that is the entire point of running it. The score
`max(probs)` ranks correct above incorrect well above chance, so at any coverage the selective risk
should sit clearly below the full-coverage error — the rule works, and the `auroc` should land
meaningfully above 0.5 on the dataset where the base model is strong (Adult, where income is fairly
predictable) and closer to 0.5 where the base model is weak (COMPAS, Law-School, where recidivism and
binarized GPA are genuinely hard and the probabilities carry little ranking signal). The actual coverage
should land near 0.80, a touch under, because the `(1 − coverage)` quantile cut accepts everything
strictly above `θ` and ties at `θ` push slightly below target. So far so good.

But the part I expect to *fail* is the fairness axis, and it should fail for a reason baked into the
single global cut. Under subgroup shift the base model is more confident on some subgroups than others —
typically the majority subgroup, where it had more data and the features line up, gets systematically
higher `max(probs)`, while a minority subgroup gets lower scores. One global threshold `θ` then accepts a
*larger* fraction of the high-confidence subgroup and a *smaller* fraction of the low-confidence one,
even though the loop demands 80% coverage overall, not per group. The deferrals concentrate on whichever
subgroup the model is least confident on. So I expect the `deferral_rate_gap` — max-minus-min subgroup
deferral rate — to be conspicuously large, not because the rule is buggy but because a group-blind cut
*cannot* equalize per-group coverage; it only controls the marginal. That is the structural ceiling of
step 1, and the diagnosis is already pointed forward: I have a *single-cut* problem and a *fairness*
gap, and the fixes split cleanly — make the cut's threshold a principled finite-sample object
(conformal), make the *score* better than raw max-probability (a learned correctness predictor), or
make the cut *per subgroup* so the coverage comes out equal (group-wise thresholds). Whatever the precise
numbers, the global confidence threshold is the weakest rung by construction: it is the optimal rule
*under the assumption that one group-blind ranking and one cut suffice*, and that assumption is exactly
what subgroup shift breaks (the full scaffold fill is in the answer).
