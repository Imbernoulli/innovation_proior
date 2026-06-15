The global confidence threshold told me, in numbers, exactly which of its two halves is fine and which is
shaky. The *ranking* half is doing real work: on Adult `auroc` is 0.853, selective risk 0.069 against a
clearly larger full-coverage error, worst-group risk 0.088 — `max(probs)` orders correct above incorrect
well, and the cut peels off the wrong predictions first. Where the base model is weak the ranking is
weak, exactly as I expected — COMPAS `auroc` 0.630, Law-School 0.614, with selective risks of 0.281 and
0.369 sitting near the base error because the probabilities carry little signal there. None of that is a
flaw in the *rule*; it is the base classifier's information showing through, and I cannot move it without
retraining, which I am forbidden from doing. The *cut* half is where I see slack I can actually attack.
Look at the achieved coverages: 0.7956 on Adult, **0.7754** on COMPAS, 0.7997 on Law-School. The target
is 0.80 every time. Adult and Law-School land within a quarter percent — fine — but COMPAS comes in two
and a half points under. That is not random noise; it is the plain empirical quantile mis-locating the
threshold on the calibration draw and then *transferring* that miss to test, with no accounting for the
finite-sample gap between "the fraction I cut on calibration" and "the fraction that will clear it on a
fresh point." The global rule treats the calibration quantile as if it were the population quantile. It
is not, and COMPAS is where the difference bites.

So the question I want to answer at this rung is narrow and worth getting exactly right: can I place the
threshold so that the accept event has a *finite-sample* guarantee — a statement that a fresh test point
clears the cut at (at least) the target rate, valid for the actual calibration size, not just
asymptotically? I am deliberately not touching the score (still `max(probs)`, since the ranking is the
base model's to give) and not yet touching the group-blindness (the deferral gaps — 0.239 on Adult,
0.144 on COMPAS, 0.237 on Law-School — are large, but that is a *different* defect, and I will not
conflate fixing the cut's calibration with fixing its group behavior). I want the cleanest possible
upgrade to the *cut*, and the right tool is a symmetry argument, not a better quantile estimator.

Separate two things the global rule blurs. One is ranking quality — does `max(probs)` put easy points
above hard ones? That sets selective risk and AUROC, and I have just measured it; it is the base model's.
The other is calibration of the cutoff — after I fix a scalar score, can I place the threshold with an
exact finite-sample accounting of how often a new point clears it? The second might be answerable even
when the first is imperfect, and crucially it does *not* need the scores to be probabilities. It needs an
exchangeability argument.

Suppose the calibration points and the test point are exchangeable — they are, here: calibration and test
are i.i.d. draws from the same distribution after the fixed split. Take any scalar score on a point,
compute `r_1, …, r_n` on calibration and `r_test` on the new point, ties broken so ranks are well
defined. Because the joint law is unchanged by permuting the `n+1` points, the test score is equally
likely to occupy any rank among the `n+1` pooled scores. Make it concrete by conditioning on the
unordered set of values: for each target rank `j` there are `n!` assignments putting `r_test` in rank `j`
out of `(n+1)!` equally likely ones, so `P(rank(r_test) = j) = 1/(n+1)`. If `r_(1) < … < r_(n)` are the
sorted calibration scores, then `r_test ≤ r_(k)` exactly when the pooled rank of `r_test` is at most `k`;
there are `k` such ranks, so

  `P(r_test ≤ r_(k)) = k/(n+1)`.

That is the entire finite-sample engine. It does not know the distribution of the score, it does not know
how the base model was trained, it only uses that the calibration scores and the test score are
exchangeable. This is exactly what the plain calibration quantile lacked — and it is why COMPAS drifted
under target: the empirical quantile aims at `k/n`, but the event I actually care about lives on the
`n+1`-point pooled scale.

Now choose the rank. I want at least a `1 − α` probability that the new point's nonconformity is no
larger than the calibration threshold `q_hat`, where `α = 1 − target_coverage` is the fraction I am
willing to defer. If `q_hat = r_(k)`, the exact probability is `k/(n+1)`, so the smallest safe integer is
`k = ceil((n+1)(1 − α))`, giving `P(r_test ≤ r_(k)) = ceil((n+1)(1 − α))/(n+1) ≥ 1 − α`. The `+1` is not
cosmetic: the test point is the unseen `(n+1)`-st member of the exchangeable bag, so the denominator is
`n+1`, not `n`. If I take the naive calibration-only rank `ceil(n(1 − α))` — which is morally what the
plain quantile does — the actual rank probability is `ceil(n(1 − α))/(n+1)`, and for finite `n` that can
fall *below* `1 − α`. That under-coverage is precisely the COMPAS symptom, and the ceiling on the
`(n+1)`-scale is the forced correction: the rank is discrete and the first integer whose rank probability
clears the bar is `ceil((n+1)(1 − α))`; rounding down breaks the lower bound.

There is an endpoint to keep separate. If `α` is so small that `ceil((n+1)(1 − α)) = n+1`, the
mathematical threshold is `q_hat = +∞` — `n` calibration scores cannot supply an `(n+1)`-st order
statistic — the accept-everything ideal. An array implementation cannot index `r_(n+1)`, so the practical
guard clamps the zero-indexed rank into `[0, n−1]`; for ordinary coverages with `ceil((n+1)(1−α)) ≤ n`
the clamp does nothing, and at the endpoint it falls back to the largest observed calibration
nonconformity. The lower endpoint mirrors this: if `α` is clipped to 1 the computed zero-indexed rank
would be `−1`, clamped to 0.

I should also ask whether this is wastefully conservative — whether I am paying real coverage to buy the
guarantee — because if so the COMPAS fix would cost me Adult and Law-School. Under continuous scores with
no ties the rank is exactly uniform, so the achieved probability is exactly `ceil((n+1)(1−α))/(n+1)`, and
since `ceil(z) < z + 1`, `P(r_test ≤ q_hat) ≤ 1 − α + 1/(n+1)`. Together with the lower bound,
`1 − α ≤ P(r_test ≤ q_hat) ≤ 1 − α + 1/(n+1)`. The correction does not push me far above target — the
slack is one rank out of `n+1`. So I expect the conformal threshold to *fix* the under-coverage where the
calibration draw was unlucky (COMPAS) without meaningfully over-shooting where the plain quantile was
already on target (Adult, Law-School). On those two the conformal rank and the empirical quantile will
land on essentially the same calibration score, so I expect their numbers to be nearly identical there —
which, if it holds, is the honest signature that this rung repairs *only* the calibration of the cut and
nothing else.

Now the score, because the conformal machinery needs a *nonconformity* score (larger = stranger) and I
have a *confidence* (larger = more acceptable). The set-prediction score `1 − probs_y` is natural when
the true label is available, but my deploy-time decision cannot use `y_test` — the policy must decide
before the reviewer reveals anything. What I do have at test is the label the model would output,
`argmax_j probs_j`, and the confidence attached to it, `max_j probs_j`. If the model is confident in its
own top prediction I accept; if not I defer. So the deployable nonconformity score is `r(x) =
1 − max_j probs(x)_j`. It is label-free, computed identically on calibration and test, which preserves
the exchangeability the whole argument rests on. And it makes the scope honest: the conformal order
statistic calibrates the *accept/defer event* for this score; it does not prove the top prediction is
correct. Correctness is the ranking question, which the AUROC and selective-risk metrics read and which
the score inherits unchanged from the base model — so I do *not* expect AUROC to move from step 1
(0.853 / 0.630 / 0.614); the conformal step touches the cut, not the order.

Assemble it in nonconformity space. With `n` calibration points, confidences `s_i = max_j probs_i`,
nonconformities `r_i = 1 − s_i`, sort the `r_i`, take the conformal rank `rank = ceil((n+1)(1 − α)) − 1`
in zero-indexed coordinates clamped to `[0, n−1]`, and `q_hat = sort(r)[rank]`. The accept condition is
`r(x_test) ≤ q_hat`. But the harness wants a score where larger is more acceptable and a comparison of
the form `acceptance_score ≥ threshold`, so I store the equivalent confidence threshold `threshold =
1 − q_hat`. Then `r(x_test) ≤ q_hat ⇔ 1 − max_j probs_test ≤ q_hat ⇔ max_j probs_test ≥ 1 − q_hat ⇔
acceptance_score(x_test) ≥ threshold`. The whole reduction is: nonconformity is `1 − max(probs)`, the
threshold is the conformal order statistic in nonconformity space, the deployed comparison is that cutoff
converted back to confidence space — a single comparison on `max(probs)`, identical in shape to step 1's
accept test, differing only in *where* the cut sits.

One limitation I want to state plainly, because it is the seam where the *next* rung enters. A single
pooled threshold controls the *marginal* accept event — `P(r_test ≤ q_hat) ≥ 1 − α` averaged over the
whole population. It does *not* force each subgroup to clear the same accept/defer rate, and it does not
equalize per-group selective risk. To get a group-conditional accept rate I would have to calibrate the
conformal rank *within* each group and pay for it with smaller per-group calibration samples. For this
pooled rule the honest promise is marginal — so I fully expect the deferral-rate gaps to stay roughly
where step 1 left them (≈0.24 / ≈0.14–0.15 / ≈0.24), maybe a hair different on COMPAS where the cut
itself moves, but not *repaired*. Fixing the marginal calibration of the cut and fixing the subgroup
distribution of deferrals are two different problems, and I am solving only the first here.

So the falsifiable expectations against step 1's numbers. (1) COMPAS actual coverage should rise from
0.7754 toward 0.80, because the `(n+1)` rank correction is exactly the fix for the calibration draw that
landed the plain quantile under target. (2) Adult (0.7956) and Law-School (0.7997) should barely move —
their plain quantile was already on target, so the conformal rank lands on essentially the same score,
and selective risk, worst-group risk, and AUROC on those two should be nearly *identical* to step 1
(0.069/0.088/0.853 on Adult; 0.369/0.414/0.614 on Law-School). (3) AUROC should not move on any dataset,
because the order is unchanged — only the cut moves. (4) The deferral-rate gaps should stay large and
roughly equal to step 1's, because a pooled threshold cannot equalize per-group coverage; if instead the
gaps collapsed, that would mean my "the conformal step only repairs the marginal cut" story is wrong. The
conformal rung is the principled-cut sibling of the confidence floor: same score, same accept-test shape,
a finite-sample guarantee bolted onto the threshold — and a deliberate refusal to pretend it has touched
the subgroup problem, which is what hands the baton to the rungs that change the score and the cut's
group structure (the full scaffold fill is in the answer).
