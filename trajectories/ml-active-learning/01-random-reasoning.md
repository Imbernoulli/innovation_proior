The only knob I own is the acquisition rule. The harness hands me a pool of `n_pool` tabular inputs sitting
there for free, a boolean mask of which ones I have already paid to label, a small neural classifier it will
retrain from scratch each round, and one empty slot — a `query(n)` that must return `n` indices into the pool,
chosen from the still-unlabeled part, that I will pay the oracle to label. Everything else is frozen: the model
family, the optimizer, the retrain-each-round loop over 20 rounds, the data handling. So the whole research
question collapses onto that one method body — given the current model and the pool, which `n` unlabeled
examples do I send out — and before I reach for anything that uses the model I should nail down the rule whose
behavior I understand completely, the one that is the *floor* by construction, the yardstick every cleverer rule
has to beat. That rule is the one that uses nothing: no model, no labels seen so far, no notion of uncertainty.
It just draws representatively. So at step 1 my edit to the scaffold is the trivial one — leave `query(n)` at
its default, a uniform draw from the unlabeled pool — and the work is understanding *why* this is the right
place to start and exactly where it will be weak, so I know what the next rung has to fix.

Start from the only thing I can prove about *any* acquisition rule, because that is what tells me what the floor
should even be. The single theorem that makes supervised learning work is the bridge from training error to test
error. The thing I actually care about is the test error of a hypothesis `c` — the probability it disagrees with
the true labeling on a fresh input drawn from the deployment distribution `P`, `err(c) = Pr_{x~P}[c(x) ≠ t(x)]`.
I never observe that directly; I only ever see the empirical error on my labeled set. The reason fitting the
labeled set tells me anything about `err(c)` is uniform convergence: a hypothesis consistent with `m` labeled
examples has `err ≤ ε` with confidence `1−δ` once `m = O((1/ε)(d ln(1/ε) + ln(1/δ)))`, `d` the VC dimension —
and that PAC bound rests on exactly one premise about *where the training examples come from*: they are an i.i.d.
sample from the very distribution I will be tested on. The sample is the bridge. If my labeled set is distributed
like `P`, low training error transfers to low test error; if I bias the draw, the bridge is gone and the
guarantee evaporates. So whatever `query(n)` I write, I had better be able to state what distribution my labeled
set ends up following, because that distribution is the thing the entire generalization story hangs on.

That immediately tells me what the *safest* possible rule is. If I want the labeled training set to stay
distributed like `P`, the cleanest way to keep that premise intact is to not let the model, the features, or the
labels seen so far tilt the draw at all — to make the selection independent of the data values. The pool itself
came from `P`; conditional on the realized pool I can only promise representativeness with respect to that finite
pool, but unconditionally, a value-independent draw from a pool that was itself drawn from `P` inherits `P`. What
does "value-independent" mean precisely? It means every still-unlabeled pool input should be equally likely to
be picked — no input preferred over another. Let me make sure that is the right formalization and not just a
vibe, because this is an old solved problem wearing a different hat. Survey sampling — Cochran's *Sampling
Techniques* — is entirely about picking a subset of a finite population so that quantities computed on the subset
estimate the whole population well, and its foundational design is simple random sampling without replacement:
choose `n` of the `N` units so every size-`n` subset is equally likely. Define the inclusion probability
`π_i = Pr[unit i is in the sample]`. By symmetry, unit `i` appears in `C(N−1,n−1)` of the `C(N,n)` possible
subsets, so `π_i = C(N−1,n−1)/C(N,n) = n/N`, the same for every `i`. Equal inclusion is exactly the property I
need, and I can say precisely why. Take any statistic I would want the labeled set to estimate about the pool —
class balance, the fraction in some region, anything expressible as a population average of `g(x_i)`. The general
unbiased estimator under unequal inclusion is Horvitz–Thompson, weighting each sampled unit by `1/π_i`; its
expectation is the population average because each unit contributes `g(x_i)/π_i` with probability `π_i`. When all
`π_i = n/N` are equal, that estimator collapses to the plain unweighted sample mean, and it stays unbiased for
every population average. So equal inclusion is not an aesthetic preference — it is the exact condition under
which the plain labeled subset is an unbiased miniature of the pool. The design survey sampling calls
"representative" is the design learning theory needs for its guarantee to hold; the two solved problems line up.

Should I sample with replacement or without? A label I already hold teaches me nothing the second time — I have
paid for it once, the label is the same — so a repeat in a batch is a wasted budget slot. And without-replacement
is also statistically tighter, and I can quantify it: the sample-mean variance under simple random sampling
without replacement is `(σ²/n)·(N−n)/(N−1)`, versus `σ²/n` with replacement, the extra factor `(N−n)/(N−1) ≤ 1`
being the finite-population correction. Without replacement strictly reduces variance, and the reduction is
largest exactly when `n` is an appreciable fraction of `N` — and drives variance to zero when `n = N`. So without
replacement dominates on both counts, no wasted budget and a tighter estimate, and I want it over the *unlabeled*
part of the pool only, because the already-bought labels would buy zero information if re-selected. That is
exactly what the scaffold default does: take the pool indices where the mask is False, permute them uniformly,
take the first `n`. A uniform permutation makes every ordered prefix equally likely, so every size-`n` subset
comes up with equal probability and the picks are automatically distinct — a uniform without-replacement draw,
every unlabeled input with inclusion probability `n/(#unlabeled)`, equal across them. One size-`n` draw per
round, no tuning, nothing data-dependent that can go wrong.

The harness even hands me, for free, the one thing that plagues adaptive rules: the diversity headache. The loop
retrains in rounds and asks for a batch of `n` at a time, to amortize retraining. For an adaptive rule that is a
real complication — greedily pick the `n` most uncertain points and they are often near-duplicates clustered in
one spot, so the batch gets wasted and the rule has to engineer diversity in. For the uniform rule there is no
such concern: a uniform draw of `n` distinct points is already spread across the pool in proportion to the pool,
and because the rule consults no model, drawing `n` at once has the same law as drawing one, marking it labeled,
retraining, and drawing again until the batch is filled. The selections inside the batch are dependent — that is
what without-replacement means — but that dependence is precisely what forbids duplicates. So the batch size is
free here, with none of the diversity headaches; another small gift of being model-free.

Now stare at where this will be weak, because that is the entire point of running it — the failure modes are
knowable in advance from the structure of the rule, and they tell me what rung 2 must attack. The first bites
when a class is rare. A uniform draw faithfully reproduces the pool's class balance in the labeled subset — that
is the "representative" property I just celebrated — so if the interesting class is 1-in-1000, a budget of 500
labels buys roughly 500 of the common class and essentially none of the rare one, and a classifier cannot learn
a class from a sample containing none of it. The very faithfulness that makes the rule unbiased is what makes it
starve the minority: representativeness reproduces the imbalance, and when the rare class is the one I care about,
reproducing the imbalance is the opposite of what I want. The second weakness is subtler and grows with the
budget. Picture the set of hypotheses still consistent with everything I have labeled so far; somewhere in input
space there is a region where these survivors still disagree — the region of uncertainty `R`. An input outside
`R` is already determined: every survivor labels it the same way, so labeling it cannot change my hypothesis and
teaches me nothing. Only inputs inside `R` can move the model. Let `α = Pr[x ∈ R]`. Early on `R` is large and
`α` is healthy; as labels accumulate `R` shrinks and `α` decays toward zero — and a uniform draw, oblivious to
`R`, lands in that useful region with probability exactly `α`. So late in training most random draws re-confirm
what the model already knows, and the information per label decays. In the cleanest case — locating a single
threshold on `[0,1]` — random labeling needs `O((1/ε)ln(1/ε))` examples while a learner that *chooses* where to
ask (binary search) needs only `O(ln(1/ε))`: an exponential gap, `1/ε` versus `ln(1/ε)`. That gap is the prize a
model-aware rule is chasing, and it is exactly what every later rung will spend the model on.

I should be honest about why I am not just reaching straight for a rule that targets `R`, because the temptation
is real and the reason I resist it is the same reason random is the right floor. The moment I bias selection
toward `R` instead of toward `P`, I break the property that gave me my only unconditional guarantee — the labeled
set is no longer an i.i.d. sample of `P`, so the PAC bridge no longer applies and any improvement becomes
conditional. Worse, a boundary-chasing rule reads the *current* model to decide what is uncertain, and early in
training that model is fit on almost nothing and is probably wrong, so it can define `R` badly and steer the
whole budget into an unrepresentative corner — a self-reinforcing sampling bias the uniform rule structurally
cannot have, precisely because it ignores the model. There is also the membership-query route — synthesize the
single most informative input — which is what buys the `ln(1/ε)` rate, but in the pool setting `query(n)` must
return *indices into real rows*, I cannot fabricate a query, and a synthesized "most informative" input could be
an uninterpretable artifact no oracle can label. So every cleverer rule trades away something the uniform rule
keeps for free: the unbiasedness, the assumption-free guarantee, the immunity to a bad early model, the
robustness to model misspecification. That is what makes this the honest reference — not despite throwing away
information, but because of it: the one rule whose statistical behavior I can state without conditions, the
learning curve any adaptive strategy must dominate across the budget range to justify the information it spends.

Across the three datasets I expect these two weaknesses to show up differently, and that shape is the prediction
I am setting up for the next rung. On **spambase** (binary, fairly balanced) a representative i.i.d. sample is
genuinely hard to beat — the boundary is simple, the classes are not pathologically imbalanced, and the budget
is a healthy fraction of the pool so the finite-population correction is actively helping; random should be
*strong* here, and a model-aware rule might barely improve on it. On **letter** (26 classes, the largest pool)
the budget is spread thin across many classes and the decaying-`α` problem bites hardest — random keeps re-buying
easy, already-separated letters while the confusable pairs (the boundary mass) stay under-sampled; this is where
a rule that chases the boundary should open the largest gap, so I expect random to be relatively *weak* on
letter, and I also expect its run-to-run variance to be widest there, because where a blind draw ends depends on
which of the 26 classes it happened to cover. On **splice** (3-class) it should sit in between. So the prediction
is sharp: random's curve is the honest reference, hardest to beat on the easy balanced binary problem and easiest
to beat where many classes and a thin budget make most random draws land in settled territory. The first thing I
will change at rung 2 is to stop drawing blindly and start letting the model say which unlabeled points it is
least sure of — turning the empty, model-free `query` into one that reads `self.predict_prob` and spends the
budget on the contested region instead of the pool's bulk. The falsifiable expectation the next rung must beat:
a model-aware rule should lift letter clearly above random's curve (that is where the adaptivity prize is
largest), gain little or nothing on spambase, and land a modest gain on splice — and if it fails to beat random
on letter, the premise that the model's boundary is informative is wrong. The distilled rule and the literal
scaffold fill are in the answer.
