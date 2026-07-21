DEHB behaved exactly as multi-fidelity should and exposed its own price in the numbers. The convergence
AUC jumped where the prior rungs were weakest — SVM 0.9813 (versus CMA-ES's 0.6644 and random search's
0.7885, a gain of 0.317 over CMA and 0.193 over random), XGBoost 0.9492 (past random search's 0.9458 and CMA's
0.7373) — and `total_evals` ballooned to 104 on XGBoost, 246 on SVM, 92 on the NN, far past the 50/40/40
budgets, which is the cheap-fidelity lever working: the same cost stretched to many more actual looks, and
triage front-loaded good configs so the best-so-far curve climbed early. The total_evals also swung across
seeds exactly as I predicted from DE's data-dependent promotion cascade — SVM ran 65, 354, 320 across the
three seeds, a 5.4× spread between min and max, where a single-fidelity method would have read a flat 40. But
three things in the table confirm the caveat I flagged, and I should read each mechanically. First, the AUCs
over 1.0: they land on the NN (1.0035, 1.0079) and one XGBoost seed (1.0031), and *not* on SVM (all three SVM
seeds 0.9733/0.9820/0.9887 sit below 1.0). That localization is telling — the overshoot is the normalization
artifact, and it appears where the cheap fidelity is noisiest: with many cheap noisy low-fidelity scores
bracketing the good ones the min-max floor sits very low and the integrated best-so-far curve overshoots, so
AUC there is partly measuring how the cheap evaluations stack up rather than pure convergence. Second, the NN
final best *slipped in the mean* — −3048.7, with seed 42 at −3086 (worse than random search's −3070) — but
the real shape is variance, not a uniform loss: the three NN seeds read −3086, −3062, −2998, a range of 88,
and seed 456 (−2998) actually *beat* random search's −3067. So triage did not make the NN uniformly worse; it
made one seed crater while another improved, which is the signature of promoting the wrong survivor when the
50-iteration rank disagrees with the 500-iteration truth. Third, and this is the quiet one, even on SVM —
where AUC soared and the cheap fidelity should be faithful — the *final best* dropped from 0.9778 to 0.9661
(seeds 0.9789/0.9561/0.9631), because triage spends so few full-fidelity looks that the true 5-fold optimum is
sometimes never evaluated. DEHB's residual weakness is structural, and it is the one every fixed
multi-fidelity schedule shares: it *guesses* the configurations-versus-fidelity tradeoff, and where the cheap
fidelity lies — or even where it is honest but the full-fidelity looks are too few — it lets the eventual
winner slip.

So before I reach for the full multi-fidelity hedge, let me isolate a different ingredient that all three
prior rungs were missing in their full form. CMA-ES learned from history but with a model too expensive to
amortize. DEHB learned from history but *model-free* (DE), and inherited the low-fidelity-correlation risk.
Neither built a clean *probabilistic model of where good configurations live* that I could trust at full
fidelity. The question this rung answers is narrow: if I commit to single-fidelity evaluation (no triage
risk at all) but spend each full evaluation on the configuration a *model* says is most promising, how far
does pure model-guided search get? This is the model component that the strongest combination later will
graft onto a multi-fidelity skeleton — so getting it right in isolation matters.

The template for model-guided search is sequential model-based optimization: keep a cheap surrogate of the
expensive loss built from the history, and each round maximize an acquisition over it to pick the next point,
evaluate, append, refit. The expensive evaluation happens once per round, so everything else must be cheap
next to it. Two choices define the method: what surrogate, and what acquisition. I want Expected Improvement
as the acquisition because it balances exploit against explore without a hand-set target: given the best
score y* seen and a surrogate's predictive distribution over the value Y(x) at a candidate, EI is the
expected amount by which x would beat the incumbent — outcomes worse than y* contribute zero improvement, and
the expectation over the surrogate's predictive distribution automatically rewards both high predicted score
(exploit) and high predictive spread that could clear y* (explore). It is the standard acquisition precisely
because that exploit/explore trade falls out of one integral with no tuning, so I commit to EI and spend the
design effort on the surrogate, which is where the space's structure actually bites. The textbook surrogate
is a Gaussian process, and it is the same one I turned away from before, for the same reasons that bind
harder here as an isolated model: a GP kernel needs a metric on the whole configuration vector that the
categorical axis (SVM kernel, NN activation) does not provide, and EI under a GP stakes all exploration on a
single predictive-variance estimate that a sparse 40-evaluation sample can silently collapse to near-zero.

The obvious escape from the GP's metric problem is a *random-forest* surrogate — the SMAC-style regression
forest that predicts a mean and a bootstrap variance and feeds them to EI. It is genuinely tempting because a
tree splits on categorical axes natively, so the SVM `kernel` and NN `activation` axes stop being a problem,
and fitting is O(n log n) rather than O(n³). But walk it a few steps and the arithmetic turns me back. A
forest still solves the *forward* regression problem — it must predict a calibrated y at each x — and it does
so by averaging leaves, so with only 10–40 observations the trees are stumps: a 40-point sample split three
or four ways per axis leaves 2–4 points per leaf, and the bootstrap-variance estimate EI relies on is then
almost pure resampling noise, not a real uncertainty. Worse, a forest's prediction is *piecewise constant*,
so between the sampled points it has no gradient and EI sees flat plateaus punctuated by steps — a poor
surface to maximize an acquisition over when I have tens of points, not thousands. The forest fixes the
categorical axis but keeps the deeper mismatch: it is estimating a whole calibrated response surface when all
I need out of the model is a *ranking* of candidates. That mismatch is the thread to pull.

Stare at the EI integral and ask what it actually needs, because the GP may be solving a harder problem than
EI requires. EI needs p(y|x): the distribution over loss for any candidate. The GP models the forward map
x → distribution-over-y directly, which is why it needs a metric on x. But I never need a calibrated number —
I draw candidates and *rank* them by EI. The data I have a lot of is the other direction: for each trial I
have (config, score), samples of the joint. The GP factors the joint as p(y|x)p(x); what if I factor it the
other way, p(x|y)p(y), and model p(x|y) — the density over *configurations* given the outcome? That is the
direction my data is easy in: group the configs by outcome and estimate a density over configs within each
group, with a per-coordinate kernel that composes natively over mixed types — no whole-vector metric needed.

Split the outcome at a quantile: pick γ and set the threshold y* so a fraction γ of observations fall on the
good side, then estimate two densities — `l(x)` over the configs whose score is in the top γ, `g(x)` over the
rest. Push EI through this factorization with Bayes (p(y|x) = p(x|y)p(y)/p(x)) and the algebra collapses to a
clean result. Writing EI for a maximization problem against incumbent y*, the improvement integral over the
good region factors because on that region p(x|y) is, by construction, exactly `l(x)` and pulls out of the
y-integral as a constant in x; the denominator p(x) is the total mixture `γ·l(x) + (1−γ)·g(x)` by total
probability over the same split; and what survives is `EI(x) ∝ (γ + (1−γ)·g(x)/l(x))^{-1}`, which is strictly
*increasing* in `l(x)/g(x)`. So EI(x) is a monotone function of the ratio `l(x)/g(x)`, and maximizing EI is
*exactly* maximizing that ratio — pick the config most likely under the good density relative to the bad one.
This is the payoff of the backwards factorization: I never had to build a regressor on x, never needed a
whole-vector metric, and the acquisition reduced to a ratio of two densities I can estimate directly from the
grouped history. The quantile choice y* is forced by the construction: GP-EI uses y* = the best observed
score, but then the good set would be a single point and `l` would have no data to fit, so I must use a
quantile γ instead, which also becomes the explore/exploit knob — larger γ admits more configs into the "good"
set (more exploration, since `l` then covers more of the space), smaller γ is greedier (a tighter elite). The
γ = 0.25 this rung uses is more exploratory than the classic 0.15, and the arithmetic under a tiny budget is
what forces the bias. Just past the 10-config warm-up, at a history of 11, `n_good = max(1, int(0.25·11)) =
int(2.75) = 2` — a two-point good set, already thin but fittable. The classic γ = 0.15 would give `int(1.65)
= 1`, a *single-point* good density, and a KDE on one point is a lone bump that concentrates the whole ratio
on one location and kills exploration exactly when I have the least data. Even at a history of 20, γ = 0.25
gives 5 good points against γ = 0.15's 3. So the more exploratory quantile is not a preference; it is the
choice that keeps `l(x)` from degenerating in the early, data-starved regime that dominates a 40-evaluation
run.

Now the implementation the scaffold fills in, and where it departs from the clean ideal. Encode each config
to a numeric vector in [0,1] (log-linear for scale knobs, and the categorical to its choice-index over the
number of choices minus one — a scalar relaxation, the same lossy flattening I accepted for CMA-ES, fine here
because EI only ranks and the continuous knobs dominate). Run `n_startup = 10` uniform-random configs first,
because below a handful of observations the densities are meaningless and I should just explore. Then split:
`n_good = max(1, int(γ·len(history)))` with γ = 0.25, threshold at the `n_good`-th best score. Estimate `l`
and `g` as simple Gaussian KDEs — `log p(x) = log(mean_i exp(−½‖x−x_i‖²/bw²) + ε)` — with a *single global
bandwidth* per density set by a Scott-style floor, `bw = max(0.05, samples.std() + ε)`. This is the key
simplification to be honest about: it is not the per-point adaptive bandwidth of the full method, just one
scale per density, which under 40 observations is a reasonable bias-variance trade (per-point bandwidths need
more data to estimate) but does smear fine structure. Finally, optimize EI by *sampling*: draw
`n_ei_candidates = 24` uniform configs, score each by `log l(x) − log g(x)`, and return the argmax — sampled
optimization rather than gradient ascent, which sidesteps the categorical axis having no gradient and is
cheap at 24 candidates. Every suggestion is full fidelity 1.0; this rung deliberately uses *no* multi-fidelity,
to measure the model in isolation. The distilled module is in the answer.

The bandwidth floor of 0.05 is a design choice worth motivating, because it is where the model's resolution
lives. In the unit box, bw = 0.05 gives each bump a width of about 10% of an axis, so ten good points tile
[0,1] without heavy overlap. A large bandwidth (0.5) flattens `l(x)` and `g(x)` toward the same constant —
`log l − log g → 0` and TPE degenerates to random search; a tiny one (0.005) makes each point an isolated
spike and any candidate not landing on a good point reads `l ≈ 0`, overfitting. The 0.05 floor is the
compromise, and because it is a *floor*, once the observations' std exceeds it the bandwidth grows with the
data.

This same bandwidth arithmetic is what quietly rescues the lossy categorical encoding. The SVM `kernel` axis
has three choices mapped to {0, 0.5, 1.0} and the NN `activation` similarly, and the scalar relaxation
imposes a spurious ordering — it pretends kernel-index 0 is "nearer" index 1 than index 2. But the inter-level
gap is 0.5, which is ten bandwidths at bw = 0.05, and the cross-level Gaussian weight is `exp(−0.5·0.25/0.0025)
= exp(−50) ≈ 0`. So on the categorical coordinate the KDE puts essentially zero mass across levels: a good
config using the RBF kernel contributes to `l` only at the RBF level and not at the others. The density on
that axis therefore collapses to a per-level count — how many good versus bad configs used each choice — which
is exactly the categorical model I would have wanted, recovered for free from the continuous KDE because the
bandwidth is far smaller than the inter-level spacing. The relaxation is lossy in principle but graceful in
practice on this substrate.

The continuous axes get the opposite, careful treatment: the scale knobs are encoded log-linearly, and that
choice matters for where the KDE places its mass. The XGBoost `learning_rate` is log-scaled over its range, so
a config at 0.01 does not encode to some tiny fraction crushed against zero — it encodes to `(log 0.01 −
log low)/(log high − log low)`, which for a plausible [0.001, 0.3] range is `(−4.605 + 6.908)/(−1.204 +
6.908) = 2.303/5.704 = 0.40`, sitting squarely in the interior of the box. Encoding in the native linear
scale would pile all the sub-0.05 learning rates into the bottom 15% of the axis and the KDE bandwidth of 0.05
would then blur them together into one indistinct bump. Log-encoding spreads the decades evenly, so the model
resolves an order-of-magnitude difference in learning rate as a healthy 0.4-wide gap in encoded space, which
is many bandwidths and cleanly separable — the encoding is doing real work to make the density model see the
structure the loss landscape actually has.

Optimizing EI by sampling 24 candidates is a *selection ratio*: each full-fidelity evaluation is the argmax
of `log l − log g` over 24 fresh draws under the current model, so against random search's best-of-one, TPE
evaluates a best-of-24 under a model already concentrated on the good region. The acquisition costs a few
hundred Gaussian weights, trivial against a train-and-score, and pushing it to hundreds of candidates would
sharpen the argmax only marginally. Sampling also sidesteps the categorical axis having no gradient to ascend
along a choice-index — so best-of-24 is both the cheap and the correct way to optimize on a mixed box.

Now where should this land against DEHB's numbers? TPE's strength is *final quality* and *low-variance*
convergence, because it spends every full-price evaluation on a model-chosen config rather than a cheap noisy
look or a blind DE trial — so I expect its NN final best to *recover* from DEHB's slip (no low-fidelity
mis-promotion can hurt it, since there is no low fidelity), landing back near random search's band rather than
DEHB's −3086 seed. But its convergence AUC should be *lower* than DEHB's inflated multi-fidelity numbers,
and for a clean reason I can put a number on: DEHB got many cheap evaluations and an AUC artifact above 1.0;
TPE gets only 40–50 *full* evaluations, and the 10-config random warm-up before the model turns on is 10 of
40 on SVM and the NN — a flat 25% of the budget spent exploring blind — and 10 of 50, 20%, on XGBoost. During
that quarter of the run the best-so-far curve climbs only as fast as random search's, so a large slice of the
AUC integral is already ceded before the model makes a single aimed suggestion. That front-loaded dead weight
is why I expect TPE to *trail DEHB on convergence AUC* (SVM well below DEHB's 0.9813, NN below DEHB's 0.9351)
while being *more reliable on final best* — the inverse of DEHB's profile. And because the warm-up fraction is
largest on the two 40-budget benchmarks, I expect the AUC gap to DEHB to be widest there and narrowest on the
50-budget XGBoost.

The falsifiable expectations against DEHB: `total_evals` back to a flat 50/40/40 with zero seed-to-seed
variance (single fidelity, the tell of no triage — the opposite of DEHB's 65-to-354 SVM swing); NN final
best recovered toward random search's level; but convergence AUC distinctly below DEHB's, because the
10-config warm-up plus full-price evaluations cannot front-load the curve the way cheap triage did. That
profile states the missing ingredient cleanly: a good model alone fixes final quality and reliability but
cannot match the anytime convergence cheap multi-fidelity buys. The two strengths are complementary, which
sets up the rungs that follow — hedge the configs-versus-fidelity tradeoff across the spectrum next, then put
a model *inside* that hedge so the best of both compound.
