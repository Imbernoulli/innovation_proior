DEHB behaved exactly as multi-fidelity should and exposed its own price in the numbers. The convergence
AUC jumped where the prior rungs were weakest — SVM 0.9813 (versus CMA-ES's 0.6644 and random search's
0.7885), XGBoost 0.9492, several seeds reading *above* 1.0 (1.0031, 1.0079) — and `total_evals` ballooned to
104 on XGBoost, 246 on SVM, 92 on the NN, far past the 50/40/40 budgets, which is the cheap-fidelity lever
working: the same cost stretched to many more actual looks, and triage front-loaded good configs so the
best-so-far curve climbed early. But two things confirm the caveat I flagged. The AUCs over 1.0 are the
normalization artifact — with many cheap noisy low-fidelity scores bracketing the good ones, the min-max
floor sits very low and the integrated curve overshoots — so AUC is now partly measuring how the cheap
evaluations stack up, not pure convergence. And the NN final best *slipped*: −3048.7 mean, with seed 42 at
−3086 (worse than random search's −3070), which is precisely the weak-rank-correlation failure I worried
about — at low fidelity the NN's 50-iteration scores rank configs differently from the 500-iteration truth,
so aggressive triage promoted the wrong survivors. DEHB's residual weakness is structural, and it is the one
both SH and Hyperband share: it *guesses* the configurations-versus-fidelity tradeoff, and where the cheap
fidelity lies, it throws away the eventual winner early.

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
design effort on the surrogate, which is where the space's structure actually bites. The textbook surrogate is a
Gaussian process, and it is elegant — analytic posterior mean and variance, closed-form EI — but it is the
wrong fit for *this* space, and naming why is what forces the alternative. A GP kernel needs a metric on the
whole configuration vector, and my spaces have a categorical axis (SVM kernel, NN activation) with no natural
metric; conditioning costs O(n³); and EI under a GP stakes all exploration on a single point estimate of the
predictive variance, which a sparse early sample (my normal starting condition at 40 evaluations) can
collapse to near-zero, killing exploration silently.

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
γ = 0.25 this rung uses is more exploratory than the classic 0.15, which is the right bias under a tiny budget
where the good set must not collapse to two or three points and over-concentrate the ratio.

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

Now where should this land against DEHB's numbers? TPE's strength is *final quality* and *low-variance*
convergence, because it spends every full-price evaluation on a model-chosen config rather than a cheap noisy
look or a blind DE trial — so I expect its NN final best to *recover* from DEHB's slip (no low-fidelity
mis-promotion can hurt it, since there is no low fidelity), landing back near random search's band rather than
DEHB's −3086 seed. But its convergence AUC should be *lower* than DEHB's inflated multi-fidelity numbers,
and for a clean reason: DEHB got many cheap evaluations and an AUC artifact above 1.0; TPE gets only 40–50
full evaluations and a 10-evaluation random warm-up before the model even turns on, so a quarter of its
budget on the small benchmarks is pure exploration with a flat early curve. I therefore expect TPE to *trail
DEHB on convergence AUC* (SVM well below 0.98, NN below DEHB's 0.935) while being *more reliable on final
best* — the inverse of DEHB's profile.

The falsifiable expectations against DEHB are then specific: `total_evals` back to exactly the budget (50/40/
40 — single fidelity, the tell that the model-only rung is doing no triage); NN final best recovered toward
random search's level (no low-fidelity mis-promotion); but convergence AUC distinctly below DEHB's, because
the 10-config warm-up and full-price evaluations cannot front-load the curve the way cheap triage did. If I
see that — budget-pinned total_evals, decent and stable final scores, but AUC that loses to DEHB — the
diagnosis is the missing ingredient stated cleanly: a *good model alone* fixes final quality and reliability
but cannot match the *anytime* convergence that cheap multi-fidelity buys. The two strengths are
complementary, which is the whole setup for the rungs that follow: hedge the configs-versus-fidelity tradeoff
across the spectrum (next), and then put a model *inside* that hedge so the best of both compound.
