CMA-ES confirmed the diagnosis in the most uncomfortable way: the more sophisticated method *lost*. Final
scores held the competitive band — XGBoost −0.4005, SVM 0.9778, NN −3034, all roughly random search's —
but the convergence AUC collapsed, and exactly where I said it would. SVM AUC fell to 0.6644 mean with a
single seed at 0.2241; XGBoost AUC dropped to 0.7373 from random search's 0.9458. That 0.2241 seed is the
whole story in one number: a CMA generation sampled wide under σ₀ = 0.3, scored nothing competitive for
several evaluations, and the best-so-far curve sat flat while the area under it bled out — and with only
~6–10 generations in a 40-evaluation budget there was no time to recover. The lesson is sharp and it is not
about CMA-ES specifically: a single-fidelity optimizer that pays an up-front model-learning cost cannot
amortize it under a few-dozen-evaluation budget, because *every* evaluation costs a full unit and there are
too few of them. The fix is therefore not a better continuous model. It is to stop paying full price for
every look. If a cheap, noisy approximation of the objective exists, I can triage many configurations
cheaply and reserve the expensive full evaluations for the survivors — buying far more than 40 effective
trials out of the same budget.

The scaffold hands me exactly this lever and I ignored it for two rungs: the fidelity parameter. Each
objective accepts `budget ∈ (0,1]` and scales its cost by it — XGBoost trains fewer trees, SVM uses fewer CV
folds, the NN runs fewer iterations — and the loop charges `total_cost += fidelity`, so an evaluation at
fidelity ⅓ costs a third of a unit and I get three of them for the price of one full look. This is why the
multi-fidelity baselines show `total_evals` far above the budget: the same budget of 40–50 cost-units stretches
to a hundred or more actual evaluations. The cheap scores are noisy but correlated with the true one — a
config that is hopeless after a third of its trees is almost certainly hopeless at full — so I can rule out
the obviously bad early and concentrate spend on the promising.

Successive halving is the clean way to triage: sample N configs, evaluate all at the lowest fidelity, keep
the top 1/η, multiply the fidelity by η, repeat to the top. Survivors get exponentially more resource; the
full evaluation is paid only for a handful. But SH has one input it cannot set for itself — N, the
configurations-versus-fidelity tradeoff. Go wide and cheap (large N) and I triage aggressively, which is
right only if the cheap fidelity predicts the expensive one well; if the correlation is weak, I will discard
at low fidelity exactly the config that would have won at full. Hyperband hedges by running SH at several
starting fidelities, but I will reach for that next rung; here I want to address a *different* weakness that
both SH and Hyperband share — they sample every configuration uniformly at random. They are brilliant at
*allocating* budget and *killing* weak configs early, but they never use the outcome of one evaluation to
decide where to look next, so their quality is capped by random sampling. CMA-ES learned from history but
couldn't triage; SH triages but doesn't learn. I want both: model-free learning *and* multi-fidelity
scheduling.

CMA-ES learned from history by fitting a covariance — a model — and that model cost too much to amortize.
The alternative is a *model-free* way to learn from history, and differential evolution is exactly that. It
keeps a population of vectors in the unit box and improves them generation by generation with three
operators. Mutation, DE/rand/1: pick three distinct population members and form a mutant `v = x_a +
F·(x_b − x_c)`. Binomial crossover: build a trial by taking each coordinate from `v` with probability `Cr`,
from the target otherwise, forcing at least one coordinate from `v` (a random index `j_rand`) so the trial is
never a pure copy. Selection: keep the trial over the target iff it scores at least as well. The genius is
the difference vector: when the population is spread out, `x_b − x_c` is large and the search explores; as it
converges, the difference vectors shrink and the search exploits — the step scale *self-adapts to the
population's own spread*, with no learning rate and no smoothness assumption. And because DE only ever
*compares* scores, it is indifferent to whether a coordinate is continuous, integer, or categorical: keep the
population in `[0,1]^D` and decode to the real (possibly discrete) space only at evaluation, and DE searches a
smooth box while the world stays discrete. Its per-step cost is constant — it never grows with the number of
evaluations the way a covariance update or a KDE fit does. It is the anti-CMA-ES learner: cheap, model-free,
discrete-friendly.

So the method this rung fills in is DE *inside* Hyperband's multi-fidelity skeleton, derived in the order the
pieces become forced by the scaffold. First the fidelity ladder. With η = 3 and `s_max = min(3,
floor(log_η(total_budget)))`, the rungs are `1/η^s` for s from s_max down to 0 — but two of those, after the
loop's clip to [0.1, 1.0], both collapse near 0.1 (e.g. 0.037 → 0.1 and 0.111 are nearly the same), so I
*dedupe* on the rounded fidelity and keep a strictly increasing ladder. This is a real adaptation to the
harness: the floor of 0.1 means I cannot have arbitrarily many distinct cheap rungs, so wasting an SH round
on two near-identical fidelities would just burn budget. Each surviving fidelity gets its own population of
`max(4, dim+1)` configs, sampled uniformly and queued for evaluation at that fidelity. Second, the
DE-with-inheritance loop. When a trial returns, I match it to its pending entry by fidelity-and-vector and do
DE selection against the target it was mutated from (an initial-population eval just fills in its score). Once
the lowest-fidelity population is fully scored, I evolve it: for each target, mutate-and-crossover a trial,
queue it at the low fidelity, and increment that fidelity's generation counter. Third — and this is DEHB's
signature departure from plain Hyperband — promotion. Every time the source fidelity completes a fresh
generation, the top `1/η` of its (already-evolved) population is *promoted* to seed the next fidelity's
population, re-evaluated there, gated by a per-fidelity generation counter so promotion happens at most once
per fresh source generation and the queue cannot grow unbounded. The crucial point is that the higher
fidelities inherit DE-evolved configurations rather than restarting from random samples, so the search at
every fidelity benefits from the learning done at the cheaper ones — the budget-allocation of SH and the
history-learning of DE share the same population. Fallback to a random full-fidelity draw only if the queue
ever empties. The distilled module is in the answer.

Now where will this actually land relative to CMA-ES and random search, and what should I be nervous about?
The wins should be on *convergence AUC*, the metric both prior rungs were weakest on, because triage means a
decent config surfaces from the cheap rungs fast and the best-so-far curve climbs early instead of sitting
flat — the opposite of CMA-ES's wide-then-slow opening. I expect DEHB's AUC to beat CMA-ES comfortably and
to be at least competitive with, probably above, random search on the harder benchmarks. I am specifically
nervous about two things. First, the AUC is computed on the *min-max-normalized* best-so-far curve, and DEHB
spends a lot of cheap evaluations — many low-fidelity scores that are noisy and can be poor — so the
normalization floor (the worst score seen) may sit very low and the curve's early portion may look ragged;
the convergence_auc can even exceed 1.0 here, which the DEHB numbers will show, an artifact of how the
normalized curve integrates when many cheap noisy evaluations bracket the good ones. Second, the *cost
accounting*: because low-fidelity evals are cheap, DEHB runs many more actual evaluations than the
single-fidelity baselines (its `total_evals` should be far above 40–50, sometimes into the hundreds when many
cheap rungs fire), and on a benchmark where the cheap fidelity predicts the expensive one *poorly* — the NN,
where 50 iterations may rank configs differently from 500 — aggressive low-fidelity triage could promote the
wrong survivors and the final best score could actually *dip* below random search's.

So the falsifiable expectations against the prior numbers are: convergence AUC should rise sharply versus
CMA-ES (whose SVM AUC was 0.6644 with a 0.2241 seed) and beat random search's 0.789/0.772 on SVM and NN,
because triage front-loads good configs; `total_evals` should climb well past the budget, confirming the
cheap evaluations are being spent; and the final best scores should stay in the competitive band but with a
real risk of the NN best *worsening* relative to random search if low-fidelity rank-correlation is weak there.
If I see exactly that — strong AUC, inflated total_evals, an NN best that slips while SVM/XGBoost AUC jump —
that is multi-fidelity working as designed, with the low-fidelity-correlation caveat exposed, and it points
at the next question: the residual weakness is that DEHB and SH alike *guess* the configurations-versus-
fidelity tradeoff. The next rung should stop guessing it and hedge across the whole spectrum.
