CMA-ES confirmed the diagnosis in the most uncomfortable way: the more sophisticated method *lost*. Final
scores held the competitive band — XGBoost −0.4005, SVM 0.9778, NN −3034, all roughly random search's —
but the convergence AUC collapsed, and exactly where I said it would. Let me read the two feedback tables
side by side and not wave at "collapsed," because the arithmetic tells me *which* collapse and *why*. On
XGBoost the AUC fell from random search's 0.9458 to 0.7373, a drop of 0.2085, a 22% relative loss. On SVM it
fell from 0.7885 to 0.6644, a drop of 0.1241, and the mean hides the real damage: CMA's three SVM seeds read
0.8408, 0.2241, 0.9283, a spread of 0.704, versus random search's 0.8899/0.5625/0.9132, a spread of 0.351.
CMA *doubled* the seed-to-seed variance on SVM, and the 0.2241 seed sits at less than a quarter of CMA's own
best SVM seed (0.9283). That single seed is the whole story in one number: a CMA generation sampled wide
under σ₀ = 0.3, scored nothing competitive for several evaluations, and the best-so-far curve sat flat while
the area under it bled out — with only ~6–10 generations in a 40-evaluation budget, there was no time to
recover.

But I have to be honest about the one benchmark that *didn't* collapse, because it sharpens the diagnosis
rather than muddying it. On the NN, CMA-ES's AUC was 0.9356 versus random search's 0.7725 — CMA *beat*
random search by 0.163 there, and its NN final best (−3033.8) edged random search's (−3050.3) by 16.5. So
the failure is not "models never help." It is benchmark-specific: CMA regressed on XGBoost and SVM and
improved on the NN. Averaged as mean-of-means the AUC comes out 0.779 for CMA against 0.836 for random
search, a net 0.057 loss — enough to rank behind random search, driven entirely by the two benchmarks where a
wide σ₀ = 0.3 opening wasted the early curve while the NN's loss surface happened to tolerate it. That
localization is the real lesson, and it is not about CMA-ES specifically: a single-fidelity optimizer that
pays an up-front model-learning cost cannot *reliably* amortize it under a few-dozen-evaluation budget,
because every evaluation costs a full unit, there are only tens of them, and whether the up-front cost pays
back depends on the luck of the opening. The fix is therefore not a better continuous model. It is to stop
paying full price for every look. If a cheap, noisy approximation of the objective exists, I can triage many
configurations cheaply and reserve the expensive full evaluations for the survivors — buying far more than 40
effective trials out of the same budget, and cutting the variance that a single unlucky wide generation just
inflicted.

The scaffold hands me exactly this lever and I ignored it for two rungs: the fidelity parameter. Each
objective accepts `budget ∈ (0,1]` and scales its cost by it — XGBoost trains fewer trees, SVM uses fewer CV
folds, the NN runs fewer iterations — and the loop charges `total_cost += fidelity`, so an evaluation at
fidelity ⅓ costs a third of a unit and I get three of them for the price of one full look. This is why the
multi-fidelity baselines show `total_evals` far above the budget: the same budget of 40–50 cost-units
stretches to a hundred or more actual evaluations. The cheap scores are noisy but correlated with the true
one — a config that is hopeless after a third of its trees is almost certainly hopeless at full — so I can
rule out the obviously bad early and concentrate spend on the promising.

Before I commit, let me be disciplined and walk the alternatives that are genuinely on the table now, because
"use fidelity" is not yet a method. Option one: keep single fidelity but swap CMA's covariance model for a
Gaussian-process surrogate with Expected Improvement. This is the textbook move and it is tempting, but it
fails on this substrate for reasons I can name concretely rather than by feel. A GP needs a metric over the
whole configuration vector, and two of my three spaces carry a categorical axis (SVM `kernel`, NN
`activation`) with no natural distance; conditioning is O(n³), and with multi-fidelity inflating the trial
count toward a hundred that is ~10⁶ operations per `suggest` call — survivable, but not the real problem. The
real problem is that a GP is *still single fidelity*: it pays a full unit per evaluation and hits the exact
amortization wall CMA just hit, and under a sparse 40-evaluation sample its predictive variance can collapse
to near-zero and silently kill exploration. So a better single-fidelity model does not address the failure I
just measured; discard it. Option two: pure successive halving with random sampling — evaluate N random
configs at low fidelity, keep the top 1/η, promote. This *does* attack the amortization wall, and it is the
minimal multi-fidelity move. But it throws away the one thing CMA got right: learning from history. SH picks
every config at random; the outcome of one evaluation never informs the next. I would be trading one half of
the problem for the other. Option three, which is where the arithmetic points: combine cheap triage with a
*model-free* way of learning from history, so I get both multi-fidelity allocation and history-driven
proposal without paying the covariance-fitting cost that just failed.

Successive halving is the clean way to triage: sample N configs, evaluate all at the lowest fidelity, keep
the top 1/η, multiply the fidelity by η, repeat to the top. Survivors get exponentially more resource; the
full evaluation is paid only for a handful. But SH has one input it cannot set for itself — N, the
configurations-versus-fidelity tradeoff. Go wide and cheap (large N) and I triage aggressively, which is
right only if the cheap fidelity predicts the expensive one well; if the correlation is weak, I will discard
at low fidelity exactly the config that would have won at full. The full hedge across several
starting fidelities is a further step I can take later; here I want to address a *different* weakness that a
plain successive-halving ladder still has — it samples every configuration uniformly at random. They are brilliant at
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
the difference vector, and it is worth pinning down with actual numbers. Take a two-dimensional slice, target
`x_t = [0.50, 0.50]`, and three population members `x_a = [0.20, 0.80]`, `x_b = [0.90, 0.10]`, `x_c =
[0.30, 0.60]`, with `F = 0.5`. Then `x_b − x_c = [0.60, −0.50]`, `F·(x_b − x_c) = [0.30, −0.25]`, and the
mutant is `x_a + [0.30, −0.25] = [0.50, 0.55]`; binomial crossover with a forced index then splices this with
`x_t`. Now watch the self-adaptation: the *magnitude* of the move, 0.30 and 0.25 in each coordinate, was set
entirely by how far apart `x_b` and `x_c` were. Early, when the population is spread across the box, a
difference like `[0.60, −0.50]` produces steps of order 0.3; late, once the population has converged so that
`x_b` and `x_c` differ by only `[0.05, −0.03]`, the same operator produces steps of order 0.02 — the search
*explores* when spread and *exploits* when converged, with no learning rate and no smoothness assumption, the
step scale reading off the population's own spread. And because DE only ever *compares* scores, it is
indifferent to whether a coordinate is continuous, integer, or categorical: keep the population in `[0,1]^D`
and decode to the real (possibly discrete) space only at evaluation, and DE searches a smooth box while the
world stays discrete. Its per-step cost is constant — O(D) for one mutation, roughly six multiplies here,
against the O(D²) covariance update CMA paid and the O(n·D) KDE fit a density model would need — it never
grows with the number of evaluations. It is the anti-CMA-ES learner: cheap, model-free, discrete-friendly,
and it self-adapts its step instead of tuning a σ that a single wide generation can wreck.

So the method this rung fills in is DE *inside* Hyperband's multi-fidelity skeleton, derived in the order the
pieces become forced by the scaffold. First the fidelity ladder, and here I have to do the arithmetic against
the harness rather than write down the textbook `1/η^s`. With η = 3 and `s_max = min(3,
floor(log_η(total_budget)))`, every benchmark lands on the same cap: `log₃ 50 = 3.56` and `log₃ 40 = 3.36`,
both floor to 3, so `s_max = 3` for all three. The raw rungs `1/3^s` for s = 3,2,1,0 are 0.037, 0.111, 0.333,
1.0, and the loop clips fidelity to a floor of 0.1, so 0.037 becomes 0.1 while 0.111, 0.333, 1.0 pass
through. I dedupe on `round(fid, 3)`: the keys are 0.1, 0.111, 0.333, 1.0 — all distinct — so the ladder is
`[0.1, 0.111, 0.333, 1.0]`, four rungs. It is worth being precise that at this cap the dedupe is *insurance,
not a binding constraint*: only s = 3 falls below the 0.1 floor, so only one rung is pushed up to it, and
0.111 stays clear of 0.1 under three-digit rounding. The guard would only bite if the cap were higher — an
`s_max ≥ 4` would push both `1/81 = 0.012` and `1/27 = 0.037` onto the same 0.1 floor, collapsing two SH
rounds into one wasted pair — which is exactly the pathology I am protecting against even though it does not
fire at η = 3, `s_max = 3`. This is a real adaptation to the harness: the floor of 0.1 means I cannot have
arbitrarily many distinct cheap rungs, so the dedupe keeps a strictly increasing ladder no matter how the cap
is set.

Each surviving fidelity gets its own population of `max(4, dim+1)` configs, sampled uniformly and queued for
evaluation at that fidelity — 7 for the 6-D XGBoost and NN spaces, 4 for the 3-D SVM space. Let me sanity-
check that this initialization even fits the budget before I trust the rest. The initial queue for XGBoost is
7 configs at each of the four fidelities, and its cost is `7·(0.1 + 0.111 + 0.333 + 1.0) = 7·1.544 = 10.8`
cost units out of the budget of 50 — so roughly a fifth of the budget seeds all four populations, leaving
~39 units for evolution and promotion, and the single full-fidelity rung (7·1.0 = 7 units) already dominates
that opening cost, which is the tell that most of the *count* of evaluations will come from the cheap rungs.
For SVM the opening is `4·1.544 = 6.2` of 40. Good — the ladder is affordable and the cheap rungs are where
the trial count balloons.

It is worth spending a moment on what each rung of that ladder actually *does* to each benchmark, because the
fidelity contract is spelled out in the scaffold and it decides where triage will be faithful and where it
will lie — and this is exactly the difference the NN risk hinges on. SVM scales its CV folds as `max(2,
int(5·budget))`. Run the four rungs through it: at 0.1, `int(5·0.1) = int(0.5) = 0`, floored to 2 folds; at
0.111, `int(0.555) = 0`, again 2; at 0.333, `int(1.665) = 1`, again 2; only at 1.0 does it reach `int(5) = 5`
folds. So on SVM the three cheap rungs are *all* 2-fold cross-validation and only the top rung is 5-fold — a
genuinely cheaper estimate but one whose ranking of configs should track the 5-fold ranking closely on a
clean binary problem like Breast Cancer, which predicts SVM is where triage should pay off most and safely.
The NN is the opposite. It scales `max_iter` as `max(50, int(500·budget))`, so the rungs are 50, 55, 166, and
500 iterations. An `MLPRegressor` at 50 iterations has barely begun to converge, and the configs that look
worst at 50 are frequently the ones with a small `learning_rate_init` or large `alpha` that are *slow but
better asymptotically* — precisely the configs a good optimizer wants, and precisely the ones cheap triage
will kill. The 50-versus-500 gap is the concrete face of the weak-rank-correlation risk I keep flagging.
XGBoost sits in between: fidelity scales `n_estimators` with a floor of 10, so the cheap rungs are shallow
ensembles that underfit but usually rank in the right order. This is the mechanism, read straight off the
contract, behind expecting SVM to win biggest, XGBoost to be safe, and the NN to be the benchmark where an
aggressive promotion can throw away the eventual winner.

Second, the DE-with-inheritance loop. When a trial returns, I match it to its pending entry by fidelity-and-
vector and do DE selection against the target it was mutated from (an initial-population eval just fills in
its score). Once the lowest-fidelity population is fully scored, I evolve it: for each target, mutate-and-
crossover a trial, queue it at the low fidelity, and increment that fidelity's generation counter. Third —
and this is DEHB's signature departure from plain Hyperband — promotion. Every time the source fidelity
completes a fresh generation, the top `1/η` of its (already-evolved) population is *promoted* to seed the
next fidelity's population, re-evaluated there, gated by a per-fidelity generation counter so promotion
happens at most once per fresh source generation and the queue cannot grow unbounded. Let me trace the
promotion accounting on the 4-member SVM populations to check it is well-defined and does not starve or
explode. The lowest rung holds 4 configs; `n_promote = max(1, 4 // 3) = max(1, 1) = 1`, so exactly one config
is promoted from 0.1 to 0.111, then one from 0.111 to 0.333, then one from 0.333 to 1.0. The generation gate
`_gen_count[hi] >= _gen_count[src]` refuses to re-promote until the source has advanced another generation,
so a single fresh low-fidelity generation triggers at most one promotion per adjacent pair — a bounded
cascade, not a runaway. On the 7-member XGBoost populations, `7 // 3 = 2`, so two configs promote at each
step. The crucial point is that the higher fidelities inherit DE-evolved configurations rather than
restarting from random samples, so the search at every fidelity benefits from the learning done at the
cheaper ones — the budget-allocation of SH and the history-learning of DE share the same population. Fallback
to a random full-fidelity draw only if the queue ever empties. The distilled module is in the answer.

Two implementation choices are forced by the substrate rather than free, and I want them on the record. The
control constants `F = 0.5` and `Cr = 0.5` are the standard DE midpoint, and under a tiny population — 4 on
SVM, 7 on the others — they matter more than they would at scale: a larger F would amplify the
difference-vector step into wild extrapolation off a population that is already sparse, while `Cr = 0.5` keeps
each trial a genuine mix of mutant and target so no single coordinate dominates, and the forced index
`j_rand` guarantees the trial differs from the target in at least one place so a generation can never stall on
pure copies. The subtler constraint is *how a returned trial is matched back to its target*. Hyperband with
random configs could match by exact `config == last.config`, because a random config is never perturbed
between queueing and evaluation. DE cannot: it mutates configs, and the harness then clips and re-encodes
them, so the config that comes back is not bit-identical to the one I queued. So matching has to run on the
*encoded* vectors with `np.allclose(..., atol=1e-3)` and a fidelity-proximity check `|fid − budget| < 0.05`,
tolerating the encode–decode–clip round-trip. This is why the decode is log-linear and deterministic — a
config encodes, decodes, clips, and re-encodes to within 1e-3 of itself — and it is a real adaptation to a
harness that hands me back a cleaned config rather than the vector I proposed.

Now let me verify the whole thing degenerates sensibly in a limiting case before I trust its behavior, then
say where it should land. Push the budget down so that `s_max = 0`: the ladder becomes the single fidelity
`[1.0]`, there is no promotion (no next rung), and DE simply evolves one full-fidelity population generation
by generation — which is exactly plain differential evolution at full price, the correct degenerate limit.
Push the other way, imagine the cheap fidelity is a *perfect* proxy for the expensive one: then every
promotion carries the true top `1/η`, the full-fidelity rung only ever evaluates genuine survivors, and the
effective number of full-cost looks I spend is a small fraction of the budget while the trial count runs into
the hundreds — the best case for AUC. Real benchmarks sit between these, and where they sit is set entirely
by the cheap-to-expensive rank correlation.

So where will this actually land relative to CMA-ES and random search, and what should I be nervous about?
The wins should be on *convergence AUC*, the metric both prior rungs were weakest on, because triage means a
decent config surfaces from the cheap rungs fast and the best-so-far curve climbs early instead of sitting
flat — the opposite of CMA-ES's wide-then-slow opening. I expect DEHB's AUC to beat CMA-ES comfortably
(whose SVM AUC was 0.6644 with that 0.2241 seed, and whose XGBoost AUC was 0.7373) and to be at least
competitive with, probably above, random search's 0.7885 on SVM and 0.7725 on the NN. I am specifically
nervous about two things. First, the AUC is computed on the *min-max-normalized* best-so-far curve, and DEHB
spends a lot of cheap evaluations — many low-fidelity scores that are noisy and can be poor — so the
normalization floor (the worst score seen) may sit very low and the curve's early portion may look ragged;
the convergence_auc can even exceed 1.0 here, an artifact of how the normalized curve integrates when many
cheap noisy evaluations bracket the good ones. Second, the *cost accounting*: because low-fidelity evals are
cheap, DEHB runs many more actual evaluations than the single-fidelity baselines — the initial-population
arithmetic above already shows the trial count detaching from the 40–50 budget, and with promotions and
generations firing it should climb into the hundreds, sometimes with a large seed-to-seed swing because DE's
promotion cascade is data-dependent (a lucky low rung that fills quickly triggers more promotions). And on a
benchmark where the cheap fidelity predicts the expensive one *poorly* — the NN, where 50 iterations may rank
configs differently from 500 — aggressive low-fidelity triage could promote the wrong survivors and the final
best score could actually *dip* below random search's, the very benchmark where CMA-ES had been fine.

So the falsifiable expectations against the prior numbers are: convergence AUC should rise sharply versus
CMA-ES and beat random search's 0.7885/0.7725 on SVM and NN, because triage front-loads good configs;
`total_evals` should climb well past the budget (into the hundreds on at least SVM), confirming the cheap
evaluations are being spent, and it should vary across seeds because promotion is data-dependent; and the
final best scores should stay in the competitive band but with a real risk of the NN best *worsening*
relative to random search's −3050 if low-fidelity rank-correlation is weak there. If I see exactly that —
strong AUC, inflated and seed-variable total_evals, an NN best that slips while SVM/XGBoost AUC jump — that
is multi-fidelity working as designed, with the low-fidelity-correlation caveat exposed, and it points at the
next question: the residual weakness is that DEHB and SH alike *guess* the configurations-versus-fidelity
tradeoff, and where the cheap fidelity lies, they throw the eventual winner away early. The next rung should
stop guessing it and hedge across the whole spectrum.
