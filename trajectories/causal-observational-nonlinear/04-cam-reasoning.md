NOTEARS-MLP's numbers land exactly on the trade I predicted, and they hand me the lever for the
strongest rung. Its precision jumped — 0.64 on ER20-Gauss seed 42, 0.58 on ER12-LowSample seed 42,
mean precision 0.43–0.50 where GraN-DAG was 0.11–0.50 and unstable — and its SHD collapsed on the dense
scenarios: SF20-GP fell from GraN-DAG's 92 to 35.7, ER20-Gauss from 76 to 51. The feature-importance
pruning did its job; the graphs are cleaner and the seed variance shrank. But look at what it cost, and
it is the precise weakness I flagged, now visible in the edge counts. On **SF20-GP**, where the graph
has 36 true edges, the recall *cratered* to a mean of 0.13 — the seeds run recall 0.194, 0.111, 0.083,
which is roughly 7, 4, 3 true edges recovered out of 36. Mean F1 0.153, which is *below* GraN-DAG's
0.273 and even below DirectLiNGAM's 0.319. This is the almost-paradoxical signature I said to watch
for: a more sophisticated pipeline scoring worse than the linear floor on one scenario, because its
sophistication is spent pruning a candidate pool the linear front-end starved. And the mechanism is
exactly the one I named — SF20-GP is the pure-GP scenario, the one whose nonlinearity has the least
linear shadow, so the linear NOTEARS skeleton plus the 0.15 correlation screen simply never *proposes*
the smoothly-nonlinear GP parents, and the gradient-boosted refinement can only prune and re-weight what
stage one offered. ER20-Gauss (F1 0.308) and ER12-LowSample (F1 0.384) came in respectably and more
stably — the mixed mechanisms there carry enough linear signal to seed the pool — but the seed-456 dips
(F1 0.109 on ER20-Gauss, 0.273 on ER12) show the linear stage still occasionally seeds a bad candidate
pool that the refinement cannot repair. The averaged F1 is `(0.153 + 0.308 + 0.384)/3 = 0.282`, a hair
above GraN-DAG's 0.277, with SHD dramatically lower — exactly the "narrow F1 win bought entirely by
precision and SHD" I predicted, and the primary metric is being held down by a single scenario's recall.

So the diagnosis is sharp and it is *not* "more pruning": the bottleneck is **candidate generation
through a linear skeleton**. Line the three rungs up and the pattern is a fork with no good branch.
Every method so far has either generated edges densely and over-connected — GraN-DAG, drawing 119–151
arrows on a 36-edge SF graph — or generated them linearly and under-connected — NOTEARS-MLP, recovering
3–7 of those 36. What I need is a method that gets the *nonlinear order right first* — without ever
passing through a linear skeleton or a free over-parameterized net — and then does disciplined edge
selection. Both prior nonlinear attempts put the nonlinear model and the structure search in the *same*
stage — GraN-DAG's net learned mechanism and adjacency jointly, NOTEARS-MLP's linear skeleton proposed
structure and the boosted stage only trimmed it — and both paid for the coupling, one in precision and
one in recall. The way out is not a better single stage; it is to stop asking one stage to do both. If I separate "get the order" from "select the edges," I can use the right tool for each,
instead of forcing one linear stage to do both jobs and failing at both on GP mechanisms.

This decoupling is the organizing idea, and it is justified by identifiability, so let me make the
argument at the population level where overfitting cannot muddy it. The DAG has a topological order;
pick a correct one and the model becomes triangular — each variable regresses only on those earlier in
the order. If somebody handed me a correct order, causal discovery would essentially dissolve into
per-node nonlinear regression plus deciding which earlier variables actually matter — ordinary variable
selection, a solved problem. So the genuinely hard, irreducibly causal part is the *order*; everything
else is regression. Let me make the "order is enough" claim precise, because it is what licenses
spending my regularization budget entirely on the edges. A *correct order* is enough for the causal
answer even with extra spurious edges: the fully-connected DAG of a correct order is a super-DAG of the
truth, and under modularity the intervention distributions it implies match the truth's, so edge
pruning is an efficiency-and-readability step, not a correctness step. That tells me where penalties
belong. The order search needs *no* regularization — identifiability supplies the gap that makes the
true order the strict optimum — and the regularization goes into the *edge selection* afterward. That is
exactly the split NOTEARS-MLP got wrong by entangling a linear skeleton with the edge selection: it
regularized (the `ℓ_1`, the thresholds) inside the stage that also had to find the structure, so the
regularization threw away true edges instead of just cleaning up spurious ones.

Now, what score recovers the order? The cleanest is the likelihood, and I should derive why it pins the
order rather than assert it. Model the additive SEM with Gaussian errors; the joint negative
log-likelihood is `sum_j [ (X_j - f_j(pa_j))² / (2σ_j²) + log σ_j ]` up to constants. Profile out the
functions `f_j` (fit them by nonlinear regression) and the noise scales `σ_j`, and in expectation the
data-fit terms each collapse to `1/2`, leaving the expected negative log-likelihood proportional to
`sum_j log σ_j`, where `σ_j²` is the residual variance of the best nonlinear regression of `X_j` on its
candidate parents. So a structure is scored purely by residual variances from ordinary nonlinear
regressions — no independence tests, no kernel HSIC. And the true order is the strict minimizer of this
*unpenalized* score. The argument is a KL one: a wrong order scoring *below* the truth would mean the
best distribution factorizing along the wrong order has smaller expected negative log-likelihood than
the true factorization, i.e. a negative KL divergence to the truth — impossible. So every wrong order
scores higher or equal, and the closedness of the nonlinear additive class makes the gap strict except
exactly in the linear-Gaussian case where the order is not identifiable anyway (there the gap is zero,
consistently with the equivalence-class stall I started the whole ladder from). This is the reason a
residual-variance order search beats DirectLiNGAM's linear residual test and NOTEARS-MLP's linear
skeleton: it reads the nonlinear asymmetry directly off the regression residuals, and it does so with a
positive, identifiability-guaranteed margin rather than a threshold I have to guess.

I want to see that margin actually exist before I build on it, so let me put it on a concrete two-node
system. Take `X_1 ~ N(0,1)` a root and `X_2 = sin(2 X_1) + 0.5 X_1 + N_2` with `N_2` small (variance
`≈ 0.09`), and score both orders by the profiled `sum_j log σ_j` using a boosted regression for each
conditional. The correct order `[1,2]` leaves residual variance `Var(X_2|X_1) = 0.082` — essentially
the true noise floor `0.09`, because the boosted regression recovers the mechanism — against the
marginal `Var(X_1) = 0.98`, for a score `log 0.98 + log 0.082 = -2.51`. The wrong order `[2,1]` cannot
undo the non-invertible bend: regressing `X_1` on `X_2` leaves residual variance `0.38`, far above the
noise floor, and with the marginal `Var(X_2) = 1.10` the score is `log 1.10 + log 0.38 = -0.87`. The
true order wins by a gap of `1.64` — positive, and comfortably larger than the estimation noise at this
sample size. So the population claim is not just a KL formality; the asymmetry is large enough to read
off finite data, which is exactly what a greedy residual-variance search will lean on.

I have two real ways to turn that population score into an algorithm, and they trade accuracy for cost.
One is to test the additive-noise asymmetry pairwise — regress each variable on each other and compare
residual-independence with a kernel statistic like HSIC — but that is `O(d²)` nonparametric independence
tests, each expensive, and pairwise verdicts need not stitch into a single consistent order; I would be
back to reconciling local decisions into a global structure. The other is a greedy score-based order
built directly from the residual-variance likelihood: repeatedly extend a partial order by the variable
whose nonlinear regression on the current prefix has the smallest residual variance. That is cheaper and
globally coherent by construction, and it is what *this task's* fill implements — as a **CAM-inspired
heuristic**, not the full machinery, so let me walk what the code actually runs and be honest about the
gap. The fullest version of this method uses penalized regression splines for every regression, a
preliminary additive neighbor-selection to shrink candidate pools, a greedy incremental-edge order
search driven by the decomposable likelihood *gain*, and a spline-significance pruning step. The
editable `run_causal_discovery` does none of those literally; each of its stages is a tractable
surrogate.

Stage one is the order. The fill builds the order *greedily by residual variance*, but with a specific,
slightly unusual rule I should examine rather than wave past. The **first** variable is chosen as the
one with the *smallest marginal variance* — the heuristic being that a root cause's variance is just its
noise variance, so roots tend to have low total variance, while a downstream node's variance is its own
noise *plus* the propagated signal variance from its ancestors. That heuristic is not a theorem, and I
can construct its failure in one line: let the root carry heavy noise, `X_1 = 1.6·(noise)` with
variance `≈ 2.6`, feeding a weak-gain child `X_2 = 0.3·tanh(X_1) + (small noise)` with variance
`≈ 0.09`. Now `Var(X_2) ≪ Var(X_1)`, so the min-marginal-variance rule picks the *child* `X_2` as the
root — exactly backwards — because the child's small mechanism gain and small noise leave it with less
total variance than its own high-variance parent. The benchmark's exponential and Laplace noises are
heavy-tailed (high variance), and steep-sigmoid mechanisms saturate (low gain), so this configuration is
not contrived; it is a regime the data actually visits. And because the greedy append never backtracks,
a wrong root can poison the whole prefix. So I expect this stage to
be the method's soft spot exactly when the noise-variance-versus-signal-gain separation is weak, which
is most acute at small `n` where the variances themselves are noisily estimated. After the root, at each
subsequent step, for every not-yet-placed variable it fits a `GradientBoostingRegressor` of that
variable on the *current ordering* (all already-placed variables) and picks the one whose residual
variance is *smallest* to append next. This is the residual-variance likelihood score realized with
boosted trees instead of splines, and greedily appending the lowest-residual-variance node is a forward
surrogate for the order search. The deviations from the canonical version — boosted trees not splines,
no preliminary neighbor screen (the regression at step `k` uses *all* `k` placed variables, not a
screened neighborhood), root by marginal variance not by likelihood gain — matter most at small `n`, but
on the evaluated graph sizes the boosted ordering is a reasonable, cheap stand-in.

Stage two is preliminary edge selection along the order, and it is the structural pivot away from
NOTEARS-MLP. For each node at position `pos`, fit a `GradientBoostingRegressor` of it on *all earlier*
variables and keep an edge from a candidate parent only if its **feature importance** exceeds `0.05`.
The point is what the candidate pool *is*: the whole upstream set, fitted nonlinearly, rather than a
linear skeleton's proposals. For a correct order that upstream set contains every true parent, so the
candidate generation cannot suffer the linear-invisibility bottleneck — the quadratic-mechanism edge
that had identically zero linear signal is now inside a boosted regression that reads it off the second
moment directly. This is the single most important difference from rung three: candidate generation is
*nonlinear and order-respecting*, so the recall NOTEARS-MLP threw away on GP mechanisms should come
back, while the importance threshold keeps the precision. Concretely, the SF20-GP edge that rung three
lost — a symmetric GP bend with `corr(parent, child) ≈ 0` — now enters a boosted regression of the child
on its full set of predecessors, and a regression tree does not need a linear signal: it splits on the
parent wherever the child's conditional mean changes, which for a symmetric bend happens on *both* sides
of the minimum. So the tree assigns that parent nonzero feature importance from its second-moment
structure alone, and the `0.05` cutoff keeps it. The edge that was invisible to every linear statistic
in rung three is recovered here by the same feature-importance mechanism that rung three used only for
pruning — the difference is entirely that here it operates on the true upstream set rather than a linear
skeleton's leftovers. That single relocation of the nonlinear model from the pruning stage to the
candidate-generation stage is what should turn rung three's 0.13 SF20-GP recall into something several
times larger.

Stage three is pruning, the regularized cleanup — and by the population argument this is exactly and
only where a penalty belongs. For any node with more than one parent, the fill runs a *partial-residual
independence test*: for each parent `p`, regress both `X_j` and `X_p` on the *other* parents (boosted
trees again), take the two residuals, and if their absolute correlation is *below 0.05* remove `p`. The
logic: if after conditioning on the other parents `X_p` carries no leftover dependence with `X_j`, then
`p` is not a genuine parent given the others. This is a surrogate for a conditional-significance test,
using residual correlation as the cheap dependence proxy, and it should give this rung the precision
NOTEARS-MLP bought *without* paying the recall NOTEARS-MLP lost — because the edges were generated
nonlinearly in the first place, the pruner is trimming a rich pool rather than a starved one.

There is a self-consistency worry I should name, because it is the same linear-blindness that sank the
previous rung, reappearing in a smaller place. The pruner measures leftover dependence by the *Pearson
correlation* of two residuals, and Pearson correlation sees only the linear part of that dependence — so
by the very argument that a symmetric bend has `corr(x, x²) = 0`, a spurious parent whose leftover
dependence with the child is purely nonlinear could slip the `0.05` cutoff and *not* be pruned. Why is
this tolerable here when it was fatal in stage-one candidate generation there? Because of *where* it
sits in the pipeline. In NOTEARS-MLP the linear screen was the *only* gate that generated candidates, so
a linearly-invisible true edge was lost forever. Here the linear correlation is the *last* gate, pruning
an already-nonlinearly-generated pool, and a missed prune costs me one false positive on a
multi-parent node, not a missed true edge — a precision nick, not a recall collapse. The asymmetry in
consequences is why the same weak statistic is a disaster in one slot and a minor leak in another, and
it is exactly the payoff of putting candidate generation, not pruning, on the nonlinear side of the
decoupling. If precision comes in below what the strict pruning suggests, this residual nonlinear
leftover is the first place I will look.

The output `B` already obeys the harness convention: `B[child, parent] = 1` means `parent -> child`,
i.e. `B[i,j] != 0` means `j -> i`. Let me note the cost, because it bounds the small-`n` risk I flagged.
Stage one fits, at step `k`, one boosted regression for each of the `d-k` remaining variables, so it is
`O(d²)` boosted fits in total — roughly `d(d-1)/2 ≈ 190` fits at `d=20` — each on a predictor set that
grows to size `d`. That is the expensive stage, and it is expensive in exactly the way that hurts at
`n=150`: a 50-tree boosted regression on a 19-dimensional predictor set with 150 points is
data-starved, so the residual-variance comparisons it ranks the order by are themselves noisy, which
feeds straight back into the greedy append taking a wrong, non-backtracked turn. The full scaffold
module is in the answer.

So the same-named-vs-paper gap, stated plainly: the task's `cam` is a *heuristic* CAM — gradient-boosted
residual-variance ordering (root by min-marginal-variance), feature-importance edge selection over all
predecessors, and partial-residual-correlation pruning — not the canonical CAM with spline mechanisms,
an additive neighbor screen, an incremental-edge likelihood-gain order search, and spline-significance
pruning. But it preserves the two ideas that matter: **decouple order from edges**, and **generate
candidates nonlinearly and order-respecting** rather than through a linear skeleton.

The falsifiable expectations against NOTEARS-MLP's measured shape. The decisive claim is that this rung
fixes the recall collapse *without* surrendering precision. On **SF20-GP**, where NOTEARS-MLP's linear
skeleton starved recall to 0.13 (F1 0.153, ~3–7 of 36 edges), the nonlinear order-respecting candidate
generation should recover the GP parents the linear stage missed — I expect recall to leap into the
0.7–0.8 range and F1 to jump past 0.8, a multiple of every prior rung, because GP-on-scale-free is
exactly the regime where a correct nonlinear order plus nonlinear edge fitting should shine and it was
exactly the regime the linear pool could not touch. On **ER20-Gauss**, the nonlinear-Gaussian case, it
should keep NOTEARS-MLP's high precision (the partial-residual pruning is strict) and restore recall,
lifting F1 well above 0.31 toward 0.7, with precision near 1.0 — the residual-variance order reads the
nonlinearity that the Gaussian noise hid from DirectLiNGAM and that the linear skeleton hid from
NOTEARS-MLP. I expect the limiter on ER20-Gauss to flip from precision to *recall*: with a strict
partial-residual pruner and a correct order, the false-positive rate should be tiny (precision near 1),
but a greedy order that places even one node slightly too early forfeits that node's true parents, which
come "after" it in my recovered order and can no longer be candidates — so the edges I lose are lost to
*order errors*, not to bad pruning, and they show up as depressed recall. That is a different failure
signature from NOTEARS-MLP's, and if the ER20-Gauss recall lands well below its precision it points the
finger squarely at the greedy order rather than the edge selection. On **ER12-LowSample**, the boosted ordering on 150 samples is the riskiest stage by the
cost argument above (a data-starved regression on a growing predictor set), so I expect this rung's
*smallest* relative gain here — F1 in the 0.5–0.6 range, clearly above NOTEARS-MLP's 0.384 but not the
blowout of the 2000-sample scenarios, because the order search is variance-limited at small `n`. If
those three land where I expect — roughly 0.85, 0.7, 0.55 — the averaged F1 is about 0.70, some two and
a half times NOTEARS-MLP's 0.282 and a genuine jump rather than the hair's-breadth gains that separated
the first three rungs from each other. The single number that would falsify the whole "decouple and go
nonlinear" thesis: if this rung's averaged F1 does not clearly clear NOTEARS-MLP's ~0.28 and its SHD
does not fall below NOTEARS-MLP's ~36 on SF20-GP, then the order is not the bottleneck and I have
misread the ladder. This should be the
strongest baseline precisely because it is the first method to get the nonlinear order right and *then*
prune — and the residual weakness it will leave is the *order search itself*: greedy,
residual-variance-driven, non-backtracking, and noisy at small `n`, so the natural next lever is to
recover the order more directly and globally from the distribution rather than by appending one node at
a time.
