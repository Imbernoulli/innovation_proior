NOTEARS-MLP's numbers land exactly on the trade I predicted, and they hand me the lever for the
strongest rung. Its precision jumped — 0.64 on ER20-Gauss seed 42, 0.58 on ER12-LowSample seed 42,
mean precision 0.43–0.50 where GraN-DAG was 0.11–0.50 and unstable — and its SHD collapsed on the dense
scenarios: SF20-GP fell from GraN-DAG's 92 to 35.7, ER20-Gauss from 76 to 51. The feature-importance
pruning did its job; the graphs are cleaner and the seed variance shrank. But look at what it cost, and
it is the precise weakness I flagged. On **SF20-GP** the recall *cratered* to 0.13 (mean F1 0.153 —
*below* GraN-DAG's 0.273 and even below DirectLiNGAM's 0.319), with all three seeds in the 0.08–0.19
recall range. The hybrid is missing the overwhelming majority of true edges on the GP scenario. That is
the linear-candidate bottleneck made visible: the linear NOTEARS skeleton plus the 0.15 correlation
screen simply never *proposes* the smoothly-nonlinear GP parents, so the gradient-boosted refinement —
which can only prune and re-weight what stage one offered — has nothing to recover them from. ER20-Gauss
(F1 0.308) and ER12-LowSample (F1 0.384) came in respectably and more stably, but the seed-456 dips
(F1 0.109 on ER20-Gauss, 0.273 on ER12) show the linear stage still occasionally seeds a bad candidate
pool that the refinement cannot repair. So the diagnosis is sharp and it is *not* "more pruning": the
bottleneck is **candidate generation through a linear skeleton**. Every method so far has either
generated edges densely and over-connected (GraN-DAG) or generated them linearly and under-connected
(NOTEARS-MLP). What I need is a method that gets the *nonlinear order right first* — without ever
passing through a linear skeleton or a free over-parameterized net — and then does disciplined edge
selection. If I separate "get the order" from "select the edges," I can use the right tool for each.

This decoupling is the organizing idea, and it is justified by identifiability, so let me make the
argument at the population level where overfitting cannot muddy it. The DAG has a topological order;
pick a correct one and the model becomes triangular — each variable regresses only on those earlier in
the order. If somebody handed me a correct order, causal discovery would essentially dissolve into
per-node nonlinear regression plus deciding which earlier variables actually matter — ordinary variable
selection, a solved problem. So the genuinely hard, irreducibly causal part is the *order*; everything
else is regression. And crucially, a *correct order* is enough for the causal answer even with extra
spurious edges: the fully-connected DAG of a correct order is a super-DAG of the truth, and under
modularity the intervention distributions it implies match the truth's, so edge pruning is an
efficiency-and-readability step, not a correctness step. That tells me where penalties belong: the order
search needs *no* regularization (identifiability supplies the gap that makes the true order the
strict optimum), and the regularization goes into the *edge selection* afterward. That is exactly the
split NOTEARS-MLP got wrong by entangling a linear skeleton with the edge selection.

Now, what score recovers the order? The cleanest is the likelihood. Model the additive SEM with
Gaussian errors; profiling out the functions, the expected negative log-likelihood collapses to
`sum_j log(sigma_j)`, where `sigma_j²` is the residual variance of the best nonlinear regression of `X_j`
on its candidate parents. So a structure is scored purely by residual variances from ordinary nonlinear
regressions — no independence tests, no kernel HSIC. And the true order is the strict minimizer of this
*unpenalized* score, because identifiability forces a positive gap: a wrong order scoring below the
truth would imply a negative KL divergence, which is impossible, so wrong orders score higher or equal,
and the closedness of the nonlinear additive class makes the gap strict. The linear-Gaussian tie
(`gap = 0`) reappears exactly where the order is not identifiable anyway. This is the whole reason a
residual-variance order search beats DirectLiNGAM's linear residual test and NOTEARS-MLP's linear
skeleton: it reads the nonlinear asymmetry directly off the regression residuals.

That is the principled CAM. But I have to be honest about what *this task's* fill of the function
actually implements, because it is a **CAM-inspired heuristic**, not the full machinery — and the gap is
the same-named-vs-paper story I must respect. The fullest CAM uses penalized regression splines (GAMs)
for every regression, a preliminary additive neighbor-selection (`gamboost`) to shrink candidate pools,
a greedy IncEdge order search driven by the decomposable likelihood *gain*, and a GAM-significance
pruning step. The editable `run_causal_discovery` does none of those literally. Let me walk what it
*does* do, in the order the code runs it, because each stage is a tractable surrogate for a CAM idea.

Stage one is the order. The fill builds the order *greedily by residual variance*, but with a specific,
slightly unusual rule. The **first** variable is chosen as the one with the *smallest marginal variance*
— the heuristic being that a root cause's variance is just its noise variance, so roots tend to have low
total variance. Then at each subsequent step, for every not-yet-placed variable it fits a
`GradientBoostingRegressor` of that variable on the *current ordering* (all already-placed variables) and
picks the one whose residual variance is *smallest* to append next. This is the residual-variance score
realized with boosted trees instead of GAMs, and greedily appending the lowest-residual-variance node is
a forward surrogate for the likelihood order search. Note the deviations from canonical CAM: boosted
trees (not penalized splines), no PNS (the regression at step `k` uses *all* `k` already-placed variables
as predictors, not a screened neighborhood), and the root chosen by marginal-variance heuristic rather
than by the likelihood gain. These matter for small `n` — regressing on a growing predictor set with
boosted trees is more variance-prone than a screened GAM — but on the evaluated graph sizes the boosted
ordering is a reasonable, cheap stand-in.

Stage two is preliminary edge selection along the order. For each node at position `pos`, fit a
`GradientBoostingRegressor` of it on *all earlier* variables and keep an edge from a candidate parent
only if its **feature importance** exceeds `0.05`. This is the analogue of CAM's edge-fitting:
importance-thresholding is a tree-based surrogate for the GAM significance test, and because it operates
on the *full* set of predecessors (which, for a correct order, contains all true parents), it does not
suffer NOTEARS-MLP's linear-candidate bottleneck — the candidate pool is the whole upstream set, fitted
nonlinearly. This is the single most important structural difference from rung three: the candidate
generation is *nonlinear and order-respecting*, not linear.

Stage three is pruning, the regularized cleanup. For any node with more than one parent, the fill runs a
*partial-residual independence test*: for each parent `p`, regress both `X_j` and `X_p` on the *other*
parents (boosted trees again), take the two residuals, and if their absolute correlation is *below 0.05*
remove `p`. The logic: if after conditioning on the other parents `X_p` carries no leftover dependence
with `X_j`, then `p` is not a genuine parent given the others. This is a surrogate for CAM's
significance-based pruning, using residual correlation as the cheap dependence proxy. It is the place the
penalty/threshold lives — exactly where I argued regularization belongs — and it is what should give CAM
the precision NOTEARS-MLP bought *without* paying the recall NOTEARS-MLP lost, because the edges were
generated nonlinearly in the first place.

The output `B` already obeys the harness convention: `B[child, parent] = 1` means `parent -> child`,
i.e. `B[i,j] != 0` means `j -> i`. The full scaffold module is in the answer.

So the same-named-vs-paper gap, stated plainly: the task's `cam` is a *heuristic* CAM — gradient-boosted
residual-variance ordering (root by min-marginal-variance), feature-importance edge selection over all
predecessors, and partial-residual-correlation pruning — not the canonical CAM with GAM splines,
`gamboost` PNS, IncEdge likelihood-gain order search, and GAM-significance pruning. But it preserves the
two ideas that matter: **decouple order from edges**, and **generate candidates nonlinearly and
order-respecting** rather than through a linear skeleton.

The falsifiable expectations against NOTEARS-MLP's measured shape. The decisive claim is that CAM fixes
the recall collapse *without* surrendering precision. On **SF20-GP**, where NOTEARS-MLP's linear
skeleton starved recall to 0.13 (F1 0.153), the nonlinear order-respecting candidate generation should
recover the GP parents the linear stage missed — I expect recall to leap into the 0.7–0.8 range and F1 to
jump past 0.8, a multiple of every prior rung, because GP-on-scale-free is exactly the regime where a
correct nonlinear order plus nonlinear edge fitting should shine. On **ER20-Gauss**, the nonlinear-
Gaussian case, CAM should keep NOTEARS-MLP's high precision (the partial-residual pruning is strict) and
restore recall, lifting F1 well above 0.31 toward 0.7, with precision near 1.0 — the residual-variance
order reads the nonlinearity that the Gaussian noise hid from DirectLiNGAM and that the linear skeleton
hid from NOTEARS-MLP. On **ER12-LowSample**, the boosted ordering on 150 samples is the riskiest stage
(regressing on a growing predictor set), so I expect CAM's *smallest* relative gain here — F1 in the
0.5–0.6 range, clearly above NOTEARS-MLP's 0.384 but not the blowout of the 2000-sample scenarios,
because the order search is variance-limited at small `n`. The single number that would falsify the
whole "decouple and go nonlinear" thesis: if CAM's averaged F1 does not clearly clear NOTEARS-MLP's
~0.28 and its SHD does not fall below NOTEARS-MLP's ~36 on SF20-GP, then the order is not the
bottleneck and I have misread the ladder. CAM is the strongest baseline precisely because it is the
first method to get the nonlinear order right and *then* prune — and the residual weakness it leaves is
the *order search itself*: it is greedy and residual-variance-driven, which is exactly what a stronger,
score-matching order recovery could improve.
