GraN-DAG's numbers confirm the diagnosis and expose the exact next lever. It did what I asked of it on
the scenario where it should: on **ER20-Gauss**, the case where DirectLiNGAM was worst (F1 0.245, SHD
96) *because* it could not see the nonlinearity, GraN-DAG lifted the mean F1 to 0.396 and SHD to 76 —
and seed 123 hit F1 0.55 at precision 0.88, which is the nonlinearity finally being read correctly. So
the bet that "model the mechanisms nonlinearly" pays off where it should, and that answers the one
question I promised to check first: the leverage really is the nonlinearity. But read the rest and the
weakness I predicted is glaring, and the edge counts make it stark. On **SF20-GP** the graph has 36
true edges, and the precision/recall pairs tell me what GraN-DAG actually drew: seed 42 was tolerable
(recall `0.472 = 17/36`, precision `0.34`, so about `50` predicted), but seeds 123 and 456 collapsed —
precision `0.143` and `0.126` against the same recall, which back out to roughly `119` and `151`
predicted edges on a 36-edge graph. It is drawing three to four times the true number of arrows and
catching only half of them, so its SHD ballooned to a mean of 92 — *worse* than DirectLiNGAM's 60, and
its precision (mean 0.20) is no better than the linear floor's. The per-seed variance is the other half
of the story: on ER20-Gauss the F1 runs 0.50, 0.55, then *collapses* to 0.137 on seed 456 (SHD 140,
worse than DirectLiNGAM's worst there), a spread of 0.41 in F1 on one scenario. On **ER12-LowSample**
it is the worst rung so far (F1 0.161, SHD 55, precision 0.11): with the graphs at ~19–23 true edges it
draws on the order of 49–64, the two-layer MLP per variable hallucinating dependence exactly as the
parameter count predicted — more weights than samples, so it interpolates noise.

The pattern across all three is one thing: **precision**. GraN-DAG's recall is respectable (0.31–0.49)
but its precision is 0.11–0.50 and unstable, so its graphs are over-connected — and the mean F1 across
scenarios, `(0.273 + 0.396 + 0.161)/3 = 0.277`, is barely above DirectLiNGAM's 0.250 despite genuinely
modeling the nonlinearity, precisely because the over-connection eats the gain. That is exactly what I
warned about: no explicit sparsity penalty, no significance pruning, order learned only implicitly
through the acyclicity constraint, so the augmented Lagrangian settles on acyclic-but-dense graphs and
the Jacobian threshold cannot reliably pick the right edges to keep. The clamp-every-500-iterations
stability patch avoided catastrophic edge loss but let the dense intermediate graphs persist — I can
see it in the 119- and 151-edge SF20-GP solutions. So the next rung's job is sharp: keep nonlinear
mechanism modeling, but impose *real* sparsity so precision climbs and SHD falls, without the per-seed
lottery of a free neural net.

The continuous-constraint machinery is still the right backbone — it is the only thing that avoids both
the combinatorial DAG search and the linear-Gaussian blindness — so I do not want to abandon NOTEARS's
`tr e^{W∘W} - d = 0`. The question is what to put *under* it, and I have three real options. The
first is to keep GraN-DAG's per-variable MLP but add an explicit sparsity term: a grouped-`ℓ_1`
penalty on the first-layer *columns*, variable-split into nonnegative parts so a bound-constrained
solver can carry it, which drives whole columns — whole parents — to zero. The dependence summary is
then `||∂_k f_j||_{L²}`, realized as the squared norm of the `k`-th first-layer column, which equals
`[W∘W]_{kj}` directly so `h` runs on it without a square root. That is the principled answer to
GraN-DAG's precision problem — sparsity where GraN-DAG had none — and I keep it as the reference for
*why* sparsity belongs here. The second option is to prune GraN-DAG's output post hoc, but that
inherits the dense, high-variance solution and only trims it, so the seed lottery survives. The third
is a two-stage decomposition: use the smooth constraint to get a cheap linear skeleton and an order,
then repair the nonlinearity with a separate regressor that has its own built-in feature selection.
This is a leaner construction than the grouped-`ℓ_1` MLP, and it is exactly what *this task's* fill of
the function implements — so I have to be honest that the name `notears_mlp` is doing a lot of work it
does not cash out, and walk what the code actually runs.

What the editable `run_causal_discovery` actually implements is a **two-stage hybrid**: a *linear*
NOTEARS to get a skeleton, then a *nonlinear* refinement with gradient-boosted regression on top. There
is no MLP anywhere in the fill, despite the name. Let me walk the construction in the order the code
runs it, because each stage answers a specific weakness of GraN-DAG.

Stage one is linear NOTEARS, the original continuous program. Standardize `X`, then minimize
`0.5/n · ||X - XW||²_F + λ_1|W|_1` subject to `h(W) = tr e^{W∘W} - d = 0`, with `λ_1 = 0.01` and the
augmented Lagrangian. I compute the acyclicity value and gradient from a twelve-term power-series
truncation of the matrix exponential rather than a library `expm`: `M = W∘W` is elementwise-small
(the `ℓ_1` penalty and standardized data keep entries well below 1), so the `1/12! ≈ 2·10⁻⁹`
prefactor crushes the tail far below the `h_tol = 1e-8` threshold, and — the point that actually
matters — `_h` and `_h_grad` use the *same* expansion, so the optimizer sees the exact gradient of
the value it is minimizing, not the gradient of a different exponential. The inner subproblem is
**L-BFGS-B** (`jac=True`, 500 inner iters), the diagonal of `W` and its gradient zeroed to forbid
self-loops. The outer schedule is the same augmented-Lagrangian pacing as the previous rung —
`ρ ×10` only when `h` fails to shrink to a quarter of its previous value, `α += ρ·h` dual ascent, 30
rounds, stop at `|h| < 1e-8` or `ρ > 1e16` — with a small random init (`0.01·randn`) to break the
zero-symmetry. Because this stage is convex-ish and low-dimensional (`d²` parameters, no hidden
units), it converges to essentially the same skeleton regardless of the init, giving a *linear*
skeleton and a *causal ordering* read off the column-sum of `|W|` (downstream nodes accumulate more
incoming weight), which I use to break the refined edges into a DAG at the end.

Now, a *linear* skeleton on nonlinear data is exactly what failed at rung one. The whole point of stage
two is to repair that linearity *without* re-introducing GraN-DAG's free-net precision problem. The
construction: threshold the linear `W` at `0.3` to get candidate parents, then *augment* the candidate
set for each node with any variable whose linear correlation with it exceeds `0.15` (so a nonlinearly-
strong-but-linearly-weak parent is not lost by the linear stage), and for each node fit a **gradient-
boosted regressor** (`GradientBoostingRegressor`, 100 trees, depth 3) of `X_j` on its candidate set.
Keep a candidate parent only if its boosted-tree *feature importance* exceeds `0.05`. This is the
nonlinear refinement: the boosted regressor sees curved mechanisms the linear `W` cannot, and the
feature-importance threshold is the sparsity discipline GraN-DAG never had — it directly answers "does
`X_j` actually use `X_k` nonlinearly," and prunes the rest. So the precision pressure that was missing
becomes the *core* of stage two, and it is a genuinely different mechanism from GraN-DAG's single
Jacobian threshold: a tree ensemble's feature importance is a *relative* budget — the importances sum
to one across the candidate set — so a `0.05` cutoff keeps at most twenty features and typically far
fewer, which is a much harder sparsity floor than a free net with an absolute clamp.

The two candidate thresholds are deliberately mismatched, and the mismatch is principled rather than
arbitrary. The `0.3` cut is applied to the linear NOTEARS coefficient, which is an `ℓ_1`-*regularized*
quantity — the penalty has already shrunk noise coefficients toward zero, so a surviving value of `0.3`
is a genuinely confident linear edge, and I can afford a high bar. The `0.15` cut is applied to the raw
Pearson correlation, which is *unregularized* and therefore noisier and biased away from zero by finite
samples, but it is there as a safety net to catch a parent the sparse linear fit dropped — so it is set
looser, trading some false candidates (which stage-two's importance test will prune anyway) for recall.
The asymmetry is exactly right for a two-tier funnel: a strict gate on the trustworthy statistic, a lax
gate on the cheap one, with a nonlinear pruner downstream to clean up whatever the lax gate lets
through. The one thing neither gate can do — and the quadratic argument above is why — is admit an edge
with no linear signal at all, so the funnel is only as wide as the linear front-end, no matter how
permissive I make the second gate.

The last step is DAG enforcement, simpler than GraN-DAG's Jacobian pass. The nonlinear refinement can
re-introduce a cycle (boosted importances respect no ordering), so I impose the linear stage's order:
`sum_i |W_{ij}|`, the column sum, is the total incoming weight of node `j`, near zero for a root and
large for a deeply downstream node, so `rank = argsort(order_score)` puts root-like nodes first and
any refined edge from a higher-rank node into a lower-rank one is deleted — acyclic by construction.
The pipeline has no seed-dependent optimizer basin: stage one is 30 L-BFGS-B rounds on a
`d²`-parameter problem, stage two exactly `d` boosted fits on candidate sets of size at most `d` —
cheap at `d = 12–20` and deterministic given the seed, so I expect seed variance to shrink relative to
GraN-DAG. Output `B` obeys `B[i,j] != 0` means `j -> i`; the full module is in the answer.

The same-named-vs-paper gap determines what I should expect. The method named `notears_mlp` here is
**not** the nonlinear MLP-NOTEARS at all: there is no neural network, no per-variable MLP, no
`||∂_k f_j||_{L²}` column-norm dependence summary, no grouped-`ℓ_1` on
first-layer weights, and no end-to-end continuous nonlinear program. It is **linear NOTEARS** (the
continuous acyclicity machinery in its original linear-coefficient form) **plus a gradient-boosted
nonlinear edge-selection refinement** — a hybrid that uses the smooth constraint only to get a linear
skeleton and an order, then repairs the nonlinearity with trees and feature-importance thresholding.
The canonical MLP-NOTEARS reasoning I keep as reference (per-node MLP, squared-column-norm into
`tr e^{W∘W}-d`, L-BFGS-B over split nonnegative weights) is *not* what runs here. The practical
consequence follows directly from the ordering of the two stages: where GraN-DAG over-connected because
it had no pruning, this fill will *under*-connect on the hardest scenarios, because the *linear* stage
one supplies the candidate pool and a linear skeleton on strongly-nonlinear GP mechanisms can miss true
edges entirely before the boosted refinement ever sees them. The refinement can only prune and re-weight
candidates the linear stage and the `0.15` correlation screen proposed; it cannot *discover* a parent
that is linearly invisible. And a GP mechanism is exactly the pathological case for the two screens: a
smooth function whose input and output can be near-uncorrelated will pass neither the `0.3` weight
threshold nor the `0.15` correlation screen, so its edge never enters the candidate pool. I can make
the failure exact rather than gesture at it. Take the cleanest symmetric bend, `X_j = X_i²` (plus
noise), with `X_i` drawn from any symmetric zero-mean distribution — which several of the noise
families here are. Then `cov(X_i, X_j) = E[X_i · X_i²] = E[X_i³] = 0` by the odd-moment symmetry, so
the Pearson correlation is *exactly* zero. A true, strong, deterministic parent–child relationship is
completely invisible to both linear screens: the `0.15` correlation cutoff sees `0`, and the linear
NOTEARS coefficient that best fits a parabola with a line is also `0`. The dependence is entirely in
the second moment, which neither stage-one screen inspects. GP draws are smoother and less extreme than
a pure parabola, but they carry exactly this kind of even-symmetric curvature, and every bit of it that
is symmetric contributes zero to the linear correlation. So the bottleneck is not the pruning; it is
*candidate generation through a linear lens*, and the quadratic case shows it is not a matter of
degree — some true edges have identically zero linear signal and can never be recovered by any
threshold on a linear statistic.

This gives me a sharper handle on *which* scenario should suffer most, and it is not simply "the most
nonlinear one." The scenarios differ in their function families: SF20-GP uses GP mechanisms throughout,
while ER20-Gauss and ER12-LowSample draw from the *mixed* family — GP, one-hidden-layer MLP, low-degree
polynomial, steep sigmoid. Those mixed mechanisms are not equally linear-invisible. A sigmoid is
monotone and has a large linear component over its active range; a low-degree polynomial with an odd
term (`x` or `x³`) carries nonzero `E[x·f(x)]`; an MLP with leaky units is piecewise-linear and
correlates strongly with its input. Only the *pure symmetric* part of a GP is linearly invisible. So
the linear screens should catch a substantial fraction of the true edges on the mixed-family ER
scenarios — enough seed candidates for the boosted refinement to work with — while on SF20-GP, where
every mechanism is a smooth GP bend with weak linear projection, the screens should starve the pool. The
prediction that falls out is specific: the recall collapse should be *concentrated on SF20-GP*, not
spread evenly across the nonlinear scenarios, precisely because SF20-GP is the one whose nonlinearity
has the least linear shadow.

So the expectations, read against GraN-DAG's measured shape. The defining trade is **precision up,
recall down**. Stage two's feature-importance pruning should push precision well above GraN-DAG's
0.11–0.50, and SHD should fall sharply on the dense-graph scenarios GraN-DAG bloated — on SF20-GP it
cannot draw 119–151 edges when a relative-importance budget caps each node's parents, so SHD should
collapse toward the order of the true edge count, and the variance shrink because there is no free-net
seed lottery resetting an optimizer on a non-convex surface. But the linear-candidate
bottleneck should *cap recall*: on **SF20-GP**, where GP mechanisms are smooth and strongly nonlinear,
the linear skeleton will miss many true edges, so I expect recall to crater (well below GraN-DAG's
0.49) and F1 to actually come in *low* there — possibly below GraN-DAG's 0.273 and even below
DirectLiNGAM's 0.319, which would be the striking, almost paradoxical signature of this method: a more
sophisticated pipeline scoring *worse* than the linear floor on one scenario precisely because its
sophistication is spent on pruning a pool the linear front-end starved. On **ER20-Gauss**, where
GraN-DAG did relatively well (F1 0.396) but with that seed-456 collapse to 0.137, the hybrid's
precision discipline should give a competitive, *more stable* result with much higher precision and no
seed-456 blowup, trading some recall for reliability. Unlike DirectLiNGAM, this method does not
*depend* on non-Gaussianity: the linear stage is least-squares plus an acyclicity constraint, the
order comes from column sums, the direction discipline is topological deletion — none appealing to
higher-order moments — so the Gaussian noise that destroyed the floor's exogeneity test costs this
pipeline nothing in principle; its ER20-Gauss weakness, if any, is recall from partly-nonlinear
parents slipping the linear pool. On **ER12-LowSample**, the linear stage is more robust to small `n`
than a per-variable MLP, and the boosted refinement on a tight 12-node candidate pool should clearly
beat GraN-DAG's 0.161 — likely the hybrid's best relative showing.

The averaged picture, then, is a narrow F1 win bought entirely by precision and SHD: precision up
everywhere, SHD falling sharply from GraN-DAG's 74–92 range, but the SF20-GP recall collapse nearly
cancelling the F1 gain, so the mean barely moves off GraN-DAG's ~0.277 while the graph gets much
cleaner. That is a revealing kind of "win": the primary metric held down by a single scenario's
recall, not by any global deficiency. The falsifiable claim is that this rung edges out GraN-DAG's F1
*and* dramatically cuts SHD, even while losing to it on SF20-GP recall specifically — and if the
SF20-GP recall does *not* crater, my linear-bottleneck diagnosis is wrong and the linear skeleton is
proposing GP parents after all. The residual weakness left for a stronger rung is now precise: the
*linear* candidate-generation bottleneck. A method that recovers the nonlinear *order* directly, never
passing through a linear skeleton, should keep this precision discipline while restoring the recall the
linear stage throws away.
