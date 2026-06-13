GraN-DAG's numbers confirm the diagnosis and expose the exact next lever. It did what I asked of it on
the scenario where it should: on **ER20-Gauss**, the case where DirectLiNGAM was worst (F1 0.245, SHD
96) *because* it could not see the nonlinearity, GraN-DAG lifted the mean F1 to 0.396 and SHD to 76 —
and seed 123 hit F1 0.55 at precision 0.88, which is the nonlinearity finally being read correctly. So
the bet that "model the mechanisms nonlinearly" pays off where it should. But read the rest and the
weakness I predicted is glaring. The per-seed variance is enormous: that same ER20-Gauss has a seed-456
collapse to F1 0.137, SHD 140 — *worse* than DirectLiNGAM's worst there — dragging the mean down. On
**SF20-GP** GraN-DAG is actually a wash or worse than DirectLiNGAM (F1 0.273 vs 0.319, and SHD 92 vs 60
— far denser), with precision a dismal 0.20 and recall 0.49: it is laying down many edges, catching
some, but the graph is bloated. On **ER12-LowSample** it is the worst rung so far (F1 0.161, SHD 55,
precision 0.11): the two-layer MLP per variable is hopelessly over-parameterized for 150 samples and
just hallucinates dependence. The pattern across all three is one thing: **precision**. GraN-DAG's
recall is respectable (0.31–0.49) but its precision is 0.11–0.50 and unstable, so its graphs are
over-connected. That is exactly what I warned about — no explicit sparsity penalty, no significance
pruning, order learned only implicitly through the acyclicity constraint, so the augmented Lagrangian
settles on acyclic-but-dense graphs and the Jacobian threshold cannot reliably pick the right edges to
keep. The clamp-every-500-iterations stability patch avoided catastrophic edge loss but let the dense
intermediate graphs persist. So the next rung's job is sharp: keep nonlinear mechanism modeling, but
impose *real* sparsity so precision climbs and SHD falls, without the per-seed lottery of a free neural
net.

The continuous-constraint machinery is still the right backbone — it is the only thing that avoids both
the combinatorial DAG search and the linear-Gaussian blindness — so I do not want to abandon NOTEARS's
`tr e^{W∘W} - d = 0`. The question is what to put *under* it. GraN-DAG put a free per-variable neural
net there and paid in precision. The fullest nonlinear-NOTEARS would keep a per-variable MLP but add an
`ℓ_1` penalty on the first-layer columns — variable-split into nonnegative parts so a bound-constrained
solver can carry it — which drives whole columns (whole parents) to zero, exactly the sparsity GraN-DAG
lacked; the dependence summary would then be `||∂_k f_j||_{L²}`, realized as the squared norm of the
`k`-th first-layer column, which equals `[W∘W]_{kj}` directly so `h` runs on it without a square root.
That is the principled answer, and I keep it as the reference for *why* sparsity belongs here. But I
have to be honest about what *this task's* fill of the function actually does, because it is a
deliberately different, leaner construction than that MLP-with-grouped-`ℓ_1` story — and the difference
is the whole same-named-vs-paper gap I must respect.

What the editable `run_causal_discovery` actually implements is a **two-stage hybrid**: a *linear*
NOTEARS to get a skeleton, then a *nonlinear* refinement with gradient-boosted regression on top. There
is no MLP anywhere in the fill, despite the name. Let me walk the construction in the order the code
runs it, because each stage answers a specific weakness of GraN-DAG.

Stage one is linear NOTEARS, exactly the original continuous program. Standardize `X`, then minimize
`0.5/n · ||X - XW||²_F + λ_1|W|_1` subject to `h(W) = tr e^{W∘W} - d = 0`, with `λ_1 = 0.01` and the
augmented Lagrangian. The acyclicity value and its gradient are computed from a *finite power series*
truncation of the matrix exponential (twelve terms) rather than a library `expm`, which is the harness's
way of keeping `_h` and `_h_grad` self-contained and consistent with each other — `expm_M = I + M + M²/2!
+ …` and `∇h = 2W ∘ expm(M)^T` with `M = W∘W`. The inner subproblem is solved with **L-BFGS-B**
(`jac=True`, 500 inner iters), the diagonal of `W` and its gradient zeroed each step to forbid
self-loops. The outer loop is the standard schedule: 30 augmented-Lagrangian rounds, `ρ ×10` whenever
`h` does not shrink to a quarter of its previous value, `α += ρ·h` dual ascent, stop when `|h| < 1e-8`
or `ρ > 1e16`. The init is a small random `W` (`0.01·randn`) to break the symmetry that stalls the
optimizer at zeros. This stage is cheap and gives me a *linear* skeleton — and crucially, a *causal
ordering* read off the column-sum of `|W|` (more downstream nodes accumulate more incoming weight),
which I will use to break ties into a DAG at the end.

Now, a *linear* skeleton on nonlinear data is exactly what failed at rung one. The whole point of stage
two is to repair that linearity *without* re-introducing GraN-DAG's free-net precision problem. The
construction: thresholid the linear `W` at `0.3` to get candidate parents, then *augment* the candidate
set for each node with any variable whose linear correlation with it exceeds `0.15` (so a nonlinearly-
strong-but-linearly-weak parent is not lost by the linear stage), and for each node fit a **gradient-
boosted regressor** (`GradientBoostingRegressor`, 100 trees, depth 3) of `X_j` on its candidate set.
Keep a candidate parent only if its boosted-tree *feature importance* exceeds `0.05`. This is the
nonlinear refinement: the boosted regressor sees curved mechanisms the linear `W` cannot, and the
feature-importance threshold is the sparsity discipline GraN-DAG never had — it directly answers "does
`X_j` actually use `X_k` nonlinearly," and prunes the rest. So the precision pressure that was missing
becomes the *core* of stage two.

The last step is DAG enforcement, and it is simpler than GraN-DAG's Jacobian pass. The nonlinear
refinement can re-introduce a cycle (boosted importances do not respect any ordering), so I impose the
topological order recovered from the *linear* stage: `order_score = sum |W|` over rows gives a downstream
ranking, `rank = argsort(order_score)` puts root-like nodes first, and any refined edge that points from
a higher-rank (downstream) node into a lower-rank (upstream) one is deleted. That guarantees acyclicity
by construction and inherits the linear stage's order. Output `B` already obeys the harness convention
`B[i,j] != 0` means `j -> i`. The full scaffold module is in the answer.

Let me be precise about the same-named-vs-paper gap, because it determines what I should expect. The
method named `notears_mlp` in this task is **not** the nonlinear MLP-NOTEARS at all: there is no neural
network, no per-variable MLP, no `||∂_k f_j||_{L²}` column-norm dependence summary, no grouped-`ℓ_1` on
first-layer weights, and no end-to-end continuous nonlinear program. It is **linear NOTEARS** (the
continuous acyclicity machinery in its original linear-coefficient form) **plus a gradient-boosted
nonlinear edge-selection refinement** — a hybrid that uses the smooth constraint only to get a linear
skeleton and an order, then repairs the nonlinearity with trees and feature-importance thresholding. The
canonical MLP-NOTEARS reasoning I keep as reference (per-node MLP, squared-column-norm into
`tr e^{W∘W}-d`, L-BFGS-B over split nonnegative weights) is *not* what runs here. The practical
consequence: where GraN-DAG over-connected because it had no pruning, this fill *under*-connects on the
hardest scenarios, because the *linear* stage one supplies the candidate pool and a linear skeleton on
strongly-nonlinear GP mechanisms can miss true edges entirely before the boosted refinement ever sees
them — the refinement can only prune and re-weight candidates the linear stage and the 0.15 correlation
screen proposed, not discover a parent that is linearly invisible.

So the expectations, read against GraN-DAG's measured shape. The defining trade is **precision up,
recall down**. Stage two's feature-importance pruning should push precision well above GraN-DAG's
0.11–0.50, and the SHD should fall on the dense-graph scenarios where GraN-DAG bloated — I expect
SF20-GP SHD to drop from GraN-DAG's 92 toward the 30s, and the variance to shrink because there is no
free-net seed lottery. But the linear-candidate bottleneck should *cap recall*: on **SF20-GP**, where
GP mechanisms are smooth and strongly nonlinear, the linear skeleton will miss many true edges, so I
expect recall to crater (well below GraN-DAG's 0.49) and F1 to actually come in *low* there — possibly
below GraN-DAG's 0.273 — a precision-heavy, recall-starved profile. On **ER20-Gauss**, where GraN-DAG
already did relatively well (F1 0.396), the hybrid's precision discipline should give a competitive,
*more stable* result — I expect F1 in the 0.3 range with much higher precision (0.5–0.6) and no seed-456
collapse, trading some recall for reliability. On **ER12-LowSample**, the linear stage is more robust to
small `n` than a per-variable MLP, and the boosted refinement on a 12-node graph with a tight candidate
pool should clearly beat GraN-DAG's 0.161 — I expect this to be the hybrid's best relative showing,
F1 in the upper 0.3s with precision well above GraN-DAG's 0.11. The single falsifiable claim that ranks
this rung above GraN-DAG: averaged across the three scenarios its F1 should edge out GraN-DAG's *and*
its SHD should be dramatically lower (the precision/SHD win is the point), even though it may lose to
GraN-DAG on SF20-GP recall specifically. And the residual weakness it leaves for a stronger rung to
exploit is now precise: the *linear* candidate-generation bottleneck. A method that recovers the
nonlinear *order* directly — without ever passing through a linear skeleton — should keep this rung's
precision discipline while restoring the recall the linear stage throws away, and beat it decisively.
