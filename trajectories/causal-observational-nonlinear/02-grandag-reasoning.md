DirectLiNGAM's numbers tell me exactly what I suspected and exactly where to push. Its F1 is low on
every scenario — 0.319 on SF20-GP, 0.245 on ER20-Gauss, 0.188 on ER12-LowSample, averaging 0.250 — and
the SHD is brutal: 60 on a 20-node scale-free graph, 96 on the 20-node ER graph. Read the precision and
recall together and the failure mode is unambiguous. Precision is ~0.18–0.24 across the board, recall
~0.18–0.46; the method is laying down a lot of arrows and most of them are wrong. Let me actually back
the edge counts out of those rates rather than eyeball them, because the exact numbers are the
diagnosis. The SF20-GP graph is scale-free on 20 nodes, and its true edge count is fixed by the
generator — a Barabási–Albert graph with two edges per arriving node gives `2·(20-2) = 36` edges, and
indeed the precision/recall pairs across all three seeds are only consistent with a true count of 36
(seed 42: recall `0.389 = 14/36`; seed 456: `0.444 = 16/36`). With 36 true edges and recall `0.389`,
DirectLiNGAM lands only about `14` of them, yet precision `0.184` means it drew `14/0.184 ≈ 76` edges
total — more than twice the true graph, roughly `62` of them false. That is the dense, wrong-direction
graph I predicted made quantitative: not "a few reversed arrows" but a triangular fill from a scrambled
order that over-draws by a factor of two and gets most of what it draws wrong.

The single sharpest confirmation is ER20-Gauss: it is the *worst* F1 (0.245) and the worst SHD (96),
which is exactly the double penalty I expected — the mechanisms are nonlinear *and* the noise is
Gaussian, so the Darmois–Skitovitch converse that the whole exogeneity lemma rests on simply fails.
And there is a second, quieter tell in the low-sample scenario that refines the picture. On
ER12-LowSample the graphs have roughly 19–23 true edges (the precision/recall pairs pin seed 42 at 19,
seed 123 at 21, seed 456 at 23), and there DirectLiNGAM actually draws *fewer* edges than truth —
around 16–22 predicted — because at `n=150` the estimated coefficients are small and the `|B|>0.01`
threshold silently deletes the weak ones. But precision is still only `0.14–0.25`, so it is not
"sparse and right," it is "sparse and wrong": of its ~16–22 drawn edges only 3–4 are correct. So the
failure mode is uniform in character (wrong direction) even though its *density* flips between the
20-node and 12-node cases. The diagnosis is therefore not "tune DirectLiNGAM" but "DirectLiNGAM is
solving the wrong problem": it breaks direction symmetry with non-Gaussian noise and a *linear* model,
and the leverage on this data is the *nonlinearity*. So the next rung must model the mechanisms as
genuinely nonlinear functions. The question is how to do that without inheriting the two diseases the
whole field fights: the super-exponential combinatorial DAG search, and the non-identifiability of
direction.

"Model nonlinearity" admits more than one architecture, so the elimination has to be on the merits.
One route is nonparametric conditional-independence testing — kernel CI tests feeding a PC-style
search. It sees nonlinear
dependence, but it inherits PC's two problems at once: the number of CI tests is combinatorial in the
worst case, and the output is a CPDAG, so it stops at the equivalence class exactly where
DirectLiNGAM's whole point was to escape it — I would be trading the linear disease for the
orientation disease. A second route is to keep NOTEARS's smooth acyclicity but leave the mechanism
*linear*, i.e. plain linear NOTEARS. That fails for a reason I can state exactly: its `W_{ij}` is a
single scalar coefficient, the entire contribution of variable `i` to `j`, so there is no functional
degree of freedom for `x_j` to bend with its parents — it would underfit the curved mechanisms and
mis-score directions the same way DirectLiNGAM did, just with a different optimizer. A third route is
pairwise additive-noise tests across all `d(d-1)/2` pairs, but those give me local orientations with
no global acyclicity guarantee and no principled way to stitch them into one DAG. What survives is the
combination I actually want: keep the continuous-constraint paradigm that avoids the discrete search,
but put a *flexible per-variable nonlinear model* under it. That is GraN-DAG, and the rest of the
derivation is making that combination actually work.

Identifiability first, because it tells me the problem is even solvable. The additive-noise assumption
`X_j = f_j(X_{pa(j)}) + N_j` with independent noise and *nonlinear* `f_j` makes the DAG identifiable
from the distribution — and crucially this holds even when the noise is *Gaussian*, as long as `f` is
nonlinear. The linear-Gaussian model is the textbook unidentifiable case; the nonlinear-Gaussian one
is not. That is the precise reason DirectLiNGAM died on ER20-Gauss and a nonlinear method should not:
the nonlinearity, not the noise shape, is what carries the direction signal there. The intuition is
the same fingerprint I leaned on before — regress effect on cause and the residual is independent of
the cause; regress the wrong way and it stays dependent — but now the *model class I fit* has to be
rich enough to represent the nonlinear mechanism, or I cannot see the fingerprint at all.
DirectLiNGAM's model class was linear, so it was blind to it by construction; this is the one place I
can be certain the next rung improves on the floor, and it is exactly the ER20-Gauss cell where I will
look first.

Now the combinatorial wall. The space of DAGs on `d` nodes grows faster than `d!`, and acyclicity is
a global discrete property, so the classical algorithms are greedy: propose an edge, check no cycle
appeared, accept or reject. DirectLiNGAM dodged this by recovering an *order* rather than searching
graphs, but it paid with the linearity. I want to keep "don't search the discrete space" *and* get
nonlinear mechanisms. The device that does this is NOTEARS's smooth acyclicity characterization, and I
need its argument carefully because I will run it twice. For a nonnegative matrix `B`, `(B^k)_{jj}`
counts closed walks of length `k` through node `j`, so `tr(B^k)` counts length-`k` cycles and `B` is
acyclic iff `tr(B^k) = 0` for every `k`. Summed over all `k` the powers overflow; the matrix
exponential reweights the length-`k` counts by `1/k!`, taming the explosion and giving the clean
statement `tr e^B = d` iff `B` is a DAG (the `d` is `tr(B⁰)=tr(I)`). For real, possibly-negative
weights, replace `B` by its Hadamard square: `h(W) = tr e^{W∘W} - d = 0`, gradient
`(e^{W∘W})^T ∘ 2W`. Solve a smooth score subject to `h(W)=0` with an augmented Lagrangian and the
whole graph updates at once — no greedy search.

So why not just use NOTEARS directly? Because its `W` *is* the coefficient matrix of a *linear* SEM —
the linear-mechanism objection I already used to eliminate it above. I want the continuous-constraint
paradigm but with a flexible, *independent* nonlinear model per variable. The natural move: give each
variable `j` its own neural network taking the other variables as input and outputting the mean of
`X_j`'s Gaussian conditional, with a learned, parent-independent noise std — that is the ANM
`X_j = f_j(parents) + N_j` written exactly. Mask the `j`-th input so a variable cannot be its own
parent.

Here is the wall that makes this nontrivial, and it is the heart of the method. In NOTEARS the
constraint lived on `W`, but `W` *was* the weighted adjacency. With a stack of neural nets there is no
single coefficient telling me whether `X_j` depends on `X_i`; variable `i` enters `NN_j` through a
tangle of weights across every layer. Before I can write `h`, I have to manufacture, out of the
network weights, a single nonnegative `(A)_{ij}` that is zero exactly when `NN_j`'s output does not
depend on input `i`. Think about how information flows. It travels from `i` to an output only along
computation *paths* through the hidden units, and a path is dead iff any weight on it is zero. So
output `k` is completely independent of input `i` iff *every* path from `i` to `k` is dead. Quantify a
path by the product of the absolute weights along it — nonnegative, zero iff some link is zero — and
"every path dead" becomes "the sum of all path products is zero," since a sum of nonnegatives
vanishes iff every term does. The key step: summing path products over all intermediate indices *is*
matrix multiplication of the absolute-value weight matrices, so with layer widths `[d, 10, 10, 1]`
the product `|W^{(3)}| |W^{(2)}| |W^{(1)}|` is a `(1,d)` row whose `i`-th entry is the total path
strength from input `i` to the output. Stack over all `d` variables, sum out the output component,
zero the diagonal, and `A` is the nonnegative `d×d` matrix `h` wanted — born nonnegative, so I do not
even need the Hadamard square. The constraint becomes `h = tr e^{A} - d = 0`, and I have run the
walk-counting argument twice: once over *network paths* to build `A`, once over *graph paths* to
constrain it.

That settles the principle. Concretely I build a per-variable MLP with **two hidden layers of ten
units and leaky-ReLU**, Xavier-initialized, weights stacked as a `(d, out, in)` tensor per layer so
all `d` networks evaluate in parallel via one `einsum`; the first layer is masked by an `adjacency`
matrix (all-ones-minus-identity) so a variable never sees itself. The score is the per-variable
Gaussian log-likelihood with a learned per-variable `log_std` (the ANM noise). The adjacency `A` is
the **path-normalized** product of absolute weight matrices, column-summed over outputs and
transposed into the `i->j` reading, and the acyclicity is `tr(matrix_exp(A)) - d`. The optimizer is
**RMSprop at lr 1e-3** with an augmented Lagrangian (`μ_0 = 1e-3`, `λ_0 = 0`), a convergence-based
dual/penalty schedule, an 80/20 train/validation split with the held-out augmented Lagrangian as the
convergence signal, and up to 30000 iterations with `h_tol = 1e-8`.

Two mechanical details in that schedule are where an augmented Lagrangian either converges or
thrashes. The dual update `λ += μ·h` is ordinary dual ascent, but the penalty escalation `μ ×10` fires
only when `h` *stops shrinking*, specifically when the new constraint value is not below `0.9×` the
previous one. The logic is that if steadily tightening `λ` is still driving `h` down, I should let it
keep working at the current `μ`; only when progress stalls does the quadratic penalty need to get
stiffer. And each time `μ` or `λ` changes I reset the RMSprop state, because RMSprop's per-parameter
running averages of squared gradients are calibrated to the *old* loss surface, and the penalty change
discontinuously reshapes that surface — carrying stale second-moment estimates across the change would
send the first few steps in wrong-sized directions. That reset is also, I note, the seed-variance
amplifier: every reset restarts the adaptive scaling from scratch on a non-convex surface, so which
basin the run rolls into is sensitive to the exact trajectory, and two seeds diverge. The
path-normalization in `w_adj` is the second detail: the raw product of absolute
weight matrices grows or shrinks systematically with depth and width (three matmuls compound the
magnitudes), so dividing by the product of *all-ones* matrices through the same mask rescales every
entry by the number of paths it summed over — turning a raw path-strength into an average path-strength
that is comparable across the network regardless of depth. Without it the `1e-3` clamp threshold would
mean different things at different scales; with it the threshold is a genuine relative cutoff.

Let me count parameters, because the count is the sharpest predictor I have of where this will fail.
Each per-variable network has `d·10` first-layer weights, `10·10` in the second, and `10·1` in the
third — at `d=20` that is `200 + 100 + 10 = 310` weights, plus biases and a noise std, fit against the
samples of that one variable. On the 2000-sample scenarios `310` parameters is comfortable. On
ER12-LowSample, `d=12` shrinks the first layer to `120`, but the sample count is only `150`, so each
conditional is fitting on the order of `230` parameters to `150` points — more parameters than data.
A leaky-ReLU MLP with more weights than samples and no explicit penalty will interpolate noise, which
means it will *manufacture* dependence on inputs that are not true parents. So before I even run it I
expect ER12-LowSample to be this method's worst scenario, dominated by over-fit spurious edges, with
large seed-to-seed variance — an arithmetic prediction, not a hunch.

Two deliberate deviations from the canonical schedule shape the numbers. First, edge clamping: rather
than masking any input whose `A` entry falls below `1e-4` every step, I apply the clamp only **every
500 iterations with a stricter `1e-3` threshold**, because per-step clamping irreversibly removes
edges too aggressively and destabilizes runs across seeds — a stability patch that is *less* prone to
killing a true edge but slower to sparsify, so it lets a dense intermediate graph persist longer.
Second, DAG enforcement by **realized sensitivity** rather than by thresholding `A`. `A` sums
*absolute* path products, so it only *upper-bounds* dependence: if any all-nonzero path from `i` to
`j` exists, `A_{ij} > 0`, but the realized function can still be flat in `x_i` — leaky-ReLU units
saturate, opposite-sign paths cancel in the actual output. The expected absolute Jacobian
`E[|∂ log-lik_j / ∂ x_i|]` measures the dependence that actually survives in the fitted function, so I
remove edges in increasing order of that Jacobian until the graph is acyclic (checked by `tr(A^k)=0`).
Thresholding `A` would lock in false parents the sensitivity read drops; the Jacobian gives pruning a
chance — but only a chance, since it is still a single threshold with no penalty behind it, and there
is no neighbor-selection screen and no significance-pruning stage. The only sparsity pressure is
constraint-plus-clamp-plus-Jacobian-threshold, with no explicit edge penalty in the objective.

So why do I expect GraN-DAG, despite genuinely modeling nonlinearity, to be a tricky middle rung
rather than a clean win over DirectLiNGAM? Its weak point is precisely where DirectLiNGAM was already
weak: precision on dense graphs. With no
explicit sparsity penalty and a per-variable network free to fit spurious dependence, and with the
order learned only implicitly through the acyclicity constraint, the augmented Lagrangian can converge
to a graph that is *acyclic but over-connected* — low precision, high recall, high SHD — especially on
the 20-node graphs where there are many candidate edges and the Jacobian threshold has to remove a lot
of them in the right order. The stability patch (clamp every 500 iters) cuts both ways: it avoids
catastrophic edge loss but lets dense intermediate graphs persist. And the per-seed variance should be
large, because RMSprop on a non-convex augmented Lagrangian with reinitialized optimizer state lands in
different basins per seed — there is no reason two seeds settle in the same one.

Reading DirectLiNGAM's shape, here are the falsifiable expectations against its numbers. GraN-DAG
models nonlinearity, so on **ER20-Gauss** — the scenario where DirectLiNGAM was worst (F1 0.245, SHD
96) *because* of the nonlinear-Gaussian double bind — it should improve the most, since the
nonlinearity is exactly what GraN-DAG sees and DirectLiNGAM could not; I expect its ER20-Gauss F1 to
clear DirectLiNGAM's by a clear margin and its SHD on that graph to fall well below 96. On **SF20-GP**,
however, I am *not* confident it beats DirectLiNGAM: the scale-free graph concentrates its 36 edges on
a few hubs, and a hub is exactly the structure an order-free continuous method mishandles — a hub feeds
many children, so if the constraint settles the hub's incident edges even slightly wrong it corrupts a
large fraction of the graph at once, and a free per-variable net has every incentive to wire spurious
edges *into* a hub because the hub is correlated with everything downstream of it. GP mechanisms are
smooth, so the no-sparsity-penalty over-connection could leave SHD *worse* than DirectLiNGAM's 60 even
if F1 is comparable — the precision should stay low, and the predicted-edge count could easily exceed
DirectLiNGAM's already-inflated `76`. On **ER12-LowSample** with only 150 samples, the `230`-parameter
per-variable MLP is over-parameterized by my count above; I expect the noisiest, possibly worst F1 of
the three, with large seed variance. So GraN-DAG should prove the nonlinearity is the leverage by
lifting ER20-Gauss off DirectLiNGAM's floor, but its lack of an explicit sparsity/pruning stage should
leave it precision-starved and SHD-heavy — the gap a later rung closes by adding disciplined edge
selection on a correctly-recovered nonlinear order. The first cell I read is ER20-Gauss: if the lift
off 0.245 does not show up there, my whole diagnosis is wrong.
