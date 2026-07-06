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

Before committing, let me lay the real options side by side, because "model nonlinearity" admits more
than one architecture and I want the elimination to be on the merits. One route is nonparametric
conditional-independence testing — kernel CI tests feeding a PC-style search. It sees nonlinear
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
need to reproduce its argument carefully because I will run it twice. Take a nonnegative matrix `B`;
`(B^k)_{jj}` counts closed walks of length `k` through node `j`, so `tr(B^k)` counts length-`k` cycles,
and `B` is acyclic iff `tr(B^k) = 0` for every `k`. Let me sanity-check the counting on the smallest
case: a single 2-cycle `1→2→1` is the matrix with `B_{12}=B_{21}=1` and zeros elsewhere; then
`B² = I`, `tr(B²) = 2 > 0`, correctly flagging the cycle, while any strictly-triangular `B` has
`B^k → 0` and every trace vanishes. As a finite sum over all `k` the powers overflow; the fix is the
matrix exponential, which reweights the length-`k` counts by `1/k!`, taming the explosion and giving
the clean statement `tr e^B = d` iff `B` is a DAG (the `d` is `tr(B⁰)=tr(I)`). For real,
possibly-negative weights, replace `B` by its Hadamard square so the count argument still applies:
`h(W) = tr e^{W∘W} - d = 0`, with gradient `(e^{W∘W})^T ∘ 2W`. Solve a smooth score subject to
`h(W)=0` with an augmented Lagrangian, and the whole graph updates at once — no greedy search.

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
path by the product of the absolute weights along it — nonnegative, and zero iff some link is zero —
and "every path dead" becomes "the sum of all path products is zero," since a sum of nonnegatives
vanishes iff every term does. The beautiful part: summing path products over all intermediate indices
*is* matrix multiplication of the absolute-value weight matrices. Let me check the shapes so the
identity is not just a slogan. With layer widths `[d, 10, 10, 1]` the per-variable weight matrices are
`|W^{(1)}|` of shape `(10,d)`, `|W^{(2)}|` of shape `(10,10)`, `|W^{(3)}|` of shape `(1,10)`; the
product `C = |W^{(3)}| |W^{(2)}| |W^{(1)}|` has shape `(1,d)`, and `C_{0i}` is exactly the total path
strength from input `i` to the scalar output. Stack that row over all `d` variables to get a `(d,d)`
matrix, sum over the (here trivial) output component, set the diagonal to zero, and `A` is the
nonnegative `d×d` matrix `h` wanted — born nonnegative, so I do not even need the Hadamard square. The
acyclicity constraint becomes `h = tr e^{A} - d = 0`, and I have run NOTEARS's walk-counting argument
twice: once over *neural-network paths* to build `A`, once over *graph paths* to constrain it.

That settles the principle. Now I have to be honest about what *this task's* fill of the function
actually implements versus the cleanest version of the method, because the harness imposes constraints
the polished method does not. The editable `run_causal_discovery` builds a per-variable MLP model with
**two hidden layers of ten units and leaky-ReLU**, Xavier-initialized, weights stacked as a `(d, out,
in)` tensor per layer so all `d` networks evaluate in parallel via one `einsum`; the first layer is
masked by an `adjacency` matrix (initialized to all-ones-minus-identity) so a variable never sees
itself. The score is the per-variable Gaussian log-likelihood with a learned per-variable `log_std`
(the ANM noise). The adjacency `A` is the **path-normalized** product of absolute weight matrices —
the raw path-product divided by the product of all-ones matrices through the same mask, which keeps the
entries on a comparable scale across depth — column-summed over outputs and transposed into the `i->j`
reading. The acyclicity is `tr(matrix_exp(A)) - d`. The optimizer is **RMSprop at lr 1e-3** with an
augmented Lagrangian (`μ_0 = 1e-3`, `λ_0 = 0`), the convergence-based dual/penalty schedule
(`λ += μ·h`, escalate `μ ×10` when the constraint stops shrinking by a factor, reset the optimizer
state on each update), 80/20 train/validation split with the held-out augmented Lagrangian as the
convergence signal, and up to 30000 iterations with `h_tol = 1e-8`.

Two mechanical details in that schedule deserve a second look because they are where an augmented
Lagrangian either converges or thrashes. The dual update `λ += μ·h` is ordinary dual ascent — it pushes
`λ` up in proportion to how badly the constraint is violated — but the penalty escalation `μ ×10` fires
only when `h` *stops shrinking*, specifically when the new constraint value is not below `0.9×` the
previous one. The logic is that if steadily tightening `λ` is still driving `h` down, I should let it
keep working at the current `μ`; only when progress stalls does the quadratic penalty need to get
stiffer. And each time `μ` or `λ` changes I reset the RMSprop state, because RMSprop's per-parameter
running averages of squared gradients are calibrated to the *old* loss surface, and the penalty change
discontinuously reshapes that surface — carrying stale second-moment estimates across the change would
send the first few steps in wrong-sized directions. That reset is also, I note, the seed-variance
amplifier: every reset restarts the adaptive scaling from scratch on a non-convex surface, so which
basin the run rolls into is sensitive to the exact trajectory, and two seeds diverge. The
path-normalization in `w_adj` is the other detail worth stating plainly: the raw product of absolute
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

Two harness-specific choices are worth naming because they are exactly the same-named-vs-paper gap that
will shape the numbers. First, edge clamping: the gcastle default masks any input whose `A` entry falls
below `1e-4` *every step*, but the task fill deliberately applies the clamp only **every 500 iterations
with a stricter `1e-3` threshold**, on the explicit reasoning that per-step clamping irreversibly
removes edges too aggressively and destabilizes runs across seeds. That is a stability patch, not the
canonical schedule, and it means the method here is *less* prone to prematurely killing a true edge but
also slower to sparsify — it will let a dense intermediate graph persist longer. Second, the final DAG
enforcement: rather than thresholding `A` (which only *upper-bounds* true dependence, since path
products can cancel in the realized function or hidden units can saturate), the fill computes the
**realized sensitivity** — the expected absolute Jacobian of each conditional's log-likelihood with
respect to each input — and removes the weakest edges in increasing order of that Jacobian until the
graph is acyclic, checked by the `tr(A^k)=0` closed-walk fact on the thresholded binary graph. This is
a faithful piece of the canonical method (sensitivity-based extraction), and it is the right strength
estimate. It is worth being precise about *why* `A` is only an upper bound and the Jacobian is the
honest quantity, because the gap between them is exactly the over-connection I am worried about. `A`
sums *absolute* path products, so it can never cancel: if there exists any path from `i` to `j` with
all-nonzero weights, `A_{ij} > 0`, and the constraint dutifully treats `i` as a potential parent of
`j`. But the *realized* function can be flat in `x_i` even when such a path exists — a leaky-ReLU unit
whose pre-activation sits on one side of the kink contributes a constant local slope, and two paths
with opposite signs can cancel in the actual output even though their absolute products both count in
`A`. The expected absolute Jacobian `E[|∂ log-lik_j / ∂ x_i|]` measures the dependence that actually
survives in the fitted function over the data distribution, so it zeroes out edges that `A` keeps but
the mechanism does not use. Thresholding `A` would therefore lock in false parents that a sensitivity
read would drop; using the Jacobian is what gives the pruning a chance — but only a chance, because it
is still a single threshold with no penalty behind it. What the harness does *not* expose, relative to the fullest version of the procedure, is any
preliminary neighbor-selection screen and any post-hoc significance-based pruning step — the fill has
neither, so the only sparsity pressure is the constraint-plus-clamp-plus-Jacobian-threshold, with no
explicit edge penalty anywhere in the objective.

So why do I expect GraN-DAG, despite being a genuinely nonlinear method, to *not* dominate the ladder,
and in particular to be a tricky middle rung rather than a clean win over DirectLiNGAM? Because its
weak point is precisely the place DirectLiNGAM was already weak: precision on dense graphs. With no
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
the three, with large seed-to-seed variance. The single claim I will hold the next rung to: GraN-DAG
should prove the nonlinearity is where the leverage is by lifting ER20-Gauss off DirectLiNGAM's floor —
but its lack of an explicit sparsity/pruning stage should leave it *precision-starved and SHD-heavy*,
so if a later rung adds disciplined edge selection on top of a correctly-recovered nonlinear order, it
should beat GraN-DAG decisively on F1 and SHD at once. That is the gap I am setting up to close. The
first cell I will read is ER20-Gauss: if the nonlinearity is genuinely the leverage, that is where the
lift off DirectLiNGAM's 0.245 must show up, and if it does not, my whole diagnosis is wrong.
