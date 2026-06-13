DirectLiNGAM's numbers tell me exactly what I suspected and exactly where to push. Its F1 is low on
every scenario — 0.319 on SF20-GP, 0.245 on ER20-Gauss, 0.188 on ER12-LowSample — and the SHD is
brutal: 60 on a 20-node scale-free graph, 96 on the 20-node ER graph. Read the precision and recall
together and the failure mode is unambiguous. Precision is ~0.18–0.24 across the board, recall ~0.18–
0.46; the method is laying down a lot of arrows and most of them are wrong. That is the dense,
wrong-direction graph I predicted: a linear residual `x_i - β x_j` on nonlinearly-generated data stays
`x_j`-dependent in *both* directions, so the exogeneity test can no longer cleanly separate source from
non-source, and the order comes out close to noise. The single sharpest confirmation is ER20-Gauss: it
is the *worst* F1 (0.245) and the worst SHD (96), which is exactly the double penalty I expected — the
mechanisms are nonlinear *and* the noise is Gaussian, so the Darmois–Skitovitch converse that the whole
exogeneity lemma rests on simply fails. The diagnosis is therefore not "tune DirectLiNGAM" but
"DirectLiNGAM is solving the wrong problem": it breaks direction symmetry with non-Gaussian noise and a
*linear* model, and the leverage on this data is the *nonlinearity*. So the next rung must model the
mechanisms as genuinely nonlinear functions. The question is how to do that without inheriting the two
diseases the whole field fights: the super-exponential combinatorial DAG search, and the
non-identifiability of direction.

Identifiability first, because it tells me the problem is even solvable. The additive-noise assumption
`X_j = f_j(X_{pa(j)}) + N_j` with independent noise and *nonlinear* `f_j` makes the DAG identifiable
from the distribution — and crucially this holds even when the noise is *Gaussian*, as long as `f` is
nonlinear. The linear-Gaussian model is the textbook unidentifiable case; the nonlinear-Gaussian one is
not. That is the precise reason DirectLiNGAM died on ER20-Gauss and a nonlinear method should not: the
nonlinearity, not the noise shape, is what carries the direction signal there. The intuition is the
same fingerprint I leaned on before — regress effect on cause and the residual is independent of the
cause; regress the wrong way and it stays dependent — but now the *model class I fit* has to be rich
enough to represent the nonlinear mechanism, or I cannot see the fingerprint at all. DirectLiNGAM's
model class was linear, so it was blind to it by construction.

Now the combinatorial wall. The space of DAGs on `d` nodes grows faster than `d!`, and acyclicity is a
global discrete property, so the classical algorithms are greedy: propose an edge, check no cycle
appeared, accept or reject. DirectLiNGAM dodged this by recovering an *order* rather than searching
graphs, but it paid with the linearity. I want to keep "don't search the discrete space" *and* get
nonlinear mechanisms. The device that does this is NOTEARS's smooth acyclicity characterization, and I
need to reproduce its argument carefully because I will run it twice. Take a nonnegative matrix `B`;
`(B^k)_{jj}` counts closed walks of length `k` through node `j`, so `tr(B^k)` counts length-`k` cycles,
and `B` is acyclic iff `tr(B^k) = 0` for every `k`. As a finite sum the powers overflow; the fix is the
matrix exponential, which reweights the length-`k` counts by `1/k!`, taming the explosion and giving
the clean statement `tr e^B = d` iff `B` is a DAG. For real, possibly-negative weights, replace `B` by
its Hadamard square so the count argument still applies: `h(W) = tr e^{W∘W} - d = 0`, with gradient
`(e^{W∘W})^T ∘ 2W`. Solve a smooth score subject to `h(W) = 0` with an augmented Lagrangian, and the
whole graph updates at once — no greedy search.

So why not just use NOTEARS? Because its `W` *is* the coefficient matrix of a *linear* SEM — the
contribution of variable `i` to `j` is the single scalar `W_{ij}`. There is no room for `X_j` to depend
nonlinearly on its parents, so on this data it would underfit the mechanisms and mis-score directions
exactly the way DirectLiNGAM did. I want the continuous-constraint paradigm but with a flexible,
*independent* nonlinear model per variable. The natural move: give each variable `j` its own neural
network taking the other variables as input and outputting the mean of `X_j`'s Gaussian conditional,
with a learned, parent-independent noise std — that is the ANM `X_j = f_j(parents) + N_j` written
exactly. Mask the `j`-th input so a variable cannot be its own parent.

Here is the wall that makes this nontrivial, and it is the heart of the method. In NOTEARS the
constraint lived on `W`, but `W` *was* the weighted adjacency. With a stack of neural nets there is no
single coefficient telling me whether `X_j` depends on `X_i`; variable `i` enters `NN_j` through a
tangle of weights across every layer. Before I can write `h`, I have to manufacture, out of the network
weights, a single nonnegative `(A)_{ij}` that is zero exactly when `NN_j`'s output does not depend on
input `i`. Think about how information flows. It travels from `i` to an output only along computation
*paths* through the hidden units, and a path is dead iff any weight on it is zero. So output `k` is
completely independent of input `i` iff *every* path from `i` to `k` is dead. Quantify a path by the
product of the absolute weights along it — nonnegative, and zero iff some link is zero — and "every path
dead" becomes "the sum of all path products is zero," since a sum of nonnegatives vanishes iff every
term does. The beautiful part: summing path products over all intermediate indices *is* matrix
multiplication of the absolute-value weight matrices. Stack them, `C = |W^{(L)}| ⋯ |W^{(1)}|`, and
`C_{ki}` is exactly the total path strength from input `i` to output `k`. Sum over the output components
to get the variable-level entry, set the diagonal to zero, and `A` is the nonnegative `d×d` matrix `h`
wanted — born nonnegative, so I do not even need the Hadamard square. The acyclicity constraint becomes
`h = tr e^{A} - d = 0`, and I have run NOTEARS's walk-counting argument twice: once over *neural-network
paths* to build `A`, once over *graph paths* to constrain it.

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

Two harness-specific choices are worth naming because they are exactly the same-named-vs-paper gap that
will shape the numbers. First, edge clamping: the gcastle default masks any input whose `A` entry falls
below `1e-4` *every step*, but the task fill deliberately applies the clamp only **every 500 iterations
with a stricter `1e-3` threshold**, on the explicit reasoning that per-step clamping irreversibly
removes edges too aggressively and destabilizes runs across seeds. That is a stability patch, not the
canonical schedule, and it means the method here is *less* prone to prematurely killing a true edge but
also slower to sparsify. Second, the final DAG enforcement: rather than thresholding `A` (which only
*upper-bounds* true dependence, since path products can cancel in the realized function or hidden units
can saturate), the fill computes the **realized sensitivity** — the expected absolute Jacobian of each
conditional's log-likelihood with respect to each input — and removes the weakest edges in increasing
order of that Jacobian until the graph is acyclic, checked by the `tr(A^k)=0` closed-walk fact on the
thresholded binary graph. This is a faithful piece of the canonical method (sensitivity-based
extraction), and it is the right strength estimate. What the harness does *not* expose, relative to the
fullest version of the procedure, is the CAM-style preliminary neighbor selection and the GAM
significance pruning — the fill has no PNS and no post-hoc significance test, so the only sparsity
pressure is the `ℓ`-free augmented Lagrangian, the edge clamping, and the Jacobian threshold. The full
scaffold module is in the answer.

So why do I expect GraN-DAG, despite being a genuinely nonlinear method, to *not* dominate the
ladder, and in particular to be a tricky middle rung rather than a clean win over DirectLiNGAM? Because
its weak point is precisely the place DirectLiNGAM was already weak: precision on dense graphs. With no
explicit sparsity penalty and a per-variable network free to fit spurious dependence, and with the
order learned only implicitly through the acyclicity constraint, the augmented Lagrangian can converge
to a graph that is *acyclic but over-connected* — low precision, high recall, high SHD — especially on
the 20-node graphs where there are many candidate edges and the Jacobian threshold has to remove a lot
of them in the right order. The stability patch (clamp every 500 iters) cuts both ways: it avoids
catastrophic edge loss but lets dense intermediate graphs persist. And the per-seed variance should be
large, because RMSprop on a non-convex augmented Lagrangian with reinitialized optimizer state lands in
different basins per seed.

Reading DirectLiNGAM's shape, here are the falsifiable expectations against its numbers. GraN-DAG
models nonlinearity, so on **ER20-Gauss** — the scenario where DirectLiNGAM was worst (F1 0.245)
*because* of the nonlinear-Gaussian double bind — it should improve the most, since the nonlinearity is
exactly what GraN-DAG sees and DirectLiNGAM could not; I expect its ER20-Gauss F1 to clear
DirectLiNGAM's by a clear margin and its SHD on that graph to fall well below 96. On **SF20-GP**,
however, I am *not* confident it beats DirectLiNGAM: the scale-free graph has hubs (high-degree nodes)
that an order-free continuous method can over-connect, and GP mechanisms are smooth, so the
no-sparsity-penalty over-connection could leave SHD *worse* than DirectLiNGAM's 60 even if F1 is
comparable — the precision should stay low. On **ER12-LowSample** with only 150 samples, a per-variable
two-layer MLP is over-parameterized; I expect the noisiest, possibly worst F1 of the three, with large
seed-to-seed variance. The single claim I will hold the next rung to: GraN-DAG should prove the
nonlinearity is where the leverage is by lifting ER20-Gauss off DirectLiNGAM's floor — but its lack of
an explicit sparsity/pruning stage should leave it *precision-starved and SHD-heavy*, so if a later rung
adds disciplined edge selection on top of a correctly-recovered nonlinear order, it should beat
GraN-DAG decisively on F1 and SHD at once. That is the gap I am setting up to close.
