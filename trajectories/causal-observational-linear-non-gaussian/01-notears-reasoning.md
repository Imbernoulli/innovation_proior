The function the harness leaves open is small — input `X`, output an adjacency matrix `B` with the
support being the DAG and `B[i,j] != 0` read as `j -> i`. The model behind the data is a linear
structural equation model: each variable is a linear combination of its parents plus its own independent
noise, `x_i = sum_j w_{ji} x_j + e_i`. Stack the incoming-weight vectors into a matrix `W` whose column
`j` holds the weights flowing *into* node `j`, and the support of `W` — which entries are nonzero —
*is* the graph. So fitting looks trivial: I want the `W` that best reconstructs the data. The
reconstruction `XW` predicts each variable from the others, the residual is `X − XW`, and the
least-squares score `ℓ(W) = (1/2n)‖X − XW‖_F^2` is smooth and convex in `W`, with gradient
`−(1/n) X^T(X − XW)`. If that were all, I would call a solver and be done — and I have statistical
cover, because the least-squares minimizer is known to recover a true DAG with high probability even in
high dimensions, and the consistency carries over to non-Gaussian noise too, with no faithfulness
assumption. The whole game, then, is the single requirement that the graph `W` induces be **acyclic**.
Everything hard about this rung lives inside that one word, and everything I write below is a way to get
a black-box solver to respect it.

I want to be honest about why I am starting *here*, with the method that ignores the one thing this
benchmark is built around. The data are non-Gaussian by construction — Laplace on ER30, exponential on
ER50, uniform on SF100 — and the whole point of the task, as the prior art makes clear, is that
non-Gaussianity buys full edge orientation that the covariance alone cannot give. A least-squares score
reads only second-order structure; it does not exploit non-Gaussianity at all. So this is deliberately
the rung that does *not* use the task's defining resource. It is the floor: a strong, general-purpose
structure learner from a *different* method family — continuous optimization rather than discrete search
or independence-testing — so that whatever the LiNGAM rungs add on top of it is measured against a real,
competent baseline rather than against nothing. If a method that throws away non-Gaussianity already
does well on the easier scenarios and stumbles on the hard one, that stumble is precisely the gap the
next rung must fill, and I will want it drawn cleanly.

Now the acyclicity obstacle, because it is the entire substance of this rung. "Acyclic" is a property of
the *support pattern* of `W` — a combinatorial property of a discrete object — and the number of DAGs
on `d` nodes grows superexponentially in `d`. I cannot enumerate it; optimizing a score over it is
NP-hard. Every classical method is some clever way of moving inside the discrete set of acyclic graphs
without leaving it, and it is worth pricing each one against these specific graphs before I discard the
family, because the prices are what push me off it. Order-based search fixes a topological ordering and
searches the space of orderings — but that space has size `d!`, and `30! ≈ 2.6·10^32`, `50! ≈ 3·10^64`,
`100! ≈ 9·10^157`. Even with a clever score I cannot walk a meaningful fraction of `10^157` orderings
for SF100; heuristics prune, but then I have no optimality and a landscape riddled with local traps.
Exact parent-set enumeration is worse in a different way: for each node I would score subsets of the
other `d−1` as candidate parent sets, `2^{d−1}` of them unbounded — `2^{29} ≈ 5·10^8` per node on ER30,
`2^{49} ≈ 6·10^14` on ER50, `2^{99} ≈ 6·10^29` on SF100 — and the only way to tame that is to cap the
in-degree, an assumption the dense Erdős–Rényi graphs break (ER30 at `p=0.25` puts the average in-degree
near `0.25·29 ≈ 7` with the late nodes higher, ER50 at `p=0.2` near `0.2·49 ≈ 10`). Greedy equivalence
search sidesteps enumeration by adding or deleting one edge at a time and checking, after every single
move, that no cycle was created — an `O(d+e)` acyclicity test inside a loop over `~d^2` candidate moves
(`900` on ER30, `10^4` on SF100), repeated for as many steps as there are edges (`~109`, `~245`, `~291`
respectively). That is tractable, but greedy edge-at-a-time search is fast and reliable only when the
true graph is sparse in the sense of few parents per node, and it inherits a second limitation that
matters more: like PC's conditional-independence route, it scores on the covariance and so recovers only
the Markov equivalence class — skeleton plus collider-forced orientations. None of these is "write down
an objective and call a solver," and all of them treat acyclicity as something to enforce operationally
by staying inside a legal discrete region, inheriting that region's curse: bad scaling, or a
bounded-in-degree assumption these graphs violate.

That phrase — "write down an objective and call a solver" — is the lever, because two neighboring
problems became exactly that. Structure learning for *undirected* graphical models turned into a convex
log-determinant program over the precision matrix, the graphical lasso, and once it was a smooth
program black-box optimizers took over. Deep nets are the same story with SGD on a differentiable loss.
The directed case never got this, for an obvious reason: the undirected case has no acyclicity
constraint, and the directed case's acyclicity constraint has no smooth closed form. So the question
sharpens to a single, concrete demand. Can I find a function `h(W)` on real matrices — smooth, with a
cheap gradient — that is exactly zero when `W`'s graph is a DAG and nonzero otherwise? If I can, then
`min_W ℓ(W) s.t. h(W) = 0` is an ordinary smooth equality-constrained problem, and the entire
discrete-search apparatus, with its factorial and exponential price tags, evaporates.

Let me write down what `h` must satisfy before I go hunting, so I can reject bad candidates on sight:
(a) `h(W) = 0` iff `W` is acyclic — the whole point; (b) ideally `h` should *quantify* how cyclic the
graph is, larger for "more cyclic," because a continuous solver works far better descending a graded
measure than chasing a flat indicator that is zero on a measure-zero set and uninformative everywhere
else; (c) `h` smooth, so I have gradients; (d) `h` and `∇h` cheap, ideally something a library already
computes. The obvious constructions die on (c) or (d). The Euclidean distance from `W` to the nearest
DAG requires projecting onto a nasty nonconvex set — no closed form, no cheap gradient. Summing edge
weights along cyclic paths means enumerating cyclic paths, which is the combinatorial monster I am
trying to escape wearing a different coat. I need an *algebraic* handle on acyclicity, something where
linear algebra does the counting for me.

Forget weights and take a binary adjacency `B`. An elementary fact does the work: `(B^k)_{ii}` counts
the closed walks of length `k` from node `i` back to itself, so `tr(B^k)` is the total number of
length-`k` closed walks over all nodes. A directed graph is acyclic precisely when it has no closed walk
of any length — a single closed walk *is* a cycle. So acyclicity is equivalent to `tr(B^k) = 0` for
every `k ≥ 1`, and a combinatorial property has become an algebraic family of vanishing traces. Now I
must package "all of these vanish" into one smooth nonnegative scalar. The naive packaging sums the
traces, and it even has a closed form: `Σ_{k≥0} B^k = (I − B)^{-1}` (the Neumann series), so
`Σ_{k≥0} tr(B^k) = tr((I − B)^{-1}) = d + Σ_{k≥1} tr(B^k)`. Clean on paper, and it would give
`tr((I − B)^{-1}) = d` iff DAG — but the Neumann series only converges when the spectral radius
`r(B) < 1`, and along the optimization path a non-DAG iterate can easily push `r(B)` past 1, at which
point `I − B` approaches singular and the "closed form" is garbage or an outright division by zero.
Wall. The finite series `Σ_{k=1}^d tr(B^k)` sidesteps convergence — a DAG's `B` is nilpotent with
`B^d = 0` so the sum truncates exactly — but now look at how the terms scale. Successive terms grow by
roughly the factor `r(B)` each step (the ratio `tr(B^{k+1})/tr(B^k)` tracks the spectral radius), so
this is a geometric series with no built-in damping: if `r(B) > 1` the terms *increase* all the way to
`k = d`, and for a dense weighted matrix the entries of `B^k` climb like `(row-sum)^k`. That is a
gradient scaled by `10^{40}`-ish quantities on one iterate and `10^{-3}` on the next — the value and,
worse, the gradient become numerically worthless, and for large or dense enough graphs they overflow
double precision outright. This wall is the instructive one: I do not need raw walk counts, I need a
*reweighting* of them that keeps every term nonnegative, converges for *every* matrix regardless of
spectral radius, and crushes the high-`k` terms so the sum stays finite and well-scaled.

The reweighting that does all three at once is `1/k!`. Then `Σ_{k≥0} tr(B^k)/k! = tr(e^B)`, the matrix
exponential. The factorial defeats every matrix: the ratio of consecutive terms is now roughly
`r(B)/(k+1)`, which falls below 1 as soon as `k > r(B)` no matter how large the spectral radius is, so
the terms rise until `k ≈ r(B)` and then plunge super-geometrically, and the series converges for *any*
square matrix. Each term stays nonnegative for binary `B` (walk counts are nonnegative), so
`tr(e^B) = d + Σ_{k≥1} tr(B^k)/k!` equals `d` exactly when every `tr(B^k)` vanishes — i.e. iff `B` is a
DAG. And `e^B` is not exotic: it is a workhorse of numerical linear algebra, computed in `O(d^3)` by
scaling-and-squaring in every scientific library, gradient included. So for binary matrices,
`B` is a DAG ⟺ `tr(e^B) = d`, and the factorial is precisely the damping the finite series lacked.

But I need `h` on *real* signed `W`, and the nonnegativity argument leaned on `B`'s nonnegative entries.
For signed `W`, `(W^k)_{ii}` can be negative and the terms in `tr(e^W)` can cancel, so `tr(e^W) = d` no
longer forces acyclicity — cancellation could produce `d` on a cyclic graph. The repair is to replace
`W` by its entrywise square. The Hadamard product `W ∘ W` is nonnegative everywhere and, crucially, has
the *same support* as `W` — the same graph, since `W_{ij} ≠ 0 ⟺ (W∘W)_{ij} ≠ 0`. Run the binary
argument on `W ∘ W`, treating it as a nonnegative weighted adjacency:
`tr(e^{W∘W}) = d + Σ_{k≥1} tr((W∘W)^k)/k!`, every term nonnegative, equal to `d` iff `W`'s graph is
acyclic. Define `h(W) = tr(e^{W∘W}) − d`. It is smooth, nonnegative, zero iff DAG, and — because it is
the factorial-reweighted total of weighted closed walks — automatically cyclicity-quantifying, which
was desideratum (b), the one a bare indicator cannot give.

The gradient I need for the solver falls out of the chain rule, and I want to check it rather than quote
it. Write `M = W ∘ W`. The matrix-calculus identity is `∂ tr(e^M)/∂M = (e^M)^T` (it comes from
differentiating `tr(M^k)` term by term: `d tr(M^k) = k tr(M^{k-1} dM)`, so the sensitivity to `M` is
`(e^M)^T`). Then `M_{kl} = W_{kl}^2` gives `∂M_{kl}/∂W_{ij} = 2W_{ij}` when `(k,l) = (i,j)` and zero
otherwise, so the double sum in the chain rule collapses to a single term:
`∂h/∂W_{ij} = (e^M)^T_{ij} · 2W_{ij}`. In matrix form `∇h(W) = (e^{W∘W})^T ∘ 2W` — the *same* matrix
exponential I already computed for the value, Hadamard-multiplied by `2W` and transposed, so the
gradient costs essentially nothing beyond the value. The transpose is not cosmetic: `W ∘ W` need not be
symmetric, and `(e^M)^T ≠ e^M` in general; dropping it would misattribute the sensitivity of the closed
walks to the wrong edges.

Let me actually put a number on `h` so I trust it, not just assert it. Take the path `1→2→3` as a binary
`B`: it is nilpotent, `B^3 = 0`, every `tr(B^k) = 0`, so `tr(e^B) = 3` and `h = 0` — a DAG scores zero,
as it must. Now close the path into a 3-cycle by adding `3→1`. Now `B` is the cyclic permutation matrix,
`B^3 = I`, and `tr(B^k) = 3` whenever `3 | k` and `0` otherwise. So
`tr(e^B) = 3·(1/0! + 1/3! + 1/6! + 1/9! + …) = 3·(1 + 0.16667 + 0.001389 + 0.0000028 + …) ≈ 3·1.16806 =
3.5042`, giving `h = tr(e^B) − 3 ≈ 0.504 > 0`. A single 3-cycle registers as `h ≈ 0.5`, and a longer
cycle of length `L` would register as `≈ L/L! = 1/(L−1)!` per node summed — smaller, correctly reflecting
that a `1000`-node cycle is "less locally cyclic" per the factorial weighting but still strictly
positive. The function does exactly what it advertises on a case I can compute by hand.

So the program exists in full: minimize `ℓ(W)` subject to `h(W) = tr(e^{W∘W}) − d = 0` — a smooth score,
a smooth scalar equality constraint, real-matrix variables, the entire matrix updated at once (global,
not local edge-by-edge search), agnostic to the noise distribution. It is *not* convex — the constraint
surface `{h = 0}` is nonconvex — so I reach stationary points, not certified optima, like every
non-exact method; I am trading global guarantees for the ability to run at all on 50 and 100 nodes. The
right tool for a single smooth equality constraint is the augmented Lagrangian, and I want to be precise
about why not a plain penalty. A pure penalty `ℓ + (ρ/2)h^2` only drives `h → 0` as `ρ → ∞`, and large
`ρ` makes the inner problem badly conditioned (the Hessian's curvature along the constraint blows up),
so I would be solving an ill-conditioned problem to get a feasible point. The augmented Lagrangian adds
a multiplier, `L^ρ(W,α) = ℓ(W) + (ρ/2)h(W)^2 + α h(W)`, and the multiplier `α` drives `h` to zero while
`ρ` stays finite. The mechanism is exact: the dual function `D(α) = min_W L^ρ` has, by the envelope
theorem, `∇D(α) = h(W*_α)` — the constraint residual at the inner optimum *is* the dual gradient — so
dual ascent is `α ← α + ρ h(W*_α)`, with `ρ` playing the role of the step size. The outer loop solves the
inner problem for `W`, nudges `α`, and raises `ρ` only when feasibility stalls: if the new infeasibility
`h_new` is not below `0.25 · h_old`, I multiply `ρ` by 10 and re-solve, otherwise I accept and step `α`.

The inner subproblem, with `ρ, α` fixed, minimizes the smooth `f(W) = ℓ + (ρ/2)h^2 + α h` plus an `ℓ1`
sparsity term `λ‖W‖_1`. The `ℓ1` is nonsmooth, but it has structure I can exploit to stay in plain
smooth optimization instead of reaching for a proximal method. Split each variable into nonnegative
positive and negative parts `W = W^+ − W^-`; at any optimum at most one side of a coordinate is
positive, so `‖W‖_1 = 1^T(W^+ + W^-)` becomes *linear* in the doubled variables. The subproblem is then
a smooth objective with simple nonnegativity bounds, which I hand straight to L-BFGS-B — a limited-memory
quasi-Newton method, chosen deliberately because each function evaluation costs an `O(d^3)` matrix
exponential and I want the fewest, most informative steps rather than the many cheap steps a first-order
method would take. The smooth gradient is `G_smooth = ∇ℓ + (ρ h + α)∇h`; in the doubled variables the
gradient is `+G_smooth + λ` for the `W^+` block and `−G_smooth + λ` for the `W^-` block, the `+λ` being
the linearized sparsity pressure pushing both parts toward zero. One more bound trick pins the diagonal
to exactly zero — `(0,0)` bounds on the diagonal coordinates — forbidding self-loops. And I center the
columns of `X` first so the intercept does not leak into edge weights.

It is worth pricing this so I know it will actually run at `d = 100`. The doubled variable vector has
length `2d^2`: that is `1800` on ER30, `5000` on ER50, `20000` on SF100 — large for L-BFGS-B but well
within its comfort zone, since limited-memory BFGS stores only a handful of vectors of that length. Each
function evaluation is one `expm` (`O(d^3)`: `2.7·10^4`, `1.25·10^5`, `10^6` flops) plus the loss
gradient `X^T(X − XW)` (`O(n d^2)`: on SF100 that is `1000·10^4 = 10^7`, which actually dominates the
`expm`). With the outer loop capped at 100 iterations and each inner L-BFGS-B solve taking on the order
of a hundred evaluations, the total `expm` count is in the low thousands — heavy but entirely feasible,
and the reason I am willing to pay `O(d^3)` per step is that the matrix exponential is the price of
enforcing acyclicity globally instead of edge-by-edge.

After the outer loop converges I have `W̃` with `h(W̃)` at machine tolerance, not exactly zero —
*almost* a DAG, with a handful of tiny spurious weights left over from a near-feasible rather than
exactly-feasible solution. I round them off with a hard threshold: zero every entry with `|W̃_{ij}| ≤ ω`.
This is justified on both sides of the cut. On the discard side, because `h` *quantifies* cyclicity, a
near-feasible solution's residual cyclic mass is concentrated in tiny edges, so the entries I am zeroing
are exactly the ones carrying the leftover infeasibility; and hard-thresholding a regression estimate is
known to reduce false discoveries. On the keep side, the true edge weights are drawn from
`[−2,−0.5] ∪ [0.5,2]`, so every real edge has magnitude at least `0.5`. Setting `ω = 0.3` puts the
threshold safely below the smallest possible true edge (a margin of `0.2`) and above the sub-threshold
spurious debris — so in the ideal case the threshold removes only false edges and touches no true one.
That margin is the reason `ω` is a robust choice rather than a tuned one. The remaining fixed constants
are the reference defaults: `ρ=1, α=0, h=∞` to start; the outer iterations capped at 100 and `ρ` capped
at `10^{16}`; feasibility declared at `h ≤ 10^{-8}`; the progress factor `0.25` and the `ρ` multiplier
`×10`; and `λ = 0.1` as the sparsity level, which sits well below the loss scale (the per-sample
normalized `ℓ` is `O(1)` while `λ‖W‖_1` with `~few` edges of magnitude `~1` per column is also `O(1)`,
so the two terms are comparable rather than one swamping the other). One harness detail is load-bearing
and easy to get catastrophically wrong: the reference implementation reads `W[i,j] ≠ 0` as `i → j`, but
this task's convention is `B[i,j] ≠ 0 ⇒ j → i`, the exact opposite, so the final adjacency must be
**transposed** before return. That single transpose is the whole difference between a correct graph and
a fully reversed one; since the metric scores direction and requires both skeleton and direction to
match for a true positive, getting the transpose wrong would zero the score on every edge. (The full
scaffold module — the augmented-Lagrangian loop and the acyclicity function `h` filling the one open
slot — is in the answer.)

Now reason about what this floor should do, because that is the entire point of running it. The
least-squares score is direction-blind in the sense that matters: it does not read the non-Gaussian
fingerprint that orients edges, so it relies on the magnitude/sparsity pattern and the acyclicity
constraint to pin orientation, which is exactly the information a Markov-equivalence-class method has and
no more. On the smaller, less dense ER30 (30 nodes, `p=0.25`, 1000 samples) the signal-to-noise per edge
is high and the sample-to-parameter ratio is comfortable — each node's incoming weights are estimated
from `1000` samples against at most `~7` real parents, and `1000/29 ≈ 34` samples per candidate
coefficient — so the continuous program should fit it well: I expect strong F1 and small SHD, with the
main risk being an unlucky seed where the dense small graph confuses a few orientations. ER50 doubles
the nodes but also doubles the samples to 2000, so the per-parameter ratio actually *improves* slightly
(`2000/49 ≈ 41`); against that, `d = 50` makes each matrix exponential and the `5000`-variable L-BFGS-B
solve much heavier and the nonconvex landscape harder to descend cleanly, so I expect a real but milder
drop than the raw node count would suggest — the extra samples and the harder optimization roughly
fighting to a draw. SF100 is the one I am worried about, and the arithmetic says why. It has 100 nodes
but only 1000 samples, a sample-to-parameter ratio of `1000/99 ≈ 10`, the thinnest of the three; it is
hub-heavy scale-free, so a hub carries many incident edges whose orientation all has to be pinned from
second-order structure alone; and its noise is *uniform*, which is light-tailed and *sub*-Gaussian, so
even a method that wanted higher-order signal would find the least of it here. The hub edges are where a
covariance-only fit is weakest: faced with many ambiguous edges incident to a hub and a sparsity penalty
that punishes keeping them, the threshold-and-acyclicity rounding will tend to *prune* the ambiguous
ones rather than risk a reversal that the acyclicity constraint would then have to unwind elsewhere. So
my mechanistic expectation is that SF100 is where this rung is weakest — lower F1, much larger SHD — and
that the error will show as an asymmetry in which recall suffers more than precision, because pruning
ambiguous hub edges drops true positives (recall) while keeping the surviving edges mostly correct
(precision). I hold that direction loosely — the acyclicity rounding could equally force a batch of
reversals that would instead dent precision — so the honest statement is: I expect SF100 to be worst and
I expect a precision/recall split; the feedback table will tell me which way it leans, and that lean is
diagnostic. Whichever way it goes, the diagnosis is already pointed at the next rung: this method's
ceiling is set by its refusal to use non-Gaussianity, so the orientation errors on the hard graph are
the gap, and the fix is to stop reading only
second-order structure and switch the same `run_causal_discovery` slot to an engine built directly on
the non-Gaussian signal this least-squares score throws away.
