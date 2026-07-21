The depth-2 floor told me where the implicit bias runs out, and it told me in numbers. On rank3-50 it
landed `test_rel_fro = 0.452` — and that is the *easy* environment, $30\%$ observed for a rank-3 matrix,
where even a gentle low-rank bias should recover almost perfectly. A relative error of $0.45$ there is a
mediocre recovery where I expected near-exact. On the canonical rank5-100 it got `0.596`, worse, and on
rank10-200 `0.923` — essentially nothing recovered, as I predicted for the deeply data-poor regime. The
tell is the `train_mse` column: `1e-9`, `2e-15`, `1e-14`. Every run drove the *training* loss to zero —
it fit the observed entries perfectly — and still missed the unobserved ones by $45$–$92\%$. So this is
not an optimization failure; depth-2 reached a zero-error interpolant every time. The problem is *which*
interpolant: the implicit bias of a depth-2 factorization, run with Adam and a not-tiny step from a
finite init, is not selecting the low-rank one sharply enough, even with samples to spare on rank3-50.
The bias I relied on to come for free is too weak, and the optimizer coarsens it further.

The two error columns say one more thing. In every environment `test_rel_fro` exceeds `full_rel_fro` by
almost exactly the factor $\sqrt{1-p}$ ($0.452$ vs $0.381$ at $p=0.3$, $0.596$ vs $0.532$ at $p=0.2$,
$0.923$ vs $0.878$ at $p=0.1$), which is what must happen when the observed entries are fit
near-perfectly and the reconstruction error is spread *uniformly* across seen and unseen entries. The
depth-2 fit is not localizing its mistakes; it is picking a globally wrong low-rank-ish matrix. So the
target for this rung is not "fit the data better" — the data is already fit — it is "select a better
matrix," which is exactly what an explicit nuclear-norm objective is for.

That diagnosis splits the path two ways. One direction is to make the implicit bias *stronger* — more
depth. But before I lean harder on the same implicit mechanism, I want a clean point of comparison: the
method that asks for the low-rank fit *explicitly*,
solved to genuine convergence rather than approximated by an adaptive optimizer from a finite init. If
depth-2's $0.45$ on rank3-50 is the implicit bias merely *approximating* nuclear-norm minimization, then
actually minimizing the nuclear norm should do at least as well, and the gap tells me how much of
depth-2's deficit was weak bias versus crude approximation.

The objective is the convex program

$$\text{minimize}\ \|X\|_\* \quad\text{subject to}\quad P_\Omega(X) = P_\Omega(M),$$

with $\|X\|_\* = \sum_i \sigma_i(X)$ the $\ell_1$ norm of the spectrum and $P_\Omega$ keeping the
observed entries. The nuclear norm is the right surrogate because it is the convex envelope of rank —
the tightest convex thing under rank — and under incoherence the program recovers an incoherent rank-$r$
matrix exactly from $O(nr\log n)$ entries. The objective is not in question; solving it cheaply at scale
is. An interior-point SDP treats this as $O(n^2)$ variables and factors a dense Newton system at
$\sim O(n^6)$ per step — at $n = 200$ that is $\sim 6\cdot10^{13}$ operations per iteration, hopelessly
past the under-30-minute wall-clock cap, and it throws away the one thing I know, that the answer is low
rank. A first-order proximal method is the opposite: its per-step cost is one SVD at $O(n^3)$, about
$8\cdot10^6$ operations at $n = 200$, seven orders cheaper, and cheaper still if I can keep the iterates
low rank. So I am committed to a proximal scheme; the design question is how to make each step both cheap
*and* faithful to the hard data constraint.

The shape of the answer looks like compressed sensing, where soft-thresholding
$S_\tau(x) = \text{sign}(x)(|x| - \tau)_+$ produces sparsity by shrinking small coefficients to exactly
zero. A low-rank matrix is one whose vector of singular values is sparse, so the matrix analogue is to
soft-threshold the singular values: for $Y = U\Sigma V^\*$,

$$D_\tau(Y) = U\,\text{diag}\big((\sigma_i - \tau)_+\big)\,V^\*.$$

If many singular values sit below $\tau$, the output has far lower rank than the input, adaptively in the
matrix's own singular basis. And this is not just an appealing operation: it is the exact proximity
operator of the nuclear norm, $D_\tau(Y) = \arg\min_X \tfrac12\|X - Y\|_F^2 + \tau\|X\|_\*$. The
subgradient optimality $0 \in X - Y + \tau\,\partial\|X\|_\*$ is satisfied by splitting $Y$'s SVD at the
threshold: the above-$\tau$ part gives $X = D_\tau(Y)$, and $Y - X = \tau(U_0 V_0^\* + W)$ with $W$
orthogonal to $U_0,V_0$ and $\|W\|_2 = \tau^{-1}\max\Sigma_1 \le 1$, so $Y - X \in \tau\,\partial\|X\|_\*$.
So $\tau$ is a direct rank dial — every singular value below it dies, every one above is pulled toward
zero by exactly $\tau$. That constant downward pull is also why the shrink alone never fits the data: the
surviving singular values come out systematically too small by $\tau$, so I will need a separate
mechanism to push them back up toward feasibility.

The naive way to enforce the constraint is to relax it — minimize
$\lambda\|X\|_\* + \tfrac12\|P_\Omega(X) - P_\Omega(M)\|_F^2$ — but that converges to the penalized
optimum, not an exact fit, and forcing the fit means shrinking $\lambda$, which shrinks the threshold
$\lambda\delta$, which stops the shrink from killing singular values, so the iterates are no longer low
rank. Cheapness (a large threshold) and accuracy (exact fit) fight. The fix is to let the residual
accumulate on its own track and shrink *that*:

$$X^k = D_\tau(Y^{k-1}),\qquad Y^k = Y^{k-1} + \delta_k\,P_\Omega(M - X^k),\qquad Y^0 = 0.$$

Now $\tau$ is decoupled from the step size and $Y$ is a running sum of residuals, supported entirely on
$\Omega$ by induction — sparse, at most $m$ nonzeros — so the only real work is one (partial) SVD per
step. Large $\tau$ now gives low-rank $X^k$ and sparse $Y^k$ *simultaneously*; the coupling that doomed
the penalized form is gone.

Read as Uzawa dual ascent, this iteration solves $\min\ \tau\|X\|_\* + \tfrac12\|X\|_F^2$ s.t.
$P_\Omega(X) = P_\Omega(M)$: the $Y$-update is exactly the dual step, and reverse-engineering the inner
minimization that a single shrink $D_\tau$ solves (using $P_\Omega Y = Y$) gives that $f$. The extra
$\tfrac12\|X\|_F^2$ is the price of a closed-form shrink and a bonus — it makes $f$ strongly convex, so
the solution is unique — and as $\tau\to\infty$ the nuclear term dominates and the minimizer converges
to the minimum-nuclear-norm solution. So large $\tau$ serves *both* accuracy and cheapness, the two
goals that fought before.

Convergence rests on that strong convexity: tracking the dual distance
$r_k = \|P_\Omega(Y^k - Y^\*)\|_F$ gives $r_k^2 \le r_{k-1}^2 - (2\delta_k - \delta_k^2)\|X^k - X^\*\|_F^2$,
a genuine decrease for $0 < \delta_k < 2$. That bound is conservative; the near-isometry
$\|P_\Omega(A)\|_F^2 \approx p\|A\|_F^2$ (with $p$ the observation ratio) means an ascent step of size
$\delta$ on the dual moves the primal effectively by $\delta p$, so I can push to $\delta = 1.2/p$ and
keep the *effective* step $\delta p = 1.2$ safely inside the decrease window: $\delta = 4$ on rank3-50,
$6$ on rank5-100, $12$ on rank10-200. This is where decoupling $\tau$ from $\delta$ pays off — a large
data step for fast convergence without touching the threshold that sets the rank. (This is not a rigorous
theorem — the iterate difference is not a fixed incoherent matrix — but it converges in practice.)

Two remaining constants. For $\tau$: on the synthetic generator $\|M\|_F \approx n\sqrt r$ and
$\|M\|_\* \approx nr$, so to keep the nuclear term about $10\times$ the Frobenius term I take $\tau = 5n$
($250$, $500$, $1000$ across the three environments) — large enough in absolute terms to force the shrink
to kill all but the few dominant modes. And a warm start: from $Y^0 = 0$ the first several shrinks return
exactly zero (while $k\delta\|P_\Omega(M)\| \le \tau$), so I leap straight to
$Y^0 = k_0\delta P_\Omega(M)$ with $k_0 = \lceil\tau/(\delta\|P_\Omega(M)\|)\rceil$, using the Frobenius
norm of the observed data in place of the spectral norm — a conservative overshoot
($\|P_\Omega(M)\|_F \ge \|P_\Omega(M)\|_2$) that lands just before the first informative iterate rather
than past it (on rank3-50, $k_0 = \lceil 250/(4\cdot 27.4)\rceil = 3$, skipping two provably-dead
iterations). Stopping is the KKT residual: $X = D_\tau(Y)$ holds by construction, so the only optimality
gap left is the constraint violation on $\Omega$, monitored as the masked squared residual, which by the
near-isometry tracks the true error on the unseen entries.

One harness deviation is consequential enough that I have to read it honestly. On rank10-200
($n\ge200$, $\text{rank\_hint}\ge10$) the fill caps the budget at *11 iterations* rather than the
requested `max_iters`, because a full $200\times200$ SVD every step over the whole budget only oscillates
and hits the wall-clock cap — a short SVT budget already reaches the intended low-but-finite
nuclear-norm baseline there. So whatever it scores on rank10-200 is "11 steps of SVT," not converged
nuclear-norm minimization, and I should not read it as the convex program's verdict.

I can also predict a qualitative flip in the `train_mse` column as direct evidence the shrink is doing
its job. Depth-2 drove training MSE to $\sim 10^{-9}$ by interpolating exactly; SVT will not, because the
$\tfrac12\|X\|_F^2$ term deliberately holds the surviving singular values below feasibility, so the
observed entries end up only approximately fit and the training residual should be visibly nonzero,
orders of magnitude above depth-2's floor. That is not a bug — it is the signature of trading exact
data-fit for a low-rank spectrum.

Against the floor, then. On the two well-sampled environments, where depth-2 left $0.452$ and $0.596$,
if that gap was weak/approximated bias rather than poor data, explicit nuclear-norm minimization run to
convergence should beat both clearly — I am now asking for the low-nuclear-norm fit directly instead of
hoping Adam approximates it. On rank10-200 I expect no such gain: $4000$ observations for a rank-10
matrix sits at the information floor where nuclear-norm minimization itself starts to fail, and the
11-iteration cap means I am not even running it to convergence, so it should stay near depth-2's $0.92$.
The clean test is the well-sampled pair — if the convex baseline beats it, depth-2's deficit was bias
strength, and the ceiling still to break is nuclear-norm strength itself.
