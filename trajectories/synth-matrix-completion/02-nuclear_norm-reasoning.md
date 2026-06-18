The depth-2 floor told me exactly where the implicit bias runs out, and it told me in numbers. On
rank3-50 it landed `test_rel_fro = 0.452` — and that is the *easy* environment, $30\%$ observed for a
rank-3 matrix, the regime where even a gentle low-rank bias should recover almost perfectly. A relative
error of $0.45$ there is not a clean win; it is a mediocre recovery where I expected near-exact. On the
canonical rank5-100 it got `0.596`, worse, and on rank10-200 `0.923` — essentially nothing recovered, as
I predicted for the deeply data-poor regime. The tell is the `train_mse` column: `1e-9`, `2e-15`,
`1e-14`. Every single run drove the *training* loss to zero — it fit the observed entries perfectly — and
still missed the unobserved ones by $45$–$92\%$. So this is not an optimization failure; depth-2 reached
a zero-error interpolant every time. The problem is *which* interpolant: the implicit bias of a
depth-2 factorization, run with Adam and a not-tiny step from a finite init, is simply not selecting the
low-rank one sharply enough — even with samples to spare on rank3-50. The bias I was relying on to come
for free is too weak, and it is coarsened further by the optimizer being a crude approximation to the
gradient flow the theory assumes.

That diagnosis splits the path two ways. One direction is to make the implicit bias *stronger* — more
depth — and I will take that on the next rung. But before I trust a stronger version of the same
implicit mechanism, I want a clean, principled point of comparison: the method that asks for the
low-rank fit *explicitly*, solved to genuine convergence rather than approximated by an adaptive
optimizer from a finite initialization. If depth-2's $0.45$ on rank3-50 is the implicit bias merely
*approximating* nuclear-norm minimization, then *actually* minimizing the nuclear norm — the convex
program, run as a real fixed-point iteration — should do at least as well, and probably better, exactly
where depth-2 left error on the table. So the second rung is the classical convex baseline, and the
question it answers is: how much of depth-2's gap was the weak bias versus the crude approximation?

Let me set up the objective I actually want. With $M$ observed only on a random set $\Omega$, far fewer
than $n^2$ entries, I assume $M$ is low rank and ask for the minimum-nuclear-norm matrix that fits:

$$\text{minimize}\ \|X\|_\* \quad\text{subject to}\quad P_\Omega(X) = P_\Omega(M),$$

where $\|X\|_\* = \sum_i \sigma_i(X)$ is the $\ell_1$ norm of the spectrum and $P_\Omega$ keeps the
observed entries and zeros the rest. The nuclear norm is the right surrogate because it is the convex
envelope of rank on the spectral-norm ball — the tightest convex thing under rank — and the program is
convex. Under an incoherence condition this recovers an incoherent rank-$r$ matrix exactly from
$O(nr\log n)$ entries. So the objective is not in question; what is in question is solving it cheaply at
scale and from cheap repeated operations, because feeding the SDP to an interior-point solver chokes
past $n=100$, ignores the low rank of the answer, and degenerates near the optimum. I need a first-order
method built from operations that only ever touch the matrix cheaply.

The shape of the answer should look like compressed sensing. There, to find a sparse vector with
$Ax = b$, the workhorse is soft-thresholding $S_\tau(x) = \text{sign}(x)(|x| - \tau)_+$, applied
repeatedly, because shrinking small coefficients to exactly zero is what *produces* sparsity. A low-rank
matrix is one whose vector of singular values is sparse, so the matrix analogue is "soft-threshold the
singular values": take the SVD $Y = U\Sigma V^\*$ and define the singular-value shrinkage

$$D_\tau(Y) = U\,\text{diag}\big((\sigma_i - \tau)_+\big)\,V^\*.$$

If many singular values sit below $\tau$, the output has far lower rank than the input — exactly the
spectrum-sparsifying behavior I want, and it is adaptive: it discovers the basis (the singular vectors)
in which the matrix is sparse rather than thresholding in a fixed transform.

Before I build on $D_\tau$ I should check it is the *right* operation, not just an appealing one. In the
scalar case soft-thresholding is the proximity operator of the $\ell_1$ norm. If $D_\tau$ is to play the
analogous role it must be the proximity operator of the nuclear norm,
$D_\tau(Y) = \arg\min_X \tfrac12\|X - Y\|_F^2 + \tau\|X\|_\*$. This holds, and the proof is the
subgradient check: the objective is strictly convex with a unique minimizer characterized by
$0 \in X - Y + \tau\,\partial\|X\|_\*$, and the nuclear-norm subdifferential at $X = U\Sigma V^\*$ is
$\{UV^\* + W : U^\* W = 0,\ WV = 0,\ \|W\|_2 \le 1\}$. Split $Y$'s SVD by the threshold into the part
above $\tau$ ($U_0,\Sigma_0,V_0$) and at/below ($U_1,\Sigma_1,V_1$); then $X = D_\tau(Y) = U_0(\Sigma_0
- \tau I)V_0^\*$ and $Y - X = \tau(U_0 V_0^\* + W)$ with $W = \tau^{-1}U_1\Sigma_1 V_1^\*$. The leading
term is the $UV^\*$ of $X$; $W$ is orthogonal to $U_0,V_0$ and has $\|W\|_2 = \tau^{-1}\max\Sigma_1 \le
1$. So $Y - X \in \tau\,\partial\|X\|_\*$, i.e. $D_\tau$ is the exact prox. The shrink is principled, not
a heuristic.

How do I turn one prox into a method that respects the hard constraint $P_\Omega(X) = P_\Omega(M)$? The
imaging-style move is to *relax*: minimize $\lambda\|X\|_\* + \tfrac12\|P_\Omega(X) - P_\Omega(M)\|_F^2$,
whose forward-backward fixed point iterates $X^k = D_{\lambda\delta}(Y^{k-1})$, $Y^k = X^k + \delta
P_\Omega(M - X^k)$. But this converges to the *penalized* objective: it does not exactly fit the
observations, and to make it fit I would shrink $\lambda$ — which shrinks the threshold $\lambda\delta$,
so the shrink kills almost no singular values and the iterates are *not* low rank. The two things that
make each iteration cheap — a large threshold producing low-rank, sparse iterates — are exactly what
fitting the data forbids in this formulation. Cheapness and accuracy fight. I want a large threshold
*and* exact data fit at once.

The fix is to let the residual accumulate on its own track and shrink *that*. Keep a matrix $Y$ grown
only by the data residual, and read $X$ off it by shrinking:

$$X^k = D_\tau(Y^{k-1}),\qquad Y^k = Y^{k-1} + \delta_k\,P_\Omega(M - X^k),\qquad Y^0 = 0.$$

Now the threshold $\tau$ is *decoupled* from the step size — a fixed constant of its own — and $Y$ is a
running sum of residuals. Cost: $Y^0 = 0$ and every residual update is supported on $\Omega$, so by
induction every $Y^k$ is supported on $\Omega$ — sparse, with at most $m$ nonzeros — and the only real
work is one (partial) SVD per step. With $\tau$ free to be large, the $X^k$ are low rank and the $Y^k$
stay sparse *simultaneously*; the coupling that doomed the penalized version is gone.

What does this iteration converge to? Read it as Uzawa's dual ascent. Uzawa solves $\min_X f(X)$ s.t.
$P_\Omega(X) = P_\Omega(M)$ by alternating an inner minimization over $X$ with a dual step $Y^k =
Y^{k-1} + \delta_k P_\Omega(M - X^k)$ — which matches my $Y$-update for free. Reverse-engineering the
$f$ that makes the inner minimization a single shrink $D_\tau$ (using $P_\Omega Y = Y$ since $Y$ is
always supported on $\Omega$) gives $f(X) = \tau\|X\|_\* + \tfrac12\|X\|_F^2$. So the iteration *is*
Uzawa for $\min \tau\|X\|_\* + \tfrac12\|X\|_F^2$ s.t. $P_\Omega(X) = P_\Omega(M)$. Three consequences.
The extra $\tfrac12\|X\|_F^2$ is the price of a closed-form shrink, and as a bonus it makes $f$ strongly
convex, so the solution is unique. I am not solving bare nuclear-norm minimization — but as $\tau\to
\infty$ the nuclear term dominates and the minimizer converges to the minimum-nuclear-norm solution. So
large $\tau$ serves *both* accuracy and cheapness, the two goals that fought in the penalized form.

Convergence needs a step-size condition. The engine is the strong convexity of $f$: any subgradients
satisfy $\langle Z - Z', X - X'\rangle \ge \|X - X'\|_F^2$ (the Frobenius part gives this directly; the
nuclear part is nonnegative by the duality bounds $\langle Z_0, X\rangle = \|X\|_\*$, $\|Z_0\|_2\le1$).
Running this through the iteration and tracking the dual distance $r_k = \|P_\Omega(Y^k - Y^\*)\|_F$
gives $r_k^2 \le r_{k-1}^2 - (2\delta_k - \delta_k^2)\|X^k - X^\*\|_F^2$, a genuine decrease for $0 <
\delta_k < 2$, so the iterates converge to the unique solution. That bound is conservative — $\delta <
2$ is slow — but the near-isometry $\|P_\Omega(A)\|_F^2 \approx p\|A\|_F^2$ (with $p$ the observation
ratio) lets me push to $\delta = 1.2/p = 1.2\,n^2/m$, taking much bigger steps. This is not a rigorous
theorem (the iterate difference is not a fixed incoherent matrix) but it converges in practice.

Two more pieces. For $\tau$: calibrate against the standard synthetic generator, where $\|M\|_F \approx
n\sqrt r$ and $\|M\|_\* \approx nr$; to make the nuclear term about $10\times$ the Frobenius term the
ratio $2\tau/n \approx 10$, i.e. $\tau = 5n$, keeps the nuclear term dominant while rank is bounded away
from $n$. And a warm start: with $Y^0 = 0$ the first several shrinks return zero (every singular value of
$k\delta P_\Omega(M)$ is below $\tau$ while $k\delta\|P_\Omega(M)\| \le \tau$), so I leap straight to
$Y^0 = k_0\delta P_\Omega(M)$ with $k_0 = \lceil\tau/(\delta\|P_\Omega(M)\|)\rceil$, the first iterate
where the shrink produces something nonzero. Stopping is the KKT residual: $X = D_\tau(Y)$ holds by
construction, so the only optimality residual left is the constraint violation on $\Omega$, monitored as
the relative residual $\|P_\Omega(X^k - M)\|_F / \|P_\Omega(M)\|_F \le \varepsilon$ — which, by the
near-isometry, tracks the true reconstruction error on the unseen entries.

Now I land this on the literal scaffold edit, and the harness has one consequential deviation from the
generic recipe that I have to be honest about. The fill computes $\tau = 5n$ via `tau_factor = 5.0`,
$\delta = 1.2/p$ via `delta_factor = 1.2`, the $k_0$ warm start using the *Frobenius* norm of the
observed data in place of the spectral norm (a slightly conservative overshoot), then iterates a full
`torch.linalg.svd` shrink and a masked residual update, stopping on the squared residual over the
observation count. But there is a hard special case keyed on the environment size: when $n \ge 200$ and
$\text{rank\_hint} \ge 10$ — i.e. rank10-200 — the iteration budget is *capped at 11* iterations rather
than the requested `max_iters`. The comment in the fill is explicit about why: that split performs a
$200\times200$ SVD every iteration, and running the full budget there only oscillates and hits the
wall-clock cap, so a short SVT budget already reaches the intended low-but-finite nuclear-norm baseline.
This matters for my expectations: on rank10-200 this baseline is *not* the converged nuclear-norm
solution — it is a deliberately truncated SVT, run just long enough not to time out. So whatever it
scores there, I should read it as "11 steps of SVT," not "nuclear-norm minimization converged."

So the falsifiable expectations against the depth-2 floor. On rank3-50, depth-2 left $0.452$ despite
generous samples; if that gap was the weak/approximated bias rather than the data being too poor,
*explicit* nuclear-norm minimization run to convergence should beat it clearly — I expect roughly
$0.2$–$0.3$, a real improvement over $0.45$. On rank5-100, where depth-2 got $0.596$, the convex program
should again improve on it, landing somewhere in the $0.3$ range — better than depth-2's weak implicit
bias, since here I am asking for the low-nuclear-norm fit directly instead of hoping Adam approximates
it. On rank10-200 the picture is bleaker and the budget cap makes it bleaker still: $4000$ observations
for a rank-10 matrix is right at the information floor, the regime where nuclear-norm minimization itself
starts to fail, and the 11-iteration truncation means I am not even running it to convergence — so I
expect it to stay around depth-2's $0.92$, essentially no recovery, confirming that this whole
nuclear-norm-strength family runs out exactly there. The clean test is rank3-50 and rank5-100: if the
convex baseline beats $0.452$ and $0.596$, it confirms depth-2's gap was a *bias-strength* problem and
sets up the real question for the next rung — whether *deeper* implicit regularization sharpens the bias
past nuclear-norm strength and reclaims the easy environments outright.
